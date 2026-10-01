"""Layer 4 — label, features, rolling Ridge A/M and 16-cell thresholds.

Reads ``SignalConfig.model`` (:class:`ModelSpec`).  Rows enter this layer only
after layers 1–3: store rows ∩ signal index ∩ point filter ∩ stock-days the day
selector kept.  Then, per signal (pandas, exact port of modelUpdate.py)::

    label → train_filter → features
      → train_and_predict_ridge_rolling  A (timing, 60d) + M (selection, 20d)
      → 16-cell prior-day quantile thresholds → pass_threshold

The frame returned by :func:`build_signal_frame` is *long*: one row per
(signal_id, tick), so a tick claimed by two signals appears twice and each
signal is trained on its own rows.  A static signal (``use_model_gate=False``)
still gets models and thresholds as diagnostics, but :func:`select_signal`
returns every row.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ..framework.dataset import _atomic_write_json, _atomic_write_parquet
from ..framework.registry import CellThresholdSpec, SignalConfig
from ..framework.store import store_dates
from .day_selector import apply_day_scores_pl, read_day_scores
from .point_filter import load_rows


SIGNAL_COL = "signal_id"


def _model(signal: SignalConfig):
    if signal.model is None:
        raise ValueError(f"{signal.condition_id}: no model configured")
    return signal.model


# --------------------------------------------------------------------------- #
# frames
# --------------------------------------------------------------------------- #


def load_signal_rows(
    signal: SignalConfig,
    *,
    store_dir: Path | str,
    points_dir: Path | str,
    params_dir: Path | str,
    dates: Sequence[str],
    use_day_selector: bool = True,
    keep_unscored: bool = True,
) -> pd.DataFrame:
    """Candidate rows (layers 1–2) minus the stock-days the day selector excluded, as pandas."""

    rows = load_rows(signal, store_dir=store_dir, points_dir=points_dir, dates=dates)
    if rows.is_empty():
        return pd.DataFrame()
    if use_day_selector and signal.day_selector is not None:
        scores = read_day_scores(params_dir, signal)
        if scores.height:
            rows = rows.filter(apply_day_scores_pl(rows, scores, keep_unscored=keep_unscored))
    return rows.to_pandas()


def prepare_signal_frame(signal: SignalConfig, rows: pd.DataFrame) -> pd.DataFrame:
    """label → train_filter → features, then tag rows with the signal id."""

    spec = _model(signal)
    if rows.empty:
        return rows
    out = spec.label(rows)
    out = spec.train_filter(out)
    out = spec.features(out)
    if spec.round_feature_decimals is not None:
        for column in spec.feature_columns:
            out[column] = out[column].round(spec.round_feature_decimals)
    out[SIGNAL_COL] = signal.st_name
    return out


def build_signal_frame(
    signals: Iterable[SignalConfig],
    *,
    dirs: Mapping[str, Path],
    dates: Sequence[str] | None = None,
    last_n_days: int | None = None,
    use_day_selector: bool = True,
) -> pd.DataFrame:
    """Long-format frame with every signal's labeled, featured rows."""

    if dates is None:
        dates = store_dates(dirs["store"])
        if last_n_days is not None:
            dates = dates[-last_n_days:]
    parts = []
    for signal in signals:
        if signal.model is None:
            continue
        rows = load_signal_rows(
            signal, store_dir=dirs["store"], points_dir=dirs["points"], params_dir=dirs["params"],
            dates=dates, use_day_selector=use_day_selector,
        )
        if rows.empty:
            continue
        parts.append(prepare_signal_frame(signal, rows))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True, copy=False)


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #


@dataclass
class SignalModelResult:
    st_name: str
    df: pd.DataFrame
    timing: dict[str, Any]
    selection: dict[str, Any]


def _append_next_day_dummy(df: pd.DataFrame, next_trade_day: str) -> pd.DataFrame:
    dummy = df.iloc[-1:].copy()
    dummy["Date"] = next_trade_day
    return pd.concat([df, dummy], ignore_index=True)


def train_signal(
    signal: SignalConfig,
    signal_df: pd.DataFrame,
    *,
    next_trade_day: str | None = None,
    n_jobs: int = -1,
) -> SignalModelResult:
    """Fit the timing (A) and selection (M) rolling Ridge models for one signal."""

    # The engine configures BLAS at import; load it only when training is requested.
    from src.research.rolling_model import train_and_predict_ridge_rolling

    spec = _model(signal)
    part = signal_df[signal_df[SIGNAL_COL] == signal.st_name].reset_index(drop=True)
    if part.empty:
        raise ValueError(f"no rows for signal {signal.st_name}")
    if next_trade_day is not None:
        part = _append_next_day_dummy(part, next_trade_day)
    common = dict(
        X_list=list(spec.feature_columns),
        feature_screening=spec.feature_screening,
        alpha=spec.alpha,
        max_position_per_stock=spec.max_position_per_stock,
        n_jobs=n_jobs,
    )
    timing = train_and_predict_ridge_rolling(
        signal_df=part, Y_col=spec.timing_target, n_training_days=spec.timing_lookback_days, **common,
    )
    selection = train_and_predict_ridge_rolling(
        signal_df=timing["df"], Y_col=spec.selection_target, n_training_days=spec.selection_lookback_days,
        day_features=[], **common,
    )
    return SignalModelResult(signal.st_name, selection["df"], timing, selection)


def train_by_signal(
    signal_df: pd.DataFrame,
    signals: Iterable[SignalConfig],
    *,
    next_trade_day: str | None = None,
    n_jobs: int = -1,
) -> dict[str, SignalModelResult]:
    by_name = {s.st_name: s for s in signals if s.model is not None}
    results: dict[str, SignalModelResult] = {}
    for st_name in signal_df[SIGNAL_COL].unique():
        signal = by_name.get(st_name)
        if signal is None:
            continue
        results[st_name] = train_signal(signal, signal_df, next_trade_day=next_trade_day, n_jobs=n_jobs)
    return results


# --------------------------------------------------------------------------- #
# thresholds (16-cell prior-day quantiles; same semantics as modelUpdate.py)
# --------------------------------------------------------------------------- #


def assign_cell2d(df: pd.DataFrame, spec: CellThresholdSpec) -> pd.Series:
    time_bucket = pd.cut(df[spec.time_col], list(spec.time_edges), labels=False)
    ref_bucket = pd.cut(df[spec.ref_col], list(spec.ref_edges), labels=False)
    return time_bucket * 10 + ref_bucket


def build_prior_quantile_tables(
    df: pd.DataFrame, pred_col: str, q: float, n_days: int, extra_dates: Sequence[str] | None = None,
) -> dict[str, dict[str, float]]:
    """{YYYYMMDD: {cell_id: threshold}} using each cell's prior ``n_days`` sessions."""

    work = df[["Date", "Cell2D", pred_col]].dropna()
    work = work.assign(Date=pd.to_datetime(work["Date"]).dt.strftime("%Y%m%d"))
    tables: dict[str, dict[str, float]] = {}
    all_dates = sorted(set(work["Date"].unique()) | set(extra_dates or []))
    for cell, part in work.groupby("Cell2D"):
        vals = part.groupby("Date")[pred_col].apply(lambda x: x.to_numpy())
        cell_dates = np.array(sorted(vals.index))
        cell_key = str(int(cell))
        for d in all_dates:
            prior = cell_dates[cell_dates < d][-n_days:]
            if len(prior) == 0:
                continue
            pool = np.concatenate([vals[w] for w in prior])
            if pool.size:
                tables.setdefault(d, {})[cell_key] = float(np.quantile(pool, q))
    return tables


def attach_cell_thresholds(
    df: pd.DataFrame, tables_a: Mapping[str, Mapping[str, float]], tables_m: Mapping[str, Mapping[str, float]]
) -> pd.DataFrame:
    out = df.copy()
    date_key = pd.to_datetime(out["Date"]).dt.strftime("%Y%m%d")
    cell_key = out["Cell2D"]

    def lookup(tables):
        return [tables.get(d, {}).get(str(int(c)), np.nan) if pd.notna(c) else np.nan for d, c in zip(date_key, cell_key)]

    out["thr_net"] = lookup(tables_a)
    out["thr_netM"] = lookup(tables_m)
    return out


@dataclass
class SignalThresholdResult:
    st_name: str
    df: pd.DataFrame
    tables_timing: dict[str, dict[str, float]]
    tables_selection: dict[str, dict[str, float]]


def threshold_signal(
    signal: SignalConfig, result: SignalModelResult, *, extra_dates: Sequence[str] | None = None,
) -> SignalThresholdResult:
    spec = _model(signal)
    df = result.df.copy()
    df["Cell2D"] = assign_cell2d(df, spec.threshold)
    tables_a = build_prior_quantile_tables(df, spec.timing_pred_col, spec.threshold.q_timing, spec.threshold.n_days, extra_dates)
    tables_m = build_prior_quantile_tables(df, spec.selection_pred_col, spec.threshold.q_selection, spec.threshold.n_days, extra_dates)
    df = attach_cell_thresholds(df, tables_a, tables_m)
    df["pass_threshold"] = (df[spec.timing_pred_col] > df["thr_net"]) & (df[spec.selection_pred_col] > df["thr_netM"])
    return SignalThresholdResult(signal.st_name, df, tables_a, tables_m)


def threshold_by_signal(
    results: Mapping[str, SignalModelResult], signals: Iterable[SignalConfig], *, extra_dates: Sequence[str] | None = None,
) -> dict[str, SignalThresholdResult]:
    by_name = {s.st_name: s for s in signals}
    return {st: threshold_signal(by_name[st], res, extra_dates=extra_dates) for st, res in results.items() if st in by_name}


# --------------------------------------------------------------------------- #
# final selection
# --------------------------------------------------------------------------- #


def select_signal(signal: SignalConfig, thresholded: SignalThresholdResult) -> pd.DataFrame:
    """Rows the signal would trade: past the model thresholds, or every row for a static book."""

    spec = _model(signal)
    df = thresholded.df
    keep = df["pass_threshold"].fillna(False).astype(bool) if spec.use_model_gate else pd.Series(True, index=df.index)
    return df[keep].sort_values(["Date", "QuoteCode", "TransTime"]).reset_index(drop=True)


def select_by_signal(
    thresholded: Mapping[str, SignalThresholdResult], signals: Iterable[SignalConfig],
) -> dict[str, pd.DataFrame]:
    by_name = {s.st_name: s for s in signals}
    return {st: select_signal(by_name[st], res) for st, res in thresholded.items() if st in by_name}


# --------------------------------------------------------------------------- #
# orchestration (what the CLI runs)
# --------------------------------------------------------------------------- #


SELECTED_COLUMNS = ("Date", "QuoteCode", "ChannelSeq", "TransTime")


def selected_path(model_dir: Path | str, signal: SignalConfig) -> Path:
    return Path(model_dir) / f"{signal.st_name}_selected.parquet"


def build_model_layer(
    signal: SignalConfig,
    *,
    dirs: Mapping[str, Path],
    dates: Sequence[str] | None = None,
    last_n_days: int | None = None,
    next_trade_day: str | None = None,
    n_jobs: int = -1,
    verbose: bool = True,
) -> SignalThresholdResult | None:
    """Train, threshold and persist the selection for one signal (``model/<st>_selected.parquet``)."""

    if signal.model is None:
        if verbose:
            print(f"[{signal.condition_id}] no model configured; skipping")
        return None
    spec = signal.model
    frame = build_signal_frame([signal], dirs=dirs, dates=dates, last_n_days=last_n_days)
    if frame.empty:
        if verbose:
            print(f"[{signal.condition_id}] no rows to model")
        return None
    result = train_signal(signal, frame, next_trade_day=next_trade_day, n_jobs=n_jobs)
    thresholded = threshold_signal(signal, result, extra_dates=[next_trade_day] if next_trade_day else None)
    selected = select_signal(signal, thresholded)
    model_dir = Path(dirs["model"])
    model_dir.mkdir(parents=True, exist_ok=True)
    out = thresholded.df[
        [*SELECTED_COLUMNS, spec.timing_pred_col, spec.selection_pred_col, "thr_net", "thr_netM", "pass_threshold"]
    ].copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y%m%d")
    out["selected"] = out.index.isin(thresholded.df.index[thresholded.df["pass_threshold"].fillna(False).astype(bool)]) if spec.use_model_gate else True
    _atomic_write_parquet(pl.from_pandas(out.reset_index(drop=True)), selected_path(model_dir, signal))
    _atomic_write_json(
        {
            "condition_id": signal.condition_id,
            "model_version": spec.version,
            "feature_columns": list(spec.feature_columns),
            "use_model_gate": spec.use_model_gate,
            "dates": int(out["Date"].nunique()),
            "rows": int(len(out)),
            "selected_rows": int(len(selected)),
            "next_trade_day": next_trade_day,
            "updated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        },
        selected_path(model_dir, signal).with_suffix(".manifest.json"),
    )
    if verbose:
        print(f"[{signal.condition_id}] model: {len(frame):,} rows trained, {len(selected):,} selected -> {selected_path(model_dir, signal)}")
    return thresholded


__all__ = [
    "SELECTED_COLUMNS",
    "SIGNAL_COL",
    "SignalModelResult",
    "SignalThresholdResult",
    "assign_cell2d",
    "attach_cell_thresholds",
    "build_model_layer",
    "build_prior_quantile_tables",
    "build_signal_frame",
    "load_signal_rows",
    "prepare_signal_frame",
    "select_by_signal",
    "select_signal",
    "selected_path",
    "threshold_by_signal",
    "threshold_signal",
    "train_by_signal",
    "train_signal",
]
