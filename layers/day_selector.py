"""Layer 3 — learned stock-day risk filter → day scores.

Reads ``SignalConfig.day_selector`` (:class:`DaySelectorSpec`).  The day-level
rules that used to be hand-set thresholds are replaced by a few simple risk
models, each predicting a ``ds_*`` stock-day target (next-day squeeze, stop
hit, ...) from D-1 features, refit every day on the strictly-prior window with
sample weights, and each cutting at an absolute threshold.  ``hard_rules``
(the price cap) stay on regardless; a stock-day is kept only when no model
excludes it.

Artifacts (``artifacts/params/``)::

    <st>_stock_days.parquet   one row per (Date, QuoteCode): D-1 features, weight, ds_* targets  (cache)
    <st>_day_scores.parquet   one row per (Date, QuoteCode): score_<model>, excluded_by, kept     (output)

Causality: features come from D-1 files (preMarket / brenchFeature named for
D), targets are that stock-day's realized values, and the models for D are fit
only on stock-days strictly before D.  Every source column is read from the
store as delivered by the feature pipeline; the only values computed here are
the ``ds_*`` targets, the sample weight and ``log_trading_turnover``.
"""

from __future__ import annotations

import datetime as _dt
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from ..framework.dataset import _atomic_write_json, _atomic_write_parquet
from ..framework.registry import DaySelectorSpec, RiskModelSpec, SignalConfig
from .point_filter import filter_hash, load_rows, read_points


# Raw stock-day columns carried in the table for the weight and the baseline rules.
CARRY_COLUMNS = (
    "trading_turnover", "trading_turnover_bf", "top1_net_buy_amount", "avg_bidLots1", "avg_askLots1",
    "d_intraday_return", "dt_major_count", "MarketCap", "sell_branch_breadth_rank", "net_buy_gini",
    "RefPrice", "hft_strick_makerSpreadBP",
)


def stock_day_table_path(params_dir: Path | str, signal: SignalConfig) -> Path:
    return Path(params_dir) / f"{signal.st_name}_stock_days.parquet"


def day_scores_path(params_dir: Path | str, signal: SignalConfig) -> Path:
    return Path(params_dir) / f"{signal.st_name}_day_scores.parquet"


def _table_manifest_path(params_dir: Path | str, signal: SignalConfig) -> Path:
    return Path(params_dir) / f"{signal.st_name}_stock_days.manifest.json"


def _spec(signal: SignalConfig, spec: DaySelectorSpec | None) -> DaySelectorSpec:
    if spec is not None:
        return spec
    if signal.day_selector is None:
        raise ValueError(f"{signal.condition_id}: no day selector configured")
    return signal.day_selector


def _derived(frame: pl.DataFrame) -> pl.DataFrame:
    """Feature transforms only (no data values are recomputed here)."""

    exprs = []
    if "trading_turnover" in frame.columns:
        exprs.append(pl.col("trading_turnover").clip(lower_bound=0).log1p().alias("log_trading_turnover"))
    if "OTC" in frame.columns:
        exprs.append(pl.col("OTC").cast(pl.Float64).alias("OTC"))
    return frame.with_columns(exprs) if exprs else frame


# --------------------------------------------------------------------------- #
# stock-day table
# --------------------------------------------------------------------------- #


def _table_identity(signal: SignalConfig, spec: DaySelectorSpec) -> dict[str, object]:
    return {"point_filter": filter_hash(signal), "targets": list(spec.targets), "features": list(spec.features), "version": spec.version, "labels": "all_ds"}


def build_stock_day_table(
    signal: SignalConfig,
    *,
    store_dir: Path | str,
    points_dir: Path | str,
    params_dir: Path | str,
    spec: DaySelectorSpec | None = None,
    force: bool = False,
    verbose: bool = True,
) -> pl.DataFrame:
    """Per (Date, QuoteCode): D-1 features (first candidate row), sample weight, ``ds_*`` targets."""

    spec = _spec(signal, spec)
    params_dir = Path(params_dir)
    params_dir.mkdir(parents=True, exist_ok=True)
    path = stock_day_table_path(params_dir, signal)
    manifest_path = _table_manifest_path(params_dir, signal)
    identity = _table_identity(signal, spec)
    recorded = None
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            recorded = json.load(handle).get("identity")
    if path.exists() and not force and recorded != identity:
        if verbose:
            print(f"[{signal.condition_id}] stock-day table inputs changed; rebuilding")
        force = True
    cache = pl.read_parquet(path) if path.exists() and not force else None
    done = set(cache["Date"].unique().to_list()) if cache is not None and cache.height else set()
    points = read_points(points_dir, signal)
    dates = sorted(points["Date"].unique().to_list()) if points.height else []
    todo = [d for d in dates if d not in done]
    latest_done = max(done) if done else None
    if latest_done in dates and latest_done not in todo:
        todo.append(latest_done)   # the latest day may have been refreshed
        if cache is not None:
            cache = cache.filter(pl.col("Date") != latest_done)
    todo.sort()
    if verbose:
        print(f"[{signal.condition_id}] stock-day table: {len(todo)} day(s) to build")
    parts = [cache] if cache is not None and cache.height else []
    feature_cols = [c for c in spec.features if c != "log_trading_turnover"]
    for position, date in enumerate(todo, start=1):
        rows = load_rows(signal, store_dir=store_dir, points_dir=points_dir, dates=[date])
        if rows.is_empty():
            continue
        labels = spec.labels(rows)
        if labels.is_empty():
            continue
        present = [c for c in feature_cols if c in rows.columns]
        carry = [c for c in CARRY_COLUMNS if c in rows.columns and c not in present]
        feats = (
            rows.sort(["QuoteCode", "TransTime"])
            .group_by("QuoteCode", maintain_order=True)
            .agg(pl.len().alias("n_rows"), *[pl.col(c).first().cast(pl.Float64).alias(c) for c in present + carry])
        )
        table = feats.join(labels.drop("Date") if "Date" in labels.columns else labels, on="QuoteCode", how="inner")
        table = _derived(table.with_columns(pl.lit(date).alias("Date")))
        weight = spec.weight(table) if spec.weight is not None else pl.Series("weight", [1.0] * table.height)
        table = table.with_columns(weight.cast(pl.Float64).alias("weight"))
        ds_cols = [c for c in labels.columns if c.startswith("ds_")]   # every label column, not only the model targets
        parts.append(table.select("Date", "QuoteCode", "n_rows", "weight", *ds_cols, *present, *carry,
                                  *[c for c in ("log_trading_turnover",) if c in table.columns and c in spec.features]))
        if verbose and (position % 50 == 0 or position == len(todo)):
            print(f"  {position}/{len(todo)} ({date}: {table.height} stock-days)")
    if not parts:
        return pl.DataFrame()
    out = pl.concat(parts, how="diagonal_relaxed").sort(["Date", "QuoteCode"])
    _atomic_write_parquet(out, path)
    _atomic_write_json({"condition_id": signal.condition_id, "identity": identity}, manifest_path)
    return out


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #


def _make_model(m: RiskModelSpec):
    h = m.hyper
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if m.family == "ridge":
        from sklearn.linear_model import Ridge

        return make_pipeline(StandardScaler(), Ridge(alpha=float(h.get("alpha", 10.0))))
    if m.family == "logistic":
        from sklearn.linear_model import LogisticRegression

        return make_pipeline(StandardScaler(), LogisticRegression(C=float(h.get("C", 1.0)), max_iter=int(h.get("max_iter", 500))))
    if m.task == "regression":
        from sklearn.ensemble import HistGradientBoostingRegressor as HGB
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    return HGB(
        max_depth=int(h.get("max_depth", 2)), max_iter=int(h.get("max_iter", 150)), learning_rate=float(h.get("learning_rate", 0.05)),
        min_samples_leaf=int(h.get("min_samples_leaf", 60)), l2_regularization=float(h.get("l2", 1.0)), random_state=0,
    )


def _fit_kwargs(model, weight: np.ndarray) -> dict[str, object]:
    """sample_weight routed to the last step of a pipeline, or to the estimator itself."""

    if hasattr(model, "steps"):
        return {f"{model.steps[-1][0]}__sample_weight": weight}
    return {"sample_weight": weight}


def _prepare_x(frame: pl.DataFrame, features: Sequence[str], medians: dict[str, float] | None = None):
    x = frame.select([pl.col(c).cast(pl.Float64) for c in features]).to_numpy()
    x = np.where(np.isfinite(x), x, np.nan)
    if medians is None:
        med = np.nanmedian(x, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        medians = {c: float(m) for c, m in zip(features, med)}
    fill = np.array([medians[c] for c in features])
    idx = np.where(np.isnan(x))
    x[idx] = np.take(fill, idx[1])
    return x, medians


def _score_one(
    m: RiskModelSpec, train: pl.DataFrame, today: pl.DataFrame, features: Sequence[str]
) -> tuple[np.ndarray, float] | None:
    """Fit one risk model on the window, score today's candidates and resolve the day's cut; None when it cannot be fit."""

    tr = train.filter(pl.col(m.target).is_finite() & (pl.col("weight") > 0))
    if tr.is_empty():
        return None
    y = tr[m.target].to_numpy().astype(float)
    if m.task == "classification":
        y = (y > 0.5).astype(int)
        if y.min() == y.max():
            return None
    x_tr, med = _prepare_x(tr, features)
    model = _make_model(m)
    model.fit(x_tr, y, **_fit_kwargs(model, tr["weight"].to_numpy().astype(float)))
    x_te, _ = _prepare_x(today, features, med)
    predict = (lambda x: model.predict_proba(x)[:, 1]) if m.task == "classification" else model.predict
    score = predict(x_te)
    if m.threshold_mode == "window_quantile":
        cut = float(np.quantile(predict(x_tr), m.threshold))   # the window's own score distribution sets the bar
    else:
        cut = float(m.threshold)
    return score, cut


def build_day_scores(
    signal: SignalConfig,
    *,
    params_dir: Path | str,
    spec: DaySelectorSpec | None = None,
    table: pl.DataFrame | None = None,
    dates: Sequence[str] | None = None,
    verbose: bool = True,
) -> pl.DataFrame:
    """Rolling: for each date, fit every risk model on strictly-prior stock-days and score that date."""

    spec = _spec(signal, spec)
    params_dir = Path(params_dir)
    if table is None:
        table = pl.read_parquet(stock_day_table_path(params_dir, signal))
    table = _derived(table)
    all_dates = sorted(table["Date"].unique().to_list())
    targets = list(dates) if dates is not None else all_dates
    out_parts: list[pl.DataFrame] = []
    fitted = 0
    for position, date in enumerate(targets, start=1):
        window = [d for d in all_dates if d < date][-spec.window_days :]
        today = table.filter(pl.col("Date") == date)
        if today.is_empty():
            continue
        train = table.filter(pl.col("Date").is_in(window))
        hard_ok = spec.hard_rules.mask(today).to_numpy() if spec.hard_rules is not None else np.ones(today.height, dtype=bool)
        cols: dict[str, object] = {}
        excluded_by = [[] for _ in range(today.height)]
        enough = bool(window) and train.height >= spec.min_train_stock_days
        for m in spec.models:
            features = [c for c in m.features if c in table.columns]
            fitted_m = _score_one(m, train, today, features) if enough and features else None
            if fitted_m is None:
                cols[m.score_column] = pl.Series(m.score_column, [None] * today.height, dtype=pl.Float64)
                cols[m.cut_column] = pl.Series(m.cut_column, [None] * today.height, dtype=pl.Float64)
                continue
            score, cut = fitted_m
            cols[m.score_column] = pl.Series(m.score_column, score.astype(float), dtype=pl.Float64)
            cols[m.cut_column] = pl.Series(m.cut_column, [cut] * today.height, dtype=pl.Float64)
            for i in np.flatnonzero(m.excludes(score, cut)):
                excluded_by[i].append(m.model_id)
        if enough:
            fitted += 1
        kept = hard_ok & np.array([not e for e in excluded_by], dtype=bool)
        scored = today.select("Date", "QuoteCode").with_columns(
            pl.Series("hard_ok", hard_ok, dtype=pl.Boolean),
            *cols.values(),
            pl.Series("excluded_by", [",".join(e) for e in excluded_by], dtype=pl.String),
            pl.Series("kept", kept, dtype=pl.Boolean),
            pl.lit("scored" if enough else "insufficient_history", dtype=pl.String).alias("reason"),
            pl.lit(spec.model_id, dtype=pl.String).alias("model"),
            pl.lit(window[0] if window else None, dtype=pl.String).alias("window_start"),
            pl.lit(window[-1] if window else None, dtype=pl.String).alias("window_end"),
            pl.lit(int(train.height) if window else 0, dtype=pl.Int64).alias("n_train"),
        )
        out_parts.append(scored)
        if verbose and (position % 50 == 0 or position == len(targets)):
            print(f"  scored {position}/{len(targets)} ({date}: {today.height} candidates, fits so far {fitted})")
    return pl.concat(out_parts, how="vertical") if out_parts else pl.DataFrame()


def write_day_scores(
    signal: SignalConfig, scores: pl.DataFrame, *, params_dir: Path | str, spec: DaySelectorSpec | None = None
) -> Path:
    spec = _spec(signal, spec)
    params_dir = Path(params_dir)
    path = day_scores_path(params_dir, signal)
    _atomic_write_parquet(scores, path)
    _atomic_write_json(
        {
            "condition_id": signal.condition_id,
            "model": spec.model_id,
            "spec_hash": spec.stable_hash(),
            "models": [m.payload() for m in spec.models],
            "window_days": spec.window_days,
            "hard_rules": spec.hard_rules.gate_id if spec.hard_rules else None,
            "dates": int(scores["Date"].n_unique()) if scores.height else 0,
            "updated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        },
        path.with_suffix(".manifest.json"),
    )
    return path


def read_day_scores(params_dir: Path | str, signal: SignalConfig) -> pl.DataFrame:
    path = day_scores_path(params_dir, signal)
    if not path.exists():
        return pl.DataFrame(schema={"Date": pl.String, "QuoteCode": pl.String, "kept": pl.Boolean})
    return pl.read_parquet(path)


# --------------------------------------------------------------------------- #
# applying the scores to rows (layer 4 reads its rows through these)
# --------------------------------------------------------------------------- #


def apply_day_scores_pl(rows: pl.DataFrame, scores: pl.DataFrame, *, keep_unscored: bool = True) -> pl.Series:
    """Boolean Series: the row's (Date, QuoteCode) was kept by the day selector.

    ``keep_unscored`` decides what happens to stock-days the selector never saw
    (no history yet, or a date it has not scored): True keeps them.
    """

    if scores.is_empty() or rows.is_empty():
        return pl.Series("kept", [True] * rows.height, dtype=pl.Boolean)
    kept = scores.filter(pl.col("kept")).select("Date", "QuoteCode").unique().with_columns(pl.lit(True).alias("_kept"))
    seen = scores.select("Date", "QuoteCode").unique().with_columns(pl.lit(True).alias("_seen"))
    keys = rows.select(pl.col("Date").cast(pl.String), pl.col("QuoteCode").cast(pl.String))
    joined = keys.join(kept, on=["Date", "QuoteCode"], how="left").join(seen, on=["Date", "QuoteCode"], how="left")
    mask = joined["_kept"].fill_null(False)
    if keep_unscored:
        mask = mask | ~joined["_seen"].fill_null(False)
    return mask.alias("kept")


def apply_day_scores_pd(df: pd.DataFrame, scores: pl.DataFrame, *, keep_unscored: bool = True) -> pd.Series:
    """pandas twin of :func:`apply_day_scores_pl` (Date may be datetime or YYYYMMDD)."""

    if scores.is_empty() or df.empty:
        return pd.Series(True, index=df.index)
    keys = pl.DataFrame({
        "Date": pd.to_datetime(df["Date"]).dt.strftime("%Y%m%d").to_numpy(),
        "QuoteCode": df["QuoteCode"].astype(str).to_numpy(),
    })
    mask = apply_day_scores_pl(keys, scores, keep_unscored=keep_unscored)
    return pd.Series(mask.to_numpy(), index=df.index)


# --------------------------------------------------------------------------- #
# evaluation: weighted realized outcome of kept vs excluded, per model by score bucket, vs baseline rules
# --------------------------------------------------------------------------- #


POSITION_CAP_NTD = 5_000_000.0   # weight 1.0 = the book fills the 500萬 per-stock cap


def _weighted_summary(frame: pl.DataFrame, pnl_col: str, extra: Sequence[str]) -> dict[str, float]:
    if frame.is_empty():
        return {"stock_days_per_day": 0.0, "weight_share": 0.0, "weighted_pnl_bp": float("nan"), "pnl_proxy_萬": 0.0, "worst_day_萬": 0.0}
    w = frame["weight"]
    daily = frame.with_columns((pl.col("weight") * pl.col(pnl_col) / 1e4 * POSITION_CAP_NTD / 1e4).alias("_pnl")).group_by("Date").agg(pl.col("_pnl").sum())
    out = {
        "stock_days_per_day": float(frame.group_by("Date").len()["len"].mean()),
        "weight_share": float(w.sum()),
        "weighted_pnl_bp": float((w * frame[pnl_col]).sum() / max(float(w.sum()), 1e-12)),
        "pnl_proxy_萬": float(daily["_pnl"].sum()),
        "worst_day_萬": float(daily["_pnl"].min()),
        "sharpe": float(daily["_pnl"].mean() / daily["_pnl"].std() * np.sqrt(240)) if daily.height > 2 and daily["_pnl"].std() > 0 else float("nan"),
    }
    for c in extra:
        if c in frame.columns:
            out[f"w_{c}"] = float((w * frame[c]).sum() / max(float(w.sum()), 1e-12))
    return out


def evaluate_day_scores(
    table: pl.DataFrame,
    scores: pl.DataFrame,
    *,
    spec: DaySelectorSpec,
    pnl_col: str = "ds_pnl_bp",
    buckets: int = 10,
) -> dict[str, pl.DataFrame]:
    """Weighted (by ``weight``) evaluation on scored stock-days.

    ``summary``: all / kept / excluded (/ baseline kept / excluded / the two
    disagreement sets).  ``per_model``: for each risk model, realized outcome by
    predicted-score bucket plus the rows above/below its threshold.
    """

    joined = _derived(table).join(scores.filter(pl.col("reason") == "scored"), on=["Date", "QuoteCode"], how="inner")
    joined = joined.filter(pl.col(pnl_col).is_finite())
    if joined.is_empty():
        raise ValueError("no scored stock-days to evaluate")
    extra = [t for t in spec.targets if t != pnl_col]
    rows = []

    def _add(name: str, mask):
        rows.append({"set": name, **_weighted_summary(joined.filter(mask), pnl_col, extra)})

    _add("all candidates", pl.lit(True))
    _add("kept", pl.col("kept"))
    _add("excluded", ~pl.col("kept"))
    if spec.baseline_rules is not None:
        joined = joined.with_columns(spec.baseline_rules.mask(joined).alias("_rule_ok"))
        _add("baseline rules kept", pl.col("_rule_ok"))
        _add("baseline rules excluded", ~pl.col("_rule_ok"))
        _add("kept by models, excluded by rules", pl.col("kept") & ~pl.col("_rule_ok"))
        _add("excluded by models, kept by rules", ~pl.col("kept") & pl.col("_rule_ok"))
    summary = pl.DataFrame(rows)

    per_model_rows = []
    for m in spec.models:
        col = m.score_column
        if col not in joined.columns:
            continue
        part = joined.filter(pl.col(col).is_not_null())
        if part.is_empty():
            continue
        part = part.with_columns(
            ((pl.col(col).rank("average") / pl.col(col).count() * buckets).ceil().clip(1, buckets)).cast(pl.Int32).alias("bucket")
        )
        for b, g in part.group_by("bucket", maintain_order=True):
            b = b[0] if isinstance(b, tuple) else b
            per_model_rows.append({"model": m.model_id, "bucket": int(b), "score_min": float(g[col].min()), "score_max": float(g[col].max()),
                                   **_weighted_summary(g, pnl_col, [m.target])})
        cut = pl.col(m.cut_column) if m.cut_column in part.columns else pl.lit(m.threshold)
        for name, mask in (("above threshold", pl.col(col) > cut), ("below threshold", pl.col(col) <= cut)):
            per_model_rows.append({"model": m.model_id, "bucket": name, "score_min": float("nan"), "score_max": float("nan"),
                                   **_weighted_summary(part.filter(mask), pnl_col, [m.target])})
    per_model = pl.DataFrame(per_model_rows) if per_model_rows else pl.DataFrame()
    return {"summary": summary, "per_model": per_model}


# --------------------------------------------------------------------------- #
# orchestration (what the CLI runs)
# --------------------------------------------------------------------------- #


def build_day_selector_layer(
    signal: SignalConfig,
    *,
    dirs: Mapping[str, Path],
    spec: DaySelectorSpec | None = None,
    force_table: bool = False,
    evaluate: bool = False,
    verbose: bool = True,
) -> pl.DataFrame:
    """Stock-day table → rolling risk scores → ``params/<st>_day_scores.parquet``."""

    spec = _spec(signal, spec)
    table = build_stock_day_table(
        signal, store_dir=dirs["store"], points_dir=dirs["points"], params_dir=dirs["params"],
        spec=spec, force=force_table, verbose=verbose,
    )
    if table.is_empty():
        if verbose:
            print(f"[{signal.condition_id}] no stock-days; day selector skipped")
        return pl.DataFrame()
    scores = build_day_scores(signal, params_dir=dirs["params"], spec=spec, table=table, verbose=verbose)
    path = write_day_scores(signal, scores, params_dir=dirs["params"], spec=spec)
    if verbose:
        kept = scores.filter(pl.col("reason") == "scored")
        share = float(kept["kept"].mean()) if kept.height else float("nan")
        print(f"[{signal.condition_id}] {spec.model_id}: {scores['Date'].n_unique()} dates, kept share {share:.3f} -> {path}")
    if evaluate:
        result = evaluate_day_scores(table, scores, spec=spec)
        with pl.Config(tbl_rows=80, tbl_cols=20, tbl_width_chars=220, float_precision=3):
            print(result["summary"]); print(result["per_model"])
    return scores


__all__ = [
    "CARRY_COLUMNS",
    "POSITION_CAP_NTD",
    "apply_day_scores_pd",
    "apply_day_scores_pl",
    "build_day_scores",
    "build_day_selector_layer",
    "build_stock_day_table",
    "day_scores_path",
    "evaluate_day_scores",
    "read_day_scores",
    "stock_day_table_path",
    "write_day_scores",
]
