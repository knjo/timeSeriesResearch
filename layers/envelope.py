"""Layer 1 — envelope + condition → signal index.

Reads ``SignalConfig.envelope`` / ``gate`` / ``selection`` / ``selection_pool``.

1. ``build_envelope_index`` (``framework/store.py``): the parameter-free envelope
   is scanned on raw day sources into ``envelope/<st>_index.parquet``.
2. ``build_store`` (``framework/store.py``): the union of every signal's
   envelope is merged once into ``store/YYYYMMDD.parquet`` with ``cond_<st>``
   tags (shared by all signals, so it is built per registry, not per signal).
3. :func:`build_condition_params`: the condition's parameters are re-picked
   every day from the frozen ``gate.grid`` on strictly-prior store rows
   (``params/<st>_condition.parquet``; every grid score is kept).  The rows
   scored are the ones layer 2 would keep (point filter applied first), so the
   winner is chosen on the same distribution the later layers see.
4. :func:`build_signal_index`: each store day is gated with that day's
   parameters into ``index/<st>_index.parquet``.

Causality: the score for target date ``D`` only uses store dates strictly
before ``D`` inside ``selection.window_days`` sessions.  The daily aggregate
cache (``params/<st>_daily_agg.parquet``) is keyed by (Date, param_set_id,
QuoteCode) so appending a day never changes an old day's aggregates.
"""

from __future__ import annotations

import datetime as _dt
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ..framework.dataset import _atomic_write_json, _atomic_write_parquet
from ..framework.events import DataRoots
from ..framework.registry import GateSpec, SignalConfig, SignalRegistry, param_set_id
from ..framework.store import build_envelope_index, build_store, read_store, store_dates


# --------------------------------------------------------------------------- #
# condition parameters (rolled daily)
# --------------------------------------------------------------------------- #


PARAM_COLUMNS = (
    "Date",
    "param_set_id",
    "params_json",
    "is_default",
    "score",
    "n_rows",
    "n_stock_days",
    "window_start",
    "window_end",
    "window_days_used",
    "is_winner",
    "reason",
)


PARAM_SCHEMA = {
    "Date": pl.String,
    "param_set_id": pl.String,
    "params_json": pl.String,
    "is_default": pl.Boolean,
    "score": pl.Float64,
    "n_rows": pl.Int64,
    "n_stock_days": pl.Int64,
    "window_start": pl.String,
    "window_end": pl.String,
    "window_days_used": pl.Int64,
    "is_winner": pl.Boolean,
    "reason": pl.String,
}


def condition_params_path(params_dir: Path | str, signal: SignalConfig) -> Path:
    return Path(params_dir) / f"{signal.st_name}_condition.parquet"


def daily_agg_path(params_dir: Path | str, signal: SignalConfig) -> Path:
    return Path(params_dir) / f"{signal.st_name}_daily_agg.parquet"


def _daily_agg_manifest_path(params_dir: Path | str, signal: SignalConfig) -> Path:
    return Path(params_dir) / f"{signal.st_name}_daily_agg.manifest.json"


def _pool_identity(signal: SignalConfig) -> dict[str, object]:
    """What the daily aggregate cache depends on besides the grid."""

    return {
        "gate": signal.gate.stable_hash(),
        "point_filter": signal.point_filter.stable_hash() if signal.point_filter else None,
        "label_col": signal.selection.label_col,
    }


def candidate_pool(signal: SignalConfig, rows: pl.DataFrame) -> pl.DataFrame:
    """Rows the condition is scored on: layer-2 point filter, then the signal's labeled pool."""

    if rows.is_empty():
        return rows
    if signal.point_filter is not None:
        rows = rows.filter(signal.point_filter.mask(rows))
        if rows.is_empty():
            return rows
    return signal.selection_pool(rows)


def _param_rows(gate: GateSpec) -> list[tuple[str, dict[str, object], bool]]:
    default_id = gate.default_param_set_id
    rows = []
    for params in gate.expand_grid():
        pid = param_set_id(params)
        rows.append((pid, params, pid == default_id))
    return rows


def _empty_daily_agg() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "Date": pl.String,
            "param_set_id": pl.String,
            "QuoteCode": pl.String,
            "sum_label": pl.Float64,
            "n_rows": pl.UInt32,
        }
    )


def compute_daily_aggregates(signal: SignalConfig, pool: pl.DataFrame) -> pl.DataFrame:
    """Per (Date, param_set, QuoteCode) label sums for one or more store days.

    ``pool`` must already be the signal's candidate pool (see
    :func:`candidate_pool`) for the dates it contains.
    """

    label_col = signal.selection.label_col
    if pool.is_empty():
        return _empty_daily_agg()
    if label_col not in pool.columns:
        raise ValueError(f"{signal.condition_id}: selection pool lacks {label_col!r}")
    parts = []
    for pid, params, _ in _param_rows(signal.gate):
        mask = signal.gate.mask(pool, params)
        gated = pool.filter(mask & pl.col(label_col).is_not_null() & pl.col(label_col).is_finite())
        if gated.is_empty():
            continue
        parts.append(
            gated.group_by(["Date", "QuoteCode"])
            .agg(pl.col(label_col).sum().alias("sum_label"), pl.len().alias("n_rows"))
            .with_columns(pl.lit(pid).alias("param_set_id"))
            .select("Date", "param_set_id", "QuoteCode", "sum_label", "n_rows")
        )
    if not parts:
        return _empty_daily_agg()
    return (
        pl.concat(parts, how="vertical")
        .with_columns(pl.col("n_rows").cast(pl.UInt32))
        .sort(["Date", "param_set_id", "QuoteCode"])
    )


def _window_scores(agg: pl.DataFrame, window_dates: Sequence[str], signal: SignalConfig) -> pl.DataFrame:
    sel = signal.selection
    window = agg.filter(pl.col("Date").is_in(list(window_dates)))
    if window.is_empty():
        return pl.DataFrame(
            schema={"param_set_id": pl.String, "score": pl.Float64, "n_rows": pl.Int64, "n_stock_days": pl.Int64}
        )
    if sel.objective == "stockday_mean_label":
        return window.group_by("param_set_id").agg(
            (pl.col("sum_label") / pl.col("n_rows")).mean().alias("score"),
            pl.col("n_rows").sum().cast(pl.Int64).alias("n_rows"),
            pl.len().cast(pl.Int64).alias("n_stock_days"),
        )
    return window.group_by("param_set_id").agg(
        (pl.col("sum_label").sum() / pl.col("n_rows").sum()).alias("score"),
        pl.col("n_rows").sum().cast(pl.Int64).alias("n_rows"),
        pl.len().cast(pl.Int64).alias("n_stock_days"),
    )


def _choose_winner(
    scored: pl.DataFrame, gate: GateSpec, signal: SignalConfig, window_days_used: int = 1
) -> tuple[str, str]:
    """Return (winner param_set_id, reason).

    Support floors: absolute rows / stock-days in the window and stock-days per
    window day, so a parameter set that fires on a handful of stock-days cannot
    win on the variance of its tiny sample.
    """

    default_id = gate.default_param_set_id
    if scored.is_empty():
        return default_id, "no_history"
    sel = signal.selection
    eligible = scored.filter(
        (pl.col("n_rows") >= sel.min_rows)
        & (pl.col("n_stock_days") >= sel.min_stock_days)
        & (pl.col("n_stock_days") >= sel.min_stock_days_per_day * max(window_days_used, 1))
    )
    if eligible.is_empty():
        return default_id, "insufficient_support"
    if len(gate.grid) == 0:
        return default_id, "fixed"
    best = eligible["score"].max()
    tied = eligible.filter(pl.col("score") == best)["param_set_id"].to_list()
    if default_id in tied:
        return default_id, "tie_default" if len(tied) > 1 else "max_score"
    return sorted(tied)[0], "max_score"


def build_condition_params(
    signal: SignalConfig,
    *,
    store_dir: Path | str,
    params_dir: Path | str,
    next_date: str | None = None,
    force: bool = False,
    verbose: bool = True,
) -> pl.DataFrame:
    """Roll the condition grid over the store and write the daily parameter table."""

    params_dir = Path(params_dir)
    params_dir.mkdir(parents=True, exist_ok=True)
    gate = signal.gate
    dates = store_dates(store_dir)
    if not dates:
        raise FileNotFoundError(f"store has no days: {store_dir}")

    agg_path = daily_agg_path(params_dir, signal)
    manifest_path = _daily_agg_manifest_path(params_dir, signal)
    identity = _pool_identity(signal)
    recorded = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            recorded = json.load(handle).get("pool", {})
    cache = pl.read_parquet(agg_path) if agg_path.exists() and not force else _empty_daily_agg()
    valid_ids = {pid for pid, _, _ in _param_rows(gate)}
    if cache.height:
        cached_ids = set(cache["param_set_id"].unique().to_list())
        if not cached_ids.issubset(valid_ids):
            if verbose:
                print(f"[{signal.condition_id}] grid changed; rebuilding daily aggregates")
            cache = _empty_daily_agg()
        elif recorded != identity:
            if verbose:
                print(f"[{signal.condition_id}] condition / point filter changed; rebuilding daily aggregates")
            cache = _empty_daily_agg()
    cached_dates = set(cache["Date"].unique().to_list()) if cache.height else set()
    todo = [d for d in dates if d not in cached_dates]
    if verbose:
        print(f"[{signal.condition_id}] params: {len(todo)} store day(s) to aggregate, grid={len(valid_ids)}")
    new_parts = []
    for position, date in enumerate(todo, start=1):
        rows = read_store(store_dir, dates=[date], condition=signal)
        agg = compute_daily_aggregates(signal, candidate_pool(signal, rows))
        if agg.height:
            new_parts.append(agg)
        if verbose and (position % 20 == 0 or position == len(todo)):
            print(f"  aggregated {position}/{len(todo)} ({date})")
    if new_parts or not agg_path.exists() or force:
        cache = pl.concat([cache, *new_parts], how="vertical").sort(["Date", "param_set_id", "QuoteCode"])
        _atomic_write_parquet(cache, agg_path)
        _atomic_write_json({"condition_id": signal.condition_id, "pool": identity}, manifest_path)

    targets = list(dates)
    if next_date is not None and next_date not in targets:
        targets.append(next_date)
        targets.sort()
    previous_path = condition_params_path(params_dir, signal)
    previous = pl.read_parquet(previous_path) if previous_path.exists() else None
    previous_winners = (
        dict(zip(previous.filter(pl.col("is_winner"))["Date"], previous.filter(pl.col("is_winner"))["param_set_id"]))
        if previous is not None and previous.height
        else {}
    )

    param_rows = _param_rows(gate)
    out_rows: list[dict[str, object]] = []
    changed = 0
    for target in targets:
        window = [d for d in dates if d < target][-signal.selection.window_days :]
        scored = _window_scores(cache, window, signal)
        winner, reason = _choose_winner(scored, gate, signal, window_days_used=len(window))
        score_map = {r["param_set_id"]: r for r in scored.to_dicts()}
        for pid, params, is_default in param_rows:
            s = score_map.get(pid, {})
            out_rows.append(
                {
                    "Date": target,
                    "param_set_id": pid,
                    "params_json": json.dumps(params, sort_keys=True, default=str),
                    "is_default": is_default,
                    "score": s.get("score"),
                    "n_rows": int(s.get("n_rows", 0) or 0),
                    "n_stock_days": int(s.get("n_stock_days", 0) or 0),
                    "window_start": window[0] if window else None,
                    "window_end": window[-1] if window else None,
                    "window_days_used": len(window),
                    "is_winner": pid == winner,
                    "reason": reason if pid == winner else None,
                }
            )
        if previous_winners.get(target) not in (None, winner):
            changed += 1
    table = pl.DataFrame(out_rows, schema=PARAM_SCHEMA).select(list(PARAM_COLUMNS))
    _atomic_write_parquet(table, previous_path)
    if verbose:
        winners = table.filter(pl.col("is_winner"))
        latest = winners.tail(1).to_dicts()[0] if winners.height else None
        print(
            f"[{signal.condition_id}] params saved: {len(targets)} date(s), "
            f"latest={latest['Date'] if latest else None} winner={latest['param_set_id'] if latest else None} "
            f"({latest['reason'] if latest else None}); winners changed vs previous file: {changed}"
        )
    return table


def read_condition_params(params_dir: Path | str, signal: SignalConfig) -> pl.DataFrame:
    path = condition_params_path(params_dir, signal)
    if not path.exists():
        return pl.DataFrame(schema={c: pl.String for c in PARAM_COLUMNS})
    return pl.read_parquet(path)


def params_for_date(table: pl.DataFrame, signal: SignalConfig, date: str) -> tuple[dict[str, object], str, str]:
    """Return (params, param_set_id, source) in effect for ``date``.

    Falls back to the latest winner strictly before ``date``, then to the gate
    defaults, so a missing parameter day never stops index construction.
    """

    default = dict(signal.gate.params)
    if table.is_empty() or "is_winner" not in table.columns:
        return default, signal.gate.default_param_set_id, "default"
    winners = table.filter(pl.col("is_winner"))
    exact = winners.filter(pl.col("Date") == date)
    if exact.height:
        row = exact.row(0, named=True)
        return json.loads(row["params_json"]), row["param_set_id"], "exact"
    prior = winners.filter(pl.col("Date") < date).sort("Date")
    if prior.height:
        row = prior.row(-1, named=True)
        return json.loads(row["params_json"]), row["param_set_id"], f"carry:{row['Date']}"
    return default, signal.gate.default_param_set_id, "default"


def winner_history(table: pl.DataFrame) -> pl.DataFrame:
    if table.is_empty():
        return table
    return table.filter(pl.col("is_winner")).select(
        "Date", "param_set_id", "params_json", "score", "n_rows", "n_stock_days", "reason"
    )


# --------------------------------------------------------------------------- #
# rolling first-live evaluation: does the prior ranking predict next-period performance?
# --------------------------------------------------------------------------- #


def _realized_scores(agg: pl.DataFrame, signal: SignalConfig) -> pl.DataFrame:
    sel = signal.selection
    if sel.objective == "stockday_mean_label":
        return agg.group_by(["Date", "param_set_id"]).agg(
            (pl.col("sum_label") / pl.col("n_rows")).mean().alias("realized"),
            pl.col("n_rows").sum().cast(pl.Int64).alias("realized_rows"),
            pl.len().cast(pl.Int64).alias("realized_stock_days"),
        )
    return agg.group_by(["Date", "param_set_id"]).agg(
        (pl.col("sum_label").sum() / pl.col("n_rows").sum()).alias("realized"),
        pl.col("n_rows").sum().cast(pl.Int64).alias("realized_rows"),
        pl.len().cast(pl.Int64).alias("realized_stock_days"),
    )


def evaluate_rolling_selection(
    signal: SignalConfig,
    *,
    params_dir: Path | str,
    top_share: float = 0.2,
    min_sets: int = 4,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Prior-rank predictiveness of the condition grid, day by day.

    For every date with prior scores for at least ``min_sets`` parameter sets and
    realized scores on that date, report the Spearman correlation between prior
    score and realized score across sets, and the realized objective of the
    winner, the default set, the oracle best, and the prior top / middle /
    bottom tiers.  Returns ``(per_date, per_month)``.
    """

    table = read_condition_params(params_dir, signal)
    agg_path = daily_agg_path(params_dir, signal)
    if table.is_empty() or not agg_path.exists():
        raise FileNotFoundError("run build_condition_params first")
    realized = _realized_scores(pl.read_parquet(agg_path), signal)
    default_id = signal.gate.default_param_set_id
    sel = signal.selection
    joined = (
        table.filter(
            pl.col("score").is_not_null()
            & (pl.col("n_rows") >= sel.min_rows)
            & (pl.col("n_stock_days") >= sel.min_stock_days)
            & (pl.col("n_stock_days") >= sel.min_stock_days_per_day * pl.col("window_days_used").clip(lower_bound=1))
        )
        .join(realized, on=["Date", "param_set_id"], how="inner")
    )
    rows: list[dict[str, object]] = []
    for date, part in joined.group_by("Date", maintain_order=True):
        date = date[0] if isinstance(date, tuple) else date
        if part.height < min_sets:
            continue
        part = part.sort("score", descending=True).with_row_index("prior_rank")
        n = part.height
        k = max(1, int(round(n * top_share)))
        prior = part["score"].rank(method="average").to_numpy()
        real = part["realized"].rank(method="average").to_numpy()
        rho = float(((prior - prior.mean()) * (real - real.mean())).sum()
                    / max(((prior - prior.mean()) ** 2).sum() ** 0.5 * ((real - real.mean()) ** 2).sum() ** 0.5, 1e-12))
        winner = part.filter(pl.col("is_winner"))
        default = part.filter(pl.col("param_set_id") == default_id)
        rows.append(
            {
                "Date": date,
                "sets": n,
                "spearman": rho,
                "winner_realized": float(winner["realized"][0]) if winner.height else None,
                "default_realized": float(default["realized"][0]) if default.height else None,
                "oracle_realized": float(part["realized"].max()),
                "top_realized": float(part.head(k)["realized"].mean()),
                "middle_realized": float(part.slice(k, max(n - 2 * k, 1))["realized"].mean()),
                "bottom_realized": float(part.tail(k)["realized"].mean()),
                "winner_prior_rank_of_realized": int(
                    part.sort("realized", descending=True).with_row_index("r").filter(pl.col("is_winner"))["r"][0]
                ) + 1 if winner.height else None,
            }
        )
    per_date = pl.DataFrame(rows).sort("Date")
    if per_date.is_empty():
        return per_date, per_date
    per_month = (
        per_date.with_columns(pl.col("Date").str.slice(0, 6).alias("month"))
        .group_by("month", maintain_order=True)
        .agg(
            pl.len().alias("days"),
            pl.col("spearman").mean().alias("spearman_mean"),
            (pl.col("spearman") > 0).mean().alias("spearman_pos_share"),
            pl.col("winner_realized").mean(),
            pl.col("default_realized").mean(),
            pl.col("top_realized").mean(),
            pl.col("middle_realized").mean(),
            pl.col("bottom_realized").mean(),
            pl.col("oracle_realized").mean(),
            (pl.col("winner_realized") > pl.col("default_realized")).mean().alias("winner_beats_default_share"),
        )
    )
    return per_date, per_month


# --------------------------------------------------------------------------- #
# signal index: each day's condition parameters applied to the tagged store
# --------------------------------------------------------------------------- #


INDEX_COLUMNS = ("Date", "QuoteCode", "ChannelSeq", "param_set_id")


def signal_index_path(index_dir: Path | str, signal: SignalConfig) -> Path:
    return Path(index_dir) / f"{signal.st_name}_index.parquet"


def _empty_index() -> pl.DataFrame:
    return pl.DataFrame(
        schema={"Date": pl.String, "QuoteCode": pl.String, "ChannelSeq": pl.UInt64, "param_set_id": pl.String}
    )


def read_signal_index(index_dir: Path | str, signal: SignalConfig, *, dates: Sequence[str] | None = None) -> pl.DataFrame:
    path = signal_index_path(index_dir, signal)
    if not path.exists():
        return _empty_index()
    frame = pl.read_parquet(path)
    if dates is not None:
        frame = frame.filter(pl.col("Date").is_in(list(dates)))
    return frame


def build_signal_index(
    signal: SignalConfig,
    *,
    store_dir: Path | str,
    params_dir: Path | str,
    index_dir: Path | str,
    start: str | None = None,
    end: str | None = None,
    force: bool = False,
    verbose: bool = True,
) -> pl.DataFrame:
    """Gate every store day with that day's parameters and persist the keys.

    Dates already present are skipped unless ``force``; the latest existing date
    is always rebuilt (its store day may have been refreshed) and so is any date
    whose stored parameter set no longer matches the winner.  A changed gate
    definition rebuilds the whole index.
    """

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    dates = [d for d in store_dates(store_dir) if (start is None or d >= start) and (end is None or d <= end)]
    existing = read_signal_index(index_dir, signal)
    manifest_path = index_dir / f"{signal.st_name}_index.manifest.json"
    recorded_gate = None
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            recorded_gate = (json.load(handle).get("gate") or {}).get("hash")
    if existing.height and recorded_gate not in (None, signal.gate.stable_hash()):
        if verbose:
            print(f"[{signal.condition_id}] gate definition changed; rebuilding the whole index")
        force = True
    params_table = read_condition_params(params_dir, signal)
    done = set(existing["Date"].unique().to_list()) if existing.height else set()
    latest_done = max(done) if done else None
    if force:
        todo = dates
    else:
        todo = [d for d in dates if d not in done or d == latest_done]
        if existing.height and params_table.height:
            stored = existing.group_by("Date").agg(pl.col("param_set_id").first())
            for date, pid in stored.rows():
                if date in dates and date not in todo and params_for_date(params_table, signal, date)[1] != pid:
                    todo.append(date)
            todo.sort()
    if existing.height:
        existing = existing.filter(~pl.col("Date").is_in(todo))
    parts = [existing] if existing.height else []
    usage: dict[str, dict[str, object]] = {}
    if verbose:
        print(f"[{signal.condition_id}] index: {len(todo)} day(s) to gate")
    for position, date in enumerate(todo, start=1):
        params, pid, source = params_for_date(params_table, signal, date)
        rows = read_store(store_dir, dates=[date], condition=signal)
        if rows.is_empty():
            continue
        mask = signal.gate.mask(rows, params)
        selected = rows.filter(mask).select("Date", "QuoteCode", "ChannelSeq").with_columns(pl.lit(pid).alias("param_set_id"))
        parts.append(selected)
        usage[date] = {"param_set_id": pid, "source": source, "rows": int(selected.height)}
        if verbose and (position % 20 == 0 or position == len(todo)):
            print(f"  gated {position}/{len(todo)} ({date}: {selected.height:,} rows, params={pid} [{source}])")
    if parts:
        index = pl.concat(parts, how="vertical").select(list(INDEX_COLUMNS)).sort(["Date", "QuoteCode", "ChannelSeq"])
    else:
        index = read_signal_index(index_dir, signal)
    _atomic_write_parquet(index, signal_index_path(index_dir, signal))
    _atomic_write_json(
        {
            "condition_id": signal.condition_id,
            "condition_hash": signal.stable_hash(),
            "gate": {"gate_id": signal.gate.gate_id, "version": signal.gate.version, "hash": signal.gate.stable_hash()},
            "rows": int(index.height),
            "dates": int(index["Date"].n_unique()) if index.height else 0,
            "watermark": index["Date"].max() if index.height else None,
            "last_run": {"dates": usage, "at": _dt.datetime.now().isoformat(timespec="seconds")},
        },
        manifest_path,
    )
    if verbose:
        print(f"[{signal.condition_id}] index saved: {index.height:,} rows -> {signal_index_path(index_dir, signal)}")
    return index


# --------------------------------------------------------------------------- #
# orchestration (what the CLI runs)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EnvelopeLayerReport:
    signals: tuple[str, ...]
    index_rows: dict[str, int]


def build_envelope_layer(
    registry: SignalRegistry,
    *,
    roots: DataRoots,
    dirs: Mapping[str, Path],
    only: Sequence[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    rescan: bool = False,
    force: bool = False,
    next_date: str | None = None,
    build_store_days: bool = True,
    verbose: bool = True,
) -> EnvelopeLayerReport:
    """Envelope scan → shared store → daily condition parameters → signal index.

    ``rescan`` re-runs the raw envelope scan for the bounded range; ``force``
    recomputes the parameter aggregates and regates the index range.
    """

    signals = registry.select(only)
    for signal in signals:
        build_envelope_index(signal, roots=roots, envelope_dir=dirs["envelope"], start=start, end=end, force=rescan, verbose=verbose)
    if build_store_days:
        build_store(
            registry, roots=roots, store_dir=dirs["store"], envelope_dir=dirs["envelope"],
            start=start, end=end, only=only, verbose=verbose,
        )
    rows: dict[str, int] = {}
    for signal in signals:
        build_condition_params(signal, store_dir=dirs["store"], params_dir=dirs["params"], next_date=next_date, force=force, verbose=verbose)
        index = build_signal_index(
            signal, store_dir=dirs["store"], params_dir=dirs["params"], index_dir=dirs["index"],
            start=start, end=end, force=force, verbose=verbose,
        )
        rows[signal.st_name] = int(index.height)
    return EnvelopeLayerReport(tuple(s.st_name for s in signals), rows)


__all__ = [
    "EnvelopeLayerReport",
    "INDEX_COLUMNS",
    "PARAM_COLUMNS",
    "build_condition_params",
    "build_envelope_layer",
    "build_signal_index",
    "candidate_pool",
    "compute_daily_aggregates",
    "condition_params_path",
    "daily_agg_path",
    "evaluate_rolling_selection",
    "params_for_date",
    "read_condition_params",
    "read_signal_index",
    "signal_index_path",
    "winner_history",
]
