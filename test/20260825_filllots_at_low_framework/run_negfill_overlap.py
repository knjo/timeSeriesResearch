"""Measure FillLots_atLow candidate overlap with the negFill reference event.

The runner builds one eligible labeled pool per day, partitions one selected
FillLots_atLow candidate against negFill on canonical event keys, and compares
all partitions with one identical control cohort.  Stock-day partitions are
also reported so exact-tick independence is not confused with same-stock-day
exposure.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from src.dataloader.txfDataLoader import TxfDataLoader
from src.research.timeSeries.condition_benchmark import reference_pool_spec
from src.research.timeSeries.framework.condition_study import (
    ConditionComparisonSpec,
    build_eligible_labeled_pool,
    compare_conditions_to_shared_control,
    summarize_cohort,
)
from src.research.timeSeries.framework.events import (
    DAY_KEY_COLS,
    DataRoots,
    EventDaySources,
    normalize_event_keys,
)
from src.research.timeSeries.framework.profiles.filllots_at_low import (
    CANDIDATE_IDS,
    FILLLOTS_AT_LOW_FAMILY_ID,
    FILLLOTS_AT_LOW_FAMILY_VERSION,
    select_filllots_at_low_candidates,
)
from src.research.timeSeries.framework.profiles.negfill import (
    SHORT_TAKER_STOP108_V1,
    select_negfill_events,
)


PROJECT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("artifacts") / "negfill_overlap_v1"
CONTRACT_ID = "filllots_at_low_negfill_overlap/v1"
NEGFILL_SELECTOR_VERSION = "negfill_spread_bid_move_v1"

COHORT_IDS = (
    "candidate",
    "exact_overlap_negfill",
    "exact_disjoint_negfill",
    "stockday_overlap_negfill",
    "stockday_disjoint_negfill",
)

SOURCE_PATTERNS = {
    "tick_data": "*_StockTick.parquet",
    "tick_feature": "*_tickFeature.parquet",
    "pre_market": "*_preMarketData.parquet",
    "txf_tick": "*_TxfTick.parquet",
}
SOURCE_IDENTITY_COLUMN = "source_identities_json"
SOURCE_IDENTITY_SOURCES = (
    "tick_data",
    "pre_market",
    "tick_feature",
    "txf_tick",
)


@dataclass(frozen=True)
class OverlapPartitions:
    """Canonical exact-key and same-stock-day partitions for one candidate."""

    candidate: pl.DataFrame
    negfill: pl.DataFrame
    exact_overlap: pl.DataFrame
    exact_disjoint: pl.DataFrame
    stockday_overlap: pl.DataFrame
    stockday_disjoint: pl.DataFrame
    overlapping_quotes: int

    @property
    def cohorts(self) -> dict[str, pl.DataFrame]:
        return {
            "candidate": self.candidate,
            "exact_overlap_negfill": self.exact_overlap,
            "exact_disjoint_negfill": self.exact_disjoint,
            "stockday_overlap_negfill": self.stockday_overlap,
            "stockday_disjoint_negfill": self.stockday_disjoint,
        }


def _canonical_day_keys(
    frame: pl.DataFrame | pl.LazyFrame,
    *,
    date: str,
) -> pl.DataFrame:
    """Normalize types, date, uniqueness, nulls, and sort order fail closed."""

    return normalize_event_keys(frame, date=date).select(DAY_KEY_COLS)


def _assert_partition(
    whole: pl.DataFrame,
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    name: str,
) -> None:
    if len(left) + len(right) != len(whole):
        raise RuntimeError(f"{name}: partition row counts do not add to whole")
    overlap = left.join(
        right,
        on=list(DAY_KEY_COLS),
        how="inner",
        validate="1:1",
    )
    if not overlap.is_empty():
        raise RuntimeError(f"{name}: partition branches overlap")


def partition_candidate_by_negfill(
    candidate_keys: pl.DataFrame | pl.LazyFrame,
    negfill_keys: pl.DataFrame | pl.LazyFrame,
    *,
    date: str,
) -> OverlapPartitions:
    """Build exact-key and same-stock-day candidate partitions.

    ``exact_overlap`` is the canonical ``C intersection N`` operation on
    ``(QuoteCode, ChannelSeq)``.  Stock-day overlap contains every candidate
    tick whose QuoteCode has at least one negFill event on the same date.
    """

    candidate = _canonical_day_keys(candidate_keys, date=date)
    negfill = _canonical_day_keys(negfill_keys, date=date)

    exact_marker = negfill.with_columns(
        pl.lit(True).alias("_is_exact_negfill")
    )
    exact_marked = (
        candidate.join(
            exact_marker,
            on=list(DAY_KEY_COLS),
            how="left",
            validate="1:1",
        )
        .with_columns(pl.col("_is_exact_negfill").fill_null(False))
    )
    if len(exact_marked) != len(candidate):
        raise RuntimeError("exact-key marker join changed candidate cardinality")
    exact_overlap = exact_marked.filter("_is_exact_negfill").select(
        DAY_KEY_COLS
    )
    exact_disjoint = exact_marked.filter(
        ~pl.col("_is_exact_negfill")
    ).select(DAY_KEY_COLS)

    negfill_quotes = negfill.select("QuoteCode").unique().with_columns(
        pl.lit(True).alias("_has_negfill_on_stock_day")
    )
    stockday_marked = (
        candidate.join(
            negfill_quotes,
            on="QuoteCode",
            how="left",
            validate="m:1",
        )
        .with_columns(pl.col("_has_negfill_on_stock_day").fill_null(False))
    )
    if len(stockday_marked) != len(candidate):
        raise RuntimeError("stock-day marker join changed candidate cardinality")
    stockday_overlap = stockday_marked.filter(
        "_has_negfill_on_stock_day"
    ).select(DAY_KEY_COLS)
    stockday_disjoint = stockday_marked.filter(
        ~pl.col("_has_negfill_on_stock_day")
    ).select(DAY_KEY_COLS)

    _assert_partition(
        candidate,
        exact_overlap,
        exact_disjoint,
        name="exact-key",
    )
    _assert_partition(
        candidate,
        stockday_overlap,
        stockday_disjoint,
        name="stock-day",
    )
    escaped_exact = exact_overlap.join(
        stockday_overlap,
        on=list(DAY_KEY_COLS),
        how="anti",
    )
    if not escaped_exact.is_empty():
        raise RuntimeError("exact negFill overlap escaped stock-day overlap")

    overlapping_quotes = len(
        candidate.select("QuoteCode")
        .unique()
        .join(
            negfill.select("QuoteCode").unique(),
            on="QuoteCode",
            how="inner",
            validate="1:1",
        )
    )
    return OverlapPartitions(
        candidate=candidate,
        negfill=negfill,
        exact_overlap=exact_overlap,
        exact_disjoint=exact_disjoint,
        stockday_overlap=stockday_overlap,
        stockday_disjoint=stockday_disjoint,
        overlapping_quotes=overlapping_quotes,
    )


def _source_dates(directory: Path, pattern: str) -> set[str]:
    return {path.name[:8] for path in directory.glob(pattern)}


def available_sessions(
    roots: DataRoots,
    *,
    start: str,
    end: str,
) -> list[str]:
    source_dirs = {
        "tick_data": roots.tick_data_dir,
        "tick_feature": roots.tick_feature_dir,
        "pre_market": roots.pre_market_dir,
        "txf_tick": roots.txf_tick_dir,
    }
    calendars = [
        _source_dates(source_dirs[name], pattern)
        for name, pattern in SOURCE_PATTERNS.items()
    ]
    common = set.intersection(*calendars)
    return sorted(date for date in common if start <= date <= end)


def _day_source_identities_json(date: str, roots: DataRoots) -> str:
    """Return the canonical identity of every source read for one study day."""

    day = EventDaySources(date, roots)
    day.require(*SOURCE_IDENTITY_SOURCES)
    identities: dict[str, dict[str, object]] = {}
    for source in SOURCE_IDENTITY_SOURCES:
        path = day.path(source)
        stat = path.stat()
        identities[source] = {
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return json.dumps(
        identities,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def _prefixed_stats(
    frame: pl.DataFrame,
    *,
    date: str,
    cohort: str,
    eligible_pool_rows: int,
    prefix: str,
) -> dict[str, object]:
    stats = summarize_cohort(
        frame,
        date=date,
        cohort=cohort,
        eligible_pool_rows=eligible_pool_rows,
    )
    return {
        f"{prefix}_{name}": value
        for name, value in stats.items()
        if name not in {"Date", "cohort", "grid", "n_eligible_pool"}
    }


def evaluate_day(
    date: str,
    *,
    roots: DataRoots,
    candidate_id: str,
    control_sample_size: int | None,
    txf_loader: TxfDataLoader,
    source_identities_json: str,
) -> pl.DataFrame:
    """Evaluate all overlap partitions against one labeled pool and control."""

    label_spec = SHORT_TAKER_STOP108_V1
    pool_spec = reference_pool_spec(label_spec)
    pool = build_eligible_labeled_pool(
        date,
        roots=roots,
        pool_spec=pool_spec,
        label_spec=label_spec,
        txf_loader=txf_loader,
    )

    day = EventDaySources(date, roots)
    day.require("tick_data", "tick_feature", "pre_market", "txf_tick")
    candidates = select_filllots_at_low_candidates(
        day,
        candidate_ids=(candidate_id,),
    )
    partitions = partition_candidate_by_negfill(
        candidates[candidate_id],
        select_negfill_events(day),
        date=date,
    )
    comparison = compare_conditions_to_shared_control(
        date,
        pool,
        partitions.cohorts,
        spec=ConditionComparisonSpec(
            condition_id="filllots_at_low_negfill_overlap",
            condition_version="1",
            control_sample_size=control_sample_size,
            control_population="eligible_pool",
        ),
        seed=int(date),
    )

    candidate_rows = len(partitions.candidate)
    exact_overlap_rows = len(partitions.exact_overlap)
    stockday_overlap_rows = len(partitions.stockday_overlap)
    candidate_quotes = partitions.candidate["QuoteCode"].n_unique()
    negfill_quotes = partitions.negfill["QuoteCode"].n_unique()
    shared_metadata: dict[str, object] = {
        "contract_id": CONTRACT_ID,
        "family_id": FILLLOTS_AT_LOW_FAMILY_ID,
        "family_version": FILLLOTS_AT_LOW_FAMILY_VERSION,
        "candidate_id": candidate_id,
        "negfill_selector_version": NEGFILL_SELECTOR_VERSION,
        "label_profile_id": label_spec.profile_id,
        "label_profile_hash": label_spec.fingerprint,
        "pool_id": pool_spec.pool_id,
        "pool_version": pool_spec.pool_version,
        "control_mode": (
            "full" if control_sample_size is None else "sampled"
        ),
        "control_sample_size": (
            0 if control_sample_size is None else control_sample_size
        ),
        SOURCE_IDENTITY_COLUMN: source_identities_json,
        "eligible_pool_rows": len(pool),
        "candidate_index_rows": candidate_rows,
        "negfill_index_rows": len(partitions.negfill),
        "exact_overlap_index_rows": exact_overlap_rows,
        "exact_disjoint_index_rows": len(partitions.exact_disjoint),
        "stockday_overlap_index_rows": stockday_overlap_rows,
        "stockday_disjoint_index_rows": len(partitions.stockday_disjoint),
        "candidate_unique_quotes": candidate_quotes,
        "negfill_unique_quotes": negfill_quotes,
        "overlapping_quotes": partitions.overlapping_quotes,
        "exact_overlap_rate_candidate": _rate(
            exact_overlap_rows,
            candidate_rows,
        ),
        "stockday_overlap_rate_candidate": _rate(
            stockday_overlap_rows,
            candidate_rows,
        ),
        "stockday_overlap_rate_quotes": _rate(
            partitions.overlapping_quotes,
            candidate_quotes,
        ),
    }

    first = comparison[COHORT_IDS[0]]
    control_stats = _prefixed_stats(
        first.control,
        date=date,
        cohort="shared_control",
        eligible_pool_rows=len(pool),
        prefix="control",
    )
    rows: list[dict[str, object]] = []
    for cohort_id in COHORT_IDS:
        result = comparison[cohort_id]
        row: dict[str, object] = {
            "Date": date,
            "cohort_id": cohort_id,
            **shared_metadata,
            "indexed_condition_rows": result.indexed_condition_rows,
            "condition_excluded_rows": result.condition_excluded_rows,
            "condition_unique_quotes": result.condition[
                "QuoteCode"
            ].n_unique(),
            "selection_rate": result.selection_rate,
        }
        row.update(
            _prefixed_stats(
                result.condition,
                date=date,
                cohort=cohort_id,
                eligible_pool_rows=len(pool),
                prefix="condition",
            )
        )
        row.update(control_stats)
        rows.append(row)
    return pl.DataFrame(rows)


def _daily_t(values: pl.Series) -> float | None:
    clean = values.drop_nulls()
    if len(clean) < 2:
        return None
    std = clean.std(ddof=1)
    if std is None or not math.isfinite(std) or std <= 0.0:
        return None
    return float(clean.mean() / std * math.sqrt(len(clean)))


def summarize_overlap(daily: pl.DataFrame) -> pl.DataFrame:
    """Aggregate additive statistics without weighting days implicitly."""

    enriched = daily.with_columns(
        pl.when(
            (pl.col("condition_n_z") > 0) & (pl.col("control_n_z") > 0)
        )
        .then(
            pl.col("condition_sum_z") / pl.col("condition_n_z")
            - pl.col("control_sum_z") / pl.col("control_n_z")
        )
        .alias("daily_delta_z"),
        pl.when(
            (pl.col("condition_n_z") > 0) & (pl.col("control_n_z") > 0)
        )
        .then(
            pl.col("condition_sum_raw_bp") / pl.col("condition_n_z")
            - pl.col("control_sum_raw_bp") / pl.col("control_n_z")
        )
        .alias("daily_delta_raw_bp"),
        pl.when(
            (pl.col("condition_n_z") > 0) & (pl.col("control_n_z") > 0)
        )
        .then(
            pl.col("condition_sum_residual_bp") / pl.col("condition_n_z")
            - pl.col("control_sum_residual_bp") / pl.col("control_n_z")
        )
        .alias("daily_delta_residual_bp"),
        pl.col("Date").str.slice(0, 4).alias("_year"),
        pl.col("Date").str.slice(0, 6).alias("_month"),
    )
    periods = (("full", None), ("2025", "2025"), ("2026", "2026"))
    rows: list[dict[str, object]] = []
    for cohort_id in COHORT_IDS:
        cohort = enriched.filter(pl.col("cohort_id") == cohort_id)
        for period, year in periods:
            period_rows = (
                cohort
                if year is None
                else cohort.filter(pl.col("_year") == year)
            )
            if period_rows.is_empty():
                continue
            bias_days = period_rows.filter(pl.col("daily_delta_z").is_not_null())
            candidate_total = int(period_rows["candidate_index_rows"].sum())
            exact_total = int(period_rows["exact_overlap_index_rows"].sum())
            stockday_total = int(
                period_rows["stockday_overlap_index_rows"].sum()
            )
            condition_n_z = int(period_rows["condition_n_z"].sum())
            control_n_z = int(period_rows["control_n_z"].sum())
            pooled_delta_z = None
            pooled_delta_raw_bp = None
            pooled_delta_residual_bp = None
            if condition_n_z > 0 and control_n_z > 0:
                pooled_delta_z = float(
                    period_rows["condition_sum_z"].sum() / condition_n_z
                    - period_rows["control_sum_z"].sum() / control_n_z
                )
                pooled_delta_raw_bp = float(
                    period_rows["condition_sum_raw_bp"].sum()
                    / condition_n_z
                    - period_rows["control_sum_raw_bp"].sum() / control_n_z
                )
                pooled_delta_residual_bp = float(
                    period_rows["condition_sum_residual_bp"].sum()
                    / condition_n_z
                    - period_rows["control_sum_residual_bp"].sum()
                    / control_n_z
                )
            month_delta = bias_days.group_by("_month").agg(
                pl.col("daily_delta_z").mean().alias("delta")
            )
            rows.append(
                {
                    "cohort_id": cohort_id,
                    "period": period,
                    "sessions": len(period_rows),
                    "bias_days": len(bias_days),
                    "indexed_condition_rows": int(
                        period_rows["indexed_condition_rows"].sum()
                    ),
                    "eligible_condition_rows": int(
                        period_rows["condition_n"].sum()
                    ),
                    "candidate_index_rows": candidate_total,
                    "exact_overlap_index_rows": exact_total,
                    "stockday_overlap_index_rows": stockday_total,
                    "exact_overlap_rate_candidate": _rate(
                        exact_total,
                        candidate_total,
                    ),
                    "stockday_overlap_rate_candidate": _rate(
                        stockday_total,
                        candidate_total,
                    ),
                    "delta_z_pooled": pooled_delta_z,
                    "delta_z_daily_mean": (
                        float(bias_days["daily_delta_z"].mean())
                        if not bias_days.is_empty()
                        else None
                    ),
                    "daily_t": _daily_t(bias_days["daily_delta_z"]),
                    "positive_days": int(
                        (bias_days["daily_delta_z"] > 0.0).sum()
                    ),
                    "positive_months": int(
                        (month_delta["delta"] > 0.0).sum()
                    ),
                    "months": len(month_delta),
                    "delta_raw_bp_pooled": pooled_delta_raw_bp,
                    "delta_residual_bp_pooled": pooled_delta_residual_bp,
                }
            )
    return pl.DataFrame(rows).sort("period", "cohort_id")


def _atomic_write_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temp_path = Path(handle.name)
    handle.close()
    try:
        frame.write_parquet(temp_path, compression="zstd")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _atomic_write_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
        encoding="utf-8",
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _validate_cached_day(
    frame: pl.DataFrame,
    *,
    date: str,
    candidate_id: str,
    control_sample_size: int | None,
    source_identities_json: str,
) -> None:
    expected_control_mode = (
        "full" if control_sample_size is None else "sampled"
    )
    checks = {
        "Date": date,
        "contract_id": CONTRACT_ID,
        "candidate_id": candidate_id,
        "label_profile_hash": SHORT_TAKER_STOP108_V1.fingerprint,
        "control_mode": expected_control_mode,
        "control_sample_size": (
            0 if control_sample_size is None else control_sample_size
        ),
        SOURCE_IDENTITY_COLUMN: source_identities_json,
    }
    for column, expected in checks.items():
        if column not in frame.columns:
            raise ValueError(f"cached {date}: missing metadata column {column}")
        actual = frame[column].unique().to_list()
        if actual != [expected]:
            raise ValueError(
                f"cached {date}: {column}={actual}, expected {expected!r}"
            )
    actual_cohorts = set(frame["cohort_id"].to_list())
    if actual_cohorts != set(COHORT_IDS) or len(frame) != len(COHORT_IDS):
        raise ValueError(f"cached {date}: incomplete or duplicate cohorts")


def _read_daily_outputs(
    output_dir: Path,
    *,
    dates: list[str] | None = None,
) -> pl.DataFrame:
    daily_dir = output_dir / "daily"
    if dates is None:
        paths = sorted(daily_dir.glob("*.parquet"))
    else:
        requested_paths = [daily_dir / f"{date}.parquet" for date in dates]
        paths = [path for path in requested_paths if path.exists()]
    if not paths:
        raise ValueError(f"no daily outputs under {output_dir}")
    return pl.concat([pl.read_parquet(path) for path in paths], how="vertical")


def _quarantine_daily_output(path: Path, quarantine_dir: Path) -> Path:
    """Atomically move a stale daily result aside without overwriting history."""

    quarantine_dir.mkdir(parents=True, exist_ok=True)
    destination = quarantine_dir / path.name
    suffix = 1
    while destination.exists():
        destination = quarantine_dir / f"{path.stem}.{suffix}{path.suffix}"
        suffix += 1
    os.replace(path, destination)
    return destination


def run(args: argparse.Namespace) -> tuple[pl.DataFrame, pl.DataFrame]:
    roots = DataRoots.from_project_root(PROJECT_ROOT)
    sessions = available_sessions(roots, start=args.start, end=args.end)[
        :: args.step
    ]
    if not sessions:
        raise ValueError("no complete source sessions in requested range")
    output_dir = Path(args.output_dir)
    daily_dir = output_dir / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    txf_loader = TxfDataLoader(roots.txf_tick_dir)
    control_sample_size = (
        None if args.control_sample == 0 else args.control_sample
    )

    failures: list[dict[str, str]] = []
    quarantined_outputs: list[dict[str, str]] = []
    for index, date in enumerate(sessions, start=1):
        source_identities_json = _day_source_identities_json(date, roots)
        output_path = daily_dir / f"{date}.parquet"
        if output_path.exists() and not args.force:
            cached = pl.read_parquet(output_path)
            _validate_cached_day(
                cached,
                date=date,
                candidate_id=args.candidate,
                control_sample_size=control_sample_size,
                source_identities_json=source_identities_json,
            )
            print(f"[{index}/{len(sessions)}] {date}: cached", flush=True)
            continue
        if output_path.exists():
            quarantine_path = _quarantine_daily_output(
                output_path,
                output_dir / "quarantine" / "daily",
            )
            quarantined_outputs.append(
                {
                    "Date": date,
                    "path": str(quarantine_path.relative_to(output_dir)),
                }
            )
        try:
            result = evaluate_day(
                date,
                roots=roots,
                candidate_id=args.candidate,
                control_sample_size=control_sample_size,
                txf_loader=txf_loader,
                source_identities_json=source_identities_json,
            )
            _atomic_write_parquet(result, output_path)
            candidate_rows = result["candidate_index_rows"][0]
            exact_rows = result["exact_overlap_index_rows"][0]
            stockday_rows = result["stockday_overlap_index_rows"][0]
            print(
                f"[{index}/{len(sessions)}] {date}: "
                f"candidate={candidate_rows:,}, exact={exact_rows:,}, "
                f"stockday={stockday_rows:,}",
                flush=True,
            )
        except Exception as exc:
            failures.append({"Date": date, "error": repr(exc)})
            if not args.continue_on_error:
                raise RuntimeError(f"{date}: overlap study failed") from exc
            print(f"[{index}/{len(sessions)}] {date}: ERROR {exc}", flush=True)
        finally:
            gc.collect()

    daily = _read_daily_outputs(output_dir, dates=sessions).filter(
        pl.col("Date").is_in(sessions)
        & (pl.col("candidate_id") == args.candidate)
        & (pl.col("label_profile_hash") == SHORT_TAKER_STOP108_V1.fingerprint)
        & (pl.col("control_sample_size") == args.control_sample)
    ).sort("Date", "cohort_id")
    expected_rows = len(sessions) * len(COHORT_IDS)
    if not failures and len(daily) != expected_rows:
        raise RuntimeError(
            f"combined daily rows={len(daily)}, expected {expected_rows}"
        )
    source_identity_rows = daily.select(
        "Date",
        SOURCE_IDENTITY_COLUMN,
    ).unique()
    if len(source_identity_rows) != daily["Date"].n_unique():
        raise RuntimeError("one study date contains multiple source identities")
    source_identities = {
        str(row["Date"]): json.loads(row[SOURCE_IDENTITY_COLUMN])
        for row in source_identity_rows.iter_rows(named=True)
    }
    summary = summarize_overlap(daily)
    # Keep the aggregate schema compatible with pre-identity overlap artifacts.
    _atomic_write_parquet(
        daily.drop(SOURCE_IDENTITY_COLUMN),
        output_dir / "daily_stats.parquet",
    )
    _atomic_write_parquet(summary, output_dir / "summary.parquet")
    _atomic_write_json(
        {
            "contract_id": CONTRACT_ID,
            "candidate_id": args.candidate,
            "candidate_family_id": FILLLOTS_AT_LOW_FAMILY_ID,
            "candidate_family_version": FILLLOTS_AT_LOW_FAMILY_VERSION,
            "negfill_selector_version": NEGFILL_SELECTOR_VERSION,
            "cohort_definitions": {
                "candidate": "all candidate canonical event keys",
                "exact_overlap_negfill": (
                    "candidate intersection negFill on QuoteCode,ChannelSeq"
                ),
                "exact_disjoint_negfill": (
                    "candidate minus negFill on QuoteCode,ChannelSeq"
                ),
                "stockday_overlap_negfill": (
                    "candidate ticks on QuoteCode-days containing negFill"
                ),
                "stockday_disjoint_negfill": (
                    "candidate ticks on QuoteCode-days without negFill"
                ),
            },
            "label_profile_id": SHORT_TAKER_STOP108_V1.profile_id,
            "label_profile_hash": SHORT_TAKER_STOP108_V1.fingerprint,
            "pool_id": reference_pool_spec(SHORT_TAKER_STOP108_V1).pool_id,
            "start": args.start,
            "end": args.end,
            "step": args.step,
            "control_sample_size": control_sample_size,
            "source_identity": {
                "column": SOURCE_IDENTITY_COLUMN,
                "sources": list(SOURCE_IDENTITY_SOURCES),
                "fields": ["path", "size", "mtime_ns"],
            },
            "source_identities": source_identities,
            "sessions": sessions,
            "failures": failures,
            "quarantined_daily_outputs": quarantined_outputs,
        },
        output_dir / "run_manifest.json",
    )
    return daily, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure exact-key and stock-day overlap between one "
            "FillLots_atLow candidate and negFill"
        )
    )
    parser.add_argument("--start", default="20250211")
    parser.add_argument("--end", default="20260824")
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument(
        "--candidate",
        required=True,
        choices=CANDIDATE_IDS,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--control-sample",
        type=int,
        default=0,
        help="0 uses the complete eligible labeled pool",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    if args.step < 1:
        parser.error("--step must be positive")
    if args.control_sample < 0:
        parser.error("--control-sample cannot be negative")
    if args.start > args.end:
        parser.error("--start cannot be after --end")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    _, overlap_summary = run(parsed)
    print(overlap_summary)
