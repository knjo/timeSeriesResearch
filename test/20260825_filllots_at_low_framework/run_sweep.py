"""Run the FillLots_atLow candidate family through the generic framework."""

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
    PrimaryEstimand,
    build_eligible_labeled_pool,
    compare_conditions_to_shared_control,
    summarize_cohort,
)
from src.research.timeSeries.framework.events import DataRoots, EventDaySources
from src.research.timeSeries.framework.labels import DirectionalLabelSpec
from src.research.timeSeries.framework.profiles.benchmark import (
    LONG_TAKER_TP109_V1,
)
from src.research.timeSeries.framework.profiles.filllots_at_low import (
    CANDIDATE_DESCRIPTIONS,
    CANDIDATE_IDS,
    FILLLOTS_AT_LOW_FAMILY_ID,
    FILLLOTS_AT_LOW_FAMILY_VERSION,
    FILLLOTS_AT_LOW_STATE_FAMILY_ID,
    FILLLOTS_AT_LOW_STATE_FAMILY_VERSION,
    STATE_CANDIDATE_DESCRIPTIONS,
    STATE_CANDIDATE_IDS,
    select_filllots_at_low_candidates,
    select_filllots_at_low_state_candidates,
)
from src.research.timeSeries.framework.profiles.negfill import (
    SHORT_TAKER_STOP108_V1,
)


PROJECT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("artifacts") / "sparse_v3_stocktime"
DEFAULT_STATE_OUTPUT_DIR = (
    Path(__file__).with_name("artifacts") / "state_recovery_sparse_v1"
)
LABEL_PROFILES = {
    "short-taker-stop108-v1": SHORT_TAKER_STOP108_V1,
    "long-taker-tp109-v1": LONG_TAKER_TP109_V1,
}
SOURCE_PATTERNS = {
    "tick_data": "*_StockTick.parquet",
    "pre_market": "*_preMarketData.parquet",
    "txf_tick": "*_TxfTick.parquet",
}
SWEEP_CONTRACT_ID = "filllots_at_low_bias_sweep/v3"
STATE_SWEEP_CONTRACT_ID = "filllots_at_low_state_bias_sweep/v1"
STOCK_TIME_BUCKET = "5m"
STOCK_TIME_EQUAL_COLUMN = "stock_time_equal_delta_z"
SOURCE_IDENTITY_COLUMN = "source_identities_json"
SOURCE_IDENTITY_SOURCES = ("tick_data", "pre_market", "txf_tick")
PRIMARY_ESTIMAND = PrimaryEstimand.SHARED_POOL


@dataclass(frozen=True)
class CandidateRegistry:
    registry_id: str
    contract_id: str
    family_id: str
    family_version: str
    candidate_ids: tuple[str, ...]
    descriptions: dict[str, str]


EVENT_SHAPING_REGISTRY = CandidateRegistry(
    registry_id="event-shaping",
    contract_id=SWEEP_CONTRACT_ID,
    family_id=FILLLOTS_AT_LOW_FAMILY_ID,
    family_version=FILLLOTS_AT_LOW_FAMILY_VERSION,
    candidate_ids=CANDIDATE_IDS,
    descriptions=CANDIDATE_DESCRIPTIONS,
)
STATE_REGISTRY = CandidateRegistry(
    registry_id="state",
    contract_id=STATE_SWEEP_CONTRACT_ID,
    family_id=FILLLOTS_AT_LOW_STATE_FAMILY_ID,
    family_version=FILLLOTS_AT_LOW_STATE_FAMILY_VERSION,
    candidate_ids=STATE_CANDIDATE_IDS,
    descriptions=STATE_CANDIDATE_DESCRIPTIONS,
)
CANDIDATE_REGISTRIES = {
    registry.registry_id: registry
    for registry in (EVENT_SHAPING_REGISTRY, STATE_REGISTRY)
}
DEFAULT_CANDIDATE_REGISTRY = EVENT_SHAPING_REGISTRY.registry_id


def _select_registry_candidates(
    day: EventDaySources,
    *,
    registry: CandidateRegistry,
    candidate_ids: tuple[str, ...],
) -> dict[str, pl.DataFrame]:
    if registry.registry_id == STATE_REGISTRY.registry_id:
        return dict(
            select_filllots_at_low_state_candidates(
                day,
                candidate_ids=candidate_ids,
            )
        )
    return dict(
        select_filllots_at_low_candidates(
            day,
            candidate_ids=candidate_ids,
        )
    )


def _source_dates(directory: Path, pattern: str) -> set[str]:
    return {path.name[:8] for path in directory.glob(pattern)}


def available_sessions(
    roots: DataRoots,
    *,
    start: str,
    end: str,
) -> list[str]:
    sources = {
        "tick_data": roots.tick_data_dir,
        "pre_market": roots.pre_market_dir,
        "txf_tick": roots.txf_tick_dir,
    }
    calendars = [
        _source_dates(sources[name], pattern)
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
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


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


def _pool_mean_z_by_quote(full_pool: pl.DataFrame) -> pl.DataFrame:
    return full_pool.group_by("QuoteCode").agg(
        pl.col("label_z").mean().alias("pool_quote_mean_z")
    )


def _stock_day_equal_delta_z(
    condition: pl.DataFrame,
    pool_by_quote: pl.DataFrame,
) -> float | None:
    """Average within-QuoteCode condition-minus-pool z for active stocks."""

    if condition.is_empty():
        return None
    condition_by_quote = condition.group_by("QuoteCode").agg(
        pl.col("label_z").mean().alias("condition_quote_mean_z")
    )
    matched = condition_by_quote.join(
        pool_by_quote,
        on="QuoteCode",
        how="inner",
        validate="1:1",
    )
    if len(matched) != len(condition_by_quote):
        raise RuntimeError("active condition QuoteCode escaped the full pool")
    value = matched.select(
        (
            pl.col("condition_quote_mean_z") - pl.col("pool_quote_mean_z")
        ).mean()
    ).item()
    return float(value) if value is not None else None


def _pool_mean_z_by_quote_time(full_pool: pl.DataFrame) -> pl.DataFrame:
    """Return complete-pool mean z for each QuoteCode and fixed time bucket."""

    return (
        full_pool.with_columns(
            pl.col("TransTime")
            .dt.truncate(STOCK_TIME_BUCKET)
            .alias("_stock_time_bucket")
        )
        .group_by("QuoteCode", "_stock_time_bucket")
        .agg(pl.col("label_z").mean().alias("pool_quote_time_mean_z"))
    )


def _stock_time_equal_delta_z(
    condition: pl.DataFrame,
    pool_by_quote_time: pl.DataFrame,
) -> float | None:
    """Equal-weight active 5-minute buckets, then equal-weight active stocks."""

    if condition.is_empty():
        return None
    condition_by_quote_time = (
        condition.with_columns(
            pl.col("TransTime")
            .dt.truncate(STOCK_TIME_BUCKET)
            .alias("_stock_time_bucket")
        )
        .group_by("QuoteCode", "_stock_time_bucket")
        .agg(pl.col("label_z").mean().alias("condition_quote_time_mean_z"))
    )
    matched = condition_by_quote_time.join(
        pool_by_quote_time,
        on=["QuoteCode", "_stock_time_bucket"],
        how="inner",
        validate="1:1",
    )
    if len(matched) != len(condition_by_quote_time):
        raise RuntimeError(
            "active condition QuoteCode/time bucket escaped the full pool"
        )
    stock_means = matched.with_columns(
        (
            pl.col("condition_quote_time_mean_z")
            - pl.col("pool_quote_time_mean_z")
        ).alias("bucket_delta_z")
    ).group_by("QuoteCode").agg(
        pl.col("bucket_delta_z").mean().alias("stock_time_equal_delta_z")
    )
    value = stock_means["stock_time_equal_delta_z"].mean()
    return float(value) if value is not None else None


def evaluate_day(
    date: str,
    *,
    roots: DataRoots,
    label_spec: DirectionalLabelSpec,
    candidate_ids: tuple[str, ...],
    control_sample_size: int | None,
    txf_loader: TxfDataLoader,
    source_identities_json: str,
    registry: CandidateRegistry = EVENT_SHAPING_REGISTRY,
) -> pl.DataFrame:
    pool_spec = reference_pool_spec(label_spec)
    pool = build_eligible_labeled_pool(
        date,
        roots=roots,
        pool_spec=pool_spec,
        label_spec=label_spec,
        txf_loader=txf_loader,
    )
    day = EventDaySources(date, roots)
    day.require("tick_data", "pre_market", "txf_tick")
    candidates = _select_registry_candidates(
        day,
        registry=registry,
        candidate_ids=candidate_ids,
    )
    comparison = compare_conditions_to_shared_control(
        date,
        pool,
        candidates,
        spec=ConditionComparisonSpec(
            condition_id=registry.family_id,
            condition_version=registry.family_version,
            control_sample_size=control_sample_size,
            control_population="eligible_pool",
            primary_estimand=PRIMARY_ESTIMAND,
        ),
        seed=int(date),
    )
    first = comparison[candidate_ids[0]]
    control_stats = _prefixed_stats(
        first.control,
        date=date,
        cohort="control",
        eligible_pool_rows=len(pool),
        prefix="control",
    )
    # Matched estimands always use the complete eligible pool.  Control
    # sampling, when explicitly requested, applies only to the shared-pool
    # comparison and must not make active QuoteCodes disappear here.
    pool_by_quote = _pool_mean_z_by_quote(pool)
    pool_by_quote_time = _pool_mean_z_by_quote_time(pool)
    rows: list[dict[str, object]] = []
    for candidate_id in candidate_ids:
        result = comparison[candidate_id]
        row: dict[str, object] = {
            "contract_id": registry.contract_id,
            "Date": date,
            "family_id": registry.family_id,
            "family_version": registry.family_version,
            "candidate_id": candidate_id,
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
            "primary_estimand": PRIMARY_ESTIMAND.value,
            "stock_time_bucket": STOCK_TIME_BUCKET,
            SOURCE_IDENTITY_COLUMN: source_identities_json,
            "eligible_pool_rows": result.eligible_pool_rows,
            "indexed_condition_rows": result.indexed_condition_rows,
            "condition_excluded_rows": result.condition_excluded_rows,
            "condition_unique_quotes": result.condition["QuoteCode"].n_unique(),
            "stock_day_equal_delta_z": _stock_day_equal_delta_z(
                result.condition,
                pool_by_quote,
            ),
            STOCK_TIME_EQUAL_COLUMN: _stock_time_equal_delta_z(
                result.condition,
                pool_by_quote_time,
            ),
            "selection_rate": result.selection_rate,
        }
        row.update(
            _prefixed_stats(
                result.condition,
                date=date,
                cohort="condition",
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


def summarize_sweep(daily: pl.DataFrame) -> pl.DataFrame:
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

    periods = (
        ("full", None),
        ("2025", "2025"),
        ("2026", "2026"),
    )
    rows: list[dict[str, object]] = []
    for candidate_id in sorted(daily["candidate_id"].unique().to_list()):
        candidate = enriched.filter(pl.col("candidate_id") == candidate_id)
        for period, year in periods:
            period_subset = (
                candidate
                if year is None
                else candidate.filter(pl.col("_year") == year)
            )
            subset = period_subset.filter(pl.col("condition_n_z") > 0)
            if subset.is_empty():
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "period": period,
                        "days": 0,
                        "events_per_day": None,
                        "quotes_per_day": None,
                        "selection_rate_median": None,
                        "delta_z_pooled": None,
                        "delta_z_daily_mean": None,
                        "daily_t": None,
                        "stock_day_equal_delta_z_daily_mean": None,
                        "stock_day_equal_daily_t": None,
                        "stock_time_equal_delta_z_daily_mean": None,
                        "stock_time_equal_daily_t": None,
                        "positive_days": 0,
                        "positive_months": 0,
                        "months": 0,
                        "delta_raw_bp_daily_mean": None,
                        "delta_residual_bp_daily_mean": None,
                    }
                )
                continue
            month_delta = subset.group_by("_month").agg(
                pl.col("daily_delta_z").mean().alias("delta")
            )
            pooled_condition_z = (
                subset["condition_sum_z"].sum()
                / subset["condition_n_z"].sum()
            )
            pooled_control_z = (
                subset["control_sum_z"].sum() / subset["control_n_z"].sum()
            )
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "period": period,
                    "days": len(subset),
                    "events_per_day": float(subset["condition_n"].mean()),
                    "quotes_per_day": float(
                        subset["condition_unique_quotes"].mean()
                    ),
                    "selection_rate_median": float(
                        subset["selection_rate"].median()
                    ),
                    "delta_z_pooled": float(
                        pooled_condition_z - pooled_control_z
                    ),
                    "delta_z_daily_mean": float(
                        subset["daily_delta_z"].mean()
                    ),
                    "daily_t": _daily_t(subset["daily_delta_z"]),
                    "stock_day_equal_delta_z_daily_mean": float(
                        subset["stock_day_equal_delta_z"].mean()
                    ),
                    "stock_day_equal_daily_t": _daily_t(
                        subset["stock_day_equal_delta_z"]
                    ),
                    "stock_time_equal_delta_z_daily_mean": float(
                        subset[STOCK_TIME_EQUAL_COLUMN].mean()
                    ),
                    "stock_time_equal_daily_t": _daily_t(
                        subset[STOCK_TIME_EQUAL_COLUMN]
                    ),
                    "positive_days": int(
                        (subset["daily_delta_z"] > 0.0).sum()
                    ),
                    "positive_months": int((month_delta["delta"] > 0.0).sum()),
                    "months": len(month_delta),
                    "delta_raw_bp_daily_mean": float(
                        subset["daily_delta_raw_bp"].mean()
                    ),
                    "delta_residual_bp_daily_mean": float(
                        subset["daily_delta_residual_bp"].mean()
                    ),
                }
            )
    return pl.DataFrame(rows).sort(
        "period",
        "delta_z_daily_mean",
        descending=[False, True],
    )


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


def _validate_cached_day(
    frame: pl.DataFrame,
    *,
    date: str,
    candidate_ids: tuple[str, ...],
    label_spec: DirectionalLabelSpec,
    pool_id: str,
    pool_version: str,
    control_sample_size: int | None,
    source_identities_json: str,
    registry: CandidateRegistry = EVENT_SHAPING_REGISTRY,
) -> None:
    checks = {
        "contract_id": registry.contract_id,
        "Date": date,
        "family_id": registry.family_id,
        "family_version": registry.family_version,
        "label_profile_hash": label_spec.fingerprint,
        "pool_id": pool_id,
        "pool_version": pool_version,
        "control_mode": (
            "full" if control_sample_size is None else "sampled"
        ),
        "control_sample_size": (
            0 if control_sample_size is None else control_sample_size
        ),
        "primary_estimand": PRIMARY_ESTIMAND.value,
        "stock_time_bucket": STOCK_TIME_BUCKET,
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
    if STOCK_TIME_EQUAL_COLUMN not in frame.columns:
        raise ValueError(
            f"cached {date}: missing metric column {STOCK_TIME_EQUAL_COLUMN}"
        )
    actual_candidates = frame["candidate_id"].to_list()
    if len(actual_candidates) != len(set(actual_candidates)):
        raise ValueError(f"cached {date}: duplicate candidate rows")
    if set(actual_candidates) != set(candidate_ids):
        raise ValueError(f"cached {date}: candidate manifest mismatch")


def run(args: argparse.Namespace) -> tuple[pl.DataFrame, pl.DataFrame]:
    roots = DataRoots.from_project_root(PROJECT_ROOT)
    sessions = available_sessions(roots, start=args.start, end=args.end)[:: args.step]
    if not sessions:
        raise ValueError("no complete source sessions in requested range")
    registry_id = getattr(args, "candidate_registry", DEFAULT_CANDIDATE_REGISTRY)
    registry = CANDIDATE_REGISTRIES[registry_id]
    unknown = sorted(set(args.candidates) - set(registry.candidate_ids))
    if unknown:
        raise ValueError(
            f"unknown candidates for registry {registry.registry_id}: {unknown}"
        )
    candidate_ids = tuple(
        candidate_id
        for candidate_id in registry.candidate_ids
        if candidate_id in set(args.candidates)
    )
    output_dir = Path(args.output_dir)
    daily_dir = output_dir / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    label_spec = LABEL_PROFILES[args.label_profile]
    pool_spec = reference_pool_spec(label_spec)
    txf_loader = TxfDataLoader(roots.txf_tick_dir)
    control_sample_size = None if args.control_sample == 0 else args.control_sample

    failures: list[dict[str, str]] = []
    quarantined_outputs: list[dict[str, str]] = []
    for index, date in enumerate(sessions, start=1):
        source_identities_json = _day_source_identities_json(date, roots)
        output_path = daily_dir / f"{date}.parquet"
        if output_path.exists() and not args.force:
            _validate_cached_day(
                pl.read_parquet(output_path),
                date=date,
                candidate_ids=candidate_ids,
                label_spec=label_spec,
                pool_id=pool_spec.pool_id,
                pool_version=pool_spec.pool_version,
                control_sample_size=control_sample_size,
                source_identities_json=source_identities_json,
                registry=registry,
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
                label_spec=label_spec,
                candidate_ids=candidate_ids,
                control_sample_size=control_sample_size,
                txf_loader=txf_loader,
                source_identities_json=source_identities_json,
                registry=registry,
            )
            _atomic_write_parquet(result, output_path)
            pool_rows = result["eligible_pool_rows"][0]
            max_events = result["condition_n"].max()
            print(
                f"[{index}/{len(sessions)}] {date}: "
                f"pool={pool_rows:,}, max_candidate={max_events:,}",
                flush=True,
            )
        except Exception as exc:
            failures.append({"Date": date, "error": repr(exc)})
            if not args.continue_on_error:
                raise RuntimeError(f"{date}: candidate sweep failed") from exc
            print(f"[{index}/{len(sessions)}] {date}: ERROR {exc}", flush=True)
        finally:
            gc.collect()

    daily = _read_daily_outputs(output_dir, dates=sessions)
    daily = daily.filter(
        pl.col("Date").is_in(sessions)
        & (pl.col("contract_id") == registry.contract_id)
        & (pl.col("family_id") == registry.family_id)
        & (pl.col("label_profile_id") == label_spec.profile_id)
        & (pl.col("label_profile_hash") == label_spec.fingerprint)
        & (pl.col("family_version") == registry.family_version)
        & (pl.col("pool_id") == pool_spec.pool_id)
        & (pl.col("pool_version") == pool_spec.pool_version)
        & (
            pl.col("control_mode")
            == ("full" if control_sample_size is None else "sampled")
        )
        & (pl.col("control_sample_size") == args.control_sample)
        & (pl.col("primary_estimand") == PRIMARY_ESTIMAND.value)
        & (pl.col("stock_time_bucket") == STOCK_TIME_BUCKET)
        & pl.col("candidate_id").is_in(candidate_ids)
    ).sort("Date", "candidate_id")
    expected_rows = len(sessions) * len(candidate_ids)
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
    summary = summarize_sweep(daily)
    # Keep the report-facing aggregate schema compatible with pre-identity v3
    # artifacts.  Per-day cache files and the run manifest retain the identity.
    _atomic_write_parquet(
        daily.drop(SOURCE_IDENTITY_COLUMN),
        output_dir / "daily_stats.parquet",
    )
    _atomic_write_parquet(summary, output_dir / "summary.parquet")
    _atomic_write_json(
        {
            "contract": registry.contract_id,
            "candidate_registry": registry.registry_id,
            "family_id": registry.family_id,
            "family_version": registry.family_version,
            "candidate_ids": list(candidate_ids),
            "candidate_definitions": {
                candidate_id: registry.descriptions[candidate_id]
                for candidate_id in candidate_ids
            },
            "label_profile_id": label_spec.profile_id,
            "label_profile_hash": label_spec.fingerprint,
            "pool_id": pool_spec.pool_id,
            "pool_version": pool_spec.pool_version,
            "stock_time_bucket": STOCK_TIME_BUCKET,
            "stock_time_equal_metric": STOCK_TIME_EQUAL_COLUMN,
            "source_identity": {
                "column": SOURCE_IDENTITY_COLUMN,
                "sources": list(SOURCE_IDENTITY_SOURCES),
                "fields": ["path", "size", "mtime_ns"],
            },
            "source_identities": source_identities,
            "stock_time_equal_weighting": (
                "condition-minus-pool mean z within QuoteCode/time bucket; "
                "equal active buckets within QuoteCode, then equal active QuoteCode"
            ),
            "start": args.start,
            "end": args.end,
            "step": args.step,
            "control_sample_size": control_sample_size,
            "primary_estimand": PRIMARY_ESTIMAND.value,
            "sessions": sessions,
            "failures": failures,
            "quarantined_daily_outputs": quarantined_outputs,
        },
        output_dir / "run_manifest.json",
    )
    return daily, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep FillLots_atLow candidate events on one shared pool"
    )
    parser.add_argument("--start", default="20250211")
    parser.add_argument("--end", default="20260824")
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--label-profile",
        choices=sorted(LABEL_PROFILES),
        default="short-taker-stop108-v1",
    )
    parser.add_argument(
        "--candidate-registry",
        choices=sorted(CANDIDATE_REGISTRIES),
        default=DEFAULT_CANDIDATE_REGISTRY,
        help="event-shaping keeps the 26-candidate sweep; state tests raw F signs",
    )
    parser.add_argument(
        "--candidate",
        dest="candidates",
        action="append",
        choices=sorted(set(CANDIDATE_IDS) | set(STATE_CANDIDATE_IDS)),
    )
    parser.add_argument(
        "--control-sample",
        type=int,
        default=0,
        help="0 uses the complete eligible pool",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    if args.step < 1:
        parser.error("--step must be positive")
    if args.control_sample < 0:
        parser.error("--control-sample cannot be negative")
    if args.candidates is None:
        args.candidates = list(
            CANDIDATE_REGISTRIES[args.candidate_registry].candidate_ids
        )
    if args.output_dir is None:
        args.output_dir = (
            DEFAULT_STATE_OUTPUT_DIR
            if args.candidate_registry == STATE_REGISTRY.registry_id
            else DEFAULT_OUTPUT_DIR
        )
    return args


if __name__ == "__main__":
    parsed = parse_args()
    _, sweep_summary = run(parsed)
    print(sweep_summary)
