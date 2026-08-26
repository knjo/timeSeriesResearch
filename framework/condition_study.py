"""Condition-vs-control studies on one shared eligible, labeled population.

The core invariant is simple: eligibility, label construction, hedge inputs,
and validity filtering happen once.  Event rows are then selected by an event
key semi-join and control rows are sampled from that exact same pool.  This
prevents a condition adapter and a baseline adapter from drifting apart.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Callable, Literal, TypeAlias

import polars as pl

from src.dataloader.txfDataLoader import TxfDataLoader, attach_txf_to_1330_return
from src.features.market_beta import add_txf_event_residual_vol

from .events import DAY_KEY_COLS, DataRoots
from .labels import (
    DEFAULT_LABEL_COLUMNS,
    DirectionalLabelSpec,
    add_directional_labels,
    required_label_columns,
)


Frame: TypeAlias = pl.DataFrame | pl.LazyFrame
FrameTransform: TypeAlias = Callable[[Frame], Frame]
ControlPopulation = Literal["eligible_pool", "non_event_pool"]


class PrimaryEstimand(StrEnum):
    """Research claim that determines which comparison is ranked first."""

    SHARED_POOL = "shared_pool"
    SAME_STOCK = "same_stock"
    SAME_STOCK_TIME = "same_stock_time"


class ConditionStudyContractError(ValueError):
    """Raised when condition and control cannot share one population contract."""


def _identity(frame: Frame) -> Frame:
    return frame


@dataclass(frozen=True)
class EligiblePoolSpec:
    """Versioned construction contract for a study's eligible tick population."""

    pool_id: str
    pool_version: str
    tick_columns: tuple[str, ...]
    pre_market_columns: tuple[str, ...] = ("QuoteCode",)
    transform_and_filter: FrameTransform = field(
        default=_identity, repr=False, compare=False
    )
    post_label_filter: FrameTransform = field(
        default=_identity, repr=False, compare=False
    )
    attach_txf: bool = True

    def __post_init__(self) -> None:
        if not self.pool_id or not self.pool_version:
            raise ValueError("pool_id and pool_version are required")
        for label, columns in (
            ("tick_columns", self.tick_columns),
            ("pre_market_columns", self.pre_market_columns),
        ):
            if len(columns) != len(set(columns)):
                raise ValueError(f"{label} cannot contain duplicates")
        missing_keys = set(DAY_KEY_COLS) - set(self.tick_columns)
        if missing_keys:
            raise ValueError(f"tick_columns missing event keys: {sorted(missing_keys)}")
        if "TransTime" not in self.tick_columns:
            raise ValueError("tick_columns must include TransTime")
        if "QuoteCode" not in self.pre_market_columns:
            raise ValueError("pre_market_columns must include QuoteCode")


@dataclass(frozen=True)
class ConditionComparisonSpec:
    """Sampling and interpretation choices, independent of the event family.

    The core always returns condition/control cohorts.  ``primary_estimand``
    records the study's ranking objective so a report cannot silently promote
    a secondary matched diagnostic.  The current long-holding discovery
    protocol uses ``shared_pool``; short-horizon studies may explicitly choose
    a horizon-aligned matched estimand in their adapter.
    """

    condition_id: str
    condition_version: str
    control_sample_size: int | None = 25_000
    control_population: ControlPopulation = "eligible_pool"
    bucket_columns: tuple[str, ...] = ()
    primary_estimand: PrimaryEstimand = PrimaryEstimand.SHARED_POOL

    def __post_init__(self) -> None:
        if not self.condition_id or not self.condition_version:
            raise ValueError("condition_id and condition_version are required")
        object.__setattr__(
            self,
            "primary_estimand",
            PrimaryEstimand(self.primary_estimand),
        )
        if self.control_sample_size is not None and self.control_sample_size < 1:
            raise ValueError("control_sample_size must be positive")
        if self.control_population not in {"eligible_pool", "non_event_pool"}:
            raise ValueError(
                "control_population must be eligible_pool or non_event_pool"
            )
        if len(self.bucket_columns) != len(set(self.bucket_columns)):
            raise ValueError("bucket_columns cannot contain duplicates")


@dataclass(frozen=True)
class ConditionDayResult:
    date: str
    eligible_pool_rows: int
    indexed_condition_rows: int
    condition_rows: int
    condition_excluded_rows: int
    control_rows: int
    condition: pl.DataFrame
    control: pl.DataFrame

    @property
    def selection_rate(self) -> float:
        if self.eligible_pool_rows == 0:
            return float("nan")
        return self.condition_rows / self.eligible_pool_rows


def _schema_names(frame: Frame) -> set[str]:
    if isinstance(frame, pl.LazyFrame):
        return set(frame.collect_schema().names())
    return set(frame.columns)


def _require_columns(frame: Frame, columns: set[str], context: str) -> None:
    missing = sorted(columns - _schema_names(frame))
    if missing:
        raise ConditionStudyContractError(f"{context}: missing columns {missing}")


def _validate_keys(frame: pl.DataFrame, context: str) -> pl.DataFrame:
    _require_columns(frame, set(DAY_KEY_COLS), context)
    keys = frame.select(DAY_KEY_COLS)
    null_count = keys.select(
        pl.any_horizontal([pl.col(column).is_null() for column in DAY_KEY_COLS])
        .sum()
        .alias("n")
    ).item()
    if null_count:
        raise ConditionStudyContractError(f"{context}: null event keys")
    duplicates = keys.group_by(DAY_KEY_COLS).len().filter(pl.col("len") > 1)
    if not duplicates.is_empty():
        raise ConditionStudyContractError(
            f"{context}: duplicate event keys; sample={duplicates.head(3).to_dicts()}"
        )
    return keys


def _apply_transform(
    transform: FrameTransform,
    frame: Frame,
    *,
    context: str,
) -> Frame:
    output = transform(frame)
    if not isinstance(output, (pl.DataFrame, pl.LazyFrame)):
        raise ConditionStudyContractError(
            f"{context} must return a polars DataFrame or LazyFrame"
        )
    return output


def build_eligible_labeled_pool(
    date: str,
    *,
    roots: DataRoots,
    pool_spec: EligiblePoolSpec,
    label_spec: DirectionalLabelSpec,
    txf_loader: TxfDataLoader | None = None,
) -> pl.DataFrame:
    """Build one population and retain only rows with a valid common label."""

    tick_path = roots.tick_data_dir / f"{date}_StockTick.parquet"
    pre_market_path = roots.pre_market_dir / f"{date}_preMarketData.parquet"
    for source, path in (("tickData", tick_path), ("preMarket", pre_market_path)):
        if not path.exists():
            raise FileNotFoundError(f"{date}: missing {source}: {path}")

    tick = pl.scan_parquet(tick_path)
    pre_market = pl.scan_parquet(pre_market_path)
    _require_columns(tick, set(pool_spec.tick_columns), f"{date} tickData")
    _require_columns(
        pre_market, set(pool_spec.pre_market_columns), f"{date} preMarket"
    )
    pre_keys = pre_market.select("QuoteCode").collect()
    if pre_keys["QuoteCode"].null_count():
        raise ConditionStudyContractError(f"{date} preMarket: null QuoteCode")
    pre_duplicates = (
        pre_keys.group_by("QuoteCode").len().filter(pl.col("len") > 1)
    )
    if not pre_duplicates.is_empty():
        raise ConditionStudyContractError(
            f"{date} preMarket: duplicate QuoteCode; "
            f"sample={pre_duplicates.head(3).to_dicts()}"
        )

    pool: Frame = tick.select(pool_spec.tick_columns).join(
        pre_market.select(pool_spec.pre_market_columns),
        on="QuoteCode",
        how="inner",
        validate="m:1",
        suffix="_pm",
    )
    pool = _apply_transform(
        pool_spec.transform_and_filter,
        pool,
        context=f"{date} pool transform_and_filter",
    )
    if isinstance(pool, pl.LazyFrame):
        pool = pool.collect()
    _validate_keys(pool, f"{date} eligible pool")

    if pool_spec.attach_txf:
        loader = txf_loader or TxfDataLoader(roots.txf_tick_dir)
        pool = add_txf_event_residual_vol(
            attach_txf_to_1330_return(pool, date, loader)
        )
    _require_columns(
        pool,
        set(required_label_columns(label_spec)),
        f"{date} label profile {label_spec.profile_id}",
    )
    labeled: Frame = add_directional_labels(pool, spec=label_spec)
    labeled = _apply_transform(
        pool_spec.post_label_filter,
        labeled,
        context=f"{date} post_label_filter",
    )
    if isinstance(labeled, pl.LazyFrame):
        labeled = labeled.collect()
    labeled = labeled.filter(pl.col(DEFAULT_LABEL_COLUMNS.valid))
    _validate_keys(labeled, f"{date} valid labeled pool")
    return labeled.with_columns(pl.lit(date).alias("Date"))


def compare_condition_to_control(
    date: str,
    eligible_pool: pl.DataFrame,
    event_keys: pl.DataFrame,
    *,
    spec: ConditionComparisonSpec,
    seed: int | None = None,
) -> ConditionDayResult:
    """Split event and control cohorts only after the shared pool is final."""

    pool_keys = _validate_keys(eligible_pool, f"{date} eligible labeled pool")
    if "Date" in event_keys.columns:
        wrong_date = event_keys.filter(pl.col("Date").cast(pl.String) != date)
        if not wrong_date.is_empty():
            raise ConditionStudyContractError(
                f"{date}: event_keys contain rows from another date"
            )
    keys = _validate_keys(event_keys, f"{date} condition event keys")
    # Events absent from the eligible pool are expected universe/validity
    # exclusions.  The semi-join is the single condition-selection operation.
    condition = eligible_pool.join(keys, on=list(DAY_KEY_COLS), how="semi")

    if spec.control_population == "non_event_pool":
        control_population = eligible_pool.join(
            keys, on=list(DAY_KEY_COLS), how="anti"
        )
    else:
        control_population = eligible_pool
    control = _sample_control(
        control_population,
        sample_size=spec.control_sample_size,
        seed=seed if seed is not None else int(date),
    )
    # Assert that neither branch escaped the already-validated population.
    for name, cohort in (("condition", condition), ("control", control)):
        cohort_keys = _validate_keys(cohort, f"{date} {name}")
        escaped = cohort_keys.join(pool_keys, on=list(DAY_KEY_COLS), how="anti")
        if not escaped.is_empty():
            raise ConditionStudyContractError(f"{date}: {name} escaped pool")

    return ConditionDayResult(
        date=date,
        eligible_pool_rows=len(eligible_pool),
        indexed_condition_rows=len(keys),
        condition_rows=len(condition),
        condition_excluded_rows=len(keys) - len(condition),
        control_rows=len(control),
        condition=condition,
        control=control,
    )


def _sample_control(
    population: pl.DataFrame,
    *,
    sample_size: int | None,
    seed: int,
) -> pl.DataFrame:
    if sample_size is None:
        return population
    sample_n = min(sample_size, len(population))
    return (
        population.sample(n=sample_n, seed=seed, shuffle=True)
        if sample_n
        else population
    )


def compare_conditions_to_shared_control(
    date: str,
    eligible_pool: pl.DataFrame,
    event_keys_by_condition: Mapping[str, pl.DataFrame],
    *,
    spec: ConditionComparisonSpec,
    seed: int | None = None,
) -> dict[str, ConditionDayResult]:
    """Compare many key-defined conditions with one identical control cohort.

    Each condition is validated independently, but candidate markers are
    joined to the eligible labeled pool together.  A row may belong to several
    conditions without being duplicated inside any one condition.  For
    ``non_event_pool`` control, the complement excludes the union of all
    supplied condition keys.
    """

    if not event_keys_by_condition:
        raise ValueError("event_keys_by_condition cannot be empty")
    if "_condition_id" in eligible_pool.columns:
        raise ConditionStudyContractError(
            "eligible pool uses reserved column _condition_id"
        )

    pool_keys = _validate_keys(eligible_pool, f"{date} eligible labeled pool")
    normalized: dict[str, pl.DataFrame] = {}
    markers: list[pl.DataFrame] = []
    for condition_id, event_keys in event_keys_by_condition.items():
        if not condition_id or not condition_id.strip():
            raise ValueError("condition ids must be non-empty")
        if "Date" in event_keys.columns:
            wrong_date = event_keys.filter(
                pl.col("Date").cast(pl.String) != date
            )
            if not wrong_date.is_empty():
                raise ConditionStudyContractError(
                    f"{date}: {condition_id} keys contain another date"
                )
        keys = _validate_keys(
            event_keys,
            f"{date} condition {condition_id}",
        )
        normalized[condition_id] = keys
        markers.append(
            keys.with_columns(pl.lit(condition_id).alias("_condition_id"))
        )

    marker_frame = pl.concat(markers, how="vertical")
    marked_conditions = eligible_pool.join(
        marker_frame,
        on=list(DAY_KEY_COLS),
        how="inner",
    )
    union_keys = marker_frame.select(DAY_KEY_COLS).unique()
    if spec.control_population == "non_event_pool":
        control_population = eligible_pool.join(
            union_keys,
            on=list(DAY_KEY_COLS),
            how="anti",
        )
    else:
        control_population = eligible_pool
    control = _sample_control(
        control_population,
        sample_size=spec.control_sample_size,
        seed=seed if seed is not None else int(date),
    )

    control_keys = _validate_keys(control, f"{date} shared control")
    escaped_control = control_keys.join(
        pool_keys,
        on=list(DAY_KEY_COLS),
        how="anti",
    )
    if not escaped_control.is_empty():
        raise ConditionStudyContractError(f"{date}: shared control escaped pool")

    results: dict[str, ConditionDayResult] = {}
    for condition_id, keys in normalized.items():
        condition = marked_conditions.filter(
            pl.col("_condition_id") == condition_id
        ).drop("_condition_id")
        condition_keys = _validate_keys(
            condition,
            f"{date} condition {condition_id} result",
        )
        escaped = condition_keys.join(
            pool_keys,
            on=list(DAY_KEY_COLS),
            how="anti",
        )
        if not escaped.is_empty():
            raise ConditionStudyContractError(
                f"{date}: condition {condition_id} escaped pool"
            )
        results[condition_id] = ConditionDayResult(
            date=date,
            eligible_pool_rows=len(eligible_pool),
            indexed_condition_rows=len(keys),
            condition_rows=len(condition),
            condition_excluded_rows=len(keys) - len(condition),
            control_rows=len(control),
            condition=condition,
            control=control,
        )
    return results


def summarize_cohort(
    frame: pl.DataFrame,
    *,
    date: str,
    cohort: str,
    eligible_pool_rows: int,
    bucket_columns: tuple[str, ...] = (),
) -> dict[str, object]:
    """Return additive statistics suitable for pooled and daily reporting."""

    required = {
        DEFAULT_LABEL_COLUMNS.raw_bp,
        DEFAULT_LABEL_COLUMNS.market_component_bp,
        DEFAULT_LABEL_COLUMNS.hedge_pnl_bp,
        DEFAULT_LABEL_COLUMNS.residual_bp,
        DEFAULT_LABEL_COLUMNS.z,
        *bucket_columns,
    }
    _require_columns(frame, required, f"{date} {cohort} summary")
    valid = frame.filter(pl.col(DEFAULT_LABEL_COLUMNS.z).is_finite())
    grid: dict[str, int] = {}
    if bucket_columns and not frame.is_empty():
        grouped = frame.group_by(list(bucket_columns)).len()
        for row in grouped.iter_rows(named=True):
            key = "|".join(str(row[column]) for column in bucket_columns)
            grid[key] = int(row["len"])

    def total(column: str) -> float:
        value = valid[column].sum()
        return float(value) if value is not None else 0.0

    z = valid[DEFAULT_LABEL_COLUMNS.z]
    return {
        "Date": date,
        "cohort": cohort,
        "n_eligible_pool": eligible_pool_rows,
        "n": len(frame),
        "n_z": len(valid),
        "sum_raw_bp": total(DEFAULT_LABEL_COLUMNS.raw_bp),
        "sum_market_component_bp": total(
            DEFAULT_LABEL_COLUMNS.market_component_bp
        ),
        "sum_hedge_pnl_bp": total(DEFAULT_LABEL_COLUMNS.hedge_pnl_bp),
        "sum_residual_bp": total(DEFAULT_LABEL_COLUMNS.residual_bp),
        "sum_z": float(z.sum()) if len(z) else 0.0,
        "sum_z2": float((z**2).sum()) if len(z) else 0.0,
        "grid": grid,
    }


def summarize_condition_day(
    result: ConditionDayResult,
    *,
    bucket_columns: tuple[str, ...] = (),
) -> tuple[dict[str, object], dict[str, object]]:
    return (
        summarize_cohort(
            result.condition,
            date=result.date,
            cohort="condition",
            eligible_pool_rows=result.eligible_pool_rows,
            bucket_columns=bucket_columns,
        ),
        summarize_cohort(
            result.control,
            date=result.date,
            cohort="control",
            eligible_pool_rows=result.eligible_pool_rows,
            bucket_columns=bucket_columns,
        ),
    )


def read_day_event_keys(
    date: str,
    *,
    event_index_path: Path | str | None = None,
    events_dir: Path | str | None = None,
) -> pl.DataFrame:
    """Read condition keys from a canonical index or a legacy per-day dataset."""

    if (event_index_path is None) == (events_dir is None):
        raise ValueError("provide exactly one of event_index_path or events_dir")
    if event_index_path is not None:
        path = Path(event_index_path)
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pl.read_parquet(path, columns=["Date", *DAY_KEY_COLS]).filter(
            pl.col("Date").cast(pl.String) == date
        )
    else:
        path = Path(events_dir) / f"{date}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        available = pl.read_parquet_schema(path)
        columns = [*DAY_KEY_COLS]
        if "Date" in available:
            columns.insert(0, "Date")
        frame = pl.read_parquet(path, columns=columns)
    return frame.select(DAY_KEY_COLS)


__all__ = [
    "ConditionComparisonSpec",
    "ConditionDayResult",
    "ConditionStudyContractError",
    "EligiblePoolSpec",
    "build_eligible_labeled_pool",
    "compare_condition_to_control",
    "compare_conditions_to_shared_control",
    "read_day_event_keys",
    "summarize_cohort",
    "summarize_condition_day",
]
