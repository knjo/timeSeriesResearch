"""Outcome-compatible robustness statistics for ``fh_neg_rejection``.

This module deliberately has no dependency on the raw-data runner.  It accepts
either a complete date-by-eight-cell aggregate table or a compact, label-valid
fresh-event cache plus one shared-control row per date.  The eight cells are the
factorial ablations of the depth, far-from-high, and bearish-state gates.  A
disabled gate means that the gate is not required; the cells are nested
candidate definitions, not mutually exclusive market regimes.

The estimand is the equal-active-day mean of::

    mean(label_z | condition cell) - mean(label_z | shared eligible pool)

Zero-event condition periods remain on the requested calendar with support
zero and an undefined effect.  They are never converted to zero effect.  The
selection-bias audit is a moving-block, single-step max-t procedure appropriate
for a family of daily ``delta_z`` series; this module does not compute a
Deflated Sharpe Ratio.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl


GATE_COLUMNS = ("use_depth", "use_far", "use_bear")
EVENT_GATE_COLUMNS = ("passes_depth", "passes_far", "passes_bear")
FULL_CELL_ID = "d1_g1_b1"
FROZEN_FAMILY_ID = "fh_neg_rejection_dgb_factorial"
FROZEN_FAMILY_VERSION = "1"
FROZEN_BOOTSTRAP_SEED = 20_260_827
FROZEN_FAMILY_ALPHA = 0.10
FROZEN_PRIMARY_BLOCK_LENGTH = 10
FROZEN_BLOCK_SENSITIVITY = (5, 20)
FROZEN_TEMPORAL_BLOCK_COUNT = 3
MIN_JOINT_VALID_BOOTSTRAP_REPETITIONS = 100
MIN_JOINT_VALID_BOOTSTRAP_RATE = 0.90
TRIAL_METADATA_ALLOWLIST = {
    "depth_ratio",
    "far_ticks",
    "bear_score",
    "description",
}
EVIDENCE_CLASSES = {
    "retrospective_diagnostic",
    "artifact_oos_non_pristine",
    "freeze_day_excluded",
    "prospective_first_live",
}
# Frozen order from study_contract.json, rather than a result-dependent order.
CELL_SPECS = (
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
)
LOO_SPECS = (
    ("D_increment", "depth", "d0_g1_b1"),
    ("G_increment", "far", "d1_g0_b1"),
    ("B_increment", "bear", "d1_g1_b0"),
)


class RobustnessContractError(ValueError):
    """Raised when an input cannot represent the frozen robustness contract."""


@dataclass(frozen=True)
class RobustnessConfig:
    """Frozen statistical and artifact choices for one evaluation."""

    family_id: str = FROZEN_FAMILY_ID
    family_version: str = FROZEN_FAMILY_VERSION
    objective: str = "shared_pool_equal_active_session_delta_z"
    selected_cell_id: str = FULL_CELL_ID
    bootstrap_seed: int = FROZEN_BOOTSTRAP_SEED
    bootstrap_reps: int = 5_000
    primary_block_length: int = FROZEN_PRIMARY_BLOCK_LENGTH
    block_length_sensitivity: tuple[int, ...] = FROZEN_BLOCK_SENSITIVITY
    family_alpha: float = FROZEN_FAMILY_ALPHA
    confidence_level: float = 0.90
    temporal_block_count: int = FROZEN_TEMPORAL_BLOCK_COUNT

    def __post_init__(self) -> None:
        if self.family_id != FROZEN_FAMILY_ID:
            raise ValueError("family_id is frozen by study_contract.json")
        if self.family_version != FROZEN_FAMILY_VERSION:
            raise ValueError("family_version is frozen by study_contract.json")
        if self.objective != "shared_pool_equal_active_session_delta_z":
            raise ValueError(
                "this evaluator only supports the frozen shared-pool estimand"
            )
        if self.selected_cell_id != FULL_CELL_ID:
            raise ValueError("selected_cell_id is frozen by study_contract.json")
        if self.bootstrap_reps < 100:
            raise ValueError("bootstrap_reps must be at least 100")
        if self.bootstrap_seed != FROZEN_BOOTSTRAP_SEED:
            raise ValueError("bootstrap_seed is frozen by study_contract.json")
        if self.primary_block_length != FROZEN_PRIMARY_BLOCK_LENGTH:
            raise ValueError("primary_block_length is frozen by study_contract.json")
        if self.block_length_sensitivity != FROZEN_BLOCK_SENSITIVITY:
            raise ValueError(
                "block_length_sensitivity is frozen by study_contract.json"
            )
        lengths = self.block_lengths
        if any(length < 1 for length in lengths):
            raise ValueError("block lengths must be positive")
        if len(lengths) != len(set(lengths)):
            raise ValueError("block lengths cannot contain duplicates")
        if self.family_alpha != FROZEN_FAMILY_ALPHA:
            raise ValueError("family_alpha is frozen by study_contract.json")
        if self.confidence_level != 0.90:
            raise ValueError("confidence_level is frozen by study_contract.json")
        if self.temporal_block_count != FROZEN_TEMPORAL_BLOCK_COUNT:
            raise ValueError("temporal_block_count is frozen by study_contract.json")

    @property
    def block_lengths(self) -> tuple[int, ...]:
        return (self.primary_block_length, *self.block_length_sensitivity)


@dataclass(frozen=True)
class RobustnessResult:
    """All machine-readable outputs produced from one common calendar."""

    requested_calendar: pl.DataFrame
    daily_effects: pl.DataFrame
    cell_summary: pl.DataFrame
    loo_daily: pl.DataFrame
    loo_summary: pl.DataFrame
    factorial_contrast_daily: pl.DataFrame
    factorial_contrasts: pl.DataFrame
    bootstrap: pl.DataFrame
    fwer_audit: pl.DataFrame
    temporal_blocks: pl.DataFrame
    trial_ledger: pl.DataFrame
    manifest: dict[str, Any]


def canonical_cell_id(depth: int | bool, far: int | bool, bear: int | bool) -> str:
    """Return the stable ID for one gate-inclusion vector."""

    return f"d{int(bool(depth))}_g{int(bool(far))}_b{int(bool(bear))}"


def canonical_cell_ids() -> tuple[str, ...]:
    return tuple(canonical_cell_id(*bits) for bits in CELL_SPECS)


def _cell_spec_by_id() -> dict[str, tuple[int, int, int]]:
    return {canonical_cell_id(*bits): bits for bits in CELL_SPECS}


def _validate_evidence_date(date: str, evidence_class: str) -> None:
    try:
        parsed = datetime.strptime(date, "%Y%m%d")
    except ValueError as exc:
        raise RobustnessContractError(
            f"requested calendar: invalid YYYYMMDD Date {date!r}"
        ) from exc
    if parsed.strftime("%Y%m%d") != date:
        raise RobustnessContractError(
            f"requested calendar: invalid YYYYMMDD Date {date!r}"
        )
    valid = {
        "retrospective_diagnostic": date <= "20260825",
        "artifact_oos_non_pristine": date == "20260826",
        "freeze_day_excluded": date == "20260827",
        "prospective_first_live": date > "20260827",
    }[evidence_class]
    if not valid:
        raise RobustnessContractError(
            f"requested calendar: Date {date} contradicts {evidence_class}"
        )


def prepare_requested_calendar(frame: pl.DataFrame) -> pl.DataFrame:
    """Validate the external, outcome-independent requested-session calendar."""

    required = ("Date", "source_complete", "run_status", "evidence_class")
    _require_columns(frame, required, "requested calendar")
    calendar = frame.select(required).with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("source_complete").cast(pl.Boolean, strict=False),
        pl.col("run_status")
        .cast(pl.String)
        .str.strip_chars()
        .str.to_lowercase(),
        pl.col("evidence_class").cast(pl.String).str.strip_chars(),
    )
    if not len(calendar):
        raise RobustnessContractError("requested calendar is empty")
    if _has_null(calendar, required):
        raise RobustnessContractError("requested calendar contains null fields")
    if calendar["Date"].n_unique() != len(calendar):
        raise RobustnessContractError("requested calendar contains duplicate Date")
    if calendar.filter(pl.col("run_status") == "").height:
        raise RobustnessContractError("requested calendar has empty run_status")
    evidence_classes = set(calendar["evidence_class"].unique().to_list())
    unknown = sorted(evidence_classes - EVIDENCE_CLASSES)
    if unknown:
        raise RobustnessContractError(
            f"requested calendar has unknown evidence_class values {unknown}"
        )
    if len(evidence_classes) != 1:
        raise RobustnessContractError(
            "one evaluation may contain exactly one evidence_class"
        )
    evidence_class = next(iter(evidence_classes))
    for date in calendar["Date"].to_list():
        _validate_evidence_date(str(date), evidence_class)
    return calendar.with_columns(
        pl.when(~pl.col("source_complete"))
        .then(pl.lit("missing_source"))
        .when(pl.col("run_status") != "success")
        .then(pl.lit("run_failed"))
        .otherwise(pl.lit("evaluation_expected"))
        .alias("calendar_status")
    ).sort("Date")


def _require_columns(frame: pl.DataFrame, columns: Sequence[str], source: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RobustnessContractError(f"{source}: missing columns {missing}")


def _has_null(frame: pl.DataFrame, columns: Sequence[str]) -> bool:
    return bool(
        frame.select(
            pl.any_horizontal([pl.col(column).is_null() for column in columns]).any()
        ).item()
    )


def _finite_values(values: Sequence[object]) -> np.ndarray:
    output = np.empty(len(values), dtype=float)
    for index, value in enumerate(values):
        output[index] = (
            float(value)
            if value is not None and math.isfinite(float(value))
            else math.nan
        )
    return output


def _validate_support_sum(
    frame: pl.DataFrame,
    *,
    count_col: str,
    sum_col: str,
    source: str,
) -> None:
    if frame.filter(pl.col(count_col) < 0).height:
        raise RobustnessContractError(f"{source}: {count_col} cannot be negative")
    bad_positive = frame.filter(
        (pl.col(count_col) > 0)
        & (pl.col(sum_col).is_null() | ~pl.col(sum_col).is_finite())
    )
    if bad_positive.height:
        raise RobustnessContractError(
            f"{source}: positive {count_col} requires a finite {sum_col}"
        )
    bad_zero = frame.filter(
        (pl.col(count_col) == 0)
        & pl.col(sum_col).is_not_null()
        & (~pl.col(sum_col).is_finite() | (pl.col(sum_col).abs() > 1e-12))
    )
    if bad_zero.height:
        raise RobustnessContractError(
            f"{source}: zero {count_col} cannot have a non-zero {sum_col}"
        )


def prepare_daily_aggregates(frame: pl.DataFrame) -> pl.DataFrame:
    """Validate and normalize a complete date-by-eight-cell aggregate table.

    Required input columns are ``Date``, ``condition_n_z``,
    ``condition_sum_z``, ``control_n_z``, and ``control_sum_z`` plus either the
    frozen ``candidate_id`` or all three ``use_*`` gate flags.  Supplying both
    is allowed only when they agree.  ``eligible_pool_rows`` is optional.
    Every requested date must have exactly one row for every cell.  A missing
    row is ambiguous with a failed run and is therefore rejected rather than
    interpreted as a zero-event period.
    """

    required = (
        "Date",
        "condition_n_z",
        "condition_sum_z",
        "control_n_z",
        "control_sum_z",
    )
    _require_columns(frame, required, "daily aggregates")
    gate_columns_present = [column in frame.columns for column in GATE_COLUMNS]
    has_gates = all(gate_columns_present)
    if any(gate_columns_present) and not has_gates:
        raise RobustnessContractError(
            "daily aggregates: provide all three use_* flags or none"
        )
    has_candidate_id = "candidate_id" in frame.columns
    if not has_gates and not has_candidate_id:
        raise RobustnessContractError(
            "daily aggregates: candidate_id or all three use_* flags required"
        )
    selected = list(required)
    if has_gates:
        selected.extend(GATE_COLUMNS)
    if has_candidate_id:
        selected.append("candidate_id")
    if "eligible_pool_rows" in frame.columns:
        selected.append("eligible_pool_rows")
    normalized = frame.select(selected).with_columns(
        pl.col("Date").cast(pl.String),
        *(
            [pl.col(column).cast(pl.Int8, strict=False) for column in GATE_COLUMNS]
            if has_gates
            else []
        ),
        *(
            [pl.col("candidate_id").cast(pl.String)]
            if has_candidate_id
            else []
        ),
        pl.col("condition_n_z").cast(pl.Int64, strict=False),
        pl.col("condition_sum_z").cast(pl.Float64, strict=False),
        pl.col("control_n_z").cast(pl.Int64, strict=False),
        pl.col("control_sum_z").cast(pl.Float64, strict=False),
        *(
            [pl.col("eligible_pool_rows").cast(pl.Int64, strict=False)]
            if "eligible_pool_rows" in selected
            else []
        ),
    )
    nonnull = (
        "Date",
        "condition_n_z",
        "control_n_z",
    )
    if _has_null(normalized, nonnull):
        raise RobustnessContractError("daily aggregates: null key or support")
    spec_by_id = _cell_spec_by_id()
    if has_candidate_id:
        if _has_null(normalized, ("candidate_id",)):
            raise RobustnessContractError("daily aggregates: null candidate_id")
        unknown_ids = sorted(
            set(normalized["candidate_id"].unique().to_list()) - set(spec_by_id)
        )
        if unknown_ids:
            raise RobustnessContractError(
                f"daily aggregates: unknown candidate_id values {unknown_ids}"
            )
    if has_gates:
        for column in GATE_COLUMNS:
            values = set(normalized[column].unique().to_list())
            if not values <= {0, 1}:
                raise RobustnessContractError(
                    f"{column} must contain only zero or one"
                )
        normalized = normalized.with_columns(
            pl.struct(GATE_COLUMNS)
            .map_elements(
                lambda row: canonical_cell_id(
                    row["use_depth"], row["use_far"], row["use_bear"]
                ),
                return_dtype=pl.String,
            )
            .alias("cell_id")
        )
        if has_candidate_id and normalized.filter(
            pl.col("candidate_id") != pl.col("cell_id")
        ).height:
            raise RobustnessContractError(
                "daily aggregates: candidate_id disagrees with use_* flags"
            )
    else:
        normalized = normalized.with_columns(
            pl.col("candidate_id").alias("cell_id"),
            *(
                pl.col("candidate_id")
                .map_elements(
                    lambda value, index=index: spec_by_id[value][index],
                    return_dtype=pl.Int8,
                )
                .alias(column)
                for index, column in enumerate(GATE_COLUMNS)
            ),
        )
    normalized = normalized.with_columns(pl.col("cell_id").alias("candidate_id"))
    duplicates = normalized.group_by("Date", "cell_id").len().filter(
        pl.col("len") != 1
    )
    if duplicates.height:
        raise RobustnessContractError("daily aggregates: duplicate date/cell rows")
    expected_cells = set(canonical_cell_ids())
    for row in normalized.group_by("Date").agg(
        pl.col("cell_id").sort().alias("cells"), pl.len().alias("rows")
    ).iter_rows(named=True):
        if int(row["rows"]) != len(expected_cells) or set(row["cells"]) != expected_cells:
            raise RobustnessContractError(
                f"{row['Date']}: incomplete eight-cell calendar"
            )

    _validate_support_sum(
        normalized,
        count_col="condition_n_z",
        sum_col="condition_sum_z",
        source="daily aggregates condition",
    )
    _validate_support_sum(
        normalized,
        count_col="control_n_z",
        sum_col="control_sum_z",
        source="daily aggregates control",
    )
    if normalized.filter(
        pl.col("condition_n_z") > pl.col("control_n_z")
    ).height:
        raise RobustnessContractError(
            "condition support cannot exceed shared label-valid control support"
        )
    specs = _cell_spec_by_id()
    for date, date_frame in normalized.partition_by(
        "Date", as_dict=True, maintain_order=True
    ).items():
        support_by_id = dict(
            date_frame.select("cell_id", "condition_n_z").iter_rows()
        )
        for strict_id, strict_bits in specs.items():
            for loose_id, loose_bits in specs.items():
                is_nested = all(
                    strict >= loose
                    for strict, loose in zip(strict_bits, loose_bits, strict=True)
                )
                if is_nested and support_by_id[strict_id] > support_by_id[loose_id]:
                    raise RobustnessContractError(
                        f"{date}: nested support violation {strict_id} > {loose_id}"
                    )

    control_check_columns = ["control_n_z", "control_sum_z"]
    if "eligible_pool_rows" in normalized.columns:
        if _has_null(normalized, ("eligible_pool_rows",)):
            raise RobustnessContractError("eligible_pool_rows cannot be null")
        if normalized.filter(pl.col("eligible_pool_rows") < 0).height:
            raise RobustnessContractError("eligible_pool_rows cannot be negative")
        if normalized.filter(
            pl.col("condition_n_z") > pl.col("eligible_pool_rows")
        ).height:
            raise RobustnessContractError(
                "condition support cannot exceed eligible_pool_rows"
            )
        control_check_columns.append("eligible_pool_rows")
    disagreement = normalized.group_by("Date").agg(
        *(
            pl.col(column).n_unique().alias(column)
            for column in control_check_columns
        )
    ).filter(
        pl.any_horizontal(
            [pl.col(column) != 1 for column in control_check_columns]
        )
    )
    if disagreement.height:
        raise RobustnessContractError(
            "daily aggregates: shared control differs across cells within a date"
        )

    condition_active = pl.col("condition_n_z") > 0
    control_active = pl.col("control_n_z") > 0
    output = normalized.with_columns(
        pl.when(condition_active)
        .then(pl.col("condition_sum_z") / pl.col("condition_n_z"))
        .alias("condition_mean_z"),
        pl.when(control_active)
        .then(pl.col("control_sum_z") / pl.col("control_n_z"))
        .alias("control_mean_z"),
        *(
            [
                pl.when(pl.col("eligible_pool_rows") > 0)
                .then(pl.col("condition_n_z") / pl.col("eligible_pool_rows"))
                .alias("selection_rate")
            ]
            if "eligible_pool_rows" in normalized.columns
            else [pl.lit(None, dtype=pl.Float64).alias("selection_rate")]
        ),
    ).with_columns(
        pl.when(condition_active & control_active)
        .then(pl.col("condition_mean_z") - pl.col("control_mean_z"))
        .alias("daily_delta_z"),
        condition_active.alias("condition_active"),
        (condition_active & control_active).alias("effect_active"),
    )
    return output.sort("Date", *GATE_COLUMNS)


def align_daily_to_requested_calendar(
    daily: pl.DataFrame,
    requested_calendar: pl.DataFrame,
) -> pl.DataFrame:
    """Align successful aggregates to every externally requested session.

    Missing-source and failed-run dates are represented by eight rows with
    null support rather than being converted to zero-event observations.
    Successful dates must contain the complete eight-cell panel.
    """

    calendar = prepare_requested_calendar(requested_calendar)
    expected_dates = set(
        calendar.filter(pl.col("calendar_status") == "evaluation_expected")[
            "Date"
        ].to_list()
    )
    actual_dates = set(daily["Date"].unique().to_list())
    missing = sorted(expected_dates - actual_dates)
    unexpected = sorted(actual_dates - expected_dates)
    if missing:
        raise RobustnessContractError(
            f"successful requested dates missing daily eight-cell rows: {missing}"
        )
    if unexpected:
        raise RobustnessContractError(
            "daily aggregates contain unrequested, missing-source, or failed "
            f"dates: {unexpected}"
        )

    cell_rows = [
        {
            "candidate_id": canonical_cell_id(depth, far, bear),
            "cell_id": canonical_cell_id(depth, far, bear),
            "use_depth": depth,
            "use_far": far,
            "use_bear": bear,
        }
        for depth, far, bear in CELL_SPECS
    ]
    skeleton = calendar.join(pl.DataFrame(cell_rows), how="cross")
    identifiers = {"Date", "candidate_id", "cell_id", *GATE_COLUMNS}
    value_columns = [column for column in daily.columns if column not in identifiers]
    panel = skeleton.join(
        daily.select("Date", "cell_id", *value_columns),
        on=("Date", "cell_id"),
        how="left",
        validate="1:1",
    ).with_columns(
        pl.when(pl.col("calendar_status") == "missing_source")
        .then(pl.lit("missing_source"))
        .when(pl.col("calendar_status") == "run_failed")
        .then(pl.lit("run_failed"))
        .when(pl.col("control_n_z") == 0)
        .then(pl.lit("zero_control_ineligible"))
        .when(pl.col("condition_n_z") == 0)
        .then(pl.lit("zero_event"))
        .otherwise(pl.lit("active"))
        .alias("period_status")
    )
    successful = panel.filter(
        pl.col("calendar_status") == "evaluation_expected"
    )
    if successful.filter(pl.col("condition_n_z").is_null()).height:
        raise RobustnessContractError(
            "successful requested date has null condition support after alignment"
        )
    return panel.sort("Date", *GATE_COLUMNS)


def build_daily_aggregates_from_event_cache(
    events: pl.DataFrame,
    daily_control: pl.DataFrame,
) -> pl.DataFrame:
    """Build all nested cells from a label-valid fresh-event compact cache.

    ``events`` requires ``Date``, ``label_z``, and the three ``passes_*`` gate
    flags plus the canonical ``QuoteCode`` and ``ChannelSeq`` event key, whose
    uniqueness is checked.  ``daily_control`` requires one row per requested
    date with ``control_n_z`` and ``control_sum_z``; ``eligible_pool_rows`` is
    optional.  Event dates must be a subset of the control calendar.
    """

    _require_columns(
        events,
        ("Date", "QuoteCode", "ChannelSeq", "label_z", *EVENT_GATE_COLUMNS),
        "event cache",
    )
    _require_columns(
        daily_control,
        ("Date", "control_n_z", "control_sum_z"),
        "daily control",
    )
    control_columns = ["Date", "control_n_z", "control_sum_z"]
    if "eligible_pool_rows" in daily_control.columns:
        control_columns.append("eligible_pool_rows")
    control = daily_control.select(control_columns).with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("control_n_z").cast(pl.Int64, strict=False),
        pl.col("control_sum_z").cast(pl.Float64, strict=False),
        *(
            [pl.col("eligible_pool_rows").cast(pl.Int64, strict=False)]
            if "eligible_pool_rows" in control_columns
            else []
        ),
    )
    if _has_null(control, ("Date", "control_n_z")):
        raise RobustnessContractError("daily control: null date or support")
    if "eligible_pool_rows" in control.columns and _has_null(
        control, ("eligible_pool_rows",)
    ):
        raise RobustnessContractError("daily control: null eligible_pool_rows")
    if not len(control):
        raise RobustnessContractError("daily control: requested calendar is empty")
    if control["Date"].n_unique() != len(control):
        raise RobustnessContractError("daily control: duplicate Date")
    _validate_support_sum(
        control,
        count_col="control_n_z",
        sum_col="control_sum_z",
        source="daily control",
    )

    event_columns = [
        "Date",
        "QuoteCode",
        "ChannelSeq",
        "label_z",
        *EVENT_GATE_COLUMNS,
    ]
    compact = events.select(event_columns).with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("label_z").cast(pl.Float64, strict=False),
        *(pl.col(column).cast(pl.Boolean, strict=False) for column in EVENT_GATE_COLUMNS),
    )
    if _has_null(compact, ("Date", "label_z", *EVENT_GATE_COLUMNS)):
        raise RobustnessContractError("event cache: null date, label, or gate")
    if compact.filter(~pl.col("label_z").is_finite()).height:
        raise RobustnessContractError("event cache: label_z must be finite")
    if _has_null(compact, ("QuoteCode", "ChannelSeq")):
        raise RobustnessContractError("event cache: null event key")
    if compact.select(
        pl.struct(["Date", "QuoteCode", "ChannelSeq"]).is_duplicated().any()
    ).item():
        raise RobustnessContractError("event cache: duplicate event key")
    unknown_dates = compact.select("Date").unique().join(
        control.select("Date"), on="Date", how="anti"
    )
    if unknown_dates.height:
        raise RobustnessContractError(
            "event cache contains dates outside the control calendar"
        )

    rows: list[dict[str, Any]] = []
    for control_row in control.sort("Date").iter_rows(named=True):
        date = str(control_row["Date"])
        day = compact.filter(pl.col("Date") == date)
        for depth, far, bear in CELL_SPECS:
            mask = pl.lit(True)
            for enabled, column in zip(
                (depth, far, bear), EVENT_GATE_COLUMNS, strict=True
            ):
                if enabled:
                    mask &= pl.col(column)
            condition = day.filter(mask)
            row = {
                "Date": date,
                "use_depth": depth,
                "use_far": far,
                "use_bear": bear,
                "condition_n_z": len(condition),
                "condition_sum_z": (
                    float(condition["label_z"].sum()) if len(condition) else 0.0
                ),
                "control_n_z": int(control_row["control_n_z"]),
                "control_sum_z": control_row["control_sum_z"],
            }
            if "eligible_pool_rows" in control_row:
                row["eligible_pool_rows"] = int(control_row["eligible_pool_rows"])
            rows.append(row)
    return prepare_daily_aggregates(pl.DataFrame(rows))


def _cell_summary(daily: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for depth, far, bear in CELL_SPECS:
        cell_id = canonical_cell_id(depth, far, bear)
        subset = daily.filter(pl.col("cell_id") == cell_id).sort("Date")
        active = subset.filter(pl.col("effect_active"))
        selection_evaluable = subset.filter(
            pl.col("period_status").is_in(["active", "zero_event"])
        )
        values = _finite_values(active["daily_delta_z"].to_list())
        values = values[np.isfinite(values)]
        support_values = active["condition_n_z"].to_numpy()
        rows.append(
            {
                "cell_id": cell_id,
                "use_depth": depth,
                "use_far": far,
                "use_bear": bear,
                "calendar_periods": len(subset),
                "condition_active_periods": int(
                    subset["condition_active"].sum() or 0
                ),
                "effect_active_periods": len(values),
                "zero_event_periods": int(
                    (subset["period_status"] == "zero_event").sum()
                ),
                "zero_control_ineligible_periods": int(
                    (subset["period_status"] == "zero_control_ineligible").sum()
                ),
                "missing_source_periods": int(
                    (subset["period_status"] == "missing_source").sum()
                ),
                "failed_run_periods": int(
                    (subset["period_status"] == "run_failed").sum()
                ),
                "missing_control_periods": int(
                    (subset["period_status"] == "zero_control_ineligible").sum()
                ),
                "condition_support": int(subset["condition_n_z"].sum() or 0),
                "mean_support_per_active_period": (
                    float(np.mean(support_values)) if len(support_values) else None
                ),
                "median_support_per_active_period": (
                    float(np.median(support_values)) if len(support_values) else None
                ),
                "equal_day_delta_z": (
                    float(np.mean(values)) if len(values) else None
                ),
                "daily_delta_z_std": (
                    float(np.std(values, ddof=1)) if len(values) > 1 else None
                ),
                "positive_day_ratio": (
                    float(np.mean(values > 0.0)) if len(values) else None
                ),
                "selection_rate_mean": (
                    float(
                        selection_evaluable["selection_rate"].drop_nulls().mean()
                    )
                    if len(selection_evaluable["selection_rate"].drop_nulls())
                    else None
                ),
                "selection_rate_periods": len(
                    selection_evaluable["selection_rate"].drop_nulls()
                ),
            }
        )
    return pl.DataFrame(rows).sort(*GATE_COLUMNS)


def _build_loo(daily: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    full = daily.filter(pl.col("cell_id") == FULL_CELL_ID).select(
        "Date",
        pl.col("period_status").alias("full_period_status"),
        pl.col("condition_n_z").alias("full_support"),
        pl.col("daily_delta_z").alias("full_daily_delta_z"),
    )
    daily_rows: list[pl.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for contrast_id, gate, reduced_cell_id in LOO_SPECS:
        reduced = daily.filter(pl.col("cell_id") == reduced_cell_id).select(
            "Date",
            pl.col("condition_n_z").alias("reduced_support"),
            pl.col("daily_delta_z").alias("reduced_daily_delta_z"),
        )
        paired = full.join(reduced, on="Date", how="inner", validate="1:1").with_columns(
            pl.lit(contrast_id).alias("contrast_id"),
            pl.lit(gate).alias("removed_gate"),
            pl.lit(FULL_CELL_ID).alias("full_cell_id"),
            pl.lit(reduced_cell_id).alias("reduced_cell_id"),
            (
                pl.col("full_daily_delta_z").is_finite()
                & pl.col("reduced_daily_delta_z").is_finite()
            ).alias("common_active"),
        ).with_columns(
            pl.when(pl.col("common_active"))
            .then(
                pl.col("full_daily_delta_z")
                - pl.col("reduced_daily_delta_z")
            )
            .alias("paired_increment_delta_z")
        ).select(
            "Date",
            "contrast_id",
            "removed_gate",
            "full_cell_id",
            "reduced_cell_id",
            "full_period_status",
            "full_support",
            "reduced_support",
            "common_active",
            "full_daily_delta_z",
            "reduced_daily_delta_z",
            "paired_increment_delta_z",
        )
        common = paired.filter(pl.col("common_active"))
        increments = _finite_values(common["paired_increment_delta_z"].to_list())
        increments = increments[np.isfinite(increments)]
        summary_rows.append(
            {
                "contrast_id": contrast_id,
                "removed_gate": gate,
                "full_cell_id": FULL_CELL_ID,
                "reduced_cell_id": reduced_cell_id,
                "calendar_periods": len(paired),
                "common_active_periods": len(increments),
                "full_support_on_common_periods": int(common["full_support"].sum()),
                "reduced_support_on_common_periods": int(
                    common["reduced_support"].sum()
                ),
                "full_equal_day_delta_z_common": (
                    float(common["full_daily_delta_z"].mean())
                    if len(common)
                    else None
                ),
                "reduced_equal_day_delta_z_common": (
                    float(common["reduced_daily_delta_z"].mean())
                    if len(common)
                    else None
                ),
                "paired_increment_delta_z": (
                    float(np.mean(increments)) if len(increments) else None
                ),
                "median_paired_increment_delta_z": (
                    float(np.median(increments)) if len(increments) else None
                ),
                "paired_increment_std": (
                    float(np.std(increments, ddof=1))
                    if len(increments) > 1
                    else None
                ),
                "positive_increment_ratio": (
                    float(np.mean(increments > 0.0)) if len(increments) else None
                ),
            }
        )
        daily_rows.append(paired)
    return (
        pl.concat(daily_rows, how="vertical").sort("Date", "contrast_id"),
        pl.DataFrame(summary_rows).sort("contrast_id"),
    )


def _factorial_contrast_specs() -> list[dict[str, Any]]:
    factor_names = ("D", "G", "B")
    specs: list[dict[str, Any]] = []
    for factor_index, factor_name in enumerate(factor_names):
        other_indices = [index for index in range(3) if index != factor_index]
        for first_state in (0, 1):
            for second_state in (0, 1):
                low = [0, 0, 0]
                low[other_indices[0]] = first_state
                low[other_indices[1]] = second_state
                high = low.copy()
                high[factor_index] = 1
                conditioned = (
                    f"{factor_names[other_indices[0]]}{first_state}_"
                    f"{factor_names[other_indices[1]]}{second_state}"
                )
                g4_id = (
                    f"{factor_name}_increment"
                    if first_state == 1 and second_state == 1
                    else None
                )
                specs.append(
                    {
                        "contrast_id": f"{factor_name}_at_{conditioned}",
                        "contrast_kind": "conditional_main_increment",
                        "decision_role": (
                            "frozen_G4_parsimony"
                            if g4_id is not None
                            else "mechanism_diagnostic"
                        ),
                        "g4_contrast_id": g4_id,
                        "coefficients": {
                            canonical_cell_id(*high): 1.0,
                            canonical_cell_id(*low): -1.0,
                        },
                    }
                )

    for left_index, right_index in ((0, 1), (0, 2), (1, 2)):
        left_name = factor_names[left_index]
        right_name = factor_names[right_index]
        remaining_index = next(
            index
            for index in range(3)
            if index not in (left_index, right_index)
        )
        remaining_name = factor_names[remaining_index]
        for remaining_state in (0, 1):
            coefficients: dict[str, float] = {}
            for left_state in (0, 1):
                for right_state in (0, 1):
                    bits = [0, 0, 0]
                    bits[left_index] = left_state
                    bits[right_index] = right_state
                    bits[remaining_index] = remaining_state
                    coefficient = 1.0 if left_state == right_state else -1.0
                    coefficients[canonical_cell_id(*bits)] = coefficient
            specs.append(
                {
                    "contrast_id": (
                        f"{left_name}{right_name}_at_"
                        f"{remaining_name}{remaining_state}"
                    ),
                    "contrast_kind": "conditional_two_way_interaction",
                    "decision_role": "mechanism_diagnostic",
                    "g4_contrast_id": None,
                    "coefficients": coefficients,
                }
            )

    specs.append(
        {
            "contrast_id": "DGB_three_way_interaction",
            "contrast_kind": "three_way_interaction",
            "decision_role": "mechanism_diagnostic",
            "g4_contrast_id": None,
            "coefficients": {
                canonical_cell_id(*bits): float((-1) ** (3 - sum(bits)))
                for bits in CELL_SPECS
            },
        }
    )
    return specs


def _build_factorial_contrasts(
    daily: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build the complete paired 2x2x2 contrast table without selecting it."""

    specs = _factorial_contrast_specs()
    dates = daily["Date"].unique().sort().to_list()
    values_by_date: dict[str, dict[str, float | None]] = {}
    for date in dates:
        values_by_date[str(date)] = {
            str(row["cell_id"]): row["daily_delta_z"]
            for row in daily.filter(pl.col("Date") == date)
            .select("cell_id", "daily_delta_z")
            .iter_rows(named=True)
        }
    daily_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for spec in specs:
        coefficients = spec["coefficients"]
        contrast_values: list[float] = []
        for date in dates:
            cell_values = values_by_date[str(date)]
            common_active = all(
                cell_values[cell_id] is not None
                and math.isfinite(float(cell_values[cell_id]))
                for cell_id in coefficients
            )
            value = (
                float(
                    sum(
                        coefficient * float(cell_values[cell_id])
                        for cell_id, coefficient in coefficients.items()
                    )
                )
                if common_active
                else None
            )
            if value is not None:
                contrast_values.append(value)
            daily_rows.append(
                {
                    "Date": str(date),
                    "contrast_id": spec["contrast_id"],
                    "contrast_kind": spec["contrast_kind"],
                    "decision_role": spec["decision_role"],
                    "g4_contrast_id": spec["g4_contrast_id"],
                    "common_active": common_active,
                    "paired_contrast_delta_z": value,
                }
            )
        values = np.asarray(contrast_values, dtype=float)
        summary_rows.append(
            {
                "contrast_id": spec["contrast_id"],
                "contrast_kind": spec["contrast_kind"],
                "decision_role": spec["decision_role"],
                "g4_contrast_id": spec["g4_contrast_id"],
                "coefficient_json": json.dumps(
                    coefficients,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "calendar_periods": len(dates),
                "common_active_periods": len(values),
                "paired_contrast_delta_z": (
                    float(values.mean()) if len(values) else None
                ),
                "median_paired_contrast_delta_z": (
                    float(np.median(values)) if len(values) else None
                ),
                "positive_contrast_ratio": (
                    float(np.mean(values > 0.0)) if len(values) else None
                ),
            }
        )
    return (
        pl.DataFrame(daily_rows, infer_schema_length=None).sort(
            "Date", "contrast_id"
        ),
        pl.DataFrame(summary_rows, infer_schema_length=None).sort(
            "contrast_kind", "contrast_id"
        ),
    )


def moving_block_indices(
    n_periods: int,
    *,
    block_length: int,
    repetitions: int,
    seed: int,
) -> np.ndarray:
    """Draw circular moving-block indices shared synchronously by all trials."""

    if n_periods < 1:
        raise ValueError("n_periods must be positive")
    if block_length < 1 or repetitions < 1:
        raise ValueError("block_length and repetitions must be positive")
    effective = min(block_length, n_periods)
    blocks_needed = math.ceil(n_periods / effective)
    seed_sequence = np.random.SeedSequence([int(seed), int(block_length)])
    rng = np.random.default_rng(seed_sequence)
    starts = rng.integers(
        0,
        n_periods,
        size=(repetitions, blocks_needed),
    )
    offsets = np.arange(effective, dtype=np.int64)
    sampled = ((starts[..., None] + offsets) % n_periods).reshape(
        repetitions, -1
    )
    return sampled[:, :n_periods]


def _sample_means(matrix: np.ndarray, indices: np.ndarray) -> np.ndarray:
    sampled = matrix[indices, :]
    finite = np.isfinite(sampled)
    counts = finite.sum(axis=1)
    sums = np.where(finite, sampled, 0.0).sum(axis=1)
    return np.divide(
        sums,
        counts,
        out=np.full(sums.shape, np.nan, dtype=float),
        where=counts > 0,
    )


def _point_means(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(matrix)
    counts = finite.sum(axis=0)
    sums = np.where(finite, matrix, 0.0).sum(axis=0)
    means = np.divide(
        sums,
        counts,
        out=np.full(sums.shape, np.nan, dtype=float),
        where=counts > 0,
    )
    return means, counts


def _bootstrap_rows(
    matrix: np.ndarray,
    target_ids: Sequence[str],
    *,
    target_kind: str,
    config: RobustnessConfig,
) -> list[dict[str, Any]]:
    point, counts = _point_means(matrix)
    centered = matrix - point
    alpha = 1.0 - config.confidence_level
    rows: list[dict[str, Any]] = []
    for block_length in config.block_lengths:
        indices = moving_block_indices(
            matrix.shape[0],
            block_length=block_length,
            repetitions=config.bootstrap_reps,
            seed=config.bootstrap_seed,
        )
        sampled = _sample_means(matrix, indices)
        sampled_centered = _sample_means(centered, indices)
        for column, target_id in enumerate(target_ids):
            values = sampled[:, column]
            null_values = sampled_centered[:, column]
            values = values[np.isfinite(values)]
            null_values = null_values[np.isfinite(null_values)]
            estimate = point[column]
            p_two_sided = None
            p_one_sided = None
            if math.isfinite(float(estimate)) and len(null_values):
                p_two_sided = float(
                    (1 + np.sum(np.abs(null_values) >= abs(float(estimate))))
                    / (len(null_values) + 1)
                )
                p_one_sided = float(
                    (1 + np.sum(null_values >= float(estimate)))
                    / (len(null_values) + 1)
                )
            rows.append(
                {
                    "target_kind": target_kind,
                    "target_id": target_id,
                    "block_length_requested": block_length,
                    "block_length_effective": min(block_length, matrix.shape[0]),
                    "primary_block_length": (
                        block_length == config.primary_block_length
                    ),
                    "calendar_periods": matrix.shape[0],
                    "active_periods": int(counts[column]),
                    "point_estimate": (
                        float(estimate) if math.isfinite(float(estimate)) else None
                    ),
                    "bootstrap_repetitions": config.bootstrap_reps,
                    "valid_bootstrap_repetitions": len(values),
                    "bootstrap_mean": float(np.mean(values)) if len(values) else None,
                    "bootstrap_se": (
                        float(np.std(values, ddof=1)) if len(values) > 1 else None
                    ),
                    "ci_low": (
                        float(np.quantile(values, alpha / 2.0))
                        if len(values)
                        else None
                    ),
                    "ci_high": (
                        float(np.quantile(values, 1.0 - alpha / 2.0))
                        if len(values)
                        else None
                    ),
                    "bootstrap_probability_gt_zero": (
                        float(np.mean(values > 0.0)) if len(values) else None
                    ),
                    "p_one_sided_positive_centered": p_one_sided,
                    "p_two_sided_centered": p_two_sided,
                    "seed": config.bootstrap_seed,
                    "method": "circular_moving_block_percentile_and_centered_test",
                }
            )
    return rows


def _effective_trial_count(matrix: np.ndarray) -> tuple[float, str, int]:
    """Return a transparent dependence diagnostic, not a test-family rewrite."""

    trial_count = matrix.shape[1]
    correlations = np.eye(trial_count, dtype=float)
    common = np.isfinite(matrix).all(axis=1)
    common_count = int(common.sum())
    if common_count >= 3:
        values = matrix[common, :]
        centered = values - values.mean(axis=0)
        norms = np.sqrt(np.square(centered).sum(axis=0))
        varying = norms > np.finfo(float).eps
        if varying.any():
            standardized = centered[:, varying] / norms[varying]
            varying_indices = np.flatnonzero(varying)
            correlations[np.ix_(varying_indices, varying_indices)] = (
                standardized.T @ standardized
            )
    eigenvalues = np.linalg.eigvalsh(correlations)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    denominator = float(np.square(eigenvalues).sum())
    effective = (
        float(np.square(eigenvalues.sum()) / denominator)
        if denominator > 0.0
        else float(trial_count)
    )
    effective = min(float(trial_count), max(1.0, effective))
    return (
        effective,
        "joint_finite_calendar_psd_corr_eigen_participation_ratio",
        common_count,
    )


def _maxstat_rows(
    matrix: np.ndarray,
    target_ids: Sequence[str],
    *,
    family_id: str,
    config: RobustnessConfig,
) -> list[dict[str, Any]]:
    point, counts = _point_means(matrix)
    centered = matrix - point
    effective_trials, effective_method, effective_common_periods = (
        _effective_trial_count(matrix)
    )
    dispersion = (
        float(np.nanstd(point, ddof=1))
        if np.isfinite(point).sum() > 1
        else None
    )
    rows: list[dict[str, Any]] = []
    for block_length in config.block_lengths:
        indices = moving_block_indices(
            matrix.shape[0],
            block_length=block_length,
            repetitions=config.bootstrap_reps,
            seed=config.bootstrap_seed,
        )
        null_means = _sample_means(centered, indices)
        joint_valid = np.isfinite(null_means).all(axis=1)
        joint_valid_count = int(joint_valid.sum())
        joint_valid_rate = joint_valid_count / config.bootstrap_reps
        if (
            joint_valid_count < MIN_JOINT_VALID_BOOTSTRAP_REPETITIONS
            or joint_valid_rate < MIN_JOINT_VALID_BOOTSTRAP_RATE
        ):
            raise RobustnessContractError(
                f"{family_id}: joint-valid max-stat bootstrap insufficient at "
                f"block length {block_length}: {joint_valid_count}/"
                f"{config.bootstrap_reps} ({joint_valid_rate:.3f})"
            )
        joint_null_means = null_means[joint_valid, :]
        scale = np.std(joint_null_means, axis=0, ddof=1)
        valid_trials = (
            np.isfinite(point)
            & np.isfinite(scale)
            & (scale > np.finfo(float).eps)
        )
        if not valid_trials.all():
            invalid_targets = [
                target_id
                for target_id, valid in zip(target_ids, valid_trials, strict=True)
                if not valid
            ]
            raise RobustnessContractError(
                f"{family_id}: non-estimable max-stat targets {invalid_targets} "
                f"at block length {block_length}"
            )
        joint_null_statistics = joint_null_means / scale
        observed_statistics = point / scale
        valid_max_statistics = np.max(joint_null_statistics, axis=1)
        critical_value = float(
            np.quantile(valid_max_statistics, 1.0 - config.family_alpha)
        )
        for column, target_id in enumerate(target_ids):
            raw_p = None
            adjusted_p = None
            statistic = observed_statistics[column]
            if math.isfinite(float(statistic)):
                own_null = joint_null_statistics[:, column]
                family_null = valid_max_statistics
                raw_p = float(
                    (1 + np.sum(own_null >= float(statistic)))
                    / (len(own_null) + 1)
                )
                adjusted_p = float(
                    (1 + np.sum(family_null >= float(statistic)))
                    / (len(family_null) + 1)
                )
            simultaneous_lower_bound = None
            if (
                critical_value is not None
                and math.isfinite(float(point[column]))
                and math.isfinite(float(scale[column]))
            ):
                simultaneous_lower_bound = float(
                    point[column] - critical_value * scale[column]
                )
            rows.append(
                {
                    "family_id": family_id,
                    "target_id": target_id,
                    "block_length_requested": block_length,
                    "block_length_effective": min(block_length, matrix.shape[0]),
                    "primary_block_length": (
                        block_length == config.primary_block_length
                    ),
                    "calendar_periods": matrix.shape[0],
                    "active_periods": int(counts[column]),
                    "point_estimate": (
                        float(point[column])
                        if math.isfinite(float(point[column]))
                        else None
                    ),
                    "bootstrap_se": (
                        float(scale[column])
                        if math.isfinite(float(scale[column]))
                        else None
                    ),
                    "max_t_statistic": (
                        float(statistic)
                        if math.isfinite(float(statistic))
                        else None
                    ),
                    "alternative": "positive",
                    "family_alpha": config.family_alpha,
                    "simultaneous_confidence_level": 1.0 - config.family_alpha,
                    "max_t_critical_value": critical_value,
                    "simultaneous_lower_bound": simultaneous_lower_bound,
                    "raw_block_bootstrap_p": raw_p,
                    "single_step_fwer_p": adjusted_p,
                    "raw_trial_count": len(target_ids),
                    "dependence_effective_trial_count": effective_trials,
                    "effective_trial_method": effective_method,
                    "effective_trial_common_finite_periods": (
                        effective_common_periods
                    ),
                    "across_trial_point_dispersion": dispersion,
                    "bootstrap_repetitions": config.bootstrap_reps,
                    "joint_valid_bootstrap_repetitions": joint_valid_count,
                    "joint_valid_bootstrap_rate": joint_valid_rate,
                    "minimum_joint_valid_repetitions": (
                        MIN_JOINT_VALID_BOOTSTRAP_REPETITIONS
                    ),
                    "minimum_joint_valid_rate": MIN_JOINT_VALID_BOOTSTRAP_RATE,
                    "seed": config.bootstrap_seed,
                    "method": "circular_moving_block_single_step_max_t_one_sided",
                    "correction_scope": "current_frozen_family_only",
                    "historical_selection_exposure_included": False,
                    "total_selection_exposure_is_lower_bound": True,
                    "current_family_correction_is_only_historical_exposure_lower_bound": True,
                }
            )
    return rows


def _matrix_from_complete_panel(
    panel: pl.DataFrame,
    *,
    dates: Sequence[str],
    id_col: str,
    value_col: str,
    target_ids: Sequence[str],
) -> np.ndarray:
    matrix = np.full((len(dates), len(target_ids)), np.nan, dtype=float)
    date_index = {date: index for index, date in enumerate(dates)}
    target_index = {target: index for index, target in enumerate(target_ids)}
    for row in panel.select("Date", id_col, value_col).iter_rows(named=True):
        value = row[value_col]
        if value is not None and math.isfinite(float(value)):
            matrix[date_index[str(row["Date"])], target_index[str(row[id_col])]] = float(
                value
            )
    return matrix


def _temporal_blocks(
    daily: pl.DataFrame,
    loo_daily: pl.DataFrame,
    *,
    config: RobustnessConfig,
) -> pl.DataFrame:
    dates = daily["Date"].unique().sort().to_list()
    block_count = min(config.temporal_block_count, len(dates))
    date_blocks = np.array_split(np.asarray(dates, dtype=object), block_count)
    rows: list[dict[str, Any]] = []
    for block_index, block_dates_array in enumerate(date_blocks, start=1):
        block_dates = [str(value) for value in block_dates_array.tolist()]
        block_id = f"T{block_index:02d}"
        for cell_id in canonical_cell_ids():
            subset = daily.filter(
                (pl.col("cell_id") == cell_id) & pl.col("Date").is_in(block_dates)
            )
            active = subset.filter(pl.col("effect_active"))
            rows.append(
                {
                    "target_kind": "cell_effect",
                    "target_id": cell_id,
                    "block_id": block_id,
                    "block_start": block_dates[0],
                    "block_end": block_dates[-1],
                    "calendar_periods": len(block_dates),
                    "active_periods": len(active),
                    "zero_event_periods": int(
                        (subset["period_status"] == "zero_event").sum()
                    ),
                    "zero_control_ineligible_periods": int(
                        (
                            subset["period_status"]
                            == "zero_control_ineligible"
                        ).sum()
                    ),
                    "missing_source_periods": int(
                        (subset["period_status"] == "missing_source").sum()
                    ),
                    "failed_run_periods": int(
                        (subset["period_status"] == "run_failed").sum()
                    ),
                    "support": int(subset["condition_n_z"].sum() or 0),
                    "equal_period_value": (
                        float(active["daily_delta_z"].mean())
                        if len(active)
                        else None
                    ),
                }
            )
        for contrast_id, gate, _ in LOO_SPECS:
            subset = loo_daily.filter(
                (pl.col("contrast_id") == contrast_id)
                & pl.col("Date").is_in(block_dates)
            )
            active = subset.filter(pl.col("common_active"))
            rows.append(
                {
                    "target_kind": "loo_increment",
                    "target_id": contrast_id,
                    "removed_gate": gate,
                    "block_id": block_id,
                    "block_start": block_dates[0],
                    "block_end": block_dates[-1],
                    "calendar_periods": len(block_dates),
                    "active_periods": len(active),
                    "zero_event_periods": None,
                    "zero_control_ineligible_periods": None,
                    "missing_source_periods": None,
                    "failed_run_periods": None,
                    "support": None,
                    "equal_period_value": (
                        float(active["paired_increment_delta_z"].mean())
                        if len(active)
                        else None
                    ),
                }
            )
    return pl.DataFrame(rows).sort("block_id", "target_kind", "target_id")


def _trial_ledger(
    summary: pl.DataFrame,
    bootstrap: pl.DataFrame,
    fwer: pl.DataFrame,
    *,
    config: RobustnessConfig,
    evaluation_start: str,
    evaluation_end: str,
    trial_metadata: Mapping[str, Mapping[str, Any]] | None,
) -> pl.DataFrame:
    metadata = dict(trial_metadata or {})
    unknown = sorted(set(metadata) - set(canonical_cell_ids()))
    if unknown:
        raise RobustnessContractError(f"trial_metadata has unknown cells: {unknown}")
    for cell_id, values in metadata.items():
        unknown_fields = sorted(set(values) - TRIAL_METADATA_ALLOWLIST)
        if unknown_fields:
            raise RobustnessContractError(
                f"trial_metadata for {cell_id} has non-parameter fields: "
                f"{unknown_fields}"
            )
    primary_bootstrap = bootstrap.filter(
        (pl.col("target_kind") == "cell_effect")
        & pl.col("primary_block_length")
    )
    primary_fwer = fwer.filter(
        (pl.col("family_id") == config.family_id)
        & pl.col("primary_block_length")
    )
    rows: list[dict[str, Any]] = []
    for summary_row in summary.iter_rows(named=True):
        cell_id = str(summary_row["cell_id"])
        parameters: dict[str, Any] = {
            "use_depth": bool(summary_row["use_depth"]),
            "use_far": bool(summary_row["use_far"]),
            "use_bear": bool(summary_row["use_bear"]),
        }
        parameters.update(metadata.get(cell_id, {}))
        parameter_json = json.dumps(
            parameters,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        fingerprint_payload = {
            "family_id": config.family_id,
            "family_version": config.family_version,
            "objective": config.objective,
            "trial_id": cell_id,
            "parameters": parameters,
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_payload,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        boot_row = primary_bootstrap.filter(pl.col("target_id") == cell_id).row(
            0, named=True
        )
        fwer_row = primary_fwer.filter(pl.col("target_id") == cell_id).row(
            0, named=True
        )
        rows.append(
            {
                "ledger_contract": "fh_neg_rejection_robustness_trial_ledger/v1",
                "trial_id": cell_id,
                "fingerprint": fingerprint,
                "family_id": config.family_id,
                "family_version": config.family_version,
                "objective": config.objective,
                "evaluation_start": evaluation_start,
                "evaluation_end": evaluation_end,
                "selected": cell_id == config.selected_cell_id,
                "parameter_json": parameter_json,
                "use_depth": summary_row["use_depth"],
                "use_far": summary_row["use_far"],
                "use_bear": summary_row["use_bear"],
                "calendar_periods": summary_row["calendar_periods"],
                "active_periods": summary_row["effect_active_periods"],
                "zero_event_periods": summary_row["zero_event_periods"],
                "zero_control_ineligible_periods": summary_row[
                    "zero_control_ineligible_periods"
                ],
                "missing_source_periods": summary_row["missing_source_periods"],
                "failed_run_periods": summary_row["failed_run_periods"],
                "support": summary_row["condition_support"],
                "selection_rate_mean_all_evaluable_periods": summary_row[
                    "selection_rate_mean"
                ],
                "selection_rate_evaluable_periods": summary_row[
                    "selection_rate_periods"
                ],
                "equal_day_delta_z": summary_row["equal_day_delta_z"],
                "primary_bootstrap_block_length": config.primary_block_length,
                "bootstrap_ci_low": boot_row["ci_low"],
                "bootstrap_ci_high": boot_row["ci_high"],
                "bootstrap_probability_gt_zero": boot_row[
                    "bootstrap_probability_gt_zero"
                ],
                "bootstrap_p_one_sided_positive": boot_row[
                    "p_one_sided_positive_centered"
                ],
                "bootstrap_p_two_sided": boot_row["p_two_sided_centered"],
                "family_raw_trial_count": fwer_row["raw_trial_count"],
                "dependence_effective_trial_count": fwer_row[
                    "dependence_effective_trial_count"
                ],
                "effective_trial_method": fwer_row["effective_trial_method"],
                "family_max_t_statistic": fwer_row["max_t_statistic"],
                "family_alpha": fwer_row["family_alpha"],
                "family_simultaneous_lower_bound": fwer_row[
                    "simultaneous_lower_bound"
                ],
                "family_fwer_p": fwer_row["single_step_fwer_p"],
                "family_joint_valid_bootstrap_repetitions": fwer_row[
                    "joint_valid_bootstrap_repetitions"
                ],
                "family_joint_valid_bootstrap_rate": fwer_row[
                    "joint_valid_bootstrap_rate"
                ],
                "family_correction_scope": fwer_row["correction_scope"],
                "historical_selection_exposure_included": fwer_row[
                    "historical_selection_exposure_included"
                ],
                "total_selection_exposure_is_lower_bound": fwer_row[
                    "total_selection_exposure_is_lower_bound"
                ],
                "current_family_correction_is_only_historical_exposure_lower_bound": fwer_row[
                    "current_family_correction_is_only_historical_exposure_lower_bound"
                ],
            }
        )
    return pl.DataFrame(rows).sort(*GATE_COLUMNS)


def evaluate_robustness(
    *,
    requested_calendar: pl.DataFrame,
    daily_aggregates: pl.DataFrame | None = None,
    event_cache: pl.DataFrame | None = None,
    daily_control: pl.DataFrame | None = None,
    config: RobustnessConfig | None = None,
    trial_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> RobustnessResult:
    """Evaluate exactly one aggregate input mode on one common calendar."""

    active_config = config or RobustnessConfig()
    calendar = prepare_requested_calendar(requested_calendar)
    if (daily_aggregates is None) == (event_cache is None):
        raise ValueError("provide exactly one of daily_aggregates or event_cache")
    if event_cache is not None:
        if daily_control is None:
            raise ValueError("daily_control is required with event_cache")
        prepared_daily = build_daily_aggregates_from_event_cache(
            event_cache, daily_control
        )
        input_mode = "event_level_compact_cache"
    else:
        if daily_control is not None:
            raise ValueError("daily_control is only valid with event_cache")
        assert daily_aggregates is not None
        prepared_daily = prepare_daily_aggregates(daily_aggregates)
        input_mode = "daily_condition_control_aggregates"

    daily = align_daily_to_requested_calendar(prepared_daily, calendar)

    dates = daily["Date"].unique().sort().to_list()
    if not dates:
        raise RobustnessContractError("evaluation calendar is empty")
    summary = _cell_summary(daily)
    loo_daily, loo_summary = _build_loo(daily)
    factorial_daily, factorial_summary = _build_factorial_contrasts(daily)

    cell_ids = canonical_cell_ids()
    cell_matrix = _matrix_from_complete_panel(
        daily,
        dates=dates,
        id_col="cell_id",
        value_col="daily_delta_z",
        target_ids=cell_ids,
    )
    loo_ids = tuple(contrast_id for contrast_id, _, _ in LOO_SPECS)
    loo_matrix = _matrix_from_complete_panel(
        loo_daily,
        dates=dates,
        id_col="contrast_id",
        value_col="paired_increment_delta_z",
        target_ids=loo_ids,
    )
    factorial_ids = tuple(
        str(value) for value in factorial_summary["contrast_id"].to_list()
    )
    factorial_matrix = _matrix_from_complete_panel(
        factorial_daily,
        dates=dates,
        id_col="contrast_id",
        value_col="paired_contrast_delta_z",
        target_ids=factorial_ids,
    )
    bootstrap_rows = _bootstrap_rows(
        cell_matrix,
        cell_ids,
        target_kind="cell_effect",
        config=active_config,
    )
    bootstrap_rows.extend(
        _bootstrap_rows(
            loo_matrix,
            loo_ids,
            target_kind="loo_increment",
            config=active_config,
        )
    )
    bootstrap_rows.extend(
        _bootstrap_rows(
            factorial_matrix,
            factorial_ids,
            target_kind="factorial_contrast",
            config=active_config,
        )
    )
    bootstrap = pl.DataFrame(bootstrap_rows).sort(
        "target_kind", "target_id", "block_length_requested"
    )
    loo_primary_bootstrap = bootstrap.filter(
        (pl.col("target_kind") == "loo_increment")
        & pl.col("primary_block_length")
    ).select(
        pl.col("target_id").alias("contrast_id"),
        "bootstrap_probability_gt_zero",
        pl.col("p_one_sided_positive_centered").alias(
            "bootstrap_p_one_sided_positive"
        ),
        pl.col("ci_low").alias("bootstrap_ci_low"),
        pl.col("ci_high").alias("bootstrap_ci_high"),
    )
    loo_summary = loo_summary.join(
        loo_primary_bootstrap, on="contrast_id", how="left", validate="1:1"
    )
    factorial_primary_bootstrap = bootstrap.filter(
        (pl.col("target_kind") == "factorial_contrast")
        & pl.col("primary_block_length")
    ).select(
        pl.col("target_id").alias("contrast_id"),
        "bootstrap_probability_gt_zero",
        pl.col("p_one_sided_positive_centered").alias(
            "bootstrap_p_one_sided_positive"
        ),
        pl.col("ci_low").alias("bootstrap_ci_low"),
        pl.col("ci_high").alias("bootstrap_ci_high"),
    )
    factorial_summary = factorial_summary.join(
        factorial_primary_bootstrap,
        on="contrast_id",
        how="left",
        validate="1:1",
    )

    fwer_rows = _maxstat_rows(
        cell_matrix,
        cell_ids,
        family_id=active_config.family_id,
        config=active_config,
    )
    fwer = pl.DataFrame(fwer_rows).sort(
        "family_id", "target_id", "block_length_requested"
    )
    temporal = _temporal_blocks(daily, loo_daily, config=active_config)
    ledger = _trial_ledger(
        summary,
        bootstrap,
        fwer,
        config=active_config,
        evaluation_start=str(dates[0]),
        evaluation_end=str(dates[-1]),
        trial_metadata=trial_metadata,
    )
    effective = fwer.filter(
        (pl.col("family_id") == active_config.family_id)
        & pl.col("primary_block_length")
    )["dependence_effective_trial_count"][0]
    manifest = {
        "contract": "fh_neg_rejection_robustness_evaluator/v1",
        "input_mode": input_mode,
        "config": asdict(active_config),
        "calendar": {
            "start": dates[0],
            "end": dates[-1],
            "periods": len(dates),
            "evidence_class": calendar["evidence_class"][0],
            "status_counts": {
                str(row["calendar_status"]): int(row["len"])
                for row in calendar.group_by("calendar_status")
                .len()
                .iter_rows(named=True)
            },
        },
        "cell_ids": list(cell_ids),
        "full_cell_id": FULL_CELL_ID,
        "loo_pairs": [
            {
                "contrast_id": contrast_id,
                "removed_gate": gate,
                "full_cell_id": FULL_CELL_ID,
                "reduced_cell_id": reduced,
            }
            for contrast_id, gate, reduced in LOO_SPECS
        ],
        "effect_zero_support_policy": "undefined_not_zero",
        "bootstrap": {
            "method": "circular_moving_block_synchronous_trials",
            "primary_block_length": active_config.primary_block_length,
            "sensitivity_block_lengths": list(
                active_config.block_length_sensitivity
            ),
            "seed": active_config.bootstrap_seed,
            "repetitions": active_config.bootstrap_reps,
        },
        "fwer_method": "circular_moving_block_single_step_max_t_one_sided",
        "fwer_alternative": "positive",
        "family_alpha": active_config.family_alpha,
        "raw_trial_count": len(cell_ids),
        "dependence_effective_trial_count": effective,
        "selection_bias_scope": {
            "correction_scope": "current_frozen_eight_cell_family_only",
            "historical_selection_exposure_included": False,
            "current_family_raw_trial_count": len(cell_ids),
            "raw_and_effective_trial_counts_are_lower_bounds_for_total_selection_exposure": True,
            "current_eight_cell_correction_is_only_a_lower_bound_on_historical_selection_exposure": True,
            "limitation": (
                "The current eight-cell max-stat audit corrects only the frozen "
                "D/G/B factorial. Earlier formal and informal selector searches "
                "remain unresolved historical selection exposure."
            ),
        },
        "parsimony_contract": {
            "paired_calendar": "full_anchor_active_sessions_with_finite_ablation_effect",
            "contrasts": {
                contrast_id: {
                    "full_cell_id": FULL_CELL_ID,
                    "reduced_cell_id": reduced,
                    "removed_gate": gate,
                }
                for contrast_id, gate, reduced in LOO_SPECS
            },
            "pass_probability_threshold": 0.80,
            "requires_positive_median_increment": True,
        },
        "factorial_mechanism_diagnostics": {
            "selection_use": "diagnostic_only",
            "conditional_main_increments": 12,
            "conditional_two_way_interactions": 6,
            "three_way_interactions": 1,
            "paired_bootstrap": True,
        },
        "dsr_used": False,
        "output_rows": {
            "requested_calendar": len(calendar),
            "daily_effects": len(daily),
            "cell_summary": len(summary),
            "loo_daily": len(loo_daily),
            "loo_summary": len(loo_summary),
            "factorial_contrast_daily": len(factorial_daily),
            "factorial_contrasts": len(factorial_summary),
            "bootstrap": len(bootstrap),
            "fwer_audit": len(fwer),
            "temporal_blocks": len(temporal),
            "trial_ledger": len(ledger),
        },
    }
    return RobustnessResult(
        requested_calendar=calendar,
        daily_effects=daily,
        cell_summary=summary,
        loo_daily=loo_daily,
        loo_summary=loo_summary,
        factorial_contrast_daily=factorial_daily,
        factorial_contrasts=factorial_summary,
        bootstrap=bootstrap,
        fwer_audit=fwer,
        temporal_blocks=temporal,
        trial_ledger=ledger,
        manifest=manifest,
    )


def write_result(result: RobustnessResult, output_dir: Path) -> None:
    """Write the evaluator outputs without modifying any upstream artifact."""

    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "requested_calendar.parquet": result.requested_calendar,
        "daily_effects.parquet": result.daily_effects,
        "cell_summary.parquet": result.cell_summary,
        "loo_daily.parquet": result.loo_daily,
        "loo_summary.parquet": result.loo_summary,
        "factorial_contrast_daily.parquet": result.factorial_contrast_daily,
        "factorial_contrasts.parquet": result.factorial_contrasts,
        "bootstrap.parquet": result.bootstrap,
        "fwer_audit.parquet": result.fwer_audit,
        "temporal_blocks.parquet": result.temporal_blocks,
        "trial_ledger.parquet": result.trial_ledger,
    }
    for name, frame in frames.items():
        frame.write_parquet(output_dir / name, compression="zstd")
    (output_dir / "run_manifest.json").write_text(
        json.dumps(result.manifest, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen fh_neg_rejection eight-cell robustness panel"
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--daily-aggregates", type=Path)
    inputs.add_argument("--event-cache", type=Path)
    parser.add_argument("--requested-calendar", type=Path, required=True)
    parser.add_argument("--daily-control", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=5_000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = RobustnessConfig(
        bootstrap_reps=args.bootstrap_reps,
    )
    if args.daily_aggregates is not None:
        if args.daily_control is not None:
            raise SystemExit("--daily-control is only valid with --event-cache")
        result = evaluate_robustness(
            requested_calendar=pl.read_parquet(args.requested_calendar),
            daily_aggregates=pl.read_parquet(args.daily_aggregates),
            config=config,
        )
    else:
        if args.daily_control is None:
            raise SystemExit("--daily-control is required with --event-cache")
        result = evaluate_robustness(
            requested_calendar=pl.read_parquet(args.requested_calendar),
            event_cache=pl.read_parquet(args.event_cache),
            daily_control=pl.read_parquet(args.daily_control),
            config=config,
        )
    write_result(result, args.output_dir)
    print(json.dumps(result.manifest, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
