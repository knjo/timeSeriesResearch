"""Frozen secondary diagnostics for the fh_neg_rejection robustness study.

This module completes the calculations frozen in ``study_contract_amendment_2``.
It consumes the full runner's ``daily_stats.parquet`` and the separately
materialized retrospective evaluator outputs.  The evaluator calendar owns
period status and evidence classification; runner sums supply raw and
beta-residual basis-point numerators.

The outputs are diagnostics only.  They do not select a condition, change the
eight-cell family, or create a deployment decision.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl


STUDY_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = STUDY_DIR / "artifacts"
DEFAULT_OUTPUT_DIR = ARTIFACT_ROOT / "secondary_diagnostics_v1"
DEFAULT_ARTIFACT_OOS_CALENDAR = (
    ARTIFACT_ROOT / "calendar_artifact_oos_v1" / "requested_calendar.parquet"
)
ARTIFACT_OOS_OUTPUT_NAME = "artifact_oos_20260826_descriptive_v1"
AMENDMENT_PATH = STUDY_DIR / "study_contract_amendment_2.json"

SECONDARY_CONTRACT_ID = "fh_neg_rejection_secondary_diagnostics/v1"
ARTIFACT_OOS_CONTRACT_ID = "fh_neg_rejection_artifact_oos_descriptive/v1"
PARENT_CONTRACT_ID = "fh_neg_rejection_robustness/fixed_factorial_v1"
EVALUATOR_CONTRACT_ID = "fh_neg_rejection_robustness_evaluator/v1"
FAMILY_ID = "fh_neg_rejection_dgb_factorial"
FAMILY_VERSION = "1"
FULL_CELL_ID = "d1_g1_b1"
RETROSPECTIVE_EVIDENCE_CLASS = "retrospective_diagnostic"

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
CELL_IDS = tuple(f"d{depth}_g{far}_b{bear}" for depth, far, bear in CELL_SPECS)
CELL_BITS = {
    cell_id: bits for cell_id, bits in zip(CELL_IDS, CELL_SPECS, strict=True)
}

RUNNER_SUM_COLUMNS = (
    "condition_sum_z",
    "control_sum_z",
    "condition_sum_raw_bp",
    "control_sum_raw_bp",
    "condition_sum_residual_bp",
    "control_sum_residual_bp",
)
PERIOD_STATUSES = {
    "active",
    "zero_event",
    "zero_control_ineligible",
    "missing_source",
    "run_failed",
}

ONE_SIDED_90_NORMAL_QUANTILE = 1.2815515655446004
FLOAT_ABS_TOLERANCE = 1.0e-12
FLOAT_REL_TOLERANCE = 1.0e-10


class SecondaryDiagnosticsContractError(ValueError):
    """Raised when an input cannot satisfy the frozen secondary contract."""


@dataclass(frozen=True)
class SecondaryDiagnosticsConfig:
    """Outcome-independent mechanics frozen by amendment 2."""

    bootstrap_seed: int = 20_260_827
    bootstrap_repetitions: int = 5_000
    block_lengths: tuple[int, ...] = (5, 10, 20)
    primary_block_length: int = 10
    confidence_level: float = 0.90
    temporal_block_count: int = 3
    hac_lags: tuple[int, ...] = (1, 5, 10, 20)
    anchor_cell_id: str = FULL_CELL_ID

    def __post_init__(self) -> None:
        if self.bootstrap_seed != 20_260_827:
            raise ValueError("bootstrap_seed is frozen by amendment 2")
        if self.bootstrap_repetitions != 5_000:
            raise ValueError("bootstrap_repetitions is frozen by amendment 2")
        if self.block_lengths != (5, 10, 20):
            raise ValueError("block_lengths are frozen by amendment 2")
        if self.primary_block_length != 10:
            raise ValueError("primary_block_length is frozen by amendment 2")
        if self.confidence_level != 0.90:
            raise ValueError("confidence_level is frozen by amendment 2")
        if self.temporal_block_count != 3:
            raise ValueError("temporal_block_count is frozen by amendment 2")
        if self.hac_lags != (1, 5, 10, 20):
            raise ValueError("hac_lags are frozen by amendment 2")
        if self.anchor_cell_id != FULL_CELL_ID:
            raise ValueError("anchor_cell_id is frozen by amendment 2")


@dataclass(frozen=True)
class SecondaryDiagnosticsResult:
    """All versioned secondary outputs for one retrospective evaluation."""

    daily_effects: pl.DataFrame
    equal_active_bp: pl.DataFrame
    monthly_effects: pl.DataFrame
    monthly_summary: pl.DataFrame
    hac_diagnostics: pl.DataFrame
    temporal_basic_intervals: pl.DataFrame
    manifest: dict[str, Any]


@dataclass(frozen=True)
class ArtifactOosDescriptiveResult:
    """Strictly separate one-session description with no inference."""

    cell_descriptive: pl.DataFrame
    loo_descriptive: pl.DataFrame
    manifest: dict[str, Any]


def _require_columns(
    frame: pl.DataFrame,
    columns: Sequence[str],
    source: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise SecondaryDiagnosticsContractError(
            f"{source}: missing columns {missing}"
        )


def _has_null(frame: pl.DataFrame, columns: Sequence[str]) -> bool:
    return bool(
        frame.select(
            pl.any_horizontal(
                [pl.col(column).is_null() for column in columns]
            ).any()
        ).item()
    )


def _is_close(left: object, right: object) -> bool:
    if left is None or right is None:
        return left is None and right is None
    left_value = float(left)
    right_value = float(right)
    if not math.isfinite(left_value) or not math.isfinite(right_value):
        return False
    return math.isclose(
        left_value,
        right_value,
        rel_tol=FLOAT_REL_TOLERANCE,
        abs_tol=FLOAT_ABS_TOLERANCE,
    )


def _validate_yyyymmdd(date: str, *, source: str) -> None:
    try:
        parsed = datetime.strptime(date, "%Y%m%d")
    except ValueError as exc:
        raise SecondaryDiagnosticsContractError(
            f"{source}: invalid YYYYMMDD Date {date!r}"
        ) from exc
    if parsed.strftime("%Y%m%d") != date:
        raise SecondaryDiagnosticsContractError(
            f"{source}: invalid YYYYMMDD Date {date!r}"
        )


def _validate_support_sum(
    frame: pl.DataFrame,
    *,
    count_column: str,
    sum_column: str,
    source: str,
) -> None:
    if frame.filter(pl.col(count_column) < 0).height:
        raise SecondaryDiagnosticsContractError(
            f"{source}: {count_column} cannot be negative"
        )
    if frame.filter(
        (pl.col(count_column) > 0)
        & (pl.col(sum_column).is_null() | ~pl.col(sum_column).is_finite())
    ).height:
        raise SecondaryDiagnosticsContractError(
            f"{source}: positive {count_column} requires finite {sum_column}"
        )
    if frame.filter(
        (pl.col(count_column) == 0)
        & pl.col(sum_column).is_not_null()
        & (
            ~pl.col(sum_column).is_finite()
            | (pl.col(sum_column).abs() > FLOAT_ABS_TOLERANCE)
        )
    ).height:
        raise SecondaryDiagnosticsContractError(
            f"{source}: zero {count_column} cannot have non-zero {sum_column}"
        )


def prepare_requested_calendar(frame: pl.DataFrame) -> pl.DataFrame:
    """Validate the evaluator's outcome-independent retrospective calendar."""

    required = (
        "Date",
        "source_complete",
        "run_status",
        "evidence_class",
        "calendar_status",
    )
    _require_columns(frame, required, "requested calendar")
    output = frame.select(required).with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("source_complete").cast(pl.Boolean, strict=False),
        pl.col("run_status").cast(pl.String),
        pl.col("evidence_class").cast(pl.String),
        pl.col("calendar_status").cast(pl.String),
    )
    if not len(output):
        raise SecondaryDiagnosticsContractError("requested calendar is empty")
    if _has_null(output, required):
        raise SecondaryDiagnosticsContractError(
            "requested calendar contains null fields"
        )
    if output["Date"].n_unique() != len(output):
        raise SecondaryDiagnosticsContractError(
            "requested calendar contains duplicate Date"
        )
    evidence_classes = set(output["evidence_class"].unique().to_list())
    if evidence_classes != {RETROSPECTIVE_EVIDENCE_CLASS}:
        raise SecondaryDiagnosticsContractError(
            "secondary diagnostics require exactly retrospective_diagnostic"
        )
    allowed_statuses = {"missing_source", "run_failed", "evaluation_expected"}
    unknown_statuses = sorted(
        set(output["calendar_status"].unique().to_list()) - allowed_statuses
    )
    if unknown_statuses:
        raise SecondaryDiagnosticsContractError(
            f"requested calendar has unknown statuses {unknown_statuses}"
        )
    for row in output.iter_rows(named=True):
        date = str(row["Date"])
        _validate_yyyymmdd(date, source="requested calendar")
        if date > "20260825":
            raise SecondaryDiagnosticsContractError(
                f"requested calendar: {date} is outside retrospective evidence"
            )
        expected_status = (
            "missing_source"
            if not bool(row["source_complete"])
            else (
                "evaluation_expected"
                if str(row["run_status"]).lower() == "success"
                else "run_failed"
            )
        )
        if row["calendar_status"] != expected_status:
            raise SecondaryDiagnosticsContractError(
                f"requested calendar: {date} status disagrees with source/run fields"
            )
    return output.sort("Date")


def prepare_artifact_oos_calendar(frame: pl.DataFrame) -> pl.DataFrame:
    """Validate the separately materialized one-session artifact-OOS calendar."""

    required = (
        "Date",
        "source_complete",
        "run_status",
        "evidence_class",
        "session_status",
    )
    _require_columns(frame, required, "artifact-OOS calendar")
    output = frame.select(required).with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("source_complete").cast(pl.Boolean, strict=False),
        pl.col("run_status").cast(pl.String),
        pl.col("evidence_class").cast(pl.String),
        pl.col("session_status").cast(pl.String),
    )
    if len(output) != 1:
        raise SecondaryDiagnosticsContractError(
            "artifact-OOS calendar must contain exactly one session"
        )
    if _has_null(output, required):
        raise SecondaryDiagnosticsContractError(
            "artifact-OOS calendar contains null fields"
        )
    row = output.row(0, named=True)
    expected = {
        "Date": "20260826",
        "source_complete": True,
        "run_status": "success",
        "evidence_class": "artifact_oos_non_pristine",
        "session_status": "evaluable",
    }
    drift = {
        key: {"expected": value, "actual": row[key]}
        for key, value in expected.items()
        if row[key] != value
    }
    if drift:
        raise SecondaryDiagnosticsContractError(
            f"artifact-OOS calendar identity/status drifted: {drift}"
        )
    return output


def prepare_evaluator_daily(
    frame: pl.DataFrame,
    calendar: pl.DataFrame,
) -> pl.DataFrame:
    """Validate the evaluator's complete date-by-eight-cell status panel."""

    required = (
        "Date",
        "cell_id",
        "use_depth",
        "use_far",
        "use_bear",
        "condition_n_z",
        "condition_sum_z",
        "control_n_z",
        "control_sum_z",
        "daily_delta_z",
        "condition_active",
        "effect_active",
        "evidence_class",
        "calendar_status",
        "period_status",
    )
    _require_columns(frame, required, "evaluator daily effects")
    output = frame.select(required).with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("cell_id").cast(pl.String),
        pl.col("use_depth").cast(pl.Int8, strict=False),
        pl.col("use_far").cast(pl.Int8, strict=False),
        pl.col("use_bear").cast(pl.Int8, strict=False),
        pl.col("condition_n_z").cast(pl.Int64, strict=False),
        pl.col("condition_sum_z").cast(pl.Float64, strict=False),
        pl.col("control_n_z").cast(pl.Int64, strict=False),
        pl.col("control_sum_z").cast(pl.Float64, strict=False),
        pl.col("daily_delta_z").cast(pl.Float64, strict=False),
        pl.col("condition_active").cast(pl.Boolean, strict=False),
        pl.col("effect_active").cast(pl.Boolean, strict=False),
        pl.col("evidence_class").cast(pl.String),
        pl.col("calendar_status").cast(pl.String),
        pl.col("period_status").cast(pl.String),
    )
    expected_rows = len(calendar) * len(CELL_IDS)
    if len(output) != expected_rows:
        raise SecondaryDiagnosticsContractError(
            f"evaluator daily effects has {len(output)} rows, expected {expected_rows}"
        )
    if output.select(pl.struct(["Date", "cell_id"]).is_duplicated().any()).item():
        raise SecondaryDiagnosticsContractError(
            "evaluator daily effects has duplicate date/cell rows"
        )
    if set(output["Date"].unique().to_list()) != set(calendar["Date"].to_list()):
        raise SecondaryDiagnosticsContractError(
            "evaluator daily effects calendar disagrees with requested calendar"
        )
    if set(output["cell_id"].unique().to_list()) != set(CELL_IDS):
        raise SecondaryDiagnosticsContractError(
            "evaluator daily effects does not contain the frozen eight cells"
        )
    if set(output["period_status"].unique().to_list()) - PERIOD_STATUSES:
        raise SecondaryDiagnosticsContractError(
            "evaluator daily effects contains an unknown period_status"
        )

    calendar_status = dict(
        calendar.select("Date", "calendar_status").iter_rows()
    )
    for row in output.iter_rows(named=True):
        date = str(row["Date"])
        cell_id = str(row["cell_id"])
        if row["evidence_class"] != RETROSPECTIVE_EVIDENCE_CLASS:
            raise SecondaryDiagnosticsContractError(
                f"evaluator daily effects: {date}/{cell_id} evidence drift"
            )
        if row["calendar_status"] != calendar_status[date]:
            raise SecondaryDiagnosticsContractError(
                f"evaluator daily effects: {date}/{cell_id} calendar status drift"
            )
        expected_bits = CELL_BITS[cell_id]
        actual_bits = (
            int(row["use_depth"]),
            int(row["use_far"]),
            int(row["use_bear"]),
        )
        if actual_bits != expected_bits:
            raise SecondaryDiagnosticsContractError(
                f"evaluator daily effects: {cell_id} gate bits drifted"
            )
        status = str(row["period_status"])
        effect_active = bool(row["effect_active"])
        if effect_active != (status == "active"):
            raise SecondaryDiagnosticsContractError(
                f"evaluator daily effects: {date}/{cell_id} active status drift"
            )
        if effect_active:
            if row["condition_n_z"] is None or row["control_n_z"] is None:
                raise SecondaryDiagnosticsContractError(
                    f"evaluator daily effects: {date}/{cell_id} missing active support"
                )
            if int(row["condition_n_z"]) <= 0 or int(row["control_n_z"]) <= 0:
                raise SecondaryDiagnosticsContractError(
                    f"evaluator daily effects: {date}/{cell_id} non-positive active support"
                )
            value = row["daily_delta_z"]
            if value is None or not math.isfinite(float(value)):
                raise SecondaryDiagnosticsContractError(
                    f"evaluator daily effects: {date}/{cell_id} active effect is not finite"
                )
        elif row["daily_delta_z"] is not None:
            raise SecondaryDiagnosticsContractError(
                f"evaluator daily effects: {date}/{cell_id} inactive effect is defined"
            )
    return output.sort("Date", "use_depth", "use_far", "use_bear")


def prepare_runner_daily(frame: pl.DataFrame) -> pl.DataFrame:
    """Validate full-run sums without selecting or summarizing outcomes."""

    required = (
        "contract_id",
        "Date",
        "family_id",
        "family_version",
        "candidate_id",
        "label_profile_id",
        "label_profile_hash",
        "pool_id",
        "pool_version",
        "condition_n_z",
        "control_n_z",
        *RUNNER_SUM_COLUMNS,
    )
    _require_columns(frame, required, "runner daily stats")
    output = frame.select(required).with_columns(
        pl.col("contract_id").cast(pl.String),
        pl.col("Date").cast(pl.String),
        pl.col("family_id").cast(pl.String),
        pl.col("family_version").cast(pl.String),
        pl.col("candidate_id").cast(pl.String),
        pl.col("label_profile_id").cast(pl.String),
        pl.col("label_profile_hash").cast(pl.String),
        pl.col("pool_id").cast(pl.String),
        pl.col("pool_version").cast(pl.String),
        pl.col("condition_n_z").cast(pl.Int64, strict=False),
        pl.col("control_n_z").cast(pl.Int64, strict=False),
        *(pl.col(column).cast(pl.Float64, strict=False) for column in RUNNER_SUM_COLUMNS),
    )
    nonnull = (
        "contract_id",
        "Date",
        "family_id",
        "family_version",
        "candidate_id",
        "label_profile_id",
        "label_profile_hash",
        "pool_id",
        "pool_version",
        "condition_n_z",
        "control_n_z",
    )
    if _has_null(output, nonnull):
        raise SecondaryDiagnosticsContractError(
            "runner daily stats contains null identity or support"
        )
    if set(output["contract_id"].unique().to_list()) != {PARENT_CONTRACT_ID}:
        raise SecondaryDiagnosticsContractError("runner contract_id drifted")
    if set(output["family_id"].unique().to_list()) != {FAMILY_ID}:
        raise SecondaryDiagnosticsContractError("runner family_id drifted")
    if set(output["family_version"].unique().to_list()) != {FAMILY_VERSION}:
        raise SecondaryDiagnosticsContractError("runner family_version drifted")
    unknown_cells = sorted(set(output["candidate_id"].unique().to_list()) - set(CELL_IDS))
    if unknown_cells:
        raise SecondaryDiagnosticsContractError(
            f"runner daily stats has unknown cells {unknown_cells}"
        )
    if output.select(
        pl.struct(["Date", "candidate_id"]).is_duplicated().any()
    ).item():
        raise SecondaryDiagnosticsContractError(
            "runner daily stats has duplicate date/candidate rows"
        )
    for date in output["Date"].unique().to_list():
        _validate_yyyymmdd(str(date), source="runner daily stats")
    for sum_column in (
        "condition_sum_z",
        "condition_sum_raw_bp",
        "condition_sum_residual_bp",
    ):
        _validate_support_sum(
            output,
            count_column="condition_n_z",
            sum_column=sum_column,
            source="runner condition",
        )
    for sum_column in (
        "control_sum_z",
        "control_sum_raw_bp",
        "control_sum_residual_bp",
    ):
        _validate_support_sum(
            output,
            count_column="control_n_z",
            sum_column=sum_column,
            source="runner control",
        )
    if output.filter(pl.col("condition_n_z") > pl.col("control_n_z")).height:
        raise SecondaryDiagnosticsContractError(
            "runner condition support exceeds shared control support"
        )
    control_columns = (
        "control_n_z",
        "control_sum_z",
        "control_sum_raw_bp",
        "control_sum_residual_bp",
        "label_profile_id",
        "label_profile_hash",
        "pool_id",
        "pool_version",
    )
    disagreement = output.group_by("Date").agg(
        *(pl.col(column).n_unique().alias(column) for column in control_columns)
    ).filter(
        pl.any_horizontal([pl.col(column) != 1 for column in control_columns])
    )
    if disagreement.height:
        raise SecondaryDiagnosticsContractError(
            "runner shared control or pipeline identity differs within a date"
        )
    return output.sort("Date", "candidate_id")


def _descriptive_effects(row: Mapping[str, Any]) -> dict[str, float | None]:
    condition_n = int(row["condition_n_z"])
    control_n = int(row["control_n_z"])
    if condition_n <= 0 or control_n <= 0:
        return {
            "daily_delta_z": None,
            "daily_delta_raw_bp": None,
            "daily_delta_residual_bp": None,
        }
    return {
        "daily_delta_z": (
            float(row["condition_sum_z"]) / condition_n
            - float(row["control_sum_z"]) / control_n
        ),
        "daily_delta_raw_bp": (
            float(row["condition_sum_raw_bp"]) / condition_n
            - float(row["control_sum_raw_bp"]) / control_n
        ),
        "daily_delta_residual_bp": (
            float(row["condition_sum_residual_bp"]) / condition_n
            - float(row["control_sum_residual_bp"]) / control_n
        ),
    }


def build_artifact_oos_descriptive(
    *,
    runner_daily_stats: pl.DataFrame,
    artifact_oos_calendar: pl.DataFrame,
    input_lineage: Mapping[str, Any] | None = None,
) -> ArtifactOosDescriptiveResult:
    """Describe 20260826 alone, with no bootstrap or inferential statistic."""

    prepare_artifact_oos_calendar(artifact_oos_calendar)
    runner = prepare_runner_daily(runner_daily_stats)
    day = runner.filter(pl.col("Date") == "20260826")
    if len(day) != len(CELL_IDS):
        raise SecondaryDiagnosticsContractError(
            "artifact-OOS runner slice must contain exactly eight cell rows"
        )
    if set(day["candidate_id"].to_list()) != set(CELL_IDS):
        raise SecondaryDiagnosticsContractError(
            "artifact-OOS runner slice does not contain the frozen cells"
        )
    rows: list[dict[str, Any]] = []
    for cell_id in CELL_IDS:
        source = day.filter(pl.col("candidate_id") == cell_id).row(
            0, named=True
        )
        depth, far, bear = CELL_BITS[cell_id]
        condition_n = int(source["condition_n_z"])
        control_n = int(source["control_n_z"])
        rows.append(
            {
                "Date": "20260826",
                "evidence_class": "artifact_oos_non_pristine",
                "cell_id": cell_id,
                "use_depth": depth,
                "use_far": far,
                "use_bear": bear,
                "condition_n_z": condition_n,
                "control_n_z": control_n,
                "condition_support_ratio": (
                    condition_n / control_n if control_n > 0 else None
                ),
                **_descriptive_effects(source),
                "effect_defined": condition_n > 0 and control_n > 0,
                "analysis_role": "one_session_descriptive_only",
                "inference_performed": False,
            }
        )
    cells = pl.DataFrame(rows, infer_schema_length=None).with_columns(
        *(
            pl.col(column).cast(pl.Int8)
            for column in ("use_depth", "use_far", "use_bear")
        )
    ).sort("use_depth", "use_far", "use_bear")

    full = cells.filter(pl.col("cell_id") == FULL_CELL_ID).row(0, named=True)
    loo_specs = (
        ("D_increment", "depth", "d0_g1_b1"),
        ("G_increment", "far", "d1_g0_b1"),
        ("B_increment", "bear", "d1_g1_b0"),
    )
    loo_rows: list[dict[str, Any]] = []
    for contrast_id, removed_gate, reduced_cell_id in loo_specs:
        reduced = cells.filter(
            pl.col("cell_id") == reduced_cell_id
        ).row(0, named=True)
        common_defined = bool(full["effect_defined"]) and bool(
            reduced["effect_defined"]
        )
        increment_values: dict[str, float | None] = {}
        for metric in (
            "daily_delta_z",
            "daily_delta_raw_bp",
            "daily_delta_residual_bp",
        ):
            output_name = metric.replace("daily_delta", "full_minus_loo")
            increment_values[output_name] = (
                float(full[metric]) - float(reduced[metric])
                if common_defined
                else None
            )
        loo_rows.append(
            {
                "Date": "20260826",
                "evidence_class": "artifact_oos_non_pristine",
                "contrast_id": contrast_id,
                "removed_gate": removed_gate,
                "full_cell_id": FULL_CELL_ID,
                "reduced_cell_id": reduced_cell_id,
                "full_condition_n_z": int(full["condition_n_z"]),
                "reduced_condition_n_z": int(reduced["condition_n_z"]),
                "control_n_z": int(full["control_n_z"]),
                **increment_values,
                "common_effect_defined": common_defined,
                "analysis_role": "one_session_descriptive_only",
                "inference_performed": False,
            }
        )
    loo = pl.DataFrame(loo_rows, infer_schema_length=None)
    frames = {
        "cell_descriptive.parquet": cells,
        "loo_descriptive.parquet": loo,
    }
    manifest = {
        "contract": ARTIFACT_OOS_CONTRACT_ID,
        "parent_contract": PARENT_CONTRACT_ID,
        "family_id": FAMILY_ID,
        "family_version": FAMILY_VERSION,
        "Date": "20260826",
        "evidence_class": "artifact_oos_non_pristine",
        "calendar_artifact": "calendar_artifact_oos_v1",
        "separation_policy": {
            "retrospective_pooled": False,
            "retrospective_outputs_referenced": False,
            "one_session_only": True,
        },
        "analysis_scope": {
            "descriptive_only": True,
            "hac_performed": False,
            "bootstrap_performed": False,
            "p_value_computed": False,
            "confidence_interval_computed": False,
            "inference_performed": False,
            "decision_created": False,
        },
        "effect_policy": {
            "z": "condition_sum_z/condition_n_z - control_sum_z/control_n_z",
            "raw_bp": (
                "condition_sum_raw_bp/condition_n_z - "
                "control_sum_raw_bp/control_n_z"
            ),
            "residual_bp": (
                "condition_sum_residual_bp/condition_n_z - "
                "control_sum_residual_bp/control_n_z"
            ),
            "inactive_effect": "undefined_not_zero",
        },
        "loo_contrasts": {
            contrast_id: {
                "full_cell_id": FULL_CELL_ID,
                "reduced_cell_id": reduced_cell_id,
                "removed_gate": removed_gate,
            }
            for contrast_id, removed_gate, reduced_cell_id in loo_specs
        },
        "input_lineage": dict(input_lineage or {}),
        "output_rows": {name: len(frame) for name, frame in frames.items()},
        "output_schemas": {name: _schema(frame) for name, frame in frames.items()},
    }
    return ArtifactOosDescriptiveResult(
        cell_descriptive=cells,
        loo_descriptive=loo,
        manifest=manifest,
    )


def validate_evaluator_manifest(manifest: Mapping[str, Any]) -> None:
    """Require the frozen retrospective evaluator identity and mechanics."""

    if manifest.get("contract") != EVALUATOR_CONTRACT_ID:
        raise SecondaryDiagnosticsContractError("evaluator contract drifted")
    if manifest.get("full_cell_id") != FULL_CELL_ID:
        raise SecondaryDiagnosticsContractError("evaluator anchor drifted")
    if tuple(manifest.get("cell_ids", ())) != CELL_IDS:
        raise SecondaryDiagnosticsContractError("evaluator cell family drifted")
    calendar = manifest.get("calendar", {})
    if calendar.get("evidence_class") != RETROSPECTIVE_EVIDENCE_CLASS:
        raise SecondaryDiagnosticsContractError(
            "evaluator manifest is not retrospective_diagnostic"
        )
    bootstrap = manifest.get("bootstrap", {})
    if bootstrap.get("method") != "circular_moving_block_synchronous_trials":
        raise SecondaryDiagnosticsContractError(
            "evaluator bootstrap is not the repaired circular method"
        )
    if int(bootstrap.get("seed", -1)) != 20_260_827:
        raise SecondaryDiagnosticsContractError("evaluator seed drifted")
    if int(bootstrap.get("repetitions", -1)) != 5_000:
        raise SecondaryDiagnosticsContractError(
            "evaluator must be the frozen 5000-repetition run"
        )


def _validate_runner_evaluator_match(
    runner: pl.DataFrame,
    evaluator_daily: pl.DataFrame,
    calendar: pl.DataFrame,
) -> pl.DataFrame:
    """Join raw/residual sums only after exact z/support lineage parity."""

    expected_dates = set(
        calendar.filter(pl.col("calendar_status") == "evaluation_expected")[
            "Date"
        ].to_list()
    )
    nonexpected_dates = set(calendar["Date"].to_list()) - expected_dates
    if runner.filter(pl.col("Date").is_in(nonexpected_dates)).height:
        raise SecondaryDiagnosticsContractError(
            "runner contains rows for a missing-source or failed evaluator date"
        )
    focused = runner.filter(pl.col("Date").is_in(expected_dates))
    expected_rows = len(expected_dates) * len(CELL_IDS)
    if len(focused) != expected_rows:
        raise SecondaryDiagnosticsContractError(
            f"runner has {len(focused)} retrospective evaluator rows, expected {expected_rows}"
        )
    for date in expected_dates:
        cells = set(
            focused.filter(pl.col("Date") == date)["candidate_id"].to_list()
        )
        if cells != set(CELL_IDS):
            raise SecondaryDiagnosticsContractError(
                f"runner {date} does not contain the frozen eight cells"
            )

    runner_columns = focused.select(
        "Date",
        pl.col("candidate_id").alias("cell_id"),
        pl.col("condition_n_z").alias("runner_condition_n_z"),
        pl.col("control_n_z").alias("runner_control_n_z"),
        *(pl.col(column).alias(f"runner_{column}") for column in RUNNER_SUM_COLUMNS),
    )
    joined = evaluator_daily.join(
        runner_columns,
        on=("Date", "cell_id"),
        how="left",
        validate="1:1",
    )
    for row in joined.iter_rows(named=True):
        date = str(row["Date"])
        cell_id = str(row["cell_id"])
        expected = row["calendar_status"] == "evaluation_expected"
        runner_n = row["runner_condition_n_z"]
        if expected != (runner_n is not None):
            raise SecondaryDiagnosticsContractError(
                f"runner/evaluator presence mismatch at {date}/{cell_id}"
            )
        if not expected:
            continue
        if int(row["condition_n_z"]) != int(runner_n):
            raise SecondaryDiagnosticsContractError(
                f"runner/evaluator condition support mismatch at {date}/{cell_id}"
            )
        if int(row["control_n_z"]) != int(row["runner_control_n_z"]):
            raise SecondaryDiagnosticsContractError(
                f"runner/evaluator control support mismatch at {date}/{cell_id}"
            )
        for evaluator_column, runner_column in (
            ("condition_sum_z", "runner_condition_sum_z"),
            ("control_sum_z", "runner_control_sum_z"),
        ):
            if not _is_close(row[evaluator_column], row[runner_column]):
                raise SecondaryDiagnosticsContractError(
                    f"runner/evaluator {evaluator_column} mismatch at {date}/{cell_id}"
                )

    active = pl.col("effect_active")
    output = joined.with_columns(
        pl.when(active)
        .then(
            pl.col("runner_condition_sum_raw_bp") / pl.col("condition_n_z")
            - pl.col("runner_control_sum_raw_bp") / pl.col("control_n_z")
        )
        .alias("daily_delta_raw_bp"),
        pl.when(active)
        .then(
            pl.col("runner_condition_sum_residual_bp") / pl.col("condition_n_z")
            - pl.col("runner_control_sum_residual_bp") / pl.col("control_n_z")
        )
        .alias("daily_delta_residual_bp"),
    )
    if output.filter(
        pl.col("effect_active")
        & (
            pl.col("daily_delta_raw_bp").is_null()
            | ~pl.col("daily_delta_raw_bp").is_finite()
            | pl.col("daily_delta_residual_bp").is_null()
            | ~pl.col("daily_delta_residual_bp").is_finite()
        )
    ).height:
        raise SecondaryDiagnosticsContractError(
            "active raw/residual BP effect is not finite"
        )
    if output.filter(
        ~pl.col("effect_active")
        & (
            pl.col("daily_delta_raw_bp").is_not_null()
            | pl.col("daily_delta_residual_bp").is_not_null()
        )
    ).height:
        raise SecondaryDiagnosticsContractError(
            "inactive raw/residual BP effect must remain undefined"
        )
    return output.select(
        "Date",
        "evidence_class",
        "calendar_status",
        "period_status",
        "cell_id",
        "use_depth",
        "use_far",
        "use_bear",
        "condition_n_z",
        "control_n_z",
        "condition_sum_z",
        "control_sum_z",
        pl.col("runner_condition_sum_raw_bp").alias("condition_sum_raw_bp"),
        pl.col("runner_control_sum_raw_bp").alias("control_sum_raw_bp"),
        pl.col("runner_condition_sum_residual_bp").alias(
            "condition_sum_residual_bp"
        ),
        pl.col("runner_control_sum_residual_bp").alias(
            "control_sum_residual_bp"
        ),
        "daily_delta_z",
        "daily_delta_raw_bp",
        "daily_delta_residual_bp",
        "condition_active",
        "effect_active",
    ).sort("Date", "use_depth", "use_far", "use_bear")


def summarize_equal_active_bp(daily: pl.DataFrame) -> pl.DataFrame:
    """Compute equal-active-session raw and residual BP point estimates."""

    rows: list[dict[str, Any]] = []
    for cell_id in CELL_IDS:
        subset = daily.filter(pl.col("cell_id") == cell_id)
        active = subset.filter(pl.col("effect_active"))
        raw_values = active["daily_delta_raw_bp"].to_numpy()
        residual_values = active["daily_delta_residual_bp"].to_numpy()
        depth, far, bear = CELL_BITS[cell_id]
        rows.append(
            {
                "cell_id": cell_id,
                "use_depth": depth,
                "use_far": far,
                "use_bear": bear,
                "calendar_sessions": len(subset),
                "active_sessions": len(active),
                "zero_event_sessions": int(
                    (subset["period_status"] == "zero_event").sum()
                ),
                "zero_control_ineligible_sessions": int(
                    (subset["period_status"] == "zero_control_ineligible").sum()
                ),
                "missing_source_sessions": int(
                    (subset["period_status"] == "missing_source").sum()
                ),
                "failed_run_sessions": int(
                    (subset["period_status"] == "run_failed").sum()
                ),
                "equal_active_raw_bp": (
                    float(np.mean(raw_values)) if len(raw_values) else None
                ),
                "equal_active_residual_bp": (
                    float(np.mean(residual_values))
                    if len(residual_values)
                    else None
                ),
                "positive_raw_bp_session_ratio": (
                    float(np.mean(raw_values > 0.0)) if len(raw_values) else None
                ),
                "positive_residual_bp_session_ratio": (
                    float(np.mean(residual_values > 0.0))
                    if len(residual_values)
                    else None
                ),
            }
        )
    return pl.DataFrame(rows).with_columns(
        *(
            pl.col(column).cast(pl.Int8)
            for column in ("use_depth", "use_far", "use_bear")
        )
    ).sort("use_depth", "use_far", "use_bear")


def build_monthly_outputs(
    daily: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Average active daily effects within month before counting signs."""

    with_month = daily.with_columns(pl.col("Date").str.slice(0, 6).alias("month"))
    monthly = (
        with_month.group_by(
            "cell_id", "use_depth", "use_far", "use_bear", "month"
        )
        .agg(
            pl.len().alias("calendar_sessions"),
            pl.col("effect_active").sum().alias("active_sessions"),
            (pl.col("period_status") == "zero_event")
            .sum()
            .alias("zero_event_sessions"),
            pl.col("daily_delta_z").mean().alias("month_mean_delta_z"),
            pl.col("daily_delta_raw_bp").mean().alias(
                "month_mean_raw_bp"
            ),
            pl.col("daily_delta_residual_bp").mean().alias(
                "month_mean_residual_bp"
            ),
        )
        .with_columns(
            (pl.col("active_sessions") > 0).alias("month_active"),
            pl.col("calendar_sessions").cast(pl.Int64),
            pl.col("active_sessions").cast(pl.Int64),
            pl.col("zero_event_sessions").cast(pl.Int64),
        )
        .sort("month", "use_depth", "use_far", "use_bear")
    )
    rows: list[dict[str, Any]] = []
    for cell_id in CELL_IDS:
        subset = monthly.filter(pl.col("cell_id") == cell_id)
        active = subset.filter(pl.col("month_active"))
        positive = int((active["month_mean_delta_z"] > 0.0).sum())
        depth, far, bear = CELL_BITS[cell_id]
        rows.append(
            {
                "cell_id": cell_id,
                "use_depth": depth,
                "use_far": far,
                "use_bear": bear,
                "requested_months": len(subset),
                "active_months": len(active),
                "positive_months": positive,
                "positive_month_ratio": (
                    positive / len(active) if len(active) else None
                ),
            }
        )
    summary = pl.DataFrame(rows).with_columns(
        *(
            pl.col(column).cast(pl.Int8)
            for column in ("use_depth", "use_far", "use_bear")
        )
    ).sort("use_depth", "use_far", "use_bear")
    return monthly, summary


def bartlett_newey_west(
    values: Sequence[float],
    requested_lag: int,
) -> dict[str, float | int | None]:
    """Return Bartlett Newey-West uncertainty for an ordered finite series."""

    series = np.asarray(values, dtype=float)
    if series.ndim != 1 or not np.isfinite(series).all():
        raise ValueError("HAC series must be one-dimensional and finite")
    if requested_lag < 0:
        raise ValueError("requested_lag cannot be negative")
    n_periods = len(series)
    mean = float(np.mean(series)) if n_periods else None
    if n_periods < 2:
        return {
            "active_sessions": n_periods,
            "requested_lag": requested_lag,
            "effective_lag": max(0, n_periods - 1),
            "mean_delta_z": mean,
            "hac_long_run_variance": None,
            "hac_standard_error": None,
            "hac_t_stat": None,
            "p_one_sided_positive_normal": None,
            "one_sided_90_lcb": None,
            "one_sided_90_ucb": None,
        }
    effective_lag = min(requested_lag, n_periods - 1)
    centered = series - float(mean)
    long_run_variance = float(np.dot(centered, centered) / n_periods)
    for lag in range(1, effective_lag + 1):
        weight = 1.0 - lag / (effective_lag + 1.0)
        autocovariance = float(
            np.dot(centered[lag:], centered[:-lag]) / n_periods
        )
        long_run_variance += 2.0 * weight * autocovariance
    if long_run_variance < 0.0 and abs(long_run_variance) <= 1.0e-14:
        long_run_variance = 0.0
    if long_run_variance < 0.0:
        raise SecondaryDiagnosticsContractError(
            "Bartlett Newey-West long-run variance became negative"
        )
    standard_error = math.sqrt(long_run_variance / n_periods)
    if standard_error <= 0.0:
        statistic = None
        p_value = None
        lower = None
        upper = None
    else:
        statistic = float(mean) / standard_error
        p_value = 0.5 * math.erfc(statistic / math.sqrt(2.0))
        lower = float(mean) - ONE_SIDED_90_NORMAL_QUANTILE * standard_error
        upper = float(mean) + ONE_SIDED_90_NORMAL_QUANTILE * standard_error
    return {
        "active_sessions": n_periods,
        "requested_lag": requested_lag,
        "effective_lag": effective_lag,
        "mean_delta_z": mean,
        "hac_long_run_variance": long_run_variance,
        "hac_standard_error": standard_error,
        "hac_t_stat": statistic,
        "p_one_sided_positive_normal": p_value,
        "one_sided_90_lcb": lower,
        "one_sided_90_ucb": upper,
    }


def build_hac_diagnostics(
    daily: pl.DataFrame,
    *,
    config: SecondaryDiagnosticsConfig,
) -> pl.DataFrame:
    """Run every frozen HAC lag for each cell, without choosing one lag."""

    rows: list[dict[str, Any]] = []
    for cell_id in CELL_IDS:
        subset = daily.filter(
            (pl.col("cell_id") == cell_id) & pl.col("effect_active")
        ).sort("Date")
        values = subset["daily_delta_z"].to_numpy()
        depth, far, bear = CELL_BITS[cell_id]
        for requested_lag in config.hac_lags:
            result = bartlett_newey_west(values, requested_lag)
            rows.append(
                {
                    "cell_id": cell_id,
                    "use_depth": depth,
                    "use_far": far,
                    "use_bear": bear,
                    **result,
                    "kernel": "Bartlett Newey-West",
                    "series_policy": (
                        "finite active-session daily_delta_z in requested-session order"
                    ),
                    "inference_role": "secondary_diagnostic_only",
                }
            )
    return pl.DataFrame(rows, infer_schema_length=None).with_columns(
        *(
            pl.col(column).cast(pl.Int8)
            for column in ("use_depth", "use_far", "use_bear")
        )
    ).sort("use_depth", "use_far", "use_bear", "requested_lag")


def circular_moving_block_indices(
    n_periods: int,
    *,
    block_length: int,
    repetitions: int,
    seed: int,
) -> np.ndarray:
    """Draw the same deterministic circular calendar resample as amendment 1."""

    if n_periods < 1:
        raise ValueError("n_periods must be positive")
    if block_length < 1 or repetitions < 1:
        raise ValueError("block_length and repetitions must be positive")
    effective = min(block_length, n_periods)
    blocks_needed = math.ceil(n_periods / effective)
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), int(block_length)])
    )
    starts = rng.integers(0, n_periods, size=(repetitions, blocks_needed))
    offsets = np.arange(effective, dtype=np.int64)
    sampled = ((starts[..., None] + offsets) % n_periods).reshape(
        repetitions, -1
    )
    return sampled[:, :n_periods]


def _basic_temporal_interval(
    values: np.ndarray,
    *,
    block_length: int,
    config: SecondaryDiagnosticsConfig,
) -> dict[str, Any]:
    finite = np.isfinite(values)
    active_sessions = int(finite.sum())
    point = float(np.mean(values[finite])) if active_sessions else None
    base = {
        "point_delta_z": point,
        "block_length_requested": block_length,
        "block_length_effective": min(block_length, len(values)),
        "primary_block_length": block_length == config.primary_block_length,
        "bootstrap_repetitions": config.bootstrap_repetitions,
        "valid_bootstrap_repetitions": 0,
        "valid_bootstrap_rate": 0.0,
        "centered_error_q10": None,
        "centered_error_q90": None,
        "one_sided_90_lcb": None,
        "one_sided_90_ucb": None,
        "interval_status": "insufficient_active_support",
    }
    if active_sessions < 2:
        return base
    centered = values.copy()
    centered[finite] -= float(point)
    indices = circular_moving_block_indices(
        len(values),
        block_length=block_length,
        repetitions=config.bootstrap_repetitions,
        seed=config.bootstrap_seed,
    )
    sampled = centered[indices]
    sampled_finite = np.isfinite(sampled)
    counts = sampled_finite.sum(axis=1)
    sums = np.where(sampled_finite, sampled, 0.0).sum(axis=1)
    errors = np.divide(
        sums,
        counts,
        out=np.full(config.bootstrap_repetitions, np.nan, dtype=float),
        where=counts > 0,
    )
    valid_errors = errors[np.isfinite(errors)]
    valid_count = len(valid_errors)
    valid_rate = valid_count / config.bootstrap_repetitions
    base["valid_bootstrap_repetitions"] = valid_count
    base["valid_bootstrap_rate"] = valid_rate
    if valid_count < 100 or valid_rate < 0.90:
        base["interval_status"] = "insufficient_bootstrap_support"
        return base
    q10 = float(np.quantile(valid_errors, 0.10))
    q90 = float(np.quantile(valid_errors, 0.90))
    base.update(
        {
            "centered_error_q10": q10,
            "centered_error_q90": q90,
            "one_sided_90_lcb": float(point) - q90,
            "one_sided_90_ucb": float(point) - q10,
            "interval_status": "ok",
        }
    )
    return base


def _validate_temporal_reference(
    reference: pl.DataFrame,
    expected_blocks: Sequence[Mapping[str, Any]],
) -> None:
    required = (
        "target_kind",
        "target_id",
        "block_id",
        "block_start",
        "block_end",
        "calendar_periods",
        "active_periods",
        "equal_period_value",
    )
    _require_columns(reference, required, "evaluator temporal blocks")
    anchor = reference.filter(
        (pl.col("target_kind") == "cell_effect")
        & (pl.col("target_id") == FULL_CELL_ID)
    ).sort("block_id")
    if len(anchor) != 3:
        raise SecondaryDiagnosticsContractError(
            "evaluator temporal blocks lacks the three anchor blocks"
        )
    by_id = {str(row["block_id"]): row for row in anchor.iter_rows(named=True)}
    for expected in expected_blocks:
        block_id = str(expected["block_id"])
        actual = by_id.get(block_id)
        if actual is None:
            raise SecondaryDiagnosticsContractError(
                f"evaluator temporal blocks is missing {block_id}"
            )
        for column in (
            "block_start",
            "block_end",
            "calendar_periods",
            "active_periods",
        ):
            if actual[column] != expected[column]:
                raise SecondaryDiagnosticsContractError(
                    f"evaluator temporal {block_id} {column} drifted"
                )
        if not _is_close(actual["equal_period_value"], expected["point_delta_z"]):
            raise SecondaryDiagnosticsContractError(
                f"evaluator temporal {block_id} point estimate drifted"
            )


def build_temporal_basic_intervals(
    daily: pl.DataFrame,
    evaluator_temporal_blocks: pl.DataFrame,
    *,
    config: SecondaryDiagnosticsConfig,
) -> pl.DataFrame:
    """Build three-block circular basic intervals for the fixed DGB anchor."""

    anchor = daily.filter(pl.col("cell_id") == config.anchor_cell_id).sort("Date")
    dates = anchor["Date"].to_list()
    if len(dates) < config.temporal_block_count:
        raise SecondaryDiagnosticsContractError(
            "requested calendar is too short for three temporal blocks"
        )
    date_blocks = np.array_split(
        np.asarray(dates, dtype=object), config.temporal_block_count
    )
    block_specs: list[dict[str, Any]] = []
    block_values: list[np.ndarray] = []
    for block_index, block_dates_array in enumerate(date_blocks, start=1):
        block_dates = [str(value) for value in block_dates_array.tolist()]
        subset = anchor.filter(pl.col("Date").is_in(block_dates)).sort("Date")
        values = np.asarray(
            [
                float(value) if value is not None else math.nan
                for value in subset["daily_delta_z"].to_list()
            ],
            dtype=float,
        )
        finite = np.isfinite(values)
        block_specs.append(
            {
                "block_id": f"T{block_index:02d}",
                "block_start": block_dates[0],
                "block_end": block_dates[-1],
                "calendar_periods": len(block_dates),
                "active_periods": int(finite.sum()),
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
                "point_delta_z": (
                    float(np.mean(values[finite])) if finite.any() else None
                ),
            }
        )
        block_values.append(values)
    _validate_temporal_reference(evaluator_temporal_blocks, block_specs)

    rows: list[dict[str, Any]] = []
    for spec, values in zip(block_specs, block_values, strict=True):
        for block_length in config.block_lengths:
            rows.append(
                {
                    "target_id": config.anchor_cell_id,
                    **spec,
                    **_basic_temporal_interval(
                        values,
                        block_length=block_length,
                        config=config,
                    ),
                    "confidence_level": config.confidence_level,
                    "bootstrap_seed": config.bootstrap_seed,
                    "method": "circular_moving_block_basic_centered_error",
                    "inactive_effect_policy": "undefined_not_zero",
                    "inference_role": "G3_secondary_temporal_diagnostic",
                }
            )
    return pl.DataFrame(rows, infer_schema_length=None).sort(
        "block_id", "block_length_requested"
    )


def _pipeline_identity(runner: pl.DataFrame, dates: set[str]) -> dict[str, Any]:
    focus = runner.filter(pl.col("Date").is_in(dates))
    output: dict[str, Any] = {}
    for column in (
        "label_profile_id",
        "label_profile_hash",
        "pool_id",
        "pool_version",
    ):
        values = focus[column].unique().to_list()
        if len(values) != 1:
            raise SecondaryDiagnosticsContractError(
                f"runner {column} changes within retrospective evaluation"
            )
        output[column] = values[0]
    return output


def _schema(frame: pl.DataFrame) -> dict[str, str]:
    return {column: str(dtype) for column, dtype in frame.schema.items()}


def evaluate_secondary_diagnostics(
    *,
    runner_daily_stats: pl.DataFrame,
    requested_calendar: pl.DataFrame,
    evaluator_daily_effects: pl.DataFrame,
    evaluator_temporal_blocks: pl.DataFrame,
    evaluator_manifest: Mapping[str, Any],
    config: SecondaryDiagnosticsConfig | None = None,
    input_lineage: Mapping[str, Any] | None = None,
) -> SecondaryDiagnosticsResult:
    """Complete all amendment-2 diagnostics without making a decision claim."""

    active_config = config or SecondaryDiagnosticsConfig()
    validate_evaluator_manifest(evaluator_manifest)
    calendar = prepare_requested_calendar(requested_calendar)
    evaluator_daily = prepare_evaluator_daily(evaluator_daily_effects, calendar)
    runner = prepare_runner_daily(runner_daily_stats)
    daily = _validate_runner_evaluator_match(runner, evaluator_daily, calendar)
    equal_active = summarize_equal_active_bp(daily)
    monthly_effects, monthly_summary = build_monthly_outputs(daily)
    hac = build_hac_diagnostics(daily, config=active_config)
    temporal = build_temporal_basic_intervals(
        daily,
        evaluator_temporal_blocks,
        config=active_config,
    )
    expected_dates = set(
        calendar.filter(pl.col("calendar_status") == "evaluation_expected")[
            "Date"
        ].to_list()
    )
    frames = {
        "secondary_daily_effects.parquet": daily,
        "equal_active_bp.parquet": equal_active,
        "monthly_effects.parquet": monthly_effects,
        "monthly_summary.parquet": monthly_summary,
        "hac_diagnostics.parquet": hac,
        "temporal_basic_intervals.parquet": temporal,
    }
    manifest = {
        "contract": SECONDARY_CONTRACT_ID,
        "parent_contract": PARENT_CONTRACT_ID,
        "amendment": "fh_neg_rejection_robustness/fixed_factorial_v1/amendment_2",
        "family_id": FAMILY_ID,
        "family_version": FAMILY_VERSION,
        "anchor_cell_id": FULL_CELL_ID,
        "evidence_class": RETROSPECTIVE_EVIDENCE_CLASS,
        "config": asdict(active_config),
        "pipeline_identity": _pipeline_identity(runner, expected_dates),
        "effect_policy": {
            "raw_bp": (
                "condition_sum_raw_bp/condition_n_z - "
                "control_sum_raw_bp/control_n_z"
            ),
            "residual_bp": (
                "condition_sum_residual_bp/condition_n_z - "
                "control_sum_residual_bp/control_n_z"
            ),
            "period_weighting": "equal finite active session",
            "inactive_effect": "undefined_not_zero",
        },
        "monthly_policy": (
            "mean daily_delta_z within YYYYMM over active sessions before "
            "counting positive active months"
        ),
        "hac": {
            "kernel": "Bartlett Newey-West",
            "lags": list(active_config.hac_lags),
            "series": (
                "finite active-session daily_delta_z in requested-session order"
            ),
            "role": "secondary_diagnostic_only",
        },
        "temporal_interval": {
            "partition": "three chronologically contiguous requested-session blocks",
            "method": "circular_moving_block_basic_centered_error",
            "block_lengths": list(active_config.block_lengths),
            "bootstrap_repetitions": active_config.bootstrap_repetitions,
            "bootstrap_seed": active_config.bootstrap_seed,
            "bounds": "LCB=theta-q90(centered_error); UCB=theta-q10(centered_error)",
            "minimum_valid_repetitions": 100,
            "minimum_valid_rate": 0.90,
        },
        "decision_scope": {
            "secondary_diagnostics_only": True,
            "condition_selection_performed": False,
            "deployment_decision_created": False,
        },
        "input_lineage": dict(input_lineage or {}),
        "output_rows": {name: len(frame) for name, frame in frames.items()},
        "output_schemas": {name: _schema(frame) for name, frame in frames.items()},
    }
    return SecondaryDiagnosticsResult(
        daily_effects=daily,
        equal_active_bp=equal_active,
        monthly_effects=monthly_effects,
        monthly_summary=monthly_summary,
        hac_diagnostics=hac,
        temporal_basic_intervals=temporal,
        manifest=manifest,
    )


def _atomic_write_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
    try:
        frame.write_parquet(temp_path, compression="zstd")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
        encoding="utf-8",
    ) as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def write_result(result: SecondaryDiagnosticsResult, output_dir: Path) -> None:
    """Write the fixed versioned artifact set without an inferred decision."""

    frames = {
        "secondary_daily_effects.parquet": result.daily_effects,
        "equal_active_bp.parquet": result.equal_active_bp,
        "monthly_effects.parquet": result.monthly_effects,
        "monthly_summary.parquet": result.monthly_summary,
        "hac_diagnostics.parquet": result.hac_diagnostics,
        "temporal_basic_intervals.parquet": result.temporal_basic_intervals,
    }
    for name, frame in frames.items():
        _atomic_write_parquet(frame, output_dir / name)
    _atomic_write_json(result.manifest, output_dir / "run_manifest.json")


def write_artifact_oos_result(
    result: ArtifactOosDescriptiveResult,
    output_dir: Path,
) -> None:
    """Write the one-session description to its own inference-free directory."""

    _atomic_write_parquet(
        result.cell_descriptive, output_dir / "cell_descriptive.parquet"
    )
    _atomic_write_parquet(
        result.loo_descriptive, output_dir / "loo_descriptive.parquet"
    )
    _atomic_write_json(result.manifest, output_dir / "run_manifest.json")


def _file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def resolve_output_dir(value: Path | str) -> Path:
    """Keep CLI outputs under the study-local artifact root."""

    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()
    artifact_root = ARTIFACT_ROOT.resolve()
    try:
        relative = resolved.relative_to(artifact_root)
    except ValueError as exc:
        raise SecondaryDiagnosticsContractError(
            f"output must be under {artifact_root}"
        ) from exc
    if not relative.parts:
        raise SecondaryDiagnosticsContractError(
            "output cannot overwrite the study artifact root"
        )
    return resolved


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build frozen amendment-2 fh_neg_rejection diagnostics"
    )
    parser.add_argument("--runner-daily-stats", type=Path, required=True)
    parser.add_argument("--evaluator-output-dir", type=Path, required=True)
    parser.add_argument(
        "--artifact-oos-calendar",
        type=Path,
        default=DEFAULT_ARTIFACT_OOS_CALENDAR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = resolve_output_dir(args.output_dir)
    evaluator_dir = args.evaluator_output_dir
    paths = {
        "runner_daily_stats": args.runner_daily_stats,
        "evaluator_requested_calendar": evaluator_dir
        / "requested_calendar.parquet",
        "evaluator_daily_effects": evaluator_dir / "daily_effects.parquet",
        "evaluator_temporal_blocks": evaluator_dir / "temporal_blocks.parquet",
        "evaluator_manifest": evaluator_dir / "run_manifest.json",
        "amendment_2": AMENDMENT_PATH,
    }
    artifact_oos_path = args.artifact_oos_calendar
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if not artifact_oos_path.is_file():
        missing.append(str(artifact_oos_path))
    if missing:
        raise SystemExit(f"missing required input artifacts: {missing}")
    evaluator_manifest = json.loads(
        paths["evaluator_manifest"].read_text(encoding="utf-8")
    )
    result = evaluate_secondary_diagnostics(
        runner_daily_stats=pl.read_parquet(paths["runner_daily_stats"]),
        requested_calendar=pl.read_parquet(
            paths["evaluator_requested_calendar"]
        ),
        evaluator_daily_effects=pl.read_parquet(
            paths["evaluator_daily_effects"]
        ),
        evaluator_temporal_blocks=pl.read_parquet(
            paths["evaluator_temporal_blocks"]
        ),
        evaluator_manifest=evaluator_manifest,
        input_lineage={
            name: _file_identity(path) for name, path in paths.items()
        },
    )
    write_result(result, output_dir)
    artifact_oos_result = build_artifact_oos_descriptive(
        runner_daily_stats=pl.read_parquet(paths["runner_daily_stats"]),
        artifact_oos_calendar=pl.read_parquet(artifact_oos_path),
        input_lineage={
            "runner_daily_stats": _file_identity(
                paths["runner_daily_stats"]
            ),
            "artifact_oos_calendar": _file_identity(artifact_oos_path),
            "amendment_2": _file_identity(AMENDMENT_PATH),
        },
    )
    artifact_oos_output_dir = output_dir / ARTIFACT_OOS_OUTPUT_NAME
    write_artifact_oos_result(
        artifact_oos_result,
        artifact_oos_output_dir,
    )
    print(
        json.dumps(
            {
                "retrospective_secondary": result.manifest,
                "artifact_oos_descriptive": artifact_oos_result.manifest,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
