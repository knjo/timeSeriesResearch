"""Evaluate the frozen 27-cell ``fh_neg_rejection`` threshold plateau.

The evaluator consumes only a completed daily aggregate panel, the completed
runner manifest, and one external retrospective requested-session calendar.
It never ranks or replaces the frozen center.  All 27 trials stay in a
separate plateau ledger and a synchronous circular max-stat family.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import polars as pl

from evaluate_robustness import (
    _effective_trial_count,
    _point_means,
    _sample_means,
    moving_block_indices,
)


STUDY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STUDY_DIR.parents[4]
CONTRACT_PATH = STUDY_DIR / "threshold_plateau_contract.json"
PARENT_CONTRACT_PATH = STUDY_DIR / "study_contract.json"
CONTRACT_BYTES = CONTRACT_PATH.read_bytes()
CONTRACT = json.loads(CONTRACT_BYTES)
CONTRACT_SHA256 = hashlib.sha256(CONTRACT_BYTES).hexdigest()
PARENT_CONTRACT_SHA256 = hashlib.sha256(PARENT_CONTRACT_PATH.read_bytes()).hexdigest()

CONTRACT_ID = "fh_neg_rejection_threshold_plateau/fixed_grid_v1"
EVALUATOR_CONTRACT = "fh_neg_rejection_threshold_plateau_evaluator/v1"
DECISION_CONTRACT = "fh_neg_rejection_threshold_plateau_decision/v1"
LEDGER_CONTRACT = "fh_neg_rejection_threshold_plateau_trial_ledger/v1"
FAMILY_ID = "fh_neg_rejection_threshold_plateau"
FAMILY_VERSION = "1"
PARENT_CONTRACT_ID = "fh_neg_rejection_robustness/fixed_factorial_v1"
CENTER_COORDINATES = (-8.0, 4.0, 2)
START_DATE, END_DATE = tuple(str(value) for value in CONTRACT["date_range"])
DEPTH_LEVELS = tuple(float(value) for value in CONTRACT["fixed_grid"]["depth_ratio_lte"])
GAP_LEVELS = tuple(float(value) for value in CONTRACT["fixed_grid"]["gap_ticks_gte"])
BEAR_LEVELS = tuple(int(value) for value in CONTRACT["fixed_grid"]["bear_score_gte"])
BLOCK_LENGTHS = tuple(int(value) for value in CONTRACT["inference"]["circular_block_lengths"])
BOOTSTRAP_SEED = int(CONTRACT["inference"]["bootstrap_seed"])
BOOTSTRAP_REPETITIONS = int(CONTRACT["inference"]["bootstrap_repetitions"])
FAMILY_ALPHA = float(CONTRACT["inference"]["alpha"])
MIN_JOINT_VALID_RATE = float(CONTRACT["inference"]["minimum_joint_valid_rate"])
MIN_JOINT_VALID_REPETITIONS = int(
    CONTRACT["inference"]["minimum_joint_valid_repetitions"]
)
MIN_ACTIVE_SESSIONS = int(
    CONTRACT["plateau_diagnostics"]["support_floor_each_cell"]["active_sessions"]
)
MIN_EVENTS = int(CONTRACT["plateau_diagnostics"]["support_floor_each_cell"]["events"])

INPUT_HASH_KEYS = (
    "daily_aggregates",
    "plateau_run_manifest",
    "requested_calendar",
    "requested_calendar_manifest",
    "threshold_contract",
)


class PlateauContractError(ValueError):
    """Raised when an input cannot represent the frozen plateau contract."""


@dataclass(frozen=True)
class ThresholdCell:
    candidate_id: str
    depth_ratio_lte: float
    gap_ticks_gte: float
    bear_score_gte: int

    def metadata(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "depth_ratio_lte": self.depth_ratio_lte,
            "gap_ticks_gte": self.gap_ticks_gte,
            "bear_score_gte": self.bear_score_gte,
        }


def candidate_id(depth: float, gap: float, bear: int) -> str:
    return f"d{int(abs(depth)):02d}_g{int(gap):02d}_b{int(bear)}"


CELLS = tuple(
    ThresholdCell(candidate_id(depth, gap, bear), depth, gap, bear)
    for depth, gap, bear in product(DEPTH_LEVELS, GAP_LEVELS, BEAR_LEVELS)
)
CELL_IDS = tuple(cell.candidate_id for cell in CELLS)
CELL_BY_ID = {cell.candidate_id: cell for cell in CELLS}
CENTER_CELL_ID = candidate_id(*CENTER_COORDINATES)

EXPECTED_DESIGN: dict[str, object] = {
    "contract": CONTRACT_ID,
    "frozen_contract_sha256": CONTRACT_SHA256,
    "parent_contract": PARENT_CONTRACT_ID,
    "parent_contract_sha256": PARENT_CONTRACT_SHA256,
    "membership_semantics": (
        "conservative_base AND NOT depth_reference_invalid; null gates fail "
        "closed; center keys must equal parent conservative_dgb"
    ),
    "center_cell_id": CENTER_CELL_ID,
    "cells": [cell.metadata() for cell in CELLS],
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )


EXPECTED_DESIGN_FINGERPRINT = hashlib.sha256(
    _canonical_json(EXPECTED_DESIGN).encode("utf-8")
).hexdigest()


@dataclass(frozen=True)
class PlateauResult:
    requested_calendar: pl.DataFrame
    daily_effects: pl.DataFrame
    cell_summary: pl.DataFrame
    year_summary: pl.DataFrame
    fwer_audit: pl.DataFrame
    neighbor_diagnostics: pl.DataFrame
    monotonicity: pl.DataFrame
    slice_reversals: pl.DataFrame
    trial_ledger: pl.DataFrame
    decision: dict[str, Any]
    manifest: dict[str, Any]


def _validate_frozen_contract() -> None:
    expected = {
        "contract_id": CONTRACT_ID,
        "parent_contract_id": PARENT_CONTRACT_ID,
        "evidence_class": "retrospective_diagnostic",
        "date_range": ["20250211", "20260825"],
        "production_promotion_authorized": False,
    }
    mismatches = {
        name: {"expected": value, "actual": CONTRACT.get(name)}
        for name, value in expected.items()
        if CONTRACT.get(name) != value
    }
    if mismatches:
        raise RuntimeError(f"frozen plateau contract identity drifted: {mismatches}")
    if (DEPTH_LEVELS, GAP_LEVELS, BEAR_LEVELS) != (
        (-6.0, -8.0, -10.0),
        (3.0, 4.0, 5.0),
        (1, 2, 3),
    ):
        raise RuntimeError("frozen plateau grid drifted")
    if len(CELLS) != 27 or len(set(CELL_IDS)) != 27:
        raise RuntimeError("frozen plateau does not contain 27 unique cells")
    if CENTER_CELL_ID != "d08_g04_b2":
        raise RuntimeError("frozen plateau center drifted")
    if BLOCK_LENGTHS != (10, 5, 20):
        raise RuntimeError("frozen circular block order drifted")
    if (
        BOOTSTRAP_SEED != 20_260_827
        or BOOTSTRAP_REPETITIONS != 5_000
        or FAMILY_ALPHA != 0.10
    ):
        raise RuntimeError("frozen plateau inference settings drifted")


_validate_frozen_contract()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256_file(path),
    }


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PlateauContractError(f"{path}: JSON root must be an object")
    return payload


def _require_columns(frame: pl.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise PlateauContractError(f"{source}: missing columns {missing}")


def _has_null(frame: pl.DataFrame, columns: Sequence[str]) -> bool:
    return bool(
        frame.select(
            pl.any_horizontal([pl.col(column).is_null() for column in columns]).any()
        ).item()
    )


def _require_equal(actual: object, expected: object, context: str) -> None:
    if actual != expected:
        raise PlateauContractError(
            f"{context}: expected {expected!r}, got {actual!r}"
        )


def _validate_hash_bindings(input_hashes: Mapping[str, str]) -> dict[str, str]:
    if set(input_hashes) != set(INPUT_HASH_KEYS):
        raise PlateauContractError(
            f"input_artifact_sha256 requires exactly {list(INPUT_HASH_KEYS)}"
        )
    output = {name: str(input_hashes[name]) for name in INPUT_HASH_KEYS}
    for name, value in output.items():
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise PlateauContractError(f"invalid SHA-256 binding for {name}")
    if output["threshold_contract"] != CONTRACT_SHA256:
        raise PlateauContractError("threshold_contract hash does not match frozen bytes")
    return output


def validate_runner_manifest(manifest: Mapping[str, object]) -> tuple[list[str], dict[str, object]]:
    expected = {
        "contract": CONTRACT_ID,
        "family_id": FAMILY_ID,
        "family_version": FAMILY_VERSION,
        "candidate_ids": list(CELL_IDS),
        "center_candidate_id": CENTER_CELL_ID,
        "frozen_contract_sha256": CONTRACT_SHA256,
        "threshold_design_fingerprint": EXPECTED_DESIGN_FINGERPRINT,
        "threshold_design": EXPECTED_DESIGN,
        "start": START_DATE,
        "end": END_DATE,
        "step": 1,
        "label_profile_id": CONTRACT["shared_semantics"]["label"],
        "pool_id": CONTRACT["shared_semantics"]["eligible_pool"],
        "pool_version": "1",
        "primary_estimand": "shared_pool",
        "control_sample_size": None,
        "evidence_class": "retrospective_diagnostic",
        "production_promotion_authorized": False,
        "selection_rule": "fixed 27-cell ledger; no ranking or center replacement",
    }
    for name, value in expected.items():
        _require_equal(manifest.get(name), value, f"plateau run manifest {name}")
    if hashlib.sha256(
        _canonical_json(manifest["threshold_design"]).encode("utf-8")
    ).hexdigest() != manifest["threshold_design_fingerprint"]:
        raise PlateauContractError("plateau threshold_design fingerprint is invalid")
    definitions = manifest.get("candidate_definitions")
    if not isinstance(definitions, dict) or set(definitions) != set(CELL_IDS):
        raise PlateauContractError("plateau candidate definitions are incomplete")
    sessions = manifest.get("sessions")
    if not isinstance(sessions, list) or not all(isinstance(value, str) for value in sessions):
        raise PlateauContractError("plateau run manifest sessions must be strings")
    if sessions != sorted(set(sessions)):
        raise PlateauContractError("plateau run manifest sessions must be sorted and unique")
    failures = manifest.get("failures")
    if not isinstance(failures, list):
        raise PlateauContractError("plateau run manifest failures must be a list")
    if failures:
        raise PlateauContractError(
            "completed plateau evaluation requires zero recorded run failures"
        )
    source_identities = manifest.get("source_identities")
    if not isinstance(source_identities, dict) or set(source_identities) != set(sessions):
        raise PlateauContractError("plateau source identity calendar is incomplete")
    for date, identities in source_identities.items():
        if not isinstance(identities, dict) or set(identities) != {
            "tick_data", "tick_feature", "pre_market", "txf_tick"
        }:
            raise PlateauContractError(f"{date}: plateau raw source identity family incomplete")
    runner_artifact = manifest.get("runner_artifact")
    if (
        not isinstance(runner_artifact, dict)
        or not isinstance(runner_artifact.get("sha256"), str)
        or len(str(runner_artifact["sha256"])) != 64
    ):
        raise PlateauContractError("plateau runner_artifact identity is incomplete")
    return list(sessions), dict(runner_artifact)


def prepare_requested_calendar(
    frame: pl.DataFrame,
    calendar_manifest: Mapping[str, object],
    *,
    run_sessions: Sequence[str],
) -> pl.DataFrame:
    required = (
        "Date",
        "evidence_class",
        "source_complete",
        "run_status",
        "session_status",
        "eligible_pool_rows",
        "control_n_z",
        "aggregate_complete",
        "membership_complete",
    )
    _require_columns(frame, required, "requested calendar")
    calendar = frame.select(required).with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("evidence_class").cast(pl.String),
        pl.col("run_status").cast(pl.String),
        pl.col("session_status").cast(pl.String),
        pl.col("source_complete").cast(pl.Boolean, strict=False),
        pl.col("aggregate_complete").cast(pl.Boolean, strict=False),
        pl.col("membership_complete").cast(pl.Boolean, strict=False),
        pl.col("eligible_pool_rows").cast(pl.Int64, strict=False),
        pl.col("control_n_z").cast(pl.Int64, strict=False),
    )
    if calendar.is_empty() or _has_null(
        calendar,
        (
            "Date", "evidence_class", "source_complete", "run_status",
            "session_status", "aggregate_complete", "membership_complete",
        ),
    ):
        raise PlateauContractError("requested calendar is empty or has null identity fields")
    if calendar["Date"].n_unique() != len(calendar):
        raise PlateauContractError("requested calendar has duplicate Date")
    calendar = calendar.sort("Date")
    dates = calendar["Date"].to_list()
    _require_equal(dates[0], START_DATE, "requested calendar start")
    _require_equal(dates[-1], END_DATE, "requested calendar end")
    _require_equal(
        set(calendar["evidence_class"].unique().to_list()),
        {"retrospective_diagnostic"},
        "requested calendar evidence class",
    )
    if calendar.filter(
        (pl.col("source_complete") & (pl.col("run_status") != "success"))
        | (pl.col("session_status") == "not_run_or_incomplete")
    ).height:
        raise PlateauContractError("requested calendar contains failed or incomplete source-complete session")
    successful = calendar.filter(pl.col("source_complete"))
    if successful.filter(~pl.col("aggregate_complete") | ~pl.col("membership_complete")).height:
        raise PlateauContractError("requested calendar completion flags are inconsistent")
    _require_equal(successful["Date"].to_list(), list(run_sessions), "calendar vs plateau sessions")
    expected_calendar_manifest = {
        "contract": "fh_neg_rejection_requested_calendar/v1",
        "start": START_DATE,
        "end": END_DATE,
        "evidence_class": "retrospective_diagnostic",
        "requested_sessions": len(calendar),
        "source_complete_sessions": len(successful),
    }
    for name, value in expected_calendar_manifest.items():
        _require_equal(calendar_manifest.get(name), value, f"calendar manifest {name}")
    status_counts = {
        str(row["session_status"]): int(row["len"])
        for row in calendar.group_by("session_status").len().iter_rows(named=True)
    }
    if "status_counts" in calendar_manifest:
        _require_equal(calendar_manifest["status_counts"], status_counts, "calendar status counts")
    return calendar


REQUIRED_DAILY_COLUMNS = (
    "contract_id",
    "Date",
    "family_id",
    "family_version",
    "candidate_id",
    "label_profile_id",
    "label_profile_hash",
    "pool_id",
    "pool_version",
    "control_mode",
    "control_sample_size",
    "primary_estimand",
    "eligible_pool_rows",
    "indexed_condition_rows",
    "condition_n",
    "condition_n_z",
    "condition_sum_z",
    "control_n_z",
    "control_sum_z",
    "selection_rate",
    "plateau_contract_sha256",
    "plateau_design_fingerprint",
    "plateau_runner_sha256",
    "parent_run_manifest_sha256",
    "parent_membership_sha256",
    "parent_membership_lineage_sha256",
    "parent_source_identities_sha256",
)


def _validate_count_and_sum(
    frame: pl.DataFrame, count: str, total: str, source: str
) -> None:
    if frame.filter(pl.col(count) < 0).height:
        raise PlateauContractError(f"{source}: negative {count}")
    if frame.filter(
        (pl.col(count) > 0) & (~pl.col(total).is_finite() | pl.col(total).is_null())
    ).height:
        raise PlateauContractError(f"{source}: positive support requires finite sum")
    if frame.filter(
        (pl.col(count) == 0)
        & pl.col(total).is_not_null()
        & ((pl.col(total).abs() > 1e-12) | ~pl.col(total).is_finite())
    ).height:
        raise PlateauContractError(f"{source}: zero support has nonzero/invalid sum")


def _is_stricter(strict: ThresholdCell, loose: ThresholdCell) -> bool:
    return (
        strict.depth_ratio_lte <= loose.depth_ratio_lte
        and strict.gap_ticks_gte >= loose.gap_ticks_gte
        and strict.bear_score_gte >= loose.bear_score_gte
    )


def prepare_daily_effects(
    frame: pl.DataFrame,
    *,
    calendar: pl.DataFrame,
    runner_manifest: Mapping[str, object],
    runner_artifact: Mapping[str, object],
) -> pl.DataFrame:
    _require_columns(frame, REQUIRED_DAILY_COLUMNS, "plateau daily aggregates")
    selected = frame.select(REQUIRED_DAILY_COLUMNS).with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("candidate_id").cast(pl.String),
        *(
            pl.col(column).cast(pl.Int64, strict=False)
            for column in (
                "eligible_pool_rows", "indexed_condition_rows", "condition_n",
                "condition_n_z", "control_n_z", "control_sample_size",
            )
        ),
        pl.col("condition_sum_z").cast(pl.Float64, strict=False),
        pl.col("control_sum_z").cast(pl.Float64, strict=False),
        pl.col("selection_rate").cast(pl.Float64, strict=False),
    )
    identity_columns = (
        "contract_id", "Date", "family_id", "family_version", "candidate_id",
        "label_profile_id", "label_profile_hash", "pool_id", "pool_version",
        "control_mode", "control_sample_size", "primary_estimand",
        "plateau_contract_sha256", "plateau_design_fingerprint",
        "plateau_runner_sha256", "parent_run_manifest_sha256",
        "parent_membership_sha256", "parent_membership_lineage_sha256",
        "parent_source_identities_sha256",
    )
    if _has_null(selected, identity_columns):
        raise PlateauContractError("plateau daily aggregates contain null identity fields")
    expected_uniform = {
        "contract_id": CONTRACT_ID,
        "family_id": FAMILY_ID,
        "family_version": FAMILY_VERSION,
        "label_profile_id": CONTRACT["shared_semantics"]["label"],
        "pool_id": CONTRACT["shared_semantics"]["eligible_pool"],
        "pool_version": "1",
        "control_mode": "full",
        "control_sample_size": 0,
        "primary_estimand": "shared_pool",
        "plateau_contract_sha256": CONTRACT_SHA256,
        "plateau_design_fingerprint": EXPECTED_DESIGN_FINGERPRINT,
        "plateau_runner_sha256": runner_artifact["sha256"],
        "parent_run_manifest_sha256": runner_manifest["parent_run_manifest"]["sha256"],
    }
    for column, expected in expected_uniform.items():
        values = selected[column].unique().to_list()
        _require_equal(values, [expected], f"daily aggregate {column}")
    for column in (
        "label_profile_hash", "parent_membership_sha256",
        "parent_membership_lineage_sha256", "parent_source_identities_sha256",
    ):
        if selected.filter(pl.col(column).str.len_chars() != 64).height:
            raise PlateauContractError(f"daily aggregate {column} is not SHA-256")
    unknown_ids = sorted(set(selected["candidate_id"].unique()) - set(CELL_IDS))
    if unknown_ids:
        raise PlateauContractError(f"daily aggregates contain unknown cells {unknown_ids}")
    if selected.select(pl.struct(["Date", "candidate_id"]).is_duplicated().any()).item():
        raise PlateauContractError("daily aggregates contain duplicate date/cell rows")
    actual_dates = selected["Date"].unique().sort().to_list()
    expected_dates = calendar.filter(pl.col("source_complete"))["Date"].to_list()
    _require_equal(actual_dates, expected_dates, "daily aggregate date calendar")
    for row in selected.group_by("Date").agg(
        pl.col("candidate_id").alias("cells"), pl.len().alias("rows")
    ).iter_rows(named=True):
        if int(row["rows"]) != 27 or set(row["cells"]) != set(CELL_IDS):
            raise PlateauContractError(f"{row['Date']}: incomplete 27-cell panel")
    if selected.filter(
        (pl.col("indexed_condition_rows") < pl.col("condition_n"))
        | (pl.col("condition_n") < pl.col("condition_n_z"))
        | (pl.col("condition_n_z") > pl.col("control_n_z"))
        | (pl.col("condition_n") > pl.col("eligible_pool_rows"))
    ).height:
        raise PlateauContractError("daily condition/control support hierarchy failed")
    _validate_count_and_sum(selected, "condition_n_z", "condition_sum_z", "condition")
    _validate_count_and_sum(selected, "control_n_z", "control_sum_z", "control")
    shared_columns = ("eligible_pool_rows", "control_n_z", "control_sum_z")
    if selected.group_by("Date").agg(
        *[pl.col(column).n_unique().alias(column) for column in shared_columns]
    ).filter(pl.any_horizontal([pl.col(column) != 1 for column in shared_columns])).height:
        raise PlateauContractError("shared control differs across cells within a date")
    calendar_shared = calendar.select(
        "Date",
        pl.col("eligible_pool_rows").alias("calendar_pool_rows"),
        pl.col("control_n_z").alias("calendar_control_n_z"),
    )
    checked = selected.join(calendar_shared, on="Date", how="left", validate="m:1")
    if checked.filter(
        (pl.col("eligible_pool_rows") != pl.col("calendar_pool_rows"))
        | (pl.col("control_n_z") != pl.col("calendar_control_n_z"))
    ).height:
        raise PlateauContractError("daily aggregate support disagrees with requested calendar")
    provided_rate = pl.when(pl.col("eligible_pool_rows") > 0).then(
        pl.col("condition_n") / pl.col("eligible_pool_rows")
    )
    if selected.filter(
        pl.col("selection_rate").is_not_null()
        & ((pl.col("selection_rate") - provided_rate).abs() > 1e-12)
    ).height:
        raise PlateauContractError("daily aggregate selection_rate is inconsistent")
    support_by_date = {
        str(date): day.select(
            "candidate_id", "indexed_condition_rows", "condition_n", "condition_n_z"
        )
        for date, day in selected.partition_by("Date", as_dict=True, maintain_order=True).items()
    }
    for date, day in support_by_date.items():
        supports = {
            str(row["candidate_id"]): row
            for row in day.iter_rows(named=True)
        }
        for strict in CELLS:
            for loose in CELLS:
                if not _is_stricter(strict, loose):
                    continue
                for column in ("indexed_condition_rows", "condition_n", "condition_n_z"):
                    if int(supports[strict.candidate_id][column]) > int(
                        supports[loose.candidate_id][column]
                    ):
                        raise PlateauContractError(
                            f"{date}: nested support violation {strict.candidate_id} > "
                            f"{loose.candidate_id} for {column}"
                        )
    metadata = pl.DataFrame([cell.metadata() for cell in CELLS])
    output = selected.join(metadata, on="candidate_id", how="left", validate="m:1")
    condition_active = pl.col("condition_n_z") > 0
    control_active = pl.col("control_n_z") > 0
    output = output.with_columns(
        pl.when(condition_active)
        .then(pl.col("condition_sum_z") / pl.col("condition_n_z"))
        .alias("condition_mean_z"),
        pl.when(control_active)
        .then(pl.col("control_sum_z") / pl.col("control_n_z"))
        .alias("control_mean_z"),
        pl.when(pl.col("eligible_pool_rows") > 0)
        .then(pl.col("condition_n") / pl.col("eligible_pool_rows"))
        .alias("realized_selection_rate"),
        condition_active.alias("condition_active"),
        (condition_active & control_active).alias("effect_active"),
    ).with_columns(
        pl.when(pl.col("effect_active"))
        .then(pl.col("condition_mean_z") - pl.col("control_mean_z"))
        .alias("daily_delta_z"),
        pl.when(pl.col("control_n_z") == 0)
        .then(pl.lit("zero_control_ineligible"))
        .when(pl.col("condition_n_z") == 0)
        .then(pl.lit("zero_event"))
        .otherwise(pl.lit("active"))
        .alias("period_status"),
        pl.col("Date").str.slice(0, 4).alias("year"),
        pl.col("Date").str.slice(0, 6).alias("month"),
    )
    return output.sort("Date", "candidate_id")


def _period_filter(frame: pl.DataFrame, period: str) -> pl.DataFrame:
    if period == "full":
        return frame
    if period not in {"2025", "2026"}:
        raise ValueError(f"unknown plateau period {period}")
    return frame.filter(pl.col("year") == period)


def summarize_cells(daily: pl.DataFrame, period: str) -> pl.DataFrame:
    subset = _period_filter(daily, period)
    rows: list[dict[str, object]] = []
    for cell in CELLS:
        cell_frame = subset.filter(pl.col("candidate_id") == cell.candidate_id)
        active = cell_frame.filter(pl.col("effect_active"))
        effects = np.asarray(active["daily_delta_z"].to_list(), dtype=float)
        effects = effects[np.isfinite(effects)]
        rates = cell_frame["realized_selection_rate"].drop_nulls()
        active_months = active["month"].n_unique() if len(active) else 0
        effect_valid_events = int(cell_frame["condition_n_z"].sum() or 0)
        rows.append(
            {
                "period": period,
                **cell.metadata(),
                "calendar_sessions": cell_frame["Date"].n_unique(),
                "active_sessions": len(effects),
                "active_months": int(active_months),
                "zero_event_sessions": int(
                    (cell_frame["period_status"] == "zero_event").sum()
                ),
                "zero_control_ineligible_sessions": int(
                    (cell_frame["period_status"] == "zero_control_ineligible").sum()
                ),
                "indexed_events": int(cell_frame["indexed_condition_rows"].sum() or 0),
                "eligible_condition_events": int(cell_frame["condition_n"].sum() or 0),
                "effect_valid_events": effect_valid_events,
                "events": effect_valid_events,
                "equal_active_session_delta_z": (
                    float(np.mean(effects)) if len(effects) else None
                ),
                "daily_delta_z_std": (
                    float(np.std(effects, ddof=1)) if len(effects) > 1 else None
                ),
                "positive_session_ratio": (
                    float(np.mean(effects > 0.0)) if len(effects) else None
                ),
                "selection_rate_equal_session_mean": (
                    float(rates.mean()) if len(rates) else None
                ),
                "selection_rate_session_median": (
                    float(rates.median()) if len(rates) else None
                ),
                "selection_rate_pooled": (
                    float(cell_frame["condition_n"].sum())
                    / float(cell_frame["eligible_pool_rows"].sum())
                    if float(cell_frame["eligible_pool_rows"].sum() or 0) > 0
                    else None
                ),
                "support_floor_active_sessions": MIN_ACTIVE_SESSIONS,
                "support_floor_events": MIN_EVENTS,
                "support_floor_pass": (
                    len(effects) >= MIN_ACTIVE_SESSIONS
                    and effect_valid_events >= MIN_EVENTS
                ),
                "support_event_definition": "effect_valid_condition_n_z",
            }
        )
    return pl.DataFrame(rows).sort("candidate_id")


def _effect_matrix(daily: pl.DataFrame) -> tuple[list[str], np.ndarray]:
    dates = daily["Date"].unique().sort().to_list()
    matrix = np.full((len(dates), len(CELL_IDS)), np.nan, dtype=float)
    date_index = {str(date): index for index, date in enumerate(dates)}
    cell_index = {cell_id: index for index, cell_id in enumerate(CELL_IDS)}
    for row in daily.select("Date", "candidate_id", "daily_delta_z").iter_rows(named=True):
        value = row["daily_delta_z"]
        if value is not None and math.isfinite(float(value)):
            matrix[date_index[str(row["Date"])], cell_index[str(row["candidate_id"])]] = float(value)
    return [str(date) for date in dates], matrix


def circular_maxstat(daily: pl.DataFrame) -> pl.DataFrame:
    dates, matrix = _effect_matrix(daily)
    point, active_counts = _point_means(matrix)
    centered = matrix - point
    effective_trials, effective_method, common_periods = _effective_trial_count(matrix)
    rows: list[dict[str, object]] = []
    for block_length in BLOCK_LENGTHS:
        indices = moving_block_indices(
            len(dates),
            block_length=block_length,
            repetitions=BOOTSTRAP_REPETITIONS,
            seed=BOOTSTRAP_SEED,
        )
        null_means = _sample_means(centered, indices)
        joint_valid = np.isfinite(null_means).all(axis=1)
        joint_count = int(joint_valid.sum())
        joint_rate = joint_count / BOOTSTRAP_REPETITIONS
        if (
            joint_count < MIN_JOINT_VALID_REPETITIONS
            or joint_rate < MIN_JOINT_VALID_RATE
        ):
            raise PlateauContractError(
                f"27-cell joint-valid bootstrap insufficient at B={block_length}: "
                f"{joint_count}/{BOOTSTRAP_REPETITIONS} ({joint_rate:.6f})"
            )
        joint_null = null_means[joint_valid, :]
        scale = np.std(joint_null, axis=0, ddof=1)
        estimable = np.isfinite(point) & np.isfinite(scale) & (
            scale > np.finfo(float).eps
        )
        if not estimable.all():
            invalid = [cell_id for cell_id, valid in zip(CELL_IDS, estimable, strict=True) if not valid]
            raise PlateauContractError(
                f"27-cell max-stat contains non-estimable trials at B={block_length}: {invalid}"
            )
        null_statistics = joint_null / scale
        observed = point / scale
        maxima = np.max(null_statistics, axis=1)
        critical = float(np.quantile(maxima, 1.0 - FAMILY_ALPHA))
        for index, cell in enumerate(CELLS):
            raw_p = float(
                (1 + np.sum(null_statistics[:, index] >= observed[index]))
                / (joint_count + 1)
            )
            adjusted_p = float(
                (1 + np.sum(maxima >= observed[index])) / (joint_count + 1)
            )
            rows.append(
                {
                    "family_id": FAMILY_ID,
                    "family_version": FAMILY_VERSION,
                    "ledger_contract": LEDGER_CONTRACT,
                    **cell.metadata(),
                    "block_length_requested": block_length,
                    "block_length_effective": min(block_length, len(dates)),
                    "primary_block_length": block_length == BLOCK_LENGTHS[0],
                    "calendar_sessions": len(dates),
                    "active_sessions": int(active_counts[index]),
                    "point_estimate": float(point[index]),
                    "bootstrap_se": float(scale[index]),
                    "max_t_statistic": float(observed[index]),
                    "max_t_critical_value": critical,
                    "simultaneous_lower_bound": float(
                        point[index] - critical * scale[index]
                    ),
                    "raw_block_bootstrap_p": raw_p,
                    "single_step_fwer_p": adjusted_p,
                    "alternative": "positive",
                    "family_alpha": FAMILY_ALPHA,
                    "simultaneous_confidence_level": 1.0 - FAMILY_ALPHA,
                    "raw_trial_count": 27,
                    "dependence_effective_trial_count": effective_trials,
                    "effective_trial_method": effective_method,
                    "effective_trial_common_finite_sessions": common_periods,
                    "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
                    "joint_valid_bootstrap_repetitions": joint_count,
                    "joint_valid_bootstrap_rate": joint_rate,
                    "minimum_joint_valid_repetitions": MIN_JOINT_VALID_REPETITIONS,
                    "minimum_joint_valid_rate": MIN_JOINT_VALID_RATE,
                    "seed": BOOTSTRAP_SEED,
                    "method": "circular_moving_block_single_step_max_t_one_sided",
                    "correction_scope": "frozen_27_cell_plateau_family_only",
                    "historical_selection_exposure_included": False,
                }
            )
    return pl.DataFrame(rows).sort("candidate_id", "block_length_requested")


def _neighbor_specs() -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in CONTRACT["plateau_diagnostics"]["six_axis_neighbors"]:
        coordinates = (
            float(raw["depth_ratio_lte"]),
            float(raw["gap_ticks_gte"]),
            int(raw["bear_score_gte"]),
        )
        neighbor_id = candidate_id(*coordinates)
        if neighbor_id not in CELL_BY_ID or neighbor_id in seen:
            raise RuntimeError("frozen six-axis neighbor family is invalid")
        differences = [
            index
            for index, (value, center) in enumerate(
                zip(coordinates, CENTER_COORDINATES, strict=True)
            )
            if value != center
        ]
        if len(differences) != 1:
            raise RuntimeError("frozen neighbor is not one axis from center")
        axis_index = differences[0]
        axis = ("depth_ratio_lte", "gap_ticks_gte", "bear_score_gte")[axis_index]
        strictness_index = {
            "depth_ratio_lte": DEPTH_LEVELS,
            "gap_ticks_gte": GAP_LEVELS,
            "bear_score_gte": BEAR_LEVELS,
        }[axis].index(coordinates[axis_index])
        center_index = {
            "depth_ratio_lte": DEPTH_LEVELS,
            "gap_ticks_gte": GAP_LEVELS,
            "bear_score_gte": BEAR_LEVELS,
        }[axis].index(CENTER_COORDINATES[axis_index])
        result.append(
            {
                "neighbor_id": neighbor_id,
                "axis": axis,
                "direction_from_center": (
                    "stricter" if strictness_index > center_index else "looser"
                ),
                **CELL_BY_ID[neighbor_id].metadata(),
            }
        )
        seen.add(neighbor_id)
    if len(result) != 6:
        raise RuntimeError("frozen neighbor family does not contain six cells")
    return result


NEIGHBOR_SPECS = _neighbor_specs()
NEIGHBOR_IDS = tuple(str(item["neighbor_id"]) for item in NEIGHBOR_SPECS)


def build_neighbor_diagnostics(summaries: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for period in ("full", "2025", "2026"):
        period_frame = summaries.filter(pl.col("period") == period)
        effect_by_id = {
            str(row["candidate_id"]): row["equal_active_session_delta_z"]
            for row in period_frame.select(
                "candidate_id", "equal_active_session_delta_z"
            ).iter_rows(named=True)
        }
        center_effect = effect_by_id.get(CENTER_CELL_ID)
        neighbor_effects = [effect_by_id.get(neighbor_id) for neighbor_id in NEIGHBOR_IDS]
        estimable = (
            center_effect is not None
            and math.isfinite(float(center_effect))
            and all(value is not None and math.isfinite(float(value)) for value in neighbor_effects)
        )
        qualified = []
        if estimable:
            assert center_effect is not None
            qualified = [
                float(value) > 0.0 and float(value) >= 0.50 * float(center_effect)
                for value in neighbor_effects
            ]
            median_effect = float(np.median(np.asarray(neighbor_effects, dtype=float)))
            median_rule = median_effect >= 0.75 * float(center_effect)
        else:
            qualified = [False] * 6
            median_effect = None
            median_rule = False
        qualifying_count = int(sum(qualified))
        period_pass = bool(
            estimable
            and float(center_effect) > 0.0
            and qualifying_count >= 5
            and median_rule
        )
        for spec, effect, qualifies in zip(
            NEIGHBOR_SPECS, neighbor_effects, qualified, strict=True
        ):
            ratio = (
                float(effect) / float(center_effect)
                if estimable and float(center_effect) != 0.0
                else None
            )
            rows.append(
                {
                    "period": period,
                    **spec,
                    "center_candidate_id": CENTER_CELL_ID,
                    "center_effect": center_effect,
                    "neighbor_effect": effect,
                    "neighbor_to_center_ratio": ratio,
                    "neighbor_positive": (
                        bool(float(effect) > 0.0) if effect is not None else False
                    ),
                    "neighbor_at_least_half_center": (
                        bool(float(effect) >= 0.5 * float(center_effect))
                        if estimable
                        else False
                    ),
                    "neighbor_qualifies": qualifies,
                    "period_estimable": estimable,
                    "period_qualifying_neighbor_count": qualifying_count,
                    "required_qualifying_neighbor_count": 5,
                    "period_median_neighbor_effect": median_effect,
                    "period_median_to_center_ratio": (
                        median_effect / float(center_effect)
                        if estimable and float(center_effect) != 0.0
                        else None
                    ),
                    "required_median_to_center_ratio": 0.75,
                    "period_local_sign_and_scale_pass": period_pass,
                    "decision_role": (
                        "explicit_gate" if period == "full" else "mandatory_disclosure"
                    ),
                }
            )
    return pl.DataFrame(rows).sort("period", "neighbor_id")


FACTOR_LEVELS: dict[str, tuple[float | int, ...]] = {
    "depth_ratio_lte": DEPTH_LEVELS,
    "gap_ticks_gte": GAP_LEVELS,
    "bear_score_gte": BEAR_LEVELS,
}


def build_monotonicity(
    summaries: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    monotonic_rows: list[dict[str, object]] = []
    slice_rows: list[dict[str, object]] = []
    coordinate_names = tuple(FACTOR_LEVELS)
    for period in ("full", "2025", "2026"):
        period_frame = summaries.filter(pl.col("period") == period)
        effects = {
            str(row["candidate_id"]): row["equal_active_session_delta_z"]
            for row in period_frame.select(
                "candidate_id", "equal_active_session_delta_z"
            ).iter_rows(named=True)
        }
        for factor_index, factor in enumerate(coordinate_names):
            levels = FACTOR_LEVELS[factor]
            level_means: list[float | None] = []
            for level in levels:
                cell_effects = [
                    effects[cell.candidate_id]
                    for cell in CELLS
                    if (
                        cell.depth_ratio_lte,
                        cell.gap_ticks_gte,
                        cell.bear_score_gte,
                    )[factor_index]
                    == level
                ]
                if len(cell_effects) == 9 and all(
                    value is not None and math.isfinite(float(value))
                    for value in cell_effects
                ):
                    level_means.append(float(np.mean(np.asarray(cell_effects, dtype=float))))
                else:
                    level_means.append(None)
            level_estimable = all(value is not None for value in level_means)
            level_deltas = [
                (
                    float(level_means[index + 1]) - float(level_means[index])
                    if level_means[index] is not None and level_means[index + 1] is not None
                    else None
                )
                for index in range(2)
            ]
            non_decreasing = bool(
                level_estimable and all(float(value) >= 0.0 for value in level_deltas)
            )
            other_names = [name for name in coordinate_names if name != factor]
            other_levels = [FACTOR_LEVELS[name] for name in other_names]
            factor_slice_rows: list[dict[str, object]] = []
            for fixed_values in product(*other_levels):
                fixed = dict(zip(other_names, fixed_values, strict=True))
                slice_id = "|".join(f"{name}={fixed[name]:g}" for name in other_names)
                for transition in range(2):
                    loose_level = levels[transition]
                    strict_level = levels[transition + 1]
                    loose_coords = dict(fixed)
                    strict_coords = dict(fixed)
                    loose_coords[factor] = loose_level
                    strict_coords[factor] = strict_level
                    loose_id = candidate_id(
                        float(loose_coords["depth_ratio_lte"]),
                        float(loose_coords["gap_ticks_gte"]),
                        int(loose_coords["bear_score_gte"]),
                    )
                    strict_id = candidate_id(
                        float(strict_coords["depth_ratio_lte"]),
                        float(strict_coords["gap_ticks_gte"]),
                        int(strict_coords["bear_score_gte"]),
                    )
                    loose_effect = effects.get(loose_id)
                    strict_effect = effects.get(strict_id)
                    comparison_estimable = (
                        loose_effect is not None
                        and strict_effect is not None
                        and math.isfinite(float(loose_effect))
                        and math.isfinite(float(strict_effect))
                    )
                    difference = (
                        float(strict_effect) - float(loose_effect)
                        if comparison_estimable
                        else None
                    )
                    factor_slice_rows.append(
                        {
                            "period": period,
                            "factor": factor,
                            "slice_id": slice_id,
                            "fixed_factor_1": other_names[0],
                            "fixed_value_1": float(fixed_values[0]),
                            "fixed_factor_2": other_names[1],
                            "fixed_value_2": float(fixed_values[1]),
                            "transition_index": transition,
                            "looser_level": float(loose_level),
                            "stricter_level": float(strict_level),
                            "looser_candidate_id": loose_id,
                            "stricter_candidate_id": strict_id,
                            "looser_effect": loose_effect,
                            "stricter_effect": strict_effect,
                            "stricter_minus_looser_effect": difference,
                            "comparison_estimable": comparison_estimable,
                            "slice_reversal": (
                                bool(float(difference) < 0.0)
                                if difference is not None
                                else None
                            ),
                            "comparison_tolerance": 0.0,
                        }
                    )
            slice_rows.extend(factor_slice_rows)
            reversal_rows = [row for row in factor_slice_rows if row["slice_reversal"] is True]
            reversal_slices = {str(row["slice_id"]) for row in reversal_rows}
            monotonic_rows.append(
                {
                    "period": period,
                    "factor": factor,
                    "strictness_level_0": float(levels[0]),
                    "strictness_level_1": float(levels[1]),
                    "strictness_level_2": float(levels[2]),
                    "level_0_mean_equal_day_effect": level_means[0],
                    "level_1_mean_equal_day_effect": level_means[1],
                    "level_2_mean_equal_day_effect": level_means[2],
                    "level_1_minus_level_0": level_deltas[0],
                    "level_2_minus_level_1": level_deltas[1],
                    "all_level_means_estimable": level_estimable,
                    "factor_level_non_decreasing": non_decreasing,
                    "slice_comparisons": len(factor_slice_rows),
                    "estimable_slice_comparisons": sum(
                        bool(row["comparison_estimable"]) for row in factor_slice_rows
                    ),
                    "slice_reversal_comparisons": len(reversal_rows),
                    "slices_with_any_reversal": len(reversal_slices),
                    "level_aggregation": (
                        "equal mean of nine cell equal-active-session effects"
                    ),
                    "decision_role": "mandatory_disclosure",
                }
            )
    return (
        pl.DataFrame(monotonic_rows).sort("period", "factor"),
        pl.DataFrame(slice_rows).sort("period", "factor", "slice_id", "transition_index"),
    )


def build_trial_ledger(
    full_summary: pl.DataFrame,
    year_summary: pl.DataFrame,
    fwer: pl.DataFrame,
) -> pl.DataFrame:
    summary_lookup = {
        (str(row["period"]), str(row["candidate_id"])): row
        for row in pl.concat([full_summary, year_summary], how="vertical").iter_rows(named=True)
    }
    fwer_lookup = {
        (str(row["candidate_id"]), int(row["block_length_requested"])): row
        for row in fwer.iter_rows(named=True)
    }
    rows: list[dict[str, object]] = []
    for index, cell in enumerate(CELLS):
        full = summary_lookup[("full", cell.candidate_id)]
        row: dict[str, object] = {
            "ledger_contract": LEDGER_CONTRACT,
            "plateau_contract_id": CONTRACT_ID,
            "family_id": FAMILY_ID,
            "family_version": FAMILY_VERSION,
            "raw_trial_index": index,
            **cell.metadata(),
            "is_frozen_center": cell.candidate_id == CENTER_CELL_ID,
            "is_axis_neighbor": cell.candidate_id in NEIGHBOR_IDS,
            "selection_role": "fixed_ledger_no_ranking_no_center_replacement",
            "support_floor_pass": full["support_floor_pass"],
        }
        for period in ("full", "2025", "2026"):
            summary = summary_lookup[(period, cell.candidate_id)]
            prefix = "overall" if period == "full" else period
            row[f"{prefix}_active_sessions"] = summary["active_sessions"]
            row[f"{prefix}_effect_valid_events"] = summary["effect_valid_events"]
            row[f"{prefix}_equal_day_delta_z"] = summary[
                "equal_active_session_delta_z"
            ]
            row[f"{prefix}_selection_rate_mean"] = summary[
                "selection_rate_equal_session_mean"
            ]
        for block_length in BLOCK_LENGTHS:
            inference = fwer_lookup[(cell.candidate_id, block_length)]
            row[f"b{block_length}_simultaneous_lcb"] = inference[
                "simultaneous_lower_bound"
            ]
            row[f"b{block_length}_single_step_fwer_p"] = inference[
                "single_step_fwer_p"
            ]
            row[f"b{block_length}_joint_valid_rate"] = inference[
                "joint_valid_bootstrap_rate"
            ]
        rows.append(row)
    return pl.DataFrame(rows).sort("raw_trial_index")


def build_decision(
    *,
    full_summary: pl.DataFrame,
    fwer: pl.DataFrame,
    neighbors: pl.DataFrame,
    monotonicity: pl.DataFrame,
    slice_reversals: pl.DataFrame,
    input_hashes: Mapping[str, str],
) -> dict[str, Any]:
    support_failed = full_summary.filter(~pl.col("support_floor_pass"))[
        "candidate_id"
    ].to_list()
    support_pass = not support_failed
    center_fwer = fwer.filter(pl.col("candidate_id") == CENTER_CELL_ID)
    if set(center_fwer["block_length_requested"].to_list()) != set(BLOCK_LENGTHS):
        raise PlateauContractError("center family uncertainty lacks a frozen block length")
    center_bounds = {
        str(row["block_length_requested"]): row["simultaneous_lower_bound"]
        for row in center_fwer.iter_rows(named=True)
    }
    center_pass = all(
        value is not None and math.isfinite(float(value)) and float(value) > 0.0
        for value in center_bounds.values()
    )
    local_by_period = {
        period: bool(
            neighbors.filter(pl.col("period") == period)[
                "period_local_sign_and_scale_pass"
            ][0]
        )
        for period in ("full", "2025", "2026")
    }
    local_full_pass = local_by_period["full"]
    explicit_gates = {
        "P1_support_floor_each_cell": {
            "decision_role": "explicit_gate",
            "pass": support_pass,
            "required_active_sessions": MIN_ACTIVE_SESSIONS,
            "required_effect_valid_events": MIN_EVENTS,
            "failed_candidate_ids": support_failed,
        },
        "P2_center_family_uncertainty": {
            "decision_role": "explicit_gate",
            "pass": center_pass,
            "rule": "center simultaneous LCB > 0 at circular block lengths 5, 10, 20",
            "center_candidate_id": CENTER_CELL_ID,
            "simultaneous_lower_bounds": center_bounds,
        },
        "P3_full_period_local_sign_and_scale": {
            "decision_role": "explicit_gate",
            "pass": local_full_pass,
            "rule": (
                "at least five of six neighbors positive and >=50% center; "
                "median neighbor effect >=75% center"
            ),
        },
    }
    failed_gate_ids = [name for name, gate in explicit_gates.items() if not gate["pass"]]
    diagnostic_pass = not failed_gate_ids
    year_stable = local_by_period["2025"] and local_by_period["2026"]
    monotonic_rows = monotonicity.to_dicts()
    factor_level_failures = [
        {"period": row["period"], "factor": row["factor"]}
        for row in monotonic_rows
        if not row["factor_level_non_decreasing"]
    ]
    reversal_count = int(
        slice_reversals.filter(pl.col("slice_reversal") == True).height  # noqa: E712
    )
    warnings: list[str] = []
    if not year_stable:
        warnings.append(
            "Full-period local pass is insufficient to call the plateau cross-year stable: "
            "at least one mandatory 2025/2026 local diagnostic failed."
        )
    if factor_level_failures or reversal_count:
        warnings.append(
            "Mandatory monotonicity disclosure contains factor-level non-monotonicity "
            "or slice reversals; these are not hidden and were not preregistered as pass gates."
        )
    warnings.append(
        "This is outcome-informed retrospective diagnostic evidence and cannot authorize deployment."
    )
    return {
        "contract": DECISION_CONTRACT,
        "plateau_contract_id": CONTRACT_ID,
        "frozen_contract_sha256": CONTRACT_SHA256,
        "family_id": FAMILY_ID,
        "family_version": FAMILY_VERSION,
        "center_candidate_id": CENTER_CELL_ID,
        "input_artifact_sha256": dict(input_hashes),
        "decision_status": (
            "diagnostic_plateau_pass" if diagnostic_pass else "diagnostic_plateau_fail"
        ),
        "diagnostic_plateau_pass": diagnostic_pass,
        "failed_gate_ids": failed_gate_ids,
        "gates": explicit_gates,
        "mandatory_disclosures": {
            "year_local_sign_and_scale": {
                "decision_role": "mandatory_disclosure_not_overall_gate",
                "period_pass": local_by_period,
                "cross_year_both_pass": year_stable,
            },
            "factor_level_monotonicity": {
                "decision_role": "mandatory_disclosure_not_overall_gate",
                "all_factor_periods_non_decreasing": not factor_level_failures,
                "failed_factor_periods": factor_level_failures,
            },
            "slice_reversals": {
                "decision_role": "mandatory_disclosure_not_overall_gate",
                "comparison_rows": len(slice_reversals),
                "reversal_rows": reversal_count,
            },
            "prevalence_and_support": {
                "decision_role": "required_report",
                "periods": ["full", "2025", "2026"],
                "support_event_definition": "effect_valid_condition_n_z",
            },
        },
        "warnings": warnings,
        "pass_interpretation": CONTRACT["pass_interpretation"],
        "fail_interpretation": CONTRACT["fail_interpretation"],
        "production_promotion_authorized": False,
        "center_replacement_authorized": False,
        "retrospective_evidence_only": True,
    }


def evaluate_threshold_plateau(
    *,
    daily_aggregates: pl.DataFrame,
    requested_calendar: pl.DataFrame,
    requested_calendar_manifest: Mapping[str, object],
    runner_manifest: Mapping[str, object],
    input_artifact_sha256: Mapping[str, str],
) -> PlateauResult:
    input_hashes = _validate_hash_bindings(input_artifact_sha256)
    sessions, runner_artifact = validate_runner_manifest(runner_manifest)
    calendar = prepare_requested_calendar(
        requested_calendar,
        requested_calendar_manifest,
        run_sessions=sessions,
    )
    daily = prepare_daily_effects(
        daily_aggregates,
        calendar=calendar,
        runner_manifest=runner_manifest,
        runner_artifact=runner_artifact,
    )
    full_summary = summarize_cells(daily, "full")
    year_summary = pl.concat(
        [summarize_cells(daily, "2025"), summarize_cells(daily, "2026")],
        how="vertical",
    ).sort("period", "candidate_id")
    combined_summary = pl.concat([full_summary, year_summary], how="vertical")
    fwer = circular_maxstat(daily)
    neighbors = build_neighbor_diagnostics(combined_summary)
    monotonicity, slice_reversals = build_monotonicity(combined_summary)
    ledger = build_trial_ledger(full_summary, year_summary, fwer)
    decision = build_decision(
        full_summary=full_summary,
        fwer=fwer,
        neighbors=neighbors,
        monotonicity=monotonicity,
        slice_reversals=slice_reversals,
        input_hashes=input_hashes,
    )
    manifest: dict[str, Any] = {
        "contract": EVALUATOR_CONTRACT,
        "plateau_contract_id": CONTRACT_ID,
        "frozen_contract_sha256": CONTRACT_SHA256,
        "threshold_design_fingerprint": EXPECTED_DESIGN_FINGERPRINT,
        "family_id": FAMILY_ID,
        "family_version": FAMILY_VERSION,
        "center_candidate_id": CENTER_CELL_ID,
        "candidate_ids": list(CELL_IDS),
        "input_artifact_sha256": input_hashes,
        "calendar": {
            "start": START_DATE,
            "end": END_DATE,
            "requested_sessions": len(calendar),
            "evidence_class": "retrospective_diagnostic",
            "years": ["2025", "2026"],
        },
        "estimand": "shared_pool_equal_active_session_delta_z",
        "zero_event_policy": "undefined_not_zero",
        "bootstrap": {
            "method": "circular_moving_block_single_step_max_t_one_sided",
            "block_lengths_in_frozen_order": list(BLOCK_LENGTHS),
            "seed": BOOTSTRAP_SEED,
            "repetitions": BOOTSTRAP_REPETITIONS,
            "family_alpha": FAMILY_ALPHA,
            "joint_valid_minimum_rate": MIN_JOINT_VALID_RATE,
            "joint_valid_minimum_repetitions": MIN_JOINT_VALID_REPETITIONS,
        },
        "decision_contract": DECISION_CONTRACT,
        "decision_status": decision["decision_status"],
        "diagnostic_plateau_pass": decision["diagnostic_plateau_pass"],
        "production_promotion_authorized": False,
        "selection_rule": "complete fixed 27-cell ledger; no ranking or center replacement",
        "disclosure_role": {
            "year_stability": "mandatory_not_overall_gate",
            "monotonicity_and_slice_reversals": "mandatory_not_overall_gate",
        },
        "output_rows": {
            "requested_calendar": len(calendar),
            "daily_effects": len(daily),
            "cell_summary": len(full_summary),
            "year_summary": len(year_summary),
            "fwer_audit": len(fwer),
            "neighbor_diagnostics": len(neighbors),
            "monotonicity": len(monotonicity),
            "slice_reversals": len(slice_reversals),
            "trial_ledger": len(ledger),
        },
    }
    return PlateauResult(
        requested_calendar=calendar,
        daily_effects=daily,
        cell_summary=full_summary,
        year_summary=year_summary,
        fwer_audit=fwer,
        neighbor_diagnostics=neighbors,
        monotonicity=monotonicity,
        slice_reversals=slice_reversals,
        trial_ledger=ledger,
        decision=decision,
        manifest=manifest,
    )


def _atomic_write_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    temp_path = Path(handle.name)
    handle.close()
    try:
        frame.write_parquet(temp_path, compression="zstd")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _atomic_write_json(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
        encoding="utf-8",
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


OUTPUT_FRAMES = (
    "requested_calendar.parquet",
    "daily_effects.parquet",
    "cell_summary.parquet",
    "year_summary.parquet",
    "fwer_audit.parquet",
    "neighbor_diagnostics.parquet",
    "monotonicity.parquet",
    "slice_reversals.parquet",
    "trial_ledger.parquet",
)


def write_result(result: PlateauResult, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "requested_calendar.parquet": result.requested_calendar,
        "daily_effects.parquet": result.daily_effects,
        "cell_summary.parquet": result.cell_summary,
        "year_summary.parquet": result.year_summary,
        "fwer_audit.parquet": result.fwer_audit,
        "neighbor_diagnostics.parquet": result.neighbor_diagnostics,
        "monotonicity.parquet": result.monotonicity,
        "slice_reversals.parquet": result.slice_reversals,
        "trial_ledger.parquet": result.trial_ledger,
    }
    for name, frame in frames.items():
        _atomic_write_parquet(frame, output_dir / name)
    _atomic_write_json(result.decision, output_dir / "plateau_decision.json")
    output_artifacts = {
        name: _file_identity(output_dir / name)
        for name in (*OUTPUT_FRAMES, "plateau_decision.json")
    }
    manifest = dict(result.manifest)
    manifest["output_artifacts"] = output_artifacts
    _atomic_write_json(manifest, output_dir / "run_manifest.json")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen 27-cell fh_neg_rejection threshold plateau"
    )
    parser.add_argument("--daily-aggregates", type=Path, required=True)
    parser.add_argument("--plateau-run-manifest", type=Path, required=True)
    parser.add_argument("--requested-calendar", type=Path, required=True)
    parser.add_argument("--requested-calendar-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _output_contains_input(output_dir: Path, input_path: Path) -> bool:
    """Return whether writing ``output_dir`` could overwrite an input file.

    A sibling output below the study directory is valid even though the frozen
    contract itself also lives in that study directory.  The unsafe direction
    is the inverse: an input file must not already be inside the requested
    output tree.
    """

    output = output_dir.resolve()
    source = input_path.resolve()
    return output == source.parent or output in source.parents


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_paths = {
        "daily_aggregates": args.daily_aggregates.resolve(),
        "plateau_run_manifest": args.plateau_run_manifest.resolve(),
        "requested_calendar": args.requested_calendar.resolve(),
        "requested_calendar_manifest": args.requested_calendar_manifest.resolve(),
        "threshold_contract": CONTRACT_PATH.resolve(),
    }
    missing = [str(path) for path in input_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"plateau evaluator inputs missing: {missing}")
    output_dir = args.output_dir.resolve()
    if any(_output_contains_input(output_dir, path) for path in input_paths.values()):
        raise PlateauContractError("evaluator output must not modify an input artifact directory")
    input_hashes = {name: _sha256_file(path) for name, path in input_paths.items()}
    result = evaluate_threshold_plateau(
        daily_aggregates=pl.read_parquet(input_paths["daily_aggregates"]),
        requested_calendar=pl.read_parquet(input_paths["requested_calendar"]),
        requested_calendar_manifest=_read_json(
            input_paths["requested_calendar_manifest"]
        ),
        runner_manifest=_read_json(input_paths["plateau_run_manifest"]),
        input_artifact_sha256=input_hashes,
    )
    manifest = write_result(result, output_dir)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
