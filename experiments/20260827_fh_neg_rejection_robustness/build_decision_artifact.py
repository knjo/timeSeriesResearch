"""Build the fail-closed final research decision for fh_neg_rejection.

The builder reads already materialized study artifacts, verifies their frozen
identities and cross-artifact contracts, and emits a compact machine-readable
decision.  It never promotes a selector: the only permitted overall status is
``RESEARCH_SHADOW_ONLY`` and deployment remains unauthorized.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import polars as pl


STUDY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STUDY_DIR.parents[4]
ARTIFACT_ROOT = STUDY_DIR / "artifacts"
DEFAULT_OUTPUT_DIR = ARTIFACT_ROOT / "final_decision_v1"

DECISION_CONTRACT = "fh_neg_rejection_final_decision/v1"
REPORT_DATA_CONTRACT = "fh_neg_rejection_final_report_data/v1"
PARENT_CONTRACT = "fh_neg_rejection_robustness/fixed_factorial_v1"
FAMILY_ID = "fh_neg_rejection_dgb_factorial"
FAMILY_VERSION = "1"
ANCHOR_ID = "d1_g1_b1"
PLATEAU_CONTRACT = "fh_neg_rejection_threshold_plateau/fixed_grid_v1"
PLATEAU_FAMILY_ID = "fh_neg_rejection_threshold_plateau"
PLATEAU_FAMILY_VERSION = "1"
PLATEAU_CENTER_ID = "d08_g04_b2"
PLATEAU_EVALUATOR_CONTRACT = (
    "fh_neg_rejection_threshold_plateau_evaluator/v1"
)
PLATEAU_DECISION_CONTRACT = (
    "fh_neg_rejection_threshold_plateau_decision/v1"
)
EXPECTED_PLATEAU_DESIGN_FINGERPRINT = (
    "226f16c8a26f9535bc6bc6462e8da5a16c8ae1f1dc0f409971a36bc8073e51ee"
)

CELL_IDS = (
    "d0_g0_b0",
    "d1_g0_b0",
    "d0_g1_b0",
    "d0_g0_b1",
    "d1_g1_b0",
    "d1_g0_b1",
    "d0_g1_b1",
    "d1_g1_b1",
)
LOO_SPECS = {
    "D_increment": {
        "full_cell_id": ANCHOR_ID,
        "reduced_cell_id": "d0_g1_b1",
        "removed_gate": "depth",
    },
    "G_increment": {
        "full_cell_id": ANCHOR_ID,
        "reduced_cell_id": "d1_g0_b1",
        "removed_gate": "far",
    },
    "B_increment": {
        "full_cell_id": ANCHOR_ID,
        "reduced_cell_id": "d1_g1_b0",
        "removed_gate": "bear",
    },
}
LOO_IDS = tuple(LOO_SPECS)
PLATEAU_CELL_IDS = tuple(
    f"d{depth:02d}_g{gap:02d}_b{bear}"
    for depth in (6, 8, 10)
    for gap in (3, 4, 5)
    for bear in (1, 2, 3)
)

INFERENCE_FILES = (
    "requested_calendar.parquet",
    "input_daily_stats.parquet",
    "daily_effects.parquet",
    "cell_summary.parquet",
    "loo_daily.parquet",
    "loo_summary.parquet",
    "factorial_contrast_daily.parquet",
    "factorial_contrasts.parquet",
    "bootstrap.parquet",
    "fwer_audit.parquet",
    "temporal_blocks.parquet",
    "trial_ledger.parquet",
    "run_manifest.json",
)
LINEAGE_FILES = (
    "cohort_support.parquet",
    "reference_status.parquet",
    "selector_input_audit.parquet",
    "session_support.parquet",
    "lineage_support_summary.json",
)
SECONDARY_FILES = (
    "secondary_daily_effects.parquet",
    "equal_active_bp.parquet",
    "monthly_effects.parquet",
    "monthly_summary.parquet",
    "hac_diagnostics.parquet",
    "temporal_basic_intervals.parquet",
    "run_manifest.json",
)
ARTIFACT_OOS_FILES = (
    "cell_descriptive.parquet",
    "loo_descriptive.parquet",
    "run_manifest.json",
)
PLATEAU_RUN_FILES = (
    "daily_stats.parquet",
    "summary.parquet",
    "run_manifest.json",
)
PLATEAU_EVALUATION_FILES = (
    "requested_calendar.parquet",
    "daily_effects.parquet",
    "cell_summary.parquet",
    "year_summary.parquet",
    "fwer_audit.parquet",
    "neighbor_diagnostics.parquet",
    "monotonicity.parquet",
    "slice_reversals.parquet",
    "trial_ledger.parquet",
    "plateau_decision.json",
    "run_manifest.json",
)
PLATEAU_INPUT_HASH_KEYS = (
    "daily_aggregates",
    "plateau_run_manifest",
    "requested_calendar",
    "requested_calendar_manifest",
    "threshold_contract",
)
PLATEAU_GATE_IDS = (
    "P1_support_floor_each_cell",
    "P2_center_family_uncertainty",
    "P3_full_period_local_sign_and_scale",
)
PLATEAU_NEIGHBOR_IDS = (
    "d06_g04_b2",
    "d10_g04_b2",
    "d08_g03_b2",
    "d08_g05_b2",
    "d08_g04_b1",
    "d08_g04_b3",
)


class DecisionArtifactError(ValueError):
    """Raised when any required input is absent, mutable, or contradictory."""


@dataclass(frozen=True)
class DecisionPaths:
    project_root: Path
    study_dir: Path
    calendar_dir: Path
    lineage_dir: Path
    positive_control: Path
    inference_dir: Path
    secondary_dir: Path
    artifact_oos_dir: Path
    historical_exposure: Path
    plateau_run_dir: Path
    plateau_evaluation_dir: Path
    output_dir: Path

    @classmethod
    def defaults(cls, output_dir: Path = DEFAULT_OUTPUT_DIR) -> "DecisionPaths":
        return cls(
            project_root=PROJECT_ROOT,
            study_dir=STUDY_DIR,
            calendar_dir=ARTIFACT_ROOT / "calendar_retrospective_v1",
            lineage_dir=ARTIFACT_ROOT / "lineage_retrospective_v1",
            positive_control=STUDY_DIR / "positive_control_reference.json",
            inference_dir=ARTIFACT_ROOT / "inference_retrospective_v1",
            secondary_dir=ARTIFACT_ROOT / "secondary_diagnostics_v1",
            artifact_oos_dir=(
                ARTIFACT_ROOT
                / "secondary_diagnostics_v1"
                / "artifact_oos_20260826_descriptive_v1"
            ),
            historical_exposure=(
                STUDY_DIR / "historical_trial_exposure.json"
            ),
            plateau_run_dir=ARTIFACT_ROOT / "threshold_plateau_v1",
            plateau_evaluation_dir=(
                ARTIFACT_ROOT / "threshold_plateau_evaluation_v1"
            ),
            output_dir=output_dir,
        )


@dataclass(frozen=True)
class DecisionResult:
    decision: dict[str, Any]
    report_data: dict[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DecisionArtifactError(message)


def _require_keys(
    payload: Mapping[str, Any],
    keys: Sequence[str],
    source: str,
) -> None:
    missing = sorted(set(keys) - set(payload))
    if missing:
        raise DecisionArtifactError(f"{source}: missing fields {missing}")


def _require_columns(
    frame: pl.DataFrame,
    columns: Sequence[str],
    source: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise DecisionArtifactError(f"{source}: missing columns {missing}")


def _finite(value: object, *, source: str) -> float:
    _require(value is not None, f"{source}: value is missing")
    output = float(value)
    _require(math.isfinite(output), f"{source}: value is not finite")
    return output


def _require_close(
    actual: object,
    expected: object,
    *,
    source: str,
    abs_tol: float = 1e-12,
) -> None:
    actual_value = _finite(actual, source=f"{source} actual")
    expected_value = _finite(expected, source=f"{source} expected")
    _require(
        math.isclose(
            actual_value,
            expected_value,
            rel_tol=0.0,
            abs_tol=abs_tol,
        ),
        f"{source}: summary disagrees with detailed evidence",
    )


def _finite_values(values: Sequence[object]) -> list[float]:
    output: list[float] = []
    for value in values:
        if value is None:
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            output.append(numeric)
    return output


def _mean(values: Sequence[float]) -> float:
    _require(bool(values), "cannot compute mean of an empty finite sequence")
    return math.fsum(values) / len(values)


def _median(values: Sequence[float]) -> float:
    _require(bool(values), "cannot compute median of an empty finite sequence")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _sample_std(values: Sequence[float]) -> float:
    _require(len(values) > 1, "sample standard deviation needs two values")
    center = _mean(values)
    return math.sqrt(
        math.fsum((value - center) ** 2 for value in values)
        / (len(values) - 1)
    )


def _chronological_blocks(
    dates: Sequence[str], block_count: int
) -> list[list[str]]:
    _require(block_count > 0, "temporal block count must be positive")
    _require(bool(dates), "temporal calendar is empty")
    count = min(block_count, len(dates))
    quotient, remainder = divmod(len(dates), count)
    blocks: list[list[str]] = []
    offset = 0
    for index in range(count):
        length = quotient + (1 if index < remainder else 0)
        blocks.append([str(value) for value in dates[offset : offset + length]])
        offset += length
    _require(offset == len(dates), "temporal block partition is incomplete")
    return blocks


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"required artifact is missing: {path}")
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "sha256": _sha256_file(path),
    }


def _load_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"required JSON artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"{path}: JSON root must be an object")
    return payload


def _require_files(directory: Path, names: Sequence[str]) -> None:
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise DecisionArtifactError(
            f"{directory}: required artifacts are missing {missing}"
        )


def _fingerprint_group(
    directory: Path,
    names: Sequence[str],
) -> dict[str, dict[str, Any]]:
    _require_files(directory, names)
    return {name: _identity(directory / name) for name in names}


def _resolve_recorded_path(
    record: Mapping[str, Any],
    *,
    project_root: Path,
    source: str,
) -> Path:
    _require_keys(record, ("path", "sha256"), source)
    raw = Path(str(record["path"]))
    path = raw if raw.is_absolute() else project_root / raw
    resolved = path.resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as exc:
        raise DecisionArtifactError(
            f"{source}: recorded path escapes project root: {resolved}"
        ) from exc
    return resolved


def _validate_recorded_identity(
    record: Mapping[str, Any],
    *,
    project_root: Path,
    source: str,
) -> dict[str, Any]:
    path = _resolve_recorded_path(
        record, project_root=project_root, source=source
    )
    actual = _identity(path)
    _require(
        actual["sha256"] == record["sha256"],
        f"{source}: recorded SHA-256 drifted",
    )
    if "size" in record:
        _require(
            actual["size"] == int(record["size"]),
            f"{source}: recorded size drifted",
        )
    return actual


def _validate_identity_map(
    records: Mapping[str, Any],
    *,
    project_root: Path,
    source: str,
) -> dict[str, dict[str, Any]]:
    _require(bool(records), f"{source}: identity map is empty")
    output: dict[str, dict[str, Any]] = {}
    for key, value in records.items():
        _require(
            isinstance(value, Mapping),
            f"{source}.{key}: identity must be an object",
        )
        output[str(key)] = _validate_recorded_identity(
            value,
            project_root=project_root,
            source=f"{source}.{key}",
        )
    return output


def _row(frame: pl.DataFrame, expression: pl.Expr, source: str) -> dict[str, Any]:
    subset = frame.filter(expression)
    _require(len(subset) == 1, f"{source}: expected exactly one row")
    return subset.row(0, named=True)


def _require_exact_keys(
    frame: pl.DataFrame,
    columns: Sequence[str],
    expected: set[tuple[object, ...]],
    source: str,
) -> None:
    _require_columns(frame, columns, source)
    actual_rows = [
        tuple(row)
        for row in frame.select(columns).iter_rows()
    ]
    _require(
        len(actual_rows) == len(set(actual_rows)),
        f"{source}: duplicate keys",
    )
    _require(set(actual_rows) == expected, f"{source}: exact key family drifted")


def _validate_calendar(paths: DecisionPaths) -> dict[str, Any]:
    parquet_path = paths.calendar_dir / "requested_calendar.parquet"
    manifest_path = paths.calendar_dir / "requested_calendar.json"
    identities = {
        "requested_calendar.parquet": _identity(parquet_path),
        "requested_calendar.json": _identity(manifest_path),
    }
    frame = pl.read_parquet(parquet_path)
    manifest = _load_json(manifest_path)
    _require_columns(
        frame,
        (
            "Date",
            "evidence_class",
            "source_complete",
            "run_status",
            "session_status",
            "control_n_z",
        ),
        "retrospective calendar",
    )
    expected_manifest = {
        "contract": "fh_neg_rejection_requested_calendar/v1",
        "factorial_contract": PARENT_CONTRACT,
        "factorial_family_id": FAMILY_ID,
        "factorial_family_version": FAMILY_VERSION,
        "evidence_class": "retrospective_diagnostic",
        "start": "20250211",
        "end": "20260825",
    }
    for key, expected in expected_manifest.items():
        _require(
            manifest.get(key) == expected,
            f"retrospective calendar manifest: {key} drifted",
        )
    _require(
        frame["Date"].n_unique() == len(frame),
        "retrospective calendar has duplicate Date",
    )
    _require(
        frame["Date"].sort().to_list() == frame["Date"].to_list(),
        "retrospective calendar is not sorted",
    )
    _require(
        set(frame["evidence_class"].unique().to_list())
        == {"retrospective_diagnostic"},
        "retrospective calendar mixes evidence classes",
    )
    _require(
        frame["Date"].min() == "20250211"
        and frame["Date"].max() == "20260825",
        "retrospective calendar date boundary drifted",
    )
    _require(
        len(frame) == int(manifest["requested_sessions"]),
        "retrospective calendar row count disagrees with manifest",
    )
    status_counts = frame.group_by("session_status").len()
    counts = dict(
        zip(
            status_counts["session_status"].to_list(),
            status_counts["len"].to_list(),
            strict=True,
        )
    )
    _require(
        set(counts) <= {"evaluable", "zero_control_ineligible"},
        "retrospective calendar contains an unrecognized session status",
    )
    _require(
        counts.get("evaluable", 0) == int(manifest["evaluable_sessions"])
        and counts.get("zero_control_ineligible", 0)
        == int(manifest["zero_control_sessions"]),
        "retrospective calendar status counts disagree with manifest",
    )
    _require(
        bool(
            frame.select(
                (
                    ((pl.col("session_status") == "evaluable")
                    & (pl.col("control_n_z") > 0))
                    | ((pl.col("session_status") == "zero_control_ineligible")
                    & (pl.col("control_n_z") == 0))
                ).all()
            ).item()
        ),
        "retrospective calendar control support contradicts session status",
    )
    _require(
        int((frame["run_status"] == "success").sum())
        == int(manifest["successful_sessions"]),
        "retrospective calendar success count drifted",
    )
    _require(
        bool(frame["source_complete"].all()),
        "retrospective calendar contains incomplete sources",
    )
    _require(
        int(manifest["failed_run_sessions"]) == 0
        and int(manifest["missing_source_sessions"]) == 0
        and int(manifest["not_run_or_incomplete_sessions"]) == 0,
        "retrospective calendar contains missing or failed sessions",
    )
    return {
        "identities": identities,
        "frame": frame,
        "manifest": manifest,
        "summary": {
            "requested_sessions": len(frame),
            "evaluable_sessions": int(manifest["evaluable_sessions"]),
            "zero_control_sessions": int(manifest["zero_control_sessions"]),
            "missing_source_sessions": int(manifest["missing_source_sessions"]),
            "failed_run_sessions": int(manifest["failed_run_sessions"]),
        },
    }


def _validate_lineage(
    paths: DecisionPaths,
    calendar: Mapping[str, Any],
) -> dict[str, Any]:
    identities = _fingerprint_group(paths.lineage_dir, LINEAGE_FILES)
    summary = _load_json(paths.lineage_dir / "lineage_support_summary.json")
    expected = {
        "contract": "fh_neg_rejection_lineage_support_audit/v1",
        "factorial_contract": PARENT_CONTRACT,
        "factorial_family_id": FAMILY_ID,
        "factorial_family_version": FAMILY_VERSION,
        "status": "pass",
    }
    for key, value in expected.items():
        _require(summary.get(key) == value, f"lineage summary: {key} drifted")
    scope = summary.get("scope")
    _require(isinstance(scope, Mapping), "lineage summary: scope is missing")
    _require(
        scope.get("evidence_class") == "retrospective_diagnostic"
        and scope.get("start") == "20250211"
        and scope.get("end") == "20260825",
        "lineage summary scope drifted",
    )
    _require(
        int(scope["requested_sessions"])
        == int(calendar["summary"]["requested_sessions"]),
        "lineage requested-session count disagrees with calendar",
    )
    _require(
        summary.get("outcome_artifacts_read") == []
        and summary.get("outcome_columns_read") == [],
        "lineage audit read outcome artifacts",
    )
    selector = summary.get("selector_static_audit")
    _require(isinstance(selector, Mapping), "selector static audit is missing")
    _require(
        selector.get("status") == "pass"
        and int(selector["forbidden_input_count"]) == 0
        and int(selector["future_syntax_finding_count"]) == 0,
        "selector static audit did not pass",
    )
    parity = summary.get("legacy_parity")
    _require(isinstance(parity, Mapping), "legacy parity audit is missing")
    _require(
        parity.get("all_applicable_sessions_passed") is True,
        "legacy parity did not pass",
    )
    input_records = summary.get("input_identities")
    _require(isinstance(input_records, Mapping), "lineage input identities missing")
    recorded = _validate_identity_map(
        input_records,
        project_root=paths.project_root,
        source="lineage.input_identities",
    )
    calendar_hash = calendar["identities"]["requested_calendar.parquet"]["sha256"]
    _require(
        recorded["requested_calendar"]["sha256"] == calendar_hash,
        "lineage calendar identity disagrees with decision calendar",
    )
    support = summary.get("support")
    _require(isinstance(support, Mapping), "lineage support is missing")
    dgb = support.get("factorial_cells", {}).get(ANCHOR_ID)
    _require(isinstance(dgb, Mapping), "lineage anchor support is missing")
    gaps = summary.get("validation_gaps")
    _require(isinstance(gaps, list), "lineage validation gaps are missing")
    return {
        "identities": identities,
        "summary": summary,
        "recorded_inputs": recorded,
        "diagnostic": {
            "status": "PASS_WITH_DISCLOSED_LIMITATIONS",
            "raw_membership_events": int(dgb["events"]),
            "raw_membership_active_sessions": int(dgb["active_sessions"]),
            "validation_gaps": gaps,
        },
    }


def _validate_positive_control(paths: DecisionPaths) -> dict[str, Any]:
    identity = _identity(paths.positive_control)
    payload = _load_json(paths.positive_control)
    _require(
        payload.get("artifact_id")
        == "fh_neg_rejection_robustness/positive_control_reference/v1",
        "positive-control artifact identity drifted",
    )
    verdict = payload.get("verdict")
    _require(isinstance(verdict, Mapping), "positive-control verdict missing")
    _require(
        verdict.get("positive_control_component") == "pass_reference",
        "positive-control component did not pass",
    )
    _require(
        verdict.get("full_g0_status") == "not_decided_by_this_artifact",
        "positive-control artifact overclaims full G0",
    )
    alignment = payload.get("contract_alignment")
    _require(isinstance(alignment, Mapping), "positive-control alignment missing")
    for component in (
        "condition",
        "directional_label",
        "eligible_pool",
        "control_and_estimand",
        "pipeline_linkage",
    ):
        item = alignment.get(component)
        _require(
            isinstance(item, Mapping) and item.get("pass") is True,
            f"positive-control alignment {component} did not pass",
        )
    _require(
        alignment["condition"].get("expected") == "FillLots_atLow < 0"
        and alignment["directional_label"].get("expected_profile_id")
        == "negfill.short_taker_stop108.v1"
        and alignment["directional_label"].get("canonical_profile_hash")
        == alignment["directional_label"].get("current_factorial_profile_hash")
        and alignment["eligible_pool"].get("expected_pool_id")
        == "tw_equity_daytrade_rank150_090030_123000_bidref_v1"
        and alignment["control_and_estimand"].get("canonical_control_mode")
        == "full"
        and alignment["control_and_estimand"].get(
            "current_factorial_control_mode"
        )
        == "full"
        and alignment["control_and_estimand"].get(
            "canonical_primary_estimand"
        )
        == "shared_pool"
        and alignment["control_and_estimand"].get(
            "current_factorial_primary_estimand"
        )
        == "shared_pool",
        "positive-control condition/profile/pool/control semantics drifted",
    )
    reference = payload.get("canonical_reference")
    _require(isinstance(reference, Mapping), "canonical positive control missing")
    preregistered = reference.get("preregistered_cell")
    _require(isinstance(preregistered, Mapping), "positive-control cell missing")
    _require(
        _finite(
            preregistered.get("equal_day_delta_z"),
            source="positive-control equal_day_delta_z",
        )
        > 0.0,
        "positive-control direction was not recovered",
    )
    cross_year = reference.get("cross_year_direction")
    _require(
        isinstance(cross_year, list) and len(cross_year) == 2,
        "positive-control cross-year disclosure is incomplete",
    )
    _require(
        all(
            _finite(row.get("equal_day_delta_z"), source="positive-control year")
            > 0.0
            for row in cross_year
        ),
        "positive-control year direction changed sign",
    )
    fingerprints = payload.get("fingerprints")
    _require(isinstance(fingerprints, Mapping), "positive-control fingerprints missing")
    verified = _validate_identity_map(
        fingerprints,
        project_root=paths.project_root,
        source="positive_control.fingerprints",
    )
    return {
        "identity": identity,
        "verified_references": verified,
        "summary": {
            "status": "PASS_REFERENCE_WITH_LIMITATIONS",
            "equal_day_delta_z": float(preregistered["equal_day_delta_z"]),
            "equal_day_raw_bp_uplift": _finite(
                preregistered["equal_day_raw_bp_uplift"],
                source="positive-control raw BP",
            ),
            "equal_day_residual_bp_uplift": _finite(
                preregistered["equal_day_residual_bp_uplift"],
                source="positive-control residual BP",
            ),
            "interpretation": reference["interpretation"],
            "lineage_limits": payload.get("lineage_limits"),
        },
    }


def _validate_inference(
    paths: DecisionPaths,
    calendar: Mapping[str, Any],
) -> dict[str, Any]:
    identities = _fingerprint_group(paths.inference_dir, INFERENCE_FILES)
    manifest = _load_json(paths.inference_dir / "run_manifest.json")
    expected = {
        "contract": "fh_neg_rejection_robustness_evaluator/v1",
        "full_cell_id": ANCHOR_ID,
        "effect_zero_support_policy": "undefined_not_zero",
        "raw_trial_count": 8,
        "fwer_method": "circular_moving_block_single_step_max_t_one_sided",
    }
    for key, value in expected.items():
        _require(manifest.get(key) == value, f"inference manifest: {key} drifted")
    _require(
        tuple(manifest.get("cell_ids", ())) == CELL_IDS,
        "inference cell family drifted",
    )
    config = manifest.get("config")
    _require(isinstance(config, Mapping), "inference config is missing")
    frozen_config = {
        "family_id": FAMILY_ID,
        "family_version": FAMILY_VERSION,
        "selected_cell_id": ANCHOR_ID,
        "bootstrap_seed": 20_260_827,
        "bootstrap_reps": 5_000,
        "primary_block_length": 10,
        "family_alpha": 0.10,
    }
    for key, value in frozen_config.items():
        _require(config.get(key) == value, f"inference config: {key} drifted")
    _require(
        manifest.get("parsimony_contract", {}).get("contrasts") == LOO_SPECS,
        "inference manifest parsimony contrast mapping drifted",
    )
    _require(
        {
            str(row.get("contrast_id")): {
                "full_cell_id": row.get("full_cell_id"),
                "reduced_cell_id": row.get("reduced_cell_id"),
                "removed_gate": row.get("removed_gate"),
            }
            for row in manifest.get("loo_pairs", [])
            if isinstance(row, Mapping)
        }
        == LOO_SPECS,
        "inference manifest LOO pairs drifted",
    )
    inference_calendar = pl.read_parquet(
        paths.inference_dir / "requested_calendar.parquet"
    )
    _require_columns(
        inference_calendar,
        ("Date", "evidence_class", "source_complete", "run_status"),
        "inference calendar",
    )
    source_calendar = calendar["frame"].select(
        "Date", "evidence_class", "source_complete", "run_status"
    )
    _require(
        inference_calendar.select(source_calendar.columns).equals(source_calendar),
        "inference calendar disagrees with external calendar",
    )
    input_daily = pl.read_parquet(paths.inference_dir / "input_daily_stats.parquet")
    _require_exact_keys(
        input_daily,
        ("Date", "candidate_id"),
        {
            (date, cell_id)
            for date in calendar["frame"]["Date"].to_list()
            for cell_id in CELL_IDS
        },
        "inference input daily panel",
    )
    runner_dir = paths.study_dir / "artifacts/dgb_factorial_v1"
    runner_daily_path = runner_dir / "daily_stats.parquet"
    runner_manifest_path = runner_dir / "run_manifest.json"
    runner_manifest = _load_json(runner_manifest_path)
    _require(
        runner_manifest.get("contract") == PARENT_CONTRACT
        and runner_manifest.get("family_id") == FAMILY_ID
        and runner_manifest.get("family_version") == FAMILY_VERSION
        and runner_manifest.get("failures") == [],
        "factorial runner identity or failure ledger drifted",
    )
    runner_daily = pl.read_parquet(runner_daily_path)
    _require_columns(runner_daily, input_daily.columns, "factorial runner daily panel")
    runner_slice = runner_daily.filter(
        pl.col("Date").is_in(calendar["frame"]["Date"].to_list())
    ).select(input_daily.columns).sort("Date", "candidate_id")
    _require(
        runner_slice.equals(input_daily.sort("Date", "candidate_id")),
        "inference input daily panel is not the exact verified runner slice",
    )
    identities["factorial_runner_daily_stats.parquet"] = _identity(
        runner_daily_path
    )
    identities["factorial_runner_manifest.json"] = _identity(
        runner_manifest_path
    )

    daily_effects = pl.read_parquet(
        paths.inference_dir / "daily_effects.parquet"
    )
    _require_columns(
        daily_effects,
        (
            "Date",
            "candidate_id",
            "cell_id",
            "condition_n_z",
            "selection_rate",
            "daily_delta_z",
            "condition_active",
            "effect_active",
            "period_status",
        ),
        "inference daily effects",
    )
    _require_exact_keys(
        daily_effects,
        ("Date", "cell_id"),
        {
            (date, cell_id)
            for date in calendar["frame"]["Date"].to_list()
            for cell_id in CELL_IDS
        },
        "inference daily effects",
    )
    _require(
        bool(
            daily_effects.select(
                (pl.col("candidate_id") == pl.col("cell_id")).all()
            ).item()
        ),
        "inference daily candidate/cell ids disagree",
    )
    recomputed_cells: dict[str, dict[str, Any]] = {}
    for cell_id in CELL_IDS:
        subset = daily_effects.filter(pl.col("cell_id") == cell_id).sort("Date")
        active = subset.filter(pl.col("effect_active"))
        values = _finite_values(active["daily_delta_z"].to_list())
        all_values = _finite_values(subset["daily_delta_z"].to_list())
        _require(
            len(values) == len(active) == len(all_values),
            f"inference {cell_id} effect-active mask contradicts finite effects",
        )
        selection = subset.filter(
            pl.col("period_status").is_in(["active", "zero_event"])
        )
        selection_rates = _finite_values(selection["selection_rate"].to_list())
        _require(
            bool(values) and bool(selection_rates),
            f"inference {cell_id} lacks active effects or selection rates",
        )
        recomputed_cells[cell_id] = {
            "calendar_periods": len(subset),
            "effect_active_periods": len(values),
            "condition_support": int(subset["condition_n_z"].sum() or 0),
            "equal_day_delta_z": _mean(values),
            "positive_day_ratio": sum(value > 0.0 for value in values)
            / len(values),
            "selection_rate_mean": _mean(selection_rates),
        }

    cell_summary = pl.read_parquet(paths.inference_dir / "cell_summary.parquet")
    _require_exact_keys(
        cell_summary,
        ("cell_id",),
        {(cell_id,) for cell_id in CELL_IDS},
        "inference cell summary",
    )
    anchor = _row(
        cell_summary,
        pl.col("cell_id") == ANCHOR_ID,
        "inference anchor summary",
    )
    for row in cell_summary.iter_rows(named=True):
        cell_id = str(row["cell_id"])
        recomputed = recomputed_cells[cell_id]
        _require(
            int(row["calendar_periods"]) == recomputed["calendar_periods"]
            and int(row["effect_active_periods"])
            == recomputed["effect_active_periods"]
            and int(row["condition_support"])
            == recomputed["condition_support"],
            f"inference {row['cell_id']} support is invalid",
        )
        for field in (
            "equal_day_delta_z",
            "positive_day_ratio",
            "selection_rate_mean",
        ):
            value = _finite(
                row[field], source=f"inference {row['cell_id']} {field}"
            )
            if field.endswith("ratio") or field == "selection_rate_mean":
                _require(0.0 <= value <= 1.0, f"inference {field} outside [0,1]")
            _require_close(
                value,
                recomputed[field],
                source=f"inference {cell_id} {field}",
            )
    loo = pl.read_parquet(paths.inference_dir / "loo_summary.parquet")
    _require_exact_keys(
        loo,
        ("contrast_id",),
        {(contrast_id,) for contrast_id in LOO_IDS},
        "inference LOO summary",
    )
    for row in loo.iter_rows(named=True):
        expected_loo = LOO_SPECS[str(row["contrast_id"])]
        for field, expected_value in expected_loo.items():
            _require(
                row.get(field) == expected_value,
                f"inference {row['contrast_id']} {field} drifted",
            )
    loo_daily = pl.read_parquet(paths.inference_dir / "loo_daily.parquet")
    _require_columns(
        loo_daily,
        (
            "Date",
            "contrast_id",
            "removed_gate",
            "full_cell_id",
            "reduced_cell_id",
            "full_support",
            "reduced_support",
            "common_active",
            "full_daily_delta_z",
            "reduced_daily_delta_z",
            "paired_increment_delta_z",
        ),
        "inference LOO daily panel",
    )
    _require_exact_keys(
        loo_daily,
        ("Date", "contrast_id"),
        {
            (date, contrast_id)
            for date in calendar["frame"]["Date"].to_list()
            for contrast_id in LOO_IDS
        },
        "inference LOO daily panel",
    )
    loo_by_id = {
        str(row["contrast_id"]): row for row in loo.iter_rows(named=True)
    }
    for contrast_id, spec in LOO_SPECS.items():
        subset = loo_daily.filter(
            pl.col("contrast_id") == contrast_id
        ).sort("Date")
        _require(
            set(subset["removed_gate"].to_list()) == {spec["removed_gate"]}
            and set(subset["full_cell_id"].to_list()) == {ANCHOR_ID}
            and set(subset["reduced_cell_id"].to_list())
            == {spec["reduced_cell_id"]},
            f"inference {contrast_id} daily mapping drifted",
        )
        common = subset.filter(pl.col("common_active"))
        increments = _finite_values(
            common["paired_increment_delta_z"].to_list()
        )
        full_values = _finite_values(common["full_daily_delta_z"].to_list())
        reduced_values = _finite_values(
            common["reduced_daily_delta_z"].to_list()
        )
        _require(
            len(increments) == len(common)
            and len(full_values) == len(common)
            and len(reduced_values) == len(common)
            and all(
                math.isclose(
                    increment,
                    full - reduced,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for increment, full, reduced in zip(
                    increments, full_values, reduced_values, strict=True
                )
            ),
            f"inference {contrast_id} common-active increments drifted",
        )
        summary_row = loo_by_id[contrast_id]
        _require(
            int(summary_row["calendar_periods"]) == len(subset)
            and int(summary_row["common_active_periods"]) == len(common)
            and int(summary_row["full_support_on_common_periods"])
            == int(common["full_support"].sum() or 0)
            and int(summary_row["reduced_support_on_common_periods"])
            == int(common["reduced_support"].sum() or 0),
            f"inference {contrast_id} LOO support summary drifted",
        )
        recomputed_loo = {
            "full_equal_day_delta_z_common": _mean(full_values),
            "reduced_equal_day_delta_z_common": _mean(reduced_values),
            "paired_increment_delta_z": _mean(increments),
            "median_paired_increment_delta_z": _median(increments),
            "paired_increment_std": _sample_std(increments),
            "positive_increment_ratio": sum(value > 0.0 for value in increments)
            / len(increments),
        }
        for field, expected_value in recomputed_loo.items():
            _require_close(
                summary_row[field],
                expected_value,
                source=f"inference {contrast_id} {field}",
            )

    factorial = pl.read_parquet(
        paths.inference_dir / "factorial_contrasts.parquet"
    )
    _require_columns(
        factorial,
        (
            "contrast_id",
            "decision_role",
            "g4_contrast_id",
            "calendar_periods",
            "common_active_periods",
            "paired_contrast_delta_z",
            "median_paired_contrast_delta_z",
            "bootstrap_probability_gt_zero",
            "bootstrap_p_one_sided_positive",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
        ),
        "inference factorial contrasts",
    )
    g4_factorial = factorial.filter(
        pl.col("decision_role") == "frozen_G4_parsimony"
    )
    _require_exact_keys(
        g4_factorial,
        ("g4_contrast_id",),
        {(contrast_id,) for contrast_id in LOO_IDS},
        "inference frozen G4 factorial contrasts",
    )
    for row in g4_factorial.iter_rows(named=True):
        contrast_id = str(row["g4_contrast_id"])
        summary_row = loo_by_id[contrast_id]
        _require(
            int(row["calendar_periods"]) == int(summary_row["calendar_periods"])
            and int(row["common_active_periods"])
            == int(summary_row["common_active_periods"]),
            f"inference {contrast_id} factorial support drifted",
        )
        for factorial_field, loo_field in (
            ("paired_contrast_delta_z", "paired_increment_delta_z"),
            (
                "median_paired_contrast_delta_z",
                "median_paired_increment_delta_z",
            ),
            (
                "bootstrap_probability_gt_zero",
                "bootstrap_probability_gt_zero",
            ),
            (
                "bootstrap_p_one_sided_positive",
                "bootstrap_p_one_sided_positive",
            ),
            ("bootstrap_ci_low", "bootstrap_ci_low"),
            ("bootstrap_ci_high", "bootstrap_ci_high"),
        ):
            _require_close(
                row[factorial_field],
                summary_row[loo_field],
                source=f"inference {contrast_id} factorial {factorial_field}",
            )

    bootstrap = pl.read_parquet(paths.inference_dir / "bootstrap.parquet")
    loo_bootstrap = bootstrap.filter(pl.col("target_kind") == "loo_increment")
    _require_exact_keys(
        loo_bootstrap,
        ("target_id", "block_length_requested"),
        {
            (contrast_id, block_length)
            for contrast_id in LOO_IDS
            for block_length in (5, 10, 20)
        },
        "inference LOO bootstrap",
    )
    _require_columns(
        loo_bootstrap,
        (
            "primary_block_length",
            "calendar_periods",
            "active_periods",
            "point_estimate",
            "bootstrap_repetitions",
            "bootstrap_probability_gt_zero",
            "p_one_sided_positive_centered",
            "ci_low",
            "ci_high",
            "seed",
            "method",
        ),
        "inference LOO bootstrap",
    )
    for row in loo_bootstrap.iter_rows(named=True):
        contrast_id = str(row["target_id"])
        summary_row = loo_by_id[contrast_id]
        block_length = int(row["block_length_requested"])
        _require(
            bool(row["primary_block_length"]) is (block_length == 10)
            and int(row["calendar_periods"])
            == int(summary_row["calendar_periods"])
            and int(row["active_periods"])
            == int(summary_row["common_active_periods"])
            and int(row["bootstrap_repetitions"]) == 5_000
            and int(row["seed"]) == 20_260_827
            and row["method"]
            == "circular_moving_block_percentile_and_centered_test",
            f"inference {contrast_id}/B{block_length} bootstrap metadata drifted",
        )
        _require_close(
            row["point_estimate"],
            summary_row["paired_increment_delta_z"],
            source=f"inference {contrast_id}/B{block_length} point",
        )
        if block_length == 10:
            for bootstrap_field, summary_field in (
                (
                    "bootstrap_probability_gt_zero",
                    "bootstrap_probability_gt_zero",
                ),
                (
                    "p_one_sided_positive_centered",
                    "bootstrap_p_one_sided_positive",
                ),
                ("ci_low", "bootstrap_ci_low"),
                ("ci_high", "bootstrap_ci_high"),
            ):
                _require_close(
                    row[bootstrap_field],
                    summary_row[summary_field],
                    source=(
                        f"inference {contrast_id}/B10 {bootstrap_field}"
                    ),
                )
    fwer = pl.read_parquet(paths.inference_dir / "fwer_audit.parquet")
    _require_exact_keys(
        fwer,
        ("target_id", "block_length_requested"),
        {
            (cell_id, block_length)
            for cell_id in CELL_IDS
            for block_length in (5, 10, 20)
        },
        "inference FWER audit",
    )
    _require_columns(
        fwer,
        (
            "family_id",
            "calendar_periods",
            "active_periods",
            "point_estimate",
            "family_alpha",
            "simultaneous_confidence_level",
            "bootstrap_repetitions",
            "seed",
            "method",
        ),
        "inference FWER audit",
    )
    _require(
        bool(
            fwer.select(
                (
                    pl.col("primary_block_length")
                    == (pl.col("block_length_requested") == 10)
                ).all()
            ).item()
        ),
        "inference FWER primary block flags drifted",
    )
    for row in fwer.iter_rows(named=True):
        cell_id = str(row["target_id"])
        recomputed = recomputed_cells[cell_id]
        block_length = int(row["block_length_requested"])
        _require(
            row["family_id"] == FAMILY_ID
            and int(row["calendar_periods"])
            == recomputed["calendar_periods"]
            and int(row["active_periods"])
            == recomputed["effect_active_periods"]
            and math.isclose(
                float(row["family_alpha"]), 0.10, rel_tol=0.0, abs_tol=0.0
            )
            and math.isclose(
                float(row["simultaneous_confidence_level"]),
                0.90,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            and int(row["bootstrap_repetitions"]) == 5_000
            and int(row["seed"]) == 20_260_827
            and row["method"]
            == "circular_moving_block_single_step_max_t_one_sided",
            f"inference FWER {cell_id}/B{block_length} metadata drifted",
        )
        _require_close(
            row["point_estimate"],
            recomputed["equal_day_delta_z"],
            source=f"inference FWER {cell_id}/B{block_length} point",
        )
        for field in (
            "simultaneous_lower_bound",
            "single_step_fwer_p",
            "dependence_effective_trial_count",
            "joint_valid_bootstrap_rate",
        ):
            value = _finite(
                row[field],
                source=(
                    f"inference FWER {row['target_id']}/"
                    f"B{row['block_length_requested']} {field}"
                ),
            )
            if field in {"single_step_fwer_p", "joint_valid_bootstrap_rate"}:
                _require(0.0 <= value <= 1.0, f"inference FWER {field} outside [0,1]")
        _require(
            int(row["raw_trial_count"]) == 8
            and int(row["joint_valid_bootstrap_repetitions"]) >= 100
            and float(row["joint_valid_bootstrap_rate"]) >= 0.90,
            "inference FWER row failed its joint-valid floor",
        )
    primary_fwer = _row(
        fwer,
        (pl.col("target_id") == ANCHOR_ID)
        & pl.col("primary_block_length"),
        "inference anchor max-stat",
    )
    _require(
        int(primary_fwer["raw_trial_count"]) == 8
        and int(primary_fwer["joint_valid_bootstrap_repetitions"]) >= 100
        and float(primary_fwer["joint_valid_bootstrap_rate"]) >= 0.90,
        "inference max-stat joint support failed closed",
    )
    ledger = pl.read_parquet(paths.inference_dir / "trial_ledger.parquet")
    _require(
        len(ledger) == 8
        and set(ledger["trial_id"].to_list()) == set(CELL_IDS)
        and int(ledger["selected"].sum()) == 1,
        "inference trial ledger is incomplete",
    )
    selected = ledger.filter(pl.col("selected")).row(0, named=True)
    _require(selected["trial_id"] == ANCHOR_ID, "selected anchor drifted")
    _require(
        selected["total_selection_exposure_is_lower_bound"] is True
        and selected["historical_selection_exposure_included"] is False,
        "inference ledger hides historical selection exposure",
    )
    temporal = pl.read_parquet(paths.inference_dir / "temporal_blocks.parquet")
    anchor_temporal = temporal.filter(
        (pl.col("target_kind") == "cell_effect")
        & (pl.col("target_id") == ANCHOR_ID)
    ).sort("block_id")
    _require_exact_keys(
        anchor_temporal,
        ("block_id",),
        {("T01",), ("T02",), ("T03",)},
        "inference temporal blocks",
    )
    _require_columns(
        anchor_temporal,
        (
            "block_start",
            "block_end",
            "calendar_periods",
            "active_periods",
            "support",
            "equal_period_value",
        ),
        "inference temporal blocks",
    )
    dates = [str(value) for value in calendar["frame"]["Date"].to_list()]
    date_blocks = _chronological_blocks(dates, 3)
    anchor_daily = daily_effects.filter(pl.col("cell_id") == ANCHOR_ID)
    for index, block_dates in enumerate(date_blocks, start=1):
        block_id = f"T{index:02d}"
        block_row = _row(
            anchor_temporal,
            pl.col("block_id") == block_id,
            f"inference temporal {block_id}",
        )
        block_daily = anchor_daily.filter(pl.col("Date").is_in(block_dates))
        active = block_daily.filter(pl.col("effect_active"))
        values = _finite_values(active["daily_delta_z"].to_list())
        _require(
            block_row["block_start"] == block_dates[0]
            and block_row["block_end"] == block_dates[-1]
            and int(block_row["calendar_periods"]) == len(block_dates)
            and int(block_row["active_periods"]) == len(values)
            and int(block_row["support"])
            == int(block_daily["condition_n_z"].sum() or 0),
            f"inference temporal {block_id} boundary or support drifted",
        )
        _require_close(
            block_row["equal_period_value"],
            _mean(values),
            source=f"inference temporal {block_id} point",
        )
    parsimony_rows = []
    for row in loo.sort("contrast_id").iter_rows(named=True):
        probability = _finite(
            row["bootstrap_probability_gt_zero"],
            source=f"{row['contrast_id']} bootstrap probability",
        )
        median = _finite(
            row["median_paired_increment_delta_z"],
            source=f"{row['contrast_id']} median increment",
        )
        parsimony_rows.append(
            {
                "contrast_id": row["contrast_id"],
                "removed_gate": row["removed_gate"],
                "full_cell_id": row["full_cell_id"],
                "reduced_cell_id": row["reduced_cell_id"],
                "common_active_periods": int(row["common_active_periods"]),
                "full_support_on_common_periods": int(
                    row["full_support_on_common_periods"]
                ),
                "reduced_support_on_common_periods": int(
                    row["reduced_support_on_common_periods"]
                ),
                "paired_increment_delta_z": float(
                    _finite(
                        row["paired_increment_delta_z"],
                        source=f"{row['contrast_id']} paired increment",
                    )
                ),
                "median_paired_increment_delta_z": median,
                "bootstrap_probability_gt_zero": probability,
                "statistical_rule_pass": probability >= 0.80 and median > 0.0,
            }
        )
        _require(
            0.0 <= probability <= 1.0
            and int(row["common_active_periods"]) > 0
            and int(row["full_support_on_common_periods"]) > 0
            and int(row["reduced_support_on_common_periods"]) > 0,
            f"{row['contrast_id']} probability or support is invalid",
        )
    return {
        "identities": identities,
        "manifest": manifest,
        "anchor": {
            "calendar_periods": int(anchor["calendar_periods"]),
            "active_periods": int(anchor["effect_active_periods"]),
            "active_months": None,
            "events": int(anchor["condition_support"]),
            "equal_day_delta_z": _finite(
                anchor["equal_day_delta_z"], source="inference anchor delta_z"
            ),
            "positive_day_ratio": _finite(
                anchor["positive_day_ratio"], source="inference positive-day ratio"
            ),
            "selection_rate_mean": _finite(
                anchor["selection_rate_mean"], source="inference selection rate"
            ),
            "maxstat_simultaneous_lcb": _finite(
                primary_fwer["simultaneous_lower_bound"],
                source="inference anchor simultaneous LCB",
            ),
            "maxstat_adjusted_p": _finite(
                primary_fwer["single_step_fwer_p"],
                source="inference anchor adjusted p",
            ),
            "raw_trial_count": int(primary_fwer["raw_trial_count"]),
            "effective_trial_count": _finite(
                primary_fwer["dependence_effective_trial_count"],
                source="inference effective trial count",
            ),
        },
        "parsimony": parsimony_rows,
        "temporal_points": anchor_temporal.select(
            "block_id",
            "block_start",
            "block_end",
            "calendar_periods",
            "active_periods",
            "equal_period_value",
        ).to_dicts(),
    }


def _validate_secondary(paths: DecisionPaths) -> dict[str, Any]:
    identities = _fingerprint_group(paths.secondary_dir, SECONDARY_FILES)
    manifest = _load_json(paths.secondary_dir / "run_manifest.json")
    expected = {
        "contract": "fh_neg_rejection_secondary_diagnostics/v1",
        "parent_contract": PARENT_CONTRACT,
        "amendment": (
            "fh_neg_rejection_robustness/fixed_factorial_v1/amendment_2"
        ),
        "family_id": FAMILY_ID,
        "family_version": FAMILY_VERSION,
        "anchor_cell_id": ANCHOR_ID,
        "evidence_class": "retrospective_diagnostic",
    }
    for key, value in expected.items():
        _require(manifest.get(key) == value, f"secondary manifest: {key} drifted")
    scope = manifest.get("decision_scope")
    _require(isinstance(scope, Mapping), "secondary decision scope missing")
    _require(
        scope.get("secondary_diagnostics_only") is True
        and scope.get("condition_selection_performed") is False
        and scope.get("deployment_decision_created") is False,
        "secondary artifact overclaims decision authority",
    )
    config = manifest.get("config")
    _require(isinstance(config, Mapping), "secondary config missing")
    expected_config = {
        "anchor_cell_id": ANCHOR_ID,
        "block_lengths": [5, 10, 20],
        "bootstrap_repetitions": 5_000,
        "bootstrap_seed": 20_260_827,
        "confidence_level": 0.90,
        "hac_lags": [1, 5, 10, 20],
        "primary_block_length": 10,
        "temporal_block_count": 3,
    }
    for key, value in expected_config.items():
        _require(config.get(key) == value, f"secondary config: {key} drifted")
    effect_policy = manifest.get("effect_policy")
    _require(
        isinstance(effect_policy, Mapping)
        and effect_policy.get("inactive_effect") == "undefined_not_zero"
        and effect_policy.get("period_weighting")
        == "equal finite active session",
        "secondary inactive-effect or weighting policy drifted",
    )
    hac_contract = manifest.get("hac")
    _require(
        isinstance(hac_contract, Mapping)
        and hac_contract.get("kernel") == "Bartlett Newey-West"
        and hac_contract.get("lags") == [1, 5, 10, 20]
        and hac_contract.get("role") == "secondary_diagnostic_only",
        "secondary HAC contract drifted",
    )
    input_lineage = manifest.get("input_lineage")
    _require(isinstance(input_lineage, Mapping), "secondary input lineage missing")
    expected_secondary_inputs = {
        "amendment_2": paths.study_dir / "study_contract_amendment_2.json",
        "evaluator_daily_effects": paths.inference_dir / "daily_effects.parquet",
        "evaluator_manifest": paths.inference_dir / "run_manifest.json",
        "evaluator_requested_calendar": (
            paths.inference_dir / "requested_calendar.parquet"
        ),
        "evaluator_temporal_blocks": paths.inference_dir / "temporal_blocks.parquet",
        "runner_daily_stats": (
            paths.study_dir / "artifacts/dgb_factorial_v1/daily_stats.parquet"
        ),
    }
    _require(
        set(input_lineage) == set(expected_secondary_inputs),
        "secondary input-lineage role family drifted",
    )
    verified_inputs = _validate_identity_map(
        input_lineage,
        project_root=paths.project_root,
        source="secondary.input_lineage",
    )
    for role, path in expected_secondary_inputs.items():
        _require(
            verified_inputs[role]["sha256"] == _identity(path)["sha256"],
            f"secondary input-lineage role {role} points to wrong artifact",
        )
    primary_daily = pl.read_parquet(paths.inference_dir / "daily_effects.parquet")
    secondary_daily = pl.read_parquet(
        paths.secondary_dir / "secondary_daily_effects.parquet"
    )
    _require_columns(
        secondary_daily,
        (
            "Date",
            "cell_id",
            "condition_n_z",
            "daily_delta_z",
            "daily_delta_raw_bp",
            "daily_delta_residual_bp",
            "effect_active",
        ),
        "secondary daily effects",
    )
    _require_exact_keys(
        secondary_daily,
        ("Date", "cell_id"),
        {
            (date, cell_id)
            for date in primary_daily["Date"].unique().sort().to_list()
            for cell_id in CELL_IDS
        },
        "secondary daily effects",
    )
    primary_bridge = primary_daily.select(
        "Date",
        "cell_id",
        "condition_n_z",
        "daily_delta_z",
        "effect_active",
    ).sort("Date", "cell_id")
    secondary_bridge = secondary_daily.select(primary_bridge.columns).sort(
        "Date", "cell_id"
    )
    _require(
        primary_bridge.equals(secondary_bridge),
        "secondary daily effects disagree with primary inference panel",
    )
    bp = pl.read_parquet(paths.secondary_dir / "equal_active_bp.parquet")
    _require_exact_keys(
        bp,
        ("cell_id",),
        {(cell_id,) for cell_id in CELL_IDS},
        "secondary BP summary",
    )
    for row in bp.iter_rows(named=True):
        cell_id = str(row["cell_id"])
        active = secondary_daily.filter(
            (pl.col("cell_id") == cell_id) & pl.col("effect_active")
        )
        raw_values = _finite_values(active["daily_delta_raw_bp"].to_list())
        residual_values = _finite_values(
            active["daily_delta_residual_bp"].to_list()
        )
        _require(
            len(raw_values) == len(active) == len(residual_values)
            and int(row["active_sessions"]) == len(active),
            f"secondary {cell_id} BP active support drifted",
        )
        _require_close(
            row["equal_active_raw_bp"],
            _mean(raw_values),
            source=f"secondary {cell_id} equal-active raw BP",
        )
        _require_close(
            row["equal_active_residual_bp"],
            _mean(residual_values),
            source=f"secondary {cell_id} equal-active residual BP",
        )
    anchor_bp = _row(bp, pl.col("cell_id") == ANCHOR_ID, "secondary anchor BP")
    monthly_effects = pl.read_parquet(
        paths.secondary_dir / "monthly_effects.parquet"
    )
    months = sorted(
        {str(value)[:6] for value in secondary_daily["Date"].to_list()}
    )
    _require_exact_keys(
        monthly_effects,
        ("cell_id", "month"),
        {(cell_id, month) for cell_id in CELL_IDS for month in months},
        "secondary monthly effects",
    )
    _require_columns(
        monthly_effects,
        (
            "calendar_sessions",
            "active_sessions",
            "month_mean_delta_z",
            "month_mean_raw_bp",
            "month_mean_residual_bp",
            "month_active",
        ),
        "secondary monthly effects",
    )
    for row in monthly_effects.iter_rows(named=True):
        cell_id = str(row["cell_id"])
        month = str(row["month"])
        detail = secondary_daily.filter(
            (pl.col("cell_id") == cell_id)
            & pl.col("Date").str.starts_with(month)
        )
        active = detail.filter(pl.col("effect_active"))
        _require(
            int(row["calendar_sessions"]) == len(detail)
            and int(row["active_sessions"]) == len(active)
            and bool(row["month_active"]) is (len(active) > 0),
            f"secondary {cell_id}/{month} monthly support drifted",
        )
        if len(active):
            for monthly_field, daily_field in (
                ("month_mean_delta_z", "daily_delta_z"),
                ("month_mean_raw_bp", "daily_delta_raw_bp"),
                ("month_mean_residual_bp", "daily_delta_residual_bp"),
            ):
                _require_close(
                    row[monthly_field],
                    _mean(_finite_values(active[daily_field].to_list())),
                    source=f"secondary {cell_id}/{month} {monthly_field}",
                )
    monthly = pl.read_parquet(paths.secondary_dir / "monthly_summary.parquet")
    _require_exact_keys(
        monthly,
        ("cell_id",),
        {(cell_id,) for cell_id in CELL_IDS},
        "secondary monthly summary",
    )
    for row in monthly.iter_rows(named=True):
        cell_id = str(row["cell_id"])
        detail = monthly_effects.filter(pl.col("cell_id") == cell_id)
        active = detail.filter(pl.col("month_active"))
        positive = active.filter(pl.col("month_mean_delta_z") > 0.0)
        _require(
            int(row.get("requested_months", len(detail))) == len(detail)
            and int(row["active_months"]) == len(active)
            and int(row["positive_months"]) == len(positive),
            f"secondary {cell_id} monthly summary support drifted",
        )
        _require_close(
            row["positive_month_ratio"],
            len(positive) / len(active),
            source=f"secondary {cell_id} positive-month ratio",
        )
    anchor_monthly = _row(
        monthly,
        pl.col("cell_id") == ANCHOR_ID,
        "secondary anchor monthly summary",
    )
    hac = pl.read_parquet(paths.secondary_dir / "hac_diagnostics.parquet")
    _require_exact_keys(
        hac,
        ("cell_id", "requested_lag"),
        {
            (cell_id, lag)
            for cell_id in CELL_IDS
            for lag in (1, 5, 10, 20)
        },
        "secondary HAC diagnostics",
    )
    anchor_hac = hac.filter(pl.col("cell_id") == ANCHOR_ID).sort("requested_lag")
    _require(
        len(anchor_hac) == 4
        and set(anchor_hac["requested_lag"].to_list()) == {1, 5, 10, 20},
        "secondary anchor HAC sensitivity is incomplete",
    )
    _require(
        set(anchor_hac["kernel"].to_list()) == {"Bartlett Newey-West"}
        and set(anchor_hac["inference_role"].to_list())
        == {"secondary_diagnostic_only"},
        "secondary anchor HAC method drifted",
    )
    for row in anchor_hac.iter_rows(named=True):
        for field in (
            "mean_delta_z",
            "hac_standard_error",
            "hac_t_stat",
            "p_one_sided_positive_normal",
            "one_sided_90_lcb",
        ):
            _finite(
                row[field],
                source=f"secondary HAC L{row['requested_lag']} {field}",
            )
    inference_anchor = _row(
        pl.read_parquet(paths.inference_dir / "cell_summary.parquet"),
        pl.col("cell_id") == ANCHOR_ID,
        "secondary inference-anchor crosscheck",
    )
    expected_active_sessions = int(inference_anchor["effect_active_periods"])
    expected_mean_delta_z = _finite(
        inference_anchor["equal_day_delta_z"],
        source="secondary inference-anchor delta_z",
    )
    _require(
        int(anchor_bp["active_sessions"]) == expected_active_sessions
        and int(anchor_monthly["active_months"]) > 0
        and all(
            int(row["active_sessions"]) == expected_active_sessions
            and math.isclose(
                float(row["mean_delta_z"]),
                expected_mean_delta_z,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for row in anchor_hac.iter_rows(named=True)
        ),
        "secondary anchor support or mean disagrees with primary inference",
    )
    intervals = pl.read_parquet(
        paths.secondary_dir / "temporal_basic_intervals.parquet"
    )
    _require_columns(
        intervals,
        (
            "target_id",
            "block_id",
            "block_start",
            "block_end",
            "active_periods",
            "point_delta_z",
            "block_length_requested",
            "primary_block_length",
            "one_sided_90_lcb",
            "one_sided_90_ucb",
            "interval_status",
        ),
        "secondary temporal intervals",
    )
    _require_exact_keys(
        intervals,
        ("target_id", "block_id", "block_length_requested"),
        {
            (ANCHOR_ID, block_id, block_length)
            for block_id in ("T01", "T02", "T03")
            for block_length in (5, 10, 20)
        },
        "secondary temporal intervals",
    )
    for block_length in (5, 10, 20):
        subset = intervals.filter(
            pl.col("block_length_requested") == block_length
        )
        _require(
            len(subset) == 3 and subset["block_id"].n_unique() == 3,
            f"secondary temporal intervals length {block_length} incomplete",
        )
    primary_temporal = pl.read_parquet(
        paths.inference_dir / "temporal_blocks.parquet"
    ).filter(
        (pl.col("target_kind") == "cell_effect")
        & (pl.col("target_id") == ANCHOR_ID)
    )
    _require_exact_keys(
        primary_temporal,
        ("block_id",),
        {("T01",), ("T02",), ("T03",)},
        "secondary primary temporal crosscheck",
    )
    primary_temporal_by_id = {
        str(row["block_id"]): row
        for row in primary_temporal.iter_rows(named=True)
    }
    interval_rows = intervals.sort(
        "block_length_requested", "block_id"
    ).select(
        "block_id",
        "block_start",
        "block_end",
        "active_periods",
        "point_delta_z",
        "block_length_requested",
        "primary_block_length",
        "one_sided_90_lcb",
        "one_sided_90_ucb",
        "interval_status",
    ).to_dicts()
    for row in interval_rows:
        for field in ("point_delta_z", "one_sided_90_lcb", "one_sided_90_ucb"):
            _finite(
                row[field],
                source=(
                    "secondary temporal "
                    f"{row['block_id']}/L{row['block_length_requested']} {field}"
                ),
            )
        _require(
            row["interval_status"] == "ok",
            "secondary temporal interval failed closed",
        )
        source_block = primary_temporal_by_id[str(row["block_id"])]
        _require(
            row["block_start"] == source_block["block_start"]
            and row["block_end"] == source_block["block_end"]
            and int(row["active_periods"]) == int(source_block["active_periods"])
            and math.isclose(
                float(row["point_delta_z"]),
                float(source_block["equal_period_value"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"secondary temporal {row['block_id']} contradicts primary block",
        )
    primary_intervals = intervals.filter(pl.col("primary_block_length")).sort(
        "block_id"
    )
    _require(len(primary_intervals) == 3, "primary temporal intervals incomplete")
    _require(
        bool((primary_intervals["interval_status"] == "ok").all()),
        "primary temporal interval failed closed",
    )
    return {
        "identities": identities,
        "verified_inputs": verified_inputs,
        "manifest": manifest,
        "anchor": {
            "active_sessions": int(anchor_bp["active_sessions"]),
            "equal_active_raw_bp": _finite(
                anchor_bp["equal_active_raw_bp"],
                source="secondary anchor raw BP",
            ),
            "equal_active_residual_bp": _finite(
                anchor_bp["equal_active_residual_bp"],
                source="secondary anchor residual BP",
            ),
            "active_months": int(anchor_monthly["active_months"]),
            "positive_months": int(anchor_monthly["positive_months"]),
            "positive_month_ratio": _finite(
                anchor_monthly["positive_month_ratio"],
                source="secondary positive-month ratio",
            ),
        },
        "hac": anchor_hac.select(
            "requested_lag",
            "active_sessions",
            "mean_delta_z",
            "hac_standard_error",
            "hac_t_stat",
            "p_one_sided_positive_normal",
            "one_sided_90_lcb",
        ).to_dicts(),
        "temporal_intervals": interval_rows,
    }


def _validate_artifact_oos(paths: DecisionPaths) -> dict[str, Any]:
    identities = _fingerprint_group(paths.artifact_oos_dir, ARTIFACT_OOS_FILES)
    manifest = _load_json(paths.artifact_oos_dir / "run_manifest.json")
    expected = {
        "contract": "fh_neg_rejection_artifact_oos_descriptive/v1",
        "parent_contract": PARENT_CONTRACT,
        "family_id": FAMILY_ID,
        "family_version": FAMILY_VERSION,
        "Date": "20260826",
        "evidence_class": "artifact_oos_non_pristine",
        "calendar_artifact": "calendar_artifact_oos_v1",
    }
    for key, value in expected.items():
        _require(manifest.get(key) == value, f"artifact-OOS: {key} drifted")
    scope = manifest.get("analysis_scope")
    separation = manifest.get("separation_policy")
    _require(
        isinstance(scope, Mapping) and isinstance(separation, Mapping),
        "artifact-OOS scope is missing",
    )
    for field in (
        "hac_performed",
        "bootstrap_performed",
        "p_value_computed",
        "confidence_interval_computed",
        "inference_performed",
        "decision_created",
    ):
        _require(scope.get(field) is False, f"artifact-OOS {field} must be false")
    _require(
        scope.get("descriptive_only") is True
        and
        separation.get("retrospective_pooled") is False
        and separation.get("one_session_only") is True,
        "artifact-OOS was not kept separate",
    )
    _require(
        separation.get("retrospective_outputs_referenced") is False,
        "artifact-OOS references retrospective outputs",
    )
    input_lineage = manifest.get("input_lineage")
    _require(isinstance(input_lineage, Mapping), "artifact-OOS lineage missing")
    expected_oos_inputs = {
        "amendment_2": paths.study_dir / "study_contract_amendment_2.json",
        "artifact_oos_calendar": (
            paths.study_dir
            / "artifacts/calendar_artifact_oos_v1/requested_calendar.parquet"
        ),
        "runner_daily_stats": (
            paths.study_dir / "artifacts/dgb_factorial_v1/daily_stats.parquet"
        ),
    }
    _require(
        set(input_lineage) == set(expected_oos_inputs),
        "artifact-OOS input-lineage role family drifted",
    )
    verified_inputs = _validate_identity_map(
        input_lineage,
        project_root=paths.project_root,
        source="artifact_oos.input_lineage",
    )
    for role, path in expected_oos_inputs.items():
        _require(
            verified_inputs[role]["sha256"] == _identity(path)["sha256"],
            f"artifact-OOS input-lineage role {role} points to wrong artifact",
        )
    cells = pl.read_parquet(paths.artifact_oos_dir / "cell_descriptive.parquet")
    _require_columns(
        cells,
        (
            "Date",
            "evidence_class",
            "cell_id",
            "condition_n_z",
            "control_n_z",
            "daily_delta_z",
            "daily_delta_raw_bp",
            "daily_delta_residual_bp",
            "effect_defined",
            "analysis_role",
            "inference_performed",
        ),
        "artifact-OOS cell table",
    )
    _require_exact_keys(
        cells,
        ("cell_id",),
        {(cell_id,) for cell_id in CELL_IDS},
        "artifact-OOS cell table",
    )
    _require(
        set(cells["Date"].to_list()) == {"20260826"}
        and set(cells["evidence_class"].to_list())
        == {"artifact_oos_non_pristine"}
        and bool(cells["effect_defined"].all())
        and not bool(cells["inference_performed"].any()),
        "artifact-OOS cell table violates one-session descriptive scope",
    )
    _require(
        set(cells["analysis_role"].to_list()) == {"one_session_descriptive_only"},
        "artifact-OOS cell analysis role drifted",
    )
    for row in cells.iter_rows(named=True):
        _require(
            int(row["condition_n_z"]) > 0 and int(row["control_n_z"]) > 0,
            f"artifact-OOS {row['cell_id']} support is invalid",
        )
        for field in (
            "daily_delta_z",
            "daily_delta_raw_bp",
            "daily_delta_residual_bp",
        ):
            _finite(
                row[field], source=f"artifact-OOS {row['cell_id']} {field}"
            )
    anchor = _row(cells, pl.col("cell_id") == ANCHOR_ID, "artifact-OOS anchor")
    loo = pl.read_parquet(paths.artifact_oos_dir / "loo_descriptive.parquet")
    _require_columns(
        loo,
        (
            "Date",
            "evidence_class",
            "contrast_id",
            "removed_gate",
            "full_condition_n_z",
            "reduced_condition_n_z",
            "full_minus_loo_z",
            "full_minus_loo_raw_bp",
            "full_minus_loo_residual_bp",
            "common_effect_defined",
            "analysis_role",
            "inference_performed",
        ),
        "artifact-OOS LOO table",
    )
    _require_exact_keys(
        loo,
        ("contrast_id",),
        {(contrast_id,) for contrast_id in LOO_IDS},
        "artifact-OOS LOO table",
    )
    _require(
        set(loo["Date"].to_list()) == {"20260826"}
        and set(loo["evidence_class"].to_list())
        == {"artifact_oos_non_pristine"}
        and bool(loo["common_effect_defined"].all())
        and not bool(loo["inference_performed"].any()),
        "artifact-OOS LOO table violates one-session descriptive scope",
    )
    for row in loo.iter_rows(named=True):
        expected_loo = LOO_SPECS[str(row["contrast_id"])]
        for field, expected_value in expected_loo.items():
            _require(
                row.get(field) == expected_value,
                f"artifact-OOS {row['contrast_id']} {field} drifted",
            )
        for field in (
            "full_minus_loo_z",
            "full_minus_loo_raw_bp",
            "full_minus_loo_residual_bp",
        ):
            _finite(
                row[field],
                source=f"artifact-OOS {row['contrast_id']} {field}",
            )
    return {
        "identities": identities,
        "verified_inputs": verified_inputs,
        "manifest": manifest,
        "anchor": {
            "Date": "20260826",
            "condition_n_z": int(anchor["condition_n_z"]),
            "control_n_z": int(anchor["control_n_z"]),
            "daily_delta_z": _finite(
                anchor["daily_delta_z"], source="artifact-OOS anchor delta_z"
            ),
            "daily_delta_raw_bp": _finite(
                anchor["daily_delta_raw_bp"], source="artifact-OOS anchor raw BP"
            ),
            "daily_delta_residual_bp": _finite(
                anchor["daily_delta_residual_bp"],
                source="artifact-OOS anchor residual BP",
            ),
            "analysis_role": anchor["analysis_role"],
        },
        "loo": loo.select(
            "contrast_id",
            "removed_gate",
            "full_condition_n_z",
            "reduced_condition_n_z",
            "full_minus_loo_z",
            "full_minus_loo_raw_bp",
            "full_minus_loo_residual_bp",
        ).sort("contrast_id").to_dicts(),
    }


def _validate_historical_exposure(paths: DecisionPaths) -> dict[str, Any]:
    identity = _identity(paths.historical_exposure)
    payload = _load_json(paths.historical_exposure)
    _require(
        payload.get("contract") == "fh_neg_rejection_historical_trial_exposure/v1",
        "historical exposure contract drifted",
    )
    inventory = payload.get("arithmetic_inventory_lower_bounds")
    implication = payload.get("implication_for_current_study")
    _require(
        isinstance(inventory, Mapping) and isinstance(implication, Mapping),
        "historical exposure summary is incomplete",
    )
    _require(
        inventory.get("valid_single_delta_z_max_stat_family") is False,
        "historical families were incorrectly pooled into one max-stat family",
    )
    _require(
        implication.get("historical_exposure_status")
        == "material and incompletely bounded above"
        and int(implication["new_dgb_factorial_current_family_size"]) == 8,
        "historical exposure implication drifted",
    )
    expected_lower_bounds = {
        "condition_selector_configurations_across_distinct_historical_families": 28,
        "condition_direction_tests_across_distinct_historical_families": 46,
        "condition_label_outcome_tests_across_distinct_historical_families": 51,
    }
    for key, expected in expected_lower_bounds.items():
        _require(
            int(inventory.get(key, -1)) == expected,
            f"historical exposure lower bound {key} drifted",
        )
    generator = payload.get("generator")
    _require(isinstance(generator, Mapping), "historical generator identity missing")
    verified: dict[str, dict[str, Any]] = {
        "generator": _validate_recorded_identity(
            generator,
            project_root=paths.project_root,
            source="historical.generator",
        )
    }
    families = payload.get("families")
    _require(isinstance(families, list) and families, "historical families missing")
    family_ids: set[str] = set()
    for family in families:
        _require(isinstance(family, Mapping), "historical family is not an object")
        family_id = str(family.get("family_id"))
        _require(family_id not in family_ids, "historical family id is duplicated")
        family_ids.add(family_id)
        sources = family.get("sources")
        _require(isinstance(sources, list) and sources, f"{family_id}: sources missing")
        for index, record in enumerate(sources):
            _require(isinstance(record, Mapping), f"{family_id}: bad source record")
            key = f"{family_id}/{index}"
            verified[key] = _validate_recorded_identity(
                record,
                project_root=paths.project_root,
                source=f"historical.{key}",
            )
    return {
        "identity": identity,
        "verified_sources": verified,
        "summary": {
            "historical_exposure_status": implication[
                "historical_exposure_status"
            ],
            "condition_selector_configurations_lower_bound": int(
                inventory[
                    "condition_selector_configurations_across_distinct_historical_families"
                ]
            ),
            "condition_direction_tests_lower_bound": int(
                inventory[
                    "condition_direction_tests_across_distinct_historical_families"
                ]
            ),
            "condition_label_outcome_tests_lower_bound": int(
                inventory[
                    "condition_label_outcome_tests_across_distinct_historical_families"
                ]
            ),
            "valid_single_delta_z_max_stat_family": False,
            "current_family_size": 8,
        },
    }


def _plateau_input_hashes(paths: DecisionPaths) -> dict[str, str]:
    return {
        "daily_aggregates": _sha256_file(
            paths.plateau_run_dir / "daily_stats.parquet"
        ),
        "plateau_run_manifest": _sha256_file(
            paths.plateau_run_dir / "run_manifest.json"
        ),
        "requested_calendar": _sha256_file(
            paths.calendar_dir / "requested_calendar.parquet"
        ),
        "requested_calendar_manifest": _sha256_file(
            paths.calendar_dir / "requested_calendar.json"
        ),
        "threshold_contract": _sha256_file(
            paths.study_dir / "threshold_plateau_contract.json"
        ),
    }


def _validate_plateau(
    paths: DecisionPaths,
    calendar: Mapping[str, Any],
) -> dict[str, Any]:
    run_identities = _fingerprint_group(paths.plateau_run_dir, PLATEAU_RUN_FILES)
    evaluation_identities = _fingerprint_group(
        paths.plateau_evaluation_dir, PLATEAU_EVALUATION_FILES
    )
    threshold_contract_path = paths.study_dir / "threshold_plateau_contract.json"
    threshold_contract_identity = _identity(threshold_contract_path)
    threshold_contract = _load_json(threshold_contract_path)
    _require(
        threshold_contract.get("contract_id") == PLATEAU_CONTRACT
        and threshold_contract.get("production_promotion_authorized") is False
        and threshold_contract.get("evidence_class") == "retrospective_diagnostic",
        "threshold plateau contract drifted",
    )
    _require(
        threshold_contract.get("date_range") == ["20250211", "20260825"]
        and threshold_contract.get("fixed_grid")
        == {
            "depth_ratio_lte": [-6, -8, -10],
            "gap_ticks_gte": [3, 4, 5],
            "bear_score_gte": [1, 2, 3],
            "cartesian_cells": 27,
            "selection_rule": (
                "Do not rank cells or replace the center. Every cell and "
                "failed/zero-support period remains in the ledger."
            ),
        },
        "threshold plateau fixed grid or date range drifted",
    )
    plateau_inference = threshold_contract.get("inference")
    _require(
        isinstance(plateau_inference, Mapping)
        and plateau_inference.get("circular_block_lengths") == [10, 5, 20]
        and plateau_inference.get("bootstrap_seed") == 20_260_827
        and plateau_inference.get("bootstrap_repetitions") == 5_000
        and plateau_inference.get("alpha") == 0.10
        and plateau_inference.get("family_correction")
        == "single-step joint-valid max-stat across all 27 fixed cells",
        "threshold plateau inference contract drifted",
    )
    run_manifest = _load_json(paths.plateau_run_dir / "run_manifest.json")
    expected_run = {
        "contract": PLATEAU_CONTRACT,
        "family_id": PLATEAU_FAMILY_ID,
        "family_version": PLATEAU_FAMILY_VERSION,
        "center_candidate_id": PLATEAU_CENTER_ID,
        "evidence_class": "retrospective_diagnostic",
        "outcome_informed_freeze": True,
        "production_promotion_authorized": False,
        "step": 1,
        "start": "20250211",
        "end": "20260825",
        "frozen_contract_sha256": threshold_contract_identity["sha256"],
    }
    for key, value in expected_run.items():
        _require(run_manifest.get(key) == value, f"plateau run: {key} drifted")
    _require(
        tuple(run_manifest.get("candidate_ids", ())) == PLATEAU_CELL_IDS,
        "plateau run candidate family drifted",
    )
    _require(run_manifest.get("failures") == [], "plateau run contains failures")
    _require(
        run_manifest.get("threshold_design_fingerprint")
        == EXPECTED_PLATEAU_DESIGN_FINGERPRINT,
        "plateau run design fingerprint drifted",
    )
    _require(
        run_manifest.get("sessions") == calendar["frame"]["Date"].to_list(),
        "plateau run session calendar drifted",
    )
    plateau_daily = pl.read_parquet(paths.plateau_run_dir / "daily_stats.parquet")
    _require_columns(
        plateau_daily,
        ("Date", "candidate_id"),
        "plateau runner daily panel",
    )
    _require(
        len(plateau_daily)
        == int(calendar["summary"]["requested_sessions"]) * 27,
        "plateau daily panel is incomplete",
    )
    _require(
        set(plateau_daily["candidate_id"].to_list()) == set(PLATEAU_CELL_IDS),
        "plateau daily candidate ids drifted",
    )
    _require(
        plateau_daily.select("Date", "candidate_id").unique().height
        == len(plateau_daily)
        and set(plateau_daily["Date"].to_list())
        == set(calendar["frame"]["Date"].to_list()),
        "plateau daily candidate-calendar keys are incomplete or duplicated",
    )

    evaluator_manifest = _load_json(
        paths.plateau_evaluation_dir / "run_manifest.json"
    )
    expected_evaluator = {
        "contract": PLATEAU_EVALUATOR_CONTRACT,
        "plateau_contract_id": PLATEAU_CONTRACT,
        "family_id": PLATEAU_FAMILY_ID,
        "family_version": PLATEAU_FAMILY_VERSION,
        "center_candidate_id": PLATEAU_CENTER_ID,
        "frozen_contract_sha256": threshold_contract_identity["sha256"],
        "threshold_design_fingerprint": run_manifest[
            "threshold_design_fingerprint"
        ],
        "decision_contract": PLATEAU_DECISION_CONTRACT,
        "estimand": "shared_pool_equal_active_session_delta_z",
        "zero_event_policy": "undefined_not_zero",
        "production_promotion_authorized": False,
    }
    for key, value in expected_evaluator.items():
        _require(
            evaluator_manifest.get(key) == value,
            f"plateau evaluator: {key} drifted",
        )
    _require(
        tuple(evaluator_manifest.get("candidate_ids", ())) == PLATEAU_CELL_IDS,
        "plateau evaluator candidate family drifted",
    )
    input_hashes = evaluator_manifest.get("input_artifact_sha256")
    _require(isinstance(input_hashes, Mapping), "plateau evaluator input hashes missing")
    _require(
        tuple(sorted(input_hashes)) == tuple(sorted(PLATEAU_INPUT_HASH_KEYS)),
        "plateau evaluator input hash keys drifted",
    )
    actual_input_hashes = _plateau_input_hashes(paths)
    _require(
        dict(input_hashes) == actual_input_hashes,
        "plateau evaluator input hashes drifted",
    )
    output_artifacts = evaluator_manifest.get("output_artifacts")
    _require(
        isinstance(output_artifacts, Mapping) and output_artifacts,
        "plateau evaluator output identities missing",
    )
    expected_output_names = set(PLATEAU_EVALUATION_FILES) - {
        "run_manifest.json"
    }
    _require(
        set(output_artifacts) == expected_output_names,
        "plateau evaluator output identity set drifted",
    )
    for name, record in output_artifacts.items():
        _require(
            name in PLATEAU_EVALUATION_FILES and name != "run_manifest.json",
            f"plateau evaluator records unknown output {name}",
        )
        _require(isinstance(record, Mapping), f"plateau output {name} identity bad")
        actual = evaluation_identities[name]
        _require(
            record.get("sha256") == actual["sha256"]
            and int(record.get("size", -1)) == actual["size"],
            f"plateau output {name} identity drifted",
        )

    evaluation_daily = pl.read_parquet(
        paths.plateau_evaluation_dir / "daily_effects.parquet"
    )
    _require_columns(
        evaluation_daily,
        (
            "Date",
            "candidate_id",
            "condition_n_z",
            "realized_selection_rate",
            "effect_active",
            "daily_delta_z",
            "plateau_design_fingerprint",
        ),
        "plateau evaluator daily panel",
    )
    _require_exact_keys(
        evaluation_daily,
        ("Date", "candidate_id"),
        {
            (date, candidate_id)
            for date in calendar["frame"]["Date"].to_list()
            for candidate_id in PLATEAU_CELL_IDS
        },
        "plateau evaluator daily panel",
    )
    _require(
        set(evaluation_daily["plateau_design_fingerprint"].to_list())
        == {EXPECTED_PLATEAU_DESIGN_FINGERPRINT},
        "plateau daily design fingerprint drifted",
    )
    plateau_recomputed: dict[tuple[str, str], dict[str, Any]] = {}
    for period in ("full", "2025", "2026"):
        period_daily = (
            evaluation_daily
            if period == "full"
            else evaluation_daily.filter(pl.col("Date").str.starts_with(period))
        )
        for candidate_id in PLATEAU_CELL_IDS:
            subset = period_daily.filter(
                pl.col("candidate_id") == candidate_id
            )
            active = subset.filter(pl.col("effect_active"))
            values = _finite_values(active["daily_delta_z"].to_list())
            rates = _finite_values(
                subset["realized_selection_rate"].to_list()
            )
            _require(
                len(values) == len(active) and bool(rates),
                f"plateau {period}/{candidate_id} active mask or prevalence drifted",
            )
            plateau_recomputed[(period, candidate_id)] = {
                "calendar_sessions": len(subset),
                "active_sessions": len(active),
                "effect_valid_events": int(
                    subset["condition_n_z"].sum() or 0
                ),
                "equal_active_session_delta_z": (
                    _mean(values) if values else None
                ),
                "selection_rate_equal_session_mean": _mean(rates),
            }

    cell_summary = pl.read_parquet(
        paths.plateau_evaluation_dir / "cell_summary.parquet"
    )
    _require_exact_keys(
        cell_summary,
        ("candidate_id",),
        {(candidate_id,) for candidate_id in PLATEAU_CELL_IDS},
        "plateau cell summary",
    )
    _require_columns(
        cell_summary,
        (
            "active_sessions",
            "effect_valid_events",
            "calendar_sessions",
            "equal_active_session_delta_z",
            "selection_rate_equal_session_mean",
            "support_floor_pass",
        ),
        "plateau cell summary",
    )
    support_flags: list[bool] = []
    for row in cell_summary.iter_rows(named=True):
        candidate_id = str(row["candidate_id"])
        recomputed = plateau_recomputed[("full", candidate_id)]
        active_sessions = int(row["active_sessions"])
        events = int(row["effect_valid_events"])
        _require(
            row.get("period") == "full"
            and int(row["calendar_sessions"])
            == recomputed["calendar_sessions"]
            and active_sessions == recomputed["active_sessions"]
            and events == recomputed["effect_valid_events"],
            f"plateau {candidate_id} summary support drifted",
        )
        computed = active_sessions >= 60 and events >= 1_000
        _require(
            row["support_floor_pass"] is computed,
            f"plateau {row['candidate_id']} support flag contradicts support",
        )
        expected_effect = recomputed["equal_active_session_delta_z"]
        if expected_effect is None:
            _require(
                row["equal_active_session_delta_z"] is None,
                f"plateau {candidate_id} non-estimable effect became finite",
            )
        else:
            _require_close(
                row["equal_active_session_delta_z"],
                expected_effect,
                source=f"plateau {candidate_id} effect",
            )
        rate = _finite(
            row["selection_rate_equal_session_mean"],
            source=f"plateau {row['candidate_id']} prevalence",
        )
        _require(0.0 <= rate <= 1.0, "plateau prevalence is outside [0,1]")
        _require_close(
            rate,
            recomputed["selection_rate_equal_session_mean"],
            source=f"plateau {candidate_id} prevalence",
        )
        support_flags.append(computed)
    computed_p1 = all(support_flags)

    year_summary = pl.read_parquet(
        paths.plateau_evaluation_dir / "year_summary.parquet"
    )
    _require_exact_keys(
        year_summary,
        ("period", "candidate_id"),
        {
            (period, candidate_id)
            for period in ("2025", "2026")
            for candidate_id in PLATEAU_CELL_IDS
        },
        "plateau year summary",
    )
    _require_columns(
        year_summary,
        (
            "calendar_sessions",
            "active_sessions",
            "effect_valid_events",
            "equal_active_session_delta_z",
            "selection_rate_equal_session_mean",
        ),
        "plateau year summary",
    )
    for row in year_summary.iter_rows(named=True):
        period = str(row["period"])
        candidate_id = str(row["candidate_id"])
        recomputed = plateau_recomputed[(period, candidate_id)]
        _require(
            int(row["calendar_sessions"])
            == recomputed["calendar_sessions"]
            and int(row["active_sessions"]) == recomputed["active_sessions"]
            and int(row["effect_valid_events"])
            == recomputed["effect_valid_events"],
            f"plateau {period}/{candidate_id} year support drifted",
        )
        expected_effect = recomputed["equal_active_session_delta_z"]
        if expected_effect is None:
            _require(
                row["equal_active_session_delta_z"] is None,
                f"plateau {period}/{candidate_id} non-estimable effect became finite",
            )
        else:
            _require_close(
                row["equal_active_session_delta_z"],
                expected_effect,
                source=f"plateau {period}/{candidate_id} effect",
            )
        _require_close(
            row["selection_rate_equal_session_mean"],
            recomputed["selection_rate_equal_session_mean"],
            source=f"plateau {period}/{candidate_id} prevalence",
        )
    fwer = pl.read_parquet(paths.plateau_evaluation_dir / "fwer_audit.parquet")
    _require_exact_keys(
        fwer,
        ("candidate_id", "block_length_requested"),
        {
            (candidate_id, block_length)
            for candidate_id in PLATEAU_CELL_IDS
            for block_length in (5, 10, 20)
        },
        "plateau FWER audit",
    )
    _require_columns(
        fwer,
        (
            "primary_block_length",
            "calendar_sessions",
            "active_sessions",
            "point_estimate",
            "family_alpha",
            "simultaneous_confidence_level",
            "raw_trial_count",
            "bootstrap_repetitions",
            "joint_valid_bootstrap_repetitions",
            "joint_valid_bootstrap_rate",
            "seed",
            "method",
        ),
        "plateau FWER audit",
    )
    for row in fwer.iter_rows(named=True):
        candidate_id = str(row["candidate_id"])
        block_length = int(row["block_length_requested"])
        recomputed = plateau_recomputed[("full", candidate_id)]
        _require(
            bool(row["primary_block_length"]) is (block_length == 10)
            and int(row["calendar_sessions"])
            == recomputed["calendar_sessions"]
            and int(row["active_sessions"]) == recomputed["active_sessions"]
            and math.isclose(
                float(row["family_alpha"]), 0.10, rel_tol=0.0, abs_tol=0.0
            )
            and math.isclose(
                float(row["simultaneous_confidence_level"]),
                0.90,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            and int(row["raw_trial_count"]) == 27
            and int(row["bootstrap_repetitions"]) == 5_000
            and int(row["joint_valid_bootstrap_repetitions"]) >= 100
            and float(row["joint_valid_bootstrap_rate"]) >= 0.90
            and int(row["seed"]) == 20_260_827
            and row["method"]
            == "circular_moving_block_single_step_max_t_one_sided",
            f"plateau FWER {candidate_id}/B{block_length} metadata drifted",
        )
        _require_close(
            row["point_estimate"],
            recomputed["equal_active_session_delta_z"],
            source=f"plateau FWER {candidate_id}/B{block_length} point",
        )
    center_fwer = fwer.filter(pl.col("candidate_id") == PLATEAU_CENTER_ID)
    center_bounds = {
        str(row["block_length_requested"]): _finite(
            row["simultaneous_lower_bound"],
            source=f"plateau center LCB B{row['block_length_requested']}",
        )
        for row in center_fwer.iter_rows(named=True)
    }
    computed_p2 = set(center_bounds) == {"5", "10", "20"} and all(
        value > 0.0 for value in center_bounds.values()
    )

    neighbors = pl.read_parquet(
        paths.plateau_evaluation_dir / "neighbor_diagnostics.parquet"
    )
    _require_exact_keys(
        neighbors,
        ("period", "neighbor_id"),
        {
            (period, neighbor_id)
            for period in ("full", "2025", "2026")
            for neighbor_id in PLATEAU_NEIGHBOR_IDS
        },
        "plateau neighbor diagnostics",
    )
    neighbor_pass_by_period: dict[str, bool] = {}
    for period in ("full", "2025", "2026"):
        period_neighbors = neighbors.filter(pl.col("period") == period)
        center_values = period_neighbors["center_effect"].unique().to_list()
        _require(len(center_values) == 1, f"plateau {period} center effect drifted")
        center_effect = _finite(
            center_values[0], source=f"plateau {period} center effect"
        )
        _require_close(
            center_effect,
            plateau_recomputed[(period, PLATEAU_CENTER_ID)][
                "equal_active_session_delta_z"
            ],
            source=f"plateau {period} neighbor-table center effect",
        )
        effects: list[float] = []
        for neighbor_row in period_neighbors.iter_rows(named=True):
            neighbor_id = str(neighbor_row["neighbor_id"])
            effect = _finite(
                neighbor_row["neighbor_effect"],
                source=f"plateau {period}/{neighbor_id} neighbor effect",
            )
            _require_close(
                effect,
                plateau_recomputed[(period, neighbor_id)][
                    "equal_active_session_delta_z"
                ],
                source=f"plateau {period}/{neighbor_id} neighbor-table effect",
            )
            effects.append(effect)
        qualifies = [
            effect > 0.0 and effect >= 0.50 * center_effect
            for effect in effects
        ]
        ordered = sorted(effects)
        median_effect = (ordered[2] + ordered[3]) / 2.0
        computed_period_pass = (
            center_effect > 0.0
            and sum(qualifies) >= 5
            and median_effect >= 0.75 * center_effect
        )
        _require(
            period_neighbors["neighbor_qualifies"].to_list() == qualifies,
            f"plateau {period} neighbor qualification flags drifted",
        )
        _require(
            set(period_neighbors["period_local_sign_and_scale_pass"].to_list())
            == {computed_period_pass},
            f"plateau {period} local pass flag drifted",
        )
        neighbor_pass_by_period[period] = computed_period_pass
    computed_p3 = neighbor_pass_by_period["full"]

    monotonicity = pl.read_parquet(
        paths.plateau_evaluation_dir / "monotonicity.parquet"
    )
    _require_exact_keys(
        monotonicity,
        ("period", "factor"),
        {
            (period, factor)
            for period in ("full", "2025", "2026")
            for factor in (
                "depth_ratio_lte",
                "gap_ticks_gte",
                "bear_score_gte",
            )
        },
        "plateau monotonicity",
    )
    factor_failures = [
        {"period": row["period"], "factor": row["factor"]}
        for row in monotonicity.iter_rows(named=True)
        if row["factor_level_non_decreasing"] is not True
    ]
    slice_reversals = pl.read_parquet(
        paths.plateau_evaluation_dir / "slice_reversals.parquet"
    )
    _require(len(slice_reversals) == 162, "plateau slice table row count drifted")
    _require(
        set(slice_reversals["period"].to_list()) == {"full", "2025", "2026"}
        and set(slice_reversals["factor"].to_list())
        == {"depth_ratio_lte", "gap_ticks_gte", "bear_score_gte"}
        and set(slice_reversals["transition_index"].to_list()) == {0, 1},
        "plateau slice comparison domains drifted",
    )
    _require_exact_keys(
        slice_reversals,
        ("period", "factor", "slice_id", "transition_index"),
        set(
            tuple(row)
            for row in slice_reversals.select(
                "period", "factor", "slice_id", "transition_index"
            ).iter_rows()
        ),
        "plateau slice comparisons",
    )
    reversal_count = int(
        slice_reversals.filter(pl.col("slice_reversal") == True).height  # noqa: E712
    )
    evaluation_calendar = pl.read_parquet(
        paths.plateau_evaluation_dir / "requested_calendar.parquet"
    )
    calendar_columns = (
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
    _require_columns(evaluation_calendar, calendar_columns, "plateau calendar")
    _require_columns(calendar["frame"], calendar_columns, "stage1 calendar")
    _require(
        evaluation_calendar.select(calendar_columns).equals(
            calendar["frame"].select(calendar_columns)
        ),
        "plateau evaluator calendar disagrees with stage1 calendar",
    )
    evaluation_daily = pl.read_parquet(
        paths.plateau_evaluation_dir / "daily_effects.parquet"
    )
    _require_columns(
        evaluation_daily,
        ("Date", "candidate_id"),
        "plateau evaluator daily panel",
    )
    _require(
        len(evaluation_daily) == len(calendar["frame"]) * 27,
        "plateau evaluator daily panel is incomplete",
    )
    _require(
        evaluation_daily.select("Date", "candidate_id").unique().height
        == len(evaluation_daily)
        and set(evaluation_daily["candidate_id"].to_list())
        == set(PLATEAU_CELL_IDS)
        and set(evaluation_daily["Date"].to_list())
        == set(calendar["frame"]["Date"].to_list()),
        "plateau evaluator candidate-calendar keys are incomplete or duplicated",
    )
    ledger = pl.read_parquet(paths.plateau_evaluation_dir / "trial_ledger.parquet")
    trial_column = "trial_id" if "trial_id" in ledger.columns else "candidate_id"
    _require_exact_keys(
        ledger,
        (trial_column,),
        {(candidate_id,) for candidate_id in PLATEAU_CELL_IDS},
        "plateau trial ledger",
    )
    decision = _load_json(
        paths.plateau_evaluation_dir / "plateau_decision.json"
    )
    expected_decision = {
        "contract": PLATEAU_DECISION_CONTRACT,
        "plateau_contract_id": PLATEAU_CONTRACT,
        "family_id": PLATEAU_FAMILY_ID,
        "family_version": PLATEAU_FAMILY_VERSION,
        "center_candidate_id": PLATEAU_CENTER_ID,
        "production_promotion_authorized": False,
    }
    for key, value in expected_decision.items():
        _require(decision.get(key) == value, f"plateau decision: {key} drifted")
    decision_hashes = decision.get("input_artifact_sha256")
    _require(
        isinstance(decision_hashes, Mapping)
        and dict(decision_hashes) == actual_input_hashes,
        "plateau decision input hashes drifted",
    )
    plateau_pass = decision.get("diagnostic_plateau_pass")
    _require(isinstance(plateau_pass, bool), "plateau pass flag is missing")
    expected_status = (
        "diagnostic_plateau_pass" if plateau_pass else "diagnostic_plateau_fail"
    )
    _require(
        decision.get("decision_status") == expected_status,
        "plateau decision status contradicts pass flag",
    )
    for field in ("failed_gate_ids", "gates", "mandatory_disclosures", "warnings"):
        _require(field in decision, f"plateau decision lacks {field}")
    gates = decision["gates"]
    _require(
        isinstance(gates, Mapping) and set(gates) == set(PLATEAU_GATE_IDS),
        "plateau explicit gate family drifted",
    )
    computed_gate_pass = {
        "P1_support_floor_each_cell": computed_p1,
        "P2_center_family_uncertainty": computed_p2,
        "P3_full_period_local_sign_and_scale": computed_p3,
    }
    for gate_id, computed_pass in computed_gate_pass.items():
        gate = gates[gate_id]
        _require(
            isinstance(gate, Mapping)
            and gate.get("decision_role") == "explicit_gate"
            and gate.get("pass") is computed_pass,
            f"plateau {gate_id} contradicts bound evidence",
        )
    expected_failed = [
        gate_id for gate_id in PLATEAU_GATE_IDS if not computed_gate_pass[gate_id]
    ]
    _require(
        decision["failed_gate_ids"] == expected_failed
        and plateau_pass is all(computed_gate_pass.values()),
        "plateau overall pass or failed-gate list contradicts explicit gates",
    )
    disclosures = decision["mandatory_disclosures"]
    _require(
        isinstance(disclosures, Mapping)
        and set(disclosures)
        == {
            "year_local_sign_and_scale",
            "factor_level_monotonicity",
            "slice_reversals",
            "prevalence_and_support",
        }
        and isinstance(decision["warnings"], list)
        and all(isinstance(value, str) for value in decision["warnings"]),
        "plateau mandatory disclosures or warnings are malformed",
    )
    year_disclosure = disclosures["year_local_sign_and_scale"]
    _require(
        year_disclosure.get("period_pass") == neighbor_pass_by_period
        and year_disclosure.get("cross_year_both_pass")
        is (neighbor_pass_by_period["2025"] and neighbor_pass_by_period["2026"]),
        "plateau year-local disclosure contradicts neighbor table",
    )
    monotonic_disclosure = disclosures["factor_level_monotonicity"]
    _require(
        monotonic_disclosure.get("failed_factor_periods") == factor_failures
        and monotonic_disclosure.get("all_factor_periods_non_decreasing")
        is (not factor_failures),
        "plateau monotonicity disclosure contradicts table",
    )
    slice_disclosure = disclosures["slice_reversals"]
    _require(
        int(slice_disclosure.get("comparison_rows", -1)) == len(slice_reversals)
        and int(slice_disclosure.get("reversal_rows", -1)) == reversal_count,
        "plateau slice-reversal disclosure contradicts table",
    )
    _require(
        evaluator_manifest.get("diagnostic_plateau_pass") is plateau_pass
        and evaluator_manifest.get("decision_status") == expected_status,
        "plateau evaluator manifest contradicts decision",
    )
    return {
        "run_identities": run_identities,
        "evaluation_identities": evaluation_identities,
        "input_hashes": actual_input_hashes,
        "run_manifest": run_manifest,
        "evaluator_manifest": evaluator_manifest,
        "decision": decision,
        "summary": {
            "diagnostic_plateau_pass": plateau_pass,
            "decision_status": expected_status,
            "failed_gate_ids": decision["failed_gate_ids"],
            "gates": decision["gates"],
            "mandatory_disclosures": decision["mandatory_disclosures"],
            "warnings": decision["warnings"],
            "mandatory_disclosure_artifacts": {
                "year_summary.parquet": "MANDATORY_DISCLOSURE_ONLY",
                "monotonicity.parquet": "MANDATORY_DISCLOSURE_ONLY",
                "slice_reversals.parquet": "MANDATORY_DISCLOSURE_ONLY",
            },
            "outcome_informed_retrospective": True,
            "production_promotion_authorized": False,
        },
    }


def _contract_identities(paths: DecisionPaths) -> dict[str, dict[str, Any]]:
    names = (
        "study_contract.json",
        "study_contract_amendment_1.json",
        "study_contract_amendment_2.json",
        "threshold_plateau_contract.json",
    )
    identities = {name: _identity(paths.study_dir / name) for name in names}
    parent = _load_json(paths.study_dir / "study_contract.json")
    _require(
        parent.get("contract_id") == PARENT_CONTRACT
        and parent.get("contract_status") == "frozen_before_v1_outcome_run"
        and parent.get("freeze_date") == "20260827"
        and parent.get("prospective_date_rule") == "Date > 20260827"
        and parent.get("scope_boundary", {}).get(
            "production_promotion_authorized"
        )
        is False,
        "parent study contract identity or deployment boundary drifted",
    )
    amendment_1 = _load_json(
        paths.study_dir / "study_contract_amendment_1.json"
    )
    _require(
        amendment_1.get("amendment_id")
        == f"{PARENT_CONTRACT}/amendment_1"
        and amendment_1.get("parent_contract_id") == PARENT_CONTRACT
        and amendment_1.get("deployment_override", {}).get(
            "stage_1_factorial_can_authorize_deployment"
        )
        is False,
        "methods amendment 1 identity or deployment override drifted",
    )
    amendment_2 = _load_json(
        paths.study_dir / "study_contract_amendment_2.json"
    )
    _require(
        amendment_2.get("amendment_id")
        == f"{PARENT_CONTRACT}/amendment_2"
        and amendment_2.get("parent_contract_id") == PARENT_CONTRACT
        and amendment_2.get("depends_on")
        == f"{PARENT_CONTRACT}/amendment_1"
        and amendment_2.get("materiality_boundary", {}).get(
            "new_post_outcome_delta_min_added"
        )
        is False,
        "methods amendment 2 identity or materiality boundary drifted",
    )
    return identities


def _gate_payloads(
    *,
    calendar: Mapping[str, Any],
    lineage: Mapping[str, Any],
    positive: Mapping[str, Any],
    inference: Mapping[str, Any],
    secondary: Mapping[str, Any],
    oos: Mapping[str, Any],
    historical: Mapping[str, Any],
    plateau: Mapping[str, Any],
) -> dict[str, Any]:
    anchor = inference["anchor"]
    secondary_anchor = secondary["anchor"]
    support_pass = (
        int(anchor["active_periods"]) >= 60
        and int(secondary_anchor["active_months"]) >= 3
        and int(anchor["events"]) >= 1_000
    )
    bias_pass = (
        float(anchor["maxstat_simultaneous_lcb"]) > 0.0
        and float(secondary_anchor["equal_active_residual_bp"]) > 0.0
    )
    temporal = secondary["temporal_intervals"]
    temporal_by_length: dict[str, dict[str, Any]] = {}
    for block_length in (5, 10, 20):
        rows = [
            row
            for row in temporal
            if int(row["block_length_requested"]) == block_length
        ]
        positive_blocks = sum(
            float(row["point_delta_z"]) > 0.0 for row in rows
        )
        no_negative_ucb = all(
            float(row["one_sided_90_ucb"]) >= 0.0 for row in rows
        )
        temporal_by_length[str(block_length)] = {
            "block_length": block_length,
            "primary": block_length == 10,
            "positive_blocks": positive_blocks,
            "no_block_upper_bound_below_zero": no_negative_ucb,
            "diagnostic_rule_pass": (
                positive_blocks >= 2 and no_negative_ucb
            ),
            "blocks": rows,
        }
    primary_temporal = temporal_by_length["10"]
    temporal_pass = bool(primary_temporal["diagnostic_rule_pass"])
    sensitivity_changes = [
        block_length
        for block_length in (5, 20)
        if bool(temporal_by_length[str(block_length)]["diagnostic_rule_pass"])
        != temporal_pass
    ]
    parsimony_pass = all(
        row["statistical_rule_pass"] for row in inference["parsimony"]
    )
    plateau_pass = bool(plateau["summary"]["diagnostic_plateau_pass"])
    return {
        "G0_lineage_and_positive_control": {
            "retrospective_diagnostic_status": (
                "PASS_WITH_DISCLOSED_LIMITATIONS"
            ),
            "prospective_deployment_status": "NOT_A_PROSPECTIVE_EFFECT_GATE",
            "lineage_status": lineage["diagnostic"]["status"],
            "positive_control_status": positive["summary"]["status"],
            "validation_gaps": lineage["diagnostic"]["validation_gaps"],
            "interpretation": (
                "Pipeline calibration and conservative selector lineage passed "
                "for retrospective diagnosis; recorded gaps remain explicit."
            ),
        },
        "G1_support": {
            "retrospective_diagnostic_status": (
                "PASS_DIAGNOSTIC_ONLY" if support_pass else "FAIL_DIAGNOSTIC"
            ),
            "prospective_deployment_status": "UNAVAILABLE_NOT_EVALUATED",
            "thresholds": {
                "minimum_active_sessions": 60,
                "minimum_active_months": 3,
                "minimum_events": 1_000,
            },
            "observed": {
                "active_sessions": anchor["active_periods"],
                "active_months": secondary_anchor["active_months"],
                "events": anchor["events"],
            },
        },
        "G2_primary_bias": {
            "retrospective_diagnostic_status": (
                "PASS_DIAGNOSTIC_ONLY" if bias_pass else "FAIL_DIAGNOSTIC"
            ),
            "prospective_deployment_status": "UNAVAILABLE_NOT_EVALUATED",
            "observed": {
                "maxstat_simultaneous_lcb": anchor[
                    "maxstat_simultaneous_lcb"
                ],
                "equal_active_residual_bp": secondary_anchor[
                    "equal_active_residual_bp"
                ],
            },
            "interpretation": (
                "The G2 rule is deployment-valid only on prospective_first_live; "
                "the observed pass is retrospective diagnostic evidence."
            ),
        },
        "G3_temporal_stability": {
            "retrospective_diagnostic_status": (
                "PASS_DIAGNOSTIC_ONLY" if temporal_pass else "FAIL_DIAGNOSTIC"
            ),
            "prospective_deployment_status": "UNAVAILABLE_NOT_EVALUATED",
            "positive_blocks": primary_temporal["positive_blocks"],
            "blocks": primary_temporal["blocks"],
            "primary_block_length": 10,
            "block_length_sensitivity": temporal_by_length,
            "sensitivity_conclusion_change_lengths": sensitivity_changes,
            "sensitivity_conclusion_available": True,
        },
        "G4_parsimony": {
            "statistical_parsimony_status": (
                "PASS_FROZEN_RULE" if parsimony_pass else "FAIL_FROZEN_RULE"
            ),
            "economic_materiality_status": (
                "NOT_ESTABLISHED_NO_PREREGISTERED_MINIMUM"
            ),
            "contrasts": inference["parsimony"],
            "interpretation": (
                "A positive statistical increment does not establish that a "
                "gate is economically necessary or material."
            ),
        },
        "G5_selection_bias_disclosure": {
            "status": "PASS_DISCLOSURE_LOWER_BOUND_ONLY",
            "current_family": {
                "raw_trials": anchor["raw_trial_count"],
                "effective_trials": anchor["effective_trial_count"],
                "maxstat_reported": True,
            },
            "historical_exposure": historical["summary"],
            "interpretation": (
                "The current max-stat covers only the frozen eight-cell family; "
                "historical exposure is material and incompletely bounded above."
            ),
        },
        "G6_evidence_class": {
            "status": "FAIL_CLOSED_NO_PROSPECTIVE_FIRST_LIVE_ARTIFACT",
            "retrospective_sessions": calendar["summary"]["requested_sessions"],
            "artifact_oos_session": oos["anchor"],
            "prospective_deployment_status": "UNAVAILABLE_NOT_EVALUATED",
            "interpretation": (
                "No Date > 20260827 prospective-first-live evaluation artifact "
                "was supplied; retrospective and 20260826 cannot authorize deployment."
            ),
        },
        "G7_threshold_plateau": {
            "retrospective_diagnostic_status": (
                "PASS_DIAGNOSTIC_ONLY" if plateau_pass else "FAIL_DIAGNOSTIC"
            ),
            "prospective_deployment_status": "NOT_PROSPECTIVE_BY_CONTRACT",
            **plateau["summary"],
            "interpretation": (
                "The outcome-informed retrospective plateau can diagnose an "
                "isolated peak but cannot authorize production promotion."
            ),
        },
    }


def build_decision(paths: DecisionPaths) -> DecisionResult:
    """Validate every required artifact and build a deployment-denying result."""

    contracts = _contract_identities(paths)
    calendar = _validate_calendar(paths)
    lineage = _validate_lineage(paths, calendar)
    positive = _validate_positive_control(paths)
    inference = _validate_inference(paths, calendar)
    secondary = _validate_secondary(paths)
    oos = _validate_artifact_oos(paths)
    historical = _validate_historical_exposure(paths)
    plateau = _validate_plateau(paths, calendar)

    inference["anchor"]["active_months"] = secondary["anchor"]["active_months"]
    gates = _gate_payloads(
        calendar=calendar,
        lineage=lineage,
        positive=positive,
        inference=inference,
        secondary=secondary,
        oos=oos,
        historical=historical,
        plateau=plateau,
    )
    input_identities = {
        "contracts": contracts,
        "calendar": calendar["identities"],
        "lineage": lineage["identities"],
        "positive_control": positive["identity"],
        "inference": inference["identities"],
        "secondary": secondary["identities"],
        "artifact_oos": oos["identities"],
        "historical_exposure": historical["identity"],
        "plateau_run": plateau["run_identities"],
        "plateau_evaluation": plateau["evaluation_identities"],
    }
    blockers = [
        "no prospective_first_live evaluation artifact",
        "G1/G2/G3/G6 prospective deployment gates are unavailable",
        "20260826 is one-session artifact-OOS descriptive evidence only",
        "historical selector exposure is a lower bound and incompletely bounded above",
        "G4 has no preregistered economic materiality minimum",
        "G7 is outcome-informed retrospective robustness only",
        "all source contracts explicitly deny production promotion",
    ]
    decision = {
        "contract": DECISION_CONTRACT,
        "study_id": "20260827_fh_neg_rejection_robustness",
        "condition_id": "fh_neg_rejection",
        "family_id": FAMILY_ID,
        "family_version": FAMILY_VERSION,
        "anchor_cell_id": ANCHOR_ID,
        "overall_status": "RESEARCH_SHADOW_ONLY",
        "deployment_authorized": False,
        "production_promotion_authorized": False,
        "deployment_handoff_authorized": False,
        "selected_replacement_condition": None,
        "evidence_classes": {
            "retrospective_diagnostic": {
                "dates": ["20250211", "20260825"],
                "deployment_eligible": False,
            },
            "artifact_oos_non_pristine": {
                "date": "20260826",
                "sessions": 1,
                "deployment_eligible": False,
                "analysis_role": "descriptive_only",
            },
            "prospective_first_live": {
                "date_rule": "Date > 20260827",
                "evaluation_status": "UNAVAILABLE_NOT_EVALUATED",
                "deployment_eligible_if_future_gates_pass": True,
            },
        },
        "gates": gates,
        "blocking_reasons": blockers,
        "input_artifact_identities": input_identities,
        "fail_closed_policy": (
            "Missing, unbound, non-finite, contradictory, or fingerprint-drifted "
            "inputs abort generation; no missing value is inferred."
        ),
    }
    report_data = {
        "contract": REPORT_DATA_CONTRACT,
        "study_id": decision["study_id"],
        "overall": {
            "status": decision["overall_status"],
            "deployment_authorized": False,
            "blocking_reasons": blockers,
        },
        "sample": {
            **calendar["summary"],
            "anchor_label_valid_events": inference["anchor"]["events"],
            "anchor_active_sessions": inference["anchor"]["active_periods"],
            "anchor_active_months": secondary["anchor"]["active_months"],
            "anchor_selection_rate_mean": inference["anchor"][
                "selection_rate_mean"
            ],
        },
        "anchor_retrospective": {
            **inference["anchor"],
            **secondary["anchor"],
            "hac_sensitivity": secondary["hac"],
            "temporal_intervals": secondary["temporal_intervals"],
        },
        "parsimony": gates["G4_parsimony"],
        "artifact_oos_20260826": {
            "status": "DESCRIPTIVE_ONLY_NOT_GATE_EVIDENCE",
            "anchor": oos["anchor"],
            "loo": oos["loo"],
        },
        "selection_bias": gates["G5_selection_bias_disclosure"],
        "threshold_plateau": gates["G7_threshold_plateau"],
        "gate_statuses": {
            gate_id: {
                key: value
                for key, value in payload.items()
                if key.endswith("status") or key == "status"
            }
            for gate_id, payload in gates.items()
        },
        "lineage_limitations": lineage["diagnostic"]["validation_gaps"],
    }
    return DecisionResult(decision=decision, report_data=report_data)


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    try:
        rendered = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise DecisionArtifactError(
            f"{path.name}: decision payload is not strict finite JSON"
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
        encoding="utf-8",
    ) as handle:
        handle.write(rendered)
        handle.write("\n")
        temp_path = Path(handle.name)
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def write_result(result: DecisionResult, output_dir: Path) -> None:
    """Write only the frozen decision and compact report-data artifacts."""

    _atomic_write_json(result.decision, output_dir / "decision.json")
    _atomic_write_json(result.report_data, output_dir / "report_data.json")


def resolve_output_dir(path: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    root = ARTIFACT_ROOT.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise DecisionArtifactError(f"output must be under {root}") from exc
    _require(bool(relative.parts), "output cannot overwrite artifact root")
    return resolved


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the fail-closed fh_neg_rejection final research decision"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output_dir = resolve_output_dir(args.output_dir)
        paths = DecisionPaths.defaults(output_dir)
        result = build_decision(paths)
        write_result(result, output_dir)
    except DecisionArtifactError as exc:
        raise SystemExit(f"decision build failed closed: {exc}") from exc
    print(
        json.dumps(
            {
                "contract": DECISION_CONTRACT,
                "overall_status": result.decision["overall_status"],
                "deployment_authorized": False,
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
