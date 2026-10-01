"""Synthetic contract tests for the frozen 27-cell plateau evaluator."""

from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl

import evaluate_threshold_plateau as study


def _dates() -> list[str]:
    first = datetime.strptime(study.START_DATE, "%Y%m%d")
    last = datetime.strptime(study.END_DATE, "%Y%m%d")
    dates_2025 = [(first + timedelta(days=index)).strftime("%Y%m%d") for index in range(60)]
    start_2026 = datetime(2026, 1, 2)
    dates_2026 = [
        (start_2026 + timedelta(days=index)).strftime("%Y%m%d") for index in range(59)
    ] + [last.strftime("%Y%m%d")]
    return dates_2025 + dates_2026


def _calendar() -> tuple[pl.DataFrame, dict[str, object]]:
    dates = _dates()
    frame = pl.DataFrame(
        {
            "Date": dates,
            "evidence_class": ["retrospective_diagnostic"] * len(dates),
            "source_complete": [True] * len(dates),
            "run_status": ["success"] * len(dates),
            "session_status": ["evaluable"] * len(dates),
            "eligible_pool_rows": [1_000] * len(dates),
            "control_n_z": [900] * len(dates),
            "aggregate_complete": [True] * len(dates),
            "membership_complete": [True] * len(dates),
        }
    )
    manifest = {
        "contract": "fh_neg_rejection_requested_calendar/v1",
        "start": study.START_DATE,
        "end": study.END_DATE,
        "evidence_class": "retrospective_diagnostic",
        "requested_sessions": len(dates),
        "source_complete_sessions": len(dates),
        "status_counts": {"evaluable": len(dates)},
    }
    return frame, manifest


def _runner_manifest() -> dict[str, object]:
    dates = _dates()
    source_identity = {
        source: {"path": f"/synthetic/{source}", "size": 1, "mtime_ns": 2}
        for source in ("tick_data", "tick_feature", "pre_market", "txf_tick")
    }
    return {
        "contract": study.CONTRACT_ID,
        "family_id": study.FAMILY_ID,
        "family_version": study.FAMILY_VERSION,
        "candidate_ids": list(study.CELL_IDS),
        "candidate_definitions": {cell_id: f"synthetic {cell_id}" for cell_id in study.CELL_IDS},
        "center_candidate_id": study.CENTER_CELL_ID,
        "frozen_contract_sha256": study.CONTRACT_SHA256,
        "threshold_design_fingerprint": study.EXPECTED_DESIGN_FINGERPRINT,
        "threshold_design": study.EXPECTED_DESIGN,
        "start": study.START_DATE,
        "end": study.END_DATE,
        "step": 1,
        "label_profile_id": study.CONTRACT["shared_semantics"]["label"],
        "pool_id": study.CONTRACT["shared_semantics"]["eligible_pool"],
        "pool_version": "1",
        "primary_estimand": "shared_pool",
        "control_sample_size": None,
        "evidence_class": "retrospective_diagnostic",
        "production_promotion_authorized": False,
        "selection_rule": "fixed 27-cell ledger; no ranking or center replacement",
        "sessions": dates,
        "failures": [],
        "source_identities": {date: source_identity for date in dates},
        "runner_artifact": {"path": "/synthetic/runner.py", "size": 10, "mtime_ns": 20, "sha256": "b" * 64},
        "parent_run_manifest": {"path": "/synthetic/parent.json", "size": 10, "mtime_ns": 20, "sha256": "c" * 64},
    }


def _input_hashes() -> dict[str, str]:
    return {
        "daily_aggregates": "1" * 64,
        "plateau_run_manifest": "2" * 64,
        "requested_calendar": "3" * 64,
        "requested_calendar_manifest": "4" * 64,
        "threshold_contract": study.CONTRACT_SHA256,
    }


def _effect(
    cell: study.ThresholdCell,
    day_index: int,
    *,
    low_neighbors: bool = False,
    slice_reversal: bool = False,
) -> float:
    depth_index = study.DEPTH_LEVELS.index(cell.depth_ratio_lte)
    gap_index = study.GAP_LEVELS.index(cell.gap_ticks_gte)
    bear_index = study.BEAR_LEVELS.index(cell.bear_score_gte)
    value = 0.14 + 0.010 * depth_index + 0.008 * gap_index + 0.006 * bear_index
    if low_neighbors and cell.candidate_id in study.NEIGHBOR_IDS:
        value = 0.02
    if slice_reversal and cell.candidate_id == "d06_g03_b1":
        value = 0.40
    return value + 0.008 * math_sine(day_index)


def math_sine(index: int) -> float:
    return float(np.sin(index / 4.0))


def _daily(
    *,
    low_neighbors: bool = False,
    slice_reversal: bool = False,
    low_support: bool = False,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    runner = _runner_manifest()
    for day_index, date in enumerate(_dates()):
        for cell in study.CELLS:
            depth_index = study.DEPTH_LEVELS.index(cell.depth_ratio_lte)
            gap_index = study.GAP_LEVELS.index(cell.gap_ticks_gte)
            bear_index = study.BEAR_LEVELS.index(cell.bear_score_gte)
            support = 80 - 10 * depth_index - 5 * gap_index - 4 * bear_index
            if low_support and cell.candidate_id == "d10_g05_b3":
                support = 8
            effect = _effect(
                cell,
                day_index,
                low_neighbors=low_neighbors,
                slice_reversal=slice_reversal,
            )
            rows.append(
                {
                    "contract_id": study.CONTRACT_ID,
                    "Date": date,
                    "family_id": study.FAMILY_ID,
                    "family_version": study.FAMILY_VERSION,
                    "candidate_id": cell.candidate_id,
                    "label_profile_id": study.CONTRACT["shared_semantics"]["label"],
                    "label_profile_hash": "a" * 64,
                    "pool_id": study.CONTRACT["shared_semantics"]["eligible_pool"],
                    "pool_version": "1",
                    "control_mode": "full",
                    "control_sample_size": 0,
                    "primary_estimand": "shared_pool",
                    "eligible_pool_rows": 1_000,
                    "indexed_condition_rows": support + 2,
                    "condition_n": support,
                    "condition_n_z": support,
                    "condition_sum_z": support * effect,
                    "control_n_z": 900,
                    "control_sum_z": 0.0,
                    "selection_rate": support / 1_000,
                    "plateau_contract_sha256": study.CONTRACT_SHA256,
                    "plateau_design_fingerprint": study.EXPECTED_DESIGN_FINGERPRINT,
                    "plateau_runner_sha256": "b" * 64,
                    "parent_run_manifest_sha256": "c" * 64,
                    "parent_membership_sha256": "d" * 64,
                    "parent_membership_lineage_sha256": "e" * 64,
                    "parent_source_identities_sha256": "f" * 64,
                }
            )
    return pl.DataFrame(rows)


def _evaluate(**daily_options) -> study.PlateauResult:
    calendar, calendar_manifest = _calendar()
    return study.evaluate_threshold_plateau(
        daily_aggregates=_daily(**daily_options),
        requested_calendar=calendar,
        requested_calendar_manifest=calendar_manifest,
        runner_manifest=_runner_manifest(),
        input_artifact_sha256=_input_hashes(),
    )


class ThresholdPlateauEvaluatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.passing = _evaluate()

    def test_frozen_grid_ids_and_metadata_are_exact(self) -> None:
        self.assertEqual(len(study.CELL_IDS), 27)
        self.assertEqual(study.CELL_IDS[0], "d06_g03_b1")
        self.assertEqual(study.CELL_IDS[-1], "d10_g05_b3")
        self.assertEqual(study.CENTER_CELL_ID, "d08_g04_b2")
        self.assertEqual(study.EXPECTED_DESIGN["cells"], [cell.metadata() for cell in study.CELLS])

    def test_passing_panel_has_complete_outputs_and_no_deployment_claim(self) -> None:
        result = self.passing
        self.assertEqual(len(result.daily_effects), 120 * 27)
        self.assertEqual(len(result.cell_summary), 27)
        self.assertEqual(len(result.year_summary), 54)
        self.assertEqual(len(result.fwer_audit), 81)
        self.assertEqual(len(result.neighbor_diagnostics), 18)
        self.assertEqual(len(result.monotonicity), 9)
        self.assertEqual(len(result.slice_reversals), 162)
        self.assertEqual(len(result.trial_ledger), 27)
        self.assertTrue(result.decision["diagnostic_plateau_pass"])
        self.assertFalse(result.decision["production_promotion_authorized"])
        self.assertEqual(result.decision["input_artifact_sha256"], _input_hashes())
        center = result.fwer_audit.filter(pl.col("candidate_id") == study.CENTER_CELL_ID)
        self.assertEqual(set(center["block_length_requested"]), {5, 10, 20})
        self.assertTrue((center["simultaneous_lower_bound"] > 0).all())
        self.assertTrue((center["joint_valid_bootstrap_rate"] >= 0.90).all())

    def test_support_floor_failure_is_a_fail_closed_decision(self) -> None:
        result = _evaluate(low_support=True)
        self.assertFalse(result.decision["diagnostic_plateau_pass"])
        self.assertIn("P1_support_floor_each_cell", result.decision["failed_gate_ids"])
        failed = result.cell_summary.filter(~pl.col("support_floor_pass"))
        self.assertEqual(failed["candidate_id"].to_list(), ["d10_g05_b3"])

    def test_local_scale_failure_is_a_fail_closed_decision(self) -> None:
        result = _evaluate(low_neighbors=True)
        self.assertFalse(result.decision["diagnostic_plateau_pass"])
        self.assertIn(
            "P3_full_period_local_sign_and_scale", result.decision["failed_gate_ids"]
        )
        periods = result.neighbor_diagnostics.group_by("period").agg(
            pl.col("period_local_sign_and_scale_pass").first().alias("passed")
        )
        self.assertFalse(periods.filter(pl.col("period") == "full")["passed"][0])

    def test_slice_reversals_are_retained_but_not_invented_as_gate(self) -> None:
        result = _evaluate(slice_reversal=True)
        reversals = result.slice_reversals.filter(pl.col("slice_reversal") == True)  # noqa: E712
        self.assertGreater(len(reversals), 0)
        self.assertEqual(
            result.decision["mandatory_disclosures"]["slice_reversals"]["reversal_rows"],
            len(reversals),
        )
        self.assertTrue(result.decision["diagnostic_plateau_pass"])
        self.assertTrue(any("monotonicity" in warning for warning in result.decision["warnings"]))

    def test_manifest_cell_order_drift_is_rejected(self) -> None:
        calendar, calendar_manifest = _calendar()
        manifest = _runner_manifest()
        manifest["candidate_ids"] = list(reversed(study.CELL_IDS))
        with self.assertRaisesRegex(study.PlateauContractError, "candidate_ids"):
            study.evaluate_threshold_plateau(
                daily_aggregates=_daily(),
                requested_calendar=calendar,
                requested_calendar_manifest=calendar_manifest,
                runner_manifest=manifest,
                input_artifact_sha256=_input_hashes(),
            )

    def test_manifest_parameter_metadata_drift_is_rejected(self) -> None:
        calendar, calendar_manifest = _calendar()
        manifest = _runner_manifest()
        design = json.loads(json.dumps(manifest["threshold_design"]))
        design["cells"][0]["depth_ratio_lte"] = -5.0
        manifest["threshold_design"] = design
        manifest["threshold_design_fingerprint"] = hashlib.sha256(
            study._canonical_json(design).encode("utf-8")
        ).hexdigest()
        with self.assertRaisesRegex(study.PlateauContractError, "threshold_design"):
            study.evaluate_threshold_plateau(
                daily_aggregates=_daily(),
                requested_calendar=calendar,
                requested_calendar_manifest=calendar_manifest,
                runner_manifest=manifest,
                input_artifact_sha256=_input_hashes(),
            )

    def test_mixed_evidence_calendar_is_rejected(self) -> None:
        calendar, calendar_manifest = _calendar()
        calendar = calendar.with_row_index("_row").with_columns(
            pl.when(pl.col("_row") == 0)
            .then(pl.lit("artifact_oos_non_pristine"))
            .otherwise(pl.col("evidence_class"))
            .alias("evidence_class")
        ).drop("_row")
        with self.assertRaisesRegex(study.PlateauContractError, "evidence class"):
            study.evaluate_threshold_plateau(
                daily_aggregates=_daily(),
                requested_calendar=calendar,
                requested_calendar_manifest=calendar_manifest,
                runner_manifest=_runner_manifest(),
                input_artifact_sha256=_input_hashes(),
            )

    def test_writer_emits_only_frozen_machine_readable_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evaluation"
            manifest = study.write_result(self.passing, output)
            expected = {
                *study.OUTPUT_FRAMES,
                "plateau_decision.json",
                "run_manifest.json",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            decision = json.loads((output / "plateau_decision.json").read_text())
            self.assertEqual(decision["contract"], study.DECISION_CONTRACT)
            self.assertEqual(manifest["contract"], study.EVALUATOR_CONTRACT)
            self.assertEqual(set(manifest["output_artifacts"]), expected - {"run_manifest.json"})

    def test_sibling_output_is_allowed_but_input_inside_output_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "inputs" / "daily.parquet"
            sibling_output = root / "evaluation"
            self.assertFalse(study._output_contains_input(sibling_output, input_path))
            self.assertTrue(study._output_contains_input(input_path.parent, input_path))
            self.assertTrue(study._output_contains_input(root, input_path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
