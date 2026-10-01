from __future__ import annotations

import datetime as dt
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import polars as pl


MODULE_PATH = Path(__file__).with_name("evaluate_secondary_diagnostics.py")
SPEC = importlib.util.spec_from_file_location(
    "fh_neg_secondary_diagnostics", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise ImportError(MODULE_PATH)
SECONDARY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SECONDARY
SPEC.loader.exec_module(SECONDARY)

EVALUATOR_PATH = Path(__file__).with_name("evaluate_robustness.py")
EVALUATOR_SPEC = importlib.util.spec_from_file_location(
    "fh_neg_secondary_integration_evaluator", EVALUATOR_PATH
)
if EVALUATOR_SPEC is None or EVALUATOR_SPEC.loader is None:  # pragma: no cover
    raise ImportError(EVALUATOR_PATH)
EVALUATOR = importlib.util.module_from_spec(EVALUATOR_SPEC)
sys.modules[EVALUATOR_SPEC.name] = EVALUATOR
EVALUATOR_SPEC.loader.exec_module(EVALUATOR)


def synthetic_dates(periods: int = 75) -> list[str]:
    start = dt.date(2026, 1, 15)
    return [
        (start + dt.timedelta(days=index)).strftime("%Y%m%d")
        for index in range(periods)
    ]


def runner_fixture(periods: int = 75) -> pl.DataFrame:
    rows = []
    zero_anchor_indices = {0, periods // 3, 2 * periods // 3}
    zero_control_index = 4
    for day_index, date in enumerate(synthetic_dates(periods)):
        control_n = 0 if day_index == zero_control_index else 100
        control_mean_z = -0.02
        control_mean_raw_bp = 2.0
        control_mean_residual_bp = -1.0
        for depth, far, bear in SECONDARY.CELL_SPECS:
            cell_id = f"d{depth}_g{far}_b{bear}"
            support = 48 - 4 * (depth + far + bear)
            if control_n == 0:
                support = 0
            if cell_id == SECONDARY.FULL_CELL_ID and day_index in zero_anchor_indices:
                support = 0
            daily_noise = ((day_index % 7) - 3) * 0.002
            delta_z = 0.03 + 0.04 * depth + 0.02 * far + 0.01 * bear + daily_noise
            delta_raw_bp = 3.0 + 0.8 * depth + 0.4 * far + 0.2 * bear + day_index * 0.01
            delta_residual_bp = 1.5 + 0.5 * depth + 0.2 * far + 0.1 * bear + day_index * 0.005
            rows.append(
                {
                    "contract_id": SECONDARY.PARENT_CONTRACT_ID,
                    "Date": date,
                    "family_id": SECONDARY.FAMILY_ID,
                    "family_version": SECONDARY.FAMILY_VERSION,
                    "candidate_id": cell_id,
                    "label_profile_id": "negfill.short_taker_stop108.v1",
                    "label_profile_hash": "synthetic-label-hash",
                    "pool_id": "synthetic-pool",
                    "pool_version": "1",
                    "condition_n_z": support,
                    "control_n_z": control_n,
                    "condition_sum_z": (
                        support * (control_mean_z + delta_z) if support else 0.0
                    ),
                    "control_sum_z": control_n * control_mean_z,
                    "condition_sum_raw_bp": (
                        support * (control_mean_raw_bp + delta_raw_bp)
                        if support
                        else 0.0
                    ),
                    "control_sum_raw_bp": control_n * control_mean_raw_bp,
                    "condition_sum_residual_bp": (
                        support
                        * (control_mean_residual_bp + delta_residual_bp)
                        if support
                        else 0.0
                    ),
                    "control_sum_residual_bp": (
                        control_n * control_mean_residual_bp
                    ),
                }
            )
    return pl.DataFrame(rows)


def requested_calendar_fixture(periods: int = 75) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Date": synthetic_dates(periods),
            "source_complete": [True] * periods,
            "run_status": ["success"] * periods,
            "evidence_class": [
                SECONDARY.RETROSPECTIVE_EVIDENCE_CLASS
            ]
            * periods,
            "calendar_status": ["evaluation_expected"] * periods,
        }
    )


def evaluator_daily_fixture(runner: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for row in runner.iter_rows(named=True):
        condition_n = int(row["condition_n_z"])
        control_n = int(row["control_n_z"])
        if control_n == 0:
            status = "zero_control_ineligible"
        elif condition_n == 0:
            status = "zero_event"
        else:
            status = "active"
        active = status == "active"
        depth, far, bear = SECONDARY.CELL_BITS[str(row["candidate_id"])]
        rows.append(
            {
                "Date": row["Date"],
                "cell_id": row["candidate_id"],
                "use_depth": depth,
                "use_far": far,
                "use_bear": bear,
                "condition_n_z": condition_n,
                "condition_sum_z": row["condition_sum_z"],
                "control_n_z": control_n,
                "control_sum_z": row["control_sum_z"],
                "daily_delta_z": (
                    row["condition_sum_z"] / condition_n
                    - row["control_sum_z"] / control_n
                    if active
                    else None
                ),
                "condition_active": condition_n > 0,
                "effect_active": active,
                "evidence_class": SECONDARY.RETROSPECTIVE_EVIDENCE_CLASS,
                "calendar_status": "evaluation_expected",
                "period_status": status,
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


def temporal_reference_fixture(daily: pl.DataFrame) -> pl.DataFrame:
    anchor = daily.filter(
        pl.col("cell_id") == SECONDARY.FULL_CELL_ID
    ).sort("Date")
    blocks = np.array_split(
        np.asarray(anchor["Date"].to_list(), dtype=object), 3
    )
    rows = []
    for block_index, block_array in enumerate(blocks, start=1):
        dates = [str(value) for value in block_array.tolist()]
        subset = anchor.filter(pl.col("Date").is_in(dates)).sort("Date")
        active = subset.filter(pl.col("effect_active"))
        rows.append(
            {
                "target_kind": "cell_effect",
                "target_id": SECONDARY.FULL_CELL_ID,
                "block_id": f"T{block_index:02d}",
                "block_start": dates[0],
                "block_end": dates[-1],
                "calendar_periods": len(dates),
                "active_periods": len(active),
                "equal_period_value": (
                    float(active["daily_delta_z"].mean()) if len(active) else None
                ),
            }
        )
    return pl.DataFrame(rows)


def evaluator_manifest_fixture(periods: int = 75) -> dict[str, object]:
    dates = synthetic_dates(periods)
    return {
        "contract": SECONDARY.EVALUATOR_CONTRACT_ID,
        "full_cell_id": SECONDARY.FULL_CELL_ID,
        "cell_ids": list(SECONDARY.CELL_IDS),
        "calendar": {
            "start": dates[0],
            "end": dates[-1],
            "periods": periods,
            "evidence_class": SECONDARY.RETROSPECTIVE_EVIDENCE_CLASS,
        },
        "bootstrap": {
            "method": "circular_moving_block_synchronous_trials",
            "seed": 20_260_827,
            "repetitions": 5_000,
        },
    }


def complete_fixture(periods: int = 75):
    runner = runner_fixture(periods)
    calendar = requested_calendar_fixture(periods)
    daily = evaluator_daily_fixture(runner)
    temporal = temporal_reference_fixture(daily)
    manifest = evaluator_manifest_fixture(periods)
    return runner, calendar, daily, temporal, manifest


def artifact_oos_fixture() -> tuple[pl.DataFrame, pl.DataFrame]:
    source_date = synthetic_dates()[10]
    runner = runner_fixture().filter(
        pl.col("Date") == source_date
    ).with_columns(pl.lit("20260826").alias("Date"))
    calendar = pl.DataFrame(
        {
            "Date": ["20260826"],
            "source_complete": [True],
            "run_status": ["success"],
            "evidence_class": ["artifact_oos_non_pristine"],
            "session_status": ["evaluable"],
        }
    )
    return runner, calendar


class SecondaryDiagnosticsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        (
            cls.runner,
            cls.calendar,
            cls.evaluator_daily,
            cls.temporal_reference,
            cls.evaluator_manifest,
        ) = complete_fixture()
        cls.result = SECONDARY.evaluate_secondary_diagnostics(
            runner_daily_stats=cls.runner,
            requested_calendar=cls.calendar,
            evaluator_daily_effects=cls.evaluator_daily,
            evaluator_temporal_blocks=cls.temporal_reference,
            evaluator_manifest=cls.evaluator_manifest,
        )
        cls.oos_runner, cls.oos_calendar = artifact_oos_fixture()
        cls.oos_result = SECONDARY.build_artifact_oos_descriptive(
            runner_daily_stats=cls.oos_runner,
            artifact_oos_calendar=cls.oos_calendar,
        )

    def test_raw_and_residual_bp_use_z_support_denominators(self) -> None:
        row = self.result.daily_effects.filter(
            (pl.col("Date") == synthetic_dates()[10])
            & (pl.col("cell_id") == SECONDARY.FULL_CELL_ID)
        ).row(0, named=True)
        expected_raw = (
            row["condition_sum_raw_bp"] / row["condition_n_z"]
            - row["control_sum_raw_bp"] / row["control_n_z"]
        )
        expected_residual = (
            row["condition_sum_residual_bp"] / row["condition_n_z"]
            - row["control_sum_residual_bp"] / row["control_n_z"]
        )
        self.assertAlmostEqual(row["daily_delta_raw_bp"], expected_raw)
        self.assertAlmostEqual(
            row["daily_delta_residual_bp"], expected_residual
        )

    def test_inactive_effects_remain_undefined(self) -> None:
        zero_event = self.result.daily_effects.filter(
            (pl.col("Date") == synthetic_dates()[0])
            & (pl.col("cell_id") == SECONDARY.FULL_CELL_ID)
        ).row(0, named=True)
        self.assertEqual(zero_event["period_status"], "zero_event")
        self.assertIsNone(zero_event["daily_delta_z"])
        self.assertIsNone(zero_event["daily_delta_raw_bp"])
        self.assertIsNone(zero_event["daily_delta_residual_bp"])

        zero_control = self.result.daily_effects.filter(
            (pl.col("Date") == synthetic_dates()[4])
            & (pl.col("cell_id") == "d0_g0_b0")
        ).row(0, named=True)
        self.assertEqual(
            zero_control["period_status"], "zero_control_ineligible"
        )
        self.assertIsNone(zero_control["daily_delta_raw_bp"])

    def test_equal_active_bp_is_an_equal_session_mean(self) -> None:
        cell_id = SECONDARY.FULL_CELL_ID
        active = self.result.daily_effects.filter(
            (pl.col("cell_id") == cell_id) & pl.col("effect_active")
        )
        summary = self.result.equal_active_bp.filter(
            pl.col("cell_id") == cell_id
        ).row(0, named=True)
        self.assertEqual(summary["active_sessions"], len(active))
        self.assertAlmostEqual(
            summary["equal_active_raw_bp"],
            active["daily_delta_raw_bp"].mean(),
        )
        self.assertAlmostEqual(
            summary["equal_active_residual_bp"],
            active["daily_delta_residual_bp"].mean(),
        )

    def test_monthly_summary_first_averages_active_days(self) -> None:
        self.assertEqual(
            len(self.result.monthly_effects),
            len(set(date[:6] for date in synthetic_dates())) * 8,
        )
        anchor = self.result.monthly_summary.filter(
            pl.col("cell_id") == SECONDARY.FULL_CELL_ID
        ).row(0, named=True)
        self.assertEqual(anchor["active_months"], 3)
        self.assertEqual(anchor["positive_months"], 3)
        self.assertAlmostEqual(anchor["positive_month_ratio"], 1.0)

        january = self.result.monthly_effects.filter(
            (pl.col("cell_id") == SECONDARY.FULL_CELL_ID)
            & (pl.col("month") == "202601")
        ).row(0, named=True)
        expected = self.result.daily_effects.filter(
            (pl.col("cell_id") == SECONDARY.FULL_CELL_ID)
            & pl.col("Date").str.starts_with("202601")
        )["daily_delta_z"].mean()
        self.assertAlmostEqual(january["month_mean_delta_z"], expected)

    def test_bartlett_newey_west_matches_manual_lag_one(self) -> None:
        result = SECONDARY.bartlett_newey_west([1.0, 2.0, 4.0, 8.0], 1)
        expected_lrv = 8.546875
        self.assertAlmostEqual(result["hac_long_run_variance"], expected_lrv)
        self.assertAlmostEqual(
            result["hac_standard_error"], math.sqrt(expected_lrv / 4.0)
        )

    def test_hac_reports_every_frozen_lag_without_selection(self) -> None:
        hac = self.result.hac_diagnostics
        self.assertEqual(len(hac), 8 * 4)
        self.assertEqual(set(hac["requested_lag"]), {1, 5, 10, 20})
        self.assertEqual(
            hac["kernel"].unique().to_list(), ["Bartlett Newey-West"]
        )
        self.assertEqual(
            hac["inference_role"].unique().to_list(),
            ["secondary_diagnostic_only"],
        )

    def test_temporal_intervals_are_three_by_three_and_basic(self) -> None:
        temporal = self.result.temporal_basic_intervals
        self.assertEqual(len(temporal), 3 * 3)
        self.assertEqual(
            set(temporal["block_length_requested"]), {5, 10, 20}
        )
        self.assertEqual(int(temporal["primary_block_length"].sum()), 3)
        self.assertTrue((temporal["interval_status"] == "ok").all())
        for row in temporal.iter_rows(named=True):
            self.assertAlmostEqual(
                row["one_sided_90_lcb"],
                row["point_delta_z"] - row["centered_error_q90"],
            )
            self.assertAlmostEqual(
                row["one_sided_90_ucb"],
                row["point_delta_z"] - row["centered_error_q10"],
            )

    def test_temporal_point_does_not_impute_zero_event_as_zero(self) -> None:
        first = self.result.temporal_basic_intervals.filter(
            (pl.col("block_id") == "T01")
            & (pl.col("block_length_requested") == 10)
        ).row(0, named=True)
        anchor = self.result.daily_effects.filter(
            (pl.col("cell_id") == SECONDARY.FULL_CELL_ID)
            & (pl.col("Date") >= first["block_start"])
            & (pl.col("Date") <= first["block_end"])
        )
        finite_mean = anchor["daily_delta_z"].mean()
        zero_imputed_mean = anchor["daily_delta_z"].fill_null(0.0).mean()
        self.assertAlmostEqual(first["point_delta_z"], finite_mean)
        self.assertNotAlmostEqual(first["point_delta_z"], zero_imputed_mean)

    def test_temporal_bootstrap_is_reproducible(self) -> None:
        second = SECONDARY.evaluate_secondary_diagnostics(
            runner_daily_stats=self.runner,
            requested_calendar=self.calendar,
            evaluator_daily_effects=self.evaluator_daily,
            evaluator_temporal_blocks=self.temporal_reference,
            evaluator_manifest=self.evaluator_manifest,
        )
        self.assertTrue(
            self.result.temporal_basic_intervals.equals(
                second.temporal_basic_intervals
            )
        )

    def test_circular_sampler_has_no_non_circular_endpoint_exclusion(self) -> None:
        indices = SECONDARY.circular_moving_block_indices(
            23,
            block_length=10,
            repetitions=50_000,
            seed=20_260_827,
        )
        inclusion = np.bincount(indices.ravel(), minlength=23)
        expected = indices.size / 23.0
        self.assertLess(float(np.max(np.abs(inclusion - expected))), expected * 0.02)
        self.assertTrue(((indices >= 0) & (indices < 23)).all())

    def test_runner_evaluator_z_mismatch_fails_closed(self) -> None:
        drifted = self.runner.with_columns(
            pl.when(
                (pl.col("Date") == synthetic_dates()[10])
                & (pl.col("candidate_id") == SECONDARY.FULL_CELL_ID)
            )
            .then(pl.col("condition_sum_z") + 0.5)
            .otherwise(pl.col("condition_sum_z"))
            .alias("condition_sum_z")
        )
        with self.assertRaisesRegex(
            SECONDARY.SecondaryDiagnosticsContractError,
            "condition_sum_z mismatch",
        ):
            SECONDARY.evaluate_secondary_diagnostics(
                runner_daily_stats=drifted,
                requested_calendar=self.calendar,
                evaluator_daily_effects=self.evaluator_daily,
                evaluator_temporal_blocks=self.temporal_reference,
                evaluator_manifest=self.evaluator_manifest,
            )

    def test_non_retrospective_calendar_is_rejected(self) -> None:
        calendar = self.calendar.with_columns(
            pl.lit("artifact_oos_non_pristine").alias("evidence_class")
        )
        with self.assertRaisesRegex(
            SECONDARY.SecondaryDiagnosticsContractError,
            "exactly retrospective_diagnostic",
        ):
            SECONDARY.prepare_requested_calendar(calendar)

    def test_temporal_reference_drift_is_rejected(self) -> None:
        drifted = self.temporal_reference.with_columns(
            pl.when(pl.col("block_id") == "T01")
            .then(pl.col("active_periods") + 1)
            .otherwise(pl.col("active_periods"))
            .alias("active_periods")
        )
        with self.assertRaisesRegex(
            SECONDARY.SecondaryDiagnosticsContractError,
            "active_periods drifted",
        ):
            SECONDARY.evaluate_secondary_diagnostics(
                runner_daily_stats=self.runner,
                requested_calendar=self.calendar,
                evaluator_daily_effects=self.evaluator_daily,
                evaluator_temporal_blocks=drifted,
                evaluator_manifest=self.evaluator_manifest,
            )

    def test_config_is_frozen(self) -> None:
        with self.assertRaisesRegex(ValueError, "bootstrap_repetitions"):
            SECONDARY.SecondaryDiagnosticsConfig(bootstrap_repetitions=200)
        with self.assertRaisesRegex(ValueError, "hac_lags"):
            SECONDARY.SecondaryDiagnosticsConfig(hac_lags=(5,))

    def test_writer_emits_versioned_schema_manifest_without_decision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "secondary_diagnostics_v1"
            SECONDARY.write_result(self.result, output_dir)
            expected_files = {
                "secondary_daily_effects.parquet",
                "equal_active_bp.parquet",
                "monthly_effects.parquet",
                "monthly_summary.parquet",
                "hac_diagnostics.parquet",
                "temporal_basic_intervals.parquet",
                "run_manifest.json",
            }
            self.assertEqual(
                {path.name for path in output_dir.iterdir()}, expected_files
            )
            manifest = json.loads(
                (output_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["contract"], SECONDARY.SECONDARY_CONTRACT_ID)
            self.assertTrue(
                manifest["decision_scope"]["secondary_diagnostics_only"]
            )
            self.assertFalse(
                manifest["decision_scope"]["deployment_decision_created"]
            )
            self.assertEqual(
                manifest["output_schemas"]["hac_diagnostics.parquet"],
                {
                    column: str(dtype)
                    for column, dtype in pl.read_parquet(
                        output_dir / "hac_diagnostics.parquet"
                    ).schema.items()
                },
            )

    def test_consumes_exact_retrospective_evaluator_result_schema(self) -> None:
        runner, calendar, _, _, _ = complete_fixture(periods=24)
        aggregate = runner.select(
            "Date",
            "candidate_id",
            "condition_n_z",
            "condition_sum_z",
            "control_n_z",
            "control_sum_z",
        )
        evaluator_result = EVALUATOR.evaluate_robustness(
            requested_calendar=calendar.drop("calendar_status"),
            daily_aggregates=aggregate,
            config=EVALUATOR.RobustnessConfig(),
        )
        result = SECONDARY.evaluate_secondary_diagnostics(
            runner_daily_stats=runner,
            requested_calendar=evaluator_result.requested_calendar,
            evaluator_daily_effects=evaluator_result.daily_effects,
            evaluator_temporal_blocks=evaluator_result.temporal_blocks,
            evaluator_manifest=evaluator_result.manifest,
        )
        self.assertEqual(len(result.daily_effects), 24 * 8)
        self.assertEqual(len(result.temporal_basic_intervals), 3 * 3)

    def test_artifact_oos_is_one_session_descriptive_only(self) -> None:
        cells = self.oos_result.cell_descriptive
        loo = self.oos_result.loo_descriptive
        self.assertEqual(len(cells), 8)
        self.assertEqual(len(loo), 3)
        self.assertEqual(cells["Date"].unique().to_list(), ["20260826"])
        self.assertEqual(
            cells["evidence_class"].unique().to_list(),
            ["artifact_oos_non_pristine"],
        )
        self.assertFalse(cells["inference_performed"].any())
        scope = self.oos_result.manifest["analysis_scope"]
        self.assertFalse(scope["hac_performed"])
        self.assertFalse(scope["bootstrap_performed"])
        self.assertFalse(scope["inference_performed"])
        self.assertFalse(
            self.oos_result.manifest["separation_policy"][
                "retrospective_pooled"
            ]
        )

    def test_artifact_oos_full_minus_loo_is_direct_difference(self) -> None:
        cells = self.oos_result.cell_descriptive
        depth = self.oos_result.loo_descriptive.filter(
            pl.col("contrast_id") == "D_increment"
        ).row(0, named=True)
        full = cells.filter(
            pl.col("cell_id") == SECONDARY.FULL_CELL_ID
        ).row(0, named=True)
        reduced = cells.filter(pl.col("cell_id") == "d0_g1_b1").row(
            0, named=True
        )
        self.assertAlmostEqual(
            depth["full_minus_loo_z"],
            full["daily_delta_z"] - reduced["daily_delta_z"],
        )
        self.assertAlmostEqual(
            depth["full_minus_loo_raw_bp"],
            full["daily_delta_raw_bp"]
            - reduced["daily_delta_raw_bp"],
        )
        self.assertAlmostEqual(
            depth["full_minus_loo_residual_bp"],
            full["daily_delta_residual_bp"]
            - reduced["daily_delta_residual_bp"],
        )

    def test_artifact_oos_writer_has_no_inference_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / SECONDARY.ARTIFACT_OOS_OUTPUT_NAME
            SECONDARY.write_artifact_oos_result(
                self.oos_result, output_dir
            )
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {
                    "cell_descriptive.parquet",
                    "loo_descriptive.parquet",
                    "run_manifest.json",
                },
            )
            names = {path.name.lower() for path in output_dir.iterdir()}
            self.assertFalse(any("hac" in name for name in names))
            self.assertFalse(any("bootstrap" in name for name in names))

    def test_artifact_oos_calendar_cannot_be_retrospective(self) -> None:
        drifted = self.oos_calendar.with_columns(
            pl.lit("retrospective_diagnostic").alias("evidence_class")
        )
        with self.assertRaisesRegex(
            SECONDARY.SecondaryDiagnosticsContractError,
            "identity/status drifted",
        ):
            SECONDARY.prepare_artifact_oos_calendar(drifted)


if __name__ == "__main__":
    unittest.main()
