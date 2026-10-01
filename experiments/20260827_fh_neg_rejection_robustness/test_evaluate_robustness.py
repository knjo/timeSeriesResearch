from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import polars as pl


MODULE_PATH = Path(__file__).with_name("evaluate_robustness.py")
SPEC = importlib.util.spec_from_file_location("fh_neg_robustness_evaluator", MODULE_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise ImportError(MODULE_PATH)
EVALUATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EVALUATOR
SPEC.loader.exec_module(EVALUATOR)


def aggregate_fixture(periods: int = 36) -> pl.DataFrame:
    start = dt.date(2026, 1, 1)
    rows = []
    for day_index in range(periods):
        date = (start + dt.timedelta(days=day_index)).strftime("%Y%m%d")
        noise = ((day_index % 5) - 2) * 0.003
        for depth, far, bear in EVALUATOR.CELL_SPECS:
            support = 20 - 2 * (depth + far + bear)
            if day_index == 0 and (depth, far, bear) == (1, 1, 1):
                support = 0
            effect = 0.01 + 0.10 * depth + 0.05 * far + 0.02 * bear + noise
            rows.append(
                {
                    "Date": date,
                    "use_depth": depth,
                    "use_far": far,
                    "use_bear": bear,
                    "condition_n_z": support,
                    "condition_sum_z": support * effect,
                    "control_n_z": 100,
                    "control_sum_z": 0.0,
                    "eligible_pool_rows": 1_000,
                }
            )
    return pl.DataFrame(rows)


def requested_calendar_fixture(periods: int = 36) -> pl.DataFrame:
    start = dt.date(2026, 1, 1)
    return pl.DataFrame(
        {
            "Date": [
                (start + dt.timedelta(days=index)).strftime("%Y%m%d")
                for index in range(periods)
            ],
            "source_complete": [True] * periods,
            "run_status": ["success"] * periods,
            "evidence_class": ["retrospective_diagnostic"] * periods,
        }
    )


class RobustnessEvaluatorTest(unittest.TestCase):
    def config(self, **overrides):
        values = {
            "bootstrap_reps": 200,
        }
        values.update(overrides)
        return EVALUATOR.RobustnessConfig(**values)

    def test_daily_aggregates_keep_zero_support_effect_undefined(self) -> None:
        daily = EVALUATOR.prepare_daily_aggregates(aggregate_fixture())
        row = daily.filter(
            (pl.col("Date") == "20260101")
            & (pl.col("cell_id") == EVALUATOR.FULL_CELL_ID)
        ).row(0, named=True)
        self.assertEqual(row["condition_n_z"], 0)
        self.assertFalse(row["condition_active"])
        self.assertFalse(row["effect_active"])
        self.assertIsNone(row["daily_delta_z"])
        self.assertEqual(len(daily), 36 * 8)

    def test_equal_day_summary_does_not_row_pool(self) -> None:
        frame = aggregate_fixture()
        frame = frame.with_columns(
            pl.when(
                (pl.col("Date") == "20260102")
                & (pl.col("use_depth") == 0)
                & (pl.col("use_far") == 0)
                & (pl.col("use_bear") == 0)
            )
            .then(1_000)
            .otherwise(pl.col("condition_n_z"))
            .alias("condition_n_z")
        ).with_columns(
            pl.when(
                (pl.col("Date") == "20260102")
                & (pl.col("use_depth") == 0)
                & (pl.col("use_far") == 0)
                & (pl.col("use_bear") == 0)
            )
            .then(pl.col("condition_n_z") * 1.0)
            .otherwise(pl.col("condition_sum_z"))
            .alias("condition_sum_z"),
            pl.lit(2_000).alias("control_n_z"),
            pl.lit(2_000).alias("eligible_pool_rows"),
        )
        result = EVALUATOR.evaluate_robustness(
            requested_calendar=requested_calendar_fixture(),
            daily_aggregates=frame,
            config=self.config(),
        )
        daily = result.daily_effects.filter(pl.col("cell_id") == "d0_g0_b0")
        expected = daily["daily_delta_z"].mean()
        actual = result.cell_summary.filter(pl.col("cell_id") == "d0_g0_b0")[
            "equal_day_delta_z"
        ].item()
        self.assertAlmostEqual(actual, expected)
        row_pooled = daily["condition_sum_z"].sum() / daily["condition_n_z"].sum()
        self.assertNotAlmostEqual(actual, row_pooled)

    def test_event_cache_builds_nested_eight_cells_and_zero_day(self) -> None:
        events = pl.DataFrame(
            {
                "Date": ["20260101", "20260101", "20260101"],
                "QuoteCode": ["A", "A", "B"],
                "ChannelSeq": [1, 2, 1],
                "label_z": [1.0, 2.0, 3.0],
                "passes_depth": [True, True, False],
                "passes_far": [True, True, False],
                "passes_bear": [True, False, False],
            }
        )
        control = pl.DataFrame(
            {
                "Date": ["20260101", "20260102"],
                "control_n_z": [10, 10],
                "control_sum_z": [0.0, 0.0],
                "eligible_pool_rows": [100, 100],
            }
        )
        daily = EVALUATOR.build_daily_aggregates_from_event_cache(events, control)
        baseline = daily.filter(
            (pl.col("Date") == "20260101")
            & (pl.col("cell_id") == "d0_g0_b0")
        ).row(0, named=True)
        full = daily.filter(
            (pl.col("Date") == "20260101")
            & (pl.col("cell_id") == "d1_g1_b1")
        ).row(0, named=True)
        zero = daily.filter(
            (pl.col("Date") == "20260102")
            & (pl.col("cell_id") == "d0_g0_b0")
        ).row(0, named=True)
        self.assertEqual(baseline["condition_n_z"], 3)
        self.assertAlmostEqual(baseline["daily_delta_z"], 2.0)
        self.assertEqual(full["condition_n_z"], 1)
        self.assertAlmostEqual(full["daily_delta_z"], 1.0)
        self.assertEqual(zero["condition_n_z"], 0)
        self.assertIsNone(zero["daily_delta_z"])

    def test_leave_one_out_uses_only_common_active_days(self) -> None:
        result = EVALUATOR.evaluate_robustness(
            requested_calendar=requested_calendar_fixture(),
            daily_aggregates=aggregate_fixture(),
            config=self.config(),
        )
        depth = result.loo_summary.filter(pl.col("removed_gate") == "depth").row(
            0, named=True
        )
        self.assertEqual(depth["common_active_periods"], 35)
        self.assertEqual(depth["contrast_id"], "D_increment")
        self.assertAlmostEqual(depth["paired_increment_delta_z"], 0.10)
        self.assertAlmostEqual(depth["median_paired_increment_delta_z"], 0.10)
        self.assertGreaterEqual(depth["bootstrap_probability_gt_zero"], 0.80)
        first = result.loo_daily.filter(
            (pl.col("Date") == "20260101")
            & (pl.col("removed_gate") == "depth")
        ).row(0, named=True)
        self.assertFalse(first["common_active"])
        self.assertIsNone(first["paired_increment_delta_z"])

        factorial = result.factorial_contrasts
        counts = {
            str(row["contrast_kind"]): int(row["len"])
            for row in factorial.group_by("contrast_kind")
            .len()
            .iter_rows(named=True)
        }
        self.assertEqual(counts["conditional_main_increment"], 12)
        self.assertEqual(counts["conditional_two_way_interaction"], 6)
        self.assertEqual(counts["three_way_interaction"], 1)
        d_corner = factorial.filter(
            pl.col("contrast_id") == "D_at_G1_B1"
        ).row(0, named=True)
        self.assertEqual(d_corner["g4_contrast_id"], "D_increment")
        self.assertAlmostEqual(
            d_corner["paired_contrast_delta_z"],
            depth["paired_increment_delta_z"],
        )
        self.assertIsNotNone(d_corner["bootstrap_probability_gt_zero"])

    def test_bootstrap_is_reproducible_and_reports_sensitivity(self) -> None:
        config = self.config()
        first = EVALUATOR.evaluate_robustness(
            requested_calendar=requested_calendar_fixture(),
            daily_aggregates=aggregate_fixture(),
            config=config,
        )
        second = EVALUATOR.evaluate_robustness(
            requested_calendar=requested_calendar_fixture(),
            daily_aggregates=aggregate_fixture(),
            config=config,
        )
        self.assertTrue(first.bootstrap.equals(second.bootstrap))
        self.assertEqual(
            set(first.bootstrap["block_length_requested"].to_list()),
            {5, 10, 20},
        )
        main = first.bootstrap.filter(pl.col("primary_block_length"))
        self.assertEqual(main["block_length_requested"].unique().to_list(), [10])

    def test_family_maxstat_is_at_least_as_conservative_as_raw(self) -> None:
        result = EVALUATOR.evaluate_robustness(
            requested_calendar=requested_calendar_fixture(),
            daily_aggregates=aggregate_fixture(),
            config=self.config(),
        )
        family = result.fwer_audit.filter(
            (pl.col("family_id") == self.config().family_id)
            & pl.col("primary_block_length")
            & pl.col("raw_block_bootstrap_p").is_not_null()
        )
        self.assertEqual(len(family), 8)
        self.assertTrue(
            family.select(
                (
                    pl.col("single_step_fwer_p")
                    >= pl.col("raw_block_bootstrap_p")
                ).all()
            ).item()
        )
        self.assertEqual(
            family["method"].unique().to_list(),
            ["circular_moving_block_single_step_max_t_one_sided"],
        )
        self.assertEqual(family["family_alpha"].unique().to_list(), [0.1])
        self.assertTrue(family["simultaneous_lower_bound"].is_not_null().all())
        self.assertTrue((family["joint_valid_bootstrap_rate"] >= 0.90).all())
        self.assertTrue(
            (family["joint_valid_bootstrap_repetitions"] >= 100).all()
        )
        self.assertEqual(
            family["effective_trial_method"].unique().to_list(),
            ["joint_finite_calendar_psd_corr_eigen_participation_ratio"],
        )
        self.assertEqual(
            family["effective_trial_common_finite_periods"].unique().to_list(),
            [35],
        )

    def test_temporal_blocks_cover_cells_and_loo_on_same_calendar(self) -> None:
        result = EVALUATOR.evaluate_robustness(
            requested_calendar=requested_calendar_fixture(),
            daily_aggregates=aggregate_fixture(),
            config=self.config(),
        )
        self.assertEqual(len(result.temporal_blocks), 3 * (8 + 3))
        block_periods = (
            result.temporal_blocks.filter(
                (pl.col("target_kind") == "cell_effect")
                & (pl.col("target_id") == EVALUATOR.FULL_CELL_ID)
            )["calendar_periods"].sum()
        )
        self.assertEqual(block_periods, 36)
        first = result.temporal_blocks.filter(
            (pl.col("block_id") == "T01")
            & (pl.col("target_kind") == "cell_effect")
            & (pl.col("target_id") == EVALUATOR.FULL_CELL_ID)
        ).row(0, named=True)
        self.assertEqual(first["zero_event_periods"], 1)

    def test_trial_ledger_is_complete_and_outcome_free_fingerprint_is_stable(self) -> None:
        result = EVALUATOR.evaluate_robustness(
            requested_calendar=requested_calendar_fixture(),
            daily_aggregates=aggregate_fixture(),
            config=self.config(),
            trial_metadata={
                EVALUATOR.FULL_CELL_ID: {"depth_ratio": -8.0, "far_ticks": 4.0}
            },
        )
        ledger = result.trial_ledger
        self.assertEqual(len(ledger), 8)
        self.assertEqual(ledger["fingerprint"].n_unique(), 8)
        self.assertEqual(int(ledger["selected"].sum()), 1)
        selected = ledger.filter(pl.col("selected")).row(0, named=True)
        self.assertEqual(selected["trial_id"], EVALUATOR.FULL_CELL_ID)
        self.assertIn('"depth_ratio":-8.0', selected["parameter_json"])
        self.assertFalse(result.manifest["dsr_used"])
        self.assertEqual(result.manifest["effect_zero_support_policy"], "undefined_not_zero")
        scope = result.manifest["selection_bias_scope"]
        self.assertFalse(scope["historical_selection_exposure_included"])
        self.assertTrue(
            scope[
                "raw_and_effective_trial_counts_are_lower_bounds_for_total_selection_exposure"
            ]
        )
        self.assertTrue(selected["total_selection_exposure_is_lower_bound"])

        with self.assertRaisesRegex(
            EVALUATOR.RobustnessContractError,
            "non-parameter fields",
        ):
            EVALUATOR.evaluate_robustness(
                requested_calendar=requested_calendar_fixture(),
                daily_aggregates=aggregate_fixture(),
                config=self.config(),
                trial_metadata={
                    EVALUATOR.FULL_CELL_ID: {"observed_delta_z": 9.9}
                },
            )

    def test_candidate_id_only_aggregate_input_uses_frozen_ids(self) -> None:
        frame = aggregate_fixture().with_columns(
            pl.struct(EVALUATOR.GATE_COLUMNS)
            .map_elements(
                lambda row: EVALUATOR.canonical_cell_id(
                    row["use_depth"], row["use_far"], row["use_bear"]
                ),
                return_dtype=pl.String,
            )
            .alias("candidate_id")
        ).drop(EVALUATOR.GATE_COLUMNS)
        daily = EVALUATOR.prepare_daily_aggregates(frame)
        self.assertEqual(set(daily["candidate_id"]), set(EVALUATOR.canonical_cell_ids()))
        self.assertTrue((daily["candidate_id"] == daily["cell_id"]).all())

    def test_nested_support_violation_is_rejected(self) -> None:
        frame = aggregate_fixture().with_columns(
            pl.when(
                (pl.col("Date") == "20260102")
                & (pl.col("use_depth") == 1)
                & (pl.col("use_far") == 1)
                & (pl.col("use_bear") == 1)
            )
            .then(25)
            .otherwise(pl.col("condition_n_z"))
            .alias("condition_n_z")
        ).with_columns(
            pl.when(
                (pl.col("Date") == "20260102")
                & (pl.col("use_depth") == 1)
                & (pl.col("use_far") == 1)
                & (pl.col("use_bear") == 1)
            )
            .then(pl.col("condition_n_z") * 0.18)
            .otherwise(pl.col("condition_sum_z"))
            .alias("condition_sum_z")
        )
        with self.assertRaisesRegex(
            EVALUATOR.RobustnessContractError, "nested support violation"
        ):
            EVALUATOR.prepare_daily_aggregates(frame)

    def test_incomplete_cell_calendar_is_rejected(self) -> None:
        incomplete = aggregate_fixture().slice(1)
        with self.assertRaisesRegex(
            EVALUATOR.RobustnessContractError, "incomplete eight-cell"
        ):
            EVALUATOR.prepare_daily_aggregates(incomplete)

    def test_shared_control_disagreement_is_rejected(self) -> None:
        frame = aggregate_fixture().with_columns(
            pl.when(
                (pl.col("Date") == "20260101")
                & (pl.col("use_depth") == 1)
                & (pl.col("use_far") == 1)
                & (pl.col("use_bear") == 1)
            )
            .then(101)
            .otherwise(pl.col("control_n_z"))
            .alias("control_n_z")
        )
        with self.assertRaisesRegex(
            EVALUATOR.RobustnessContractError, "shared control differs"
        ):
            EVALUATOR.prepare_daily_aggregates(frame)

    def test_circular_blocks_remove_edge_inclusion_bias(self) -> None:
        values = np.zeros(23, dtype=float)
        values[0] = 100.0
        centered = values - values.mean()
        indices = EVALUATOR.moving_block_indices(
            len(centered),
            block_length=10,
            repetitions=50_000,
            seed=EVALUATOR.FROZEN_BOOTSTRAP_SEED,
        )
        bootstrap_means = centered[indices].mean(axis=1)
        self.assertLess(abs(float(bootstrap_means.mean())), 0.10)
        self.assertTrue(((indices >= 0) & (indices < len(centered))).all())

    def test_calendar_retains_missing_failed_and_zero_control_statuses(self) -> None:
        calendar = requested_calendar_fixture().with_columns(
            pl.when(pl.col("Date") == "20260102")
            .then(False)
            .otherwise(pl.col("source_complete"))
            .alias("source_complete"),
            pl.when(pl.col("Date") == "20260103")
            .then(pl.lit("failed"))
            .otherwise(pl.col("run_status"))
            .alias("run_status"),
        )
        daily = aggregate_fixture().filter(
            ~pl.col("Date").is_in(["20260102", "20260103"])
        )
        result = EVALUATOR.evaluate_robustness(
            requested_calendar=calendar,
            daily_aggregates=daily,
            config=self.config(),
        )
        missing = result.daily_effects.filter(
            (pl.col("Date") == "20260102")
            & (pl.col("cell_id") == EVALUATOR.FULL_CELL_ID)
        ).row(0, named=True)
        failed = result.daily_effects.filter(
            (pl.col("Date") == "20260103")
            & (pl.col("cell_id") == EVALUATOR.FULL_CELL_ID)
        ).row(0, named=True)
        self.assertEqual(missing["period_status"], "missing_source")
        self.assertEqual(failed["period_status"], "run_failed")
        self.assertIsNone(missing["condition_n_z"])
        self.assertIsNone(failed["daily_delta_z"])
        self.assertEqual(len(result.daily_effects), 36 * 8)
        summary = result.cell_summary.filter(
            pl.col("cell_id") == EVALUATOR.FULL_CELL_ID
        ).row(0, named=True)
        self.assertEqual(summary["missing_source_periods"], 1)
        self.assertEqual(summary["failed_run_periods"], 1)

    def test_successful_calendar_date_cannot_silently_disappear(self) -> None:
        daily = aggregate_fixture().filter(pl.col("Date") != "20260102")
        with self.assertRaisesRegex(
            EVALUATOR.RobustnessContractError,
            "successful requested dates missing",
        ):
            EVALUATOR.evaluate_robustness(
                requested_calendar=requested_calendar_fixture(),
                daily_aggregates=daily,
                config=self.config(),
            )

    def test_one_run_cannot_pool_evidence_classes(self) -> None:
        calendar = pl.DataFrame(
            {
                "Date": ["20260825", "20260826"],
                "source_complete": [True, True],
                "run_status": ["success", "success"],
                "evidence_class": [
                    "retrospective_diagnostic",
                    "artifact_oos_non_pristine",
                ],
            }
        )
        with self.assertRaisesRegex(
            EVALUATOR.RobustnessContractError,
            "exactly one evidence_class",
        ):
            EVALUATOR.prepare_requested_calendar(calendar)

    def test_zero_control_is_not_zero_event(self) -> None:
        frame = aggregate_fixture().with_columns(
            pl.when(pl.col("Date") == "20260103")
            .then(0)
            .otherwise(pl.col("condition_n_z"))
            .alias("condition_n_z"),
            pl.when(pl.col("Date") == "20260103")
            .then(0.0)
            .otherwise(pl.col("condition_sum_z"))
            .alias("condition_sum_z"),
            pl.when(pl.col("Date") == "20260103")
            .then(0)
            .otherwise(pl.col("control_n_z"))
            .alias("control_n_z"),
            pl.when(pl.col("Date") == "20260103")
            .then(0.0)
            .otherwise(pl.col("control_sum_z"))
            .alias("control_sum_z"),
        )
        result = EVALUATOR.evaluate_robustness(
            requested_calendar=requested_calendar_fixture(),
            daily_aggregates=frame,
            config=self.config(),
        )
        statuses = set(
            result.daily_effects.filter(pl.col("Date") == "20260103")[
                "period_status"
            ]
        )
        self.assertEqual(statuses, {"zero_control_ineligible"})
        full = result.cell_summary.filter(
            pl.col("cell_id") == EVALUATOR.FULL_CELL_ID
        ).row(0, named=True)
        self.assertEqual(full["zero_control_ineligible_periods"], 1)
        self.assertEqual(full["selection_rate_periods"], 35)

    def test_frozen_identity_cannot_be_overridden(self) -> None:
        with self.assertRaisesRegex(ValueError, "family_id is frozen"):
            EVALUATOR.RobustnessConfig(family_id="other")
        with self.assertRaisesRegex(ValueError, "selected_cell_id is frozen"):
            EVALUATOR.RobustnessConfig(selected_cell_id="d0_g0_b0")

    def test_maxstat_fails_closed_when_joint_valid_rate_is_too_low(self) -> None:
        rng = np.random.default_rng(7)
        matrix = rng.normal(size=(100, 8))
        matrix[:, -1] = np.nan
        matrix[[0, 50], -1] = [0.0, 1.0]
        with self.assertRaisesRegex(
            EVALUATOR.RobustnessContractError,
            "joint-valid max-stat bootstrap insufficient",
        ):
            EVALUATOR._maxstat_rows(
                matrix,
                EVALUATOR.canonical_cell_ids(),
                family_id=EVALUATOR.FROZEN_FAMILY_ID,
                config=self.config(),
            )


if __name__ == "__main__":
    unittest.main()
