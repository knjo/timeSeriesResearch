from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path

import numpy as np
from scipy import stats


MODULE_PATH = Path(__file__).with_name("analysis_utils.py")
SPEC = importlib.util.spec_from_file_location("netfill_long_cover_analysis", MODULE_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot import {MODULE_PATH}")
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


class CalendarMetricTests(unittest.TestCase):
    def test_calendar_sharpe_uses_sample_dispersion(self) -> None:
        values = np.array([1.0, 2.0, 3.0, -1.0])
        expected = np.mean(values) / np.std(values, ddof=1) * np.sqrt(252.0)
        self.assertTrue(np.isclose(ANALYSIS.calendar_sharpe(values), expected))

    def test_zero_variance_sharpe_is_undefined(self) -> None:
        self.assertTrue(math.isnan(ANALYSIS.calendar_sharpe([1.0, 1.0])))

    def test_max_drawdown_includes_initial_zero(self) -> None:
        self.assertTrue(
            np.isclose(ANALYSIS.max_drawdown([-2.0, 1.0, -3.0, 5.0]), -4.0)
        )

    def test_expected_shortfall_uses_worst_ceiling_count(self) -> None:
        values = np.arange(21.0)
        self.assertTrue(
            np.isclose(
                ANALYSIS.historical_expected_shortfall(values, tail_fraction=0.05),
                0.5,
            )
        )


class MovingBlockMaxTTests(unittest.TestCase):
    @staticmethod
    def _trials() -> dict[str, np.ndarray]:
        rng = np.random.default_rng(71)
        return {
            "weak": rng.normal(0.0, 1.0, 80),
            "strong": rng.normal(0.45, 1.0, 80),
            "adverse": rng.normal(-0.20, 1.0, 80),
        }

    def test_max_t_is_deterministic_and_familywise_conservative(self) -> None:
        first = ANALYSIS.moving_block_max_t(
            self._trials(),
            block_length=5,
            repetitions=999,
            seed=22,
        )
        second = ANALYSIS.moving_block_max_t(
            dict(reversed(list(self._trials().items()))),
            block_length=5,
            repetitions=999,
            seed=22,
        )
        self.assertEqual(first, second)
        for row in first["trials"].values():
            self.assertGreaterEqual(
                row["max_t_adjusted_p"] + 1e-15,
                row["raw_one_sided_p"],
            )
        self.assertLess(
            first["trials"]["strong"]["max_t_adjusted_p"],
            0.10,
        )

    def test_constant_trial_is_retained_as_non_estimable(self) -> None:
        result = ANALYSIS.moving_block_max_t(
            {
                "varying": np.arange(12.0),
                "constant": np.zeros(12),
            },
            repetitions=99,
        )
        constant = result["trials"]["constant"]
        self.assertFalse(constant["estimable"])
        self.assertTrue(np.isclose(constant["max_t_adjusted_p"], 1.0))

    def test_unequal_calendars_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal length"):
            ANALYSIS.moving_block_max_t({"a": [1.0, 2.0], "b": [1.0]})


class SharpeProbabilityTests(unittest.TestCase):
    def test_probabilistic_sharpe_matches_bailey_formula(self) -> None:
        values = np.array([0.4, -0.2, 0.1, 0.5, -0.1, 0.3, 0.2, -0.3])
        result = ANALYSIS.probabilistic_sharpe_ratio(values)
        daily_sharpe = np.mean(values) / np.std(values, ddof=1)
        skewness = stats.skew(values, bias=False)
        kurtosis = stats.kurtosis(values, fisher=False, bias=False)
        denominator = np.sqrt(
            (
                1.0
                - skewness * daily_sharpe
                + ((kurtosis - 1.0) / 4.0) * daily_sharpe**2
            )
            / (len(values) - 1)
        )
        expected = stats.norm.cdf(daily_sharpe / denominator)
        self.assertTrue(np.isclose(result["psr_probability"], expected))
        self.assertTrue(np.isclose(result["sample_pearson_kurtosis"], kurtosis))

    def test_more_trials_raise_dsr_benchmark_and_lower_probability(self) -> None:
        rng = np.random.default_rng(312)
        values = rng.normal(0.08, 1.0, 180)
        results = ANALYSIS.deflated_sharpe_sensitivity(
            values,
            trial_counts=(6, 28),
        )
        self.assertGreater(
            results[28]["dsr_benchmark_annualized_sharpe"],
            results[6]["dsr_benchmark_annualized_sharpe"],
        )
        self.assertLessEqual(
            results[28]["dsr_probability"],
            results[6]["dsr_probability"] + 1e-15,
        )
        self.assertTrue(results[6]["trial_sharpe_std_fallback_used"])

    def test_empirical_trial_dispersion_is_preserved(self) -> None:
        rng = np.random.default_rng(91)
        values = rng.normal(0.1, 1.0, 100)
        result = ANALYSIS.deflated_sharpe_ratio(
            values,
            trial_count=6,
            trial_sharpe_std_annualized=0.7,
        )
        self.assertTrue(np.isclose(result["trial_sharpe_std_annualized"], 0.7))
        self.assertFalse(result["trial_sharpe_std_fallback_used"])


class CoverPortfolioTests(unittest.TestCase):
    def test_position_only_normalization_uses_aggregate_capital(self) -> None:
        scale = ANALYSIS.position_only_normalization(
            [100.0, 100.0, 100.0, 100.0],
            [50.0, 50.0, 50.0, 50.0],
        )
        self.assertTrue(np.isclose(scale, 2.0))

    def test_fixed_weight_cover_metrics(self) -> None:
        result = ANALYSIS.cover_portfolio_metrics(
            [-10.0, 5.0, -5.0, 10.0],
            [5.0, -1.0, 4.0, -2.0],
            short_position=[100.0] * 4,
            long_position=[50.0] * 4,
            native_long_multipliers=(0.0, 0.5, 1.0),
        )
        self.assertTrue(np.isclose(result["long_position_normalization_scale"], 2.0))
        diagnostics = result["native_cover_diagnostics"]
        self.assertTrue(
            np.isclose(
                diagnostics["mean_native_long_pnl_on_short_loss_days"],
                4.5,
            )
        )
        self.assertTrue(
            np.isclose(
                diagnostics["mean_native_long_pnl_on_worst_short_days"],
                5.0,
            )
        )
        self.assertTrue(
            np.isclose(
                diagnostics["downside_beta_long_on_short_loss_days"],
                -0.2,
            )
        )
        baseline = result["native_cash_frontier"][0]
        self.assertTrue(np.isclose(baseline["native_long_multiplier_h"], 0.0))
        self.assertTrue(np.isclose(baseline["total_pnl"], 0.0))
        anchor = result["native_cash_frontier"][1]
        self.assertTrue(np.isclose(anchor["native_long_multiplier_h"], 0.5))
        self.assertTrue(np.isclose(anchor["total_pnl"], 3.0))
        self.assertTrue(np.isclose(anchor["max_drawdown"], -7.5))
        self.assertGreater(anchor["max_drawdown_delta_vs_short"], 0.0)
        normalized_anchor = result["position_normalized_frontier"][1]
        self.assertTrue(np.isclose(normalized_anchor["total_pnl"], 6.0))
        self.assertTrue(np.isclose(normalized_anchor["max_drawdown"], -5.0))
        self.assertEqual(result["position_normalization_role"], "secondary_descriptive_only")

    def test_cover_requires_calendar_alignment(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal length"):
            ANALYSIS.cover_portfolio_metrics(
                [1.0, 2.0],
                [1.0],
                short_position=2.0,
                long_position=1.0,
            )


if __name__ == "__main__":
    unittest.main()
