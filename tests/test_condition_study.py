from __future__ import annotations

import unittest

import polars as pl

from src.research.timeSeries.framework.condition_study import (
    ConditionComparisonSpec,
    ConditionStudyContractError,
    PrimaryEstimand,
    compare_condition_to_control,
    compare_conditions_to_shared_control,
    summarize_condition_day,
)


def _pool() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Date": ["20260102"] * 4,
            "QuoteCode": ["A", "A", "B", "B"],
            "ChannelSeq": [1, 2, 1, 2],
            "label_raw_bp": [10.0, 20.0, -10.0, 30.0],
            "label_market_component_bp": [1.0, 2.0, 3.0, 4.0],
            "label_hedge_pnl_bp": [-1.0, -2.0, -3.0, -4.0],
            "label_residual_bp": [9.0, 18.0, -13.0, 26.0],
            "label_z": [0.9, 1.8, -1.3, 2.6],
            "bucket": ["x", "x", "y", "y"],
        }
    )


class ConditionStudyTest(unittest.TestCase):
    def test_primary_estimand_is_explicit_and_validated(self) -> None:
        default = ConditionComparisonSpec("example", "1")
        matched = ConditionComparisonSpec(
            "example",
            "1",
            primary_estimand="same_stock_time",
        )
        self.assertEqual(default.primary_estimand, PrimaryEstimand.SHARED_POOL)
        self.assertEqual(
            matched.primary_estimand,
            PrimaryEstimand.SAME_STOCK_TIME,
        )
        with self.assertRaises(ValueError):
            ConditionComparisonSpec(
                "example",
                "1",
                primary_estimand="five_minute",  # type: ignore[arg-type]
            )

    def test_multiple_conditions_share_one_full_control(self) -> None:
        results = compare_conditions_to_shared_control(
            "20260102",
            _pool(),
            {
                "first": pl.DataFrame(
                    {"QuoteCode": ["A", "Z"], "ChannelSeq": [1, 9]}
                ),
                "second": pl.DataFrame(
                    {"QuoteCode": ["A", "B"], "ChannelSeq": [2, 2]}
                ),
            },
            spec=ConditionComparisonSpec(
                "family",
                "1",
                control_sample_size=None,
            ),
        )
        self.assertEqual(set(results), {"first", "second"})
        self.assertEqual(results["first"].condition_rows, 1)
        self.assertEqual(results["first"].condition_excluded_rows, 1)
        self.assertEqual(results["second"].condition_rows, 2)
        self.assertEqual(results["first"].control_rows, 4)
        self.assertTrue(
            results["first"].control.equals(results["second"].control)
        )
        self.assertTrue(results["first"].control.equals(_pool()))

    def test_condition_and_control_are_split_from_one_final_pool(self) -> None:
        keys = pl.DataFrame({"QuoteCode": ["A", "Z"], "ChannelSeq": [2, 9]})
        result = compare_condition_to_control(
            "20260102",
            _pool(),
            keys,
            spec=ConditionComparisonSpec(
                "example", "1", control_sample_size=100
            ),
        )
        self.assertEqual(result.eligible_pool_rows, 4)
        self.assertEqual(result.indexed_condition_rows, 2)
        self.assertEqual(result.condition_rows, 1)
        self.assertEqual(result.condition_excluded_rows, 1)
        self.assertEqual(result.control_rows, 4)
        self.assertAlmostEqual(result.selection_rate, 0.25)
        self.assertEqual(result.condition["ChannelSeq"].to_list(), [2])

        condition, control = summarize_condition_day(
            result, bucket_columns=("bucket",)
        )
        self.assertEqual(condition["sum_z"], 1.8)
        self.assertEqual(condition["n_eligible_pool"], 4)
        self.assertEqual(control["n"], 4)

    def test_non_event_control_is_explicit(self) -> None:
        keys = pl.DataFrame({"QuoteCode": ["A"], "ChannelSeq": [2]})
        result = compare_condition_to_control(
            "20260102",
            _pool(),
            keys,
            spec=ConditionComparisonSpec(
                "example",
                "1",
                control_sample_size=100,
                control_population="non_event_pool",
            ),
        )
        self.assertEqual(result.control_rows, 3)
        self.assertNotIn(2, result.control.filter(pl.col("QuoteCode") == "A")["ChannelSeq"])

    def test_duplicate_event_keys_fail_closed(self) -> None:
        duplicate = pl.DataFrame(
            {"QuoteCode": ["A", "A"], "ChannelSeq": [1, 1]}
        )
        with self.assertRaisesRegex(ConditionStudyContractError, "duplicate"):
            compare_condition_to_control(
                "20260102",
                _pool(),
                duplicate,
                spec=ConditionComparisonSpec("example", "1"),
            )


if __name__ == "__main__":
    unittest.main()
