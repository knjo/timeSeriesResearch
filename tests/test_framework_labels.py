from __future__ import annotations

from dataclasses import replace
import unittest

import polars as pl
from polars.testing import assert_frame_equal

from src.research.timeSeries.framework.labels import (
    CenterSpec,
    add_centered_labels,
    add_directional_labels,
)
from src.research.timeSeries.framework.profiles.benchmark import LONG_TAKER_TP109_V1
from src.research.timeSeries.framework.profiles.fhl import LONG_MAKER_TP109_V1
from src.research.timeSeries.framework.profiles.negfill import SHORT_TAKER_STOP108_V1


def _label_fixture() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Date": ["20260102", "20260102"],
            "QuoteCode": ["2330", "2330"],
            "BidPrice1": [100.0, 100.0],
            "AskPrice1": [101.0, 101.0],
            "Close": [90.0, 90.0],
            "FutureHigh": [108.0, 109.1],
            "RefPrice": [100.0, 100.0],
            "TickSize": [0.1, 0.1],
            "txf_beta_60d": [1.2, 1.2],
            "txf_to_1330_bp": [100.0, 100.0],
            "txf_residual_vol_to_1330": [0.02, 0.02],
        }
    )


class DirectionalLabelsTest(unittest.TestCase):
    def test_label_fingerprint_covers_the_full_economic_contract(self) -> None:
        identical = replace(SHORT_TAKER_STOP108_V1)
        changed_hedge = replace(
            SHORT_TAKER_STOP108_V1,
            hedge=replace(SHORT_TAKER_STOP108_V1.hedge, hedge_ratio=0.5),
        )
        self.assertEqual(
            SHORT_TAKER_STOP108_V1.fingerprint,
            identical.stable_hash(),
        )
        self.assertNotEqual(
            SHORT_TAKER_STOP108_V1.fingerprint,
            changed_hedge.fingerprint,
        )

    def test_short_profile_reproduces_stop_and_hedge_sign(self) -> None:
        result = add_directional_labels(
            _label_fixture(), spec=SHORT_TAKER_STOP108_V1
        )

        # Equality with the 1.08 barrier does not trigger the strict stop.
        self.assertAlmostEqual(result["label_exit_price"][0], 90.0)
        self.assertAlmostEqual(result["label_raw_bp"][0], 1_000.0)
        self.assertAlmostEqual(result["label_market_component_bp"][0], 120.0)
        self.assertAlmostEqual(result["label_hedge_pnl_bp"][0], 120.0)
        self.assertAlmostEqual(result["label_residual_bp"][0], 1_120.0)
        self.assertAlmostEqual(result["label_z"][0], 5.6)

        # Above the barrier exits at Ref*1.08 plus two adverse ticks.
        self.assertAlmostEqual(result["label_exit_price"][1], 108.2)
        self.assertAlmostEqual(result["label_raw_bp"][1], -820.0)
        self.assertAlmostEqual(result["label_residual_bp"][1], -700.0)
        self.assertAlmostEqual(result["label_z"][1], -3.5)
        self.assertTrue(result["label_valid"].all())

    def test_long_taker_and_maker_keep_distinct_entry_semantics(self) -> None:
        frame = _label_fixture()
        taker = add_directional_labels(frame, spec=LONG_TAKER_TP109_V1)
        maker = add_directional_labels(frame, spec=LONG_MAKER_TP109_V1)

        # The second row triggers the 1.09 take-profit.  A1 taker entry pays
        # the spread while B1 maker entry does not.
        expected_taker_raw = (109.0 - 101.0) / 101.0 * 10_000.0
        self.assertAlmostEqual(taker["label_entry_price"][1], 101.0)
        self.assertAlmostEqual(maker["label_entry_price"][1], 100.0)
        self.assertAlmostEqual(taker["label_raw_bp"][1], expected_taker_raw)
        self.assertAlmostEqual(maker["label_raw_bp"][1], 900.0)

        # A long residual removes the positive market component.
        self.assertAlmostEqual(taker["label_market_component_bp"][1], 120.0)
        self.assertAlmostEqual(taker["label_hedge_pnl_bp"][1], -120.0)
        self.assertAlmostEqual(
            taker["label_residual_bp"][1], expected_taker_raw - 120.0
        )
        self.assertAlmostEqual(maker["label_residual_bp"][1], 780.0)

    def test_long_take_profit_trigger_is_strict(self) -> None:
        frame = _label_fixture().with_columns(
            pl.lit(109.0).alias("FutureHigh")
        )
        result = add_directional_labels(frame, spec=LONG_TAKER_TP109_V1)
        self.assertTrue(result["label_exit_price"].is_close(90.0).all())

    def test_invalid_entry_and_vol_fail_closed_without_infinity(self) -> None:
        frame = _label_fixture().with_columns(
            pl.Series("BidPrice1", [0.0, 100.0]),
            pl.Series("txf_residual_vol_to_1330", [0.02, 0.0]),
        )
        result = add_directional_labels(frame, spec=SHORT_TAKER_STOP108_V1)
        self.assertEqual(result["label_valid"].to_list(), [False, False])
        self.assertEqual(result["label_z"].null_count(), 2)
        finite_count = result.select(
            pl.col("label_z").is_finite().fill_null(False).sum()
        ).item()
        self.assertEqual(finite_count, 0)

    def test_eager_and_lazy_results_match(self) -> None:
        frame = _label_fixture()
        eager = add_directional_labels(
            frame, spec=SHORT_TAKER_STOP108_V1
        )
        lazy = add_directional_labels(
            frame.lazy(), spec=SHORT_TAKER_STOP108_V1
        ).collect()
        assert_frame_equal(eager, lazy)

    def test_centered_labels_preserve_source_and_group_contracts(self) -> None:
        frame = pl.DataFrame(
            {
                "Date": ["20260102", "20260102", "20260102"],
                "QuoteCode": ["A", "A", "B"],
                "label_z": [1.0, 3.0, 10.0],
            }
        )
        result = add_centered_labels(frame)
        self.assertEqual(result["label_z"].to_list(), [1.0, 3.0, 10.0])
        expected_stock = [-1.0, 1.0, 0.0]
        expected_day = [
            1.0 - 14.0 / 3.0,
            3.0 - 14.0 / 3.0,
            10.0 - 14.0 / 3.0,
        ]
        for actual, expected in zip(
            result["label_z_within_stock_day"], expected_stock, strict=True
        ):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(
            result["label_z_within_day"], expected_day, strict=True
        ):
            self.assertAlmostEqual(actual, expected)

        custom = add_centered_labels(
            frame.lazy(),
            centers=(CenterSpec("centered_by_quote", ("QuoteCode",)),),
        ).collect()
        self.assertEqual(custom["centered_by_quote"].to_list(), expected_stock)

    def test_missing_inputs_are_reported_before_execution(self) -> None:
        with self.assertRaisesRegex(ValueError, "label input missing columns"):
            add_directional_labels(
                pl.DataFrame({"BidPrice1": [100.0]}),
                spec=SHORT_TAKER_STOP108_V1,
            )


if __name__ == "__main__":
    unittest.main()
