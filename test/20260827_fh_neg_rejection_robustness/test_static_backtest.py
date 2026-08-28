from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import polars as pl


RUNNER_PATH = Path(__file__).with_name("run_static_backtest.py")
SPEC = importlib.util.spec_from_file_location("fh_static_backtest_tested", RUNNER_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot import {RUNNER_PATH}")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class StaticBacktestContractTests(unittest.TestCase):
    def test_legal_tick_size_boundaries(self) -> None:
        prices = [9.9, 10.0, 49.9, 50.0, 99.9, 100.0, 499.5, 500.0, 999.0, 1000.0]
        expected = [0.01, 0.05, 0.05, 0.1, 0.1, 0.5, 0.5, 1.0, 1.0, 5.0]
        actual = (
            pl.DataFrame({"price": prices})
            .select(RUNNER.legal_tick_size(pl.col("price")).alias("tick"))["tick"]
            .to_list()
        )
        self.assertEqual(actual, expected)

    def test_rounded_pool_boundary_is_strict(self) -> None:
        ratios = [-0.0150004, -0.0149994, 0.0499994, 0.0499996]
        frame = pl.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "RefPrice": [100.0] * 4,
                "BidPrice1": [100.0 * (1.0 + ratio) for ratio in ratios],
                "ToRef": ratios,
            }
        )
        kept = RUNNER.apply_robustness_pool_boundary(frame)
        self.assertEqual(kept["id"].to_list(), [1, 2])

    def test_short_barrier_label_uses_recomputed_tick(self) -> None:
        frame = pl.DataFrame(
            {
                "BidPrice1": [99.9],
                "Close": [98.0],
                "FutureHigh": [109.0],
                "RefPrice": [100.0],
                "TickSize": [0.5],
                "label_raw_bp": [-100.0],
                "label_residual_bp": [-98.0],
                "label_z": [-0.98],
                "txf_residual_vol_to_1330": [0.01],
            }
        )
        output = RUNNER.recompute_robustness_labels(frame)
        expected_exit = 108.0 + 2.0 * 0.1
        expected_raw = (99.9 - expected_exit) / 99.9 * 10_000.0
        self.assertAlmostEqual(output["TickSize"].item(), 0.1)
        self.assertAlmostEqual(output["label_exit_price_robustness"].item(), expected_exit)
        self.assertAlmostEqual(output["label_raw_bp"].item(), expected_raw)
        self.assertAlmostEqual(output["label_residual_bp"].item(), expected_raw + 2.0)
        self.assertAlmostEqual(output["label_z"].item(), (expected_raw + 2.0) / 100.0)

    def test_calendar_daily_keeps_zero_signal_session(self) -> None:
        signals = pl.DataFrame(
            {
                "Date": ["20250102"],
                "QuoteCode": ["2330"],
                "BidPrice1": [100.0],
                "label_raw_bp": [30.0],
                "label_residual_bp": [35.0],
            }
        )
        daily = RUNNER._daily_frame(
            signals, ["20250102", "20250103"], fee_bp=19.3
        )
        self.assertEqual(len(daily), 2)
        self.assertEqual(daily["trades"].to_list(), [1, 0])
        self.assertTrue(
            np.isclose(daily["net_pnl"][0], 100.0 * (30.0 - 19.3) / 10.0)
        )
        self.assertEqual(daily["net_pnl"][1], 0.0)


if __name__ == "__main__":
    unittest.main()
