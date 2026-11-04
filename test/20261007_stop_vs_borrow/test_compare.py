"""Economic branch tests, missing data guards and parity with BackTest/tick helper."""
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from compare import (POLICIES, mark_policies, prepare_signals, price_policies,
                     resolve_next_open, summarize, touch_times)
from src.research.backTest import BackTest
from src.research.tool import first_touch_times


def fixture():
    t = datetime(2026, 1, 2, 10)
    # TP first; SL then TP; SL then close; SL then borrow; neither, exact 109%.
    frame = pl.DataFrame({"Date": ["20260102"] * 5, "QuoteCode": ["2330"] * 5,
        "ChannelSeq": list(range(5)), "TransTime": [t] * 5, "BidPrice1": [100.] * 5,
        "RefPrice": [100.] * 5, "Close": [110., 110., 102., 110., 109.], "TickSize": [.5] * 5,
        "txf_residual_vol_to_1330": [.02] * 5, "nextDayOpen": [112.] * 5,
        "TakerSell_CloseBP": [-9999.] * 5})
    frame = prepare_signals(frame).with_columns(
        pl.Series("low_time", [t + timedelta(minutes=1), t + timedelta(minutes=2), None, None, None], dtype=pl.Datetime("us")),
        pl.Series("high_time", [t + timedelta(minutes=2), t + timedelta(minutes=1), t + timedelta(minutes=1), t + timedelta(minutes=1), None], dtype=pl.Datetime("us")),
        pl.lit(112.).alias("comparison_next_open"))
    return mark_policies(frame)


class ComparisonTests(unittest.TestCase):
    def test_exit_branches_and_no_stale_labels(self):
        ledger = price_policies(fixture())
        self.assertEqual(ledger["tp_sl_reason"].to_list(), ["tp", "sl", "sl", "sl", "close"])
        self.assertEqual(ledger["tp_no_sl_borrow_reason"].to_list(), ["tp", "tp", "close", "borrow", "close"])
        np.testing.assert_allclose(ledger["tp_sl_exit"], [96.001, 104.5, 104.5, 104.5, 109.])
        np.testing.assert_allclose(ledger["tp_no_sl_borrow_exit"], [96.001, 96.001, 102., 115.36, 109.])
        self.assertAlmostEqual(ledger["borrow_fee_twd"][3], 3360.)
        self.assertAlmostEqual(ledger["tp_no_sl_borrow_pnl_twd"][3], -15553.)
        # No touch must use Close, even if original label contains a legacy stop.
        self.assertAlmostEqual(ledger["tp_sl_bp"][4], -900.)

    def test_tie_is_stop_for_baseline_but_tp_without_stop(self):
        f = fixture().head(1).with_columns(pl.col("high_time").alias("low_time"))
        f = mark_policies(f)
        self.assertEqual(f["tp_sl_reason"][0], "sl")
        self.assertEqual(f["tp_no_sl_borrow_reason"][0], "tp")

    def test_required_missing_open_fails_and_irrelevant_missing_is_ok(self):
        f = fixture().with_columns(pl.lit(None, dtype=pl.Float64).alias("comparison_next_open"))
        with self.assertRaisesRegex(ValueError, "Unpriced borrow"):
            price_policies(f)
        self.assertEqual(price_policies(f.filter(pl.col("tp_no_sl_borrow_reason") != "borrow")).height, 4)

    def test_market_open_uses_next_session_and_checks_existing_value(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            for date, price in (("20260102", 100.), ("20260105", 112.)):
                pl.DataFrame({"quote_code": ["2330"], "open_price": [price]}).write_parquet(path / f"{date}_marketData.parquet")
            f = fixture().drop("comparison_next_open").with_columns(pl.lit(None, dtype=pl.Float64).alias("nextDayOpen"))
            filled, audit = resolve_next_open(f, path)
            self.assertEqual(audit["comparison_next_date"][0], "20260105")
            self.assertEqual(audit["filled_missing_rows"][0], 1)
            self.assertAlmostEqual(filled["comparison_next_open"][0], 112.)
            with self.assertRaisesRegex(ValueError, "disagrees"):
                resolve_next_open(f.with_columns(pl.lit(111.).alias("nextDayOpen")), path)

    def test_backtest_pnl_parity_and_zero_initial_drawdown(self):
        ledger = price_policies(fixture())
        for policy in POLICIES:
            bt = BackTest(ledger.to_pandas(), bp_col=f"{policy}_bp")
            np.testing.assert_allclose(bt.df["PnL"], ledger[f"{policy}_pnl_twd"])
        result = summarize(ledger, ["20260102", "20260105"])
        self.assertEqual(result["daily"].height, 2)
        row = result["summary"].filter((pl.col("period") == "all") & (pl.col("policy") == "tp_sl")).row(0, named=True)
        self.assertAlmostEqual(row["mdd_wan"], row["net_pnl_wan"])

    def test_touch_parity_strict_entry_and_missing_tick_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            t = datetime(2026, 1, 2, 10)
            f = prepare_signals(pl.DataFrame({"Date": ["20260102"], "QuoteCode": ["2330"],
                "TransTime": [t], "BidPrice1": [100.], "RefPrice": [100.], "Close": [99.],
                "TickSize": [.5], "txf_residual_vol_to_1330": [.02]}))
            with self.assertRaises(FileNotFoundError):
                touch_times(f, path, verbose=False)
            pl.DataFrame({"QuoteCode": ["2330"] * 4, "TrialMatch": [0, 1, 0, 0],
                "TransTime": [t, t + timedelta(seconds=1), t + timedelta(seconds=2), t + timedelta(seconds=3)],
                "FillPrice": [90., 90., 105., 96.]}).write_parquet(path / "20260102_StockTick.parquet")
            out, _ = touch_times(f, path, verbose=False)
            canonical = first_touch_times(f.to_pandas(), low_col="comparison_tp_price", high_col="comparison_sl_price", tick_dir=path)
            np.testing.assert_array_equal(out["low_time"].to_numpy(), canonical.low_time.values)
            np.testing.assert_array_equal(out["high_time"].to_numpy(), canonical.high_time.values)
            self.assertEqual(out["low_time"][0], t + timedelta(seconds=3))


if __name__ == "__main__":
    unittest.main()
