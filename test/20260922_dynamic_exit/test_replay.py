"""Synthetic paths test timing, late barriers, causal centers and empty books."""
import unittest

import numpy as np

from data import MINUTE, StockTicks, load_day, minute_boundaries, timestamp, trailing_centers
from engine import bands, replay


def simulate(t, kind, fill, ask, *, mode=0, ratio=1.05, activation=20,
             center=None, exec_price=None, exec_time=None, entry=0, scope=0):
    t = np.array(t, dtype=np.int64)
    f = lambda v: np.array(v, dtype=float)
    ask = f(ask)
    return replay(t, np.array(kind, dtype=np.int8), f(fill), ask, t,
                  ask if exec_price is None else f(exec_price), t if exec_time is None else np.array(exec_time, dtype=np.int64),
                  np.full(len(t), 100.) if center is None else f(center), np.ones(len(t)),
                  np.array([entry], dtype=np.int64), f([100]), f([100]), f([.02]), f([.02]), f([99]),
                  100, mode, 2.5, ratio, activation, scope)[0]


class ReplayTests(unittest.TestCase):
    def test_late_reduction_uses_current_ask(self):
        out = simulate([10, 20], [0, 2], [107, 107], [107.1, 107.2])
        self.assertAlmostEqual(out[0], 107.2)
        self.assertAlmostEqual(out[1], 20)
        self.assertAlmostEqual(out[3], 1)

    def test_no_retroactive_early_stop(self):
        out = simulate([10, 20, 30], [0, 2, 0], [106, 104, 106], [106.1, 104.1, 106.1])
        self.assertAlmostEqual(out[1], 30)

    def test_original_stop_precedes_cascade(self):
        out = simulate([10, 20], [0, 2], [109, 104], [109.1, 104.1])
        self.assertAlmostEqual(out[1], 10)

    def test_new_entry_beyond_new_line(self):
        out = simulate([20, 30], [2, 0], [104, 107], [104.1, 107.1], entry=25)
        self.assertAlmostEqual(out[1], 30)
        self.assertAlmostEqual(out[0], 107.1)

    def test_new_only_does_not_close_old_position(self):
        out = simulate([20, 30], [2, 0], [107, 107], [107.1, 107.1], scope=2)
        self.assertAlmostEqual(out[2], 0)

    def test_pending_order_waits_for_real_ask(self):
        out = simulate([10], [0], [109], [0], exec_time=[40], exec_price=[110])
        self.assertAlmostEqual(out[1], 40)
        self.assertAlmostEqual(out[0], 110)
        self.assertAlmostEqual(out[4], 1)

    def test_locked_to_close_is_mark_not_fill(self):
        out = simulate([10], [0], [109], [0], exec_time=[-1], exec_price=[np.nan])
        self.assertAlmostEqual(out[5], 1)
        self.assertAlmostEqual(out[0], 99)

    def test_empty_ask_at_reset_still_arms_cover(self):
        out = simulate([20], [2], [107], [0], exec_time=[40], exec_price=[108])
        self.assertAlmostEqual(out[2], 1)
        self.assertAlmostEqual(out[1], 40)

    def test_cap_and_nearer_sides(self):
        sl, tp = bands(6, 100., 100., .02, .03, 110., 2.5, 105., 95.)
        self.assertAlmostEqual(sl, 105.)
        self.assertAlmostEqual(tp, 101.75)
        sl, tp = bands(3, 100., 100., .02, .03, 110., 2.5, 105., 95.)
        self.assertAlmostEqual(sl, 108.)

    def test_moving_tp_can_realize_loss(self):
        out = simulate([20], [1], [101], [101.1], mode=6, center=[108])
        self.assertAlmostEqual(out[2], 2)
        self.assertGreater(out[0], 100)

    def test_ratchet_cannot_relax(self):
        sl, tp = bands(7, 100., 100., .02, .02, 100., 2.5, 101., 98.)
        self.assertAlmostEqual(sl, 101.)
        self.assertAlmostEqual(tp, 98.)

    def test_reset_observed_survives_later_trade_exit(self):
        out = simulate([20, 30], [1, 0], [100, 109], [100.1, 109.1], mode=6, center=[100, 100])
        self.assertAlmostEqual(out[10], 1)
        self.assertAlmostEqual(out[3], 0)

    def test_unhit_close_keeps_latest_barrier(self):
        out = simulate([20], [1], [100], [100.1], mode=3, center=[101])
        self.assertAlmostEqual(out[6], 106.05)
        self.assertAlmostEqual(out[7], 95.95)

    def test_closing_auction_has_no_immediate_minute_fills(self):
        t = minute_boundaries("20260922")
        self.assertEqual(t[-1], timestamp("20260922", "13:24:00"))
        self.assertTrue(np.all(t < timestamp("20260922", "13:25:00")))

    def test_pending_order_cannot_use_closing_auction_ask(self):
        tk = StockTicks(np.array([10, 90]), np.array([109., 110.]), np.ones(2), np.array([0., 110.]),
                        np.ones(2), 100, np.array([], dtype=np.int64), np.array([]), np.array([]), 80)
        ev = tk.events()
        self.assertEqual(ev[6][0], -1)
        self.assertEqual(ev[6][1], -1)

    def test_trial_phase_invalidates_prior_formal_ask(self):
        import tempfile
        from pathlib import Path
        import polars as pl
        date = "20260922"
        t = [timestamp(date,s) for s in ["09:40:00","09:59:30","10:00:30"]]
        frame=pl.DataFrame({"QuoteCode":["A"]*3,"TransTime":pl.Series(t,dtype=pl.Datetime("us")),
                            "ChannelSeq":[1,2,3],"TrialMatch":[0,1,0],"FillPrice":[100.,999.,101.],
                            "FillLots":[1,100,1],"AskPrice1":[100.1,999.1,101.1],"AskLots1":[1,1,1]})
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"ticks.parquet"
            frame.write_parquet(path)
            tick=load_day(path,date,["A"])["A"]
        ev=tick.events()
        at=np.flatnonzero((ev[0]==timestamp(date,"10:00:00")) & (ev[1]==1))[0]
        self.assertAlmostEqual(ev[3][at],0.)
        self.assertAlmostEqual(ev[5][at],101.1)
        self.assertEqual(ev[6][at],timestamp(date,"10:00:30"))
        self.assertAlmostEqual(ev[7][at],100.)

    def test_no_future_prices_in_vwap(self):
        t = np.array([0, 29*MINUTE, 30*MINUTE, 31*MINUTE])
        vwap, sma = trailing_centers(t, np.array([100., 102., 999., 500.]), np.array([1., 3., 100., 200.]),
                                     np.array([30*MINUTE]))
        self.assertAlmostEqual(vwap[0], 101.5)
        self.assertAlmostEqual(sma[0], 101.)

    def test_future_packet_same_timestamp_does_not_change_trade(self):
        tk = StockTicks(np.array([10, 10, 30]), np.array([109., 0., 0.]), np.array([1., 0., 0.]),
                        np.array([109.1, 107.1, 106.]), np.ones(3), 100,
                        np.array([], dtype=np.int64), np.array([]), np.array([]))
        ev = tk.events()
        self.assertAlmostEqual(ev[3][0], 109.1)

    def test_cross_stock_switch_uses_distinct_actual_stops(self):
        import polars as pl
        from unittest.mock import patch
        from run_study import process_day
        from spec import candidates
        sig = pl.DataFrame({"Date": ["20260922"]*3, "QuoteCode": ["A", "A", "B"],
                            "ChannelSeq": [1, 2, 3], "trade_id": [0, 1, 2],
                            "TransTime": pl.Series([0, 1, 0], dtype=pl.Datetime("us")),
                            "BidPrice1": [100.]*3, "RefPrice": [100.]*3, "TickSize": [.1]*3,
                            "Close": [99.]*3, "txf_residual_vol_to_1330": [.02]*3,
                            "txf_residual_vol_0900_1330_60d": [.02]*3})
        def stock(t, p):
            return StockTicks(np.array(t), np.array(p), np.ones(len(t)), np.array(p)+.1,
                              np.ones(len(t)), 100, np.array([], dtype=np.int64), np.array([]), np.array([]))
        ticks = {"A": stock([10, 30], [109., 104.]), "B": stock([9, 30], [107., 104.])}
        specs = [c for c in candidates() if c.name in ["ref08", "cascade_k1_to05", "cascade_k2_to05"]]
        with patch("run_study.load_day", return_value=ticks):
            result, stops = process_day(sig, None, specs)
        self.assertEqual(len(stops), 1)
        b1 = result.filter((pl.col("QuoteCode") == "B") & (pl.col("candidate") == "cascade_k1_to05"))
        self.assertAlmostEqual(b1["exit_price"][0], 107.1)
        self.assertEqual(b1["exit_time"].cast(pl.Int64)[0], 10)
        b2 = result.filter((pl.col("QuoteCode") == "B") & (pl.col("candidate") == "cascade_k2_to05"))
        self.assertEqual(b2["reason"][0], 0)


if __name__ == "__main__":
    unittest.main()
