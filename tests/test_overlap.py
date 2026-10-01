from __future__ import annotations

import unittest
from datetime import datetime

import polars as pl

from src.research.intradaySignal.framework.overlap import overlap_report


def _frame(rows):
    return pl.DataFrame(
        rows,
        schema={"Date": pl.String, "QuoteCode": pl.String, "ChannelSeq": pl.UInt64, "TransTime": pl.Datetime("us"), "y": pl.Float64},
        orient="row",
    )


class OverlapTest(unittest.TestCase):
    def test_rulers(self) -> None:
        t = lambda h, m, s=0: datetime(2026, 9, 1, h, m, s)  # noqa: E731
        a = _frame([
            ("20260901", "2330", 1, t(9, 5), 10.0),      # exact match with B
            ("20260901", "2330", 2, t(9, 6), 20.0),      # same stock-day, B tick 60s away
            ("20260901", "2317", 1, t(9, 5), 30.0),      # stock-day disjoint
            ("20260902", "2330", 1, t(9, 5), 40.0),      # day 2, exact match
            ("20260902", "2317", 1, t(9, 5), 50.0),      # day 2, disjoint
        ])
        b = _frame([
            ("20260901", "2330", 1, t(9, 5), 1.0),
            ("20260901", "2330", 9, t(9, 7), 2.0),
            ("20260902", "2330", 1, t(9, 5), 3.0),
            ("20260902", "2454", 1, t(9, 5), 4.0),
        ])
        r = overlap_report(a, b, a_name="fh", b_name="negfill", label_col="y", proximity_windows=(60, 300))
        self.assertEqual(r.tick_shared, 2)
        self.assertAlmostEqual(r.a_in_b_share, 2 / 5)
        self.assertAlmostEqual(r.b_in_a_share, 2 / 4)
        self.assertAlmostEqual(r.tick_jaccard, 2 / 7)
        self.assertEqual(r.stock_day_shared, 2)               # (0901,2330), (0902,2330)
        self.assertAlmostEqual(r.a_stock_day_in_b_share, 2 / 4)
        # ticks 1,2 (0901 2330) and 4 (0902 2330) have a B tick within 300s; within 60s: tick 1 (0s), tick 2 (60s), tick 4
        self.assertAlmostEqual(r.proximity[300], 3 / 5)
        self.assertAlmostEqual(r.proximity[60], 3 / 5)
        self.assertEqual(r.common_days, 2)
        self.assertEqual(r.a_exact_disjoint_rows, 3)
        self.assertAlmostEqual(r.a_exact_disjoint_label_mean, (20 + 30 + 50) / 3)
        self.assertEqual(r.a_stock_day_disjoint_rows, 2)
        self.assertAlmostEqual(r.a_stock_day_disjoint_label_mean, 40.0)
        table = r.as_table()
        self.assertIn("exact tick Jaccard", table["metric"].to_list())


if __name__ == "__main__":
    unittest.main()
