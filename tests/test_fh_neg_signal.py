from __future__ import annotations

import unittest
from datetime import datetime

import polars as pl

from src.research.intradaySignal.signals._short_common import PRICE_CAP, short_point_filter_spec
from src.research.intradaySignal.signals.fh_neg import CONDITION, FH_NEG, POINT_FILTER


def _rows() -> pl.DataFrame:
    t0 = datetime(2026, 9, 18, 10, 0, 0)
    # row 0: 3 minutes after a high 3% above open, deep selling, 5 ticks below, 3 bearish votes, first visit
    # row 1: same but 40 minutes after the high
    # row 2: 3 minutes after the high but shallow selling (depth -2), second visit
    # row 3: high never above open (opened at high, slid), 3 minutes after
    return pl.DataFrame(
        {
            "TransTime": [t0, t0, t0, t0],
            "fh_neg__high_time": [
                datetime(2026, 9, 18, 9, 57), datetime(2026, 9, 18, 9, 20),
                datetime(2026, 9, 18, 9, 57), datetime(2026, 9, 18, 9, 57),
            ],
            "FillLots_atHigh": [-900.0, -900.0, -200.0, -900.0],
            "avg_bidLots1": [50.0, 50.0, 50.0, 50.0],
            "avg_askLots1": [50.0, 50.0, 50.0, 50.0],
            "ToHigh": [0.05, 0.05, 0.05, 0.05],
            "RefPrice": [100.0, 100.0, 100.0, 100.0],
            "BidPrice1": [101.0, 101.0, 101.0, 101.0],
            "TickSize": [1.0, 1.0, 1.0, 1.0],
            "B1_A1B1": [0.2, 0.2, 0.2, 0.2],
            "MicroMomentum_30": [-0.5, -0.5, -0.5, -0.5],
            "LOB_NetPressure_30": [-1.0, -1.0, -1.0, -1.0],
            "fh_neg__visit_index": [1, 1, 2, 1],
            "fh_neg__pair_changed": [True, True, False, True],
            "RecordHigh": [103.0, 103.0, 103.0, 100.0],
            "Open": [100.0, 100.0, 100.0, 100.0],
            "OTC": [0, 0, 0, 0],
            "MD_L1Rate_30": [0.1, 0.1, 0.1, 0.1],
            "SpreadPairElapsed": [0.05, 0.05, 0.05, 0.05],
            "AmountRank_canDayTrade": [500, 500, 500, 500],
            "day_amount_rank": [120, 120, 120, 120],
            "TrialMatch": [0, 0, 0, 0],
            "ToRef": [0.01, 0.01, 0.01, 0.01],   # tickFeature column (Mid-based); the condition reads it as delivered
        },
        schema_overrides={"fh_neg__visit_index": pl.UInt32},
    )


class FhNegConditionTest(unittest.TestCase):
    def test_default_condition_is_5_minutes_at_or_above_open(self) -> None:
        # default = <=5 min since the high + bid >= open + ToRef/rank edge; row 1 is 40 minutes old
        self.assertEqual(CONDITION.mask(_rows()).to_list(), [True, False, True, True])
        self.assertEqual(CONDITION.mask(_rows(), {"bear_min": 3}).to_list(), [True, False, True, True])
        self.assertEqual(CONDITION.mask(_rows(), {"to_open_min": None}).to_list(), [True, False, True, True])
        self.assertEqual(CONDITION.mask(_rows(), {"to_open_min": 0.02}).to_list(), [False, False, False, False])
        self.assertEqual(CONDITION.mask(_rows(), {"min_since_high_max": 15.0, "to_open_min": -0.02}).to_list(), [True, False, True, True])

    def test_study_condition_is_one_grid_point(self) -> None:
        rows = _rows()
        study = {"min_since_high_max": None, "depth_max": -8.0, "gap_min": 4.0, "bear_min": 2, "max_visit": 1}
        self.assertEqual(CONDITION.mask(rows, study).to_list(), [True, True, False, True])
        self.assertEqual(CONDITION.mask(rows, {**study, "high_vs_open_min": 0.02}).to_list(), [True, True, False, False])
        self.assertEqual(CONDITION.mask(rows, {"min_since_high_max": 5.0}).to_list(), [True, False, True, True])
        self.assertEqual(len(CONDITION.expand_grid()), 20)
        self.assertEqual({c["min_since_high_max"] for c in CONDITION.expand_grid()}, {5.0, 7.5, 10.0, 15.0})
        self.assertEqual(CONDITION.mask(rows, {"run_up_min": 0.02}).to_list(), [True, False, True, False])
        self.assertEqual(CONDITION.expand_grid()[0], dict(CONDITION.params))


class ShortPointFilterTest(unittest.TestCase):
    def test_session_trial_and_optional_tick_filters(self) -> None:
        rows = pl.DataFrame(
            {
                "TransTime": [datetime(2026, 9, 18, 10, 0), datetime(2026, 9, 18, 12, 31), datetime(2026, 9, 18, 9, 0, 10), datetime(2026, 9, 18, 11, 0)],
                "TrialMatch": [0, 0, 0, 1],
                "BidPrice1": [101.0, 101.0, 101.0, 101.0],
                "OTC": [0, 0, 0, 0],
                "MD_L1Rate_30": [0.1, 0.1, 0.1, 0.1],
                "SpreadPairElapsed": [0.05, 0.05, 0.05, 0.05],
                "AmountRank_canDayTrade": [500, 500, 500, 500],
                "day_amount_rank": [120, 120, 120, 120],
            }
        )
        # fh: session window + formal match only
        self.assertEqual(POINT_FILTER.mask(rows).to_list(), [True, False, False, False])
        self.assertEqual(POINT_FILTER.mask(rows, {"md_l1rate_min": 0.25}).to_list(), [False, False, False, False])
        self.assertEqual(POINT_FILTER.mask(rows, {"rank_max": 100}).to_list(), [False, False, False, False])
        self.assertEqual(POINT_FILTER.mask(rows, {"elapsed_min": 0.1}).to_list(), [False, False, False, False])
        self.assertEqual(POINT_FILTER.mask(rows, {"session_close": "12:35:00"}).to_list(), [True, True, False, False])
        with self.assertRaises(ValueError):
            short_point_filter_spec("x", nope=1)

    def test_day_rules_fold_into_the_point_filter(self) -> None:
        from src.research.intradaySignal.signals.negfill import NEGFILL, POINT_FILTER as NF_POINTS
        base = dict(TransTime=[datetime(2026, 9, 18, 10, 0)] * 3, TrialMatch=[0, 0, 0], BidPrice1=[101.0] * 3, OTC=[1, 1, 1],
                    MD_L1Rate_30=[0.5] * 3, SpreadPairElapsed=[0.5] * 3, AmountRank_canDayTrade=[50] * 3, day_amount_rank=[50] * 3,
                    MarketCap=[1e9, 1e9, 9e11], d_intraday_return=[0.0, 0.05, 0.0], dt_major_count=[0, 0, 0], sell_branch_breadth_rank=[0.5] * 3,
                    net_buy_gini=[0.1] * 3, top1_net_buy_amount=[1.0] * 3, trading_turnover_bf=[100.0] * 3, RefPrice=[50.0] * 3, hft_strick_makerSpreadBP=[0.0] * 3)
        rows = pl.DataFrame(base)
        # row 1: R1 (up 5% yesterday, no major buyer); row 2: market cap >= 8e11
        self.assertEqual(NF_POINTS.mask(rows).to_list(), [True, False, False])
        self.assertIsNone(NEGFILL.day_selector)
        self.assertEqual(NEGFILL.model.threshold.q_timing, 0.05)

    def test_price_cap_is_a_hard_rule(self) -> None:
        rows = pl.DataFrame({"RefPrice": [100.0, 3000.0, 3001.0, None]})
        self.assertEqual(PRICE_CAP.mask(rows).to_list(), [True, True, False, False])
        self.assertEqual(PRICE_CAP.stage, "hard_rule")


class RegistryWiringTest(unittest.TestCase):
    def test_sections(self) -> None:
        self.assertEqual(FH_NEG.condition_id, "fh_neg_v3")
        self.assertEqual(FH_NEG.stage, "research")
        self.assertEqual(FH_NEG.envelope_columns, ("pair_changed", "large_fill", "visit_index", "high_time"))
        self.assertIs(FH_NEG.gate, CONDITION)
        self.assertIs(FH_NEG.point_filter, POINT_FILTER)
        self.assertEqual(FH_NEG.point_filter.stage, "point_filter")
        self.assertEqual([m.model_id for m in FH_NEG.day_selector.models], ["stop", "squeeze", "open_pos"])
        self.assertIs(FH_NEG.day_selector.hard_rules, PRICE_CAP)
        self.assertEqual(FH_NEG.day_selector.baseline_rules.gate_id, "legacy_day_rules")
        self.assertFalse(FH_NEG.model.use_model_gate)


if __name__ == "__main__":
    unittest.main()
