"""Focused contracts for the study-local FillHigh signal adapter."""

from __future__ import annotations

import datetime as dt
import unittest

import polars as pl

from study import (
    FEATURE_MANIFEST,
    REQUIRED_MODEL_INPUT_COLUMNS,
    add_study_features,
    add_threshold_buckets,
    apply_capacity,
)


def _feature_fixture() -> pl.DataFrame:
    values: dict[str, list[object]] = {
        column: [1.0] for column in REQUIRED_MODEL_INPUT_COLUMNS
    }
    values.update(
        {
            "Date": ["20260102"],
            "QuoteCode": ["2330"],
            "ChannelSeq": [1],
            "TransTime": [dt.datetime(2026, 1, 2, 10, 0)],
            "OTC": [False],
            "TrialMatch": [0],
            "BidPrice1": [100.0],
            "RefPrice": [98.0],
            "Open": [99.0],
            "TickSize": [0.5],
            "Spread": [1.0],
            "FillLots": [20.0],
            "FillLots_atLow": [-30.0],
            "FillLots_atHigh": [-50.0],
            "avg_bidLots1": [10.0],
            "avg_askLots1": [15.0],
            "BidLots1": [10.0],
            "BidLots2": [5.0],
            "BidLots4": [4.0],
            "BidLots5": [6.0],
            "AskLots1": [12.0],
            "AskLots2": [3.0],
            "AskLots4": [5.0],
            "AskLots5": [5.0],
            "ToHigh": [0.02],
            "MD_ElaspeTime_30": [9.0],
            "LOB_NetPressure_30": [-9.0],
            "QL_Asymmetry": [-3.0],
        }
    )
    return pl.DataFrame(values)


class FillHighSignalStudyTest(unittest.TestCase):
    def test_manifest_is_fixed_and_future_free(self) -> None:
        self.assertEqual(len(FEATURE_MANIFEST.features), 38)
        self.assertEqual(
            FEATURE_MANIFEST.fingerprint,
            "09a29ca300424620a1cbf6331e585072d91a9a6fa2c8dd7ea4e4f1d6e6559468",
        )
        self.assertFalse(
            any(
                name.startswith("Future")
                or name in {"Close", "txf_to_1330_bp"}
                for name in FEATURE_MANIFEST.ordered_names
            )
        )

    def test_feature_transform_uses_explicit_depth_and_signed_context(self) -> None:
        row = add_study_features(_feature_fixture()).row(0, named=True)
        self.assertEqual(row["ToRef"], round((100.0 - 98.0) / 98.0, 6))
        self.assertEqual(row["FillLots_atHigh_re"], -2.0)
        self.assertEqual(row["FillLots_atLow_re"], -1.2)
        self.assertEqual(row["B2_Last"], 0.5)
        self.assertEqual(row["A2_Last"], 0.2)
        self.assertEqual(row["Spread_ticks"], 2.0)
        self.assertEqual(row["filllots_at_high_gap_ticks"], 3.92)
        self.assertLess(row["LOB_NetPressure_30_signed_log"], 0.0)

    def test_notebook_bucket_contract_fails_closed_outside_range(self) -> None:
        result = add_threshold_buckets(
            pl.DataFrame(
                {
                    "RemainSeconds": [3000, 3301, 7000, 11000, 15000],
                    "ToRef": [-0.001, 0.005, 0.015, 0.025, 0.04],
                }
            )
        )
        self.assertEqual(
            result["time_bucket"].to_list(),
            [None, "t0", "t1", "t2", "t3"],
        )
        self.assertEqual(
            result["to_ref_bucket"].to_list(),
            [None, "r0", "r1", "r2", "r3"],
        )

    def test_capacity_is_one_based_and_strict(self) -> None:
        frame = pl.DataFrame(
            {
                "Date": ["20260102"] * 3,
                "QuoteCode": ["2330"] * 3,
                "TransTime": [
                    dt.datetime(2026, 1, 2, 9, 31, second)
                    for second in range(3)
                ],
                "ChannelSeq": [1, 2, 3],
                "BidPrice1": [100.0, 100.0, 100.0],
                "avg_bidLots1": [1.0, 1.0, 1.0],
                "avg_askLots1": [2.0, 2.0, 2.0],
            }
        )
        result = apply_capacity(frame)
        self.assertEqual(result["accLots"].to_list(), [1, 2])
        self.assertEqual(result["stock_position_wan"].to_list(), [10.0, 20.0])


if __name__ == "__main__":
    unittest.main()
