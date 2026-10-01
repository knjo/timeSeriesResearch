from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import polars as pl

from src.research.intradaySignal.framework.events import DataRoots, EventDaySources
from src.research.intradaySignal.framework.profiles.filllots_at_high import (
    CANDIDATE_DESCRIPTIONS,
    CANDIDATE_IDS,
    FILLLOTS_AT_HIGH_FEATURE_FAMILY_VERSION,
    build_filllots_at_high_candidate_frame,
    select_filllots_at_high_feature_candidates,
)


DATE = "20260102"


class FillLotsAtHighProfileTest(unittest.TestCase):
    def _roots(self, root: Path) -> DataRoots:
        roots = DataRoots.from_project_root(root)
        for directory in (
            roots.tick_data_dir,
            roots.tick_feature_dir,
            roots.pre_market_dir,
        ):
            directory.mkdir(parents=True)
        return roots

    def _write_fixture(self, roots: DataRoots) -> None:
        pl.DataFrame(
            {
                "QuoteCode": ["A"] * 11,
                "ChannelSeq": pl.Series(range(1, 12), dtype=pl.UInt64),
                "FillLots": [0, 1, 0, 20, 30, 0, 400, 1, 0, 500, 400],
                "InOut": [0, -1, 0, 1, -1, 0, -1, -1, 0, 1, 1],
                "TrialMatch": [0] * 11,
                "marketOpen": [True] * 11,
                "RecordHigh": [10.0] * 6 + [11.0] * 4 + [12.0],
                "RefPrice": [10.0] * 11,
                "BidPrice1": [
                    10.0,
                    10.0,
                    9.9,
                    9.9,
                    9.9,
                    9.9,
                    10.9,
                    10.9,
                    11.0,
                    11.0,
                    11.9,
                ],
                "AskPrice1": [
                    10.1,
                    10.1,
                    10.1,
                    10.1,
                    10.1,
                    10.2,
                    11.0,
                    11.0,
                    11.1,
                    11.1,
                    12.0,
                ],
            }
        ).write_parquet(roots.tick_data_dir / f"{DATE}_StockTick.parquet")

        # Deliberately supply a wrong precomputed state.  The profile must
        # rebuild it from raw ticks before attaching these causal features.
        pl.DataFrame(
            {
                "QuoteCode": ["A"] * 11,
                "ChannelSeq": pl.Series(range(1, 12), dtype=pl.UInt64),
                "FillLots_atHigh": [999.0] * 11,
                "B1_A1B1": [
                    0.5,
                    0.5,
                    0.2,
                    0.8,
                    0.8,
                    0.5,
                    0.2,
                    0.2,
                    0.8,
                    0.8,
                    0.2,
                ],
                "LOB_NetPressure_30": [
                    0.0,
                    -0.5,
                    -0.5,
                    0.5,
                    0.5,
                    0.0,
                    -0.5,
                    -0.5,
                    0.5,
                    0.5,
                    -0.5,
                ],
                "MicroMomentum_30": [
                    0.0,
                    -0.2,
                    -0.2,
                    0.2,
                    0.2,
                    0.0,
                    -0.2,
                    -0.2,
                    0.2,
                    0.2,
                    -0.2,
                ],
                "ToHigh": [
                    0.0,
                    0.0,
                    0.005,
                    0.005,
                    0.025,
                    0.025,
                    0.025,
                    0.025,
                    0.005,
                    0.005,
                    0.005,
                ],
                "TickSize": [0.05] * 11,
            }
        ).write_parquet(
            roots.tick_feature_dir / f"{DATE}_tickFeature.parquet"
        )
        pl.DataFrame(
            {
                "QuoteCode": ["A"],
                "avg_bidLots1": [10.0],
                "avg_askLots1": [10.0],
                "big_buy_lots": [5.0],
                "big_sell_lots": [5.0],
            }
        ).write_parquet(
            roots.pre_market_dir / f"{DATE}_preMarketData.parquet"
        )

    def test_every_candidate_requires_price_change_or_large_fill(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            roots = self._roots(Path(temporary_directory))
            self._write_fixture(roots)
            day = EventDaySources(DATE, roots)
            frame = build_filllots_at_high_candidate_frame(day)
            selected = select_filllots_at_high_feature_candidates(day)

            self.assertEqual(FILLLOTS_AT_HIGH_FEATURE_FAMILY_VERSION, "1")
            self.assertEqual(tuple(selected), CANDIDATE_IDS)
            self.assertEqual(tuple(CANDIDATE_DESCRIPTIONS), CANDIDATE_IDS)
            self.assertEqual(
                frame.filter(pl.col("ChannelSeq") == 7)[
                    "FillLots_atHigh"
                ].item(),
                -400,
            )
            self.assertEqual(
                frame.filter(pl.col("ChannelSeq") == 11)[
                    "FillLots_atHigh"
                ].item(),
                400,
            )

            def sequences(candidate_id: str) -> list[int]:
                return selected[candidate_id]["ChannelSeq"].to_list()

            # A small same-price fill changes the cumulative state but is not a
            # fresh sample under either user-approved sampling branch.
            self.assertNotIn(2, {seq for candidate in selected.values() for seq in candidate["ChannelSeq"]})
            # A later small same-price fill remains blocked even in a deep state.
            self.assertNotIn(8, {seq for candidate in selected.values() for seq in candidate["ChannelSeq"]})

            self.assertEqual(sequences("fh_pos_sample"), [4, 10, 11])
            self.assertEqual(sequences("fh_neg_sample"), [3, 5, 6, 7, 9])
            self.assertEqual(sequences("fh_pos_depth2"), [10, 11])
            self.assertEqual(sequences("fh_neg_depth8"), [7, 9])
            self.assertEqual(sequences("fh_pos_near"), [4, 10, 11])
            self.assertEqual(sequences("fh_neg_far"), [5, 6, 7])
            self.assertEqual(sequences("fh_pos_breakout"), [10])
            self.assertEqual(sequences("fh_pos_exhaustion"), [11])
            self.assertEqual(sequences("fh_neg_rejection"), [7])
            self.assertEqual(sequences("fh_neg_absorption"), [9])

            for candidate_id, keys in selected.items():
                self.assertEqual(len(keys), len(keys.unique()))
                self.assertIn(candidate_id, CANDIDATE_DESCRIPTIONS)

    def test_subset_and_unknown_candidate_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            roots = self._roots(Path(temporary_directory))
            self._write_fixture(roots)
            day = EventDaySources(DATE, roots)
            subset = select_filllots_at_high_feature_candidates(
                day,
                candidate_ids=("fh_neg_sample", "fh_pos_depth2"),
            )
            self.assertEqual(
                tuple(subset),
                ("fh_neg_sample", "fh_pos_depth2"),
            )
            with self.assertRaisesRegex(ValueError, "unknown"):
                select_filllots_at_high_feature_candidates(
                    day,
                    candidate_ids=("not_registered",),
                )
            with self.assertRaisesRegex(ValueError, "duplicates"):
                select_filllots_at_high_feature_candidates(
                    day,
                    candidate_ids=("fh_neg_sample", "fh_neg_sample"),
                )

    def test_missing_prior_big_lot_threshold_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            roots = self._roots(Path(temporary_directory))
            self._write_fixture(roots)
            pl.DataFrame(
                {
                    "QuoteCode": ["A"],
                    "avg_bidLots1": [10.0],
                    "avg_askLots1": [10.0],
                    "big_buy_lots": pl.Series([None], dtype=pl.Float64),
                    "big_sell_lots": pl.Series([None], dtype=pl.Float64),
                }
            ).write_parquet(
                roots.pre_market_dir / f"{DATE}_preMarketData.parquet"
            )
            selected = select_filllots_at_high_feature_candidates(
                EventDaySources(DATE, roots)
            )
            selected_sequences = {
                sequence
                for keys in selected.values()
                for sequence in keys["ChannelSeq"]
            }

            # These rows have economically large current fills but revisit an
            # existing price pair.  Without a causal D-1 threshold, the large
            # fill branch must not silently fall back to the floor of 20.
            self.assertTrue({4, 5, 10}.isdisjoint(selected_sequences))
            self.assertIn(3, selected_sequences)


if __name__ == "__main__":
    unittest.main()
