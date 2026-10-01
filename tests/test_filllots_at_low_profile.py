from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import polars as pl

from src.research.intradaySignal.framework.events import DataRoots, EventDaySources
from src.research.intradaySignal.framework.profiles.filllots_at_low import (
    CANDIDATE_DESCRIPTIONS,
    CANDIDATE_IDS,
    FILLLOTS_AT_LOW_FAMILY_VERSION,
    FILLLOTS_AT_LOW_STATE_FAMILY_VERSION,
    STATE_CANDIDATE_DESCRIPTIONS,
    STATE_CANDIDATE_IDS,
    build_filllots_at_low_candidate_frame,
    select_filllots_at_low_candidates,
    select_filllots_at_low_state_candidates,
)


DATE = "20260102"


class FillLotsAtLowProfileTest(unittest.TestCase):
    def test_candidates_use_raw_prior_state_and_symmetric_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            roots = DataRoots.from_project_root(Path(tmp))
            for directory in (
                roots.tick_data_dir,
                roots.tick_feature_dir,
                roots.pre_market_dir,
                roots.tick_bar_dir,
                roots.txf_tick_dir,
            ):
                directory.mkdir(parents=True)

            pl.DataFrame(
                {
                    "QuoteCode": ["A"] * 8 + ["B"],
                    "ChannelSeq": pl.Series(
                        [1, 2, 3, 4, 5, 6, 7, 8, 1], dtype=pl.UInt64
                    ),
                    "FillLots_atLow": [
                        0,
                        -1,
                        -3,
                        -10,
                        -1,
                        1,
                        5,
                        -2,
                        -20,
                    ],
                }
            ).write_parquet(
                roots.tick_feature_dir / f"{DATE}_tickFeature.parquet"
            )
            pl.DataFrame(
                {
                    "QuoteCode": ["A"] * 8 + ["B"],
                    "ChannelSeq": pl.Series(
                        [1, 2, 3, 4, 5, 6, 7, 8, 1], dtype=pl.UInt64
                    ),
                    "FillLots": [0, 1, 2, 7, 1, 2, 4, 7, 20],
                    "InOut": [0, -1, -1, -1, -1, 1, 1, -1, -1],
                    "TrialMatch": [0, 0, 0, 0, 0, 0, 0, 1, 0],
                    "marketOpen": [True] * 9,
                    "RecordLow": [
                        10.0,
                        10.0,
                        10.0,
                        10.0,
                        9.0,
                        9.0,
                        9.0,
                        9.0,
                        8.0,
                    ],
                    "FillPrice": [
                        10.0,
                        10.1,
                        10.1,
                        10.1,
                        9.0,
                        9.1,
                        9.1,
                        9.1,
                        8.0,
                    ],
                }
            ).write_parquet(
                roots.tick_data_dir / f"{DATE}_StockTick.parquet"
            )
            pl.DataFrame(
                {
                    "QuoteCode": ["A"],
                    "avg_bidLots1": [1.0],
                    "avg_askLots1": [1.0],
                }
            ).write_parquet(
                roots.pre_market_dir / f"{DATE}_preMarketData.parquet"
            )

            selected = select_filllots_at_low_candidates(
                EventDaySources(DATE, roots)
            )
            self.assertEqual(FILLLOTS_AT_LOW_FAMILY_VERSION, "3")
            self.assertEqual(tuple(selected), CANDIDATE_IDS)
            self.assertEqual(tuple(CANDIDATE_DESCRIPTIONS), CANDIDATE_IDS)
            states = select_filllots_at_low_state_candidates(
                EventDaySources(DATE, roots)
            )
            self.assertEqual(FILLLOTS_AT_LOW_STATE_FAMILY_VERSION, "1")
            self.assertEqual(tuple(states), STATE_CANDIDATE_IDS)
            self.assertEqual(
                tuple(STATE_CANDIDATE_DESCRIPTIONS),
                STATE_CANDIDATE_IDS,
            )
            self.assertEqual(
                states["negative_state"].to_dicts(),
                [
                    {"QuoteCode": "A", "ChannelSeq": 2},
                    {"QuoteCode": "A", "ChannelSeq": 3},
                    {"QuoteCode": "A", "ChannelSeq": 4},
                    {"QuoteCode": "A", "ChannelSeq": 5},
                    {"QuoteCode": "B", "ChannelSeq": 1},
                ],
            )
            self.assertEqual(
                states["positive_state"]["ChannelSeq"].to_list(),
                [6, 7, 8],
            )
            subset_ids = (
                "positive_epoch_entry",
                "depth_lt_neg1_epoch_entry",
            )
            subset = select_filllots_at_low_candidates(
                EventDaySources(DATE, roots),
                candidate_ids=subset_ids,
            )
            self.assertEqual(tuple(subset), subset_ids)
            with self.assertRaisesRegex(ValueError, "unknown"):
                select_filllots_at_low_candidates(
                    EventDaySources(DATE, roots),
                    candidate_ids=("not_registered",),
                )

            def sequences(candidate: str, quote: str = "A") -> list[int]:
                return (
                    selected[candidate]
                    .filter(pl.col("QuoteCode") == quote)["ChannelSeq"]
                    .to_list()
                )

            self.assertEqual(sequences("negative_state_update"), [2, 3, 4, 5])
            self.assertEqual(
                sequences("negative_sell_flow_update"), [2, 3, 4, 5]
            )
            self.assertEqual(sequences("negative_buy_flow_update"), [])
            self.assertEqual(sequences("negative_epoch_start"), [5])
            self.assertEqual(sequences("negative_same_epoch_cross"), [2])
            self.assertEqual(sequences("negative_epoch_entry"), [2, 5])
            self.assertEqual(
                sequences("depth_lt_neg1_state_update"), [3, 4]
            )
            self.assertEqual(
                sequences("depth_lt_neg2_state_update"), [4]
            )
            self.assertEqual(
                sequences("depth_lt_neg4_state_update"), [4]
            )
            self.assertEqual(sequences("depth_lt_neg8_state_update"), [])
            self.assertEqual(
                sequences("depth_lt_neg1_epoch_entry"), [3]
            )
            self.assertEqual(
                sequences("depth_lt_neg2_epoch_entry"), [4]
            )
            self.assertEqual(
                sequences("depth_lt_neg4_epoch_entry"), [4]
            )
            self.assertEqual(sequences("depth_lt_neg8_epoch_entry"), [])
            self.assertEqual(sequences("positive_state_update"), [6, 7])
            self.assertEqual(sequences("positive_same_epoch_cross"), [6])
            self.assertEqual(sequences("positive_epoch_entry"), [6])
            self.assertEqual(
                sequences("depth_gt_pos1_epoch_entry"), [7]
            )
            # The first chronological raw row starts an epoch.  It is an
            # entry if it is a formal trade and already has negative state,
            # but it is not a state update because no prior row exists.
            self.assertEqual(sequences("negative_state_update", quote="B"), [])
            self.assertEqual(sequences("negative_epoch_start", quote="B"), [1])
            self.assertEqual(sequences("negative_epoch_entry", quote="B"), [1])
            self.assertEqual(
                sequences("depth_lt_neg1_epoch_entry", quote="B"), []
            )
            # A TrialMatch row contributes zero to recomputed state, and the
            # explicit formal-trade gate blocks it from every candidate.
            for candidate in CANDIDATE_IDS:
                self.assertNotIn(8, sequences(candidate))

            def keys(candidate: str) -> set[tuple[str, int]]:
                return {
                    (row["QuoteCode"], row["ChannelSeq"])
                    for row in selected[candidate].to_dicts()
                }

            negative_updates = keys("negative_state_update")
            sell_updates = keys("negative_sell_flow_update")
            buy_updates = keys("negative_buy_flow_update")
            self.assertEqual(sell_updates | buy_updates, negative_updates)
            self.assertFalse(sell_updates & buy_updates)

            epoch_starts = keys("negative_epoch_start")
            same_epoch_crosses = keys("negative_same_epoch_cross")
            self.assertEqual(
                epoch_starts | same_epoch_crosses,
                keys("negative_epoch_entry"),
            )
            self.assertFalse(epoch_starts & same_epoch_crosses)

            depth_ladder = [
                keys("depth_lt_neg1_state_update"),
                keys("depth_lt_neg2_state_update"),
                keys("depth_lt_neg4_state_update"),
                keys("depth_lt_neg8_state_update"),
            ]
            for shallow, deep in zip(depth_ladder, depth_ladder[1:]):
                self.assertTrue(deep <= shallow)

    def test_epoch_entry_means_each_reentry_not_first_entry_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            roots = DataRoots.from_project_root(Path(tmp))
            roots.tick_data_dir.mkdir(parents=True)
            roots.pre_market_dir.mkdir(parents=True)
            pl.DataFrame(
                {
                    "QuoteCode": ["A"] * 4,
                    "ChannelSeq": pl.Series([1, 2, 3, 4], dtype=pl.UInt64),
                    "FillLots": [2, 4, 4, 4],
                    "InOut": [-1, 1, -1, 1],
                    "TrialMatch": [0] * 4,
                    "marketOpen": [True] * 4,
                    "RecordLow": [10.0] * 4,
                    "FillPrice": [10.0] * 4,
                }
            ).write_parquet(
                roots.tick_data_dir / f"{DATE}_StockTick.parquet"
            )
            pl.DataFrame(
                {
                    "QuoteCode": ["A"],
                    "avg_bidLots1": [0.5],
                    "avg_askLots1": [0.5],
                }
            ).write_parquet(
                roots.pre_market_dir / f"{DATE}_preMarketData.parquet"
            )

            selected = select_filllots_at_low_candidates(
                EventDaySources(DATE, roots)
            )
            self.assertEqual(
                selected["positive_epoch_entry"]["ChannelSeq"].to_list(),
                [2, 4],
            )
            self.assertEqual(
                selected["depth_gt_pos1_epoch_entry"]["ChannelSeq"].to_list(),
                [2, 4],
            )
            self.assertEqual(
                selected["depth_lt_neg1_epoch_entry"]["ChannelSeq"].to_list(),
                [1, 3],
            )
            self.assertIn(
                "re-enters",
                CANDIDATE_DESCRIPTIONS["depth_gt_pos1_epoch_entry"],
            )

    def test_missing_trial_feature_row_does_not_hide_first_formal_update(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            roots = DataRoots.from_project_root(Path(tmp))
            for directory in (
                roots.tick_data_dir,
                roots.tick_feature_dir,
                roots.pre_market_dir,
            ):
                directory.mkdir(parents=True)

            pl.DataFrame(
                {
                    "QuoteCode": ["A", "A"],
                    "ChannelSeq": pl.Series([1, 2], dtype=pl.UInt64),
                    "FillLots": [0, 3],
                    "InOut": [0, -1],
                    "TrialMatch": [1, 0],
                    "marketOpen": [False, True],
                    "RecordLow": [0.0, 9.0],
                    "FillPrice": [10.0, 9.0],
                }
            ).write_parquet(
                roots.tick_data_dir / f"{DATE}_StockTick.parquet"
            )
            # Historical tickFeature omits the trial row.  Its supplied state is
            # deliberately wrong: family v3 must not read this derived column.
            pl.DataFrame(
                {
                    "QuoteCode": ["A"],
                    "ChannelSeq": pl.Series([2], dtype=pl.UInt64),
                    "FillLots_atLow": [999],
                }
            ).write_parquet(
                roots.tick_feature_dir / f"{DATE}_tickFeature.parquet"
            )
            pl.DataFrame(
                {
                    "QuoteCode": ["A"],
                    "avg_bidLots1": [1.0],
                    "avg_askLots1": [1.0],
                }
            ).write_parquet(
                roots.pre_market_dir / f"{DATE}_preMarketData.parquet"
            )

            day = EventDaySources(DATE, roots)
            frame = build_filllots_at_low_candidate_frame(day)
            first_formal = frame.filter(pl.col("ChannelSeq") == 2).row(
                0, named=True
            )
            self.assertEqual(first_formal["FillLots_atLow"], -3)
            self.assertEqual(first_formal["_previous_filllots_at_low"], 0)
            self.assertFalse(first_formal["_previous_market_open"])
            self.assertTrue(first_formal["_state_changed"])
            self.assertTrue(first_formal["_epoch_started"])

            selected = select_filllots_at_low_candidates(day)
            for candidate_id in (
                "negative_state_update",
                "negative_epoch_start",
                "negative_epoch_entry",
                "depth_lt_neg1_state_update",
                "depth_lt_neg1_epoch_entry",
            ):
                self.assertEqual(
                    selected[candidate_id]["ChannelSeq"].to_list(),
                    [2],
                )


if __name__ == "__main__":
    unittest.main()
