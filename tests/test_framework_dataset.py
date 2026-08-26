from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, time
from pathlib import Path

import polars as pl

from src.research.timeSeries.framework.dataset import (
    EventDatasetSpec,
    DatasetContractError,
    MergeOptions,
    UniverseSpec,
    build_event_dataset,
    materialize_event_day,
    read_event_dataset,
)
from src.research.timeSeries.framework.events import DataRoots
from src.research.timeSeries.framework.profiles.tw_equity import (
    NEXTDAY_ALLOW_DAY_TRADE_UNIVERSE_V1,
)


DATE = "20240102"


def _roots(root: Path) -> DataRoots:
    roots = DataRoots.from_project_root(root)
    for path in (
        roots.tick_data_dir,
        roots.tick_feature_dir,
        roots.pre_market_dir,
        roots.tick_bar_dir,
        roots.txf_tick_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return roots


def _write_minimal_sources(roots: DataRoots) -> None:
    event_time = datetime(2024, 1, 2, 9, 7)
    pl.DataFrame(
        {
            "RecvTime": [event_time],
            "QuoteCode": ["2330"],
            "ChannelSeq": [1],
            "TransTime": [event_time],
            "BidPrice1": [100.0],
        },
        schema_overrides={"ChannelSeq": pl.UInt64},
    ).write_parquet(roots.tick_data_dir / f"{DATE}_StockTick.parquet")
    pl.DataFrame(
        {
            "QuoteCode": ["2330"],
            "ChannelSeq": [1],
            "feature_x": [2.0],
        },
        schema_overrides={"ChannelSeq": pl.UInt64},
    ).write_parquet(roots.tick_feature_dir / f"{DATE}_tickFeature.parquet")
    pl.DataFrame(
        {
            "QuoteCode": ["2330"],
            "nextday_allow_day_trade_mark": ["X"],
        }
    ).write_parquet(roots.pre_market_dir / f"{DATE}_preMarketData.parquet")
    # The stock did not trade between 09:04 and 09:10.  The 09:10 cross-
    # section therefore reuses its 09:04 stock tick, but is not knowable at a
    # 09:07 event.  Alignment must use the completed TimeSlot, not TransTime.
    pl.DataFrame(
        {
            "QuoteCode": ["2330", "2330"],
            "TimeSlot": [time(9, 5), time(9, 10)],
            "TransTime": [
                datetime(2024, 1, 2, 9, 4),
                datetime(2024, 1, 2, 9, 4),
            ],
            "AmountRank": [10, 1],
        }
    ).write_parquet(roots.tick_bar_dir / f"{DATE}_tickBar.parquet")


def _index_rows() -> pl.DataFrame:
    return pl.DataFrame(
        {"Date": [DATE], "QuoteCode": ["2330"], "ChannelSeq": [1]},
        schema_overrides={"ChannelSeq": pl.UInt64},
    )


class FrameworkDatasetTest(unittest.TestCase):
    def test_source_order_policy_preserves_legacy_tick_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            roots = _roots(Path(tmp))
            _write_minimal_sources(roots)
            result = materialize_event_day(
                DATE,
                _index_rows(),
                roots=roots,
                options=MergeOptions(
                    attach_txf=False,
                    preserve_tick_data_column_order=True,
                ),
            )
            self.assertEqual(
                result.frame.columns[:5],
                [
                    "RecvTime",
                    "QuoteCode",
                    "ChannelSeq",
                    "TransTime",
                    "BidPrice1",
                ],
            )

    def test_generic_merge_is_row_preserving_and_causal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            roots = _roots(Path(tmp))
            _write_minimal_sources(roots)
            result = materialize_event_day(
                DATE,
                _index_rows(),
                roots=roots,
                options=MergeOptions(attach_txf=False),
            )
            self.assertEqual(len(result.frame), 1)
            self.assertEqual(result.frame["AmountRank"][0], 10)
            self.assertEqual(result.frame["TimeSlot"][0], time(9, 5))
            self.assertEqual(result.audit.indexed_rows, 1)
            self.assertEqual(result.audit.universe_removed_rows, 0)

    def test_universe_is_explicit_and_missing_eligibility_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            roots = _roots(Path(tmp))
            _write_minimal_sources(roots)
            path = roots.pre_market_dir / f"{DATE}_preMarketData.parquet"
            pl.DataFrame({"QuoteCode": ["2330"]}).write_parquet(path)
            with self.assertRaisesRegex(DatasetContractError, "requires"):
                materialize_event_day(
                    DATE,
                    _index_rows(),
                    roots=roots,
                    options=MergeOptions(
                        universe=NEXTDAY_ALLOW_DAY_TRADE_UNIVERSE_V1,
                        attach_txf=False,
                    ),
                )

    def test_missing_tick_key_is_not_silently_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            roots = _roots(Path(tmp))
            _write_minimal_sources(roots)
            rows = pl.concat(
                [
                    _index_rows(),
                    pl.DataFrame(
                        {
                            "Date": [DATE],
                            "QuoteCode": ["2317"],
                            "ChannelSeq": [2],
                        },
                        schema_overrides={"ChannelSeq": pl.UInt64},
                    ),
                ]
            )
            with self.assertRaisesRegex(DatasetContractError, "lost 1"):
                materialize_event_day(
                    DATE,
                    rows,
                    roots=roots,
                    options=MergeOptions(attach_txf=False),
                )

    def test_reader_rejects_schema_drift_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pl.DataFrame({"x": [1]}).write_parquet(root / "20240102.parquet")
            pl.DataFrame({"x": [1], "y": [2]}).write_parquet(
                root / "20240103.parquet"
            )
            with self.assertRaisesRegex(DatasetContractError, "schema drift"):
                read_event_dataset(root)
            union = read_event_dataset(root, schema_policy="union")
            self.assertEqual(union.shape, (2, 2))

    def test_universe_callback_can_only_choose_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            roots = _roots(Path(tmp))
            _write_minimal_sources(roots)

            def malicious(frame: pl.DataFrame) -> pl.DataFrame:
                return frame.with_columns(pl.lit(-999.0).alias("BidPrice1"))

            result = materialize_event_day(
                DATE,
                _index_rows(),
                roots=roots,
                options=MergeOptions(
                    universe=UniverseSpec("malicious", "1", malicious),
                    attach_txf=False,
                ),
            )
            self.assertEqual(result.frame["BidPrice1"][0], 100.0)

    def test_versioned_build_tracks_sources_and_noop_does_not_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roots = _roots(root)
            _write_minimal_sources(roots)
            index_path = root / "event_index.parquet"
            output_dir = root / "events_v1"
            _index_rows().write_parquet(index_path)
            spec = EventDatasetSpec(
                event_id="example",
                dataset_version="1",
                index_path=index_path,
                output_dir=output_dir,
                merge_options=MergeOptions(attach_txf=False),
            )
            first = build_event_dataset(spec, roots=roots)
            self.assertEqual(first.saved_dates, (DATE,))
            output_path = output_dir / f"{DATE}.parquet"
            first_mtime = output_path.stat().st_mtime_ns

            second = build_event_dataset(spec, roots=roots)
            self.assertEqual(second.saved_dates, ())
            self.assertEqual(second.skipped_dates, (DATE,))
            self.assertEqual(output_path.stat().st_mtime_ns, first_mtime)

            # A source identity change refreshes exactly the stale partition.
            feature_path = roots.tick_feature_dir / f"{DATE}_tickFeature.parquet"
            feature = pl.read_parquet(feature_path).with_columns(
                pl.lit(3.0).alias("feature_x")
            )
            feature.write_parquet(feature_path)
            refreshed = build_event_dataset(spec, roots=roots)
            self.assertEqual(refreshed.saved_dates, (DATE,))
            self.assertEqual(
                pl.read_parquet(output_path)["feature_x"][0], 3.0
            )

    def test_invalid_index_does_not_publish_dataset_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roots = _roots(root)
            _write_minimal_sources(roots)
            index_path = root / "bad_index.parquet"
            _index_rows().with_columns(
                pl.col("ChannelSeq").cast(pl.Int64)
            ).write_parquet(index_path)
            spec = EventDatasetSpec(
                "bad",
                "1",
                index_path,
                root / "output",
                MergeOptions(attach_txf=False),
            )
            with self.assertRaises(Exception):
                build_event_dataset(spec, roots=roots)
            self.assertFalse(spec.manifest_path.exists())

    def test_strict_reader_treats_column_order_as_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pl.DataFrame({"x": [1], "y": [2]}).write_parquet(
                root / "20240102.parquet"
            )
            pl.DataFrame({"y": [3], "x": [4]}).write_parquet(
                root / "20240103.parquet"
            )
            with self.assertRaisesRegex(DatasetContractError, "schema drift"):
                read_event_dataset(root)


if __name__ == "__main__":
    unittest.main()
