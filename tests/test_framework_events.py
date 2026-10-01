from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import polars as pl

from src.research.intradaySignal.framework.events import (
    EVENT_KEY_COLS,
    DataRoots,
    EventDaySources,
    EventIndexManifestError,
    EventIndexSpec,
    EventIndexValidationError,
    build_event_index,
    normalize_event_keys,
    validate_event_index,
)
from src.research.intradaySignal.framework.profiles.negfill import (
    negfill_event_index_spec,
    select_negfill_events,
)


def _make_roots(root: Path) -> DataRoots:
    roots = DataRoots.from_project_root(root)
    for directory in (
        roots.tick_data_dir,
        roots.tick_feature_dir,
        roots.pre_market_dir,
        roots.tick_bar_dir,
        roots.txf_tick_dir,
    ):
        directory.mkdir(parents=True)
    return roots


def _write_day_sources(roots: DataRoots, date: str) -> None:
    pl.DataFrame(
        {
            "QuoteCode": ["B", "A", "A"],
            "ChannelSeq": pl.Series([2, 3, 1], dtype=pl.UInt64),
            "feature_flag": [True, True, True],
        }
    ).write_parquet(roots.tick_feature_dir / f"{date}_tickFeature.parquet")
    pl.DataFrame(
        {
            "QuoteCode": ["A", "B", "A"],
            "ChannelSeq": pl.Series([1, 2, 3], dtype=pl.UInt64),
            "tick_flag": [True, True, False],
        }
    ).write_parquet(roots.tick_data_dir / f"{date}_StockTick.parquet")
    pl.DataFrame(
        {
            "QuoteCode": ["A", "B"],
            "eligible": [True, True],
        }
    ).write_parquet(roots.pre_market_dir / f"{date}_preMarketData.parquet")


def _multi_source_selector(day: EventDaySources) -> pl.LazyFrame:
    return (
        day.scan_tick_feature()
        .join(
            day.scan_tick_data(),
            on=["QuoteCode", "ChannelSeq"],
            how="inner",
            validate="1:1",
        )
        .join(day.scan_pre_market(), on="QuoteCode", how="inner", validate="m:1")
        .filter(pl.col("feature_flag") & pl.col("tick_flag") & pl.col("eligible"))
        .select("QuoteCode", "ChannelSeq")
    )


class EventFrameworkTest(unittest.TestCase):
    def test_data_roots_use_pipeline_config_but_keep_temp_root_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fallback = DataRoots.from_project_root(root)
            self.assertEqual(fallback.tick_data_dir, root / "data" / "tickData")

            configured_base = root / "external" / "Data"
            config_dir = root / "config"
            config_dir.mkdir()
            (config_dir / "pipeline.yaml").write_text(
                """
data_storage:
  base_dir: external/Data
  tick_dir: stockTicks
  txf_tick_dir: futuresTicks
""".lstrip(),
                encoding="utf-8",
            )

            configured = DataRoots.from_project_root(root)
            self.assertEqual(configured.tick_data_dir, configured_base / "stockTicks")
            self.assertEqual(
                configured.txf_tick_dir,
                configured_base / "futuresTicks",
            )
            self.assertEqual(
                configured.pre_market_dir,
                configured_base / "preMarket",
            )

    def test_negfill_profile_owns_versioned_event_index_spec(self) -> None:
        spec = negfill_event_index_spec(Path("negfill.parquet"))

        self.assertEqual(spec.event_id, "negfill")
        self.assertEqual(spec.selector_version, "negfill_spread_bid_move_v1")
        self.assertIs(spec.selector, select_negfill_events)
        self.assertEqual(spec.required_sources, ("tick_feature",))
        self.assertEqual(spec.calendar_source, "tick_feature")
        self.assertFalse(spec.allow_unmanifested_existing)

    def test_multi_source_selector_builds_sorted_canonical_index_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roots = _make_roots(root)
            _write_day_sources(roots, "20240103")
            _write_day_sources(roots, "20240102")
            index_path = root / "events.parquet"
            spec = EventIndexSpec(
                event_id="multi_source",
                selector_version="v1",
                selector=_multi_source_selector,
                index_path=index_path,
                required_sources=("tick_feature", "tick_data", "pre_market"),
            )

            report = build_event_index(spec, roots=roots)
            self.assertEqual(report.scanned_dates, ("20240102", "20240103"))
            self.assertEqual(report.row_count, 4)
            self.assertEqual(report.unique_key_count, 4)
            self.assertEqual(report.processed_date_count, 2)

            result = report.read_index()
            self.assertEqual(tuple(result.columns), EVENT_KEY_COLS)
            self.assertEqual(result.schema["Date"], pl.String)
            self.assertEqual(result.schema["QuoteCode"], pl.String)
            self.assertEqual(result.schema["ChannelSeq"], pl.UInt64)
            self.assertEqual(
                result.select(EVENT_KEY_COLS).rows(),
                [
                    ("20240102", "A", 1),
                    ("20240102", "B", 2),
                    ("20240103", "A", 1),
                    ("20240103", "B", 2),
                ],
            )

            with report.manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertEqual(manifest["event_id"], "multi_source")
            self.assertEqual(manifest["rows_by_date"], {"20240102": 2, "20240103": 2})

            with patch(
                "src.research.intradaySignal.framework.events._atomic_write_parquet"
            ) as parquet_writer:
                resumed = build_event_index(spec, roots=roots)
            parquet_writer.assert_not_called()
            self.assertEqual(resumed.scanned_dates, ())
            self.assertEqual(resumed.skipped_dates, ("20240102", "20240103"))
            self.assertEqual(resumed.row_count, 4)

    def test_legacy_adoption_is_metadata_only_and_tracks_unverified_dates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roots = _make_roots(root)
            _write_day_sources(roots, "20240102")
            index_path = root / "legacy_events.parquet"
            pl.DataFrame(
                {
                    "Date": ["20240102", "20240102"],
                    "QuoteCode": ["A", "B"],
                    "ChannelSeq": pl.Series([1, 2], dtype=pl.UInt64),
                }
            ).write_parquet(index_path)
            inode_before = index_path.stat().st_ino
            bytes_before = index_path.read_bytes()
            spec = EventIndexSpec(
                event_id="legacy_multi_source",
                selector_version="v1",
                selector=_multi_source_selector,
                index_path=index_path,
                required_sources=("tick_feature", "tick_data", "pre_market"),
                allow_unmanifested_existing=True,
            )

            with patch(
                "src.research.intradaySignal.framework.events._atomic_write_parquet"
            ) as parquet_writer:
                adopted = build_event_index(spec, roots=roots)
            parquet_writer.assert_not_called()
            self.assertEqual(adopted.scanned_dates, ())
            self.assertEqual(index_path.stat().st_ino, inode_before)
            self.assertEqual(index_path.read_bytes(), bytes_before)

            with spec.manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertTrue(manifest["legacy_adopted"])
            self.assertFalse(manifest["selector_reproduction_verified"])
            self.assertEqual(manifest["unverified_dates"], ["20240102"])
            self.assertEqual(
                set(manifest["source_identities"]["20240102"]),
                {"tick_feature", "tick_data", "pre_market"},
            )

            # Mutating a declared source changes its identity.  The next build
            # must rerun this date and thereby verify it under the new selector.
            pl.DataFrame(
                {
                    "QuoteCode": ["A", "A"],
                    "ChannelSeq": pl.Series([1, 3], dtype=pl.UInt64),
                    "tick_flag": [True, False],
                }
            ).write_parquet(roots.tick_data_dir / "20240102_StockTick.parquet")
            refreshed = build_event_index(spec, roots=roots)
            self.assertEqual(refreshed.scanned_dates, ("20240102",))
            self.assertEqual(refreshed.refreshed_dates, ("20240102",))
            self.assertEqual(
                refreshed.read_index().rows(),
                [("20240102", "A", 1)],
            )
            with spec.manifest_path.open("r", encoding="utf-8") as handle:
                refreshed_manifest = json.load(handle)
            self.assertTrue(refreshed_manifest["legacy_adopted"])
            self.assertTrue(
                refreshed_manifest["selector_reproduction_verified"]
            )
            self.assertEqual(refreshed_manifest["unverified_dates"], [])

    def test_duplicate_selector_keys_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roots = _make_roots(root)
            _write_day_sources(roots, "20240102")

            def duplicates(_: EventDaySources) -> pl.DataFrame:
                return pl.DataFrame(
                    {
                        "QuoteCode": ["A", "A"],
                        "ChannelSeq": pl.Series([1, 1], dtype=pl.UInt64),
                    }
                )

            spec = EventIndexSpec("dupes", "v1", duplicates, root / "events.parquet")
            with self.assertRaises(EventIndexValidationError):
                build_event_index(spec, roots=roots)
            self.assertFalse(spec.index_path.exists())

    def test_missing_declared_source_fails_instead_of_becoming_zero_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roots = _make_roots(root)
            pl.DataFrame(
                {
                    "QuoteCode": ["A"],
                    "ChannelSeq": pl.Series([1], dtype=pl.UInt64),
                }
            ).write_parquet(roots.tick_feature_dir / "20240102_tickFeature.parquet")
            spec = EventIndexSpec(
                "missing_source",
                "v1",
                _multi_source_selector,
                root / "events.parquet",
                required_sources=("tick_feature", "tick_data", "pre_market"),
            )
            with self.assertRaises(FileNotFoundError):
                build_event_index(spec, roots=roots)

    def test_manifest_rejects_selector_version_drift_without_full_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roots = _make_roots(root)
            _write_day_sources(roots, "20240102")
            path = root / "events.parquet"
            v1 = EventIndexSpec(
                "event",
                "v1",
                _multi_source_selector,
                path,
                required_sources=("tick_feature", "tick_data", "pre_market"),
            )
            build_event_index(v1, roots=roots)

            v2 = EventIndexSpec(
                "event",
                "v2",
                _multi_source_selector,
                path,
                required_sources=("tick_feature", "tick_data", "pre_market"),
            )
            with self.assertRaises(EventIndexManifestError):
                build_event_index(v2, roots=roots)

            rebuilt = build_event_index(v2, roots=roots, force=True)
            self.assertEqual(rebuilt.selector_version, "v2")
            self.assertEqual(rebuilt.scanned_dates, ("20240102",))

    def test_schema_null_uniqueness_and_sort_validators(self) -> None:
        normalized = normalize_event_keys(
            pl.DataFrame(
                {
                    "QuoteCode": ["B", "A"],
                    "ChannelSeq": pl.Series([2, 1], dtype=pl.Int64),
                }
            ),
            date="20240102",
        )
        self.assertEqual(normalized.rows(), [("20240102", "A", 1), ("20240102", "B", 2)])
        validate_event_index(normalized)

        with self.assertRaises(EventIndexValidationError):
            validate_event_index(normalized.reverse())
        with self.assertRaises(EventIndexValidationError):
            normalize_event_keys(
                pl.DataFrame(
                    {
                        "QuoteCode": ["A", None],
                        "ChannelSeq": pl.Series([1, 2], dtype=pl.Int64),
                    }
                ),
                date="20240102",
            )
        with self.assertRaises(EventIndexValidationError):
            normalize_event_keys(
                pl.DataFrame({"QuoteCode": ["A"], "ChannelSeq": ["1"]}),
                date="20240102",
            )

if __name__ == "__main__":
    unittest.main()
