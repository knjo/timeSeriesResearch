from __future__ import annotations

import ast
import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import polars as pl

from src.research.timeSeries import build_index as deployed_index
from src.research.timeSeries import data_loader as deployed_loader


def _write_event_source(root: Path, date: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{date}_tickFeature.parquet"
    pl.DataFrame(
        {
            "QuoteCode": ["A", "A", "A"],
            "ChannelSeq": pl.Series([2, 1, 3], dtype=pl.UInt64),
            "Spread": [2.0, 1.0, 2.0],
            "BidPreMove": [0.5, 0.5, 0.5],
            "FillLots_atLow": [-1.0, -1.0, -1.0],
        }
    ).write_parquet(path)
    return path


class DeployedPipelineTest(unittest.TestCase):
    def test_filter_sorts_before_diff_and_requires_fixed_event_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = deployed_index.filter_one_day(
                _write_event_source(root, "20240102")
            )
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.rows(), [("20240102", "A", 2)])

            missing_path = root / "20240103_tickFeature.parquet"
            pl.DataFrame(
                {
                    "QuoteCode": ["A"],
                    "ChannelSeq": pl.Series([1], dtype=pl.UInt64),
                    "Spread": [1.0],
                    "FillLots_atLow": [-1.0],
                }
            ).write_parquet(missing_path)
            with self.assertRaises(deployed_index.DeployedIndexError):
                deployed_index.filter_one_day(missing_path)

    def test_single_incremental_call_appends_new_day_without_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "tickFeature"
            output_path = root / "data_index.parquet"
            _write_event_source(source_dir, "20240102")

            first = deployed_index.build_index(
                tick_feature_dir=source_dir,
                output_path=output_path,
            )
            self.assertEqual(first["Date"].unique().to_list(), ["20240102"])

            _write_event_source(source_dir, "20240103")
            second = deployed_index.build_index(
                tick_feature_dir=source_dir,
                output_path=output_path,
            )
            self.assertEqual(
                second["Date"].unique().sort().to_list(),
                ["20240102", "20240103"],
            )
            self.assertFalse(output_path.with_suffix(".manifest.json").exists())
            self.assertEqual(
                stat.S_IMODE(output_path.stat().st_mode),
                0o664,
            )

            mtime = output_path.stat().st_mtime_ns
            resumed = deployed_index.build_index(
                tick_feature_dir=source_dir,
                output_path=output_path,
            )
            self.assertEqual(resumed.rows(), second.rows())
            self.assertEqual(output_path.stat().st_mtime_ns, mtime)

    def test_bounded_force_replaces_only_requested_day(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "tickFeature"
            output_path = root / "data_index.parquet"
            _write_event_source(source_dir, "20240102")
            day_two = _write_event_source(source_dir, "20240103")
            original = deployed_index.build_index(
                tick_feature_dir=source_dir,
                output_path=output_path,
            )

            changed = pl.read_parquet(day_two).with_columns(
                pl.lit(1.0).alias("Spread")
            )
            changed.write_parquet(day_two)
            rebuilt = deployed_index.build_index(
                forcefetch=True,
                start="20240103",
                end="20240103",
                tick_feature_dir=source_dir,
                output_path=output_path,
            )

            self.assertEqual(
                rebuilt.filter(pl.col("Date") == "20240102").rows(),
                original.filter(pl.col("Date") == "20240102").rows(),
            )
            self.assertEqual(
                rebuilt.filter(pl.col("Date") == "20240103").height,
                0,
            )

    def test_daily_loader_writes_only_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_path = root / "data_index.parquet"
            output_dir = root / "data"
            pl.DataFrame(
                {
                    "Date": ["20240102"],
                    "QuoteCode": ["A"],
                    "ChannelSeq": pl.Series([1], dtype=pl.UInt64),
                }
            ).write_parquet(index_path)
            fake_day = pl.DataFrame(
                {
                    "QuoteCode": ["A"],
                    "ChannelSeq": pl.Series([1], dtype=pl.UInt64),
                    "Date": ["20240102"],
                }
            )
            source_names = (
                "tickData",
                "tickFeature",
                "preMarket",
                "tickBar",
                "txfTickData",
            )
            paths = deployed_loader.DeploymentDataPaths(
                *(root / name for name in source_names)
            )

            with (
                patch.object(deployed_loader, "TxfDataLoader", return_value=object()),
                patch.object(deployed_loader, "_merge_one_day", return_value=fake_day),
            ):
                report = deployed_loader.build_merged_data(
                    index_path=index_path,
                    output_dir=output_dir,
                    data_paths=paths,
                )

            self.assertEqual(report.saved_dates, ("20240102",))
            self.assertEqual(
                sorted(path.name for path in output_dir.iterdir()),
                ["20240102.parquet"],
            )
            self.assertEqual(
                stat.S_IMODE((output_dir / "20240102.parquet").stat().st_mode),
                0o664,
            )

    def test_reader_ignores_non_parquet_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pl.DataFrame({"x": [1]}).write_parquet(root / "20240102.parquet")
            (root / "unexpected.json").write_text(
                json.dumps({"not": "a daily point"}), encoding="utf-8"
            )
            result = deployed_loader.read_time_series(output_dir=root)
            self.assertEqual(result.to_dicts(), [{"x": 1}])

    def test_deployed_entrypoints_do_not_import_research_framework(self) -> None:
        for path in (
            Path(deployed_index.__file__),
            Path(deployed_loader.__file__),
        ):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imported_modules = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_modules.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module is not None:
                    imported_modules.append(node.module)
            self.assertFalse(
                any(
                    module.startswith("src.research.timeSeries.framework")
                    for module in imported_modules
                ),
                f"deployed entrypoint imports research framework: {path}",
            )


if __name__ == "__main__":
    unittest.main()
