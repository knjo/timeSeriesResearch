from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import polars as pl

from src.research.intradaySignal.framework.events import EventIndexValidationError


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "20260825_filllots_at_low_framework"
    / "run_negfill_overlap.py"
)
SPEC = importlib.util.spec_from_file_location(
    "filllots_negfill_overlap_runner",
    RUNNER_PATH,
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import guard
    raise RuntimeError(f"cannot import overlap runner from {RUNNER_PATH}")
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def _keys(rows: list[tuple[str, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={"QuoteCode": pl.String, "ChannelSeq": pl.UInt64},
        orient="row",
    )


def _daily_overlap_frame(
    date: str,
    candidate_id: str,
    *,
    source_identities_json: str = "{}",
) -> pl.DataFrame:
    condition_counts = [2, 1, 1, 2, 0]
    return pl.DataFrame(
        {
            "Date": [date] * len(RUNNER.COHORT_IDS),
            "cohort_id": list(RUNNER.COHORT_IDS),
            "contract_id": [RUNNER.CONTRACT_ID] * len(RUNNER.COHORT_IDS),
            "family_id": [RUNNER.FILLLOTS_AT_LOW_FAMILY_ID]
            * len(RUNNER.COHORT_IDS),
            "family_version": [RUNNER.FILLLOTS_AT_LOW_FAMILY_VERSION]
            * len(RUNNER.COHORT_IDS),
            "candidate_id": [candidate_id] * len(RUNNER.COHORT_IDS),
            "negfill_selector_version": [RUNNER.NEGFILL_SELECTOR_VERSION]
            * len(RUNNER.COHORT_IDS),
            "label_profile_id": [RUNNER.SHORT_TAKER_STOP108_V1.profile_id]
            * len(RUNNER.COHORT_IDS),
            "label_profile_hash": [RUNNER.SHORT_TAKER_STOP108_V1.fingerprint]
            * len(RUNNER.COHORT_IDS),
            "pool_id": ["fixture_pool"] * len(RUNNER.COHORT_IDS),
            "pool_version": ["1"] * len(RUNNER.COHORT_IDS),
            "control_mode": ["full"] * len(RUNNER.COHORT_IDS),
            "control_sample_size": [0] * len(RUNNER.COHORT_IDS),
            RUNNER.SOURCE_IDENTITY_COLUMN: [source_identities_json]
            * len(RUNNER.COHORT_IDS),
            "eligible_pool_rows": [100] * len(RUNNER.COHORT_IDS),
            "candidate_index_rows": [4] * len(RUNNER.COHORT_IDS),
            "negfill_index_rows": [3] * len(RUNNER.COHORT_IDS),
            "exact_overlap_index_rows": [1] * len(RUNNER.COHORT_IDS),
            "exact_disjoint_index_rows": [3] * len(RUNNER.COHORT_IDS),
            "stockday_overlap_index_rows": [3] * len(RUNNER.COHORT_IDS),
            "stockday_disjoint_index_rows": [1] * len(RUNNER.COHORT_IDS),
            "candidate_unique_quotes": [3] * len(RUNNER.COHORT_IDS),
            "negfill_unique_quotes": [3] * len(RUNNER.COHORT_IDS),
            "overlapping_quotes": [2] * len(RUNNER.COHORT_IDS),
            "exact_overlap_rate_candidate": [0.25] * len(RUNNER.COHORT_IDS),
            "stockday_overlap_rate_candidate": [0.75]
            * len(RUNNER.COHORT_IDS),
            "stockday_overlap_rate_quotes": [2.0 / 3.0]
            * len(RUNNER.COHORT_IDS),
            "indexed_condition_rows": condition_counts,
            "condition_excluded_rows": [0] * len(RUNNER.COHORT_IDS),
            "condition_unique_quotes": condition_counts,
            "selection_rate": [count / 100.0 for count in condition_counts],
            "condition_n": condition_counts,
            "condition_n_z": condition_counts,
            "condition_sum_z": [float(count) for count in condition_counts],
            "condition_sum_raw_bp": [
                float(count) for count in condition_counts
            ],
            "condition_sum_residual_bp": [
                float(count) for count in condition_counts
            ],
            "control_n": [100] * len(RUNNER.COHORT_IDS),
            "control_n_z": [100] * len(RUNNER.COHORT_IDS),
            "control_sum_z": [0.0] * len(RUNNER.COHORT_IDS),
            "control_sum_raw_bp": [0.0] * len(RUNNER.COHORT_IDS),
            "control_sum_residual_bp": [0.0] * len(RUNNER.COHORT_IDS),
        }
    )


class FillLotsNegfillOverlapRunnerTest(unittest.TestCase):
    def test_exact_and_stockday_partitions_are_distinct(self) -> None:
        candidate = _keys(
            [("A", 1), ("A", 2), ("B", 1), ("C", 1)]
        )
        negfill = _keys([("A", 2), ("B", 9), ("D", 1)])

        result = RUNNER.partition_candidate_by_negfill(
            candidate,
            negfill,
            date="20260825",
        )

        self.assertEqual(
            result.exact_overlap.to_dicts(),
            [{"QuoteCode": "A", "ChannelSeq": 2}],
        )
        self.assertEqual(
            result.exact_disjoint.to_dicts(),
            [
                {"QuoteCode": "A", "ChannelSeq": 1},
                {"QuoteCode": "B", "ChannelSeq": 1},
                {"QuoteCode": "C", "ChannelSeq": 1},
            ],
        )
        self.assertEqual(
            result.stockday_overlap.to_dicts(),
            [
                {"QuoteCode": "A", "ChannelSeq": 1},
                {"QuoteCode": "A", "ChannelSeq": 2},
                {"QuoteCode": "B", "ChannelSeq": 1},
            ],
        )
        self.assertEqual(
            result.stockday_disjoint.to_dicts(),
            [{"QuoteCode": "C", "ChannelSeq": 1}],
        )
        self.assertEqual(result.overlapping_quotes, 2)

    def test_duplicate_candidate_keys_fail_closed(self) -> None:
        candidate = _keys([("A", 1), ("A", 1)])
        negfill = _keys([("A", 1)])

        with self.assertRaises(EventIndexValidationError):
            RUNNER.partition_candidate_by_negfill(
                candidate,
                negfill,
                date="20260825",
            )

    def test_cached_day_rejects_replaced_source_and_legacy_identity(self) -> None:
        date = "20250102"
        candidate_id = RUNNER.CANDIDATE_IDS[0]
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            roots = RUNNER.DataRoots(
                tick_data_dir=root / "tick_data",
                tick_feature_dir=root / "tick_feature",
                pre_market_dir=root / "pre_market",
                tick_bar_dir=root / "tick_bar",
                txf_tick_dir=root / "txf_tick",
            )
            for directory in (
                roots.tick_data_dir,
                roots.tick_feature_dir,
                roots.pre_market_dir,
                roots.tick_bar_dir,
                roots.txf_tick_dir,
            ):
                directory.mkdir()
            source_paths = {
                "tick_data": (
                    roots.tick_data_dir / f"{date}_StockTick.parquet"
                ),
                "pre_market": (
                    roots.pre_market_dir / f"{date}_preMarketData.parquet"
                ),
                "tick_feature": (
                    roots.tick_feature_dir / f"{date}_tickFeature.parquet"
                ),
                "txf_tick": roots.txf_tick_dir / f"{date}_TxfTick.parquet",
            }
            for source, path in source_paths.items():
                path.write_bytes(f"{source}-v1".encode())

            original_identity = RUNNER._day_source_identities_json(date, roots)
            identity_payload = json.loads(original_identity)
            self.assertEqual(
                set(identity_payload),
                set(RUNNER.SOURCE_IDENTITY_SOURCES),
            )
            for source in RUNNER.SOURCE_IDENTITY_SOURCES:
                self.assertEqual(
                    set(identity_payload[source]),
                    {"path", "size", "mtime_ns"},
                )

            cached = _daily_overlap_frame(
                date,
                candidate_id,
                source_identities_json=original_identity,
            )
            RUNNER._validate_cached_day(
                cached,
                date=date,
                candidate_id=candidate_id,
                control_sample_size=None,
                source_identities_json=original_identity,
            )

            # Replacing one dependency in place must invalidate the daily cache.
            source_paths["tick_feature"].write_bytes(
                b"tick-feature-v2-with-a-different-size"
            )
            replacement_identity = RUNNER._day_source_identities_json(
                date,
                roots,
            )
            self.assertNotEqual(replacement_identity, original_identity)
            with self.assertRaisesRegex(
                ValueError,
                RUNNER.SOURCE_IDENTITY_COLUMN,
            ):
                RUNNER._validate_cached_day(
                    cached,
                    date=date,
                    candidate_id=candidate_id,
                    control_sample_size=None,
                    source_identities_json=replacement_identity,
                )

            # Pre-identity daily artifacts are intentionally not reusable.
            with self.assertRaisesRegex(
                ValueError,
                RUNNER.SOURCE_IDENTITY_COLUMN,
            ):
                RUNNER._validate_cached_day(
                    cached.drop(RUNNER.SOURCE_IDENTITY_COLUMN),
                    date=date,
                    candidate_id=candidate_id,
                    control_sample_size=None,
                    source_identities_json=original_identity,
                )

    def test_run_aggregates_requested_sessions_and_records_identities(self) -> None:
        date = "20250102"
        candidate_id = RUNNER.CANDIDATE_IDS[0]
        pool_spec = SimpleNamespace(pool_id="fixture_pool")
        with tempfile.TemporaryDirectory() as temp_directory:
            output_dir = Path(temp_directory) / "artifacts"
            daily_dir = output_dir / "daily"
            daily_dir.mkdir(parents=True)
            unrelated_path = daily_dir / "19990101.parquet"
            pl.DataFrame({"unrelated": [True]}).write_parquet(unrelated_path)
            evaluated = _daily_overlap_frame(
                date,
                candidate_id,
                source_identities_json="{}",
            )
            args = Namespace(
                start=date,
                end=date,
                step=1,
                output_dir=output_dir,
                candidate=candidate_id,
                control_sample=0,
                force=False,
                continue_on_error=False,
            )
            with (
                mock.patch.object(
                    RUNNER.DataRoots,
                    "from_project_root",
                    return_value=SimpleNamespace(txf_tick_dir=Path("unused")),
                ),
                mock.patch.object(
                    RUNNER,
                    "available_sessions",
                    return_value=[date],
                ),
                mock.patch.object(
                    RUNNER,
                    "reference_pool_spec",
                    return_value=pool_spec,
                ),
                mock.patch.object(RUNNER, "TxfDataLoader", return_value=object()),
                mock.patch.object(
                    RUNNER,
                    "_day_source_identities_json",
                    return_value="{}",
                ),
                mock.patch.object(
                    RUNNER,
                    "evaluate_day",
                    return_value=evaluated,
                ) as evaluate,
            ):
                daily, summary = RUNNER.run(args)

            self.assertEqual(len(daily), len(RUNNER.COHORT_IDS))
            self.assertEqual(daily["Date"].unique().to_list(), [date])
            self.assertIn(RUNNER.SOURCE_IDENTITY_COLUMN, daily.columns)
            self.assertFalse(summary.is_empty())
            evaluate.assert_called_once()
            self.assertEqual(
                evaluate.call_args.kwargs["source_identities_json"],
                "{}",
            )
            cached = pl.read_parquet(daily_dir / f"{date}.parquet")
            self.assertIn(RUNNER.SOURCE_IDENTITY_COLUMN, cached.columns)
            report_daily = pl.read_parquet(output_dir / "daily_stats.parquet")
            self.assertNotIn(RUNNER.SOURCE_IDENTITY_COLUMN, report_daily.columns)
            self.assertTrue(unrelated_path.exists())

            manifest = json.loads((output_dir / "run_manifest.json").read_text())
            self.assertEqual(manifest["sessions"], [date])
            self.assertEqual(
                manifest["source_identity"],
                {
                    "column": RUNNER.SOURCE_IDENTITY_COLUMN,
                    "sources": list(RUNNER.SOURCE_IDENTITY_SOURCES),
                    "fields": ["path", "size", "mtime_ns"],
                },
            )
            self.assertEqual(manifest["source_identities"], {date: {}})
            self.assertEqual(manifest["quarantined_daily_outputs"], [])


if __name__ == "__main__":
    unittest.main()
