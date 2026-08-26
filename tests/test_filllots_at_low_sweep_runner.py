from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import polars as pl


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "test"
    / "20260825_filllots_at_low_framework"
    / "run_sweep.py"
)
SPEC = importlib.util.spec_from_file_location(
    "filllots_at_low_sweep_runner",
    RUNNER_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load sweep runner: {RUNNER_PATH}")
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


class FillLotsAtLowSweepRunnerTest(unittest.TestCase):
    @staticmethod
    def _day_frame(
        date: str,
        candidate_id: str,
        *,
        label_profile_id: str,
        label_profile_hash: str,
        pool_id: str,
        pool_version: str,
        source_identities_json: str = "{}",
        condition_sum_z: float = 2.0,
    ) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "contract_id": [RUNNER.SWEEP_CONTRACT_ID],
                "Date": [date],
                "family_id": [RUNNER.FILLLOTS_AT_LOW_FAMILY_ID],
                "family_version": [RUNNER.FILLLOTS_AT_LOW_FAMILY_VERSION],
                "candidate_id": [candidate_id],
                "label_profile_id": [label_profile_id],
                "label_profile_hash": [label_profile_hash],
                "pool_id": [pool_id],
                "pool_version": [pool_version],
                "control_mode": ["full"],
                "control_sample_size": [0],
                "primary_estimand": [RUNNER.PRIMARY_ESTIMAND.value],
                "stock_time_bucket": [RUNNER.STOCK_TIME_BUCKET],
                RUNNER.SOURCE_IDENTITY_COLUMN: [source_identities_json],
                "eligible_pool_rows": [100],
                "indexed_condition_rows": [2],
                "condition_excluded_rows": [0],
                "condition_unique_quotes": [1],
                "stock_day_equal_delta_z": [condition_sum_z / 2.0],
                RUNNER.STOCK_TIME_EQUAL_COLUMN: [condition_sum_z / 2.0],
                "selection_rate": [0.02],
                "condition_n": [2],
                "condition_n_z": [2],
                "condition_sum_z": [condition_sum_z],
                "condition_sum_raw_bp": [condition_sum_z],
                "condition_sum_residual_bp": [condition_sum_z],
                "control_n": [100],
                "control_n_z": [100],
                "control_sum_z": [0.0],
                "control_sum_raw_bp": [0.0],
                "control_sum_residual_bp": [0.0],
            }
        )

    def test_available_sessions_does_not_require_tick_feature(self) -> None:
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
            (roots.tick_data_dir / "20250102_StockTick.parquet").touch()
            (roots.pre_market_dir / "20250102_preMarketData.parquet").touch()
            (roots.txf_tick_dir / "20250102_TxfTick.parquet").touch()

            self.assertEqual(
                RUNNER.available_sessions(
                    roots,
                    start="20250101",
                    end="20250103",
                ),
                ["20250102"],
            )

    def test_state_registry_uses_unmodified_state_selector(self) -> None:
        day = object()
        candidate_ids = ("negative_state", "positive_state")
        selected = {
            candidate_id: pl.DataFrame(
                {
                    "QuoteCode": [candidate_id],
                    "ChannelSeq": pl.Series([1], dtype=pl.UInt64),
                }
            )
            for candidate_id in candidate_ids
        }
        with (
            mock.patch.object(
                RUNNER,
                "select_filllots_at_low_state_candidates",
                return_value=selected,
            ) as state_selector,
            mock.patch.object(
                RUNNER,
                "select_filllots_at_low_candidates",
            ) as shaping_selector,
        ):
            result = RUNNER._select_registry_candidates(
                day,
                registry=RUNNER.STATE_REGISTRY,
                candidate_ids=candidate_ids,
            )

        self.assertEqual(tuple(result), candidate_ids)
        state_selector.assert_called_once_with(
            day,
            candidate_ids=candidate_ids,
        )
        shaping_selector.assert_not_called()

    def test_state_registry_has_distinct_cache_identity(self) -> None:
        date = "20250102"
        candidate_id = "negative_state"
        label_spec = RUNNER.LABEL_PROFILES["short-taker-stop108-v1"]
        cached = self._day_frame(
            date,
            candidate_id,
            label_profile_id=label_spec.profile_id,
            label_profile_hash=label_spec.fingerprint,
            pool_id="fixture_pool",
            pool_version="1",
        ).with_columns(
            pl.lit(RUNNER.STATE_SWEEP_CONTRACT_ID).alias("contract_id"),
            pl.lit(RUNNER.FILLLOTS_AT_LOW_STATE_FAMILY_ID).alias("family_id"),
            pl.lit(RUNNER.FILLLOTS_AT_LOW_STATE_FAMILY_VERSION).alias(
                "family_version"
            ),
        )

        RUNNER._validate_cached_day(
            cached,
            date=date,
            candidate_ids=(candidate_id,),
            label_spec=label_spec,
            pool_id="fixture_pool",
            pool_version="1",
            control_sample_size=None,
            source_identities_json="{}",
            registry=RUNNER.STATE_REGISTRY,
        )
        with self.assertRaisesRegex(ValueError, "contract_id"):
            RUNNER._validate_cached_day(
                cached,
                date=date,
                candidate_ids=(candidate_id,),
                label_spec=label_spec,
                pool_id="fixture_pool",
                pool_version="1",
                control_sample_size=None,
                source_identities_json="{}",
            )

    def test_default_output_directory_is_registry_specific(self) -> None:
        with mock.patch.object(sys, "argv", ["run_sweep.py"]):
            event_args = RUNNER.parse_args()
        with mock.patch.object(
            sys,
            "argv",
            ["run_sweep.py", "--candidate-registry", "state"],
        ):
            state_args = RUNNER.parse_args()

        self.assertEqual(event_args.output_dir, RUNNER.DEFAULT_OUTPUT_DIR)
        self.assertEqual(
            state_args.output_dir,
            RUNNER.DEFAULT_STATE_OUTPUT_DIR,
        )
        self.assertNotEqual(event_args.output_dir, state_args.output_dir)

    def test_cached_day_rejects_same_path_source_replacement(self) -> None:
        date = "20250102"
        candidate_id = RUNNER.CANDIDATE_IDS[0]
        label_spec = RUNNER.LABEL_PROFILES["short-taker-stop108-v1"]
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
                roots.pre_market_dir,
                roots.txf_tick_dir,
            ):
                directory.mkdir()
            tick_path = roots.tick_data_dir / f"{date}_StockTick.parquet"
            pre_market_path = (
                roots.pre_market_dir / f"{date}_preMarketData.parquet"
            )
            txf_path = roots.txf_tick_dir / f"{date}_TxfTick.parquet"
            tick_path.write_bytes(b"tick-v1")
            pre_market_path.write_bytes(b"pre-market-v1")
            txf_path.write_bytes(b"txf-v1")

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
            cached = self._day_frame(
                date,
                candidate_id,
                label_profile_id=label_spec.profile_id,
                label_profile_hash=label_spec.fingerprint,
                pool_id="fixture_pool",
                pool_version="1",
                source_identities_json=original_identity,
            )
            RUNNER._validate_cached_day(
                cached,
                date=date,
                candidate_ids=(candidate_id,),
                label_spec=label_spec,
                pool_id="fixture_pool",
                pool_version="1",
                control_sample_size=None,
                source_identities_json=original_identity,
            )

            # Replace one dependency in place under the exact same filename.
            tick_path.write_bytes(b"tick-v2-with-a-different-size")
            replacement_identity = RUNNER._day_source_identities_json(date, roots)
            self.assertNotEqual(replacement_identity, original_identity)
            with self.assertRaisesRegex(
                ValueError,
                RUNNER.SOURCE_IDENTITY_COLUMN,
            ):
                RUNNER._validate_cached_day(
                    cached,
                    date=date,
                    candidate_ids=(candidate_id,),
                    label_spec=label_spec,
                    pool_id="fixture_pool",
                    pool_version="1",
                    control_sample_size=None,
                    source_identities_json=replacement_identity,
                )

    def test_stock_day_equal_delta_uses_same_quote_pool_baseline(self) -> None:
        pool = pl.DataFrame(
            {
                "QuoteCode": ["A", "A", "B", "B", "C"],
                "label_z": [1.0, 3.0, 0.0, 2.0, 100.0],
            }
        )
        condition = pl.DataFrame(
            {
                "QuoteCode": ["A", "B"],
                "label_z": [3.0, 0.0],
            }
        )

        # A contributes 3 - mean(1,3) = +1; B contributes
        # 0 - mean(0,2) = -1.  Quote C cannot affect the matched result.
        self.assertAlmostEqual(
            RUNNER._stock_day_equal_delta_z(
                condition,
                RUNNER._pool_mean_z_by_quote(pool),
            ),
            0.0,
        )
        self.assertIsNone(
            RUNNER._stock_day_equal_delta_z(
                condition.head(0),
                RUNNER._pool_mean_z_by_quote(pool),
            )
        )

    def test_summarize_sweep_preserves_all_inactive_candidates_and_periods(
        self,
    ) -> None:
        label_spec = RUNNER.LABEL_PROFILES["short-taker-stop108-v1"]
        rows = []
        for date in ("20250102", "20260102"):
            for candidate_id in ("always_zero_a", "always_zero_b"):
                rows.append(
                    self._day_frame(
                        date,
                        candidate_id,
                        label_profile_id=label_spec.profile_id,
                        label_profile_hash=label_spec.fingerprint,
                        pool_id="fixture_pool",
                        pool_version="1",
                    ).with_columns(
                        pl.lit(0, dtype=pl.Int64).alias("condition_n"),
                        pl.lit(0, dtype=pl.Int64).alias("condition_n_z"),
                        pl.lit(0, dtype=pl.Int64).alias(
                            "condition_unique_quotes"
                        ),
                        pl.lit(0.0).alias("condition_sum_z"),
                        pl.lit(0.0).alias("condition_sum_raw_bp"),
                        pl.lit(0.0).alias("condition_sum_residual_bp"),
                        pl.lit(0.0).alias("selection_rate"),
                        pl.lit(None, dtype=pl.Float64).alias(
                            "stock_day_equal_delta_z"
                        ),
                        pl.lit(None, dtype=pl.Float64).alias(
                            RUNNER.STOCK_TIME_EQUAL_COLUMN
                        ),
                    )
                )

        summary = RUNNER.summarize_sweep(
            pl.concat(rows, how="vertical")
        ).sort("candidate_id", "period")

        self.assertEqual(len(summary), 6)
        self.assertEqual(
            set(summary["candidate_id"].to_list()),
            {"always_zero_a", "always_zero_b"},
        )
        for candidate_id in ("always_zero_a", "always_zero_b"):
            candidate = summary.filter(
                pl.col("candidate_id") == candidate_id
            )
            self.assertEqual(
                set(candidate["period"].to_list()),
                {"full", "2025", "2026"},
            )
            for column in ("days", "positive_days", "positive_months", "months"):
                self.assertEqual(candidate[column].to_list(), [0, 0, 0])
            nullable = set(candidate.columns) - {
                "candidate_id",
                "period",
                "days",
                "positive_days",
                "positive_months",
                "months",
            }
            for column in nullable:
                self.assertEqual(candidate[column].null_count(), 3)

    def test_summarize_sweep_keeps_inactive_candidate_without_changing_active(
        self,
    ) -> None:
        label_spec = RUNNER.LABEL_PROFILES["short-taker-stop108-v1"]
        active = pl.concat(
            [
                self._day_frame(
                    date,
                    "active",
                    label_profile_id=label_spec.profile_id,
                    label_profile_hash=label_spec.fingerprint,
                    pool_id="fixture_pool",
                    pool_version="1",
                    condition_sum_z=condition_sum_z,
                )
                for date, condition_sum_z in (
                    ("20250102", 2.0),
                    ("20250103", 4.0),
                )
            ],
            how="vertical",
        )
        inactive = active.with_columns(
            pl.lit("inactive").alias("candidate_id"),
            pl.lit(0, dtype=pl.Int64).alias("condition_n"),
            pl.lit(0, dtype=pl.Int64).alias("condition_n_z"),
            pl.lit(0, dtype=pl.Int64).alias("condition_unique_quotes"),
            pl.lit(0.0).alias("condition_sum_z"),
            pl.lit(0.0).alias("condition_sum_raw_bp"),
            pl.lit(0.0).alias("condition_sum_residual_bp"),
            pl.lit(0.0).alias("selection_rate"),
            pl.lit(None, dtype=pl.Float64).alias("stock_day_equal_delta_z"),
            pl.lit(None, dtype=pl.Float64).alias(
                RUNNER.STOCK_TIME_EQUAL_COLUMN
            ),
        )

        baseline = RUNNER.summarize_sweep(active).filter(
            pl.col("period").is_in(["full", "2025"])
        ).sort("period")
        mixed = RUNNER.summarize_sweep(
            pl.concat([active, inactive], how="vertical")
        )
        mixed_active = mixed.filter(
            (pl.col("candidate_id") == "active")
            & pl.col("period").is_in(["full", "2025"])
        ).sort("period")
        self.assertEqual(baseline.to_dicts(), mixed_active.to_dicts())

        inactive_summary = mixed.filter(
            pl.col("candidate_id") == "inactive"
        )
        self.assertEqual(len(inactive_summary), 3)
        self.assertEqual(inactive_summary["days"].to_list(), [0, 0, 0])
        self.assertEqual(inactive_summary["delta_z_pooled"].null_count(), 3)

    def test_stock_day_equal_delta_fails_if_active_quote_escapes_pool(self) -> None:
        pool = pl.DataFrame({"QuoteCode": ["A"], "label_z": [1.0]})
        condition = pl.DataFrame({"QuoteCode": ["B"], "label_z": [2.0]})
        with self.assertRaisesRegex(RuntimeError, "escaped"):
            RUNNER._stock_day_equal_delta_z(
                condition,
                RUNNER._pool_mean_z_by_quote(pool),
            )

    def test_stock_time_equal_delta_weights_buckets_then_stocks(self) -> None:
        pool = pl.DataFrame(
            {
                "QuoteCode": ["A", "A", "A", "A", "B", "B", "C"],
                "TransTime": [
                    datetime(2025, 1, 2, 9, 1),
                    datetime(2025, 1, 2, 9, 2),
                    datetime(2025, 1, 2, 9, 6),
                    datetime(2025, 1, 2, 9, 8),
                    datetime(2025, 1, 2, 9, 1),
                    datetime(2025, 1, 2, 9, 3),
                    datetime(2025, 1, 2, 9, 1),
                ],
                "label_z": [2.0, -2.0, 14.0, 6.0, 0.0, 8.0, 100.0],
            }
        )
        condition = pool[[0, 2, 4]]

        # A contributes an equal-bucket mean of ((2 - 0) + (14 - 10))/2 = 3.
        # B contributes 0 - 4 = -4. C is inactive, so the stock-equal result
        # is (3 - 4)/2 = -0.5.
        self.assertAlmostEqual(
            RUNNER._stock_time_equal_delta_z(
                condition,
                RUNNER._pool_mean_z_by_quote_time(pool),
            ),
            -0.5,
        )
        self.assertIsNone(
            RUNNER._stock_time_equal_delta_z(
                condition.head(0),
                RUNNER._pool_mean_z_by_quote_time(pool),
            )
        )

    def test_evaluate_day_matched_estimands_ignore_control_sampling(self) -> None:
        candidate_id = RUNNER.CANDIDATE_IDS[0]
        label_spec = RUNNER.LABEL_PROFILES["short-taker-stop108-v1"]
        pool = pl.DataFrame(
            {
                "QuoteCode": ["A", "A", "B", "B"],
                "ChannelSeq": pl.Series([1, 2, 1, 2], dtype=pl.UInt64),
                "TransTime": [
                    datetime(2025, 1, 2, 9, 1),
                    datetime(2025, 1, 2, 9, 6),
                    datetime(2025, 1, 2, 9, 1),
                    datetime(2025, 1, 2, 9, 6),
                ],
                "label_raw_bp": [1.0, 3.0, 0.0, 4.0],
                "label_market_component_bp": [0.0] * 4,
                "label_hedge_pnl_bp": [0.0] * 4,
                "label_residual_bp": [1.0, 3.0, 0.0, 4.0],
                "label_z": [1.0, 3.0, 0.0, 4.0],
            }
        )
        condition = pool.filter(
            (pl.col("QuoteCode") == "B") & (pl.col("ChannelSeq") == 2)
        )
        # The sampled shared control deliberately omits active QuoteCode B.
        # Both matched estimands must nevertheless use the complete pool.
        sampled_control = pool.filter(
            (pl.col("QuoteCode") == "A") & (pl.col("ChannelSeq") == 1)
        )
        result = SimpleNamespace(
            eligible_pool_rows=len(pool),
            indexed_condition_rows=1,
            condition_excluded_rows=0,
            condition=condition,
            control=sampled_control,
            selection_rate=1.0 / len(pool),
        )
        pool_spec = SimpleNamespace(pool_id="fixture_pool", pool_version="1")
        day = mock.Mock()

        with (
            mock.patch.object(
                RUNNER,
                "reference_pool_spec",
                return_value=pool_spec,
            ),
            mock.patch.object(
                RUNNER,
                "build_eligible_labeled_pool",
                return_value=pool,
            ),
            mock.patch.object(RUNNER, "EventDaySources", return_value=day),
            mock.patch.object(
                RUNNER,
                "select_filllots_at_low_candidates",
                return_value={
                    candidate_id: condition.select("QuoteCode", "ChannelSeq")
                },
            ),
            mock.patch.object(
                RUNNER,
                "compare_conditions_to_shared_control",
                return_value={candidate_id: result},
            ),
        ):
            evaluated = RUNNER.evaluate_day(
                "20250102",
                roots=SimpleNamespace(),
                label_spec=label_spec,
                candidate_ids=(candidate_id,),
                control_sample_size=1,
                txf_loader=object(),
                source_identities_json="{}",
            )

        row = evaluated.row(0, named=True)
        self.assertEqual(row["control_n"], 1)
        self.assertAlmostEqual(row["stock_day_equal_delta_z"], 2.0)
        self.assertAlmostEqual(row[RUNNER.STOCK_TIME_EQUAL_COLUMN], 0.0)
        day.require.assert_called_once_with("tick_data", "pre_market", "txf_tick")

    def test_cached_day_contract_includes_pool_version(self) -> None:
        candidate_id = RUNNER.CANDIDATE_IDS[0]
        label_spec = RUNNER.LABEL_PROFILES["short-taker-stop108-v1"]
        cached = self._day_frame(
            "20250102",
            candidate_id,
            label_profile_id=label_spec.profile_id,
            label_profile_hash=label_spec.fingerprint,
            pool_id="fixture_pool",
            pool_version="1",
        )

        RUNNER._validate_cached_day(
            cached,
            date="20250102",
            candidate_ids=(candidate_id,),
            label_spec=label_spec,
            pool_id="fixture_pool",
            pool_version="1",
            control_sample_size=None,
            source_identities_json="{}",
        )
        with self.assertRaisesRegex(ValueError, RUNNER.SOURCE_IDENTITY_COLUMN):
            RUNNER._validate_cached_day(
                cached.drop(RUNNER.SOURCE_IDENTITY_COLUMN),
                date="20250102",
                candidate_ids=(candidate_id,),
                label_spec=label_spec,
                pool_id="fixture_pool",
                pool_version="1",
                control_sample_size=None,
                source_identities_json="{}",
            )
        with self.assertRaisesRegex(ValueError, "pool_version"):
            RUNNER._validate_cached_day(
                cached,
                date="20250102",
                candidate_ids=(candidate_id,),
                label_spec=label_spec,
                pool_id="fixture_pool",
                pool_version="2",
                control_sample_size=None,
                source_identities_json="{}",
            )
        wrong_bucket = cached.with_columns(
            pl.lit("10m").alias("stock_time_bucket")
        )
        with self.assertRaisesRegex(ValueError, "stock_time_bucket"):
            RUNNER._validate_cached_day(
                wrong_bucket,
                date="20250102",
                candidate_ids=(candidate_id,),
                label_spec=label_spec,
                pool_id="fixture_pool",
                pool_version="1",
                control_sample_size=None,
                source_identities_json="{}",
            )
        with self.assertRaisesRegex(ValueError, RUNNER.STOCK_TIME_EQUAL_COLUMN):
            RUNNER._validate_cached_day(
                cached.drop(RUNNER.STOCK_TIME_EQUAL_COLUMN),
                date="20250102",
                candidate_ids=(candidate_id,),
                label_spec=label_spec,
                pool_id="fixture_pool",
                pool_version="1",
                control_sample_size=None,
                source_identities_json="{}",
            )

    def test_force_failure_quarantines_stale_day_before_summary(self) -> None:
        candidate_id = RUNNER.CANDIDATE_IDS[0]
        label_spec = RUNNER.LABEL_PROFILES["short-taker-stop108-v1"]
        pool_spec = SimpleNamespace(pool_id="fixture_pool", pool_version="1")
        sessions = ["20250102", "20250103"]
        with tempfile.TemporaryDirectory() as temp_directory:
            output_dir = Path(temp_directory) / "artifacts"
            daily_dir = output_dir / "daily"
            daily_dir.mkdir(parents=True)
            for date in sessions:
                self._day_frame(
                    date,
                    candidate_id,
                    label_profile_id=label_spec.profile_id,
                    label_profile_hash=label_spec.fingerprint,
                    pool_id=pool_spec.pool_id,
                    pool_version=pool_spec.pool_version,
                    condition_sum_z=200.0,
                ).write_parquet(daily_dir / f"{date}.parquet")

            successful = self._day_frame(
                sessions[0],
                candidate_id,
                label_profile_id=label_spec.profile_id,
                label_profile_hash=label_spec.fingerprint,
                pool_id=pool_spec.pool_id,
                pool_version=pool_spec.pool_version,
            )

            def evaluate(date: str, **_: object) -> pl.DataFrame:
                if date == sessions[1]:
                    raise RuntimeError("expected recomputation failure")
                return successful

            args = Namespace(
                start=sessions[0],
                end=sessions[-1],
                step=1,
                output_dir=output_dir,
                label_profile="short-taker-stop108-v1",
                candidates=[candidate_id],
                control_sample=0,
                force=True,
                continue_on_error=True,
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
                    return_value=sessions,
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
                mock.patch.object(RUNNER, "evaluate_day", side_effect=evaluate),
            ):
                daily, _ = RUNNER.run(args)

            self.assertEqual(daily["Date"].to_list(), [sessions[0]])
            self.assertAlmostEqual(daily["condition_sum_z"][0], 2.0)
            self.assertIn(RUNNER.SOURCE_IDENTITY_COLUMN, daily.columns)
            report_daily = pl.read_parquet(output_dir / "daily_stats.parquet")
            self.assertNotIn(RUNNER.SOURCE_IDENTITY_COLUMN, report_daily.columns)
            self.assertFalse((daily_dir / f"{sessions[1]}.parquet").exists())
            quarantined = sorted(
                (output_dir / "quarantine" / "daily").glob("*.parquet")
            )
            self.assertEqual(len(quarantined), 2)
            manifest = json.loads((output_dir / "run_manifest.json").read_text())
            self.assertEqual(
                [failure["Date"] for failure in manifest["failures"]],
                [sessions[1]],
            )
            self.assertEqual(len(manifest["quarantined_daily_outputs"]), 2)
            self.assertEqual(
                manifest["stock_time_bucket"],
                RUNNER.STOCK_TIME_BUCKET,
            )
            self.assertEqual(
                manifest["stock_time_equal_metric"],
                RUNNER.STOCK_TIME_EQUAL_COLUMN,
            )
            self.assertEqual(
                manifest["primary_estimand"],
                RUNNER.PRIMARY_ESTIMAND.value,
            )
            self.assertEqual(
                manifest["source_identity"],
                {
                    "column": RUNNER.SOURCE_IDENTITY_COLUMN,
                    "sources": list(RUNNER.SOURCE_IDENTITY_SOURCES),
                    "fields": ["path", "size", "mtime_ns"],
                },
            )
            self.assertEqual(manifest["source_identities"], {sessions[0]: {}})


if __name__ == "__main__":
    unittest.main()
