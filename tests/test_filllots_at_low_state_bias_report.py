from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal


REPORT_PATH = (
    Path(__file__).resolve().parents[1]
    / "test"
    / "20260826_filllots_at_low_state_bias"
    / "report_state_bias.py"
)
SPEC = importlib.util.spec_from_file_location(
    "filllots_at_low_state_bias_report",
    REPORT_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load state bias report: {REPORT_PATH}")
REPORT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPORT
SPEC.loader.exec_module(REPORT)


class FillLotsAtLowStateBiasReportTest(unittest.TestCase):
    dates = ("20250102", "20250103", "20250106", "20250107")

    @staticmethod
    def _daily(
        *,
        direction: str,
        inactive_negative_last_day: bool = False,
    ) -> pl.DataFrame:
        label = (
            REPORT.SHORT_TAKER_STOP108_V1
            if direction == "short"
            else REPORT.LONG_TAKER_TP109_V1
        )
        negative_effects = (0.0, 2.0, 0.0, 2.0)
        positive_effects = (0.0, -1.0, 0.0, -1.0)
        rows: list[dict[str, object]] = []
        for day_index, date in enumerate(
            FillLotsAtLowStateBiasReportTest.dates
        ):
            control_mean = (
                100.0
                if inactive_negative_last_day
                and day_index == len(FillLotsAtLowStateBiasReportTest.dates) - 1
                else 0.0
            )
            negative_n = (
                0
                if inactive_negative_last_day
                and day_index == len(FillLotsAtLowStateBiasReportTest.dates) - 1
                else 40
            )
            for candidate_id, count, effect in (
                ("negative_state", negative_n, negative_effects[day_index]),
                ("positive_state", 59, positive_effects[day_index]),
            ):
                condition_mean = control_mean + effect
                rows.append(
                    {
                        "contract_id": REPORT.STATE_SWEEP_CONTRACT_ID,
                        "Date": date,
                        "family_id": REPORT.FILLLOTS_AT_LOW_STATE_FAMILY_ID,
                        "family_version": (
                            REPORT.FILLLOTS_AT_LOW_STATE_FAMILY_VERSION
                        ),
                        "candidate_id": candidate_id,
                        "label_profile_id": label.profile_id,
                        "label_profile_hash": label.fingerprint,
                        "pool_id": "fixture_pool",
                        "pool_version": "1",
                        "control_mode": "full",
                        "control_sample_size": 0,
                        "primary_estimand": "shared_pool",
                        "stock_time_bucket": "5m",
                        "eligible_pool_rows": 100,
                        "indexed_condition_rows": count,
                        "condition_excluded_rows": 0,
                        "condition_unique_quotes": 2 if count else 0,
                        "stock_day_equal_delta_z": effect if count else None,
                        "stock_time_equal_delta_z": effect if count else None,
                        "selection_rate": count / 100.0,
                        "condition_n": count,
                        "condition_n_z": count,
                        "condition_sum_raw_bp": count * condition_mean * 10.0,
                        "condition_sum_market_component_bp": 0.0,
                        "condition_sum_hedge_pnl_bp": 0.0,
                        "condition_sum_residual_bp": count * condition_mean * 10.0,
                        "condition_sum_z": count * condition_mean,
                        "condition_sum_z2": (
                            count * (condition_mean * condition_mean + 1.0)
                        ),
                        "control_n": 100,
                        "control_n_z": 100,
                        "control_sum_raw_bp": 100 * control_mean * 10.0,
                        "control_sum_market_component_bp": 0.0,
                        "control_sum_hedge_pnl_bp": 0.0,
                        "control_sum_residual_bp": 100 * control_mean * 10.0,
                        "control_sum_z": 100 * control_mean,
                        "control_sum_z2": 100 * (control_mean * control_mean + 1.0),
                    }
                )
        return pl.DataFrame(rows)

    @staticmethod
    def _write_input(
        directory: Path,
        daily: pl.DataFrame,
    ) -> None:
        directory.mkdir()
        daily.write_parquet(directory / "daily_stats.parquet")
        sessions = sorted(daily["Date"].unique().to_list())
        identities = {
            date: {
                source: {
                    "path": f"/{date}/{source}",
                    "size": 1,
                    "mtime_ns": 1,
                }
                for source in ("tick_data", "pre_market", "txf_tick")
            }
            for date in sessions
        }
        manifest = {
            "contract": REPORT.STATE_SWEEP_CONTRACT_ID,
            "candidate_registry": REPORT.STATE_REGISTRY_ID,
            "family_id": REPORT.FILLLOTS_AT_LOW_STATE_FAMILY_ID,
            "family_version": REPORT.FILLLOTS_AT_LOW_STATE_FAMILY_VERSION,
            "candidate_ids": list(REPORT.STATE_CANDIDATE_IDS),
            "label_profile_id": daily["label_profile_id"][0],
            "label_profile_hash": daily["label_profile_hash"][0],
            "pool_id": "fixture_pool",
            "pool_version": "1",
            "primary_estimand": "shared_pool",
            "sessions": sessions,
            "failures": [],
            "source_identities": identities,
        }
        (directory / "run_manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

    def _input_pair(
        self,
        root: Path,
        *,
        inactive_negative_last_day: bool = False,
    ) -> tuple[Path, Path]:
        short = root / "short"
        long = root / "long"
        self._write_input(
            short,
            self._daily(
                direction="short",
                inactive_negative_last_day=inactive_negative_last_day,
            ),
        )
        self._write_input(
            long,
            self._daily(
                direction="long",
                inactive_negative_last_day=inactive_negative_last_day,
            ),
        )
        return short, long

    def test_report_is_row_order_invariant_and_keeps_empty_year(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            short, long = self._input_pair(Path(temp_directory))
            expected = REPORT.build_report((short, long), hac_lag=1)
            for directory in (short, long):
                path = directory / "daily_stats.parquet"
                shuffled = pl.read_parquet(path).sample(
                    fraction=1.0,
                    shuffle=True,
                    seed=17,
                )
                shuffled.write_parquet(path)
            actual = REPORT.build_report((short, long), hac_lag=1)

            for expected_frame, actual_frame in zip(expected, actual):
                assert_frame_equal(expected_frame, actual_frame)
            summary = actual[0]
            self.assertEqual(summary.shape[0], 12)
            empty_2026 = summary.filter(pl.col("period") == "2026")
            self.assertEqual(empty_2026.shape[0], 4)
            self.assertEqual(empty_2026["sessions"].to_list(), [0, 0, 0, 0])
            self.assertEqual(empty_2026["event_rows"].to_list(), [0, 0, 0, 0])
            self.assertEqual(
                summary.filter(pl.col("period") == "full").shape[0],
                4,
            )

    def test_inactive_day_does_not_add_control_only_rows_to_pooled_delta(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            short, long = self._input_pair(
                Path(temp_directory),
                inactive_negative_last_day=True,
            )
            summary, _, _ = REPORT.build_report((short, long), hac_lag=1)
            negative = summary.filter(
                (pl.col("direction") == "short")
                & (pl.col("candidate_id") == "negative_state")
                & (pl.col("period") == "full")
            ).row(0, named=True)

            # The final day has control mean=100 but no negative-state rows.
            # Its control must not dilute the three active-day pooled estimand.
            self.assertAlmostEqual(negative["delta_z_pooled"], 2.0 / 3.0)
            self.assertEqual(negative["active_days"], 3)

    def test_partition_rejects_missing_candidate_day(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            short, long = self._input_pair(Path(temp_directory))
            path = short / "daily_stats.parquet"
            damaged = pl.read_parquet(path).filter(
                ~(
                    (pl.col("Date") == self.dates[-1])
                    & (pl.col("candidate_id") == "positive_state")
                )
            )
            damaged.write_parquet(path)
            with self.assertRaisesRegex(
                REPORT.StateBiasReportError,
                "every candidate on every session",
            ):
                REPORT.build_report((short, long), hac_lag=1)


if __name__ == "__main__":
    unittest.main()
