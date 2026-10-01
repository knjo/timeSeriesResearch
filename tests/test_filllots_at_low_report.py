from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl


REPORT_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "20260825_filllots_at_low_framework"
    / "report_sweep.py"
)
SPEC = importlib.util.spec_from_file_location("filllots_at_low_report", REPORT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load report module: {REPORT_PATH}")
REPORT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPORT
SPEC.loader.exec_module(REPORT)


def _daily_fixture() -> pl.DataFrame:
    dates_and_delta = (
        ("20250102", 1.0),
        ("20250203", 2.0),
        ("20250204", 1.0),
        ("20260102", 2.0),
        ("20260103", 1.0),
        ("20260104", 2.0),
        ("20260202", None),
    )
    rows: list[dict[str, object]] = []
    for candidate_id, offset in (("candidate_a", 0.0), ("candidate_b", -1.0)):
        for date, base_delta in dates_and_delta:
            active = base_delta is not None
            delta = base_delta + offset if active else 0.0
            condition_count = 2 if active else 0
            rows.append(
                {
                    "contract_id": "fixture_sweep/v1",
                    "Date": date,
                    "family_id": "fixture",
                    "family_version": "1",
                    "candidate_id": candidate_id,
                    "label_profile_id": "short",
                    "label_profile_hash": "hash",
                    "pool_id": "pool",
                    "pool_version": "1",
                    "control_mode": "full",
                    "control_sample_size": 0,
                    "eligible_pool_rows": 100,
                    "condition_n": condition_count,
                    "condition_n_z": condition_count,
                    "condition_sum_z": 2.0 * delta,
                    "control_n": 4,
                    "control_n_z": 4,
                    "control_sum_z": 0.0,
                }
            )
    return pl.DataFrame(rows)


def _daily_fixture_with_stock_equal() -> pl.DataFrame:
    return _daily_fixture().with_columns(
        pl.when(pl.col("condition_n_z") > 0)
        .then(pl.col("condition_sum_z") / pl.col("condition_n_z"))
        .otherwise(None)
        .alias("stock_day_equal_delta_z"),
    )


def _daily_fixture_with_stock_time_equal() -> pl.DataFrame:
    return _daily_fixture_with_stock_equal().with_columns(
        pl.lit("5m").alias("stock_time_bucket"),
        pl.when(pl.col("condition_n_z") > 0)
        .then(
            pl.col("condition_sum_z") / pl.col("condition_n_z")
            + pl.when(pl.col("candidate_id") == "candidate_a")
            .then(-1.0)
            .otherwise(1.0)
        )
        .otherwise(None)
        .alias("stock_time_equal_delta_z"),
    )


def _row(
    frame: pl.DataFrame, candidate_id: str, key: str, value: str
) -> dict[str, object]:
    selected = frame.filter(
        (pl.col("candidate_id") == candidate_id) & (pl.col(key) == value)
    )
    if len(selected) != 1:
        raise AssertionError(f"expected one row, found {len(selected)}")
    return selected.row(0, named=True)


class FillLotsAtLowReportTest(unittest.TestCase):
    def test_hand_checked_period_hac_month_and_bh_statistics(self) -> None:
        summary, monthly = REPORT.build_report(_daily_fixture(), hac_lag=5)
        full_a = _row(summary, "candidate_a", "period", "full")
        full_b = _row(summary, "candidate_b", "period", "full")

        self.assertEqual(full_a["sessions"], 7)
        self.assertEqual(full_a["reported_days"], 7)
        self.assertEqual(full_a["active_days"], 6)
        self.assertAlmostEqual(full_a["session_coverage"], 6.0 / 7.0)
        self.assertEqual(full_a["event_rows"], 12)
        self.assertAlmostEqual(full_a["delta_z_pooled"], 1.5)
        self.assertAlmostEqual(full_a["equal_day_delta_z_mean"], 1.5)
        self.assertAlmostEqual(full_a["equal_day_t"], 3.0 * math.sqrt(5.0))

        # For [1, 2, 1, 2, 1, 2], Bartlett HAC(5) long-run variance
        # is 1/24.  The mean standard error is 1/12, so the t-stat is 18.
        self.assertAlmostEqual(full_a["hac_long_run_variance"], 1.0 / 24.0)
        self.assertAlmostEqual(full_a["hac_se"], 1.0 / 12.0)
        self.assertAlmostEqual(full_a["hac_t"], 18.0)
        self.assertAlmostEqual(
            full_a["raw_p_two_sided"], math.erfc(18.0 / math.sqrt(2.0))
        )

        # Active monthly means are 1, 3/2, and 5/3.  Their equal-month
        # mean is 25/18 and their one-sample t-stat is 25/sqrt(13).
        self.assertEqual(full_a["calendar_months"], 4)
        self.assertEqual(full_a["active_months"], 3)
        self.assertAlmostEqual(full_a["monthly_delta_z_mean"], 25.0 / 18.0)
        self.assertAlmostEqual(full_a["monthly_t"], 25.0 / math.sqrt(13.0))
        self.assertEqual(full_a["positive_months"], 3)
        self.assertAlmostEqual(full_a["positive_month_ratio"], 1.0)
        self.assertFalse(full_a["stock_day_equal_available"])
        self.assertIsNone(full_a["stock_day_equal_days"])
        self.assertIsNone(full_a["stock_day_equal_delta_z_mean"])
        self.assertIsNone(full_a["stock_day_equal_t"])
        self.assertIsNone(full_a["stock_day_equal_hac_t"])
        self.assertIsNone(full_a["stock_day_equal_bh_q_full"])
        self.assertFalse(full_a["stock_time_equal_available"])
        self.assertIsNone(full_a["stock_time_equal_days"])
        self.assertIsNone(full_a["stock_time_equal_delta_z_mean"])
        self.assertIsNone(full_a["stock_time_equal_t"])
        self.assertIsNone(full_a["stock_time_equal_hac_t"])
        self.assertIsNone(full_a["stock_time_equal_bh_q_full"])
        self.assertIsNone(full_a["stock_time_bucket"])

        # Candidate B has the same residual sequence shifted down by one,
        # hence the same HAC standard error and a HAC t-stat of six.
        self.assertAlmostEqual(full_b["delta_z_pooled"], 0.5)
        self.assertAlmostEqual(full_b["hac_t"], 6.0)
        p_a = math.erfc(18.0 / math.sqrt(2.0))
        p_b = math.erfc(6.0 / math.sqrt(2.0))
        self.assertAlmostEqual(full_a["bh_q_full"], 2.0 * p_a)
        self.assertAlmostEqual(full_b["bh_q_full"], p_b)
        year_a = _row(summary, "candidate_a", "period", "2025")
        self.assertIsNone(year_a["bh_q_full"])

        inactive_month = _row(monthly, "candidate_a", "month", "202602")
        self.assertEqual(inactive_month["sessions"], 1)
        self.assertEqual(inactive_month["active_days"], 0)
        self.assertAlmostEqual(inactive_month["session_coverage"], 0.0)
        self.assertIsNone(inactive_month["equal_day_delta_z_mean"])
        self.assertIsNone(inactive_month["positive_month"])

    def test_optional_stock_day_equal_statistics(self) -> None:
        summary, _ = REPORT.build_report(
            _daily_fixture_with_stock_equal(), hac_lag=5
        )
        full_a = _row(summary, "candidate_a", "period", "full")

        self.assertTrue(full_a["stock_day_equal_available"])
        self.assertEqual(full_a["stock_day_equal_days"], 6)
        self.assertAlmostEqual(full_a["stock_day_equal_delta_z_mean"], 1.5)
        self.assertAlmostEqual(
            full_a["stock_day_equal_t"], 3.0 * math.sqrt(5.0)
        )
        self.assertAlmostEqual(
            full_a["stock_day_equal_p_two_sided"],
            REPORT.student_t_two_sided_p(3.0 * math.sqrt(5.0), 5),
        )
        self.assertAlmostEqual(full_a["stock_day_equal_hac_se"], 1.0 / 12.0)
        self.assertAlmostEqual(full_a["stock_day_equal_hac_t"], 18.0)
        self.assertAlmostEqual(
            full_a["stock_day_equal_hac_p_two_sided"],
            math.erfc(18.0 / math.sqrt(2.0)),
        )
        full_b = _row(summary, "candidate_b", "period", "full")
        p_a = math.erfc(18.0 / math.sqrt(2.0))
        p_b = math.erfc(6.0 / math.sqrt(2.0))
        self.assertAlmostEqual(full_a["stock_day_equal_bh_q_full"], 2.0 * p_a)
        self.assertAlmostEqual(full_b["stock_day_equal_bh_q_full"], p_b)
        year_a = _row(summary, "candidate_a", "period", "2025")
        self.assertIsNone(year_a["stock_day_equal_bh_q_full"])

    def test_optional_stock_time_statistics_use_independent_bh_family(self) -> None:
        summary, _ = REPORT.build_report(
            _daily_fixture_with_stock_time_equal(),
            hac_lag=5,
        )
        full_a = _row(summary, "candidate_a", "period", "full")
        full_b = _row(summary, "candidate_b", "period", "full")

        self.assertEqual(full_a["stock_time_bucket"], "5m")
        self.assertTrue(full_a["stock_time_equal_available"])
        self.assertEqual(full_a["stock_time_equal_days"], 6)
        self.assertAlmostEqual(full_a["stock_time_equal_delta_z_mean"], 0.5)
        self.assertAlmostEqual(full_a["stock_time_equal_hac_t"], 6.0)
        self.assertAlmostEqual(full_b["stock_time_equal_delta_z_mean"], 1.5)
        self.assertAlmostEqual(full_b["stock_time_equal_hac_t"], 18.0)
        self.assertAlmostEqual(
            full_a["stock_time_equal_p_two_sided"],
            REPORT.student_t_two_sided_p(
                full_a["stock_time_equal_t"],
                full_a["stock_time_equal_days"] - 1,
            ),
        )
        p_six = math.erfc(6.0 / math.sqrt(2.0))
        p_eighteen = math.erfc(18.0 / math.sqrt(2.0))
        self.assertAlmostEqual(full_a["stock_time_equal_bh_q_full"], p_six)
        self.assertAlmostEqual(
            full_b["stock_time_equal_bh_q_full"],
            2.0 * p_eighteen,
        )
        # The stock-day family has the opposite candidate ranking. Its q-values
        # must not be pooled with the stock-time p-values.
        self.assertAlmostEqual(full_a["stock_day_equal_bh_q_full"], 2.0 * p_eighteen)
        self.assertAlmostEqual(full_b["stock_day_equal_bh_q_full"], p_six)
        year_a = _row(summary, "candidate_a", "period", "2025")
        self.assertIsNone(year_a["stock_time_equal_bh_q_full"])

    def test_legacy_quote_equal_pair_is_not_treated_as_matched(self) -> None:
        legacy = _daily_fixture().with_columns(
            pl.lit(1.0).alias("condition_quote_equal_mean_z"),
            pl.lit(0.0).alias("control_quote_equal_mean_z"),
        )
        summary, _ = REPORT.build_report(legacy)
        full_a = _row(summary, "candidate_a", "period", "full")
        self.assertFalse(full_a["stock_day_equal_available"])
        self.assertIsNone(full_a["stock_day_equal_delta_z_mean"])
        self.assertIsNone(full_a["stock_day_equal_hac_t"])

    def test_student_t_probability_and_contract_failure(self) -> None:
        self.assertAlmostEqual(REPORT.student_t_two_sided_p(1.0, 1), 0.5)
        duplicate = pl.concat([_daily_fixture(), _daily_fixture().head(1)])
        with self.assertRaisesRegex(REPORT.SweepReportError, "duplicate"):
            REPORT.build_report(duplicate)

        fractional_count = _daily_fixture().with_row_index("_row").with_columns(
            pl.when(pl.col("_row") == 0)
            .then(pl.lit(1.5))
            .otherwise(pl.col("condition_n"))
            .alias("condition_n")
        ).drop("_row")
        with self.assertRaisesRegex(
            REPORT.SweepReportError,
            "condition_n must contain nonnegative integers",
        ):
            REPORT.build_report(fractional_count)

    def test_rejects_mixed_estimand_metadata(self) -> None:
        replacements = {
            "contract_id": "other_sweep/v1",
            "control_mode": "sampled",
            "control_sample_size": 10,
        }
        for column, replacement in replacements.items():
            with self.subTest(column=column):
                mixed = _daily_fixture().with_row_index("_row").with_columns(
                    pl.when(pl.col("_row") == 0)
                    .then(pl.lit(replacement))
                    .otherwise(pl.col(column))
                    .alias(column)
                ).drop("_row")
                with self.assertRaisesRegex(REPORT.SweepReportError, column):
                    REPORT.build_report(mixed)

        stock_time = _daily_fixture_with_stock_time_equal()
        mixed_bucket = stock_time.with_row_index("_row").with_columns(
            pl.when(pl.col("_row") == 0)
            .then(pl.lit("10m"))
            .otherwise(pl.col("stock_time_bucket"))
            .alias("stock_time_bucket")
        ).drop("_row")
        with self.assertRaisesRegex(REPORT.SweepReportError, "stock_time_bucket"):
            REPORT.build_report(mixed_bucket)

    def test_stock_time_metric_requires_bucket_identity(self) -> None:
        metric_without_bucket = _daily_fixture_with_stock_time_equal().drop(
            "stock_time_bucket"
        )
        with self.assertRaisesRegex(REPORT.SweepReportError, "stock_time_bucket"):
            REPORT.build_report(metric_without_bucket)

        bucket_without_metric = _daily_fixture().with_columns(
            pl.lit("5m").alias("stock_time_bucket")
        )
        with self.assertRaisesRegex(
            REPORT.SweepReportError,
            "stock_time_equal_delta_z",
        ):
            REPORT.build_report(bucket_without_metric)

    def test_cli_concatenates_multiple_artifact_directories(self) -> None:
        daily = _daily_fixture_with_stock_equal()
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            artifact_2025 = root / "dense2025_v2"
            artifact_2026 = root / "dense2026_v2"
            output = root / "combined_report"
            artifact_2025.mkdir()
            artifact_2026.mkdir()
            daily.filter(pl.col("Date").str.starts_with("2025")).write_parquet(
                artifact_2025 / "daily_stats.parquet"
            )
            daily.filter(pl.col("Date").str.starts_with("2026")).write_parquet(
                artifact_2026 / "daily_stats.parquet"
            )

            single_result = REPORT.main([str(artifact_2025), "--quiet"])
            self.assertEqual(single_result, 0)
            self.assertTrue(
                (artifact_2025 / "report" / "period_summary.parquet").exists()
            )

            result = REPORT.main(
                [
                    str(artifact_2025),
                    str(artifact_2026),
                    "--output-dir",
                    str(output),
                    "--quiet",
                ]
            )

            self.assertEqual(result, 0)
            summary = pl.read_parquet(output / "period_summary.parquet")
            full_a = _row(summary, "candidate_a", "period", "full")
            self.assertEqual(full_a["sessions"], 7)
            self.assertEqual(full_a["stock_day_equal_days"], 6)
            manifest = json.loads((output / "report.json").read_text())
            self.assertIsNone(manifest["source"])
            self.assertEqual(len(manifest["sources"]), 2)

    def test_cli_requires_output_directory_for_multiple_inputs(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                REPORT.parse_args(["dense2025_v2", "dense2026_v2"])
        single = REPORT.parse_args(["dense2025_v2"])
        self.assertIsNone(single.output_dir)


if __name__ == "__main__":
    unittest.main()
