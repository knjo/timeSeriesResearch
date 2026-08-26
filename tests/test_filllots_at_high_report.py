from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl


REPORT_PATH = (
    Path(__file__).resolve().parents[1]
    / "test"
    / "20260826_filllots_at_high_feature_bias"
    / "report_sweep.py"
)
SPEC = importlib.util.spec_from_file_location(
    "filllots_at_high_feature_report_test_target",
    REPORT_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load FillLots-at-high report: {REPORT_PATH}")
REPORT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPORT
SPEC.loader.exec_module(REPORT)


def _source_manifest() -> dict[str, object]:
    date = "20250102"
    return {
        "source_identity": {
            "sources": list(REPORT.REQUIRED_SOURCE_IDENTITIES),
            "fields": list(REPORT.SOURCE_IDENTITY_FIELDS),
        },
        "source_identities": {
            date: {
                source: {
                    "path": f"/{date}/{source}.parquet",
                    "size": 100,
                    "mtime_ns": 200,
                }
                for source in REPORT.REQUIRED_SOURCE_IDENTITIES
            }
        },
    }


def _daily_fixture(direction: str) -> pl.DataFrame:
    label_id, label_hash = REPORT.EXPECTED_DIRECTION_LABELS[direction]
    dates = ("20250102", "20250103", "20260102", "20260103")
    rows: list[dict[str, object]] = []
    for candidate_index, candidate_id in enumerate(REPORT.CANDIDATE_IDS, start=1):
        for day_index, date in enumerate(dates):
            effect = candidate_index / 20.0 + (day_index % 2) * 0.25
            rows.append(
                {
                    "contract_id": REPORT.SWEEP_CONTRACT_ID,
                    "Date": date,
                    "family_id": REPORT.FILLLOTS_AT_HIGH_FEATURE_FAMILY_ID,
                    "family_version": (
                        REPORT.FILLLOTS_AT_HIGH_FEATURE_FAMILY_VERSION
                    ),
                    "candidate_id": candidate_id,
                    "label_profile_id": label_id,
                    "label_profile_hash": label_hash,
                    "pool_id": "fixture_pool",
                    "pool_version": "1",
                    "control_mode": "full",
                    "control_sample_size": 0,
                    "primary_estimand": "shared_pool",
                    "stock_time_bucket": "5m",
                    "eligible_pool_rows": 100,
                    "condition_n": 2,
                    "condition_n_z": 2,
                    "condition_sum_z": 2.0 * effect,
                    "control_n": 100,
                    "control_n_z": 100,
                    "control_sum_z": 0.0,
                    "condition_unique_quotes": 2,
                    "selection_rate": 0.02,
                    "condition_sum_raw_bp": 20.0 * effect,
                    "control_sum_raw_bp": 0.0,
                    "condition_sum_residual_bp": 16.0 * effect,
                    "control_sum_residual_bp": 0.0,
                    "stock_day_equal_delta_z": effect,
                    "stock_time_equal_delta_z": effect,
                }
            )
    return pl.DataFrame(rows)


def _write_artifact(directory: Path, direction: str) -> None:
    directory.mkdir()
    daily = _daily_fixture(direction)
    daily.write_parquet(directory / "daily_stats.parquet")
    sessions = sorted(daily["Date"].unique().to_list())
    identity = _source_manifest()
    identity["source_identities"] = {
        date: {
            source: {
                "path": f"/{date}/{source}.parquet",
                "size": 100,
                "mtime_ns": 200,
            }
            for source in REPORT.REQUIRED_SOURCE_IDENTITIES
        }
        for date in sessions
    }
    label_id, label_hash = REPORT.EXPECTED_DIRECTION_LABELS[direction]
    manifest = {
        "contract": REPORT.SWEEP_CONTRACT_ID,
        "candidate_registry": REPORT.REGISTRY_ID,
        "family_id": REPORT.FILLLOTS_AT_HIGH_FEATURE_FAMILY_ID,
        "family_version": REPORT.FILLLOTS_AT_HIGH_FEATURE_FAMILY_VERSION,
        "candidate_ids": list(REPORT.CANDIDATE_IDS),
        "candidate_definitions": REPORT.CANDIDATE_DESCRIPTIONS,
        "label_profile_id": label_id,
        "label_profile_hash": label_hash,
        "pool_id": "fixture_pool",
        "pool_version": "1",
        "stock_time_bucket": "5m",
        "primary_estimand": "shared_pool",
        "control_sample_size": None,
        "start": sessions[0],
        "end": sessions[-1],
        "step": 1,
        "sessions": sessions,
        "failures": [],
        "source_identity": identity["source_identity"],
        "source_identities": identity["source_identities"],
    }
    (directory / "run_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


class FillLotsAtHighReportTest(unittest.TestCase):
    def test_holm_uses_fixed_family_and_preserves_null(self) -> None:
        actual = REPORT.holm_adjust(
            {"a": 0.01, "b": 0.04, "c": 0.03, "missing": None},
            family_size=4,
        )
        self.assertAlmostEqual(actual["a"], 0.04)
        self.assertAlmostEqual(actual["b"], 0.09)
        self.assertAlmostEqual(actual["c"], 0.09)
        self.assertIsNone(actual["missing"])

        rows = []
        index = 0
        for direction in ("short", "long"):
            for candidate_id in REPORT.CANDIDATE_IDS:
                index += 1
                for period in ("full", "2025", "2026"):
                    rows.append(
                        {
                            "direction": direction,
                            "candidate_id": candidate_id,
                            "period": period,
                            "hac_p_two_sided": index / 1000.0,
                        }
                    )
        adjusted = REPORT._add_joint_holm(pl.DataFrame(rows))
        self.assertEqual(
            adjusted.filter(pl.col("period") == "full").height,
            24,
        )
        self.assertEqual(
            adjusted.filter(pl.col("period") != "full")[
                "joint_holm_p_full"
            ].null_count(),
            48,
        )

    def test_source_identity_registry_must_be_exact_and_complete(self) -> None:
        valid = _source_manifest()
        REPORT._validate_source_identities(
            valid,
            sessions=("20250102",),
            context="fixture",
        )

        for sources in (
            ["tick_data", "pre_market", "txf_tick"],
            [*REPORT.REQUIRED_SOURCE_IDENTITIES, "future_source"],
        ):
            with self.subTest(sources=sources):
                damaged = copy.deepcopy(valid)
                damaged["source_identity"]["sources"] = sources
                with self.assertRaisesRegex(
                    REPORT.HighFeatureReportError,
                    "include exactly",
                ):
                    REPORT._validate_source_identities(
                        damaged,
                        sessions=("20250102",),
                        context="fixture",
                    )

        damaged_day = copy.deepcopy(valid)
        del damaged_day["source_identities"]["20250102"]["tick_feature"]
        with self.assertRaisesRegex(
            REPORT.HighFeatureReportError,
            "identities are incomplete",
        ):
            REPORT._validate_source_identities(
                damaged_day,
                sessions=("20250102",),
                context="fixture",
            )

    def test_joint_build_has_all_tables_and_bp_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            short = root / "short"
            long = root / "long"
            _write_artifact(short, "short")
            _write_artifact(long, "long")

            summary, monthly, daily, distribution, sensitivity = (
                REPORT.build_report(short_dir=short, long_dir=long)
            )

            self.assertEqual(summary.height, 72)
            self.assertEqual(monthly.height, 48)
            self.assertEqual(daily.height, 96)
            self.assertEqual(distribution.height, 72)
            self.assertEqual(sensitivity.height, 120)
            for column in (
                "mean_events_per_active_day",
                "mean_quotes_per_active_day",
                "selection_rate_mean",
                "equal_day_delta_raw_bp_mean",
                "raw_bp_hac_t",
                "raw_bp_hac_p_two_sided",
                "equal_day_delta_residual_bp_mean",
                "residual_bp_hac_t",
                "residual_bp_hac_p_two_sided",
            ):
                self.assertIn(column, summary.columns)
            full = summary.filter(pl.col("period") == "full")
            self.assertEqual(full.height, 24)
            self.assertEqual(full["joint_holm_p_full"].null_count(), 0)
            non_full = summary.filter(pl.col("period") != "full")
            self.assertEqual(
                non_full["joint_holm_p_full"].null_count(),
                non_full.height,
            )


if __name__ == "__main__":
    unittest.main()
