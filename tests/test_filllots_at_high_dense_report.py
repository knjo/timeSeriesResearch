from __future__ import annotations

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
    / "report_dense.py"
)
SPEC = importlib.util.spec_from_file_location(
    "filllots_at_high_dense_report_test_target",
    REPORT_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load dense reporter: {REPORT_PATH}")
REPORT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPORT
SPEC.loader.exec_module(REPORT)


SHORT_CANDIDATE = "fh_neg_rejection"
LONG_CANDIDATE = "fh_pos_breakout"
DATES = (
    "20250102",
    "20250203",
    "20250304",
    "20260102",
    "20260203",
    "20260304",
)


def _sparse_summary() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    targets = {
        "short": SHORT_CANDIDATE,
        "long": LONG_CANDIDATE,
    }
    for direction in ("short", "long"):
        for candidate_id in REPORT.CANDIDATE_IDS:
            selected = candidate_id == targets[direction]
            for period in ("full", "2025", "2026"):
                rows.append(
                    {
                        "direction": direction,
                        "candidate_id": candidate_id,
                        "period": period,
                        "active_days": 10,
                        "session_coverage": 1.0,
                        "events_per_active_day": 100.0,
                        "equal_day_delta_z_mean": 0.5 if selected else -0.5,
                        "hac_p_two_sided": 0.001 if selected else 0.9,
                        "bh_q_full": 0.01 if selected else 0.9,
                        "joint_holm_p_full": 0.02 if selected else 1.0,
                        "positive_month_ratio": 0.9 if selected else 0.1,
                    }
                )
    return pl.DataFrame(rows)


def _write_promotion(root: Path) -> Path:
    sparse_path = root / "sparse_period_summary.parquet"
    _sparse_summary().write_parquet(sparse_path)
    study_spec = json.loads(
        REPORT.STUDY_SPEC_PATH.read_text(encoding="utf-8")
    )
    gate, selected = REPORT.SELECTOR.select_promotions(
        pl.read_parquet(sparse_path),
        study_spec,
    )
    promotion = {
        "contract": REPORT.PROMOTION_CONTRACT_ID,
        "study_spec": {
            "path": str(REPORT.STUDY_SPEC_PATH.resolve()),
            "sha256": REPORT.SPARSE._sha256(REPORT.STUDY_SPEC_PATH),
        },
        "period_summary": {
            "path": str(sparse_path.resolve()),
            "sha256": REPORT.SPARSE._sha256(sparse_path),
        },
        "selected": selected,
        "gate_rows": gate.to_dicts(),
    }
    path = root / "promotion.json"
    path.write_text(json.dumps(promotion), encoding="utf-8")
    return path


def _daily(direction: str, candidate_id: str) -> pl.DataFrame:
    label_id, label_hash = REPORT.SPARSE.EXPECTED_DIRECTION_LABELS[direction]
    effects = (
        (1.0, 2.0, 1.0, 2.0, 1.0, 2.0)
        if direction == "short"
        else (1.0, 1.5, 1.0, -0.5, -0.3, -0.4)
    )
    rows: list[dict[str, object]] = []
    for date, effect in zip(DATES, effects, strict=True):
        rows.append(
            {
                "contract_id": REPORT.SPARSE.SWEEP_CONTRACT_ID,
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
                "condition_n": 20,
                "condition_n_z": 20,
                "condition_sum_z": 20.0 * effect,
                "control_n": 100,
                "control_n_z": 100,
                "control_sum_z": 0.0,
                "condition_unique_quotes": 10,
                "selection_rate": 0.20,
                "condition_sum_raw_bp": 200.0 * effect,
                "control_sum_raw_bp": 0.0,
                "condition_sum_residual_bp": 160.0 * effect,
                "control_sum_residual_bp": 0.0,
                "stock_day_equal_delta_z": effect * 0.9,
                "stock_time_equal_delta_z": effect * 0.8,
            }
        )
    return pl.DataFrame(rows)


def _identities() -> dict[str, dict[str, dict[str, object]]]:
    return {
        date: {
            source: {
                "path": f"/{date}/{source}.parquet",
                "size": 100,
                "mtime_ns": 200,
            }
            for source in REPORT.SPARSE.REQUIRED_SOURCE_IDENTITIES
        }
        for date in DATES
    }


def _write_dense(
    directory: Path,
    *,
    direction: str,
    candidate_id: str,
) -> None:
    directory.mkdir()
    daily = _daily(direction, candidate_id)
    daily.write_parquet(directory / "daily_stats.parquet")
    label_id, label_hash = REPORT.SPARSE.EXPECTED_DIRECTION_LABELS[direction]
    manifest = {
        "contract": REPORT.SPARSE.SWEEP_CONTRACT_ID,
        "candidate_registry": REPORT.SPARSE.REGISTRY_ID,
        "family_id": REPORT.FILLLOTS_AT_HIGH_FEATURE_FAMILY_ID,
        "family_version": REPORT.FILLLOTS_AT_HIGH_FEATURE_FAMILY_VERSION,
        "candidate_ids": [candidate_id],
        "candidate_definitions": {
            candidate_id: REPORT.CANDIDATE_DESCRIPTIONS[candidate_id]
        },
        "label_profile_id": label_id,
        "label_profile_hash": label_hash,
        "pool_id": "fixture_pool",
        "pool_version": "1",
        "stock_time_bucket": "5m",
        "primary_estimand": "shared_pool",
        "control_sample_size": None,
        "start": DATES[0],
        "end": "20260825",
        "step": 1,
        "sessions": list(DATES),
        "failures": [],
        "source_identity": {
            "sources": list(REPORT.SPARSE.REQUIRED_SOURCE_IDENTITIES),
            "fields": list(REPORT.SPARSE.SOURCE_IDENTITY_FIELDS),
        },
        "source_identities": _identities(),
    }
    (directory / "run_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


class FillLotsAtHighDenseReportTest(unittest.TestCase):
    def test_validate_dense_daily_accepts_zero_pool_nan_selection_rate(
        self,
    ) -> None:
        zero_pool_date = DATES[0]
        daily = _daily("long", LONG_CANDIDATE).with_columns(
            *[
                pl.when(pl.col("Date") == zero_pool_date)
                .then(pl.lit(value))
                .otherwise(pl.col(column))
                .alias(column)
                for column, value in {
                    "eligible_pool_rows": 0,
                    "condition_n": 0,
                    "condition_n_z": 0,
                    "condition_sum_z": 0.0,
                    "control_n": 0,
                    "control_n_z": 0,
                    "control_sum_z": 0.0,
                    "condition_unique_quotes": 0,
                    "selection_rate": float("nan"),
                    "condition_sum_raw_bp": 0.0,
                    "control_sum_raw_bp": 0.0,
                    "condition_sum_residual_bp": 0.0,
                    "control_sum_residual_bp": 0.0,
                    "stock_day_equal_delta_z": None,
                    "stock_time_equal_delta_z": None,
                }.items()
            ]
        )

        self.assertIsNone(
            REPORT._validate_dense_daily(
                daily,
                direction="long",
                candidate_id=LONG_CANDIDATE,
                sessions=DATES,
            )
        )

    def test_dense_report_validates_promotion_and_applies_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            promotion = _write_promotion(root)
            short = root / "short"
            long = root / "long"
            _write_dense(
                short,
                direction="short",
                candidate_id=SHORT_CANDIDATE,
            )
            _write_dense(
                long,
                direction="long",
                candidate_id=LONG_CANDIDATE,
            )

            summary, monthly, daily, distribution, sensitivity, decisions = (
                REPORT.build_report(
                    promotion_json=promotion,
                    short_dir=short,
                    long_dir=long,
                )
            )

            self.assertEqual(summary.height, 6)
            self.assertEqual(monthly.height, 12)
            self.assertEqual(daily.height, 12)
            self.assertEqual(distribution.height, 6)
            self.assertEqual(sensitivity.height, 10)
            self.assertEqual(len(decisions), 2)
            decision_by_direction = {
                row["direction"]: row for row in decisions
            }
            self.assertTrue(decision_by_direction["short"]["accepted"])
            self.assertFalse(decision_by_direction["long"]["accepted"])
            self.assertIn("equal_day_delta_raw_bp_mean", summary.columns)
            self.assertIn("equal_day_delta_residual_bp_mean", summary.columns)
            self.assertIn("stock_day_equal_delta_z_mean", summary.columns)
            self.assertIn("stock_time_equal_delta_z_mean", summary.columns)
            self.assertIn("dense_acceptance_decision", summary.columns)

            output = root / "report"
            REPORT.write_report(
                promotion_json=promotion,
                short_dir=short,
                long_dir=long,
                output_dir=output,
            )
            for filename in (
                "period_summary.parquet",
                "monthly_stats.parquet",
                "daily_effects.parquet",
                "distribution_stats.parquet",
                "hac_sensitivity.parquet",
                "decision.json",
                "report.json",
            ):
                self.assertTrue((output / filename).is_file(), filename)
            decision_payload = json.loads(
                (output / "decision.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                decision_payload["contract"],
                REPORT.DECISION_CONTRACT_ID,
            )
            self.assertEqual(len(decision_payload["decisions"]), 2)

    def test_promotion_hash_and_dense_candidate_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            promotion = _write_promotion(root)
            payload = json.loads(promotion.read_text(encoding="utf-8"))
            payload["period_summary"]["sha256"] = "0" * 64
            promotion.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(REPORT.DenseReportError, "sha256"):
                REPORT._load_promotion(promotion)

            promotion = _write_promotion(root)
            short = root / "short"
            _write_dense(
                short,
                direction="short",
                candidate_id=LONG_CANDIDATE,
            )
            loaded = REPORT._load_promotion(promotion)
            with self.assertRaisesRegex(
                REPORT.DenseReportError,
                "candidate set is not the promotion",
            ):
                REPORT._load_dense_input(
                    short,
                    direction="short",
                    candidate_id=loaded.selected["short"],
                    study_spec=loaded.study_spec,
                )


if __name__ == "__main__":
    unittest.main()
