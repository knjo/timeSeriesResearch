from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import polars as pl


REPORT_PATH = (
    Path(__file__).resolve().parents[1]
    / "test"
    / "20260826_filllots_at_high_feature_bias"
    / "report_dense_complement.py"
)
SPEC = importlib.util.spec_from_file_location(
    "filllots_at_high_dense_complement_report_test_target",
    REPORT_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load complement reporter: {REPORT_PATH}")
REPORT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPORT
SPEC.loader.exec_module(REPORT)


SHORT_CANDIDATE = "fh_neg_rejection"
LONG_CANDIDATE = "fh_pos_breakout"


def _dense_dates() -> tuple[str, ...]:
    current = date(2025, 2, 11)
    dates: list[str] = []
    while len(dates) < REPORT.EXPECTED_DENSE_SESSIONS:
        if current.weekday() < 5:
            dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return tuple(dates)


DENSE_DATES = _dense_dates()
SPARSE_DATES = DENSE_DATES[:: REPORT.SPARSE_STEP]


def _write_promotion(root: Path) -> Path:
    targets = {"short": SHORT_CANDIDATE, "long": LONG_CANDIDATE}
    rows: list[dict[str, object]] = []
    for direction in ("short", "long"):
        for candidate_id in REPORT.DENSE.CANDIDATE_IDS:
            promoted = candidate_id == targets[direction]
            for period in ("full", "2025", "2026"):
                rows.append(
                    {
                        "direction": direction,
                        "candidate_id": candidate_id,
                        "period": period,
                        "active_days": 76,
                        "session_coverage": 1.0,
                        "events_per_active_day": 100.0,
                        "equal_day_delta_z_mean": 0.5 if promoted else -0.5,
                        "hac_p_two_sided": 0.001 if promoted else 0.9,
                        "bh_q_full": 0.01 if promoted else 0.9,
                        "joint_holm_p_full": 0.02 if promoted else 1.0,
                        "positive_month_ratio": 0.9 if promoted else 0.1,
                    }
                )
    period_path = root / "sparse_period_summary.parquet"
    period = pl.DataFrame(rows)
    period.write_parquet(period_path)
    study_spec = json.loads(
        REPORT.DENSE.STUDY_SPEC_PATH.read_text(encoding="utf-8")
    )
    gate, selected = REPORT.DENSE.SELECTOR.select_promotions(
        period,
        study_spec,
    )
    payload = {
        "contract": REPORT.DENSE.PROMOTION_CONTRACT_ID,
        "study_spec": {
            "path": str(REPORT.DENSE.STUDY_SPEC_PATH.resolve()),
            "sha256": REPORT.DENSE.SPARSE._sha256(
                REPORT.DENSE.STUDY_SPEC_PATH
            ),
        },
        "period_summary": {
            "path": str(period_path.resolve()),
            "sha256": REPORT.DENSE.SPARSE._sha256(period_path),
        },
        "selected": selected,
        "gate_rows": gate.to_dicts(),
    }
    path = root / "promotion.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _identities(
    dates: tuple[str, ...],
) -> dict[str, dict[str, dict[str, object]]]:
    return {
        day: {
            source: {
                "path": f"/{day}/{source}.parquet",
                "size": 100,
                "mtime_ns": 200,
            }
            for source in REPORT.DENSE.SPARSE.REQUIRED_SOURCE_IDENTITIES
        }
        for day in dates
    }


def _daily(
    *,
    direction: str,
    candidate_ids: tuple[str, ...],
    dates: tuple[str, ...],
) -> pl.DataFrame:
    label_id, label_hash = REPORT.DENSE.SPARSE.EXPECTED_DIRECTION_LABELS[
        direction
    ]
    rows: list[dict[str, object]] = []
    for candidate_index, candidate_id in enumerate(candidate_ids):
        for day_index, day in enumerate(dates):
            inactive = (
                day == REPORT.NOTABLE_POSSIBLY_INACTIVE_DATE
                and candidate_id
                in {SHORT_CANDIDATE, LONG_CANDIDATE}
            )
            count = 0 if inactive else 20
            effect = 1.0 + ((day_index + candidate_index) % 3) * 0.20
            rows.append(
                {
                    "contract_id": REPORT.DENSE.SPARSE.SWEEP_CONTRACT_ID,
                    "Date": day,
                    "family_id": (
                        REPORT.DENSE.FILLLOTS_AT_HIGH_FEATURE_FAMILY_ID
                    ),
                    "family_version": (
                        REPORT.DENSE.FILLLOTS_AT_HIGH_FEATURE_FAMILY_VERSION
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
                    "condition_n": count,
                    "condition_n_z": count,
                    "condition_sum_z": count * effect,
                    "control_n": 100,
                    "control_n_z": 100,
                    "control_sum_z": 0.0,
                    "condition_unique_quotes": 10 if count else 0,
                    "selection_rate": count / 100.0,
                    "condition_sum_raw_bp": count * effect * 10.0,
                    "control_sum_raw_bp": 0.0,
                    "condition_sum_residual_bp": count * effect * 8.0,
                    "control_sum_residual_bp": 0.0,
                    "stock_day_equal_delta_z": effect * 0.9 if count else None,
                    "stock_time_equal_delta_z": effect * 0.8 if count else None,
                }
            )
    return pl.DataFrame(rows)


def _write_artifact(
    directory: Path,
    *,
    direction: str,
    candidate_ids: tuple[str, ...],
    dates: tuple[str, ...],
    step: int,
) -> None:
    directory.mkdir()
    _daily(
        direction=direction,
        candidate_ids=candidate_ids,
        dates=dates,
    ).write_parquet(directory / "daily_stats.parquet")
    label_id, label_hash = REPORT.DENSE.SPARSE.EXPECTED_DIRECTION_LABELS[
        direction
    ]
    manifest = {
        "contract": REPORT.DENSE.SPARSE.SWEEP_CONTRACT_ID,
        "candidate_registry": REPORT.DENSE.SPARSE.REGISTRY_ID,
        "family_id": REPORT.DENSE.FILLLOTS_AT_HIGH_FEATURE_FAMILY_ID,
        "family_version": (
            REPORT.DENSE.FILLLOTS_AT_HIGH_FEATURE_FAMILY_VERSION
        ),
        "candidate_ids": list(candidate_ids),
        "candidate_definitions": {
            candidate_id: REPORT.DENSE.CANDIDATE_DESCRIPTIONS[candidate_id]
            for candidate_id in candidate_ids
        },
        "label_profile_id": label_id,
        "label_profile_hash": label_hash,
        "pool_id": "fixture_pool",
        "pool_version": "1",
        "stock_time_bucket": "5m",
        "primary_estimand": "shared_pool",
        "control_sample_size": None,
        "start": DENSE_DATES[0],
        "end": "20260825",
        "step": step,
        "sessions": list(dates),
        "failures": [],
        "source_identity": {
            "sources": list(
                REPORT.DENSE.SPARSE.REQUIRED_SOURCE_IDENTITIES
            ),
            "fields": list(REPORT.DENSE.SPARSE.SOURCE_IDENTITY_FIELDS),
        },
        "source_identities": _identities(dates),
    }
    (directory / "run_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def _write_all(root: Path) -> dict[str, Path]:
    paths = {
        "promotion": _write_promotion(root),
        "dense_short": root / "dense_short",
        "dense_long": root / "dense_long",
        "sparse_short": root / "sparse_short",
        "sparse_long": root / "sparse_long",
    }
    _write_artifact(
        paths["dense_short"],
        direction="short",
        candidate_ids=(SHORT_CANDIDATE,),
        dates=DENSE_DATES,
        step=1,
    )
    _write_artifact(
        paths["dense_long"],
        direction="long",
        candidate_ids=(LONG_CANDIDATE,),
        dates=DENSE_DATES,
        step=1,
    )
    _write_artifact(
        paths["sparse_short"],
        direction="short",
        candidate_ids=REPORT.DENSE.CANDIDATE_IDS,
        dates=SPARSE_DATES,
        step=5,
    )
    _write_artifact(
        paths["sparse_long"],
        direction="long",
        candidate_ids=REPORT.DENSE.CANDIDATE_IDS,
        dates=SPARSE_DATES,
        step=5,
    )
    return paths


class FillLotsAtHighDenseComplementReportTest(unittest.TestCase):
    def test_builds_exact_calendar_complement_and_labels_decision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = _write_all(root)
            result = REPORT.build_report(
                promotion_json=paths["promotion"],
                dense_short_dir=paths["dense_short"],
                dense_long_dir=paths["dense_long"],
                sparse_short_dir=paths["sparse_short"],
                sparse_long_dir=paths["sparse_long"],
            )
            summary, _, daily, distribution, sensitivity, decisions = result
            self.assertEqual(summary.height, 6)
            self.assertEqual(daily.height, 2 * 301)
            self.assertEqual(distribution.height, 6)
            self.assertEqual(sensitivity.height, 10)
            self.assertEqual(len(decisions), 2)
            self.assertTrue(
                set(daily["Date"].unique().to_list()).isdisjoint(SPARSE_DATES)
            )
            inactive = daily.filter(
                pl.col("Date") == REPORT.NOTABLE_POSSIBLY_INACTIVE_DATE
            )
            self.assertEqual(inactive.height, 2)
            self.assertEqual(inactive["active"].to_list(), [False, False])
            self.assertIn("equal_day_delta_raw_bp_mean", summary.columns)
            self.assertIn("stock_time_equal_delta_z_mean", summary.columns)
            self.assertIn("robustness_scope", summary.columns)

            output = root / "output"
            REPORT.write_report(
                promotion_json=paths["promotion"],
                dense_short_dir=paths["dense_short"],
                dense_long_dir=paths["dense_long"],
                sparse_short_dir=paths["sparse_short"],
                sparse_long_dir=paths["sparse_long"],
                output_dir=output,
            )
            payload = json.loads(
                (output / "complement_decision.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(payload["preregistered_dense_acceptance"])
            self.assertFalse(payload["future_out_of_sample"])
            self.assertEqual(
                payload["calendar"]["complement_requested_sessions"],
                301,
            )
            self.assertEqual(
                payload["classification"],
                "descriptive_internal_calendar_complement_robustness",
            )

    def test_rejects_sparse_dense_identity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = _write_all(root)
            date_to_damage = SPARSE_DATES[0]
            for key in ("sparse_short", "sparse_long"):
                manifest_path = paths[key] / "run_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["source_identities"][date_to_damage]["tick_data"][
                    "size"
                ] += 1
                manifest_path.write_text(
                    json.dumps(manifest),
                    encoding="utf-8",
                )
            with self.assertRaisesRegex(
                REPORT.ComplementReportError,
                "source identities differ",
            ):
                REPORT._load_inputs(
                    promotion_json=paths["promotion"],
                    dense_short_dir=paths["dense_short"],
                    dense_long_dir=paths["dense_long"],
                    sparse_short_dir=paths["sparse_short"],
                    sparse_long_dir=paths["sparse_long"],
                )


if __name__ == "__main__":
    unittest.main()
