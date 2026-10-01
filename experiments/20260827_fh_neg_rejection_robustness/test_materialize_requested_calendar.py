"""Focused contracts for the external factorial requested-session calendar."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import polars as pl

import materialize_requested_calendar as study


def _manifest(
    *,
    start: str,
    end: str,
    sessions: list[str],
    failures: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    return {
        "contract": study.FACTORIAL_CONTRACT,
        "family_id": study.FACTORIAL_FAMILY_ID,
        "family_version": study.FACTORIAL_FAMILY_VERSION,
        "candidate_ids": list(study.CELL_IDS),
        "start": start,
        "end": end,
        "step": 1,
        "sessions": sessions,
        "failures": failures or [],
    }


def _source_path(project_root: Path, source: str, date: str) -> Path:
    directory, suffix = study.REQUIRED_SOURCES[source]
    return project_root / "data" / directory / f"{date}{suffix}"


def _add_sources(
    project_root: Path,
    date: str,
    *,
    omit: tuple[str, ...] = (),
) -> None:
    for source in study.REQUIRED_SOURCES:
        if source in omit:
            continue
        path = _source_path(project_root, source, date)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


def _daily_rows(
    date: str,
    *,
    eligible_pool_rows: int = 100,
    control_n_z: int = 80,
    condition_n_z: int = 4,
    candidate_ids: tuple[str, ...] = study.CELL_IDS,
) -> list[dict[str, object]]:
    return [
        {
            "Date": date,
            "candidate_id": candidate_id,
            "eligible_pool_rows": eligible_pool_rows,
            "condition_n_z": condition_n_z,
            "control_n_z": control_n_z,
        }
        for candidate_id in candidate_ids
    ]


def _add_membership(run_dir: Path, date: str) -> None:
    daily = run_dir / "membership_daily" / f"{date}.parquet"
    lineage = run_dir / "membership_lineage" / f"{date}.json"
    daily.parent.mkdir(parents=True, exist_ok=True)
    lineage.parent.mkdir(parents=True, exist_ok=True)
    daily.write_bytes(b"membership-present")
    lineage.write_text("{}\n", encoding="utf-8")


def _write_run(
    run_dir: Path,
    manifest: dict[str, object],
    daily_rows: list[dict[str, object]],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    pl.DataFrame(daily_rows).write_parquet(run_dir / "daily_stats.parquet")


class RequestedCalendarContractTest(unittest.TestCase):
    def test_full_run_can_be_sliced_into_one_evidence_class(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run"
            dates = ["20260825", "20260826"]
            for date in dates:
                _add_sources(root, date)
                _add_membership(run_dir, date)
            _write_run(
                run_dir,
                _manifest(
                    start=dates[0],
                    end=dates[-1],
                    sessions=dates,
                ),
                [row for date in dates for row in _daily_rows(date)],
            )

            retrospective, retrospective_manifest = study.build_calendar(
                project_root=root,
                run_dir=run_dir,
                start="20260825",
                end="20260825",
            )
            artifact_oos, artifact_manifest = study.build_calendar(
                project_root=root,
                run_dir=run_dir,
                start="20260826",
                end="20260826",
            )

            self.assertEqual(retrospective["Date"].to_list(), ["20260825"])
            self.assertEqual(
                retrospective_manifest["evidence_class"],
                "retrospective_diagnostic",
            )
            self.assertEqual(artifact_oos["Date"].to_list(), ["20260826"])
            self.assertEqual(
                artifact_manifest["evidence_class"],
                "artifact_oos_non_pristine",
            )
            with self.assertRaisesRegex(ValueError, "exactly one evidence class"):
                study.build_calendar(
                    project_root=root,
                    run_dir=run_dir,
                    start="20260825",
                    end="20260826",
                )

    def test_statuses_are_distinct_and_zero_event_remains_candidate_level(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run"
            evaluable = "20260102"
            zero_control = "20260103"
            failed = "20260104"
            missing_source = "20260105"
            not_run = "20260106"
            wrong_cells = "20260107"
            zero_event = "20260108"
            dates = [
                evaluable,
                zero_control,
                failed,
                missing_source,
                not_run,
                wrong_cells,
                zero_event,
            ]
            for date in dates:
                _add_sources(
                    root,
                    date,
                    omit=("pre_market",) if date == missing_source else (),
                )
            for date in (evaluable, zero_control, wrong_cells, zero_event):
                _add_membership(run_dir, date)
            malformed_family = (*study.CELL_IDS[:-1], "not_a_frozen_cell")
            daily_rows = [
                *_daily_rows(evaluable),
                *_daily_rows(
                    zero_control,
                    eligible_pool_rows=0,
                    control_n_z=0,
                ),
                *_daily_rows(wrong_cells, candidate_ids=malformed_family),
                *_daily_rows(zero_event, condition_n_z=0),
            ]
            complete_sessions = [date for date in dates if date != missing_source]
            _write_run(
                run_dir,
                _manifest(
                    start=dates[0],
                    end=dates[-1],
                    sessions=complete_sessions,
                    failures=[{"Date": failed, "error": "synthetic failure"}],
                ),
                daily_rows,
            )

            calendar, manifest = study.build_calendar(
                project_root=root,
                run_dir=run_dir,
                start=dates[0],
                end=dates[-1],
            )
            by_date = {
                row["Date"]: row for row in calendar.iter_rows(named=True)
            }

            self.assertEqual(by_date[evaluable]["run_status"], "success")
            self.assertEqual(by_date[evaluable]["session_status"], "evaluable")
            self.assertEqual(
                by_date[zero_control]["session_status"],
                "zero_control_ineligible",
            )
            self.assertEqual(by_date[failed]["run_status"], "failed")
            self.assertEqual(by_date[failed]["session_status"], "run_failure")
            self.assertEqual(
                by_date[missing_source]["session_status"],
                "missing_source",
            )
            self.assertEqual(by_date[not_run]["run_status"], "not_run")
            self.assertEqual(
                by_date[not_run]["session_status"],
                "not_run_or_incomplete",
            )
            self.assertFalse(by_date[wrong_cells]["exact_frozen_cells"])
            self.assertEqual(by_date[wrong_cells]["run_status"], "incomplete")
            self.assertEqual(by_date[zero_event]["session_status"], "evaluable")
            self.assertEqual(manifest["evaluable_sessions"], 2)
            self.assertEqual(manifest["zero_control_sessions"], 1)
            self.assertEqual(manifest["missing_source_sessions"], 1)
            self.assertEqual(manifest["failed_run_sessions"], 1)
            self.assertEqual(manifest["not_run_or_incomplete_sessions"], 2)

    def test_premarket_only_file_does_not_invent_a_requested_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run"
            trading_date = "20260102"
            stray_date = "20260103"
            _add_sources(root, trading_date)
            premarket = _source_path(root, "pre_market", stray_date)
            premarket.parent.mkdir(parents=True, exist_ok=True)
            premarket.write_bytes(b"")
            _add_membership(run_dir, trading_date)
            _write_run(
                run_dir,
                _manifest(
                    start=trading_date,
                    end=stray_date,
                    sessions=[trading_date],
                ),
                _daily_rows(trading_date),
            )

            calendar, manifest = study.build_calendar(
                project_root=root,
                run_dir=run_dir,
                start=trading_date,
                end=stray_date,
            )
            self.assertEqual(calendar["Date"].to_list(), [trading_date])
            self.assertEqual(
                manifest["premarket_only_dates_excluded"],
                [stray_date],
            )

    def test_manifest_identity_and_dense_sampling_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run"
            date = "20260102"
            _add_sources(root, date)
            _add_membership(run_dir, date)
            manifest = _manifest(start=date, end=date, sessions=[date])
            _write_run(run_dir, manifest, _daily_rows(date))

            manifest["family_id"] = "wrong_family"
            _write_run(run_dir, manifest, _daily_rows(date))
            with self.assertRaisesRegex(ValueError, "run identity"):
                study.build_calendar(
                    project_root=root,
                    run_dir=run_dir,
                    start=date,
                    end=date,
                )

            manifest["family_id"] = study.FACTORIAL_FAMILY_ID
            manifest["step"] = 5
            _write_run(run_dir, manifest, _daily_rows(date))
            with self.assertRaisesRegex(ValueError, "step=1"):
                study.build_calendar(
                    project_root=root,
                    run_dir=run_dir,
                    start=date,
                    end=date,
                )


if __name__ == "__main__":
    unittest.main()
