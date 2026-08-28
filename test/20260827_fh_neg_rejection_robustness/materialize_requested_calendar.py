"""Materialize the external status calendar required by the robustness audit."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[5]
FACTORIAL_CONTRACT = "fh_neg_rejection_robustness/fixed_factorial_v1"
FACTORIAL_FAMILY_ID = "fh_neg_rejection_dgb_factorial"
FACTORIAL_FAMILY_VERSION = "1"
REQUIRED_SOURCES = {
    "tick_data": ("tickData", "_StockTick.parquet"),
    "tick_feature": ("tickFeature", "_tickFeature.parquet"),
    "pre_market": ("preMarket", "_preMarketData.parquet"),
    "txf_tick": ("txfTickData", "_TxfTick.parquet"),
}
CELL_IDS = (
    "d0_g0_b0",
    "d1_g0_b0",
    "d0_g1_b0",
    "d0_g0_b1",
    "d1_g1_b0",
    "d1_g0_b1",
    "d0_g1_b1",
    "d1_g1_b1",
)
CELL_COUNT = len(CELL_IDS)


def evidence_class(date: str) -> str:
    if date <= "20260825":
        return "retrospective_diagnostic"
    if date == "20260826":
        return "artifact_oos_non_pristine"
    if date == "20260827":
        return "freeze_day_excluded"
    return "prospective_first_live"


def _validate_date(value: object, *, field: str) -> str:
    date = str(value)
    try:
        parsed = datetime.strptime(date, "%Y%m%d")
    except ValueError as exc:
        raise ValueError(f"{field} must be a valid YYYYMMDD date: {date!r}") from exc
    if parsed.strftime("%Y%m%d") != date:
        raise ValueError(f"{field} must be a valid YYYYMMDD date: {date!r}")
    return date


def _source_dates(directory: Path, suffix: str) -> set[str]:
    dates: set[str] = set()
    for path in directory.glob(f"*{suffix}"):
        if len(path.name) >= 8 and path.name[:8].isdigit():
            dates.add(_validate_date(path.name[:8], field=f"source filename {path.name}"))
    return dates


def _validate_run_manifest(
    run_manifest: dict[str, object],
    *,
    start: str,
    end: str,
) -> tuple[list[str], dict[str, str]]:
    expected_identity = {
        "contract": FACTORIAL_CONTRACT,
        "family_id": FACTORIAL_FAMILY_ID,
        "family_version": FACTORIAL_FAMILY_VERSION,
    }
    mismatches = {
        field: {"expected": expected, "actual": run_manifest.get(field)}
        for field, expected in expected_identity.items()
        if str(run_manifest.get(field)) != expected
    }
    if mismatches:
        raise ValueError(f"unexpected factorial run identity: {mismatches}")
    if run_manifest.get("candidate_ids") != list(CELL_IDS):
        raise ValueError("run manifest does not contain the frozen eight-cell family")
    if run_manifest.get("step") != 1:
        raise ValueError("requested calendar requires a dense factorial run with step=1")

    manifest_start = _validate_date(run_manifest.get("start"), field="manifest start")
    manifest_end = _validate_date(run_manifest.get("end"), field="manifest end")
    if not manifest_start <= start <= end <= manifest_end:
        raise ValueError(
            "calendar range must be a nonempty subset of the factorial run range: "
            f"calendar={start}..{end}, run={manifest_start}..{manifest_end}"
        )

    raw_sessions = run_manifest.get("sessions")
    if not isinstance(raw_sessions, list):
        raise ValueError("run manifest sessions must be a list")
    sessions = [
        _validate_date(value, field="manifest session") for value in raw_sessions
    ]
    if sessions != sorted(set(sessions)):
        raise ValueError("run manifest sessions must be unique and sorted")
    if any(not manifest_start <= date <= manifest_end for date in sessions):
        raise ValueError("run manifest session lies outside its declared range")

    raw_failures = run_manifest.get("failures")
    if not isinstance(raw_failures, list):
        raise ValueError("run manifest failures must be a list")
    failures: dict[str, str] = {}
    for item in raw_failures:
        if not isinstance(item, dict) or "Date" not in item:
            raise ValueError("each run failure must contain Date")
        date = _validate_date(item["Date"], field="failure Date")
        if date not in sessions:
            raise ValueError(f"failure Date {date} is absent from manifest sessions")
        if date in failures:
            raise ValueError(f"duplicate failure Date {date}")
        failures[date] = str(item.get("error", "unknown"))
    return sessions, failures


def _atomic_write_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temp_path = Path(handle.name)
    handle.close()
    try:
        frame.write_parquet(temp_path, compression="zstd")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _atomic_write_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
        encoding="utf-8",
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def build_calendar(
    *,
    project_root: Path,
    run_dir: Path,
    start: str,
    end: str,
) -> tuple[pl.DataFrame, dict[str, object]]:
    start = _validate_date(start, field="calendar start")
    end = _validate_date(end, field="calendar end")
    run_manifest_path = run_dir / "run_manifest.json"
    daily_path = run_dir / "daily_stats.parquet"
    if not run_manifest_path.is_file() or not daily_path.is_file():
        raise FileNotFoundError("completed run_manifest.json and daily_stats.parquet required")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    manifest_sessions, failures = _validate_run_manifest(
        run_manifest,
        start=start,
        end=end,
    )

    source_dates = {
        source: _source_dates(
            project_root / "data" / directory,
            suffix,
        )
        for source, (directory, suffix) in REQUIRED_SOURCES.items()
    }
    # tickData is the raw event-stream calendar.  Include any date visible in
    # either intraday derivative source as a safeguard against a missing raw
    # tick file; preMarket-only stray files cannot create a trading session.
    intraday_calendar = (
        source_dates["tick_data"]
        | source_dates["tick_feature"]
        | source_dates["txf_tick"]
    )
    requested = sorted(
        date for date in intraday_calendar if start <= date <= end
    )
    premarket_only_excluded = sorted(
        date
        for date in source_dates["pre_market"] - intraday_calendar
        if start <= date <= end
    )
    if not requested:
        raise ValueError("external source calendar has no requested sessions")
    evidence_classes = {evidence_class(date) for date in requested}
    if len(evidence_classes) != 1:
        raise ValueError(
            "one requested calendar may contain exactly one evidence class; "
            f"found {sorted(evidence_classes)}"
        )
    daily = pl.read_parquet(daily_path)
    required_daily = {
        "Date",
        "candidate_id",
        "eligible_pool_rows",
        "condition_n_z",
        "control_n_z",
    }
    missing_daily = sorted(required_daily - set(daily.columns))
    if missing_daily:
        raise ValueError(f"daily_stats missing columns: {missing_daily}")
    daily_status = {
        str(row["Date"]): row
        for row in daily.group_by("Date").agg(
            pl.len().alias("cell_rows"),
            pl.col("candidate_id").alias("candidate_ids"),
            pl.col("eligible_pool_rows").n_unique().alias("pool_value_count"),
            pl.col("eligible_pool_rows").first().alias("eligible_pool_rows"),
            pl.col("control_n_z").n_unique().alias("control_value_count"),
            pl.col("control_n_z").first().alias("control_n_z"),
        ).iter_rows(named=True)
    }

    rows: list[dict[str, object]] = []
    for date in requested:
        source_flags = {
            f"has_{source}": date in dates
            for source, dates in source_dates.items()
        }
        source_complete = all(source_flags.values())
        status = daily_status.get(date)
        exact_cells = bool(
            status is not None
            and set(status["candidate_ids"]) == set(CELL_IDS)
        )
        aggregate_complete = bool(
            status is not None
            and int(status["cell_rows"]) == CELL_COUNT
            and exact_cells
            and int(status["pool_value_count"]) == 1
            and int(status["control_value_count"]) == 1
        )
        membership_paths = (
            run_dir / "membership_daily" / f"{date}.parquet",
            run_dir / "membership_lineage" / f"{date}.json",
        )
        membership_complete = all(path.is_file() for path in membership_paths)
        membership_any = any(path.is_file() for path in membership_paths)
        if not source_complete:
            run_status = "not_attempted"
        elif date in failures:
            run_status = "failed"
        elif aggregate_complete and membership_complete:
            run_status = "success"
        elif status is None and not membership_any:
            run_status = "not_run"
        else:
            run_status = "incomplete"
        eligible_pool_rows = (
            int(status["eligible_pool_rows"]) if status is not None else None
        )
        control_n_z = int(status["control_n_z"]) if status is not None else None
        if eligible_pool_rows is not None and eligible_pool_rows < 0:
            raise ValueError(f"{date}: eligible_pool_rows must be nonnegative")
        if control_n_z is not None and control_n_z < 0:
            raise ValueError(f"{date}: control_n_z must be nonnegative")
        if not source_complete:
            session_status = "missing_source"
        elif run_status == "failed":
            session_status = "run_failure"
        elif run_status != "success":
            session_status = "not_run_or_incomplete"
        elif not eligible_pool_rows or not control_n_z:
            session_status = "zero_control_ineligible"
        else:
            session_status = "evaluable"
        rows.append(
            {
                "Date": date,
                "evidence_class": evidence_class(date),
                **source_flags,
                "source_complete": source_complete,
                "run_status": run_status,
                "session_status": session_status,
                "eligible_pool_rows": eligible_pool_rows,
                "control_n_z": control_n_z,
                "exact_frozen_cells": exact_cells,
                "aggregate_complete": aggregate_complete,
                "membership_complete": membership_complete,
                "run_error": failures.get(date),
            }
        )
    calendar = pl.DataFrame(rows).sort("Date")
    expected_complete = {
        date for date in manifest_sessions if start <= date <= end
    }
    actual_complete = set(
        calendar.filter(pl.col("source_complete"))["Date"].to_list()
    )
    if expected_complete != actual_complete:
        raise ValueError(
            "run manifest sessions disagree with external source calendar: "
            f"manifest_only={sorted(expected_complete - actual_complete)[:5]}, "
            f"calendar_only={sorted(actual_complete - expected_complete)[:5]}"
        )
    incomplete = calendar.filter(pl.col("run_status") != "success")
    manifest: dict[str, object] = {
        "contract": "fh_neg_rejection_requested_calendar/v1",
        "factorial_contract": FACTORIAL_CONTRACT,
        "factorial_family_id": FACTORIAL_FAMILY_ID,
        "factorial_family_version": FACTORIAL_FAMILY_VERSION,
        "start": start,
        "end": end,
        "evidence_class": next(iter(evidence_classes)),
        "requested_calendar_basis": (
            "union of tickData, tickFeature, and txfTickData filename dates; "
            "preMarket-only dates excluded"
        ),
        "premarket_only_dates_excluded": premarket_only_excluded,
        "coverage_limitation": (
            "a trading session absent from all three intraday directory "
            "calendars cannot be discovered without a separate exchange calendar"
        ),
        "requested_sessions": len(calendar),
        "source_complete_sessions": int(calendar["source_complete"].sum()),
        "successful_sessions": int((calendar["run_status"] == "success").sum()),
        "evaluable_sessions": int((calendar["session_status"] == "evaluable").sum()),
        "zero_control_sessions": int(
            (calendar["session_status"] == "zero_control_ineligible").sum()
        ),
        "non_successful_sessions": len(incomplete),
        "missing_source_sessions": int(
            (calendar["session_status"] == "missing_source").sum()
        ),
        "failed_run_sessions": int(
            (calendar["session_status"] == "run_failure").sum()
        ),
        "not_run_or_incomplete_sessions": int(
            (calendar["session_status"] == "not_run_or_incomplete").sum()
        ),
        "status_counts": {
            str(row["session_status"]): int(row["len"])
            for row in calendar.group_by("session_status").len().iter_rows(named=True)
        },
        "evidence_class_counts": {
            str(row["evidence_class"]): int(row["len"])
            for row in calendar.group_by("evidence_class").len().iter_rows(named=True)
        },
    }
    return calendar, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build external factorial status calendar")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory; defaults to --run-dir for backward compatibility",
    )
    parser.add_argument("--start", default="20250211")
    parser.add_argument("--end", default="20260825")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    calendar_frame, calendar_manifest = build_calendar(
        project_root=PROJECT_ROOT,
        run_dir=args.run_dir,
        start=args.start,
        end=args.end,
    )
    output_dir = args.output_dir or args.run_dir
    _atomic_write_parquet(calendar_frame, output_dir / "requested_calendar.parquet")
    _atomic_write_json(calendar_manifest, output_dir / "requested_calendar.json")
    print(json.dumps(calendar_manifest, indent=2, sort_keys=True))
