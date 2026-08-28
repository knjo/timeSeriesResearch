"""Validate and summarize selector lineage without reading any outcomes.

This audit is intentionally downstream of a *completed* factorial run and its
external requested-session calendar.  It never reads ``daily_stats.parquet``:
all counts below are raw selector membership/support, not label-valid support
or performance.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import polars as pl


STUDY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STUDY_DIR.parents[4]
CONTRACT_PATH = STUDY_DIR / "study_contract.json"
DEFAULT_PROFILE_SOURCE = (
    PROJECT_ROOT
    / "src/research/timeSeries/framework/profiles/filllots_at_high.py"
)
DEFAULT_RUNNER_SOURCE = STUDY_DIR / "run_factorial.py"

CONTRACT_BYTES = CONTRACT_PATH.read_bytes()
CONTRACT = json.loads(CONTRACT_BYTES)
CONTRACT_SHA256 = hashlib.sha256(CONTRACT_BYTES).hexdigest()
CONTRACT_ID = str(CONTRACT["contract_id"])
FAMILY = CONTRACT["primary_candidate_family"]
FAMILY_ID = str(FAMILY["family_id"])
FAMILY_VERSION = str(FAMILY["family_version"])
CELL_IDS = tuple(str(cell["candidate_id"]) for cell in FAMILY["cells"])
CANONICAL_KEY = tuple(str(value) for value in CONTRACT["sample_unit"]["canonical_key"])
MEMBERSHIP_KEY = CANONICAL_KEY
REFERENCE_COLUMNS = (
    "avg_bidLots1",
    "avg_askLots1",
    "big_buy_lots",
    "big_sell_lots",
)
REFERENCE_STATUS_COLUMNS = tuple(
    f"{column}_lineage_status" for column in REFERENCE_COLUMNS
)
REFERENCE_STATUSES = {
    "not_ambiguous",
    "proven_genuine_one",
    "stored_one_mismatch",
    "unresolved_missing_raw_support",
}
BRANCHES = {
    "md_only",
    "large_fill_only",
    "md_and_large_fill",
    "invalid_large_fill_only",
}
SUCCESS_SESSION_STATUSES = {"evaluable", "zero_control_ineligible"}

REQUIRED_MEMBERSHIP_COLUMNS = {
    *MEMBERSHIP_KEY,
    "membership_mask",
    "conservative_base",
    "legacy_dgb",
    "conservative_dgb",
    "pass_d_legacy",
    "pass_d",
    "pass_g",
    "pass_b",
    "InOut",
    "_md_price_change",
    "_large_fill",
    "large_fill_conservative",
    "fresh_branch_conservative",
    "avg_bid_ambiguous_one",
    "avg_ask_ambiguous_one",
    "big_buy_ambiguous_one",
    "big_sell_ambiguous_one",
    "depth_reference_invalid",
    "large_fill_reference_invalid",
    *REFERENCE_STATUS_COLUMNS,
}
REQUIRED_CALENDAR_COLUMNS = {
    "Date",
    "evidence_class",
    "source_complete",
    "run_status",
    "session_status",
    "membership_complete",
    "aggregate_complete",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256_file(path),
    }


def _raw_source_identity(path: Path) -> dict[str, object]:
    """Identity shape stored by the shared runner for raw source files."""

    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return payload


def _atomic_write_json(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
        encoding="utf-8",
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _atomic_write_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    temp_path = Path(handle.name)
    handle.close()
    try:
        frame.write_parquet(temp_path, compression="zstd")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _require_columns(frame: pl.DataFrame, required: Iterable[str], context: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{context}: missing columns {missing}")


def _count_true(frame: pl.DataFrame, column: str) -> int:
    value = frame.select(pl.col(column).cast(pl.UInt64).sum()).item()
    return int(value or 0)


def _unique_quotes(frame: pl.DataFrame) -> set[str]:
    if frame.is_empty():
        return set()
    return {str(value) for value in frame["QuoteCode"].drop_nulls().unique()}


def _value_counts(frame: pl.DataFrame, column: str) -> dict[str, int]:
    if frame.is_empty():
        return {}
    return {
        str(row[column]): int(row["len"])
        for row in frame.group_by(column).len().iter_rows(named=True)
    }


def _require_equal(actual: object, expected: object, context: str) -> None:
    if actual != expected:
        raise ValueError(f"{context}: expected {expected!r}, got {actual!r}")


@dataclass
class SupportAccumulator:
    events: int = 0
    stock_days: int = 0
    active_sessions: int = 0
    stocks: set[str] = field(default_factory=set)

    def add(self, frame: pl.DataFrame) -> None:
        if frame.is_empty():
            return
        quotes = _unique_quotes(frame)
        self.events += len(frame)
        self.stock_days += len(quotes)
        self.active_sessions += 1
        self.stocks.update(quotes)

    def row(
        self, *, cohort_type: str, cohort_id: str, stratum: str = "all"
    ) -> dict[str, object]:
        return {
            "cohort_type": cohort_type,
            "cohort_id": cohort_id,
            "stratum": stratum,
            "events": self.events,
            "stock_days": self.stock_days,
            "stocks": len(self.stocks),
            "active_sessions": self.active_sessions,
        }


def _validate_run_manifest(
    manifest: Mapping[str, object], reference_path: Path
) -> tuple[list[str], dict[str, object]]:
    expected = {
        "contract": CONTRACT_ID,
        "family_id": FAMILY_ID,
        "family_version": FAMILY_VERSION,
        "frozen_contract_sha256": CONTRACT_SHA256,
        "label_profile_id": str(CONTRACT["directional_label"]["profile_id"]),
        "pool_id": str(CONTRACT["sample_unit"]["eligible_pool_id"]),
        "pool_version": "1",
        "primary_estimand": "shared_pool",
        "stock_time_bucket": "5m",
        "control_sample_size": None,
        "primary_membership": "conservative_reference_validated",
        "legacy_parity_membership": "diagnostic_only",
        "candidate_frame_builds_per_processed_day": 1,
        "step": 1,
    }
    for field_name, value in expected.items():
        _require_equal(manifest.get(field_name), value, f"run manifest {field_name}")
    _require_equal(
        manifest.get("candidate_ids"), list(CELL_IDS), "run manifest candidate_ids"
    )
    design_fingerprint = manifest.get("factorial_design_fingerprint")
    if not isinstance(design_fingerprint, str) or len(design_fingerprint) != 64:
        raise ValueError("run manifest lacks a completed factorial design fingerprint")
    sessions = manifest.get("sessions")
    if not isinstance(sessions, list) or not all(isinstance(x, str) for x in sessions):
        raise ValueError("run manifest sessions must be a string list")
    if sessions != sorted(set(sessions)):
        raise ValueError("run manifest sessions must be sorted and unique")
    source_identities = manifest.get("source_identities")
    if not isinstance(source_identities, dict):
        raise ValueError("run manifest source_identities must be an object")
    _require_equal(
        manifest.get("reference_lineage_artifact"),
        _file_identity(reference_path),
        "run manifest reference_lineage_artifact",
    )
    failures = manifest.get("failures")
    if not isinstance(failures, list):
        raise ValueError("run manifest failures must be a list")
    failure_dates: list[str] = []
    for failure in failures:
        if not isinstance(failure, dict) or not isinstance(failure.get("Date"), str):
            raise ValueError("run manifest failure entries require a Date")
        failure_dates.append(str(failure["Date"]))
    if len(failure_dates) != len(set(failure_dates)):
        raise ValueError("run manifest failures contain duplicate dates")
    return sessions, source_identities


def _calendar_count(frame: pl.DataFrame, expression: pl.Expr) -> int:
    return int(frame.select(expression.cast(pl.UInt64).sum()).item() or 0)


def _validate_calendar(
    calendar: pl.DataFrame,
    calendar_manifest: Mapping[str, object],
    run_manifest: Mapping[str, object],
    run_sessions: Sequence[str],
) -> list[str]:
    _require_columns(calendar, REQUIRED_CALENDAR_COLUMNS, "requested calendar")
    if calendar.is_empty():
        raise ValueError("requested calendar is empty")
    if calendar["Date"].null_count() or calendar["Date"].n_unique() != len(calendar):
        raise ValueError("requested calendar Date must be non-null and unique")
    dates = [str(value) for value in calendar["Date"].to_list()]
    if dates != sorted(dates):
        raise ValueError("requested calendar Date must be sorted")
    evidence_classes = calendar["evidence_class"].unique().to_list()
    if len(evidence_classes) != 1:
        raise ValueError("requested calendar must contain one evidence class")
    expected_calendar_identity = {
        "contract": "fh_neg_rejection_requested_calendar/v1",
        "factorial_contract": CONTRACT_ID,
        "factorial_family_id": FAMILY_ID,
        "factorial_family_version": FAMILY_VERSION,
        "requested_sessions": len(calendar),
        "evidence_class": evidence_classes[0],
    }
    for name, value in expected_calendar_identity.items():
        _require_equal(
            calendar_manifest.get(name), value, f"requested calendar manifest {name}"
        )
    successful = calendar.filter(pl.col("run_status") == "success")
    expected_counts = {
        "source_complete_sessions": _calendar_count(calendar, pl.col("source_complete")),
        "successful_sessions": len(successful),
        "evaluable_sessions": _calendar_count(
            calendar, pl.col("session_status") == "evaluable"
        ),
        "zero_control_sessions": _calendar_count(
            calendar, pl.col("session_status") == "zero_control_ineligible"
        ),
        "missing_source_sessions": _calendar_count(
            calendar, pl.col("session_status") == "missing_source"
        ),
        "failed_run_sessions": _calendar_count(
            calendar, pl.col("session_status") == "run_failure"
        ),
        "not_run_or_incomplete_sessions": _calendar_count(
            calendar, pl.col("session_status") == "not_run_or_incomplete"
        ),
    }
    for name, value in expected_counts.items():
        _require_equal(calendar_manifest.get(name), value, f"calendar count {name}")
    status_counts = {
        str(row["session_status"]): int(row["len"])
        for row in calendar.group_by("session_status").len().iter_rows(named=True)
    }
    evidence_counts = {
        str(row["evidence_class"]): int(row["len"])
        for row in calendar.group_by("evidence_class").len().iter_rows(named=True)
    }
    if "status_counts" in calendar_manifest:
        _require_equal(
            calendar_manifest.get("status_counts"), status_counts, "calendar status_counts"
        )
    if "evidence_class_counts" in calendar_manifest:
        _require_equal(
            calendar_manifest.get("evidence_class_counts"),
            evidence_counts,
            "calendar evidence_class_counts",
        )
    if expected_counts["not_run_or_incomplete_sessions"]:
        raise ValueError("requested scope is not a completed run: incomplete sessions remain")
    if successful.filter(
        ~pl.col("session_status").is_in(sorted(SUCCESS_SESSION_STATUSES))
        | ~pl.col("membership_complete")
        | ~pl.col("aggregate_complete")
        | ~pl.col("source_complete")
    ).height:
        raise ValueError("successful calendar rows have inconsistent completion flags")
    if calendar.filter(
        (pl.col("session_status") == "run_failure")
        & (pl.col("run_status") != "failed")
    ).height:
        raise ValueError("calendar run_failure rows are not marked failed")
    scoped_run_sessions = sorted(
        date for date in run_sessions if dates[0] <= date <= dates[-1]
    )
    source_complete_dates = calendar.filter(pl.col("source_complete"))["Date"].to_list()
    _require_equal(
        scoped_run_sessions,
        source_complete_dates,
        "run manifest sessions vs external source-complete calendar",
    )
    failure_dates = {
        str(item["Date"])
        for item in run_manifest.get("failures", [])
        if dates[0] <= str(item["Date"]) <= dates[-1]
    }
    calendar_failures = set(
        calendar.filter(pl.col("run_status") == "failed")["Date"].to_list()
    )
    _require_equal(failure_dates, calendar_failures, "run failure date lineage")
    return dates


def _validate_reference_audit(
    frame: pl.DataFrame, manifest: Mapping[str, object]
) -> dict[str, dict[str, int]]:
    _require_equal(
        manifest.get("contract"),
        "fh_neg_rejection_reference_lineage_audit/v1",
        "reference audit contract",
    )
    _require_equal(manifest.get("outcome_columns_read"), [], "reference outcome read set")
    if frame.is_empty():
        _require_equal(manifest.get("ambiguous_stock_sessions"), 0, "reference rows")
        _require_equal(
            manifest.get("invalid_or_mismatched_stock_sessions"),
            0,
            "reference invalid rows",
        )
        _require_equal(
            manifest.get("reference_status_counts"),
            {column: {} for column in REFERENCE_COLUMNS},
            "reference status counts",
        )
        return {column: {} for column in REFERENCE_COLUMNS}
    _require_columns(
        frame,
        {"Date", "QuoteCode", "has_invalid_ambiguous_reference", *REFERENCE_STATUS_COLUMNS},
        "reference lineage",
    )
    if frame.select(pl.struct(["Date", "QuoteCode"]).is_duplicated().any()).item():
        raise ValueError("reference lineage contains duplicate stock-session keys")
    if frame.select(pl.any_horizontal(pl.col(["Date", "QuoteCode"]).is_null()).any()).item():
        raise ValueError("reference lineage contains null stock-session keys")
    invalid_exprs: list[pl.Expr] = []
    status_counts: dict[str, dict[str, int]] = {}
    for column, status_column in zip(REFERENCE_COLUMNS, REFERENCE_STATUS_COLUMNS):
        if frame[status_column].null_count():
            raise ValueError(f"reference lineage {status_column} contains null status")
        observed = set(str(value) for value in frame[status_column].drop_nulls().unique())
        unknown = sorted(observed - REFERENCE_STATUSES)
        if unknown:
            raise ValueError(f"reference lineage {status_column} has unknown statuses {unknown}")
        status_counts[column] = _value_counts(frame, status_column)
        invalid_exprs.append(
            pl.col(status_column).is_in(
                ["stored_one_mismatch", "unresolved_missing_raw_support"]
            )
        )
    recomputed_invalid = pl.any_horizontal(invalid_exprs)
    if frame.filter(
        pl.col("has_invalid_ambiguous_reference") != recomputed_invalid
    ).height:
        raise ValueError("reference lineage invalid flag disagrees with statuses")
    invalid = _count_true(frame, "has_invalid_ambiguous_reference")
    _require_equal(manifest.get("ambiguous_stock_sessions"), len(frame), "reference rows")
    _require_equal(
        manifest.get("invalid_or_mismatched_stock_sessions"), invalid, "reference invalid rows"
    )
    _require_equal(
        manifest.get("reference_status_counts"), status_counts, "reference status counts"
    )
    return status_counts


def _membership_mask(frame: pl.DataFrame) -> pl.Series:
    mask = pl.Series("_expected_mask", [0] * len(frame), dtype=pl.UInt16)
    for index, cell in enumerate(FAMILY["cells"]):
        expression = pl.col("conservative_base")
        if cell["D"]:
            expression = expression & pl.col("pass_d")
        if cell["G"]:
            expression = expression & pl.col("pass_g")
        if cell["B"]:
            expression = expression & pl.col("pass_b")
        bit = frame.select(expression.fill_null(False).cast(pl.UInt16)).to_series()
        mask = mask + bit * (1 << index)
    return mask


def _validate_membership_logic(frame: pl.DataFrame, date: str) -> None:
    _require_columns(frame, REQUIRED_MEMBERSHIP_COLUMNS, f"{date} membership")
    if frame.select(pl.any_horizontal(pl.col(list(MEMBERSHIP_KEY)).is_null()).any()).item():
        raise ValueError(f"{date}: null canonical membership key")
    if frame.select(pl.struct(list(MEMBERSHIP_KEY)).is_duplicated().any()).item():
        raise ValueError(f"{date}: duplicate canonical membership key")
    if frame.filter(pl.col("Date") != date).height:
        raise ValueError(f"{date}: membership Date disagrees with filename")
    expected_legacy = pl.col("pass_d_legacy") & pl.col("pass_g") & pl.col("pass_b")
    expected_conservative = (
        pl.col("conservative_base")
        & pl.col("pass_d")
        & pl.col("pass_g")
        & pl.col("pass_b")
    )
    if frame.filter(pl.col("legacy_dgb") != expected_legacy).height:
        raise ValueError(f"{date}: legacy DGB logical invariant failed")
    if frame.filter(pl.col("conservative_dgb") != expected_conservative).height:
        raise ValueError(f"{date}: conservative DGB logical invariant failed")
    expected_base = pl.col("_md_price_change") | pl.col("large_fill_conservative")
    if frame.filter(pl.col("conservative_base") != expected_base).height:
        raise ValueError(f"{date}: conservative fresh-base logical invariant failed")
    expected_mask = _membership_mask(frame)
    if frame["membership_mask"].cast(pl.UInt16).to_list() != expected_mask.to_list():
        raise ValueError(f"{date}: membership mask disagrees with frozen cells")
    observed_branches = set(str(value) for value in frame["fresh_branch_conservative"].unique())
    unknown_branches = sorted(observed_branches - BRANCHES)
    if unknown_branches:
        raise ValueError(f"{date}: unknown fresh branches {unknown_branches}")
    branch_expr = (
        pl.when(pl.col("_md_price_change") & pl.col("large_fill_conservative"))
        .then(pl.lit("md_and_large_fill"))
        .when(pl.col("_md_price_change"))
        .then(pl.lit("md_only"))
        .when(pl.col("large_fill_conservative"))
        .then(pl.lit("large_fill_only"))
        .otherwise(pl.lit("invalid_large_fill_only"))
    )
    if frame.filter(pl.col("fresh_branch_conservative") != branch_expr).height:
        raise ValueError(f"{date}: fresh branch label invariant failed")
    if frame.filter(
        pl.col("large_fill_conservative") & ~pl.col("InOut").is_in([-1, 1])
    ).height:
        raise ValueError(f"{date}: conservative large-fill row lacks a valid side")


def _lineage_expected_counts(frame: pl.DataFrame) -> dict[str, object]:
    candidate_counts = {
        candidate_id: _count_true(
            frame.with_columns(
                ((pl.col("membership_mask") & (1 << index)) != 0).alias("_member")
            ),
            "_member",
        )
        for index, candidate_id in enumerate(CELL_IDS)
    }
    return {
        "legacy_base_rows": len(frame),
        "conservative_base_rows": _count_true(frame, "conservative_base"),
        "legacy_dgb_rows": _count_true(frame, "legacy_dgb"),
        "conservative_dgb_rows": _count_true(frame, "conservative_dgb"),
        "legacy_dgb_removed_by_reference_validation": frame.filter(
            pl.col("legacy_dgb") & ~pl.col("conservative_dgb")
        ).height,
        "candidate_raw_rows": candidate_counts,
        "ambiguous_one_exposure": {
            "avg_bid_event_rows": _count_true(frame, "avg_bid_ambiguous_one"),
            "avg_ask_event_rows": _count_true(frame, "avg_ask_ambiguous_one"),
            "big_buy_event_rows": _count_true(frame, "big_buy_ambiguous_one"),
            "big_sell_event_rows": _count_true(frame, "big_sell_ambiguous_one"),
            "depth_invalid_event_rows": _count_true(frame, "depth_reference_invalid"),
            "depth_invalid_quotes": len(
                _unique_quotes(frame.filter(pl.col("depth_reference_invalid")))
            ),
            "side_large_fill_invalid_event_rows": _count_true(
                frame, "large_fill_reference_invalid"
            ),
            "side_large_fill_invalid_quotes": len(
                _unique_quotes(frame.filter(pl.col("large_fill_reference_invalid")))
            ),
        },
        "fresh_branch_counts": _value_counts(frame, "fresh_branch_conservative"),
        "legacy_large_fill_inout_counts": _value_counts(
            frame.filter(pl.col("_large_fill")), "InOut"
        ),
        "conservative_large_fill_inout_counts": _value_counts(
            frame.filter(pl.col("large_fill_conservative")), "InOut"
        ),
    }


def _validate_membership_lineage(
    *,
    date: str,
    frame: pl.DataFrame,
    lineage: Mapping[str, object],
    membership_path: Path,
    reference_path: Path,
    run_manifest: Mapping[str, object],
    source_identities: Mapping[str, object],
) -> str:
    expected = {
        "contract": CONTRACT_ID,
        "frozen_contract_sha256": CONTRACT_SHA256,
        "design_fingerprint": run_manifest["factorial_design_fingerprint"],
        "family_id": FAMILY_ID,
        "family_version": FAMILY_VERSION,
        "Date": date,
        "membership_artifact": _file_identity(membership_path),
        "reference_lineage_artifact": _file_identity(reference_path),
        "source_identities": source_identities,
        "source_identities_sha256": _canonical_json_sha256(source_identities),
    }
    for name, value in expected.items():
        _require_equal(lineage.get(name), value, f"{date} membership lineage {name}")
    for name, value in _lineage_expected_counts(frame).items():
        _require_equal(lineage.get(name), value, f"{date} membership lineage {name}")
    parity = lineage.get("legacy_parity")
    if not isinstance(parity, dict):
        raise ValueError(f"{date}: legacy parity payload is missing")
    status = str(parity.get("status"))
    legacy_rows = _count_true(frame, "legacy_dgb")
    if status == "pass":
        for field_name in ("missing_from_rebuilt", "extra_in_rebuilt"):
            _require_equal(parity.get(field_name), 0, f"{date} parity {field_name}")
        _require_equal(parity.get("expected_legacy_rows"), legacy_rows, f"{date} parity expected")
        _require_equal(parity.get("rebuilt_legacy_rows"), legacy_rows, f"{date} parity rebuilt")
    elif status == "not_applicable_outside_legacy_artifact":
        _require_equal(parity.get("rebuilt_legacy_rows"), legacy_rows, f"{date} parity rebuilt")
    else:
        raise ValueError(f"{date}: legacy parity has non-passing status {status!r}")
    return status


def _ast_literal(node: ast.AST, env: Mapping[str, object]) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in env:
        return env[node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        values: list[object] = []
        for element in node.elts:
            if isinstance(element, ast.Starred):
                expanded = _ast_literal(element.value, env)
                if not isinstance(expanded, (tuple, list)):
                    raise ValueError("static selector tuple expansion is not a sequence")
                values.extend(expanded)
            else:
                values.append(_ast_literal(element, env))
        return tuple(values)
    if isinstance(node, ast.Dict):
        return {
            str(_ast_literal(key, env)): _ast_literal(value, env)
            for key, value in zip(node.keys, node.values)
            if key is not None
        }
    raise ValueError(f"unsupported static selector literal: {ast.dump(node)}")


def _assignment(tree: ast.AST, name: str, env: Mapping[str, object]) -> object:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                value = node.value
                if value is None:
                    break
                return _ast_literal(value, env)
    raise ValueError(f"static selector source lacks assignment {name}")


def _profile_required_columns(tree: ast.AST, env: Mapping[str, object]) -> dict[str, tuple[str, ...]]:
    function = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "build_filllots_at_high_candidate_frame"
        ),
        None,
    )
    if function is None:
        raise ValueError("profile source lacks build_filllots_at_high_candidate_frame")
    source_names = {
        "tick": "tick_data",
        "tick_feature": "tick_feature",
        "pre_market": "pre_market",
    }
    result: dict[str, tuple[str, ...]] = {}
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "_require_columns" or len(node.args) < 2:
            continue
        if not isinstance(node.args[0], ast.Name) or node.args[0].id not in source_names:
            raise ValueError("profile _require_columns uses an unknown raw source")
        values = _ast_literal(node.args[1], env)
        if not isinstance(values, tuple) or not all(isinstance(value, str) for value in values):
            raise ValueError("profile required selector columns are not static strings")
        result[source_names[node.args[0].id]] = values
    if set(result) != set(source_names.values()):
        raise ValueError("profile source does not statically declare all selector inputs")
    return result


def _snake_tokens(column: str) -> list[str]:
    snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", column).lower()
    return [token for token in re.split(r"[^a-z0-9]+", snake) if token]


FORBIDDEN_INPUT_TOKENS = {
    "label",
    "target",
    "future",
    "forward",
    "next",
    "exit",
    "close",
    "pnl",
    "profit",
    "return",
    "returns",
    "outcome",
    "lead",
    "horizon",
}


def _input_risk(column: str) -> tuple[bool, str | None]:
    matched = sorted(set(_snake_tokens(column)) & FORBIDDEN_INPUT_TOKENS)
    if matched:
        return True, f"outcome/future-name token(s): {','.join(matched)}"
    return False, None


def _number_literal(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _number_literal(node.operand)
        return -value if value is not None else None
    return None


def _future_syntax_findings(tree: ast.AST, path: Path) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        method = node.func.attr
        if method == "shift" and node.args:
            periods = _number_literal(node.args[0])
            if periods is not None and periods < 0:
                findings.append(
                    {
                        "source_path": str(path.resolve()),
                        "line": node.lineno,
                        "finding": "negative_shift",
                        "detail": f"shift({periods:g})",
                    }
                )
        if method in {"lead", "forward"}:
            findings.append(
                {
                    "source_path": str(path.resolve()),
                    "line": node.lineno,
                    "finding": "future_method_name",
                    "detail": method,
                }
            )
        if method == "join_asof":
            for keyword in node.keywords:
                if (
                    keyword.arg == "strategy"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value == "forward"
                ):
                    findings.append(
                        {
                            "source_path": str(path.resolve()),
                            "line": node.lineno,
                            "finding": "forward_asof_join",
                            "detail": "join_asof(strategy='forward')",
                        }
                    )
    return findings


def _pl_col_names(tree: ast.AST, function_names: set[str]) -> set[str]:
    names: set[str] = set()
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in function_names
    ]
    for function in functions:
        for node in ast.walk(function):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "col"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "pl"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                names.add(node.args[0].value)
    return names


def audit_selector_sources(
    *, runner_source: Path, profile_source: Path
) -> tuple[pl.DataFrame, dict[str, object]]:
    runner_tree = ast.parse(runner_source.read_text(encoding="utf-8"), str(runner_source))
    profile_tree = ast.parse(profile_source.read_text(encoding="utf-8"), str(profile_source))
    env = {"DAY_KEY_COLS": tuple(CANONICAL_KEY[1:])}
    runner_required = _assignment(runner_tree, "REQUIRED_SELECTOR_COLUMNS", env)
    if not isinstance(runner_required, dict):
        raise ValueError("runner REQUIRED_SELECTOR_COLUMNS is not a static mapping")
    normalized_runner = {
        str(source): tuple(str(column) for column in columns)
        for source, columns in runner_required.items()
    }
    profile_required = _profile_required_columns(profile_tree, env)
    if normalized_runner != profile_required:
        raise ValueError(
            "runner/profile selector input declarations disagree: "
            f"runner={normalized_runner}, profile={profile_required}"
        )
    rows: list[dict[str, object]] = []
    forbidden: list[dict[str, str]] = []
    for source, columns in normalized_runner.items():
        for column in columns:
            is_forbidden, reason = _input_risk(column)
            row = {
                "audit_scope": "raw_required_input",
                "source": source,
                "column": column,
                "declared_in_runner": True,
                "declared_in_profile": True,
                "forbidden_outcome_or_future_name": is_forbidden,
                "risk_reason": reason,
            }
            rows.append(row)
            if is_forbidden:
                forbidden.append({"source": source, "column": column, "reason": str(reason)})
    expression_inventories = {
        "runner_selector_expression": _pl_col_names(
            runner_tree,
            {"build_factorial_gate_frame", "_cell_membership_expression"},
        ),
        "profile_selector_expression": _pl_col_names(
            profile_tree, {"build_filllots_at_high_candidate_frame"}
        ),
    }
    for scope, columns in expression_inventories.items():
        for column in sorted(columns):
            is_forbidden, reason = _input_risk(column)
            rows.append(
                {
                    "audit_scope": scope,
                    "source": "derived_or_joined_frame",
                    "column": column,
                    "declared_in_runner": scope == "runner_selector_expression",
                    "declared_in_profile": scope == "profile_selector_expression",
                    "forbidden_outcome_or_future_name": is_forbidden,
                    "risk_reason": reason,
                }
            )
            if is_forbidden:
                forbidden.append(
                    {
                        "source": scope,
                        "column": column,
                        "reason": str(reason),
                    }
                )
    syntax_findings = [
        *_future_syntax_findings(runner_tree, runner_source),
        *_future_syntax_findings(profile_tree, profile_source),
    ]
    if forbidden or syntax_findings:
        raise ValueError(
            "selector static future/outcome audit failed: "
            f"columns={forbidden}, syntax={syntax_findings}"
        )
    frame = pl.DataFrame(rows).sort("audit_scope", "source", "column")
    summary = {
        "status": "pass",
        "method": (
            "AST equality of runner/profile raw required-column declarations; "
            "column-name denylist; negative shift/lead/forward-asof syntax scan"
        ),
        "raw_input_count": sum(row["audit_scope"] == "raw_required_input" for row in rows),
        "selector_expression_column_count": sum(
            row["audit_scope"] != "raw_required_input" for row in rows
        ),
        "forbidden_input_count": 0,
        "future_syntax_finding_count": 0,
        "sources": {
            "runner": _file_identity(runner_source),
            "profile": _file_identity(profile_source),
        },
        "scope_limit": (
            "syntactic audit of current source; it does not prove upstream feature "
            "producer causality or bind these code bytes to the completed run"
        ),
    }
    return frame, summary


def _support_add(
    accumulators: dict[tuple[str, str, str], SupportAccumulator],
    frame: pl.DataFrame,
    cohort_type: str,
    cohort_id: str,
    stratum: str = "all",
) -> None:
    accumulators.setdefault(
        (cohort_type, cohort_id, stratum), SupportAccumulator()
    ).add(frame)


def _session_support_row(
    frame: pl.DataFrame, *, date: str, parity_status: str
) -> dict[str, object]:
    return {
        "Date": date,
        "membership_validated": True,
        "legacy_parity_status": parity_status,
        "legacy_base_events": len(frame),
        "legacy_base_stocks": len(_unique_quotes(frame)),
        "conservative_base_events": _count_true(frame, "conservative_base"),
        "conservative_base_stocks": len(
            _unique_quotes(frame.filter(pl.col("conservative_base")))
        ),
        "legacy_dgb_events": _count_true(frame, "legacy_dgb"),
        "conservative_dgb_events": _count_true(frame, "conservative_dgb"),
        "base_removed_events": frame.filter(~pl.col("conservative_base")).height,
        "dgb_removed_events": frame.filter(
            pl.col("legacy_dgb") & ~pl.col("conservative_dgb")
        ).height,
        "depth_invalid_events": _count_true(frame, "depth_reference_invalid"),
        "large_fill_reference_invalid_events": _count_true(
            frame, "large_fill_reference_invalid"
        ),
        "md_only_events": frame.filter(
            pl.col("fresh_branch_conservative") == "md_only"
        ).height,
        "large_fill_only_events": frame.filter(
            pl.col("fresh_branch_conservative") == "large_fill_only"
        ).height,
        "md_and_large_fill_events": frame.filter(
            pl.col("fresh_branch_conservative") == "md_and_large_fill"
        ).height,
        "invalid_large_fill_only_events": frame.filter(
            pl.col("fresh_branch_conservative") == "invalid_large_fill_only"
        ).height,
        "conservative_large_fill_buy_events": frame.filter(
            pl.col("large_fill_conservative") & (pl.col("InOut") == 1)
        ).height,
        "conservative_large_fill_sell_events": frame.filter(
            pl.col("large_fill_conservative") & (pl.col("InOut") == -1)
        ).height,
    }


def build_lineage_support_summary(
    *,
    run_dir: Path,
    requested_calendar_path: Path,
    requested_calendar_manifest_path: Path,
    reference_lineage_path: Path,
    reference_lineage_manifest_path: Path,
    runner_source: Path = DEFAULT_RUNNER_SOURCE,
    profile_source: Path = DEFAULT_PROFILE_SOURCE,
) -> tuple[dict[str, object], pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Return fully validated artifacts without writing or reading outcomes."""

    paths = (
        run_dir / "run_manifest.json",
        requested_calendar_path,
        requested_calendar_manifest_path,
        reference_lineage_path,
        reference_lineage_manifest_path,
        runner_source,
        profile_source,
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"lineage/support audit inputs missing: {missing}")
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest = _read_json(run_manifest_path)
    calendar_manifest = _read_json(requested_calendar_manifest_path)
    reference_manifest = _read_json(reference_lineage_manifest_path)
    calendar = pl.read_parquet(requested_calendar_path)
    reference = pl.read_parquet(reference_lineage_path)
    run_sessions, source_identities = _validate_run_manifest(
        run_manifest, reference_lineage_path
    )
    dates = _validate_calendar(
        calendar, calendar_manifest, run_manifest, run_sessions
    )
    full_reference_status_counts = _validate_reference_audit(reference, reference_manifest)
    reference_start = reference_manifest.get("start")
    reference_end = reference_manifest.get("end")
    if (
        not isinstance(reference_start, str)
        or not isinstance(reference_end, str)
        or reference_start > dates[0]
        or reference_end < dates[-1]
    ):
        raise ValueError(
            "reference lineage audit period does not cover the requested calendar: "
            f"reference={reference_start}..{reference_end}, calendar={dates[0]}..{dates[-1]}"
        )
    selector_frame, selector_summary = audit_selector_sources(
        runner_source=runner_source, profile_source=profile_source
    )

    scoped_reference = reference.filter(pl.col("Date").is_in(dates)) if not reference.is_empty() else reference
    scoped_reference_status_rows: list[dict[str, object]] = []
    for column, status_column in zip(REFERENCE_COLUMNS, REFERENCE_STATUS_COLUMNS):
        counts = _value_counts(scoped_reference, status_column)
        for status in sorted(REFERENCE_STATUSES | set(counts)):
            scoped_reference_status_rows.append(
                {
                    "reference_column": column,
                    "lineage_status": status,
                    "stock_sessions": counts.get(status, 0),
                    "scope": "requested_calendar",
                }
            )
    reference_status_frame = pl.DataFrame(scoped_reference_status_rows).sort(
        "reference_column", "lineage_status"
    )
    reference_by_date = {
        str(date): scoped_reference.filter(pl.col("Date") == date)
        for date in scoped_reference["Date"].unique().to_list()
    } if not scoped_reference.is_empty() else {}

    accumulators: dict[tuple[str, str, str], SupportAccumulator] = {}
    daily_rows: list[dict[str, object]] = []
    parity_counts: dict[str, int] = {}
    successful_dates = set(
        calendar.filter(pl.col("run_status") == "success")["Date"].to_list()
    )
    daily_files = {path.stem: path for path in (run_dir / "membership_daily").glob("*.parquet")}
    lineage_files = {path.stem: path for path in (run_dir / "membership_lineage").glob("*.json")}
    missing_membership = sorted(
        date for date in successful_dates if date not in daily_files or date not in lineage_files
    )
    if missing_membership:
        raise ValueError(f"successful sessions lack membership artifacts: {missing_membership}")

    for calendar_row in calendar.iter_rows(named=True):
        date = str(calendar_row["Date"])
        base_row = dict(calendar_row)
        if date not in successful_dates:
            base_row.update(
                {
                    "membership_validated": False,
                    "legacy_parity_status": None,
                    "legacy_base_events": None,
                    "legacy_base_stocks": None,
                    "conservative_base_events": None,
                    "conservative_base_stocks": None,
                    "legacy_dgb_events": None,
                    "conservative_dgb_events": None,
                    "base_removed_events": None,
                    "dgb_removed_events": None,
                    "depth_invalid_events": None,
                    "large_fill_reference_invalid_events": None,
                    "md_only_events": None,
                    "large_fill_only_events": None,
                    "md_and_large_fill_events": None,
                    "invalid_large_fill_only_events": None,
                    "conservative_large_fill_buy_events": None,
                    "conservative_large_fill_sell_events": None,
                }
            )
            daily_rows.append(base_row)
            continue
        if date not in source_identities:
            raise ValueError(f"{date}: successful session lacks run source identities")
        membership_path = daily_files[date]
        lineage_path = lineage_files[date]
        frame = pl.read_parquet(membership_path)
        _validate_membership_logic(frame, date)
        lineage = _read_json(lineage_path)
        parity_status = _validate_membership_lineage(
            date=date,
            frame=frame,
            lineage=lineage,
            membership_path=membership_path,
            reference_path=reference_lineage_path,
            run_manifest=run_manifest,
            source_identities=source_identities[date],
        )
        parity_counts[parity_status] = parity_counts.get(parity_status, 0) + 1
        reference_day = reference_by_date.get(date)
        if reference_day is not None and not reference_day.is_empty():
            joined = frame.join(
                reference_day.select("QuoteCode", *REFERENCE_STATUS_COLUMNS),
                on="QuoteCode",
                how="inner",
                validate="m:1",
                suffix="_reference_audit",
            )
            for status_column in REFERENCE_STATUS_COLUMNS:
                audit_column = f"{status_column}_reference_audit"
                if joined.filter(
                    ~pl.col(status_column).eq_missing(pl.col(audit_column))
                ).height:
                    raise ValueError(f"{date}: membership/reference audit status mismatch for {status_column}")

        _support_add(accumulators, frame, "base", "legacy_base")
        _support_add(
            accumulators,
            frame.filter(pl.col("conservative_base")),
            "base",
            "conservative_base",
        )
        _support_add(accumulators, frame.filter(pl.col("legacy_dgb")), "anchor", "legacy_dgb")
        _support_add(
            accumulators,
            frame.filter(pl.col("conservative_dgb")),
            "anchor",
            "conservative_dgb",
        )
        _support_add(
            accumulators,
            frame.filter(~pl.col("conservative_base")),
            "removal",
            "legacy_base_removed_by_reference_validation",
        )
        _support_add(
            accumulators,
            frame.filter(pl.col("legacy_dgb") & ~pl.col("conservative_dgb")),
            "removal",
            "legacy_dgb_removed_by_reference_validation",
        )
        for index, candidate_id in enumerate(CELL_IDS):
            _support_add(
                accumulators,
                frame.filter((pl.col("membership_mask") & (1 << index)) != 0),
                "factorial_cell",
                candidate_id,
            )
        for branch in sorted(BRANCHES):
            branch_frame = frame.filter(pl.col("fresh_branch_conservative") == branch)
            _support_add(accumulators, branch_frame, "fresh_branch", branch)
            if branch in {"large_fill_only", "md_and_large_fill"}:
                for side, side_name in ((-1, "sell"), (1, "buy")):
                    _support_add(
                        accumulators,
                        branch_frame.filter(pl.col("InOut") == side),
                        "fresh_branch_large_fill_side",
                        branch,
                        side_name,
                    )
        for side, side_name in ((-1, "sell"), (1, "buy")):
            _support_add(
                accumulators,
                frame.filter(pl.col("_large_fill") & (pl.col("InOut") == side)),
                "legacy_large_fill_side",
                "legacy_large_fill",
                side_name,
            )
            _support_add(
                accumulators,
                frame.filter(
                    pl.col("large_fill_conservative") & (pl.col("InOut") == side)
                ),
                "conservative_large_fill_side",
                "conservative_large_fill",
                side_name,
            )
            _support_add(
                accumulators,
                frame.filter(
                    pl.col("_large_fill")
                    & ~pl.col("large_fill_conservative")
                    & (pl.col("InOut") == side)
                ),
                "invalid_large_fill_side",
                "invalid_large_fill",
                side_name,
            )
        for column in (
            "depth_reference_invalid",
            "large_fill_reference_invalid",
            "avg_bid_ambiguous_one",
            "avg_ask_ambiguous_one",
            "big_buy_ambiguous_one",
            "big_sell_ambiguous_one",
        ):
            _support_add(
                accumulators,
                frame.filter(pl.col(column)),
                "reference_exposure",
                column,
            )
        base_row.update(_session_support_row(frame, date=date, parity_status=parity_status))
        daily_rows.append(base_row)

    support_rows = [
        accumulator.row(cohort_type=key[0], cohort_id=key[1], stratum=key[2])
        for key, accumulator in sorted(accumulators.items())
    ]
    support_frame = pl.DataFrame(support_rows).sort("cohort_type", "cohort_id", "stratum")
    session_frame = pl.DataFrame(daily_rows).sort("Date")
    status_counts = {
        str(row["session_status"]): int(row["len"])
        for row in calendar.group_by("session_status").len().iter_rows(named=True)
    }
    run_status_counts = {
        str(row["run_status"]): int(row["len"])
        for row in calendar.group_by("run_status").len().iter_rows(named=True)
    }
    support_lookup = {
        (row["cohort_type"], row["cohort_id"], row["stratum"]): row
        for row in support_rows
    }
    def support(cohort_type: str, cohort_id: str, stratum: str = "all") -> dict[str, object]:
        row = support_lookup.get((cohort_type, cohort_id, stratum))
        if row is not None:
            return {
                key: row[key]
                for key in ("events", "stock_days", "stocks", "active_sessions")
            }
        return {"events": 0, "stock_days": 0, "stocks": 0, "active_sessions": 0}

    extra_daily_dates = sorted(set(daily_files) - set(dates))
    extra_lineage_dates = sorted(set(lineage_files) - set(dates))
    summary: dict[str, object] = {
        "contract": "fh_neg_rejection_lineage_support_audit/v1",
        "factorial_contract": CONTRACT_ID,
        "factorial_family_id": FAMILY_ID,
        "factorial_family_version": FAMILY_VERSION,
        "canonical_key": list(CANONICAL_KEY),
        "status": "pass",
        "outcome_artifacts_read": [],
        "outcome_columns_read": [],
        "support_semantics": (
            "raw selector membership on successful run sessions; not label-valid "
            "support and not a performance estimand"
        ),
        "scope": {
            "start": dates[0],
            "end": dates[-1],
            "evidence_class": calendar["evidence_class"][0],
            "requested_sessions": len(calendar),
            "successful_sessions": len(successful_dates),
            "session_status_counts": status_counts,
            "run_status_counts": run_status_counts,
            "extra_membership_daily_dates_outside_calendar_not_read": extra_daily_dates,
            "extra_membership_lineage_dates_outside_calendar_not_read": extra_lineage_dates,
        },
        "input_identities": {
            "run_manifest": _file_identity(run_manifest_path),
            "requested_calendar": _file_identity(requested_calendar_path),
            "requested_calendar_manifest": _file_identity(requested_calendar_manifest_path),
            "reference_lineage": _file_identity(reference_lineage_path),
            "reference_lineage_manifest": _file_identity(reference_lineage_manifest_path),
        },
        "legacy_parity": {
            "status_counts": parity_counts,
            "all_applicable_sessions_passed": set(parity_counts) <= {
                "pass", "not_applicable_outside_legacy_artifact"
            },
        },
        "support": {
            "legacy_base": support("base", "legacy_base"),
            "conservative_base": support("base", "conservative_base"),
            "legacy_dgb": support("anchor", "legacy_dgb"),
            "conservative_dgb": support("anchor", "conservative_dgb"),
            "factorial_cells": {
                candidate_id: support("factorial_cell", candidate_id)
                for candidate_id in CELL_IDS
            },
        },
        "conservative_reference_removals": {
            "base": support("removal", "legacy_base_removed_by_reference_validation"),
            "dgb": support("removal", "legacy_dgb_removed_by_reference_validation"),
        },
        "invalid_reference_exposure": {
            column: support("reference_exposure", column)
            for column in (
                "depth_reference_invalid",
                "large_fill_reference_invalid",
                "avg_bid_ambiguous_one",
                "avg_ask_ambiguous_one",
                "big_buy_ambiguous_one",
                "big_sell_ambiguous_one",
            )
        },
        "fresh_branches": {
            branch: {
                "all": support("fresh_branch", branch),
                **(
                    {
                        "large_fill_side": {
                            side: support("fresh_branch_large_fill_side", branch, side)
                            for side in ("buy", "sell")
                        }
                    }
                    if branch in {"large_fill_only", "md_and_large_fill"}
                    else {}
                ),
            }
            for branch in sorted(BRANCHES)
        },
        "large_fill_side": {
            "legacy": {
                side: support("legacy_large_fill_side", "legacy_large_fill", side)
                for side in ("buy", "sell")
            },
            "conservative": {
                side: support(
                    "conservative_large_fill_side", "conservative_large_fill", side
                )
                for side in ("buy", "sell")
            },
            "removed_by_reference_validation": {
                side: support("invalid_large_fill_side", "invalid_large_fill", side)
                for side in ("buy", "sell")
            },
        },
        "reference_audit": {
            "full_artifact_status_counts": full_reference_status_counts,
            "scoped_ambiguous_stock_sessions": len(scoped_reference),
            "scoped_invalid_or_mismatched_stock_sessions": (
                _count_true(scoped_reference, "has_invalid_ambiguous_reference")
                if not scoped_reference.is_empty()
                else 0
            ),
        },
        "selector_static_audit": selector_summary,
        "validation_gaps": [
            {
                "id": "calendar_not_exchange_authoritative",
                "detail": str(calendar_manifest.get("coverage_limitation")),
            },
            {
                "id": "raw_source_identity_has_no_content_hash",
                "detail": (
                    "run source identities bind path/size/mtime_ns, not raw-source SHA-256; "
                    "the lineage hash only hashes that metadata payload"
                ),
            },
            {
                "id": "selector_code_not_bound_to_run_manifest",
                "detail": (
                    "the static audit hashes current runner/profile source, but the run "
                    "manifest does not store those source hashes"
                ),
            },
            {
                "id": "upstream_feature_causality_not_proven_by_ast",
                "detail": (
                    "the AST audit rejects obvious future names/operations but cannot prove "
                    "the producer semantics of tickFeature columns"
                ),
            },
            {
                "id": "reference_audit_only_ambiguous_stored_one",
                "detail": (
                    "the reference reconstruction audits producer-sanitized value 1; it "
                    "does not reconstruct every non-one reference"
                ),
            },
            {
                "id": "support_is_pre_label_membership",
                "detail": (
                    "membership support precedes shared-pool and label-valid filtering; "
                    "condition_n in outcome aggregates may be smaller"
                ),
            },
        ],
    }
    return summary, session_frame, support_frame, reference_status_frame, selector_frame


def write_outputs(
    *,
    output_dir: Path,
    summary: Mapping[str, object],
    session_frame: pl.DataFrame,
    support_frame: pl.DataFrame,
    reference_status_frame: pl.DataFrame,
    selector_frame: pl.DataFrame,
) -> None:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(summary, output_dir / "lineage_support_summary.json")
    _atomic_write_parquet(session_frame, output_dir / "session_support.parquet")
    _atomic_write_parquet(support_frame, output_dir / "cohort_support.parquet")
    _atomic_write_parquet(reference_status_frame, output_dir / "reference_status.parquet")
    _atomic_write_parquet(selector_frame, output_dir / "selector_input_audit.parquet")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate completed fh_neg_rejection lineage and raw support"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--requested-calendar", type=Path, required=True)
    parser.add_argument("--requested-calendar-manifest", type=Path, required=True)
    parser.add_argument("--reference-lineage", type=Path, required=True)
    parser.add_argument("--reference-lineage-manifest", type=Path, required=True)
    parser.add_argument("--runner-source", type=Path, default=DEFAULT_RUNNER_SOURCE)
    parser.add_argument("--profile-source", type=Path, default=DEFAULT_PROFILE_SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    run_dir = args.run_dir.resolve()
    if output_dir == run_dir or run_dir in output_dir.parents:
        raise ValueError("lineage/support output must be outside the immutable run directory")
    results = build_lineage_support_summary(
        run_dir=run_dir,
        requested_calendar_path=args.requested_calendar.resolve(),
        requested_calendar_manifest_path=args.requested_calendar_manifest.resolve(),
        reference_lineage_path=args.reference_lineage.resolve(),
        reference_lineage_manifest_path=args.reference_lineage_manifest.resolve(),
        runner_source=args.runner_source.resolve(),
        profile_source=args.profile_source.resolve(),
    )
    write_outputs(
        output_dir=output_dir,
        summary=results[0],
        session_frame=results[1],
        support_frame=results[2],
        reference_status_frame=results[3],
        selector_frame=results[4],
    )
    print(json.dumps(results[0], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
