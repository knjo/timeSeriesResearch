"""Run the frozen 27-cell fh_neg_rejection threshold plateau.

The selector never rebuilds FillLots-at-high features.  It reads the completed
parent D/G/B compact membership surface, validates that surface and its lineage
against the current four raw-source identities, then applies the frozen
threshold grid to the conservative base.  The generic sweep runner continues
to own the full eligible pool, label, shared control, and daily aggregates.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Mapping, Sequence

import polars as pl

from src.research.intradaySignal.framework.events import (
    DAY_KEY_COLS,
    DataRoots,
    EventDaySources,
)


STUDY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STUDY_DIR.parents[4]
RUNNER_PATH = Path(__file__).resolve()
ARTIFACT_ROOT = STUDY_DIR / "artifacts"
DEFAULT_OUTPUT_DIR = ARTIFACT_ROOT / "threshold_plateau_v1"
CONTRACT_PATH = STUDY_DIR / "threshold_plateau_contract.json"
PARENT_CONTRACT_PATH = STUDY_DIR / "study_contract.json"
PARENT_RUN_DIR = ARTIFACT_ROOT / "dgb_factorial_v1"
PARENT_RUN_MANIFEST_PATH = PARENT_RUN_DIR / "run_manifest.json"
REFERENCE_LINEAGE_PATH = ARTIFACT_ROOT / "reference_lineage_v1/reference_lineage.parquet"
COMMON_RUNNER_PATH = (
    STUDY_DIR.parent / "20260825_filllots_at_low_framework" / "run_sweep.py"
)

CONTRACT_BYTES = CONTRACT_PATH.read_bytes()
CONTRACT = json.loads(CONTRACT_BYTES)
CONTRACT_SHA256 = hashlib.sha256(CONTRACT_BYTES).hexdigest()
CONTRACT_ID = "fh_neg_rejection_threshold_plateau/fixed_grid_v1"
FAMILY_ID = "fh_neg_rejection_threshold_plateau"
FAMILY_VERSION = "1"
REGISTRY_ID = "fh-neg-rejection-threshold-plateau-v1"
PARENT_CONTRACT_ID = "fh_neg_rejection_robustness/fixed_factorial_v1"
PARENT_FAMILY_ID = "fh_neg_rejection_dgb_factorial"
PARENT_FAMILY_VERSION = "1"
PARENT_CELL_IDS = (
    "d0_g0_b0",
    "d1_g0_b0",
    "d0_g1_b0",
    "d0_g0_b1",
    "d1_g1_b0",
    "d1_g0_b1",
    "d0_g1_b1",
    "d1_g1_b1",
)
PARENT_CONTRACT_SHA256 = hashlib.sha256(PARENT_CONTRACT_PATH.read_bytes()).hexdigest()
REQUIRED_SOURCES = (
    "tick_data",
    "tick_feature",
    "pre_market",
    "txf_tick",
)
SOURCE_PATTERNS = {
    "tick_data": "*_StockTick.parquet",
    "tick_feature": "*_tickFeature.parquet",
    "pre_market": "*_preMarketData.parquet",
    "txf_tick": "*_TxfTick.parquet",
}
REQUIRED_MEMBERSHIP_COLUMNS = (
    "Date",
    *DAY_KEY_COLS,
    "conservative_base",
    "conservative_dgb",
    "filllots_at_high_depth_ratio",
    "filllots_at_high_gap_ticks",
    "bear_score",
    "depth_reference_invalid",
)
CONTRACT_START, CONTRACT_END = tuple(CONTRACT["date_range"])
LABEL_PROFILE = "short-taker-stop108-v1"


@dataclass(frozen=True)
class ThresholdCell:
    candidate_id: str
    depth_ratio_lte: float
    gap_ticks_gte: float
    bear_score_gte: int

    @property
    def description(self) -> str:
        return (
            "conservative_base and valid depth reference; "
            f"depth_ratio<={self.depth_ratio_lte:g}; "
            f"gap_ticks>={self.gap_ticks_gte:g}; "
            f"bear_score>={self.bear_score_gte}"
        )


def _candidate_id(depth: float, gap: float, bear: int) -> str:
    depth_magnitude = int(abs(depth))
    return f"d{depth_magnitude:02d}_g{int(gap):02d}_b{bear}"


def _validated_contract_grid() -> tuple[ThresholdCell, ...]:
    expected = {
        "contract_id": CONTRACT_ID,
        "parent_contract_id": PARENT_CONTRACT_ID,
        "status": "frozen_after_stage1_outcomes_before_any_threshold_plateau_outcome",
        "evidence_class": "retrospective_diagnostic",
        "production_promotion_authorized": False,
    }
    mismatches = {
        key: {"expected": value, "actual": CONTRACT.get(key)}
        for key, value in expected.items()
        if CONTRACT.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"unexpected frozen plateau contract: {mismatches}")
    if CONTRACT.get("outcome_exposure_before_freeze", {}).get(
        "threshold_plateau_outcomes_seen"
    ) is not False:
        raise RuntimeError("plateau contract no-outcome freeze marker is absent")
    grid = CONTRACT["fixed_grid"]
    depths = tuple(float(value) for value in grid["depth_ratio_lte"])
    gaps = tuple(float(value) for value in grid["gap_ticks_gte"])
    bears = tuple(int(value) for value in grid["bear_score_gte"])
    if depths != (-6.0, -8.0, -10.0):
        raise RuntimeError(f"unexpected frozen depth grid: {depths}")
    if gaps != (3.0, 4.0, 5.0):
        raise RuntimeError(f"unexpected frozen gap grid: {gaps}")
    if bears != (1, 2, 3):
        raise RuntimeError(f"unexpected frozen bear grid: {bears}")
    cells = tuple(
        ThresholdCell(
            candidate_id=_candidate_id(depth, gap, bear),
            depth_ratio_lte=depth,
            gap_ticks_gte=gap,
            bear_score_gte=bear,
        )
        for depth, gap, bear in product(depths, gaps, bears)
    )
    if len(cells) != int(grid["cartesian_cells"]) or len(cells) != 27:
        raise RuntimeError("frozen threshold grid is not the declared 27 cells")
    if len({cell.candidate_id for cell in cells}) != len(cells):
        raise RuntimeError("threshold candidate ids are not unique")
    center = CONTRACT["shared_semantics"]["center"]
    if center != {
        "depth_ratio_lte": -8,
        "gap_ticks_gte": 4,
        "bear_score_gte": 2,
    }:
        raise RuntimeError(f"unexpected frozen center: {center}")
    return cells


THRESHOLD_CELLS = _validated_contract_grid()
CELL_IDS = tuple(cell.candidate_id for cell in THRESHOLD_CELLS)
CELL_BY_ID = {cell.candidate_id: cell for cell in THRESHOLD_CELLS}
CENTER_CELL_ID = _candidate_id(-8.0, 4.0, 2)
CELL_DESCRIPTIONS = {
    cell.candidate_id: cell.description for cell in THRESHOLD_CELLS
}
DESIGN_PAYLOAD = {
    "contract": CONTRACT_ID,
    "frozen_contract_sha256": CONTRACT_SHA256,
    "parent_contract": PARENT_CONTRACT_ID,
    "parent_contract_sha256": PARENT_CONTRACT_SHA256,
    "membership_semantics": (
        "conservative_base AND NOT depth_reference_invalid; null gates fail "
        "closed; center keys must equal parent conservative_dgb"
    ),
    "center_cell_id": CENTER_CELL_ID,
    "cells": [
        {
            "candidate_id": cell.candidate_id,
            "depth_ratio_lte": cell.depth_ratio_lte,
            "gap_ticks_gte": cell.gap_ticks_gte,
            "bear_score_gte": cell.bear_score_gte,
        }
        for cell in THRESHOLD_CELLS
    ],
}
DESIGN_FINGERPRINT = hashlib.sha256(
    json.dumps(
        DESIGN_PAYLOAD,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


def _load_common_runner():
    spec = importlib.util.spec_from_file_location(
        "_fh_neg_rejection_plateau_common_runner",
        COMMON_RUNNER_PATH,
    )
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"cannot load common runner: {COMMON_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


COMMON = _load_common_runner()
PLATEAU_REGISTRY = COMMON.CandidateRegistry(
    registry_id=REGISTRY_ID,
    contract_id=CONTRACT_ID,
    family_id=FAMILY_ID,
    family_version=FAMILY_VERSION,
    candidate_ids=CELL_IDS,
    descriptions=CELL_DESCRIPTIONS,
)
_COMMON_EVALUATE_DAY = COMMON.evaluate_day
_COMMON_VALIDATE_CACHED_DAY = COMMON._validate_cached_day


@dataclass(frozen=True)
class ParentBinding:
    run_dir: Path
    manifest: dict[str, object]
    manifest_identity: dict[str, object]
    reference_lineage_identity: dict[str, object]
    sessions: frozenset[str]
    failures: frozenset[str]


@dataclass
class RunContext:
    output_dir: Path
    parent: ParentBinding
    membership_audits: dict[str, dict[str, object]] = field(default_factory=dict)


_RUN_CONTEXT: RunContext | None = None


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


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _validate_date(value: str, label: str) -> None:
    try:
        parsed = datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise ValueError(f"{label} must be a valid YYYYMMDD date") from exc
    if parsed.strftime("%Y%m%d") != value:
        raise ValueError(f"{label} must be exactly eight YYYYMMDD digits")


def _source_dates(directory: Path, pattern: str) -> set[str]:
    return {path.name[:8] for path in directory.glob(pattern)}


def available_sessions(
    roots: DataRoots,
    *,
    start: str,
    end: str,
) -> list[str]:
    calendars = [
        _source_dates(roots.source_dir(source), SOURCE_PATTERNS[source])
        for source in REQUIRED_SOURCES
    ]
    common = set.intersection(*calendars)
    return sorted(date for date in common if start <= date <= end)


def _current_source_identities(
    date: str,
    roots: DataRoots,
) -> dict[str, dict[str, object]]:
    day = EventDaySources(date, roots)
    day.require(*REQUIRED_SOURCES)
    identities: dict[str, dict[str, object]] = {}
    for source in REQUIRED_SOURCES:
        path = day.path(source)
        stat = path.stat()
        identities[source] = {
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return identities


def _current_source_identities_json(date: str, roots: DataRoots) -> str:
    return _canonical_json(_current_source_identities(date, roots))


def _validate_parent_factorial_design(manifest: dict[str, object]) -> None:
    design = manifest.get("factorial_design")
    fingerprint = manifest.get("factorial_design_fingerprint")
    if not isinstance(design, dict) or not isinstance(fingerprint, str):
        raise ValueError("parent manifest lacks factorial design identity")
    actual = hashlib.sha256(_canonical_json(design).encode("utf-8")).hexdigest()
    if actual != fingerprint:
        raise ValueError("parent factorial design fingerprint is invalid")
    if design.get("contract") != PARENT_CONTRACT_ID:
        raise ValueError("parent factorial design contract drifted")
    if design.get("frozen_contract_sha256") != PARENT_CONTRACT_SHA256:
        raise ValueError("parent factorial design is not bound to study_contract.json")


def load_parent_binding(parent_run_dir: Path = PARENT_RUN_DIR) -> ParentBinding:
    run_dir = Path(parent_run_dir).resolve()
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"parent run manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "contract": PARENT_CONTRACT_ID,
        "family_id": PARENT_FAMILY_ID,
        "family_version": PARENT_FAMILY_VERSION,
        "frozen_contract_sha256": PARENT_CONTRACT_SHA256,
        "candidate_ids": list(PARENT_CELL_IDS),
        "primary_membership": "conservative_reference_validated",
        "step": 1,
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"parent run manifest identity mismatch: {mismatches}")
    _validate_parent_factorial_design(manifest)

    sessions_value = manifest.get("sessions")
    failures_value = manifest.get("failures")
    if not isinstance(sessions_value, list) or not all(
        isinstance(value, str) for value in sessions_value
    ):
        raise ValueError("parent manifest sessions must be a string list")
    if sessions_value != sorted(set(sessions_value)):
        raise ValueError("parent manifest sessions must be unique and sorted")
    if not isinstance(failures_value, list):
        raise ValueError("parent manifest failures must be a list")
    failures: set[str] = set()
    for item in failures_value:
        if not isinstance(item, dict) or not isinstance(item.get("Date"), str):
            raise ValueError("parent failure entries must contain Date")
        failures.add(str(item["Date"]))
    plateau_failures = {
        date for date in failures if CONTRACT_START <= date <= CONTRACT_END
    }
    if plateau_failures:
        raise ValueError(
            f"parent run has failures in plateau range: {sorted(plateau_failures)}"
        )

    source_identities = manifest.get("source_identities")
    if not isinstance(source_identities, dict):
        raise ValueError("parent manifest lacks per-session source identities")
    missing_source_lineage = sorted(set(sessions_value) - set(source_identities))
    if missing_source_lineage:
        raise ValueError(
            "parent manifest lacks source identities for sessions: "
            f"{missing_source_lineage[:5]}"
        )
    for date in sessions_value:
        identities = source_identities[date]
        if not isinstance(identities, dict) or set(identities) != set(REQUIRED_SOURCES):
            raise ValueError(f"parent {date} source identity family is incomplete")

    if not REFERENCE_LINEAGE_PATH.is_file():
        raise FileNotFoundError(
            f"reference lineage artifact is missing: {REFERENCE_LINEAGE_PATH}"
        )
    reference_identity = _file_identity(REFERENCE_LINEAGE_PATH)
    if manifest.get("reference_lineage_artifact") != reference_identity:
        raise ValueError("parent manifest reference-lineage identity drifted")
    return ParentBinding(
        run_dir=run_dir,
        manifest=manifest,
        manifest_identity=_file_identity(manifest_path),
        reference_lineage_identity=reference_identity,
        sessions=frozenset(sessions_value),
        failures=frozenset(failures),
    )


def validate_parent_calendar(
    binding: ParentBinding,
    *,
    roots: DataRoots,
    start: str,
    end: str,
) -> list[str]:
    sessions = available_sessions(roots, start=start, end=end)
    parent_sessions = sorted(
        date for date in binding.sessions if start <= date <= end
    )
    if sessions != parent_sessions:
        raise ValueError(
            "parent sessions disagree with current four-source calendar: "
            f"parent_only={sorted(set(parent_sessions) - set(sessions))[:5]}, "
            f"source_only={sorted(set(sessions) - set(parent_sessions))[:5]}"
        )
    membership_dates = {
        path.stem for path in (binding.run_dir / "membership_daily").glob("*.parquet")
    }
    lineage_dates = {
        path.stem for path in (binding.run_dir / "membership_lineage").glob("*.json")
    }
    missing_membership = sorted(set(sessions) - membership_dates)
    missing_lineage = sorted(set(sessions) - lineage_dates)
    if missing_membership or missing_lineage:
        raise ValueError(
            "parent membership cache is incomplete: "
            f"membership={missing_membership[:5]}, lineage={missing_lineage[:5]}"
        )
    return sessions


def _parent_membership_paths(
    binding: ParentBinding,
    date: str,
) -> tuple[Path, Path]:
    return (
        binding.run_dir / "membership_daily" / f"{date}.parquet",
        binding.run_dir / "membership_lineage" / f"{date}.json",
    )


def validate_parent_membership_day(
    *,
    date: str,
    roots: DataRoots,
    binding: ParentBinding,
) -> dict[str, object]:
    if date not in binding.sessions:
        raise ValueError(f"{date}: absent from completed parent run sessions")
    if date in binding.failures:
        raise ValueError(f"{date}: parent run recorded a failure")
    manifest_path = binding.run_dir / "run_manifest.json"
    if _file_identity(manifest_path) != binding.manifest_identity:
        raise ValueError(f"{date}: parent run manifest changed during plateau run")
    if _file_identity(REFERENCE_LINEAGE_PATH) != binding.reference_lineage_identity:
        raise ValueError(f"{date}: frozen reference-lineage artifact drifted")

    membership_path, lineage_path = _parent_membership_paths(binding, date)
    if not membership_path.is_file() or not lineage_path.is_file():
        raise ValueError(f"{date}: parent membership or lineage file is missing")
    membership_identity = _file_identity(membership_path)
    lineage_identity = _file_identity(lineage_path)
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    current_sources = _current_source_identities(date, roots)
    current_sources_json = _canonical_json(current_sources)
    expected = {
        "contract": PARENT_CONTRACT_ID,
        "frozen_contract_sha256": PARENT_CONTRACT_SHA256,
        "design_fingerprint": binding.manifest["factorial_design_fingerprint"],
        "family_id": PARENT_FAMILY_ID,
        "family_version": PARENT_FAMILY_VERSION,
        "Date": date,
        "source_identities": current_sources,
        "source_identities_sha256": _sha256_text(current_sources_json),
        "membership_artifact": membership_identity,
        "reference_lineage_artifact": binding.reference_lineage_identity,
    }
    mismatches = {
        key: {"expected": value, "actual": lineage.get(key)}
        for key, value in expected.items()
        if lineage.get(key) != value
    }
    parent_sources = binding.manifest["source_identities"].get(date)
    if parent_sources != current_sources:
        mismatches["parent_manifest_source_identities"] = {
            "expected": current_sources,
            "actual": parent_sources,
        }
    if mismatches:
        raise ValueError(f"{date}: parent membership lineage mismatch: {mismatches}")

    schema = pl.read_parquet_schema(membership_path)
    missing_columns = sorted(set(REQUIRED_MEMBERSHIP_COLUMNS) - set(schema.names()))
    if missing_columns:
        raise ValueError(
            f"{date}: parent membership missing columns: {missing_columns}"
        )
    return {
        "Date": date,
        "parent_manifest_sha256": binding.manifest_identity["sha256"],
        "parent_membership_sha256": membership_identity["sha256"],
        "parent_membership_lineage_sha256": lineage_identity["sha256"],
        "parent_source_identities_sha256": _sha256_text(current_sources_json),
        "membership_path": str(membership_path.resolve()),
        "lineage_path": str(lineage_path.resolve()),
    }


def load_parent_membership_frame(
    *,
    date: str,
    roots: DataRoots,
    binding: ParentBinding,
) -> tuple[pl.DataFrame, dict[str, object]]:
    audit = validate_parent_membership_day(
        date=date,
        roots=roots,
        binding=binding,
    )
    membership_path, _ = _parent_membership_paths(binding, date)
    membership = pl.read_parquet(membership_path).select(
        *REQUIRED_MEMBERSHIP_COLUMNS
    )
    if membership.filter(pl.col("Date") != date).height:
        raise ValueError(f"{date}: parent membership contains another Date")
    if membership.select(
        pl.struct(list(DAY_KEY_COLS)).is_duplicated().any()
    ).item():
        raise ValueError(f"{date}: parent membership has duplicate event keys")
    return membership, audit


def _membership_expression(cell: ThresholdCell) -> pl.Expr:
    return (
        pl.col("conservative_base").fill_null(False)
        & ~pl.col("depth_reference_invalid").fill_null(True)
        & (
            pl.col("filllots_at_high_depth_ratio") <= cell.depth_ratio_lte
        ).fill_null(False)
        & (
            pl.col("filllots_at_high_gap_ticks") >= cell.gap_ticks_gte
        ).fill_null(False)
        & (pl.col("bear_score") >= cell.bear_score_gte).fill_null(False)
    )


def select_threshold_candidates_from_membership(
    membership: pl.DataFrame,
    *,
    candidate_ids: Sequence[str] = CELL_IDS,
) -> Mapping[str, pl.DataFrame]:
    missing_columns = sorted(
        set(REQUIRED_MEMBERSHIP_COLUMNS) - set(membership.columns)
    )
    if missing_columns:
        raise ValueError(f"membership frame missing columns: {missing_columns}")
    selected_ids = tuple(candidate_ids)
    unknown = sorted(set(selected_ids) - set(CELL_IDS))
    if unknown:
        raise ValueError(f"unknown threshold cells: {unknown}")
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("threshold candidate_ids cannot contain duplicates")
    if membership.select(
        pl.struct(list(DAY_KEY_COLS)).is_duplicated().any()
    ).item():
        raise ValueError("membership frame contains duplicate event keys")
    selected = {
        candidate_id: (
            membership.filter(_membership_expression(CELL_BY_ID[candidate_id]))
            .select(*DAY_KEY_COLS)
            .sort(list(DAY_KEY_COLS))
        )
        for candidate_id in selected_ids
    }
    if CENTER_CELL_ID in selected:
        expected_center = (
            membership.filter(pl.col("conservative_dgb").fill_null(False))
            .select(*DAY_KEY_COLS)
            .sort(list(DAY_KEY_COLS))
        )
        if not selected[CENTER_CELL_ID].equals(expected_center):
            raise ValueError(
                "center threshold keys differ from parent conservative_dgb"
            )
    return selected


def _active_parent_binding() -> ParentBinding:
    if _RUN_CONTEXT is None:
        return load_parent_binding()
    return _RUN_CONTEXT.parent


def _select_candidates(
    day: EventDaySources,
    *,
    registry,
    candidate_ids: tuple[str, ...],
) -> dict[str, pl.DataFrame]:
    if registry.registry_id != REGISTRY_ID:
        raise ValueError(f"unexpected threshold registry: {registry.registry_id}")
    day.require(*REQUIRED_SOURCES)
    membership, audit = load_parent_membership_frame(
        date=day.date,
        roots=day.roots,
        binding=_active_parent_binding(),
    )
    if _RUN_CONTEXT is not None:
        _RUN_CONTEXT.membership_audits[day.date] = audit
    return dict(
        select_threshold_candidates_from_membership(
            membership,
            candidate_ids=candidate_ids,
        )
    )


def _runner_sha256() -> str:
    return _sha256_file(RUNNER_PATH)


def _plateau_cache_metadata(audit: Mapping[str, object]) -> dict[str, str]:
    return {
        "plateau_contract_sha256": CONTRACT_SHA256,
        "plateau_design_fingerprint": DESIGN_FINGERPRINT,
        "plateau_runner_sha256": _runner_sha256(),
        "parent_run_manifest_sha256": str(audit["parent_manifest_sha256"]),
        "parent_membership_sha256": str(audit["parent_membership_sha256"]),
        "parent_membership_lineage_sha256": str(
            audit["parent_membership_lineage_sha256"]
        ),
        "parent_source_identities_sha256": str(
            audit["parent_source_identities_sha256"]
        ),
    }


def _require_uniform_plateau_metadata(
    frame: pl.DataFrame,
    *,
    date: str,
    audit: Mapping[str, object],
) -> None:
    for column, expected in _plateau_cache_metadata(audit).items():
        if column not in frame.columns:
            raise ValueError(f"cached {date}: missing plateau metadata {column}")
        actual = frame[column].unique().to_list()
        if actual != [expected]:
            raise ValueError(
                f"cached {date}: {column}={actual}, expected {expected!r}"
            )


def _evaluate_day_with_plateau_lineage(*args, **kwargs) -> pl.DataFrame:
    frame = _COMMON_EVALUATE_DAY(*args, **kwargs)
    date = str(kwargs.get("date", args[0] if args else ""))
    if _RUN_CONTEXT is None or date not in _RUN_CONTEXT.membership_audits:
        raise RuntimeError(f"{date}: plateau membership audit was not recorded")
    metadata = _plateau_cache_metadata(
        _RUN_CONTEXT.membership_audits[date]
    )
    return frame.with_columns(
        *[pl.lit(value).alias(column) for column, value in metadata.items()]
    )


def _validate_cached_day_with_plateau_lineage(
    frame: pl.DataFrame,
    **kwargs,
) -> None:
    _COMMON_VALIDATE_CACHED_DAY(frame, **kwargs)
    date = str(kwargs["date"])
    binding = _active_parent_binding()
    roots = DataRoots.from_project_root(PROJECT_ROOT)
    audit = validate_parent_membership_day(
        date=date,
        roots=roots,
        binding=binding,
    )
    expected_source_json = _current_source_identities_json(date, roots)
    if kwargs["source_identities_json"] != expected_source_json:
        raise ValueError(f"cached {date}: current source identity canonicalization drifted")
    _require_uniform_plateau_metadata(frame, date=date, audit=audit)
    if _RUN_CONTEXT is not None:
        _RUN_CONTEXT.membership_audits[date] = audit


def _configure_common_runner() -> None:
    COMMON.PROJECT_ROOT = PROJECT_ROOT
    COMMON.CANDIDATE_REGISTRIES = {REGISTRY_ID: PLATEAU_REGISTRY}
    COMMON.DEFAULT_CANDIDATE_REGISTRY = REGISTRY_ID
    COMMON.SOURCE_IDENTITY_SOURCES = REQUIRED_SOURCES
    COMMON.available_sessions = available_sessions
    COMMON._select_registry_candidates = _select_candidates
    COMMON.evaluate_day = _evaluate_day_with_plateau_lineage
    COMMON._validate_cached_day = _validate_cached_day_with_plateau_lineage


def resolve_output_dir(value: Path | str) -> Path:
    path = Path(value)
    resolved = (
        path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    )
    artifact_root = ARTIFACT_ROOT.resolve()
    parent_root = PARENT_RUN_DIR.resolve()
    if resolved == artifact_root or artifact_root not in resolved.parents:
        raise ValueError(
            "output must be a versioned child of "
            f"{artifact_root}; got {resolved}"
        )
    if resolved == parent_root or parent_root in resolved.parents:
        raise ValueError("plateau output cannot be written into the parent run")
    return resolved


def _validate_requested_range(start: str, end: str) -> None:
    _validate_date(start, "start")
    _validate_date(end, "end")
    if start > end:
        raise ValueError("start must be <= end")
    if not CONTRACT_START <= start <= end <= CONTRACT_END:
        raise ValueError(
            "plateau range must remain within the frozen retrospective range "
            f"{CONTRACT_START}..{CONTRACT_END}"
        )


def _validate_label_and_pool_contract() -> tuple[object, object]:
    label_spec = COMMON.LABEL_PROFILES[LABEL_PROFILE]
    pool_spec = COMMON.reference_pool_spec(label_spec)
    shared = CONTRACT["shared_semantics"]
    if label_spec.profile_id != shared["label"]:
        raise ValueError("common evaluator label does not match plateau contract")
    if pool_spec.pool_id != shared["eligible_pool"]:
        raise ValueError("common evaluator pool does not match plateau contract")
    return label_spec, pool_spec


def _boundary_membership_audit(
    sessions: Sequence[str],
    *,
    roots: DataRoots,
    binding: ParentBinding,
) -> dict[str, object]:
    dates = tuple(dict.fromkeys((sessions[0], sessions[-1])))
    return {
        date: validate_parent_membership_day(
            date=date,
            roots=roots,
            binding=binding,
        )
        for date in dates
    }


def build_dry_run_contract(args: argparse.Namespace) -> dict[str, object]:
    _configure_common_runner()
    roots = DataRoots.from_project_root(PROJECT_ROOT)
    binding = load_parent_binding()
    sessions = validate_parent_calendar(
        binding,
        roots=roots,
        start=args.start,
        end=args.end,
    )
    if not sessions:
        raise ValueError("no complete source sessions in requested range")
    label_spec, pool_spec = _validate_label_and_pool_contract()
    return {
        "dry_run": True,
        "contract": CONTRACT_ID,
        "frozen_contract_sha256": CONTRACT_SHA256,
        "family_id": FAMILY_ID,
        "family_version": FAMILY_VERSION,
        "design_fingerprint": DESIGN_FINGERPRINT,
        "runner_sha256": _runner_sha256(),
        "cell_count": len(CELL_IDS),
        "center_cell_id": CENTER_CELL_ID,
        "candidate_ids": list(CELL_IDS),
        "label_profile_id": label_spec.profile_id,
        "label_profile_hash": label_spec.fingerprint,
        "pool_id": pool_spec.pool_id,
        "pool_version": pool_spec.pool_version,
        "start": args.start,
        "end": args.end,
        "step": 1,
        "session_count": len(sessions),
        "first_session": sessions[0],
        "last_session": sessions[-1],
        "output_dir": str(args.output_dir),
        "resume": not args.force,
        "parent_run_manifest": binding.manifest_identity,
        "parent_membership_semantics": (
            "conservative_base; invalid depth and null gates fail closed"
        ),
        "boundary_membership_audit": _boundary_membership_audit(
            sessions,
            roots=roots,
            binding=binding,
        ),
    }


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
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _augment_run_manifest(
    output_dir: Path,
    *,
    binding: ParentBinding,
) -> None:
    path = output_dir / "run_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "contract": CONTRACT_ID,
        "family_id": FAMILY_ID,
        "family_version": FAMILY_VERSION,
        "candidate_ids": list(CELL_IDS),
        "step": 1,
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"plateau run manifest identity mismatch: {mismatches}")
    payload.update(
        {
            "frozen_contract_sha256": CONTRACT_SHA256,
            "threshold_design_fingerprint": DESIGN_FINGERPRINT,
            "threshold_design": DESIGN_PAYLOAD,
            "center_candidate_id": CENTER_CELL_ID,
            "runner_artifact": _file_identity(RUNNER_PATH),
            "parent_run_manifest": binding.manifest_identity,
            "parent_reference_lineage": binding.reference_lineage_identity,
            "parent_membership_run_dir": str(binding.run_dir),
            "parent_membership_validation": (
                "each processed or resumed day matched current tick_data, "
                "tick_feature, pre_market, and txf_tick identities plus the "
                "parent membership and lineage file hashes"
            ),
            "candidate_surface_builds_per_processed_day": 0,
            "parent_membership_cache_reads_per_processed_day": 1,
            "membership_semantics": (
                "conservative_base and not depth_reference_invalid; "
                "threshold nulls fail closed; center key parity with parent "
                "conservative_dgb required"
            ),
            "selection_rule": (
                "fixed 27-cell ledger; no ranking or center replacement"
            ),
            "evidence_class": "retrospective_diagnostic",
            "outcome_informed_freeze": True,
            "production_promotion_authorized": False,
        }
    )
    _atomic_write_json(payload, path)


def run(args: argparse.Namespace) -> tuple[pl.DataFrame, pl.DataFrame]:
    global _RUN_CONTEXT

    _configure_common_runner()
    if _RUN_CONTEXT is not None:
        raise RuntimeError("threshold plateau runner is already active")
    roots = DataRoots.from_project_root(PROJECT_ROOT)
    binding = load_parent_binding()
    sessions = validate_parent_calendar(
        binding,
        roots=roots,
        start=args.start,
        end=args.end,
    )
    if not sessions:
        raise ValueError("no complete source sessions in requested range")
    _validate_label_and_pool_contract()
    _RUN_CONTEXT = RunContext(output_dir=args.output_dir, parent=binding)
    try:
        daily, summary = COMMON.run(args)
    finally:
        _RUN_CONTEXT = None
    _augment_run_manifest(args.output_dir, binding=binding)
    return daily, summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen 27-cell fh_neg_rejection threshold plateau from "
            "the validated parent membership cache"
        )
    )
    parser.add_argument("--start", default=CONTRACT_START)
    parser.add_argument("--end", default=CONTRACT_END)
    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    cache_mode = parser.add_mutually_exclusive_group()
    cache_mode.add_argument(
        "--resume",
        dest="force",
        action="store_false",
        help="reuse only fully lineage-matched plateau daily caches (default)",
    )
    cache_mode.add_argument(
        "--force",
        dest="force",
        action="store_true",
        help="quarantine and recompute plateau daily caches only",
    )
    parser.set_defaults(force=False)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate frozen identities, calendar, and boundary parent caches",
    )
    args = parser.parse_args(argv)
    try:
        _validate_requested_range(args.start, args.end)
        args.output_dir = resolve_output_dir(args.output_dir)
    except ValueError as exc:
        parser.error(str(exc))
    args.step = 1
    args.label_profile = LABEL_PROFILE
    args.control_sample = 0
    args.candidates = list(CELL_IDS)
    args.candidate_registry = REGISTRY_ID
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        print(
            json.dumps(
                build_dry_run_contract(args),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    run(args)
    print(
        json.dumps(
            {
                "status": "complete",
                "contract": CONTRACT_ID,
                "output_dir": str(args.output_dir),
                "outcomes_printed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
