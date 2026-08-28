"""Run the frozen D/G/B ablation factorial for ``fh_neg_rejection``.

The base cohort is every fresh ``FillLots_atHigh < 0`` event.  The three
binary factors mean "require this gate" when set to one and "omit this gate"
when set to zero; zero never means the complement of the gate.  This produces
the eight nested ablation cells needed to measure the incremental contribution
of depth (D), distance from high (G), and bearish book state (B).

Each study day calls the public FillLots-at-high candidate-frame builder once.
All eight key sets are then filtered from that in-memory frame and passed to
the existing shared-pool sweep runner, which owns atomic daily aggregate
caches, source lineage, label/pool fingerprints, and resume validation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import polars as pl

from src.research.timeSeries.framework.events import (
    DAY_KEY_COLS,
    DataRoots,
    EventDaySources,
)
from src.research.timeSeries.framework.profiles.filllots_at_high import (
    FAR_HIGH_TICKS,
    FILLLOTS_AT_HIGH_FEATURE_FAMILY_VERSION,
    MICRO_MOMENTUM_THRESHOLD_BP,
    NEGATIVE_DEPTH_RATIO,
    WEAK_BID_SHARE,
    build_filllots_at_high_candidate_frame,
)


STUDY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STUDY_DIR.parents[4]
ARTIFACT_ROOT = STUDY_DIR / "artifacts"
DEFAULT_OUTPUT_DIR = ARTIFACT_ROOT / "dgb_factorial_v1"
STUDY_CONTRACT_PATH = STUDY_DIR / "study_contract.json"
COMMON_RUNNER_PATH = (
    STUDY_DIR.parent / "20260825_filllots_at_low_framework" / "run_sweep.py"
)
LEGACY_INDEX_PATH = (
    STUDY_DIR.parent
    / "20260826_filllots_at_high_signal"
    / "artifacts/event_index_v1/fh_neg_rejection.parquet"
)
LEGACY_INDEX_MANIFEST_PATH = LEGACY_INDEX_PATH.with_suffix(".manifest.json")
REFERENCE_LINEAGE_DIR = ARTIFACT_ROOT / "reference_lineage_v1"
REFERENCE_LINEAGE_PATH = REFERENCE_LINEAGE_DIR / "reference_lineage.parquet"
REFERENCE_LINEAGE_MANIFEST_PATH = (
    REFERENCE_LINEAGE_DIR / "reference_lineage.json"
)

FROZEN_CONTRACT_BYTES = STUDY_CONTRACT_PATH.read_bytes()
FROZEN_CONTRACT = json.loads(FROZEN_CONTRACT_BYTES)
CONTRACT_FINGERPRINT = hashlib.sha256(FROZEN_CONTRACT_BYTES).hexdigest()
CONTRACT_ID = str(FROZEN_CONTRACT["contract_id"])
if CONTRACT_ID != "fh_neg_rejection_robustness/fixed_factorial_v1":
    raise RuntimeError(f"unexpected frozen contract id: {CONTRACT_ID}")
FAMILY_CONTRACT = FROZEN_CONTRACT["primary_candidate_family"]
FAMILY_ID = str(FAMILY_CONTRACT["family_id"])
FAMILY_VERSION = str(FAMILY_CONTRACT["family_version"])
REGISTRY_ID = "fh-neg-rejection-dgb-factorial-v1"
BEAR_SCORE_THRESHOLD = 2

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
REQUIRED_SELECTOR_COLUMNS = {
    "tick_data": (
        *DAY_KEY_COLS,
        "FillLots",
        "InOut",
        "TrialMatch",
        "marketOpen",
        "RecordHigh",
        "RefPrice",
        "BidPrice1",
        "AskPrice1",
    ),
    "tick_feature": (
        *DAY_KEY_COLS,
        "B1_A1B1",
        "LOB_NetPressure_30",
        "MicroMomentum_30",
        "ToHigh",
        "TickSize",
    ),
    "pre_market": (
        "QuoteCode",
        "avg_bidLots1",
        "avg_askLots1",
        "big_buy_lots",
        "big_sell_lots",
    ),
}


@dataclass(frozen=True)
class FactorCell:
    """One nested gate-inclusion cell in the D/G/B factorial."""

    candidate_id: str
    require_depth: bool
    require_gap: bool
    require_bear: bool

    @property
    def description(self) -> str:
        gates = (
            f"D={'require ratio<=-8' if self.require_depth else 'omitted'}",
            f"G={'require gap>=4 ticks' if self.require_gap else 'omitted'}",
            f"B={'require bear_score>=2' if self.require_bear else 'omitted'}",
        )
        return "fresh and F_high<0 base; " + "; ".join(gates)


FACTOR_CELLS = tuple(
    FactorCell(
        candidate_id=str(cell["candidate_id"]),
        require_depth=bool(cell["D"]),
        require_gap=bool(cell["G"]),
        require_bear=bool(cell["B"]),
    )
    for cell in FAMILY_CONTRACT["cells"]
)
CELL_IDS = tuple(cell.candidate_id for cell in FACTOR_CELLS)
EXPECTED_CELL_IDS = (
    "d0_g0_b0",
    "d1_g0_b0",
    "d0_g1_b0",
    "d0_g0_b1",
    "d1_g1_b0",
    "d1_g0_b1",
    "d0_g1_b1",
    "d1_g1_b1",
)
if CELL_IDS != EXPECTED_CELL_IDS:
    raise RuntimeError(
        f"frozen factorial cell order drifted: {CELL_IDS}"
    )
CELL_BY_ID = {cell.candidate_id: cell for cell in FACTOR_CELLS}
CELL_DESCRIPTIONS = {
    cell.candidate_id: cell.description for cell in FACTOR_CELLS
}

BASE_DEFINITION = "_fresh_event and FillLots_atHigh < 0"
GATE_DEFINITIONS = {
    "D": f"filllots_at_high_depth_ratio <= {NEGATIVE_DEPTH_RATIO:g}",
    "G": f"filllots_at_high_gap_ticks >= {FAR_HIGH_TICKS:g}",
    "B": (
        "bear_score >= 2, where bear_score is the sum of "
        f"B1_A1B1 <= {WEAK_BID_SHARE:g}, "
        f"MicroMomentum_30 <= -{MICRO_MOMENTUM_THRESHOLD_BP:g}, and "
        "LOB_NetPressure_30 < 0"
    ),
}
DESIGN_PAYLOAD = {
    "contract": CONTRACT_ID,
    "frozen_contract_sha256": CONTRACT_FINGERPRINT,
    "upstream_fillhigh_family_version": (
        FILLLOTS_AT_HIGH_FEATURE_FAMILY_VERSION
    ),
    "base_definition": BASE_DEFINITION,
    "gate_definitions": GATE_DEFINITIONS,
    "zero_level_semantics": "gate omitted, not gate complement",
    "cells": [
        {
            "candidate_id": cell.candidate_id,
            "require_depth": cell.require_depth,
            "require_gap": cell.require_gap,
            "require_bear": cell.require_bear,
        }
        for cell in FACTOR_CELLS
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
        "_fh_neg_rejection_factorial_common_runner",
        COMMON_RUNNER_PATH,
    )
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"cannot load common runner: {COMMON_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


COMMON = _load_common_runner()
FACTORIAL_REGISTRY = COMMON.CandidateRegistry(
    registry_id=REGISTRY_ID,
    contract_id=CONTRACT_ID,
    family_id=FAMILY_ID,
    family_version=FAMILY_VERSION,
    candidate_ids=CELL_IDS,
    descriptions=CELL_DESCRIPTIONS,
)
_COMMON_VALIDATE_CACHED_DAY = COMMON._validate_cached_day


@dataclass
class RunContext:
    output_dir: Path
    force: bool


_RUN_CONTEXT: RunContext | None = None


def _require_frame_columns(frame: pl.DataFrame, columns: Sequence[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"factorial candidate frame missing columns: {missing}")


def build_factorial_gate_frame(
    frame: pl.DataFrame,
    *,
    reference_lineage: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build one compact legacy-base row with conservative D/G/B flags.

    The upstream profile has no validity bit for producer-sanitized reference
    values of one.  We therefore preserve its exact legacy membership for
    parity, while the primary v1 cells fail closed on ambiguous depth and
    same-side large-fill references.  A valid MD-price-change branch survives
    an ambiguous large-fill reference.
    """

    required = (
        *DAY_KEY_COLS,
        "_fresh_event",
        "_md_price_change",
        "_large_fill",
        "FillLots_atHigh",
        "FillLots",
        "InOut",
        "filllots_at_high_depth_ratio",
        "filllots_at_high_gap_ticks",
        "_large_fill_threshold",
        "avg_bidLots1",
        "avg_askLots1",
        "big_buy_lots",
        "big_sell_lots",
        "B1_A1B1",
        "MicroMomentum_30",
        "LOB_NetPressure_30",
    )
    _require_frame_columns(frame, required)
    lineage_columns = (
        "QuoteCode",
        "avg_bidLots1_lineage_status",
        "avg_askLots1_lineage_status",
        "big_buy_lots_lineage_status",
        "big_sell_lots_lineage_status",
    )
    if reference_lineage is None:
        reference_lineage = pl.DataFrame(
            schema={column: pl.String for column in lineage_columns}
        )
    _require_frame_columns(reference_lineage, lineage_columns)
    if reference_lineage["QuoteCode"].n_unique() != len(reference_lineage):
        raise ValueError("reference lineage contains duplicate QuoteCode rows")
    frame = frame.join(
        reference_lineage.select(*lineage_columns),
        on="QuoteCode",
        how="left",
        validate="m:1",
    )
    bear_score = (
        (pl.col("B1_A1B1") <= WEAK_BID_SHARE).cast(pl.UInt8)
        + (
            pl.col("MicroMomentum_30")
            <= -MICRO_MOMENTUM_THRESHOLD_BP
        ).cast(pl.UInt8)
        + (pl.col("LOB_NetPressure_30") < 0.0).cast(pl.UInt8)
    )
    avg_bid_ambiguous = (
        (pl.col("avg_bidLots1") - 1.0).abs() < 1.0e-9
    ).fill_null(True)
    avg_ask_ambiguous = (
        (pl.col("avg_askLots1") - 1.0).abs() < 1.0e-9
    ).fill_null(True)
    big_buy_ambiguous = (pl.col("big_buy_lots") == 1).fill_null(True)
    big_sell_ambiguous = (pl.col("big_sell_lots") == 1).fill_null(True)
    avg_bid_invalid = avg_bid_ambiguous & ~pl.col(
        "avg_bidLots1_lineage_status"
    ).is_in(("not_ambiguous",)).fill_null(False)
    avg_ask_invalid = avg_ask_ambiguous & ~pl.col(
        "avg_askLots1_lineage_status"
    ).is_in(("not_ambiguous",)).fill_null(False)
    valid_big_statuses = ("not_ambiguous", "proven_genuine_one")
    big_buy_invalid = big_buy_ambiguous & ~pl.col(
        "big_buy_lots_lineage_status"
    ).is_in(valid_big_statuses).fill_null(False)
    big_sell_invalid = big_sell_ambiguous & ~pl.col(
        "big_sell_lots_lineage_status"
    ).is_in(valid_big_statuses).fill_null(False)
    side_large_fill_invalid = (
        pl.when(pl.col("InOut") > 0)
        .then(big_buy_invalid)
        .when(pl.col("InOut") < 0)
        .then(big_sell_invalid)
        .otherwise(False)
        .fill_null(True)
    )
    return (
        frame.filter(
            pl.col("_fresh_event").fill_null(False)
            & (pl.col("FillLots_atHigh") < 0).fill_null(False)
        )
        .select(
            *DAY_KEY_COLS,
            "FillLots_atHigh",
            "FillLots",
            "InOut",
            "filllots_at_high_depth_ratio",
            "filllots_at_high_gap_ticks",
            "_md_price_change",
            "_large_fill",
            "_large_fill_threshold",
            "avg_bidLots1",
            "avg_askLots1",
            "big_buy_lots",
            "big_sell_lots",
            "avg_bidLots1_lineage_status",
            "avg_askLots1_lineage_status",
            "big_buy_lots_lineage_status",
            "big_sell_lots_lineage_status",
            "B1_A1B1",
            "MicroMomentum_30",
            "LOB_NetPressure_30",
        )
        .with_columns(
            bear_score.alias("bear_score"),
            avg_bid_ambiguous.alias("avg_bid_ambiguous_one"),
            avg_ask_ambiguous.alias("avg_ask_ambiguous_one"),
            big_buy_ambiguous.alias("big_buy_ambiguous_one"),
            big_sell_ambiguous.alias("big_sell_ambiguous_one"),
            (avg_bid_ambiguous | avg_ask_ambiguous).alias(
                "depth_reference_ambiguous_one"
            ),
            (avg_bid_invalid | avg_ask_invalid).alias(
                "depth_reference_invalid"
            ),
            side_large_fill_invalid.alias(
                "large_fill_reference_invalid"
            ),
        )
        .with_columns(
            (
                pl.col("filllots_at_high_depth_ratio")
                <= NEGATIVE_DEPTH_RATIO
            )
            .fill_null(False)
            .alias("pass_d_legacy"),
            (
                pl.col("filllots_at_high_gap_ticks") >= FAR_HIGH_TICKS
            )
            .fill_null(False)
            .alias("pass_g"),
            (pl.col("bear_score") >= BEAR_SCORE_THRESHOLD)
            .fill_null(False)
            .alias("pass_b"),
            (
                pl.col("_large_fill")
                & ~pl.col("large_fill_reference_invalid")
            ).alias("large_fill_conservative"),
        )
        .with_columns(
            (
                pl.col("_md_price_change")
                | pl.col("large_fill_conservative")
            ).alias("conservative_base"),
            (
                pl.col("pass_d_legacy")
                & ~pl.col("depth_reference_invalid")
            ).alias("pass_d"),
            (
                pl.col("pass_d_legacy")
                & pl.col("pass_g")
                & pl.col("pass_b")
            ).alias("legacy_dgb"),
        )
        .with_columns(
            (
                pl.col("conservative_base")
                & pl.col("pass_d")
                & pl.col("pass_g")
                & pl.col("pass_b")
            ).alias("conservative_dgb"),
            pl.when(
                pl.col("_md_price_change")
                & pl.col("large_fill_conservative")
            )
            .then(pl.lit("md_and_large_fill"))
            .when(pl.col("_md_price_change"))
            .then(pl.lit("md_only"))
            .when(pl.col("large_fill_conservative"))
            .then(pl.lit("large_fill_only"))
            .otherwise(pl.lit("invalid_large_fill_only"))
            .alias("fresh_branch_conservative"),
        )
    )


def select_factorial_candidates_from_frame(
    frame: pl.DataFrame,
    *,
    candidate_ids: Sequence[str] = CELL_IDS,
) -> Mapping[str, pl.DataFrame]:
    """Select requested factorial key sets without another raw-source scan."""

    selected_ids = tuple(candidate_ids)
    unknown = sorted(set(selected_ids) - set(CELL_IDS))
    if unknown:
        raise ValueError(f"unknown factorial cells: {unknown}")
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("factorial candidate_ids cannot contain duplicates")

    gates = build_factorial_gate_frame(frame)
    return select_factorial_candidates_from_gate_frame(
        gates,
        candidate_ids=selected_ids,
    )


def select_factorial_candidates_from_gate_frame(
    gates: pl.DataFrame,
    *,
    candidate_ids: Sequence[str] = CELL_IDS,
) -> Mapping[str, pl.DataFrame]:
    """Select cells from one already-built compact membership surface."""

    selected_ids = tuple(candidate_ids)
    unknown = sorted(set(selected_ids) - set(CELL_IDS))
    if unknown:
        raise ValueError(f"unknown factorial cells: {unknown}")
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("factorial candidate_ids cannot contain duplicates")
    selected: dict[str, pl.DataFrame] = {}
    for candidate_id in selected_ids:
        cell = CELL_BY_ID[candidate_id]
        expression = _cell_membership_expression(cell)
        selected[candidate_id] = (
            gates.filter(expression)
            .select(*DAY_KEY_COLS)
            .sort(list(DAY_KEY_COLS))
        )
    return selected


def _cell_membership_expression(cell: FactorCell) -> pl.Expr:
    expression = pl.col("conservative_base")
    if cell.require_depth:
        expression = expression & pl.col("pass_d")
    if cell.require_gap:
        expression = expression & pl.col("pass_g")
    if cell.require_bear:
        expression = expression & pl.col("pass_b")
    return expression


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


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def _quarantine_file(path: Path, quarantine_dir: Path) -> Path:
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    destination = quarantine_dir / path.name
    suffix = 1
    while destination.exists():
        destination = quarantine_dir / f"{path.stem}.{suffix}{path.suffix}"
        suffix += 1
    os.replace(path, destination)
    return destination


def _membership_paths(output_dir: Path, date: str) -> tuple[Path, Path]:
    return (
        output_dir / "membership_daily" / f"{date}.parquet",
        output_dir / "membership_lineage" / f"{date}.json",
    )


def _reference_lineage_for_date(date: str) -> pl.DataFrame:
    if not REFERENCE_LINEAGE_PATH.is_file():
        raise FileNotFoundError(
            f"missing frozen reference lineage: {REFERENCE_LINEAGE_PATH}"
        )
    return (
        pl.scan_parquet(REFERENCE_LINEAGE_PATH)
        .filter(pl.col("Date") == date)
        .select(
            "QuoteCode",
            "avg_bidLots1_lineage_status",
            "avg_askLots1_lineage_status",
            "big_buy_lots_lineage_status",
            "big_sell_lots_lineage_status",
        )
        .collect()
    )


def _legacy_artifact_parity(
    date: str,
    rebuilt_legacy_keys: pl.DataFrame,
    *,
    source_identities_json: str,
) -> dict[str, object]:
    """Compare rebuilt legacy DGB keys to the frozen old event index."""

    manifest = json.loads(
        LEGACY_INDEX_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    processed_dates = set(manifest.get("processed_dates", ()))
    if date not in processed_dates:
        return {
            "status": "not_applicable_outside_legacy_artifact",
            "rebuilt_legacy_rows": len(rebuilt_legacy_keys),
        }
    current_sources = json.loads(source_identities_json)
    expected_sources = manifest["source_identities"][date]
    mismatched_sources = {
        source: {
            "expected": expected_sources[source],
            "actual": current_sources[source],
        }
        for source in ("tick_data", "tick_feature", "pre_market")
        if current_sources.get(source) != expected_sources.get(source)
    }
    if mismatched_sources:
        raise ValueError(
            f"{date}: legacy parity source identities drifted: "
            f"{mismatched_sources}"
        )
    expected = (
        pl.scan_parquet(LEGACY_INDEX_PATH)
        .filter(pl.col("Date") == date)
        .select(*DAY_KEY_COLS)
        .collect()
        .sort(list(DAY_KEY_COLS))
    )
    rebuilt = rebuilt_legacy_keys.sort(list(DAY_KEY_COLS))
    missing = expected.join(rebuilt, on=list(DAY_KEY_COLS), how="anti")
    extra = rebuilt.join(expected, on=list(DAY_KEY_COLS), how="anti")
    report = {
        "status": "pass" if not len(missing) and not len(extra) else "fail",
        "expected_legacy_rows": len(expected),
        "rebuilt_legacy_rows": len(rebuilt),
        "missing_from_rebuilt": len(missing),
        "extra_in_rebuilt": len(extra),
    }
    if report["status"] != "pass":
        raise ValueError(f"{date}: legacy fh_neg_rejection parity failed: {report}")
    return report


def _membership_mask_expression() -> pl.Expr:
    bits = [
        pl.when(_cell_membership_expression(cell))
        .then(pl.lit(1 << index, dtype=pl.UInt16))
        .otherwise(pl.lit(0, dtype=pl.UInt16))
        for index, cell in enumerate(FACTOR_CELLS)
    ]
    return pl.sum_horizontal(bits).cast(pl.UInt16).alias("membership_mask")


def _count_true(frame: pl.DataFrame, column: str) -> int:
    value = frame.select(pl.col(column).cast(pl.UInt64).sum()).item()
    return int(value or 0)


def _unique_quotes_when(frame: pl.DataFrame, column: str) -> int:
    return int(
        frame.filter(pl.col(column))["QuoteCode"].n_unique()
        if len(frame)
        else 0
    )


def _value_counts(frame: pl.DataFrame, column: str) -> dict[str, int]:
    if not len(frame):
        return {}
    return {
        str(row[column]): int(row["len"])
        for row in frame.group_by(column).len().iter_rows(named=True)
    }


def _membership_lineage_payload(
    date: str,
    gates: pl.DataFrame,
    selected: Mapping[str, pl.DataFrame],
    *,
    source_identities_json: str,
    parity: dict[str, object],
    membership_path: Path,
) -> dict[str, object]:
    ambiguous_dgb = gates.filter(
        pl.col("legacy_dgb") & ~pl.col("conservative_dgb")
    )
    return {
        "contract": CONTRACT_ID,
        "frozen_contract_sha256": CONTRACT_FINGERPRINT,
        "design_fingerprint": DESIGN_FINGERPRINT,
        "family_id": FAMILY_ID,
        "family_version": FAMILY_VERSION,
        "Date": date,
        "source_identities_sha256": _sha256_text(source_identities_json),
        "source_identities": json.loads(source_identities_json),
        "reference_lineage_artifact": _file_identity(REFERENCE_LINEAGE_PATH),
        "membership_artifact": _file_identity(membership_path),
        "legacy_parity": parity,
        "legacy_base_rows": len(gates),
        "conservative_base_rows": _count_true(gates, "conservative_base"),
        "legacy_dgb_rows": _count_true(gates, "legacy_dgb"),
        "conservative_dgb_rows": _count_true(gates, "conservative_dgb"),
        "legacy_dgb_removed_by_reference_validation": len(ambiguous_dgb),
        "candidate_raw_rows": {
            candidate_id: len(keys) for candidate_id, keys in selected.items()
        },
        "ambiguous_one_exposure": {
            "avg_bid_event_rows": _count_true(gates, "avg_bid_ambiguous_one"),
            "avg_ask_event_rows": _count_true(gates, "avg_ask_ambiguous_one"),
            "big_buy_event_rows": _count_true(gates, "big_buy_ambiguous_one"),
            "big_sell_event_rows": _count_true(gates, "big_sell_ambiguous_one"),
            "depth_invalid_event_rows": _count_true(
                gates, "depth_reference_invalid"
            ),
            "depth_invalid_quotes": _unique_quotes_when(
                gates, "depth_reference_invalid"
            ),
            "side_large_fill_invalid_event_rows": _count_true(
                gates, "large_fill_reference_invalid"
            ),
            "side_large_fill_invalid_quotes": _unique_quotes_when(
                gates, "large_fill_reference_invalid"
            ),
        },
        "fresh_branch_counts": _value_counts(
            gates, "fresh_branch_conservative"
        ),
        "legacy_large_fill_inout_counts": _value_counts(
            gates.filter(pl.col("_large_fill")), "InOut"
        ),
        "conservative_large_fill_inout_counts": _value_counts(
            gates.filter(pl.col("large_fill_conservative")), "InOut"
        ),
    }


def _publish_membership_day(
    day: EventDaySources,
    gates: pl.DataFrame,
    selected: Mapping[str, pl.DataFrame],
) -> None:
    if _RUN_CONTEXT is None:  # unit-level selector use
        return
    output_dir = _RUN_CONTEXT.output_dir
    membership_path, lineage_path = _membership_paths(output_dir, day.date)
    for path in (membership_path, lineage_path):
        if path.exists():
            _quarantine_file(
                path,
                output_dir / "quarantine" / path.parent.name,
            )
    source_identities_json = COMMON._day_source_identities_json(
        day.date, day.roots
    )
    rebuilt_legacy_keys = (
        gates.filter(pl.col("legacy_dgb"))
        .select(*DAY_KEY_COLS)
        .sort(list(DAY_KEY_COLS))
    )
    parity = _legacy_artifact_parity(
        day.date,
        rebuilt_legacy_keys,
        source_identities_json=source_identities_json,
    )
    membership = (
        gates.with_columns(
            pl.lit(day.date).alias("Date"),
            _membership_mask_expression(),
        )
        .select(
            "Date",
            *DAY_KEY_COLS,
            "membership_mask",
            "conservative_base",
            "legacy_dgb",
            "conservative_dgb",
            "pass_d_legacy",
            "pass_d",
            "pass_g",
            "pass_b",
            "FillLots_atHigh",
            "FillLots",
            "InOut",
            "filllots_at_high_depth_ratio",
            "filllots_at_high_gap_ticks",
            "bear_score",
            "_md_price_change",
            "_large_fill",
            "large_fill_conservative",
            "fresh_branch_conservative",
            "_large_fill_threshold",
            "avg_bidLots1",
            "avg_askLots1",
            "big_buy_lots",
            "big_sell_lots",
            "avg_bid_ambiguous_one",
            "avg_ask_ambiguous_one",
            "big_buy_ambiguous_one",
            "big_sell_ambiguous_one",
            "depth_reference_invalid",
            "large_fill_reference_invalid",
            "avg_bidLots1_lineage_status",
            "avg_askLots1_lineage_status",
            "big_buy_lots_lineage_status",
            "big_sell_lots_lineage_status",
            "B1_A1B1",
            "MicroMomentum_30",
            "LOB_NetPressure_30",
        )
        .sort(["Date", *DAY_KEY_COLS])
    )
    if membership.select(
        pl.struct(["Date", *DAY_KEY_COLS]).is_duplicated().any()
    ).item():
        raise ValueError(f"{day.date}: duplicate compact membership keys")
    _atomic_write_parquet(membership, membership_path)
    lineage = _membership_lineage_payload(
        day.date,
        gates,
        selected,
        source_identities_json=source_identities_json,
        parity=parity,
        membership_path=membership_path,
    )
    _atomic_write_json(lineage, lineage_path)


def _validate_membership_cache(
    *,
    date: str,
    source_identities_json: str,
) -> None:
    if _RUN_CONTEXT is None:
        return
    membership_path, lineage_path = _membership_paths(
        _RUN_CONTEXT.output_dir, date
    )
    if not membership_path.is_file() or not lineage_path.is_file():
        raise ValueError(f"cached {date}: compact membership cache is missing")
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    expected = {
        "contract": CONTRACT_ID,
        "frozen_contract_sha256": CONTRACT_FINGERPRINT,
        "design_fingerprint": DESIGN_FINGERPRINT,
        "family_id": FAMILY_ID,
        "family_version": FAMILY_VERSION,
        "Date": date,
        "source_identities_sha256": _sha256_text(source_identities_json),
    }
    mismatches = {
        key: {"expected": value, "actual": lineage.get(key)}
        for key, value in expected.items()
        if lineage.get(key) != value
    }
    if lineage.get("membership_artifact") != _file_identity(membership_path):
        mismatches["membership_artifact"] = "file identity drift"
    if lineage.get("reference_lineage_artifact") != _file_identity(
        REFERENCE_LINEAGE_PATH
    ):
        mismatches["reference_lineage_artifact"] = "file identity drift"
    if mismatches:
        raise ValueError(f"cached {date}: membership lineage mismatch {mismatches}")


def _validate_cached_day_with_membership(
    frame: pl.DataFrame,
    **kwargs,
) -> None:
    _COMMON_VALIDATE_CACHED_DAY(frame, **kwargs)
    _validate_membership_cache(
        date=kwargs["date"],
        source_identities_json=kwargs["source_identities_json"],
    )


def _select_candidates(
    day: EventDaySources,
    *,
    registry,
    candidate_ids: tuple[str, ...],
) -> dict[str, pl.DataFrame]:
    """Build the raw candidate frame exactly once for this study day."""

    if registry.registry_id != REGISTRY_ID:
        raise ValueError(f"unexpected registry: {registry.registry_id}")
    day.require(*REQUIRED_SOURCES)
    frame = build_filllots_at_high_candidate_frame(day)
    gates = build_factorial_gate_frame(
        frame,
        reference_lineage=_reference_lineage_for_date(day.date),
    )
    selected = dict(
        select_factorial_candidates_from_gate_frame(
            gates,
            candidate_ids=candidate_ids,
        )
    )
    _publish_membership_day(day, gates, selected)
    return selected


def _configure_common_runner() -> None:
    COMMON.PROJECT_ROOT = PROJECT_ROOT
    COMMON.CANDIDATE_REGISTRIES = {REGISTRY_ID: FACTORIAL_REGISTRY}
    COMMON.DEFAULT_CANDIDATE_REGISTRY = REGISTRY_ID
    COMMON.SOURCE_IDENTITY_SOURCES = REQUIRED_SOURCES
    COMMON.available_sessions = available_sessions
    COMMON._select_registry_candidates = _select_candidates
    COMMON._validate_cached_day = _validate_cached_day_with_membership


def resolve_output_dir(value: Path | str) -> Path:
    """Fail closed unless output is a versioned child of this study."""

    path = Path(value)
    resolved = (
        path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    )
    artifact_root = ARTIFACT_ROOT.resolve()
    if resolved == artifact_root or artifact_root not in resolved.parents:
        raise ValueError(
            "output must be a versioned child of "
            f"{artifact_root}; got {resolved}"
        )
    return resolved


def _validate_date(value: str, label: str) -> None:
    try:
        parsed = datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise ValueError(f"{label} must be a valid YYYYMMDD date") from exc
    if parsed.strftime("%Y%m%d") != value:
        raise ValueError(f"{label} must be exactly eight YYYYMMDD digits")


def _validate_boundary_schemas(
    roots: DataRoots,
    sessions: Sequence[str],
) -> dict[str, object]:
    """Inspect Parquet metadata only; do not materialize raw rows."""

    boundary_dates = tuple(dict.fromkeys((sessions[0], sessions[-1])))
    audit: dict[str, object] = {}
    for date in boundary_dates:
        day = EventDaySources(date, roots)
        day.require(*REQUIRED_SOURCES)
        date_audit: dict[str, object] = {}
        for source in REQUIRED_SOURCES:
            path = day.path(source)
            schema = pl.read_parquet_schema(path)
            required = REQUIRED_SELECTOR_COLUMNS.get(source, ())
            missing = sorted(set(required) - set(schema.names()))
            if missing:
                raise ValueError(
                    f"{date} {source} missing required columns: {missing}"
                )
            stat = path.stat()
            date_audit[source] = {
                "path": str(path.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "column_count": len(schema),
            }
        audit[date] = date_audit
    return audit


def build_dry_run_contract(args: argparse.Namespace) -> dict[str, object]:
    roots = DataRoots.from_project_root(PROJECT_ROOT)
    sessions = available_sessions(
        roots,
        start=args.start,
        end=args.end,
    )[:: args.step]
    if not sessions:
        raise ValueError("no complete source sessions in requested range")
    label_spec = COMMON.LABEL_PROFILES[args.label_profile]
    pool_spec = COMMON.reference_pool_spec(label_spec)
    return {
        "dry_run": True,
        "contract": CONTRACT_ID,
        "frozen_contract_sha256": CONTRACT_FINGERPRINT,
        "family_id": FAMILY_ID,
        "family_version": FAMILY_VERSION,
        "design_fingerprint": DESIGN_FINGERPRINT,
        "design": DESIGN_PAYLOAD,
        "label_profile_id": label_spec.profile_id,
        "label_profile_hash": label_spec.fingerprint,
        "pool_id": pool_spec.pool_id,
        "pool_version": pool_spec.pool_version,
        "start": args.start,
        "end": args.end,
        "step": args.step,
        "session_count": len(sessions),
        "first_session": sessions[0],
        "last_session": sessions[-1],
        "output_dir": str(args.output_dir),
        "resume": not args.force,
        "primary_membership": "conservative_reference_validated",
        "legacy_parity_membership": "diagnostic_only",
        "reference_lineage_artifact": _file_identity(
            REFERENCE_LINEAGE_PATH
        ),
        "boundary_source_audit": _validate_boundary_schemas(roots, sessions),
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


def _augment_run_manifest(output_dir: Path) -> None:
    path = output_dir / "run_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "contract": CONTRACT_ID,
        "family_id": FAMILY_ID,
        "family_version": FAMILY_VERSION,
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"run manifest identity mismatch: {mismatches}")
    payload.update(
        {
            "frozen_contract_sha256": CONTRACT_FINGERPRINT,
            "factorial_design_fingerprint": DESIGN_FINGERPRINT,
            "factorial_design": DESIGN_PAYLOAD,
            "factorial_semantics": (
                "0 omits a gate; 1 requires it; cells are nested, not disjoint"
            ),
            "candidate_frame_builds_per_processed_day": 1,
            "primary_membership": "conservative_reference_validated",
            "legacy_parity_membership": "diagnostic_only",
            "reference_lineage_artifact": _file_identity(
                REFERENCE_LINEAGE_PATH
            ),
        }
    )
    _atomic_write_json(payload, path)


def run(args: argparse.Namespace) -> tuple[pl.DataFrame, pl.DataFrame]:
    global _RUN_CONTEXT

    _configure_common_runner()
    if _RUN_CONTEXT is not None:
        raise RuntimeError("factorial runner is already active")
    _RUN_CONTEXT = RunContext(args.output_dir, args.force)
    try:
        daily, summary = COMMON.run(args)
    finally:
        _RUN_CONTEXT = None
    _augment_run_manifest(args.output_dir)
    return daily, summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 8-cell nested D/G/B fh_neg_rejection robustness factorial"
        )
    )
    parser.add_argument("--start", default="20250211")
    parser.add_argument("--end", default="20260826")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--label-profile",
        choices=sorted(COMMON.LABEL_PROFILES),
        default="short-taker-stop108-v1",
    )
    parser.add_argument(
        "--control-sample",
        type=int,
        default=0,
        help="0 uses the complete shared eligible pool",
    )
    cache_mode = parser.add_mutually_exclusive_group()
    cache_mode.add_argument(
        "--resume",
        dest="force",
        action="store_false",
        help=(
            "reuse only daily caches whose contract and source identities "
            "still match (default)"
        ),
    )
    cache_mode.add_argument(
        "--force",
        dest="force",
        action="store_true",
        help="quarantine and recompute requested daily caches",
    )
    parser.set_defaults(force=False)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate contract, calendar, output boundary, and source schemas",
    )
    args = parser.parse_args(argv)
    _validate_date(args.start, "start")
    _validate_date(args.end, "end")
    if args.start > args.end:
        parser.error("start must be <= end")
    if args.step < 1:
        parser.error("step must be positive")
    if args.control_sample < 0:
        parser.error("control-sample cannot be negative")
    try:
        args.output_dir = resolve_output_dir(args.output_dir)
    except ValueError as exc:
        parser.error(str(exc))
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
    _, summary = run(args)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
