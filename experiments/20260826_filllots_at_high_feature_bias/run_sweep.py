"""Run the preregistered FillLots-at-high feature family.

The estimator and cache/report contracts are reused from the canonical
FillLots-at-low framework study.  This adapter changes only the event registry,
its required source calendar, and its selector.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import polars as pl

from src.research.intradaySignal.framework.events import DataRoots, EventDaySources
from src.research.intradaySignal.framework.profiles.filllots_at_high import (
    CANDIDATE_DESCRIPTIONS,
    CANDIDATE_IDS,
    FILLLOTS_AT_HIGH_FEATURE_FAMILY_ID,
    FILLLOTS_AT_HIGH_FEATURE_FAMILY_VERSION,
    select_filllots_at_high_feature_candidates,
)


PROJECT_ROOT = Path(__file__).resolve().parents[5]
COMMON_RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "20260825_filllots_at_low_framework"
    / "run_sweep.py"
)
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("artifacts") / "sparse_v1"
CONTRACT_ID = "filllots_at_high_feature_bias_sweep/v1"
REGISTRY_ID = "filllots-at-high-feature-v1"
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


def _load_common_runner():
    spec = importlib.util.spec_from_file_location(
        "_filllots_at_low_common_runner",
        COMMON_RUNNER_PATH,
    )
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"cannot load common runner: {COMMON_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


COMMON = _load_common_runner()
HIGH_REGISTRY = COMMON.CandidateRegistry(
    registry_id=REGISTRY_ID,
    contract_id=CONTRACT_ID,
    family_id=FILLLOTS_AT_HIGH_FEATURE_FAMILY_ID,
    family_version=FILLLOTS_AT_HIGH_FEATURE_FAMILY_VERSION,
    candidate_ids=CANDIDATE_IDS,
    descriptions=CANDIDATE_DESCRIPTIONS,
)


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


def _select_candidates(
    day: EventDaySources,
    *,
    registry,
    candidate_ids: tuple[str, ...],
) -> dict[str, pl.DataFrame]:
    if registry.registry_id != REGISTRY_ID:
        raise ValueError(f"unexpected registry: {registry.registry_id}")
    day.require(*REQUIRED_SOURCES)
    return dict(
        select_filllots_at_high_feature_candidates(
            day,
            candidate_ids=candidate_ids,
        )
    )


def _configure_common_runner() -> None:
    COMMON.PROJECT_ROOT = PROJECT_ROOT
    COMMON.CANDIDATE_REGISTRIES = {REGISTRY_ID: HIGH_REGISTRY}
    COMMON.DEFAULT_CANDIDATE_REGISTRY = REGISTRY_ID
    COMMON.SOURCE_IDENTITY_SOURCES = REQUIRED_SOURCES
    COMMON.available_sessions = available_sessions
    COMMON._select_registry_candidates = _select_candidates


def run(args: argparse.Namespace) -> tuple[pl.DataFrame, pl.DataFrame]:
    _configure_common_runner()
    return COMMON.run(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep preregistered FillLots_atHigh fresh-event and causal "
            "tick-feature candidates on one shared eligible pool"
        )
    )
    parser.add_argument("--start", default="20250211")
    parser.add_argument("--end", default="20260825")
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--label-profile",
        choices=sorted(COMMON.LABEL_PROFILES),
        default="short-taker-stop108-v1",
    )
    parser.add_argument(
        "--candidate",
        dest="candidates",
        action="append",
        choices=CANDIDATE_IDS,
    )
    parser.add_argument(
        "--control-sample",
        type=int,
        default=0,
        help="0 uses the complete eligible pool",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    if args.step < 1:
        parser.error("--step must be positive")
    if args.control_sample < 0:
        parser.error("--control-sample cannot be negative")
    if args.candidates is None:
        args.candidates = list(CANDIDATE_IDS)
    args.candidate_registry = REGISTRY_ID
    return args


if __name__ == "__main__":
    parsed = parse_args()
    _, sweep_summary = run(parsed)
    print(sweep_summary)
