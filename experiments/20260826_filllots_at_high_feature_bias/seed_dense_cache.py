"""Seed a one-candidate dense run with compatible sparse daily caches."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import polars as pl

from src.research.intradaySignal.framework.profiles.filllots_at_high import (
    CANDIDATE_IDS,
)


def _atomic_write_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        frame.write_parquet(temporary, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def seed_cache(source_dir: Path, target_dir: Path, candidate_id: str) -> int:
    if candidate_id not in CANDIDATE_IDS:
        raise ValueError(f"unknown candidate: {candidate_id}")
    manifest_path = source_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("failures") != []:
        raise ValueError("source sparse run has failures")
    if manifest.get("candidate_ids") != list(CANDIDATE_IDS):
        raise ValueError("source sparse run is not the frozen full registry")
    source_daily = source_dir / "daily"
    paths = sorted(source_daily.glob("*.parquet"))
    if len(paths) != len(manifest.get("sessions", [])):
        raise ValueError("source daily cache does not cover manifest sessions")

    target_daily = target_dir / "daily"
    seeded = 0
    for source_path in paths:
        selected = pl.read_parquet(source_path).filter(
            pl.col("candidate_id") == candidate_id
        )
        if len(selected) != 1:
            raise ValueError(
                f"{source_path}: expected one {candidate_id} row, got {len(selected)}"
            )
        target_path = target_daily / source_path.name
        if target_path.exists():
            existing = pl.read_parquet(target_path)
            if not existing.equals(selected, null_equal=True):
                raise ValueError(f"existing dense cache differs: {target_path}")
            continue
        _atomic_write_parquet(selected, target_path)
        seeded += 1
    return seeded


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--target-dir", required=True, type=Path)
    parser.add_argument("--candidate", required=True, choices=CANDIDATE_IDS)
    args = parser.parse_args()
    seeded = seed_cache(args.source_dir, args.target_dir, args.candidate)
    print(f"Seeded {seeded} compatible daily caches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
