"""Focused synthetic contracts for the fixed threshold-plateau runner."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import polars as pl

import run_threshold_plateau as study
from src.research.intradaySignal.framework.events import DataRoots, EventDaySources


def _membership_fixture() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Date": ["20260102"] * 8,
            "QuoteCode": ["2330"] * 8,
            "ChannelSeq": list(range(1, 9)),
            "conservative_base": [
                True,
                True,
                True,
                False,
                True,
                True,
                True,
                True,
            ],
            "conservative_dgb": [
                False,
                True,
                False,
                False,
                False,
                False,
                True,
                False,
            ],
            "filllots_at_high_depth_ratio": [
                -7.0,
                -9.0,
                -11.0,
                -20.0,
                -20.0,
                None,
                -11.0,
                -20.0,
            ],
            "filllots_at_high_gap_ticks": [
                4.0,
                5.0,
                3.0,
                10.0,
                10.0,
                10.0,
                5.0,
                None,
            ],
            "bear_score": [2, 3, 1, 3, 3, 3, 3, 3],
            "depth_reference_invalid": [
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                None,
            ],
        }
    )


def _create_sources(root: Path, date: str) -> DataRoots:
    roots = DataRoots.from_project_root(root)
    day = EventDaySources(date, roots)
    for source in study.REQUIRED_SOURCES:
        path = day.path(source)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{date}:{source}".encode("utf-8"))
    return roots


def _write_synthetic_parent(
    parent_dir: Path,
    *,
    roots: DataRoots,
    date: str,
) -> None:
    membership_dir = parent_dir / "membership_daily"
    lineage_dir = parent_dir / "membership_lineage"
    membership_dir.mkdir(parents=True, exist_ok=True)
    lineage_dir.mkdir(parents=True, exist_ok=True)
    membership_path = membership_dir / f"{date}.parquet"
    _membership_fixture().with_columns(pl.lit(date).alias("Date")).write_parquet(
        membership_path
    )
    sources = study._current_source_identities(date, roots)
    design = {
        "contract": study.PARENT_CONTRACT_ID,
        "frozen_contract_sha256": study.PARENT_CONTRACT_SHA256,
        "cells": list(study.PARENT_CELL_IDS),
    }
    design_fingerprint = study.hashlib.sha256(
        study._canonical_json(design).encode("utf-8")
    ).hexdigest()
    manifest = {
        "contract": study.PARENT_CONTRACT_ID,
        "family_id": study.PARENT_FAMILY_ID,
        "family_version": study.PARENT_FAMILY_VERSION,
        "frozen_contract_sha256": study.PARENT_CONTRACT_SHA256,
        "candidate_ids": list(study.PARENT_CELL_IDS),
        "primary_membership": "conservative_reference_validated",
        "step": 1,
        "sessions": [date],
        "failures": [],
        "source_identities": {date: sources},
        "factorial_design": design,
        "factorial_design_fingerprint": design_fingerprint,
        "reference_lineage_artifact": study._file_identity(
            study.REFERENCE_LINEAGE_PATH
        ),
    }
    (parent_dir / "run_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    sources_json = study._canonical_json(sources)
    lineage = {
        "contract": study.PARENT_CONTRACT_ID,
        "frozen_contract_sha256": study.PARENT_CONTRACT_SHA256,
        "design_fingerprint": design_fingerprint,
        "family_id": study.PARENT_FAMILY_ID,
        "family_version": study.PARENT_FAMILY_VERSION,
        "Date": date,
        "source_identities": sources,
        "source_identities_sha256": study._sha256_text(sources_json),
        "membership_artifact": study._file_identity(membership_path),
        "reference_lineage_artifact": study._file_identity(
            study.REFERENCE_LINEAGE_PATH
        ),
    }
    (lineage_dir / f"{date}.json").write_text(
        json.dumps(lineage),
        encoding="utf-8",
    )


class ThresholdPlateauContractTest(unittest.TestCase):
    def test_frozen_grid_has_exactly_27_cells_and_center(self) -> None:
        self.assertEqual(len(study.THRESHOLD_CELLS), 27)
        self.assertEqual(len(set(study.CELL_IDS)), 27)
        self.assertEqual(study.CENTER_CELL_ID, "d08_g04_b2")
        center = study.CELL_BY_ID[study.CENTER_CELL_ID]
        self.assertAlmostEqual(center.depth_ratio_lte, -8.0)
        self.assertAlmostEqual(center.gap_ticks_gte, 4.0)
        self.assertEqual(center.bear_score_gte, 2)

    def test_selector_preserves_base_and_fails_closed_on_invalid_depth(self) -> None:
        selected = study.select_threshold_candidates_from_membership(
            _membership_fixture()
        )
        loose = set(selected["d06_g03_b1"]["ChannelSeq"].to_list())
        center = set(selected[study.CENTER_CELL_ID]["ChannelSeq"].to_list())
        strict = set(selected["d10_g05_b3"]["ChannelSeq"].to_list())

        self.assertEqual(loose, {1, 2, 3, 7})
        self.assertEqual(center, {2, 7})
        self.assertEqual(strict, {7})
        self.assertTrue(strict < center < loose)
        for keys in selected.values():
            chosen = set(keys["ChannelSeq"].to_list())
            self.assertNotIn(4, chosen)
            self.assertNotIn(5, chosen)
            self.assertNotIn(6, chosen)
            self.assertNotIn(8, chosen)

    def test_parent_lineage_matches_all_four_current_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            date = "20260102"
            roots = _create_sources(root, date)
            parent_dir = root / "parent"
            _write_synthetic_parent(parent_dir, roots=roots, date=date)
            binding = study.load_parent_binding(parent_dir)

            membership, audit = study.load_parent_membership_frame(
                date=date,
                roots=roots,
                binding=binding,
            )
            self.assertEqual(len(membership), 8)
            self.assertEqual(audit["Date"], date)
            study._require_uniform_plateau_metadata(
                pl.DataFrame([study._plateau_cache_metadata(audit)]),
                date=date,
                audit=audit,
            )

            tick_feature = EventDaySources(date, roots).path("tick_feature")
            tick_feature.write_bytes(b"source identity changed")
            with self.assertRaisesRegex(ValueError, "lineage mismatch"):
                study.validate_parent_membership_day(
                    date=date,
                    roots=roots,
                    binding=binding,
                )

    def test_resume_metadata_and_output_boundary_fail_closed(self) -> None:
        audit = {
            "parent_manifest_sha256": "a",
            "parent_membership_sha256": "b",
            "parent_membership_lineage_sha256": "c",
            "parent_source_identities_sha256": "d",
        }
        valid = pl.DataFrame([study._plateau_cache_metadata(audit)])
        study._require_uniform_plateau_metadata(
            valid,
            date="20260102",
            audit=audit,
        )
        stale = valid.with_columns(
            pl.lit("wrong").alias("plateau_design_fingerprint")
        )
        with self.assertRaisesRegex(ValueError, "design_fingerprint"):
            study._require_uniform_plateau_metadata(
                stale,
                date="20260102",
                audit=audit,
            )

        accepted = study.resolve_output_dir(
            study.ARTIFACT_ROOT / "threshold_plateau_unit_v1"
        )
        self.assertEqual(
            accepted,
            (study.ARTIFACT_ROOT / "threshold_plateau_unit_v1").resolve(),
        )
        for rejected in (study.ARTIFACT_ROOT, study.PARENT_RUN_DIR):
            with self.subTest(path=rejected):
                with self.assertRaises(ValueError):
                    study.resolve_output_dir(rejected)


if __name__ == "__main__":
    unittest.main()
