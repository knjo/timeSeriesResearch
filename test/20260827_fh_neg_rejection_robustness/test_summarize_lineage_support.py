"""Synthetic, outcome-free tests for the lineage/support summarizer."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import polars as pl

import summarize_lineage_support as study


DATE = "20260102"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _membership_frame(*, duplicate: bool = False) -> pl.DataFrame:
    null_statuses = {column: None for column in study.REFERENCE_STATUS_COLUMNS}
    rows = [
        {
            "Date": DATE,
            "QuoteCode": "A",
            "ChannelSeq": 1,
            "membership_mask": 255,
            "conservative_base": True,
            "legacy_dgb": True,
            "conservative_dgb": True,
            "pass_d_legacy": True,
            "pass_d": True,
            "pass_g": True,
            "pass_b": True,
            "InOut": -1,
            "_md_price_change": True,
            "_large_fill": False,
            "large_fill_conservative": False,
            "fresh_branch_conservative": "md_only",
            "avg_bid_ambiguous_one": False,
            "avg_ask_ambiguous_one": False,
            "big_buy_ambiguous_one": False,
            "big_sell_ambiguous_one": False,
            "depth_reference_invalid": False,
            "large_fill_reference_invalid": False,
            **null_statuses,
        },
        {
            "Date": DATE,
            "QuoteCode": "B",
            "ChannelSeq": 2,
            "membership_mask": 77,
            "conservative_base": True,
            "legacy_dgb": False,
            "conservative_dgb": False,
            "pass_d_legacy": False,
            "pass_d": False,
            "pass_g": True,
            "pass_b": True,
            "InOut": -1,
            "_md_price_change": False,
            "_large_fill": True,
            "large_fill_conservative": True,
            "fresh_branch_conservative": "large_fill_only",
            "avg_bid_ambiguous_one": False,
            "avg_ask_ambiguous_one": False,
            "big_buy_ambiguous_one": False,
            "big_sell_ambiguous_one": False,
            "depth_reference_invalid": False,
            "large_fill_reference_invalid": False,
            **null_statuses,
        },
        {
            "Date": DATE,
            "QuoteCode": "C",
            "ChannelSeq": 3,
            "membership_mask": 3,
            "conservative_base": True,
            "legacy_dgb": False,
            "conservative_dgb": False,
            "pass_d_legacy": True,
            "pass_d": True,
            "pass_g": False,
            "pass_b": False,
            "InOut": 1,
            "_md_price_change": True,
            "_large_fill": True,
            "large_fill_conservative": True,
            "fresh_branch_conservative": "md_and_large_fill",
            "avg_bid_ambiguous_one": False,
            "avg_ask_ambiguous_one": False,
            "big_buy_ambiguous_one": False,
            "big_sell_ambiguous_one": False,
            "depth_reference_invalid": False,
            "large_fill_reference_invalid": False,
            **null_statuses,
        },
        {
            "Date": DATE,
            "QuoteCode": "D",
            "ChannelSeq": 4,
            "membership_mask": 0,
            "conservative_base": False,
            "legacy_dgb": True,
            "conservative_dgb": False,
            "pass_d_legacy": True,
            "pass_d": False,
            "pass_g": True,
            "pass_b": True,
            "InOut": -1,
            "_md_price_change": False,
            "_large_fill": True,
            "large_fill_conservative": False,
            "fresh_branch_conservative": "invalid_large_fill_only",
            "avg_bid_ambiguous_one": True,
            "avg_ask_ambiguous_one": False,
            "big_buy_ambiguous_one": False,
            "big_sell_ambiguous_one": True,
            "depth_reference_invalid": True,
            "large_fill_reference_invalid": True,
            "avg_bidLots1_lineage_status": "stored_one_mismatch",
            "avg_askLots1_lineage_status": "not_ambiguous",
            "big_buy_lots_lineage_status": "not_ambiguous",
            "big_sell_lots_lineage_status": "unresolved_missing_raw_support",
        },
    ]
    if duplicate:
        rows.append(dict(rows[0]))
    return pl.DataFrame(rows).with_columns(pl.col("membership_mask").cast(pl.UInt16))


def _reference_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Date": [DATE],
            "QuoteCode": ["D"],
            "has_invalid_ambiguous_reference": [True],
            "avg_bidLots1_lineage_status": ["stored_one_mismatch"],
            "avg_askLots1_lineage_status": ["not_ambiguous"],
            "big_buy_lots_lineage_status": ["not_ambiguous"],
            "big_sell_lots_lineage_status": ["unresolved_missing_raw_support"],
        }
    )


def _calendar_frame(*, incomplete: bool = False) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Date": [DATE],
            "evidence_class": ["retrospective_diagnostic"],
            "source_complete": [True],
            "run_status": ["not_run" if incomplete else "success"],
            "session_status": [
                "not_run_or_incomplete" if incomplete else "evaluable"
            ],
            "membership_complete": [not incomplete],
            "aggregate_complete": [not incomplete],
            "eligible_pool_rows": [100],
            "control_n_z": [80],
        }
    )


class SyntheticInputs:
    def __init__(self, root: Path, *, duplicate: bool = False, incomplete: bool = False):
        self.run_dir = root / "run"
        self.run_dir.mkdir()
        (self.run_dir / "membership_daily").mkdir()
        (self.run_dir / "membership_lineage").mkdir()
        self.calendar_path = root / "requested_calendar.parquet"
        self.calendar_manifest_path = root / "requested_calendar.json"
        self.reference_path = root / "reference_lineage.parquet"
        self.reference_manifest_path = root / "reference_lineage.json"
        self.output_dir = root / "output"

        reference = _reference_frame()
        reference.write_parquet(self.reference_path)
        status_counts = {
            column: study._value_counts(reference, status_column)
            for column, status_column in zip(
                study.REFERENCE_COLUMNS, study.REFERENCE_STATUS_COLUMNS
            )
        }
        _write_json(
            self.reference_manifest_path,
            {
                "contract": "fh_neg_rejection_reference_lineage_audit/v1",
                "outcome_columns_read": [],
                "start": DATE,
                "end": DATE,
                "ambiguous_stock_sessions": 1,
                "invalid_or_mismatched_stock_sessions": 1,
                "reference_status_counts": status_counts,
            },
        )

        membership = _membership_frame(duplicate=duplicate)
        membership_path = self.run_dir / "membership_daily" / f"{DATE}.parquet"
        membership.write_parquet(membership_path)
        source_identity = {
            source: {
                "path": f"/synthetic/{source}/{DATE}",
                "size": index + 10,
                "mtime_ns": index + 100,
            }
            for index, source in enumerate(
                ("tick_data", "tick_feature", "pre_market", "txf_tick")
            )
        }
        design_fingerprint = "a" * 64
        lineage = {
            "contract": study.CONTRACT_ID,
            "frozen_contract_sha256": study.CONTRACT_SHA256,
            "design_fingerprint": design_fingerprint,
            "family_id": study.FAMILY_ID,
            "family_version": study.FAMILY_VERSION,
            "Date": DATE,
            "membership_artifact": study._file_identity(membership_path),
            "reference_lineage_artifact": study._file_identity(self.reference_path),
            "source_identities": source_identity,
            "source_identities_sha256": study._canonical_json_sha256(source_identity),
            "legacy_parity": {
                "status": "pass",
                "expected_legacy_rows": 2,
                "rebuilt_legacy_rows": 2,
                "missing_from_rebuilt": 0,
                "extra_in_rebuilt": 0,
            },
            **study._lineage_expected_counts(membership),
        }
        _write_json(
            self.run_dir / "membership_lineage" / f"{DATE}.json", lineage
        )
        _write_json(
            self.run_dir / "run_manifest.json",
            {
                "contract": study.CONTRACT_ID,
                "family_id": study.FAMILY_ID,
                "family_version": study.FAMILY_VERSION,
                "frozen_contract_sha256": study.CONTRACT_SHA256,
                "label_profile_id": study.CONTRACT["directional_label"]["profile_id"],
                "pool_id": study.CONTRACT["sample_unit"]["eligible_pool_id"],
                "pool_version": "1",
                "primary_estimand": "shared_pool",
                "stock_time_bucket": "5m",
                "control_sample_size": None,
                "primary_membership": "conservative_reference_validated",
                "legacy_parity_membership": "diagnostic_only",
                "candidate_frame_builds_per_processed_day": 1,
                "step": 1,
                "candidate_ids": list(study.CELL_IDS),
                "factorial_design_fingerprint": design_fingerprint,
                "sessions": [DATE],
                "failures": [],
                "source_identities": {DATE: source_identity},
                "reference_lineage_artifact": study._file_identity(
                    self.reference_path
                ),
            },
        )
        calendar = _calendar_frame(incomplete=incomplete)
        calendar.write_parquet(self.calendar_path)
        _write_json(
            self.calendar_manifest_path,
            {
                "contract": "fh_neg_rejection_requested_calendar/v1",
                "factorial_contract": study.CONTRACT_ID,
                "factorial_family_id": study.FAMILY_ID,
                "factorial_family_version": study.FAMILY_VERSION,
                "requested_sessions": 1,
                "evidence_class": "retrospective_diagnostic",
                "source_complete_sessions": 1,
                "successful_sessions": 0 if incomplete else 1,
                "evaluable_sessions": 0 if incomplete else 1,
                "zero_control_sessions": 0,
                "missing_source_sessions": 0,
                "failed_run_sessions": 0,
                "not_run_or_incomplete_sessions": 1 if incomplete else 0,
                "coverage_limitation": "synthetic calendar has no exchange source",
            },
        )

    def build(self):
        return study.build_lineage_support_summary(
            run_dir=self.run_dir,
            requested_calendar_path=self.calendar_path,
            requested_calendar_manifest_path=self.calendar_manifest_path,
            reference_lineage_path=self.reference_path,
            reference_lineage_manifest_path=self.reference_manifest_path,
        )


class LineageSupportSummaryTest(unittest.TestCase):
    def test_completed_run_is_validated_and_summarized_without_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs = SyntheticInputs(Path(directory))
            summary, sessions, cohorts, reference_status, selector = inputs.build()
            study.write_outputs(
                output_dir=inputs.output_dir,
                summary=summary,
                session_frame=sessions,
                support_frame=cohorts,
                reference_status_frame=reference_status,
                selector_frame=selector,
            )
            self.assertEqual(summary["outcome_columns_read"], [])
            self.assertEqual(summary["support"]["legacy_base"]["events"], 4)
            self.assertEqual(summary["support"]["conservative_base"]["events"], 3)
            self.assertEqual(summary["conservative_reference_removals"]["dgb"]["events"], 1)
            self.assertEqual(
                summary["fresh_branches"]["large_fill_only"]["large_fill_side"]["sell"]["events"],
                1,
            )
            self.assertEqual(summary["selector_static_audit"]["status"], "pass")
            self.assertEqual(sessions["membership_validated"].to_list(), [True])
            expected = {
                "lineage_support_summary.json",
                "session_support.parquet",
                "cohort_support.parquet",
                "reference_status.parquet",
                "selector_input_audit.parquet",
            }
            self.assertEqual({path.name for path in inputs.output_dir.iterdir()}, expected)

    def test_duplicate_canonical_membership_key_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs = SyntheticInputs(Path(directory), duplicate=True)
            with self.assertRaisesRegex(ValueError, "duplicate canonical"):
                inputs.build()
            self.assertFalse(inputs.output_dir.exists())

    def test_reference_artifact_identity_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs = SyntheticInputs(Path(directory))
            manifest_path = inputs.run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["reference_lineage_artifact"]["sha256"] = "0" * 64
            _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "reference_lineage_artifact"):
                inputs.build()

    def test_incomplete_requested_calendar_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs = SyntheticInputs(Path(directory), incomplete=True)
            with self.assertRaisesRegex(ValueError, "not a completed run"):
                inputs.build()

    def test_static_audit_rejects_future_named_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = root / "runner.py"
            profile = root / "profile.py"
            mapping = (
                "{'tick_data': ('QuoteCode', 'ChannelSeq', 'future_return'), "
                "'tick_feature': ('QuoteCode', 'ChannelSeq'), "
                "'pre_market': ('QuoteCode',)}"
            )
            runner.write_text(f"REQUIRED_SELECTOR_COLUMNS = {mapping}\n", encoding="utf-8")
            profile.write_text(
                "def build_filllots_at_high_candidate_frame(day):\n"
                "    _require_columns(tick, ('QuoteCode', 'ChannelSeq', 'future_return'))\n"
                "    _require_columns(tick_feature, ('QuoteCode', 'ChannelSeq'))\n"
                "    _require_columns(pre_market, ('QuoteCode',))\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "future/outcome audit failed"):
                study.audit_selector_sources(runner_source=runner, profile_source=profile)

    def test_static_audit_rejects_negative_shift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = root / "runner.py"
            profile = root / "profile.py"
            mapping = (
                "{'tick_data': ('QuoteCode', 'ChannelSeq'), "
                "'tick_feature': ('QuoteCode', 'ChannelSeq'), "
                "'pre_market': ('QuoteCode',)}"
            )
            runner.write_text(f"REQUIRED_SELECTOR_COLUMNS = {mapping}\n", encoding="utf-8")
            profile.write_text(
                "def build_filllots_at_high_candidate_frame(day):\n"
                "    _require_columns(tick, ('QuoteCode', 'ChannelSeq'))\n"
                "    _require_columns(tick_feature, ('QuoteCode', 'ChannelSeq'))\n"
                "    _require_columns(pre_market, ('QuoteCode',))\n"
                "    return x.shift(-1)\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "negative_shift"):
                study.audit_selector_sources(runner_source=runner, profile_source=profile)


if __name__ == "__main__":
    unittest.main(verbosity=2)
