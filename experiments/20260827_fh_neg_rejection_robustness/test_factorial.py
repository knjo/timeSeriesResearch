"""Focused contracts for the fh_neg_rejection D/G/B factorial runner."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import polars as pl

import run_factorial as study


def _synthetic_candidate_frame() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    sequence = 1
    for pass_d in (False, True):
        for pass_g in (False, True):
            for pass_b in (False, True):
                rows.append(
                    {
                        "QuoteCode": "2330",
                        "ChannelSeq": sequence,
                        "_fresh_event": True,
                        "FillLots_atHigh": -100,
                        "filllots_at_high_depth_ratio": (
                            -8.0 if pass_d else -7.0
                        ),
                        "filllots_at_high_gap_ticks": (
                            4.0 if pass_g else 3.0
                        ),
                        "B1_A1B1": 0.35 if pass_b else 0.36,
                        "MicroMomentum_30": -0.10 if pass_b else -0.09,
                        "LOB_NetPressure_30": 1.0,
                    }
                )
                sequence += 1
    rows.extend(
        (
            {
                "QuoteCode": "2330",
                "ChannelSeq": sequence,
                "_fresh_event": False,
                "FillLots_atHigh": -100,
                "filllots_at_high_depth_ratio": -9.0,
                "filllots_at_high_gap_ticks": 5.0,
                "B1_A1B1": 0.2,
                "MicroMomentum_30": -0.2,
                "LOB_NetPressure_30": -1.0,
            },
            {
                "QuoteCode": "2330",
                "ChannelSeq": sequence + 1,
                "_fresh_event": True,
                "FillLots_atHigh": 100,
                "filllots_at_high_depth_ratio": -9.0,
                "filllots_at_high_gap_ticks": 5.0,
                "B1_A1B1": 0.2,
                "MicroMomentum_30": -0.2,
                "LOB_NetPressure_30": -1.0,
            },
        )
    )
    return pl.DataFrame(rows).with_columns(
        pl.lit(True).alias("_md_price_change"),
        pl.lit(False).alias("_large_fill"),
        pl.lit(-100).alias("FillLots"),
        pl.lit(-1).alias("InOut"),
        pl.lit(20).alias("_large_fill_threshold"),
        pl.lit(10.0).alias("avg_bidLots1"),
        pl.lit(10.0).alias("avg_askLots1"),
        pl.lit(100).alias("big_buy_lots"),
        pl.lit(100).alias("big_sell_lots"),
    )


class FactorialContractTest(unittest.TestCase):
    def test_eight_nested_cells_have_expected_support(self) -> None:
        selected = study.select_factorial_candidates_from_frame(
            _synthetic_candidate_frame()
        )
        self.assertEqual(tuple(selected), study.CELL_IDS)
        self.assertEqual(len(selected), 8)
        for cell in study.FACTOR_CELLS:
            required = sum(
                (
                    cell.require_depth,
                    cell.require_gap,
                    cell.require_bear,
                )
            )
            self.assertEqual(
                len(selected[cell.candidate_id]),
                2 ** (3 - required),
                cell.candidate_id,
            )

    def test_full_cell_is_original_fh_neg_rejection_logic(self) -> None:
        selected = study.select_factorial_candidates_from_frame(
            _synthetic_candidate_frame(),
            candidate_ids=("d1_g1_b1",),
        )["d1_g1_b1"]
        self.assertEqual(selected["ChannelSeq"].to_list(), [8])

    def test_daily_selector_builds_candidate_frame_once(self) -> None:
        class FakeDay:
            def __init__(self) -> None:
                self.required: tuple[str, ...] = ()
                self.date = "20990101"

            def require(self, *sources: str) -> None:
                self.required = tuple(sources)

        day = FakeDay()
        with patch.object(
            study,
            "build_filllots_at_high_candidate_frame",
            return_value=_synthetic_candidate_frame(),
        ) as builder:
            selected = study._select_candidates(
                day,
                registry=study.FACTORIAL_REGISTRY,
                candidate_ids=study.CELL_IDS,
            )
        builder.assert_called_once_with(day)
        self.assertEqual(day.required, study.REQUIRED_SOURCES)
        self.assertEqual(len(selected), 8)

    def test_output_is_confined_to_new_study_artifacts(self) -> None:
        accepted = study.resolve_output_dir(
            study.ARTIFACT_ROOT / "unit_test_v1"
        )
        self.assertEqual(accepted, (study.ARTIFACT_ROOT / "unit_test_v1").resolve())
        rejected = (
            study.ARTIFACT_ROOT,
            study.PROJECT_ROOT / "src/research/intradaySignal/data/new",
            study.PROJECT_ROOT
            / "src/research/intradaySignal/experiments/20260826_filllots_at_high_signal/artifacts/new",
        )
        for path in rejected:
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    study.resolve_output_dir(path)

    def test_design_fingerprint_is_bound_to_cache_version(self) -> None:
        self.assertEqual(len(study.DESIGN_FINGERPRINT), 64)
        self.assertEqual(study.FAMILY_VERSION, "1")
        self.assertEqual(study.CONTRACT_ID, study.FROZEN_CONTRACT["contract_id"])
        self.assertIn("d1_g1_b1", study.CELL_DESCRIPTIONS)

    def test_ambiguous_one_fails_closed_but_legacy_membership_survives(self) -> None:
        frame = _synthetic_candidate_frame().filter(
            pl.col("ChannelSeq") == 8
        ).with_columns(
            pl.lit(1.0).alias("avg_bidLots1"),
            pl.lit(False).alias("_md_price_change"),
            pl.lit(True).alias("_large_fill"),
            pl.lit(1).alias("big_sell_lots"),
        )
        lineage = pl.DataFrame(
            {
                "QuoteCode": ["2330"],
                "avg_bidLots1_lineage_status": ["stored_one_mismatch"],
                "avg_askLots1_lineage_status": ["not_ambiguous"],
                "big_buy_lots_lineage_status": ["not_ambiguous"],
                "big_sell_lots_lineage_status": [
                    "unresolved_missing_raw_support"
                ],
            }
        )
        gates = study.build_factorial_gate_frame(
            frame,
            reference_lineage=lineage,
        )
        self.assertTrue(gates["legacy_dgb"][0])
        self.assertFalse(gates["conservative_dgb"][0])
        self.assertFalse(gates["conservative_base"][0])
        self.assertTrue(gates["depth_reference_invalid"][0])
        self.assertTrue(gates["large_fill_reference_invalid"][0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
