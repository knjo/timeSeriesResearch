from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import polars as pl


MODULE_PATH = Path(__file__).with_name("run_backtest.py")
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("netfill_long_cover_runner", MODULE_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot import {MODULE_PATH}")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class DrawdownTests(unittest.TestCase):
    def test_initial_zero_is_in_high_water_mark(self) -> None:
        cumulative, high_water, drawdown = RUNNER._drawdown(
            np.array([-5.0, 2.0, 1.0])
        )
        self.assertTrue(np.allclose(cumulative, [-5.0, -3.0, -2.0]))
        self.assertTrue(np.allclose(high_water, [0.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(drawdown, [-5.0, -3.0, -2.0]))


class CapacityTests(unittest.TestCase):
    def test_capacity_uses_ask_and_is_candidate_specific(self) -> None:
        frame = pl.DataFrame(
            {
                "candidate_id": ["a", "a", "b"],
                "Date": ["20250102"] * 3,
                "QuoteCode": ["1234"] * 3,
                "TransTime": [
                    "2025-01-02 09:01:00",
                    "2025-01-02 09:02:00",
                    "2025-01-02 09:01:30",
                ],
                "ChannelSeq": [1, 2, 1],
                "AskPrice1": [100.0, 101.0, 100.0],
                "avg_bidLots1": [2.5] * 3,
                "avg_askLots1": [2.5] * 3,
            }
        ).with_columns(
            pl.col("TransTime").str.to_datetime(),
            pl.col("ChannelSeq").cast(pl.UInt64),
        )
        result = RUNNER._apply_capacity(frame)
        self.assertEqual(len(result), 3)
        candidate_a = result.filter(pl.col("candidate_id") == "a").sort(
            "ChannelSeq"
        )
        self.assertEqual(candidate_a["accLots"].to_list(), [1, 2])
        self.assertTrue(
            np.allclose(candidate_a["stock_position_wan"].to_numpy(), [10.0, 20.1])
        )
        candidate_b = result.filter(pl.col("candidate_id") == "b")
        self.assertEqual(candidate_b["accLots"].to_list(), [1])


class DailyPnlTests(unittest.TestCase):
    def test_daily_uses_ask_entry_and_retains_zero_days(self) -> None:
        schema = RUNNER._empty_accepted().schema
        row = {column: None for column in schema}
        row.update(
            {
                "contract_sha256": "x",
                "Date": "20250102",
                "candidate_id": RUNNER.CANDIDATES[0],
                "candidate_family": "FillLots_atHigh",
                "QuoteCode": "1234",
                "ChannelSeq": 1,
                "BidPrice1": 90.0,
                "AskPrice1": 100.0,
                "label_raw_bp": 30.0,
                "label_residual_bp": 35.0,
            }
        )
        signal = pl.DataFrame([row], schema=schema, strict=False)
        daily = RUNNER._daily_frame(
            signal,
            ["20250102", "20250103"],
            fee_bp=20.0,
        )
        selected = daily.filter(
            pl.col("candidate_id") == RUNNER.CANDIDATES[0]
        ).sort("Date")
        self.assertTrue(np.allclose(selected["gross_pnl"].to_numpy(), [300.0, 0.0]))
        self.assertTrue(np.allclose(selected["net_pnl"].to_numpy(), [100.0, 0.0]))
        self.assertTrue(
            np.allclose(selected["position"].to_numpy(), [100_000.0, 0.0])
        )

    def test_missing_residual_is_diagnostic_only_not_raw_trade_eligibility(self) -> None:
        schema = RUNNER._empty_accepted().schema
        rows = []
        for channel_seq, ask, raw_bp, residual_bp in (
            (1, 100.0, 30.0, None),
            (2, 200.0, 40.0, 50.0),
        ):
            row = {column: None for column in schema}
            row.update(
                {
                    "contract_sha256": "x",
                    "Date": "20250102",
                    "candidate_id": RUNNER.CANDIDATES[0],
                    "candidate_family": "FillLots_atHigh",
                    "QuoteCode": "1234",
                    "ChannelSeq": channel_seq,
                    "AskPrice1": ask,
                    "label_raw_bp": raw_bp,
                    "label_residual_bp": residual_bp,
                }
            )
            rows.append(row)
        signals = pl.DataFrame(rows, schema=schema, strict=False)

        daily = RUNNER._daily_frame(signals, ["20250102"], fee_bp=20.0)
        selected = daily.filter(
            pl.col("candidate_id") == RUNNER.CANDIDATES[0]
        ).row(0, named=True)

        self.assertEqual(selected["trades"], 2)
        self.assertTrue(np.isclose(selected["position"], 300_000.0))
        self.assertTrue(np.isclose(selected["gross_pnl"], 1_100.0))
        self.assertTrue(np.isclose(selected["net_pnl"], 500.0))
        self.assertEqual(selected["residual_trades"], 1)
        self.assertTrue(np.isclose(selected["residual_position"], 200_000.0))
        self.assertTrue(np.isclose(selected["residual_gross_pnl"], 1_000.0))
        self.assertTrue(np.isclose(selected["residual_net_pnl"], 600.0))

    def test_all_zero_support_candidates_are_retained_on_dense_calendar(self) -> None:
        calendar = ["20250102", "20250103", "20250106"]
        daily = RUNNER._daily_frame(
            RUNNER._empty_accepted(),
            calendar,
            fee_bp=RUNNER.PRIMARY_FEE_BP,
        )

        self.assertEqual(len(daily), len(calendar) * len(RUNNER.CANDIDATES))
        self.assertEqual(
            set(daily["candidate_id"].unique().to_list()),
            set(RUNNER.CANDIDATES),
        )
        for column in (
            "gross_pnl",
            "net_pnl",
            "residual_gross_pnl",
            "residual_net_pnl",
            "residual_position",
            "position",
            "trades",
            "residual_trades",
            "stocks",
            "cumulative_net_pnl",
            "high_water_mark",
            "drawdown",
        ):
            self.assertTrue(
                np.allclose(daily[column].to_numpy(), 0.0),
                msg=column,
            )


class ContractTests(unittest.TestCase):
    def test_contract_candidate_order_is_exact(self) -> None:
        contract, digest = RUNNER._read_contract()
        self.assertEqual(contract["candidate_family"]["trial_count"], 6)
        self.assertEqual(len(digest), 64)

    def test_runtime_contract_and_materialization_identity_are_fail_closed(self) -> None:
        contract, contract_sha = RUNNER._read_contract()
        first_payload, first_json, first_sha = RUNNER._materialization_identity(
            contract_sha
        )
        second_payload, second_json, second_sha = RUNNER._materialization_identity(
            contract_sha
        )

        self.assertEqual(contract["runtime_assertions"]["artifact_version"], "static_long_cover_v2")
        self.assertEqual(first_payload, second_payload)
        self.assertEqual(first_json, second_json)
        self.assertEqual(first_sha, second_sha)
        self.assertEqual(first_payload["contract_sha256"], contract_sha)
        self.assertEqual(
            first_json,
            json.dumps(
                first_payload,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        self.assertEqual(
            first_sha,
            hashlib.sha256(first_json.encode("utf-8")).hexdigest(),
        )

        with mock.patch.object(RUNNER, "PRICE_CAP", RUNNER.PRICE_CAP + 1.0):
            with self.assertRaisesRegex(RuntimeError, "runtime assertions drifted"):
                RUNNER._read_contract()
            _, _, drifted_sha = RUNNER._materialization_identity(contract_sha)
        self.assertNotEqual(first_sha, drifted_sha)


class FamilyBiasTests(unittest.TestCase):
    def test_zero_support_trials_are_retained_as_non_estimable(self) -> None:
        calendar = [f"202501{day:02d}" for day in range(1, 31)]
        daily = RUNNER._daily_frame(
            RUNNER._empty_accepted(),
            calendar,
            fee_bp=RUNNER.PRIMARY_FEE_BP,
        )
        short = pl.DataFrame(
            {
                "Date": calendar,
                "net_pnl": np.linspace(-15.0, 14.0, len(calendar)),
            }
        )
        contract, _ = RUNNER._read_contract()
        contract = copy.deepcopy(contract)
        selection = contract["selection_bias_accounting"]
        selection["bootstrap_repetitions"] = 49
        selection["block_sensitivity_repetitions"] = 29

        result = RUNNER._build_family_bias_adjustment(daily, short, contract)

        self.assertEqual(result["moving_block_max_t"]["trial_count"], 6)
        self.assertEqual(
            result["current_family_dependence"]["non_estimable_candidates"],
            list(RUNNER.CANDIDATES),
        )
        self.assertIsNone(
            result["current_family_dependence"]["effective_trial_count"]
        )
        for candidate in RUNNER.CANDIDATES:
            primary = result["moving_block_max_t"]["trials"][candidate]
            tail = result["cover_tail_moving_block_max_t"]["trials"][candidate]
            self.assertFalse(primary["estimable"], msg=candidate)
            self.assertTrue(np.isclose(primary["max_t_adjusted_p"], 1.0))
            self.assertFalse(tail["estimable"], msg=candidate)
            self.assertTrue(np.isclose(tail["max_t_adjusted_p"], 1.0))
            self.assertTrue(
                np.isclose(
                    tail[
                        "conditional_mean_native_long_pnl_on_worst_short_days"
                    ],
                    0.0,
                )
            )
            for trial_count in selection["dsr_like_trial_counts"]:
                dsr = result["dsr_like"]["candidates"][candidate][
                    str(trial_count)
                ]
                self.assertFalse(dsr["estimable"], msg=(candidate, trial_count))


class DecisionClassificationTests(unittest.TestCase):
    def test_different_candidates_cannot_jointly_pass(self) -> None:
        decision, joint = RUNNER._classify_decision(["a"], ["b"])
        self.assertEqual(
            decision,
            "RETROSPECTIVE_STANDALONE_AND_COVER_SCREENS_ARE_DISJOINT",
        )
        self.assertEqual(joint, [])

    def test_same_candidate_must_pass_both_gates(self) -> None:
        decision, joint = RUNNER._classify_decision(["a", "b"], ["b", "c"])
        self.assertEqual(
            decision,
            "RETROSPECTIVE_JOINT_LONG_COVER_SCREEN_PASSED",
        )
        self.assertEqual(joint, ["b"])

    def test_build_decision_requires_same_candidate_gate_intersection(self) -> None:
        standalone_only = RUNNER.CANDIDATES[0]
        cover_only = RUNNER.CANDIDATES[1]
        periods = []
        for candidate in RUNNER.CANDIDATES:
            nonnegative = candidate == standalone_only
            for period in (
                "full_retrospective",
                "year_2025",
                "year_2026_historical",
            ):
                periods.append(
                    {
                        "candidate_id": candidate,
                        "period": period,
                        "cap_weighted_net_bp": 1.0 if nonnegative else -1.0,
                    }
                )
        period_summary = pl.DataFrame(periods)

        family_bias = {
            "moving_block_max_t": {
                "trials": {
                    candidate: {
                        "max_t_adjusted_p": (
                            0.05 if candidate == standalone_only else 1.0
                        )
                    }
                    for candidate in RUNNER.CANDIDATES
                }
            },
            "cover_tail_moving_block_max_t": {
                "trials": {
                    candidate: {
                        "max_t_adjusted_p": (
                            0.05 if candidate == cover_only else 1.0
                        )
                    }
                    for candidate in RUNNER.CANDIDATES
                }
            },
            "dsr_like": {
                "candidates": {
                    candidate: {
                        "28": {"dsr_probability": None},
                    }
                    for candidate in RUNNER.CANDIDATES
                }
            },
        }
        cover_rows = []
        for candidate in RUNNER.CANDIDATES:
            cover_pass = candidate == cover_only
            cover_rows.append(
                {
                    "candidate_id": candidate,
                    "accounting_mode": "native_cash_add_on_primary",
                    "long_multiplier_h": 0.5,
                    "mean_native_long_pnl_on_worst_short_days": (
                        1.0 if cover_pass else -1.0
                    ),
                    "pearson_net_daily_pnl": -0.1 if cover_pass else 0.1,
                    "spearman_net_daily_pnl": -0.1 if cover_pass else 0.1,
                    "calendar_sharpe_delta_vs_short": (
                        0.1 if cover_pass else -0.1
                    ),
                    "max_drawdown_delta_vs_short": (
                        1.0 if cover_pass else -1.0
                    ),
                }
            )
        cover_summary = pl.DataFrame(cover_rows)
        contract, _ = RUNNER._read_contract()

        result = RUNNER._build_decision(
            period_summary,
            family_bias,
            cover_summary,
            contract,
        )

        self.assertEqual(
            result["decision"],
            "RETROSPECTIVE_STANDALONE_AND_COVER_SCREENS_ARE_DISJOINT",
        )
        self.assertEqual(
            result["standalone_research_screen_candidates"],
            [standalone_only],
        )
        self.assertEqual(result["cover_research_screen_candidates"], [cover_only])
        self.assertEqual(result["joint_research_screen_candidates"], [])


if __name__ == "__main__":
    unittest.main()
