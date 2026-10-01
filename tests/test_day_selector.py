from __future__ import annotations

import tempfile
import unittest

import numpy as np
import pandas as pd
import polars as pl

from src.research.intradaySignal.framework.registry import DaySelectorSpec, GateSpec, RiskModelSpec
from src.research.intradaySignal.layers.day_selector import (
    apply_day_scores_pd,
    apply_day_scores_pl,
    build_day_scores,
    evaluate_day_scores,
    read_day_scores,
    write_day_scores,
)
from src.research.intradaySignal.signals.fh_neg import FH_NEG


def _table(n_days: int = 12, n_stocks: int = 40, seed: int = 0) -> pl.DataFrame:
    """Stock-day table: ds_risk rises with x, ds_stop_hit fires when x is large; weight from turnover."""

    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        date = f"202601{d + 1:02d}"
        for s in range(n_stocks):
            x = rng.normal()
            rows.append({
                "Date": date, "QuoteCode": f"S{s:03d}", "n_rows": 5, "weight": 1.0 if s < 30 else 0.2,
                "ds_risk": 0.5 * x + rng.normal() * 0.3, "ds_stop_hit": float(x > 1.0), "ds_pnl_bp": -20.0 * x + rng.normal() * 5,
                "MarketCap": float(np.exp(x)), "OTC": float(s % 2), "RefPrice": 50.0 + s if s < 38 else 5000.0, "trading_turnover": 1e8 * np.exp(x),
            })
    return pl.DataFrame(rows)


def _cap(frame: pl.DataFrame, params) -> pl.Series:
    return frame.select((pl.col("RefPrice") <= params["ref_price_max"]).alias("g"))["g"]


FEATURES = ("MarketCap", "OTC", "RefPrice", "log_trading_turnover")
RISK = RiskModelSpec(model_id="risk", target="ds_risk", threshold=0.4, task="regression", family="ridge", features=FEATURES)
STOP = RiskModelSpec(model_id="stop", target="ds_stop_hit", threshold=0.5, task="classification", family="logistic", features=FEATURES)


def _spec(models=(RISK, STOP), **kw) -> DaySelectorSpec:
    base = dict(selector_id="t", labels=lambda f: f, models=models, window_days=5, min_train_stock_days=50,
                hard_rules=GateSpec("cap", "1", {"ref_price_max": 1000.0}, _cap, stage="hard_rule"))
    base.update(kw)
    return DaySelectorSpec(**base)


class RiskModelSpecTest(unittest.TestCase):
    def test_contract(self) -> None:
        with self.assertRaises(ValueError):
            RiskModelSpec(model_id="x", target="label_residual_bp", threshold=0.0)      # ds_ prefix required
        with self.assertRaises(ValueError):
            RiskModelSpec(model_id="x", target="ds_a", threshold=0.0, task="classification", family="ridge")
        with self.assertRaises(ValueError):
            DaySelectorSpec(selector_id="d", labels=lambda f: f, models=(RISK, RISK))   # duplicate ids
        spec = _spec()
        self.assertEqual(spec.targets, ("ds_risk", "ds_stop_hit"))
        self.assertEqual(spec.features, FEATURES)
        self.assertEqual(RISK.excludes(np.array([0.3, 0.5])).tolist(), [False, True])


class DaySelectorTest(unittest.TestCase):
    def test_scores_are_strictly_prior_additive_and_hard_ruled(self) -> None:
        table = _table()
        scores = build_day_scores(FH_NEG, params_dir="/tmp", spec=_spec(), table=table, verbose=False)
        first = scores.filter(pl.col("Date") == "20260101")
        self.assertTrue((first["reason"] == "insufficient_history").all())
        self.assertTrue(first["score_risk"].is_null().all())
        self.assertEqual(int((~first["kept"]).sum()), 2)                 # the two 5000-priced stocks: hard rule only
        later = scores.filter(pl.col("Date") == "20260110")
        self.assertEqual(later["window_end"][0], "20260109")
        self.assertEqual(later["window_start"][0], "20260105")
        self.assertEqual(later["n_train"][0], 5 * 40)
        # additive: kept == hard_ok & no model excluded; excluded_by names the models
        expect = later["hard_ok"] & (later["excluded_by"] == "")
        self.assertEqual(later["kept"].to_list(), expect.to_list())
        self.assertTrue(later.filter(pl.col("score_risk") > 0.4)["excluded_by"].str.contains("risk").all())
        self.assertTrue(later.filter(pl.col("score_risk") <= 0.4)["excluded_by"].str.contains("risk").not_().all())
        self.assertGreater(int((~later["kept"]).sum()), 2)

    def test_evaluation_is_weighted_and_per_model(self) -> None:
        table = _table(n_days=20)
        spec = _spec()
        scores = build_day_scores(FH_NEG, params_dir="/tmp", spec=spec, table=table, verbose=False)
        result = evaluate_day_scores(table, scores, spec=spec)
        summary = {r["set"]: r for r in result["summary"].to_dicts()}
        self.assertGreater(summary["kept"]["weighted_pnl_bp"], summary["excluded"]["weighted_pnl_bp"])   # pnl falls with x, risk rises with x
        per_model = result["per_model"]
        self.assertEqual(set(per_model["model"].to_list()), {"risk", "stop"})
        top = per_model.filter((pl.col("model") == "risk") & (pl.col("bucket") == "10"))
        bottom = per_model.filter((pl.col("model") == "risk") & (pl.col("bucket") == "1"))
        self.assertLess(float(top["weighted_pnl_bp"][0]), float(bottom["weighted_pnl_bp"][0]))

    def test_families_instantiate(self) -> None:
        table = _table(n_days=8)
        models = (
            RiskModelSpec(model_id="a", target="ds_risk", threshold=0.4, family="hist_gb", features=FEATURES, hyper={"max_iter": 20}),
            RiskModelSpec(model_id="b", target="ds_stop_hit", threshold=0.5, task="classification", family="hist_gb", features=FEATURES, hyper={"max_iter": 20}),
        )
        scores = build_day_scores(FH_NEG, params_dir="/tmp", spec=_spec(models=models), table=table, verbose=False)
        self.assertTrue((scores.filter(pl.col("Date") == "20260108")["reason"] == "scored").all())

    def test_apply_scores_and_round_trip(self) -> None:
        scores = pl.DataFrame({"Date": ["20260110", "20260110"], "QuoteCode": ["A", "B"], "kept": [True, False], "reason": ["scored", "scored"]})
        rows = pl.DataFrame({"Date": ["20260110", "20260110", "20260110", "20260111"], "QuoteCode": ["A", "B", "C", "A"]})
        self.assertEqual(apply_day_scores_pl(rows, scores).to_list(), [True, False, True, True])              # unscored kept
        self.assertEqual(apply_day_scores_pl(rows, scores, keep_unscored=False).to_list(), [True, False, False, False])
        pdf = pd.DataFrame({"Date": pd.to_datetime(["2026-01-10", "2026-01-10"]), "QuoteCode": ["A", "B"]})
        self.assertEqual(apply_day_scores_pd(pdf, scores).tolist(), [True, False])
        with tempfile.TemporaryDirectory() as d:
            write_day_scores(FH_NEG, scores, params_dir=d, spec=_spec())
            self.assertEqual(read_day_scores(d, FH_NEG).height, 2)


if __name__ == "__main__":
    unittest.main()
