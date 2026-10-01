from __future__ import annotations

import json
import unittest

import polars as pl

from src.research.intradaySignal.framework.dataset import ALL_EVENTS_UNIVERSE, UniverseSpec
from src.research.intradaySignal.framework.registry import (
    DaySelectorSpec,
    GateSpec,
    RiskModelSpec,
    RollingSelection,
    SignalConfig,
    SignalRegistry,
    param_set_id,
)
from src.research.intradaySignal.framework.store import _tag_frame
from src.research.intradaySignal.layers.envelope import candidate_pool, compute_daily_aggregates, params_for_date


def _gate(frame: pl.DataFrame, params) -> pl.Series:
    return frame.select((pl.col("x") >= params["x_min"]).alias("g"))["g"]


def _selector(day):  # pragma: no cover - structural placeholder
    raise AssertionError("envelope selector must not run in unit tests")


def _pool(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(pl.col("y").alias("label_residual_bp"))


def _signal(grid=None, universe=ALL_EVENTS_UNIVERSE, point_filter=None) -> SignalConfig:
    gate = GateSpec("g", "1", {"x_min": 1.0}, _gate, grid=grid or {})
    return SignalConfig(
        st_name="sig",
        version="1",
        envelope=_selector,
        envelope_required_sources=("tick_feature",),
        gate=gate,
        selection_pool=_pool,
        selection=RollingSelection(window_days=3, min_rows=1, min_stock_days=1),
        universe=universe,
        point_filter=point_filter,
    )


class GateSpecTest(unittest.TestCase):
    def test_grid_expansion_puts_default_first_and_is_stable(self) -> None:
        gate = GateSpec("g", "1", {"a": 1, "b": 2.0}, _gate, grid={"a": (3, 1, 2), "b": (2.0, 4.0)})
        combos = gate.expand_grid()
        self.assertEqual(len(combos), 6)
        self.assertEqual(combos[0], {"a": 1, "b": 2.0})
        self.assertEqual(param_set_id(combos[0]), gate.default_param_set_id)
        self.assertEqual(len({param_set_id(c) for c in combos}), 6)

    def test_grid_rejects_unknown_parameter(self) -> None:
        with self.assertRaises(ValueError):
            GateSpec("g", "1", {"a": 1}, _gate, grid={"zzz": (1, 2)})

    def test_only_condition_gates_carry_a_grid(self) -> None:
        with self.assertRaises(ValueError):
            GateSpec("g", "1", {"x_min": 1.0}, _gate, grid={"x_min": (1.0, 2.0)}, stage="point_filter")
        with self.assertRaises(ValueError):
            GateSpec("g", "1", {"x_min": 1.0}, _gate, stage="post_model")

    def test_mask_enforces_boolean_contract_and_unknown_params(self) -> None:
        gate = GateSpec("g", "1", {"x_min": 1.0}, _gate)
        frame = pl.DataFrame({"x": [0.0, 1.0, 2.0, None]})
        self.assertEqual(gate.mask(frame).to_list(), [False, True, True, False])
        self.assertEqual(gate.mask_pd(frame.to_pandas()).tolist(), [False, True, True, False])
        with self.assertRaises(ValueError):
            gate.mask(frame, {"nope": 1})


class SignalConfigTest(unittest.TestCase):
    def test_sections_are_stage_checked(self) -> None:
        with self.assertRaises(ValueError):
            _signal(point_filter=GateSpec("p", "1", {"x_min": 0.0}, _gate, stage="condition"))
        with self.assertRaises(ValueError):
            DaySelectorSpec(selector_id="d", labels=_pool, models=(RiskModelSpec(model_id="m", target="ds_y", threshold=0.0),),
                            hard_rules=GateSpec("h", "1", {"x_min": 0.0}, _gate))
        signal = _signal(point_filter=GateSpec("p", "1", {"x_min": 0.0}, _gate, stage="point_filter"))
        self.assertEqual(signal.condition_id, "sig_v1")
        self.assertEqual(signal.tag_column, "cond_sig")
        self.assertIsNone(signal.day_selector)
        self.assertIsNone(signal.model)

    def test_duplicate_names_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SignalRegistry((_signal(), _signal()))

    def test_select_unknown_name_fails(self) -> None:
        registry = SignalRegistry((_signal(),))
        with self.assertRaises(KeyError):
            registry.select(["other"])
        self.assertEqual(registry.select(None)[0].st_name, "sig")


class TagFrameTest(unittest.TestCase):
    def test_tags_are_envelope_hit_and_universe_and_untagged_rows_drop(self) -> None:
        universe = UniverseSpec("u", "1", lambda f: f.filter(pl.col("ok")))
        signal = _signal(universe=universe)
        frame = pl.DataFrame(
            {
                "QuoteCode": ["A", "A", "B", "C"],
                "ChannelSeq": [1, 2, 1, 1],
                "ok": [True, False, True, True],
                "x": [1.0, 1.0, 1.0, 1.0],
            }
        ).with_columns(pl.col("ChannelSeq").cast(pl.UInt64))
        keys = pl.DataFrame({"QuoteCode": ["A", "A", "B"], "ChannelSeq": [1, 2, 1]}).with_columns(
            pl.col("ChannelSeq").cast(pl.UInt64)
        )
        tagged = _tag_frame(frame, {"sig": keys}, [signal])
        self.assertEqual(tagged.select("QuoteCode", "ChannelSeq").rows(), [("A", 1), ("B", 1)])
        self.assertTrue(tagged["cond_sig"].all())


class ParamsTest(unittest.TestCase):
    def test_daily_aggregates_are_per_param_set_and_stock_day(self) -> None:
        signal = _signal(grid={"x_min": (1.0, 2.0)})
        pool = pl.DataFrame(
            {
                "Date": ["20260101"] * 4,
                "QuoteCode": ["A", "A", "B", "B"],
                "x": [1.0, 2.0, 2.0, 0.5],
                "y": [10.0, 20.0, 30.0, 40.0],
            }
        )
        agg = compute_daily_aggregates(signal, candidate_pool(signal, pool))
        default_id = signal.gate.default_param_set_id
        strict_id = param_set_id({"x_min": 2.0})
        by = {(r["param_set_id"], r["QuoteCode"]): r for r in agg.to_dicts()}
        self.assertEqual(by[(default_id, "A")]["n_rows"], 2)
        self.assertEqual(by[(default_id, "A")]["sum_label"], 30.0)
        self.assertEqual(by[(strict_id, "A")]["n_rows"], 1)
        self.assertEqual(by[(strict_id, "B")]["sum_label"], 30.0)

    def test_candidate_pool_applies_the_point_filter_first(self) -> None:
        signal = _signal(point_filter=GateSpec("p", "1", {"x_min": 2.0}, _gate, stage="point_filter"))
        pool = pl.DataFrame({"Date": ["20260101"] * 3, "QuoteCode": ["A", "A", "B"], "x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
        self.assertEqual(candidate_pool(signal, pool)["x"].to_list(), [2.0, 3.0])

    def test_params_for_date_carries_last_winner_then_default(self) -> None:
        signal = _signal()
        table = pl.DataFrame(
            {
                "Date": ["20260102"],
                "param_set_id": ["p_x"],
                "params_json": [json.dumps({"x_min": 5.0})],
                "is_winner": [True],
            }
        )
        self.assertEqual(params_for_date(table, signal, "20260101")[2], "default")
        self.assertEqual(params_for_date(table, signal, "20260102")[0], {"x_min": 5.0})
        params, _, source = params_for_date(table, signal, "20260105")
        self.assertEqual(params, {"x_min": 5.0})
        self.assertTrue(source.startswith("carry:"))


if __name__ == "__main__":
    unittest.main()
