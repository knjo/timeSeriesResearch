from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd
import polars as pl

from src.research.intradaySignal.framework.registry import GateSpec, ModelSpec, RollingSelection, SignalConfig
from src.research.intradaySignal.layers.day_selector import write_day_scores
from src.research.intradaySignal.layers.model import (
    SignalThresholdResult,
    load_signal_rows,
    prepare_signal_frame,
    select_signal,
)
from src.research.intradaySignal.signals._short_common import short_day_selector


DATE = "20260110"


def _gate(frame: pl.DataFrame, params) -> pl.Series:
    return frame.select((pl.col("x") >= params["x_min"]).alias("g"))["g"]


def _identity(df: pd.DataFrame) -> pd.DataFrame:
    return df


def _signal(use_model_gate: bool = True) -> SignalConfig:
    return SignalConfig(
        st_name="sig",
        version="1",
        envelope=lambda day: None,
        envelope_required_sources=("tick_feature",),
        gate=GateSpec("g", "1", {"x_min": 0.0}, _gate),
        selection=RollingSelection(min_rows=1, min_stock_days=1),
        selection_pool=lambda f: f,
        day_selector=short_day_selector("sig_day"),
        model=ModelSpec(
            label=_identity, train_filter=_identity, features=lambda df: df.assign(f1=df["x"] * 2),
            feature_columns=("f1",), timing_target="y", selection_target="y", raw_label_col="y",
            use_model_gate=use_model_gate,
        ),
    )


class ModelLayerTest(unittest.TestCase):
    def test_rows_enter_the_model_after_points_and_day_selector(self) -> None:
        signal = _signal()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "store").mkdir(); (root / "points").mkdir(); (root / "params").mkdir()
            t = datetime(2026, 1, 10, 10, 0)
            pl.DataFrame({
                "Date": [DATE] * 4, "QuoteCode": ["A", "A", "B", "C"], "ChannelSeq": [1, 2, 1, 1],
                "TransTime": [t] * 4, "x": [1.0, 2.0, 3.0, 4.0], "y": [0.1, 0.2, 0.3, 0.4], "cond_sig": [True] * 4,
            }, schema_overrides={"ChannelSeq": pl.UInt64}).write_parquet(root / "store" / f"{DATE}.parquet")
            # layer 2 kept A#1, B#1, C#1 (A#2 removed); layer 3 excluded C
            pl.DataFrame({"Date": [DATE] * 3, "QuoteCode": ["A", "B", "C"], "ChannelSeq": [1, 1, 1]},
                         schema_overrides={"ChannelSeq": pl.UInt64}).write_parquet(root / "points" / "sig_points.parquet")
            write_day_scores(signal, pl.DataFrame({"Date": [DATE] * 3, "QuoteCode": ["A", "B", "C"], "score": [1.0, 0.5, -1.0],
                                                   "kept": [True, True, False], "reason": ["scored"] * 3}), params_dir=root / "params")
            rows = load_signal_rows(signal, store_dir=root / "store", points_dir=root / "points", params_dir=root / "params", dates=[DATE])
            self.assertEqual(sorted(zip(rows["QuoteCode"], rows["ChannelSeq"])), [("A", 1), ("B", 1)])
            without = load_signal_rows(signal, store_dir=root / "store", points_dir=root / "points", params_dir=root / "params", dates=[DATE], use_day_selector=False)
            self.assertEqual(len(without), 3)
            frame = prepare_signal_frame(signal, rows)
            self.assertEqual(frame["signal_id"].unique().tolist(), ["sig"])
            self.assertEqual(frame["f1"].tolist(), [2.0, 6.0])

    def test_select_signal_respects_the_model_gate_switch(self) -> None:
        df = pd.DataFrame({"Date": [DATE] * 3, "QuoteCode": ["A", "B", "C"], "TransTime": [1, 2, 3], "pass_threshold": [True, False, None]})
        thr = SignalThresholdResult("sig", df, {}, {})
        self.assertEqual(select_signal(_signal(True), thr)["QuoteCode"].tolist(), ["A"])
        self.assertEqual(select_signal(_signal(False), thr)["QuoteCode"].tolist(), ["A", "B", "C"])


if __name__ == "__main__":
    unittest.main()
