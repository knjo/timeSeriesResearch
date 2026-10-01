from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, time
from pathlib import Path

import polars as pl

from src.research.intradaySignal.framework.dataset import ALL_EVENTS_UNIVERSE
from src.research.intradaySignal.framework.events import DataRoots
from src.research.intradaySignal.framework.profiles.tw_equity import NEXTDAY_ALLOW_DAY_TRADE_UNIVERSE_V1
from src.research.intradaySignal.framework.registry import GateSpec, RollingSelection, SignalConfig, param_set_id
from src.research.intradaySignal.framework.store import (
    StoreContractError,
    build_envelope_index,
    build_store,
    read_envelope_index,
    store_manifest_path,
)
from src.research.intradaySignal.layers.envelope import (
    build_condition_params,
    build_signal_index,
    read_condition_params,
    read_signal_index,
)
from src.research.intradaySignal.layers.point_filter import build_points, load_rows, read_points


DATE = "20240102"


def _roots(root: Path) -> DataRoots:
    roots = DataRoots.from_project_root(root)
    for path in (roots.tick_data_dir, roots.tick_feature_dir, roots.pre_market_dir, roots.tick_bar_dir, roots.txf_tick_dir):
        path.mkdir(parents=True, exist_ok=True)
    return roots


def _write_sources(roots: DataRoots) -> None:
    t = datetime(2024, 1, 2, 9, 7)
    pl.DataFrame(
        {
            "RecvTime": [t, t, t],
            "QuoteCode": ["2317", "2330", "2330"],
            "ChannelSeq": [1, 1, 2],
            "TransTime": [t, t, t],
            "BidPrice1": [50.0, 100.0, 101.0],
        },
        schema_overrides={"ChannelSeq": pl.UInt64},
    ).write_parquet(roots.tick_data_dir / f"{DATE}_StockTick.parquet")
    pl.DataFrame(
        {"QuoteCode": ["2317", "2330", "2330"], "ChannelSeq": [1, 1, 2], "feature_x": [3.0, 2.0, 3.0]},
        schema_overrides={"ChannelSeq": pl.UInt64},
    ).write_parquet(roots.tick_feature_dir / f"{DATE}_tickFeature.parquet")
    pl.DataFrame(
        {
            "QuoteCode": ["2317", "2330"],
            "nextday_allow_day_trade_mark": ["X", "X"],
            "txf_beta_60d": [1.0, 1.1],
            "txf_beta_enrichment_ok": [True, True],
        }
    ).write_parquet(roots.pre_market_dir / f"{DATE}_preMarketData.parquet")
    pl.DataFrame(
        {"QuoteCode": ["2330"], "TimeSlot": [time(9, 5)], "TransTime": [datetime(2024, 1, 2, 9, 4)], "AmountRank": [10]}
    ).write_parquet(roots.tick_bar_dir / f"{DATE}_tickBar.parquet")


def _gate(frame: pl.DataFrame, params) -> pl.Series:
    return frame.select((pl.col("BidPrice1") >= params["bid_min"]).alias("g"))["g"]


def _pool(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(pl.col("BidPrice1").alias("label_residual_bp"))


def _select_a(day):
    return day.scan_tick_feature().filter(pl.col("feature_x") == 2.0).select("QuoteCode", "ChannelSeq")


def _select_b(day):
    return (
        day.scan_tick_feature()
        .filter(pl.col("feature_x") == 3.0)
        .select("QuoteCode", "ChannelSeq", pl.col("feature_x").alias("score"))
    )


SIG_A = SignalConfig(
    st_name="alpha",
    version="1",
    envelope=_select_a,
    envelope_required_sources=("tick_feature",),
    gate=GateSpec("ga", "1", {"bid_min": 0.0}, _gate),
    selection_pool=_pool,
    selection=RollingSelection(window_days=2, min_rows=1, min_stock_days=1),
    universe=NEXTDAY_ALLOW_DAY_TRADE_UNIVERSE_V1,
)
SIG_B = SignalConfig(
    st_name="beta",
    version="1",
    envelope=_select_b,
    envelope_required_sources=("tick_feature",),
    gate=GateSpec("gb", "1", {"bid_min": 0.0}, _gate, grid={"bid_min": (0.0, 60.0)}),
    selection_pool=_pool,
    selection=RollingSelection(window_days=2, min_rows=1, min_stock_days=1, min_stock_days_per_day=0.0),
    universe=ALL_EVENTS_UNIVERSE,
    envelope_columns=("score",),
)
SIG_B_FILTERED = SignalConfig(
    st_name="beta",
    version="1",
    envelope=_select_b,
    envelope_required_sources=("tick_feature",),
    gate=SIG_B.gate,
    selection_pool=_pool,
    selection=SIG_B.selection,
    universe=ALL_EVENTS_UNIVERSE,
    envelope_columns=("score",),
    point_filter=GateSpec("pb", "1", {"bid_min": 60.0}, _gate, stage="point_filter"),
)


class StorePipelineTest(unittest.TestCase):
    def test_envelope_store_patch_params_index_points_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            roots = _roots(root)
            _write_sources(roots)
            env_dir, store_dir = root / "envelope", root / "store"
            params_dir, index_dir, points_dir = root / "params", root / "index", root / "points"

            # envelope: build, then a second run has nothing newer than the watermark
            first = build_envelope_index(SIG_A, roots=roots, envelope_dir=env_dir, verbose=False)
            self.assertEqual(first.processed_dates, (DATE,))
            self.assertEqual(first.total_rows, 1)
            again = build_envelope_index(SIG_A, roots=roots, envelope_dir=env_dir, verbose=False)
            self.assertEqual(again.processed_dates, ())
            self.assertEqual(again.watermark, DATE)

            # store with one signal
            report = build_store([SIG_A], roots=roots, store_dir=store_dir, envelope_dir=env_dir, verbose=False)
            self.assertEqual(report.built_dates, (DATE,))
            day = pl.read_parquet(store_dir / f"{DATE}.parquet")
            self.assertEqual(day.height, 1)
            self.assertTrue(day["cond_alpha"].all())
            self.assertEqual(day["QuoteCode"].to_list(), ["2330"])

            # registering a second signal patches the existing day in place
            build_envelope_index(SIG_B, roots=roots, envelope_dir=env_dir, verbose=False)
            env_b = read_envelope_index(env_dir, SIG_B)
            self.assertEqual(env_b.columns, ["Date", "QuoteCode", "ChannelSeq", "score"])
            report = build_store(
                [SIG_A, SIG_B], roots=roots, store_dir=store_dir, envelope_dir=env_dir, refresh_latest=False, verbose=False,
            )
            self.assertEqual(report.patched_dates, (DATE,))
            day = pl.read_parquet(store_dir / f"{DATE}.parquet").sort(["QuoteCode", "ChannelSeq"])
            self.assertEqual(day.height, 3)
            self.assertEqual(day["cond_alpha"].to_list(), [False, True, False])
            self.assertEqual(day["cond_beta"].to_list(), [True, False, True])
            self.assertEqual(day["beta__score"].to_list(), [3.0, None, 3.0])
            manifest = json.loads(store_manifest_path(store_dir).read_text())
            self.assertEqual(manifest["days"][DATE]["mode"], "patch")
            self.assertEqual(manifest["days"][DATE]["conditions"]["alpha"]["rows"], 1)
            self.assertEqual(manifest["days"][DATE]["conditions"]["beta"]["rows"], 2)
            self.assertEqual(manifest["days"][DATE]["conditions"]["beta"]["envelope_rows"], 2)

            # the store is shared: a subset build would drop the other signal's rows -> refused
            with self.assertRaises(StoreContractError):
                build_store([SIG_A], roots=roots, store_dir=store_dir, envelope_dir=env_dir, verbose=False)
            with self.assertRaises(StoreContractError):
                build_store([SIG_B], roots=roots, store_dir=store_dir, envelope_dir=env_dir, force=True, verbose=False)
            self.assertEqual(pl.read_parquet(store_dir / f"{DATE}.parquet").height, 3)

            # an unchanged registry is a no-op
            report = build_store(
                [SIG_A, SIG_B], roots=roots, store_dir=store_dir, envelope_dir=env_dir, refresh_latest=False, verbose=False,
            )
            self.assertEqual(report.skipped_dates, (DATE,))

            # params: the only store day has no strictly-prior history -> defaults
            table = build_condition_params(SIG_B, store_dir=store_dir, params_dir=params_dir, next_date="20240103", verbose=False)
            self.assertEqual(sorted(table["Date"].unique().to_list()), [DATE, "20240103"])
            first_day = table.filter((pl.col("Date") == DATE) & pl.col("is_winner")).row(0, named=True)
            self.assertEqual(first_day["reason"], "no_history")
            self.assertEqual(first_day["param_set_id"], SIG_B.gate.default_param_set_id)
            next_day = table.filter((pl.col("Date") == "20240103") & pl.col("is_winner")).row(0, named=True)
            self.assertEqual(next_day["window_start"], DATE)
            self.assertEqual(next_day["window_end"], DATE)
            # strict gate keeps only BidPrice1 >= 60 -> mean 101 beats the default's mean of (50+101)/2
            self.assertEqual(next_day["param_set_id"], param_set_id({"bid_min": 60.0}))
            self.assertEqual(next_day["reason"], "max_score")
            scores = table.filter(pl.col("Date") == "20240103").sort("param_set_id")
            self.assertEqual(scores.height, 2)
            self.assertTrue((scores["n_stock_days"] > 0).all())
            self.assertEqual(read_condition_params(params_dir, SIG_B).height, table.height)

            # the point filter is applied before the condition is scored: only 2330@101 is left,
            # both sets tie on it and the default wins
            table = build_condition_params(SIG_B_FILTERED, store_dir=store_dir, params_dir=params_dir, next_date="20240103", verbose=False)
            next_day = table.filter((pl.col("Date") == "20240103") & pl.col("is_winner")).row(0, named=True)
            self.assertEqual(next_day["param_set_id"], SIG_B.gate.default_param_set_id)
            self.assertEqual(next_day["reason"], "tie_default")
            self.assertEqual(next_day["n_rows"], 1)

            # index: gate with that day's parameters (defaults on the first day) -> both beta rows
            index = build_signal_index(SIG_B_FILTERED, store_dir=store_dir, params_dir=params_dir, index_dir=index_dir, verbose=False)
            self.assertEqual(index.height, 2)
            self.assertEqual(index["param_set_id"].unique().to_list(), [SIG_B.gate.default_param_set_id])
            self.assertEqual(read_signal_index(index_dir, SIG_B, dates=[DATE]).height, 2)

            # points: the fixed filter keeps 2330@101 only; load_rows returns that store row
            points = build_points(SIG_B_FILTERED, store_dir=store_dir, index_dir=index_dir, points_dir=points_dir, verbose=False)
            self.assertEqual(points.height, 1)
            self.assertEqual(points.row(0)[:2], (DATE, "2330"))
            rows = load_rows(SIG_B_FILTERED, store_dir=store_dir, points_dir=points_dir, dates=[DATE])
            self.assertEqual(rows.height, 1)
            self.assertEqual(rows["BidPrice1"][0], 101.0)
            # no point filter -> index passes through
            passthrough = build_points(SIG_B, store_dir=store_dir, index_dir=index_dir, points_dir=root / "points2", verbose=False)
            self.assertEqual(passthrough.height, 2)
            self.assertEqual(read_points(root / "points2", SIG_B).height, 2)


if __name__ == "__main__":
    unittest.main()
