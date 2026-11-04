"""Replay the frozen candidate family against the same accepted short entries."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from src.pipeline_storage import load_pipeline_storage
from src.features.market_beta import residual_vol_time_scale
from data import load_day
from engine import replay
from spec import candidates, contract

HERE = Path(__file__).resolve().parent
COLUMNS = ["Date", "QuoteCode", "ChannelSeq", "TransTime", "BidPrice1", "RefPrice", "TickSize", "Close",
           "FutureHigh", "TakerSell_CloseBP", "txf_residual_vol_to_1330", "txf_residual_vol_0900_1330_60d"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def replay_stock(g, tick, spec, activation=-1, event=None):
    if event is None:
        event = tick.events(average=spec.average)
    a = lambda col: g[col].to_numpy().astype(float)
    return replay(*event, g["TransTime"].cast(pl.Int64).to_numpy(), a("BidPrice1"), a("RefPrice"),
                  a("txf_residual_vol_to_1330"), a("txf_residual_vol_0900_1330_60d"), a("Close"),
                  tick.close_time, spec.mode, spec.span, spec.ratio, activation, spec.scope)


def make_rows(g, values, spec, activation):
    return g.with_columns(
        pl.lit(spec.name).alias("candidate"), pl.lit(spec.family).alias("family"),
        pl.lit(spec.baseline).alias("baseline"), pl.lit(activation, dtype=pl.Int64).alias("activation_us"),
        pl.Series("exit_price", values[:, 0]),
        pl.Series("exit_time", values[:, 1].astype(np.int64)).cast(pl.Datetime("us")),
        pl.Series("reason", values[:, 2].astype(np.int8)),
        pl.Series("reset_exit", values[:, 3] > 0.5),
        pl.Series("reset_observed", values[:, 10] > 0.5),
        pl.Series("waited_for_ask", values[:, 4] > 0.5),
        pl.Series("unfilled_close_mark", values[:, 5] > 0.5),
        pl.Series("sl_at_exit", values[:, 6]), pl.Series("tp_at_exit", values[:, 7]),
        pl.Series("trigger_us", values[:, 8].astype(np.int64)),
        pl.Series("quote_age_seconds", values[:, 9]),
    ).with_columns(
        (pl.col("BidPrice1") * 1000).alias("notional"),
        ((pl.col("BidPrice1")-pl.col("exit_price"))/pl.col("BidPrice1")*1e4).alias("gross_bp"),
        (pl.col("TransTime").cast(pl.Int64) < pl.col("activation_us")).alias("entered_before_activation"),
    ).with_columns(
        (pl.col("gross_bp")-19.3).alias("net_bp"),
        ((pl.col("BidPrice1")-pl.col("exit_price"))*1000 - pl.col("notional")*19.3/1e4).alias("pnl_twd"),
        pl.when((pl.col("reason") != 0) & ~pl.col("unfilled_close_mark"))
          .then(pl.col("TickSize")*2*1000).otherwise(0.0).alias("stress_cost_twd"),
    )


def process_day(sig, tick_path, specs):
    date = sig["Date"][0]
    ticks = load_day(tick_path, date, sig["QuoteCode"].unique().to_list())
    groups = sig.partition_by("QuoteCode", as_dict=True, maintain_order=True)
    baseline_values, stops = {}, []
    for (code,), g in groups.items():
        out = replay_stock(g, ticks[code], specs[0])
        baseline_values[code] = out
        stopped = (out[:, 2] > 0.5) & (out[:, 2] < 1.5) & (out[:, 5] < 0.5)
        if np.any(stopped):
            stops.append((int(out[stopped, 1].min()), code))
    stops.sort()
    switch = {k: stops[k-1][0] if len(stops) >= k else ticks[next(iter(ticks))].close_time + 1
              for k in range(1, 7)}
    output = []
    for (code,), g in groups.items():
        tick = ticks[code]
        extra = sorted({v for v in switch.values() if v < tick.close_time})
        events = {avg: tick.events(extra, average=avg) for avg in ("vwap", "minute_last_sma")}
        for spec in specs:
            activation = switch[spec.k] if spec.k else -1
            values = baseline_values[code] if spec.name == "ref08" else replay_stock(
                g, tick, spec, activation, events[spec.average])
            output.append(make_rows(g, values, spec, activation))
    return pl.concat(output), [{"Date": date, "k": i+1, "stop_time_us": t, "QuoteCode": code}
                               for i, (t, code) in enumerate(stops)]


def validate_signals(signal):
    if signal.unique(["Date", "QuoteCode", "ChannelSeq"]).height != signal.height:
        raise ValueError("Duplicate canonical entry keys")
    for col in ["BidPrice1", "RefPrice", "TickSize", "Close", "txf_residual_vol_to_1330", "txf_residual_vol_0900_1330_60d"]:
        a = signal[col].to_numpy()
        if not np.all(np.isfinite(a) & (a > 0)):
            raise ValueError(f"Invalid {col}")
    times = signal["TransTime"].to_numpy().astype("datetime64[us]")
    end = times.astype("datetime64[D]").astype("datetime64[us]") + np.timedelta64(810, "m")
    remain = (end-times).astype("timedelta64[us]").astype(float)/1e6
    calc = signal["txf_residual_vol_0900_1330_60d"].to_numpy() * residual_vol_time_scale(np.clip(remain/16200, 0, 1))
    if not np.allclose(calc, signal["txf_residual_vol_to_1330"].to_numpy(), rtol=1e-6, atol=1e-9):
        raise ValueError("Entry sigma does not match canonical clock")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="00000000")
    parser.add_argument("--end", default="99999999")
    parser.add_argument("--output", type=Path, default=HERE / "artifacts" / "v4")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    signal_path = ROOT / "src/research/timeSeries/signalwithRisk_df.parquet"
    source_hash = digest(signal_path)
    signal = (pl.read_parquet(signal_path, columns=COLUMNS).with_row_index("trade_id")
              .filter(pl.col("Date").is_between(pl.lit(args.start), pl.lit(args.end))).sort("Date", "TransTime", "QuoteCode", "ChannelSeq"))
    validate_signals(signal)
    tick_dir = load_pipeline_storage().tick_dir
    specs = candidates()
    frozen = contract()
    code_hash = {p.name: digest(p) for p in sorted(HERE.glob("*.py"))}
    identity = {"signal_sha256": source_hash, "contract": frozen, "code": code_hash,
                "dates": signal["Date"].unique().sort().to_list()}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if not args.resume or old["fingerprint"] != fingerprint:
            raise ValueError("Output exists or fingerprint changed; choose a fresh output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "days").mkdir(exist_ok=True)
    manifest = {**identity, "fingerprint": fingerprint, "signal_path": str(signal_path),
                "tick_root": str(tick_dir), "n_rows": signal.height, "n_days": signal["Date"].n_unique(),
                "status": "running", "tick_sources": []}
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    signal.write_parquet(args.output / "input_signals.parquet")
    start = time.monotonic()
    for index, ((date,), sig) in enumerate(signal.partition_by("Date", as_dict=True, maintain_order=True).items()):
        path = tick_dir / f"{date}_StockTick.parquet"
        stat = path.stat()  # Missing day fails; never silently drops a bad day.
        source = {"date": date, "path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        manifest["tick_sources"].append(source)
        day_path = args.output / "days" / f"{date}.parquet"
        source_path = args.output / "days" / f"{date}_source.json"
        if args.resume and day_path.exists():
            if not source_path.exists() or json.loads(source_path.read_text()) != source:
                raise ValueError(f"Cached source changed or missing source metadata: {date}")
        if not (args.resume and day_path.exists()):
            rows, stops = process_day(sig, path, specs)
            if rows.height != sig.height * len(specs):
                raise AssertionError("Unequal entry support")
            if rows.filter(pl.col("exit_time") < pl.col("TransTime")).height:
                raise AssertionError("Exit before entry")
            if path.stat().st_mtime_ns != stat.st_mtime_ns:
                raise RuntimeError(f"Source changed during replay: {path}")
            (args.output / "days" / f"{date}_stops.json").write_text(json.dumps(stops))
            source_path.write_text(json.dumps(source))
            temporary = day_path.with_suffix(".parquet.tmp")
            rows.write_parquet(temporary)
            temporary.replace(day_path)
        if index % 10 == 0 or index+1 == manifest["n_days"]:
            print(f"{index+1}/{manifest['n_days']} {date}: {sig.height} entries; {time.monotonic()-start:.1f}s", flush=True)
    if digest(signal_path) != source_hash:
        raise RuntimeError("Signal file changed during replay")
    manifest["status"] = "replay_complete"
    manifest["elapsed_seconds"] = time.monotonic()-start
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    from summarize import summarize
    summarize(args.output)


if __name__ == "__main__":
    main()
