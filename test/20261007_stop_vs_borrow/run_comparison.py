"""Run and save the fixed, paired stop-loss versus borrow-cost comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import polars as pl

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from compare import compare_exit_policies, plot_comparison


def identity(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "src/research/timeSeries/signalwithRisk_df.parquet")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "artifacts/v1")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Use a new output version: {args.output}")
    source_identity = identity(args.input)
    source = pl.read_parquet(args.input)
    result = compare_exit_policies(source)
    args.output.mkdir(parents=True)
    for name in ("ledger", "summary", "daily", "monthly", "stock_days", "transitions", "open_audit"):
        result[name].write_parquet(args.output / f"{name}.parquet")
        if name != "ledger":
            result[name].write_csv(args.output / f"{name}.csv")
    plot_comparison(result).savefig(args.output / "comparison.png", dpi=160)
    sources = [identity(Path(p)) for p in result["open_audit"]["market_source"].unique().sort()]
    manifest = {"input": source_identity, "parameters": result["parameters"],
                "uncertainty": result["uncertainty"],
                "code": [identity(Path(__file__)), identity(Path(__file__).with_name("compare.py")),
                         identity(ROOT / "src/research/tool.py"), identity(ROOT / "src/research/backTest.py")],
                "market_sources": sources, "tick_source_stat_fingerprints": result["tick_sources"],
                "outputs": [identity(p) for p in sorted(args.output.glob("*"))]}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(result["summary"].to_pandas().to_string(index=False))
    print(result["transitions"].to_pandas().to_string(index=False))
    print(json.dumps(result["uncertainty"], indent=2))


if __name__ == "__main__":
    main()
