"""CLI: run the intradaySignal layers for the registered signals.

    uv run python src/research/intradaySignal/pipeline/run.py envelope                # scan → store → params → index (incremental)
    uv run python src/research/intradaySignal/pipeline/run.py envelope --only negfill --next-date 20261001
    uv run python src/research/intradaySignal/pipeline/run.py envelope --rescan --start 20260901 --end 20260930
    uv run python src/research/intradaySignal/pipeline/run.py points   --only negfill
    uv run python src/research/intradaySignal/pipeline/run.py day_selector --only negfill --evaluate
    uv run python src/research/intradaySignal/pipeline/run.py model    --only negfill --last-n-days 230
    uv run python src/research/intradaySignal/pipeline/run.py all      --only negfill

Layers read the store in order; ``all`` runs them back to back with the
incremental defaults.  Every layer supports ``--only <st ...>`` and
``--output-root`` (redirect artifacts for tests).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.research.intradaySignal.config import REGISTRY, resolve_dirs  # noqa: E402
from src.research.intradaySignal.framework.events import DataRoots  # noqa: E402


LAYERS = ("envelope", "points", "day_selector", "model", "all")


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--only", nargs="*", default=None, help="st_name(s); default = every registered signal")
    parser.add_argument("--output-root", default=None, help="Redirect artifacts (tests)")
    parser.add_argument("--quiet", action="store_true")


def _bounds(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start", default=None, help="YYYYMMDD inclusive")
    parser.add_argument("--end", default=None, help="YYYYMMDD inclusive")
    parser.add_argument("--force", action="store_true", help="Rebuild the selected range")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run intradaySignal layers")
    sub = parser.add_subparsers(dest="layer", required=True)

    p = sub.add_parser("envelope", help="layer 1: envelope scan → store → condition params → signal index")
    _common(p); _bounds(p)
    p.add_argument("--rescan", action="store_true", help="Re-run the raw envelope scan for --start/--end")
    p.add_argument("--all-history", action="store_true", help="Allow --rescan without --start (every source date, 2023+)")
    p.add_argument("--no-store", action="store_true", help="Skip the shared store merge")
    p.add_argument("--next-date", default=None, help="Also emit condition params for a date without store rows yet")
    p.add_argument("--evaluate", action="store_true", help="Print the rolling first-live evaluation of the grid")

    p = sub.add_parser("points", help="layer 2: fixed tick-level filter → candidate points")
    _common(p); _bounds(p)

    p = sub.add_parser("day_selector", help="layer 3: learned stock-day gate → day scores")
    _common(p)
    p.add_argument("--force-table", action="store_true", help="Rebuild the stock-day table")
    p.add_argument("--evaluate", action="store_true", help="Print next-day realized target by score bucket")

    p = sub.add_parser("model", help="layer 4: rolling Ridge A/M + thresholds → selection")
    _common(p)
    p.add_argument("--last-n-days", type=int, default=None, help="Train on the last N store days (default: all)")
    p.add_argument("--next-date", default=None, help="Append a dummy row so the next session gets predictions")
    p.add_argument("--n-jobs", type=int, default=-1)

    p = sub.add_parser("all", help="layers 1-4 with incremental defaults")
    _common(p)
    p.add_argument("--last-n-days", type=int, default=None)
    p.add_argument("--next-date", default=None)
    return parser


def run_envelope(args, dirs) -> None:
    from src.research.intradaySignal.layers.envelope import build_envelope_layer, evaluate_rolling_selection

    if args.rescan and args.start is None and not args.all_history:
        raise SystemExit("--rescan without --start rebuilds every source date since 2023; pass --start or --all-history")
    roots = DataRoots.from_project_root(PROJECT_ROOT)
    build_envelope_layer(
        REGISTRY, roots=roots, dirs=dirs, only=args.only, start=args.start, end=args.end,
        rescan=args.rescan, force=args.force, next_date=args.next_date,
        build_store_days=not args.no_store, verbose=not args.quiet,
    )
    if args.evaluate:
        import polars as pl

        for signal in REGISTRY.select(args.only):
            if not signal.gate.grid:
                continue
            per_date, per_month = evaluate_rolling_selection(signal, params_dir=dirs["params"])
            with pl.Config(tbl_rows=60, tbl_cols=20, tbl_width_chars=200, float_precision=3):
                print(f"[{signal.condition_id}] rolling first-live evaluation ({per_date.height} days)")
                print(per_month)


def run_points(args, dirs) -> None:
    from src.research.intradaySignal.layers.point_filter import build_points

    for signal in REGISTRY.select(args.only):
        build_points(
            signal, store_dir=dirs["store"], index_dir=dirs["index"], points_dir=dirs["points"],
            start=args.start, end=args.end, force=args.force, verbose=not args.quiet,
        )


def run_day_selector(args, dirs) -> None:
    from src.research.intradaySignal.layers.day_selector import build_day_selector_layer

    for signal in REGISTRY.select(args.only):
        if signal.day_selector is None:
            print(f"[{signal.condition_id}] no day selector configured; skipping")
            continue
        build_day_selector_layer(signal, dirs=dirs, force_table=args.force_table, evaluate=args.evaluate, verbose=not args.quiet)


def run_model(args, dirs) -> None:
    from src.research.intradaySignal.layers.model import build_model_layer

    for signal in REGISTRY.select(args.only):
        build_model_layer(
            signal, dirs=dirs, last_n_days=args.last_n_days, next_trade_day=args.next_date,
            n_jobs=getattr(args, "n_jobs", -1), verbose=not args.quiet,
        )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    dirs = resolve_dirs(args.output_root)
    if args.layer == "envelope":
        run_envelope(args, dirs)
    elif args.layer == "points":
        run_points(args, dirs)
    elif args.layer == "day_selector":
        run_day_selector(args, dirs)
    elif args.layer == "model":
        run_model(args, dirs)
    else:
        ns = argparse.Namespace(**vars(args))
        ns.start = ns.end = None; ns.force = False; ns.rescan = False; ns.all_history = False
        ns.no_store = False; ns.evaluate = False; ns.force_table = False; ns.n_jobs = -1
        run_envelope(ns, dirs); run_points(ns, dirs); run_day_selector(ns, dirs); run_model(ns, dirs)


if __name__ == "__main__":
    main()
