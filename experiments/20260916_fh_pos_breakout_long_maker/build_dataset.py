"""Build the study-local event index, causal row-level dataset, and maker fill seconds."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import polars as pl

from src.research.intradaySignal.framework.dataset import build_event_dataset
from src.research.intradaySignal.framework.events import build_event_index

from study import (
    EVENT_DATASET_DIR,
    EVENT_INDEX_PATH,
    FILL_DIR,
    STUDY_END,
    STUDY_START,
    add_delayed_maker_fill_seconds,
    data_roots,
    event_dataset_spec,
    event_index_spec,
    maker_fill_dir,
    select_long_events,
)


class ProgressSelector:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self, day):
        selected = select_long_events(day)
        self.count += 1
        if self.count == 1 or self.count % 10 == 0:
            print(
                f"[event-index] processed={self.count} date={day.date} raw_candidates={len(selected)}",
                flush=True,
            )
        return selected


def _chunks(values: list[str], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _atomic_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False)
    handle.close()
    frame.write_parquet(handle.name)
    Path(handle.name).replace(path)


def build_fill_seconds(dates: list[str], *, force: bool) -> None:
    roots = data_roots()
    fill_dir = maker_fill_dir()
    FILL_DIR.mkdir(parents=True, exist_ok=True)
    for index, date in enumerate(dates, start=1):
        out_path = FILL_DIR / f"{date}.parquet"
        if out_path.exists() and not force:
            continue
        source = EVENT_DATASET_DIR / f"{date}.parquet"
        if not source.exists():
            continue
        events = pl.read_parquet(source, columns=["Date", "QuoteCode", "ChannelSeq", "TransTime", "BidPrice1"])
        with_fill = add_delayed_maker_fill_seconds(events, date=date, roots=roots, fill_dir=fill_dir)
        _atomic_parquet(
            with_fill.select(
                "Date", "QuoteCode", "ChannelSeq", "maker_buy_fill_seconds", "maker_fill_branch", "maker_fill_lookup_channelseq"
            ),
            out_path,
        )
        if index == 1 or index % 20 == 0:
            filled = with_fill["maker_buy_fill_seconds"]
            print(
                f"[maker-fill] {index}/{len(dates)} {date} rows={with_fill.height} "
                f"null={filled.null_count()} lt600={(filled < 600).sum()}",
                flush=True,
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=STUDY_START)
    parser.add_argument("--end", default=STUDY_END)
    parser.add_argument("--stage", choices=("all", "index", "dataset", "fill"), default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--chunk-days", type=int, default=10)
    args = parser.parse_args()
    if args.chunk_days < 1:
        parser.error("--chunk-days must be positive")

    roots = data_roots()
    if args.stage in {"all", "index"}:
        progress = ProgressSelector()
        report = build_event_index(
            event_index_spec(selector=progress), roots=roots, force=args.force, start=args.start, end=args.end
        )
        print(report, flush=True)

    dates: list[str] = []
    if args.stage in {"all", "dataset", "fill"}:
        if not EVENT_INDEX_PATH.exists():
            raise FileNotFoundError(EVENT_INDEX_PATH)
        index = pl.read_parquet(EVENT_INDEX_PATH, columns=["Date"])
        dates = [d for d in index["Date"].unique().sort().to_list() if args.start <= d <= args.end]

    if args.stage in {"all", "dataset"}:
        saved = skipped = 0
        for chunk in _chunks(dates, args.chunk_days):
            report = build_event_dataset(
                event_dataset_spec(), roots=roots, force=args.force, start=chunk[0], end=chunk[-1]
            )
            saved += len(report.saved_dates)
            skipped += len(report.skipped_dates)
            print(f"[event-dataset] through={chunk[-1]} saved={saved} skipped={skipped}", flush=True)

    if args.stage in {"all", "fill"}:
        build_fill_seconds(dates, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
