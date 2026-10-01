"""Build the study-local event index and causal row-level dataset."""

from __future__ import annotations

import argparse

import polars as pl

from src.research.intradaySignal.framework.dataset import build_event_dataset
from src.research.intradaySignal.framework.events import build_event_index

from study import (
    EVENT_INDEX_PATH,
    STUDY_END,
    STUDY_START,
    data_roots,
    event_dataset_spec,
    event_index_spec,
    select_short_events,
)


class ProgressSelector:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self, day):
        selected = select_short_events(day)
        self.count += 1
        if self.count == 1 or self.count % 10 == 0:
            print(
                f"[event-index] processed={self.count} date={day.date} "
                f"raw_candidates={len(selected)}",
                flush=True,
            )
        return selected


def _chunks(values: list[str], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=STUDY_START)
    parser.add_argument("--end", default=STUDY_END)
    parser.add_argument(
        "--stage", choices=("all", "index", "dataset"), default="all"
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--chunk-days", type=int, default=10)
    args = parser.parse_args()
    if args.chunk_days < 1:
        parser.error("--chunk-days must be positive")

    roots = data_roots()
    if args.stage in {"all", "index"}:
        progress = ProgressSelector()
        report = build_event_index(
            event_index_spec(selector=progress),
            roots=roots,
            force=args.force,
            start=args.start,
            end=args.end,
        )
        print(report, flush=True)

    if args.stage in {"all", "dataset"}:
        if not EVENT_INDEX_PATH.exists():
            raise FileNotFoundError(EVENT_INDEX_PATH)
        index = pl.read_parquet(EVENT_INDEX_PATH, columns=["Date"])
        dates = [
            value
            for value in index["Date"].unique().sort().to_list()
            if args.start <= value <= args.end
        ]
        saved = 0
        skipped = 0
        for chunk in _chunks(dates, args.chunk_days):
            report = build_event_dataset(
                event_dataset_spec(),
                roots=roots,
                force=args.force,
                start=chunk[0],
                end=chunk[-1],
            )
            saved += len(report.saved_dates)
            skipped += len(report.skipped_dates)
            print(
                f"[event-dataset] through={chunk[-1]} "
                f"saved={saved} skipped={skipped}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
