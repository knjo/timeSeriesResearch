"""Build the legacy negFill event index through the generic event framework.

The public ``build_index(forcefetch=False)`` entry point and default artifact
path are retained for existing notebooks and scripts.  The versioned selector
and index spec live in the negFill profile; this module is only a compatibility
adapter and CLI.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.research.timeSeries.framework.events import (
    DataRoots,
    build_event_index,
    normalize_event_keys,
)
from src.research.timeSeries.framework.profiles.negfill import (
    negfill_event_index_spec,
    select_negfill_keys,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TICK_FEATURE_DIR = PROJECT_ROOT / "data" / "tickFeature"
OUTPUT_DIR = PROJECT_ROOT / "src" / "research" / "timeSeries"
OUTPUT_PATH = OUTPUT_DIR / "data_index.parquet"
DATA_ROOTS = DataRoots.from_project_root(PROJECT_ROOT)


def filter_one_day(path: Path) -> pl.DataFrame | None:
    """Compatibility helper that selects negFill keys from one tickFeature file."""

    path = Path(path)
    date = path.stem.split("_", 1)[0]
    selected = select_negfill_keys(pl.scan_parquet(path), source=str(path))
    result = normalize_event_keys(selected, date=date)
    return result if result.height else None


def build_index(
    forcefetch: bool = False,
    start: str | None = None,
    end: str | None = None,
) -> pl.DataFrame:
    """Build or incrementally refresh the negFill event index."""

    report = build_event_index(
        negfill_event_index_spec(
            OUTPUT_PATH,
            # The repository already has a pre-framework index. It is adopted
            # only after the framework validates its full key contract.
            allow_unmanifested_existing=True,
        ),
        roots=DATA_ROOTS,
        force=forcefetch,
        start=start,
        end=end,
    )
    print(
        "Saved negFill data_index: "
        f"{report.row_count} events across {report.event_date_count} event dates; "
        f"scanned={len(report.scanned_dates)}, skipped={len(report.skipped_dates)} "
        f"-> {report.index_path}"
    )
    return report.read_index()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the negFill event index")
    parser.add_argument("--force", action="store_true", help="Force rebuild")
    parser.add_argument("--start", default=None, help="YYYYMMDD, inclusive")
    parser.add_argument("--end", default=None, help="YYYYMMDD, inclusive")
    args = parser.parse_args()
    build_index(forcefetch=args.force, start=args.start, end=args.end)
