"""Layer 2 — fixed tick-level removals → candidate points.

Reads ``SignalConfig.point_filter`` (a ``point_filter`` :class:`GateSpec`, no
grid).  Every index row of a day is tested once; the surviving keys are written
to ``points/<st>_points.parquet`` (``Date, QuoteCode, ChannelSeq``).  A signal
without a point filter passes its index through unchanged.

Layers 3 and 4 read their rows through :func:`load_rows`, so "candidate" means
the same thing everywhere: store rows ∩ signal index ∩ point filter.

Incremental contract: a date is redone when it is new, when it is the latest
index date (the store day may have been refreshed), or when the index for that
date changed (row count or parameter set).  A changed filter definition
rebuilds every date.
"""

from __future__ import annotations

import datetime as _dt
import json
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from ..framework.dataset import _atomic_write_json, _atomic_write_parquet
from ..framework.registry import SignalConfig
from ..framework.store import read_store
from .envelope import read_signal_index


POINTS_COLUMNS = ("Date", "QuoteCode", "ChannelSeq")


def points_path(points_dir: Path | str, signal: SignalConfig) -> Path:
    return Path(points_dir) / f"{signal.st_name}_points.parquet"


def _manifest_path(points_dir: Path | str, signal: SignalConfig) -> Path:
    return Path(points_dir) / f"{signal.st_name}_points.manifest.json"


def _empty_points() -> pl.DataFrame:
    return pl.DataFrame(schema={"Date": pl.String, "QuoteCode": pl.String, "ChannelSeq": pl.UInt64})


def filter_hash(signal: SignalConfig) -> str:
    return signal.point_filter.stable_hash() if signal.point_filter is not None else "none"


def read_points(points_dir: Path | str, signal: SignalConfig, *, dates: Sequence[str] | None = None) -> pl.DataFrame:
    path = points_path(points_dir, signal)
    if not path.exists():
        return _empty_points()
    frame = pl.read_parquet(path)
    if dates is not None:
        frame = frame.filter(pl.col("Date").is_in(list(dates)))
    return frame


def point_mask(signal: SignalConfig, rows: pl.DataFrame) -> pl.Series:
    """The layer-2 mask for store rows (all True when the signal has no point filter)."""

    if signal.point_filter is None:
        return pl.Series("points", [True] * rows.height, dtype=pl.Boolean)
    return signal.point_filter.mask(rows)


def build_points(
    signal: SignalConfig,
    *,
    store_dir: Path | str,
    index_dir: Path | str,
    points_dir: Path | str,
    start: str | None = None,
    end: str | None = None,
    force: bool = False,
    verbose: bool = True,
) -> pl.DataFrame:
    """Apply the signal's point filter to every index day and persist the surviving keys."""

    points_dir = Path(points_dir)
    points_dir.mkdir(parents=True, exist_ok=True)
    index = read_signal_index(index_dir, signal)
    dates = sorted(
        d for d in (index["Date"].unique().to_list() if index.height else [])
        if (start is None or d >= start) and (end is None or d <= end)
    )
    signature = {
        row["Date"]: (int(row["n"]), row["pid"])
        for row in (
            index.group_by("Date").agg(pl.len().alias("n"), pl.col("param_set_id").first().alias("pid")).to_dicts()
            if index.height else []
        )
    }
    manifest_path = _manifest_path(points_dir, signal)
    manifest: dict[str, object] = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    current_hash = filter_hash(signal)
    recorded_hash = (manifest.get("point_filter") or {}).get("hash")
    existing = read_points(points_dir, signal)
    if existing.height and recorded_hash not in (None, current_hash):
        if verbose:
            print(f"[{signal.condition_id}] point filter changed; rebuilding all points")
        force = True
    recorded: dict[str, dict[str, object]] = dict(manifest.get("dates", {})) if not force else {}
    latest_index_date = dates[-1] if dates else None
    if force:
        todo = list(dates)
    else:
        todo = [
            d for d in dates
            if d not in recorded
            or d == latest_index_date
            or recorded[d].get("index_rows") != signature[d][0]
            or recorded[d].get("param_set_id") != signature[d][1]
        ]
    if existing.height:
        existing = existing.filter(~pl.col("Date").is_in(todo))
    parts = [existing] if existing.height else []
    if verbose:
        print(f"[{signal.condition_id}] points: {len(todo)} day(s) to filter")
    for position, date in enumerate(todo, start=1):
        keys = index.filter(pl.col("Date") == date).select("QuoteCode", "ChannelSeq")
        rows = read_store(store_dir, dates=[date], condition=signal)
        if rows.is_empty() or keys.is_empty():
            recorded[date] = {"index_rows": signature[date][0], "param_set_id": signature[date][1], "rows": 0}
            continue
        rows = rows.join(keys, on=["QuoteCode", "ChannelSeq"], how="semi")
        kept = rows.filter(point_mask(signal, rows)).select(list(POINTS_COLUMNS))
        parts.append(kept)
        recorded[date] = {"index_rows": signature[date][0], "param_set_id": signature[date][1], "rows": int(kept.height)}
        if verbose and (position % 20 == 0 or position == len(todo)):
            print(f"  filtered {position}/{len(todo)} ({date}: {kept.height:,} of {rows.height:,} index rows kept)")
    if parts:
        points = pl.concat(parts, how="vertical").select(list(POINTS_COLUMNS)).sort(list(POINTS_COLUMNS))
    else:
        points = _empty_points()
    _atomic_write_parquet(points, points_path(points_dir, signal))
    _atomic_write_json(
        {
            "condition_id": signal.condition_id,
            "point_filter": (
                {"gate_id": signal.point_filter.gate_id, "version": signal.point_filter.version, "hash": current_hash}
                if signal.point_filter is not None else {"gate_id": None, "version": None, "hash": current_hash}
            ),
            "rows": int(points.height),
            "dates": dict(sorted(recorded.items())),
            "watermark": points["Date"].max() if points.height else None,
            "updated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        },
        manifest_path,
    )
    if verbose:
        print(f"[{signal.condition_id}] points saved: {points.height:,} rows -> {points_path(points_dir, signal)}")
    return points


def load_rows(
    signal: SignalConfig,
    *,
    store_dir: Path | str,
    points_dir: Path | str,
    dates: Sequence[str],
    columns: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Store rows for ``dates`` that survived layers 1–2 (the signal's candidate points)."""

    points = read_points(points_dir, signal, dates=list(dates))
    if points.is_empty():
        return pl.DataFrame()
    parts = []
    for date in dates:
        keys = points.filter(pl.col("Date") == date).select("QuoteCode", "ChannelSeq")
        if keys.is_empty():
            continue
        rows = read_store(store_dir, dates=[date], condition=signal, columns=columns)
        if rows.is_empty():
            continue
        parts.append(rows.join(keys, on=["QuoteCode", "ChannelSeq"], how="semi"))
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts, how="diagonal_relaxed")


__all__ = [
    "POINTS_COLUMNS",
    "build_points",
    "filter_hash",
    "load_rows",
    "point_mask",
    "points_path",
    "read_points",
]
