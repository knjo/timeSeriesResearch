"""Shared multi-signal store: per-signal envelope indexes and tagged daily rows.

Layout (under ``artifacts/``, ignored by git)::

    artifacts/envelope/<st_name>_index.parquet       Date, QuoteCode, ChannelSeq [, envelope cols]
    artifacts/envelope/<st_name>_index.manifest.json
    artifacts/store/YYYYMMDD.parquet                 one merged row per tick, cond_<st> tags
    artifacts/store/manifest.json                    merge options + per-day, per-condition envelope hash

Every condition's envelope is materialized once into the same daily file: the
union of all envelope keys is merged against the raw sources exactly once, then
each condition contributes a boolean ``cond_<st_name>`` column (envelope hit AND
its own universe).  Rows that no condition claims are dropped.

Incremental contract
--------------------
* Envelope indexes append source dates newer than their own watermark; ``force``
  with bounds replaces that inclusive range, ``force`` without bounds rebuilds.
* Store days are skipped when their manifest entry carries the same envelope
  hash for every registered condition, except the latest existing day which is
  refreshed (its sources may have been incomplete when first built).  A day
  whose envelope hash changed for some condition (new signal, new envelope
  version) is *patched*: only that condition's keys not already present are
  merged and every tag is recomputed.  Keys dropped by a universe are therefore
  never re-merged night after night.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from src.dataloader.txfDataLoader import TxfDataLoader

from .dataset import (
    ALL_EVENTS_UNIVERSE,
    MergeOptions,
    _atomic_write_json,
    _atomic_write_parquet,
    materialize_event_day,
)
from .events import (
    DAY_KEY_COLS,
    EVENT_KEY_COLS,
    DataRoots,
    EventDaySources,
    _SOURCE_GLOBS,
    _validate_date,
    normalize_event_keys,
)
from .registry import SignalConfig, SignalRegistry


STORE_FORMAT_VERSION = 1
KEY_COLS = tuple(DAY_KEY_COLS)


class StoreContractError(RuntimeError):
    """Raised when the store or an envelope index violates its contract."""


# --------------------------------------------------------------------------- #
# merge options
# --------------------------------------------------------------------------- #


def store_merge_options_v1() -> MergeOptions:
    """Store merge = deployed negFill legacy alignment + brenchFeature.

    The universe is deliberately ``ALL_EVENTS`` here: each condition applies its
    own universe when its tag is computed, so one merged row can serve several
    signals with different eligibility rules.
    """

    return MergeOptions(
        universe=ALL_EVENTS_UNIVERSE,
        require_pre_market=False,
        attach_tick_bar=True,
        require_tick_bar=False,
        tick_bar_alignment="legacy_stock_tick_time",
        attach_txf=True,
        require_txf=False,
        strict_cardinality=True,
        feature_suffix="_right",
        schema_policy="union",
        preserve_tick_data_column_order=True,
        attach_brench=True,
        require_brench=False,
    )


def _merge_options_payload(options: MergeOptions) -> dict[str, object]:
    return {
        "universe": [options.universe.universe_id, options.universe.universe_version],
        "require_pre_market": options.require_pre_market,
        "attach_tick_bar": options.attach_tick_bar,
        "require_tick_bar": options.require_tick_bar,
        "tick_bar_alignment": options.tick_bar_alignment,
        "attach_txf": options.attach_txf,
        "require_txf": options.require_txf,
        "feature_suffix": options.feature_suffix,
        "schema_policy": options.schema_policy,
        "preserve_tick_data_column_order": options.preserve_tick_data_column_order,
        "attach_brench": options.attach_brench,
    }


# --------------------------------------------------------------------------- #
# envelope indexes
# --------------------------------------------------------------------------- #


def envelope_index_path(envelope_dir: Path | str, condition: SignalConfig) -> Path:
    return Path(envelope_dir) / f"{condition.st_name}_index.parquet"


def _envelope_manifest_path(envelope_dir: Path | str, condition: SignalConfig) -> Path:
    return Path(envelope_dir) / f"{condition.st_name}_index.manifest.json"


def _empty_envelope(condition: SignalConfig) -> pl.DataFrame:
    schema: dict[str, pl.DataType] = {
        "Date": pl.String,
        "QuoteCode": pl.String,
        "ChannelSeq": pl.UInt64,
    }
    return pl.DataFrame(schema=schema)


def _validate_envelope(frame: pl.DataFrame, condition: SignalConfig) -> None:
    expected = [*EVENT_KEY_COLS, *condition.envelope_columns]
    if list(frame.columns) != expected:
        raise StoreContractError(
            f"{condition.condition_id}: envelope columns must be {expected}, got {frame.columns}"
        )
    keys = frame.select(EVENT_KEY_COLS)
    if keys.schema["Date"] != pl.String or keys.schema["QuoteCode"] != pl.String:
        raise StoreContractError(f"{condition.condition_id}: Date/QuoteCode must be String")
    if keys.schema["ChannelSeq"] != pl.UInt64:
        raise StoreContractError(f"{condition.condition_id}: ChannelSeq must be UInt64")
    nulls = {c: keys[c].null_count() for c in EVENT_KEY_COLS if keys[c].null_count()}
    if nulls:
        raise StoreContractError(f"{condition.condition_id}: null keys {nulls}")
    duplicates = keys.group_by(list(EVENT_KEY_COLS)).len().filter(pl.col("len") > 1)
    if not duplicates.is_empty():
        raise StoreContractError(
            f"{condition.condition_id}: duplicate envelope keys {duplicates.head(3).to_dicts()}"
        )


def select_envelope_day(condition: SignalConfig, day: EventDaySources) -> pl.DataFrame:
    """Run one condition's envelope selector for one day and normalize it."""

    raw = condition.envelope(day)
    if isinstance(raw, pl.LazyFrame):
        raw = raw.collect()
    if not isinstance(raw, pl.DataFrame):
        raise StoreContractError(f"{condition.condition_id}: envelope must return a polars frame")
    missing = [c for c in condition.envelope_columns if c not in raw.columns]
    if missing:
        raise StoreContractError(
            f"{condition.condition_id}: envelope output lacks declared columns {missing}"
        )
    keys = normalize_event_keys(raw.select(list(KEY_COLS)), date=day.date)
    if not condition.envelope_columns:
        return keys
    extras = raw.select([*KEY_COLS, *condition.envelope_columns]).with_columns(
        pl.col("ChannelSeq").cast(pl.UInt64, strict=True),
        pl.col("QuoteCode").cast(pl.String, strict=True),
    )
    if extras.select(list(KEY_COLS)).is_duplicated().any():
        raise StoreContractError(
            f"{condition.condition_id}: envelope columns are not unique per key"
        )
    out = keys.join(extras, on=list(KEY_COLS), how="left").select(
        [*EVENT_KEY_COLS, *condition.envelope_columns]
    )
    return out.sort(list(EVENT_KEY_COLS))


def _source_dates(roots: DataRoots, condition: SignalConfig) -> list[str]:
    calendar_source = condition.envelope_required_sources[0]
    directory = roots.source_dir(calendar_source)
    dates: list[str] = []
    for path in sorted(directory.glob(_SOURCE_GLOBS[calendar_source])):
        date = path.name.split("_", 1)[0]
        try:
            _validate_date(date, f"source filename {path.name}")
        except ValueError:
            continue
        dates.append(date)
    return dates


@dataclass(frozen=True)
class EnvelopeBuildReport:
    condition_id: str
    processed_dates: tuple[str, ...]
    skipped_missing_sources: tuple[str, ...]
    total_rows: int
    watermark: str | None


def read_envelope_index(
    envelope_dir: Path | str,
    condition: SignalConfig,
    *,
    dates: Sequence[str] | None = None,
) -> pl.DataFrame:
    path = envelope_index_path(envelope_dir, condition)
    if not path.exists():
        return _empty_envelope(condition)
    frame = pl.read_parquet(path)
    if dates is not None:
        frame = frame.filter(pl.col("Date").is_in(list(dates)))
    return frame


def build_envelope_index(
    condition: SignalConfig,
    *,
    roots: DataRoots,
    envelope_dir: Path | str,
    start: str | None = None,
    end: str | None = None,
    force: bool = False,
    verbose: bool = True,
) -> EnvelopeBuildReport:
    """Incrementally build one condition's parameter-free envelope index."""

    _validate_bounds(start, end)
    envelope_dir = Path(envelope_dir)
    envelope_dir.mkdir(parents=True, exist_ok=True)
    path = envelope_index_path(envelope_dir, condition)
    manifest_path = _envelope_manifest_path(envelope_dir, condition)

    full_rebuild = force and start is None and end is None
    existing = read_envelope_index(envelope_dir, condition)
    if existing.height:
        manifest = _read_json(manifest_path)
        stored_id = manifest.get("condition_id")
        if stored_id not in (None, condition.condition_id):
            if not force:
                raise StoreContractError(
                    f"{path} belongs to {stored_id}, not {condition.condition_id}; "
                    "a new condition version needs --force (with --start to bound the rescan)"
                )
            # A version bump never mixes rows from two definitions: start from an
            # empty index and rebuild the requested range only.
            if verbose:
                print(f"[{condition.condition_id}] replacing {stored_id} index; rebuilding the requested range")
            existing = _empty_envelope(condition)
            full_rebuild = True
        else:
            _validate_envelope(existing, condition)
    base = _empty_envelope(condition) if full_rebuild else existing
    if full_rebuild and (start is not None or end is not None):
        # bounded full rebuild: the index is exactly the requested range
        pass

    all_dates = _source_dates(roots, condition)
    bounded = [d for d in all_dates if (start is None or d >= start) and (end is None or d <= end)]
    watermark = existing["Date"].max() if existing.height else None
    if not force and watermark is not None:
        targets = [d for d in bounded if d > watermark]
    else:
        targets = bounded
    if force and not full_rebuild and existing.height:
        base = existing.filter(~pl.col("Date").is_in(targets))

    parts: list[pl.DataFrame] = []
    processed: list[str] = []
    skipped: list[str] = []
    if verbose:
        print(f"[{condition.condition_id}] envelope: {len(targets)} source date(s) to scan")
    for position, date in enumerate(targets, start=1):
        day = EventDaySources(date=date, roots=roots)
        missing = [s for s in condition.envelope_required_sources if not day.path(s).is_file()]
        if missing:
            skipped.append(date)
            if verbose:
                print(f"  [{position}/{len(targets)}] {date}: skip, missing {missing}")
            continue
        selected = select_envelope_day(condition, day)
        processed.append(date)
        if selected.height:
            parts.append(selected)
        if verbose:
            print(f"  [{position}/{len(targets)}] {date}: {selected.height:,} keys")

    frames = [f for f in (base, *parts) if f.height]
    if frames:
        index = pl.concat(frames, how="vertical").sort(list(EVENT_KEY_COLS))
    else:
        index = _empty_envelope(condition)
        for column in condition.envelope_columns:
            index = index.with_columns(pl.lit(None).alias(column))
    _validate_envelope(index, condition)
    if targets or full_rebuild:
        _atomic_write_parquet(index, path)
        _atomic_write_json(
            {
                "condition_id": condition.condition_id,
                "st_name": condition.st_name,
                "version": condition.version,
                "condition_hash": condition.stable_hash(),
                "envelope_columns": list(condition.envelope_columns),
                "envelope_required_sources": list(condition.envelope_required_sources),
                "dates": int(index["Date"].n_unique()) if index.height else 0,
                "rows": int(index.height),
                "watermark": index["Date"].max() if index.height else None,
                "updated_at": _dt.datetime.now().isoformat(timespec="seconds"),
            },
            manifest_path,
        )
    if verbose:
        print(
            f"[{condition.condition_id}] envelope saved: {index.height:,} keys, "
            f"watermark={index['Date'].max() if index.height else None} -> {path}"
        )
    return EnvelopeBuildReport(
        condition_id=condition.condition_id,
        processed_dates=tuple(processed),
        skipped_missing_sources=tuple(skipped),
        total_rows=int(index.height),
        watermark=index["Date"].max() if index.height else None,
    )


# --------------------------------------------------------------------------- #
# store
# --------------------------------------------------------------------------- #


def store_manifest_path(store_dir: Path | str) -> Path:
    return Path(store_dir) / "manifest.json"


def store_day_path(store_dir: Path | str, date: str) -> Path:
    return Path(store_dir) / f"{date}.parquet"


def store_dates(store_dir: Path | str) -> list[str]:
    root = Path(store_dir)
    if not root.exists():
        return []
    dates = []
    for path in sorted(root.glob("*.parquet")):
        try:
            _validate_date(path.stem, path.name)
        except ValueError:
            continue
        dates.append(path.stem)
    return dates


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise StoreContractError(f"{path}: manifest must be a JSON object")
    return payload


def _validate_bounds(start: str | None, end: str | None) -> None:
    if start is not None:
        _validate_date(start, "start")
    if end is not None:
        _validate_date(end, "end")
    if start is not None and end is not None and start > end:
        raise ValueError(f"start must be <= end, got {start} > {end}")


def _tag_frame(
    frame: pl.DataFrame,
    condition_keys: Mapping[str, pl.DataFrame],
    conditions: Sequence[SignalConfig],
) -> pl.DataFrame:
    """Recompute every ``cond_<st>`` column and drop rows nobody claims."""

    out = frame.drop([c.tag_column for c in conditions if c.tag_column in frame.columns])
    out = out.with_row_index("_store_order")
    key_view = out.select(["_store_order", *KEY_COLS])
    tag_columns: list[str] = []
    for condition in conditions:
        keys = condition_keys.get(condition.st_name)
        if keys is None or keys.is_empty():
            out = out.with_columns(pl.lit(False).alias(condition.tag_column))
            tag_columns.append(condition.tag_column)
            continue
        hits = (
            key_view.join(
                keys.select(list(KEY_COLS)).unique().with_columns(pl.lit(True).alias("_hit")),
                on=list(KEY_COLS),
                how="left",
            )
            .sort("_store_order")["_hit"]
            .fill_null(False)
        )
        allowed = condition.universe.apply(out).select(list(KEY_COLS)).unique()
        in_universe = (
            key_view.join(allowed.with_columns(pl.lit(True).alias("_ok")), on=list(KEY_COLS), how="left")
            .sort("_store_order")["_ok"]
            .fill_null(False)
        )
        out = out.with_columns((hits & in_universe).alias(condition.tag_column))
        tag_columns.append(condition.tag_column)
    all_tags = [c for c in out.columns if c.startswith("cond_")]
    out = out.with_columns([pl.col(c).fill_null(False) for c in all_tags])
    out = out.filter(pl.any_horizontal([pl.col(c) for c in all_tags]))
    return out.sort("_store_order").drop("_store_order")


def _attach_envelope_columns(
    frame: pl.DataFrame,
    condition_keys: Mapping[str, pl.DataFrame],
    conditions: Sequence[SignalConfig],
) -> pl.DataFrame:
    out = frame
    for condition in conditions:
        if not condition.envelope_columns:
            continue
        renamed = {c: condition.envelope_store_column(c) for c in condition.envelope_columns}
        out = out.drop([c for c in renamed.values() if c in out.columns])
        keys = condition_keys.get(condition.st_name)
        if keys is None:
            for column in renamed.values():
                out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias(column))
            continue
        extras = keys.select([*KEY_COLS, *condition.envelope_columns]).rename(renamed)
        out = out.with_row_index("_store_order").join(extras, on=list(KEY_COLS), how="left")
        out = out.sort("_store_order").drop("_store_order")
    return out


@dataclass(frozen=True)
class StoreBuildReport:
    built_dates: tuple[str, ...]
    patched_dates: tuple[str, ...]
    refreshed_dates: tuple[str, ...]
    skipped_dates: tuple[str, ...]
    empty_dates: tuple[str, ...]


def _keys_hash(keys: pl.DataFrame, salt: str = "") -> str:
    """Stable digest of a key set (+ condition identity); detects envelope changes per day."""

    if keys.is_empty():
        return "empty" if not salt else hashlib.sha1(f"empty|{salt}".encode("utf-8")).hexdigest()
    joined = (
        keys.select(list(KEY_COLS))
        .unique()
        .sort(list(KEY_COLS))
        .select(
            pl.concat_str(
                [pl.col("QuoteCode"), pl.lit(":"), pl.col("ChannelSeq").cast(pl.String)]
            )
            .str.join("\n")
            .alias("s")
        )
        .item()
    )
    return hashlib.sha1(f"{joined}|{salt}".encode("utf-8")).hexdigest()


def _condition_salt(condition: SignalConfig) -> str:
    return f"{condition.condition_id}|{','.join(condition.envelope_columns)}"


def _condition_keys_for_day(
    envelopes: Mapping[str, pl.DataFrame], date: str
) -> dict[str, pl.DataFrame]:
    # Empty frames are kept (not dropped) so envelope-column dtypes survive on
    # days where a condition selected nothing.
    return {
        st_name: index.filter(pl.col("Date") == date).drop("Date")
        for st_name, index in envelopes.items()
    }


def _union_keys(condition_keys: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    frames = [k.select(list(KEY_COLS)) for k in condition_keys.values() if k.height]
    if not frames:
        return pl.DataFrame(schema={"QuoteCode": pl.String, "ChannelSeq": pl.UInt64})
    return pl.concat(frames, how="vertical").unique().sort(list(KEY_COLS))


def build_store_day(
    date: str,
    condition_keys: Mapping[str, pl.DataFrame],
    conditions: Sequence[SignalConfig],
    *,
    roots: DataRoots,
    options: MergeOptions,
    txf_loader: TxfDataLoader | None = None,
    existing: pl.DataFrame | None = None,
    merge_keys: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, str]:
    """Return the tagged store frame for one day and the build mode used.

    With ``existing`` the day is patched: only ``merge_keys`` (keys the caller
    determined are new for this day) are merged, then every tag is recomputed.
    ``merge_keys=None`` falls back to "union minus keys already present".
    """

    union = _union_keys(condition_keys)
    if union.is_empty() and existing is None:
        return pl.DataFrame(), "empty"
    mode = "build"
    if existing is not None and existing.height:
        present = existing.select(list(KEY_COLS)).unique()
        if merge_keys is None:
            merge_keys = union.join(present, on=list(KEY_COLS), how="anti")
        else:
            merge_keys = merge_keys.select(list(KEY_COLS)).join(present, on=list(KEY_COLS), how="anti")
        if merge_keys.is_empty():
            merged = existing
            mode = "retag"
        else:
            new_rows = materialize_event_day(
                date,
                merge_keys.with_columns(pl.lit(date).alias("Date")).select(list(EVENT_KEY_COLS)),
                roots=roots,
                options=options,
                txf_loader=txf_loader,
            ).frame
            old_cols = {c for c in existing.columns if not c.startswith("cond_")}
            if old_cols != set(new_rows.columns):
                only_old = sorted(old_cols - set(new_rows.columns))
                only_new = sorted(set(new_rows.columns) - old_cols)
                print(
                    f"  [WARN] {date}: patch schema drift (only existing: {only_old[:5]}, "
                    f"only new: {only_new[:5]}); using diagonal concat"
                )
            merged = pl.concat([existing, new_rows], how="diagonal_relaxed")
            mode = "patch"
    else:
        merged = materialize_event_day(
            date,
            union.with_columns(pl.lit(date).alias("Date")).select(list(EVENT_KEY_COLS)),
            roots=roots,
            options=options,
            txf_loader=txf_loader,
        ).frame
    merged = _attach_envelope_columns(merged, condition_keys, conditions)
    tagged = _tag_frame(merged, condition_keys, conditions)
    tagged = tagged.sort(list(KEY_COLS))
    return tagged, mode


def build_store(
    registry: SignalRegistry | Sequence[SignalConfig],
    *,
    roots: DataRoots,
    store_dir: Path | str,
    envelope_dir: Path | str,
    start: str | None = None,
    end: str | None = None,
    force: bool = False,
    refresh_latest: bool = True,
    only: Sequence[str] | None = None,
    options: MergeOptions | None = None,
    verbose: bool = True,
) -> StoreBuildReport:
    """Materialize the union of every registered envelope into tagged daily files."""

    _validate_bounds(start, end)
    # The store is shared: a day is always materialized and tagged for EVERY registered
    # signal.  Building with a subset would drop the other signals' rows (and tags) from
    # any day it rebuilds or refreshes, so ``only`` merely validates the names here; it
    # narrows the envelope scan upstream, never the store.
    if only is not None and isinstance(registry, SignalRegistry):
        registry.select(only)
    conditions = tuple(registry)
    if not conditions:
        raise ValueError("no conditions to build")
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    options = options or store_merge_options_v1()
    manifest_path = store_manifest_path(store_dir)
    manifest = _read_json(manifest_path)
    stored_options = manifest.get("merge_options")
    if stored_options and stored_options != _merge_options_payload(options):
        raise StoreContractError(
            f"{store_dir}: existing store was built with different merge options; "
            "use a new store directory or rebuild with --force after clearing it"
        )

    envelopes = {c.st_name: read_envelope_index(envelope_dir, c) for c in conditions}
    all_dates = sorted({d for idx in envelopes.values() for d in idx["Date"].unique().to_list()})
    dates = [d for d in all_dates if (start is None or d >= start) and (end is None or d <= end)]
    existing_dates = store_dates(store_dir)
    latest_existing = existing_dates[-1] if existing_dates else None

    txf_loader = TxfDataLoader(roots.txf_tick_dir)
    built: list[str] = []
    patched: list[str] = []
    refreshed: list[str] = []
    skipped: list[str] = []
    empty: list[str] = []
    day_manifest: dict[str, object] = dict(manifest.get("days", {}))
    expected_tags = {c.tag_column for c in conditions}

    if verbose:
        print(f"store: {len(dates)} envelope date(s), {len(existing_dates)} existing day file(s)")
    for position, date in enumerate(dates, start=1):
        path = store_day_path(store_dir, date)
        condition_keys = _condition_keys_for_day(envelopes, date)
        env_hashes = {
            c.st_name: _keys_hash(condition_keys.get(c.st_name, _union_keys({})), _condition_salt(c))
            for c in conditions
        }
        if not any(k.height for k in condition_keys.values()) and not path.exists():
            empty.append(date)
            continue
        recorded_raw = day_manifest.get(date, {})
        recorded = recorded_raw.get("conditions", {}) if isinstance(recorded_raw, dict) else {}
        existing = None
        merge_keys = None
        mode_hint = "build"
        if path.exists():
            # fail closed: never rebuild / refresh / patch a day for fewer signals than it carries
            foreign = sorted(
                c for c in pl.read_parquet_schema(path) if c.startswith("cond_") and c not in expected_tags
            )
            if foreign:
                raise StoreContractError(
                    f"{path}: day carries tags {foreign} for signals not in this build; "
                    "build the store with the full registry"
                )
        if path.exists() and not force:
            existing = pl.read_parquet(path)
            if refresh_latest and date == latest_existing:
                existing = None
                mode_hint = "refresh"
            else:
                has_tags = expected_tags.issubset(existing.columns)
                changed = [
                    st
                    for st, digest in env_hashes.items()
                    if not isinstance(recorded.get(st), dict)
                    or recorded[st].get("envelope_hash") != digest
                ]
                if has_tags and not changed:
                    skipped.append(date)
                    continue
                candidate = _union_keys({st: condition_keys[st] for st in changed if st in condition_keys})
                merge_keys = candidate
                mode_hint = "patch"
        try:
            frame, mode = build_store_day(
                date,
                condition_keys,
                conditions,
                roots=roots,
                options=options,
                txf_loader=txf_loader,
                existing=existing,
                merge_keys=merge_keys,
            )
        except FileNotFoundError as exc:
            print(f"  [{position}/{len(dates)}] {date}: skip, {exc}")
            skipped.append(date)
            continue
        if frame.is_empty():
            empty.append(date)
            if verbose:
                print(f"  [{position}/{len(dates)}] {date}: 0 rows after tagging")
            continue
        _atomic_write_parquet(frame, path)
        counts = {c.st_name: int(frame[c.tag_column].sum()) for c in conditions}
        day_conditions = {
            st: info for st, info in recorded.items() if isinstance(info, dict) and st not in counts
        }
        for c in conditions:
            keys = condition_keys.get(c.st_name)
            day_conditions[c.st_name] = {
                "rows": counts[c.st_name],
                "envelope_rows": int(keys.height) if keys is not None else 0,
                "envelope_hash": env_hashes[c.st_name],
            }
        day_manifest[date] = {
            "rows": int(frame.height),
            "columns": int(frame.width),
            "conditions": day_conditions,
            "mode": mode_hint if mode_hint != "build" else mode,
            "built_at": _dt.datetime.now().isoformat(timespec="seconds"),
        }
        if mode_hint == "refresh":
            refreshed.append(date)
        elif mode == "patch" or mode == "retag":
            patched.append(date)
        else:
            built.append(date)
        if verbose:
            print(
                f"  [{position}/{len(dates)}] {date}: {frame.height:,} rows "
                f"({', '.join(f'{k}={v:,}' for k, v in counts.items())}) [{day_manifest[date]['mode']}]"
            )

    payload = {
        "format_version": STORE_FORMAT_VERSION,
        "merge_options": _merge_options_payload(options),
        "conditions": {
            **dict(manifest.get("conditions", {})),
            **{
                c.st_name: {
                    "condition_id": c.condition_id,
                    "version": c.version,
                    "condition_hash": c.stable_hash(),
                    "envelope_columns": [c.envelope_store_column(x) for x in c.envelope_columns],
                    "universe": [c.universe.universe_id, c.universe.universe_version],
                }
                for c in conditions
            },
        },
        "days": dict(sorted(day_manifest.items())),
        "updated_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    _atomic_write_json(payload, manifest_path)
    if verbose:
        print(
            f"store done: built={len(built)} patched={len(patched)} refreshed={len(refreshed)} "
            f"skipped={len(skipped)} empty={len(empty)} -> {store_dir}"
        )
    return StoreBuildReport(
        tuple(built), tuple(patched), tuple(refreshed), tuple(skipped), tuple(empty)
    )


def read_store(
    store_dir: Path | str,
    *,
    dates: Sequence[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    columns: Sequence[str] | None = None,
    condition: SignalConfig | str | None = None,
    lazy: bool = False,
) -> pl.DataFrame | pl.LazyFrame:
    """Read store days; ``condition`` keeps only rows tagged for that signal."""

    _validate_bounds(start, end)
    root = Path(store_dir)
    available = store_dates(root)
    if dates is not None:
        wanted = [d for d in dates if d in set(available)]
        missing = sorted(set(dates) - set(available))
        if missing:
            print(f"  [WARN] store is missing {len(missing)} requested date(s): {missing[:5]}")
    else:
        wanted = [d for d in available if (start is None or d >= start) and (end is None or d <= end)]
    if not wanted:
        empty = pl.DataFrame()
        return empty.lazy() if lazy else empty
    tag = None
    if condition is not None:
        tag = condition if isinstance(condition, str) else condition.tag_column
        if not tag.startswith("cond_"):
            tag = f"cond_{tag}"
    frames = []
    for date in wanted:
        scan = pl.scan_parquet(store_day_path(root, date))
        if tag is not None:
            scan = scan.filter(pl.col(tag))
        if columns is not None:
            keep = [c for c in columns if c in scan.collect_schema().names()]
            scan = scan.select(keep)
        frames.append(scan)
    out = pl.concat(frames, how="diagonal_relaxed")
    return out if lazy else out.collect()


__all__ = [
    "EnvelopeBuildReport",
    "STORE_FORMAT_VERSION",
    "StoreBuildReport",
    "StoreContractError",
    "build_envelope_index",
    "build_store",
    "build_store_day",
    "envelope_index_path",
    "read_envelope_index",
    "read_store",
    "select_envelope_day",
    "store_dates",
    "store_day_path",
    "store_manifest_path",
    "store_merge_options_v1",
]
