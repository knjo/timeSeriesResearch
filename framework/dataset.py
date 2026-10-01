"""Generic, audited materialization of event-index rows.

The event selector and the dataset merger are deliberately separate.  An event
index says *which* ticks are interesting; this module only attaches source data
to those immutable keys.  Strategy-specific universe filters are explicit,
versioned :class:`UniverseSpec` objects rather than hidden clauses in the
merger.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal

import polars as pl

from src.dataloader.preMarketJoin import join_pre_market_features
from src.dataloader.txfDataLoader import TxfDataLoader, attach_txf_to_1330_return
from src.features.market_beta import add_txf_event_residual_vol

from .events import DataRoots, EVENT_KEY_COLS, validate_event_index


TICK_KEY_COLS = ("QuoteCode", "ChannelSeq")
TickBarAlignment = Literal["causal_completed_slot", "legacy_stock_tick_time"]
SchemaPolicy = Literal["strict", "union"]


class DatasetContractError(ValueError):
    """Raised when a source violates the event-dataset contract."""


class MissingDatasetSourceError(FileNotFoundError):
    """Raised when a source declared as required is unavailable."""


FrameFilter = Callable[[pl.DataFrame], pl.DataFrame]


@dataclass(frozen=True)
class UniverseSpec:
    """A named, versioned filter applied after source attachment.

    ``filter_fn`` must only remove rows.  The merger verifies that it neither
    duplicates events nor invents new event keys.  ``None`` means the complete
    event-index population.
    """

    universe_id: str
    universe_version: str
    filter_fn: FrameFilter | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.universe_id or not self.universe_version:
            raise ValueError("universe_id and universe_version are required")
        if self.filter_fn is not None and not callable(self.filter_fn):
            raise TypeError("filter_fn must be callable")

    def apply(self, frame: pl.DataFrame) -> pl.DataFrame:
        if self.filter_fn is None:
            return frame
        filtered = self.filter_fn(frame)
        if not isinstance(filtered, pl.DataFrame):
            raise DatasetContractError(
                f"universe {self.universe_id!r} must return a polars DataFrame"
            )
        return filtered


ALL_EVENTS_UNIVERSE = UniverseSpec("all_events", "1")


@dataclass(frozen=True)
class MergeOptions:
    """Source and causality choices for a materialized event dataset."""

    universe: UniverseSpec = ALL_EVENTS_UNIVERSE
    require_pre_market: bool = True
    attach_tick_bar: bool = True
    require_tick_bar: bool = True
    tick_bar_alignment: TickBarAlignment = "causal_completed_slot"
    tick_bar_columns: tuple[str, ...] | None = None
    attach_txf: bool = True
    require_txf: bool = False
    strict_cardinality: bool = True
    feature_suffix: str = "_feature"
    schema_policy: SchemaPolicy = "strict"
    preserve_tick_data_column_order: bool = False
    attach_brench: bool = False
    require_brench: bool = False

    def __post_init__(self) -> None:
        if self.tick_bar_alignment not in {
            "causal_completed_slot",
            "legacy_stock_tick_time",
        }:
            raise ValueError(
                f"unknown tick_bar_alignment={self.tick_bar_alignment!r}"
            )
        if self.schema_policy not in {"strict", "union"}:
            raise ValueError(f"unknown schema_policy={self.schema_policy!r}")
        if self.tick_bar_columns is not None and len(
            self.tick_bar_columns
        ) != len(set(self.tick_bar_columns)):
            raise ValueError("tick_bar_columns cannot contain duplicates")
        if self.require_txf and not self.attach_txf:
            raise ValueError("require_txf=True requires attach_txf=True")
        if not self.feature_suffix:
            raise ValueError("feature_suffix cannot be empty")



@dataclass(frozen=True)
class EventDatasetSpec:
    event_id: str
    dataset_version: str
    index_path: Path
    output_dir: Path
    merge_options: MergeOptions = field(default_factory=MergeOptions)
    allow_unmanifested_existing: bool = False

    def __post_init__(self) -> None:
        if not self.event_id or not self.dataset_version:
            raise ValueError("event_id and dataset_version are required")
        object.__setattr__(self, "index_path", Path(self.index_path))
        object.__setattr__(self, "output_dir", Path(self.output_dir))

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "_dataset_manifest.json"


@dataclass(frozen=True)
class DayMergeAudit:
    date: str
    indexed_rows: int
    materialized_rows: int
    universe_removed_rows: int
    tick_bar_alignment: str | None
    tick_bar_attached: bool
    tick_bar_matched_rows: int | None
    txf_available_rows: int | None


@dataclass(frozen=True)
class DayMergeResult:
    frame: pl.DataFrame
    audit: DayMergeAudit


@dataclass(frozen=True)
class DatasetBuildReport:
    event_id: str
    dataset_version: str
    saved_dates: tuple[str, ...]
    skipped_dates: tuple[str, ...]
    day_audits: tuple[DayMergeAudit, ...]


def _source_path(roots: DataRoots, source: str, date: str) -> Path:
    templates = {
        "tick_data": roots.tick_data_dir / f"{date}_StockTick.parquet",
        "tick_feature": roots.tick_feature_dir / f"{date}_tickFeature.parquet",
        "pre_market": roots.pre_market_dir / f"{date}_preMarketData.parquet",
        "tick_bar": roots.tick_bar_dir / f"{date}_tickBar.parquet",
    }
    if source == "brench_feature":
        return roots.source_dir("brench_feature") / f"{date}_brenchFeature.parquet"
    return templates[source]


def _read_required(path: Path, source: str, date: str) -> pl.DataFrame:
    if not path.exists():
        raise MissingDatasetSourceError(f"{date}: missing {source}: {path}")
    return pl.read_parquet(path)


def _require_columns(
    frame: pl.DataFrame | pl.LazyFrame,
    columns: tuple[str, ...] | list[str],
    context: str,
) -> None:
    available = (
        set(frame.collect_schema().names())
        if isinstance(frame, pl.LazyFrame)
        else set(frame.columns)
    )
    missing = [column for column in columns if column not in available]
    if missing:
        raise DatasetContractError(f"{context}: missing columns {missing}")


def _assert_unique(frame: pl.DataFrame, keys: tuple[str, ...], context: str) -> None:
    if frame.is_empty():
        return
    nulls = frame.select(
        pl.any_horizontal([pl.col(key).is_null() for key in keys]).any()
    ).item()
    if bool(nulls):
        raise DatasetContractError(f"{context}: null event key in {keys}")
    duplicates = frame.group_by(list(keys)).len().filter(pl.col("len") > 1)
    if not duplicates.is_empty():
        sample = duplicates.head(3).to_dicts()
        raise DatasetContractError(
            f"{context}: duplicate keys {keys}; sample={sample}"
        )


def _assert_same_event_set(
    before: pl.DataFrame,
    after: pl.DataFrame,
    *,
    context: str,
    allow_subset: bool = False,
) -> None:
    keys = list(TICK_KEY_COLS)
    before_keys = before.select(keys)
    after_keys = after.select(keys)
    _assert_unique(after_keys, TICK_KEY_COLS, context)
    invented = after_keys.join(before_keys, on=keys, how="anti")
    if not invented.is_empty():
        raise DatasetContractError(f"{context}: invented event keys")
    if not allow_subset:
        missing = before_keys.join(after_keys, on=keys, how="anti")
        if not missing.is_empty():
            raise DatasetContractError(
                f"{context}: lost {len(missing)} of {len(before_keys)} event rows"
            )


def _prepare_day_index(date: str, index_rows: pl.DataFrame) -> pl.DataFrame:
    _require_columns(index_rows, list(TICK_KEY_COLS), f"{date} event index")
    if "Date" in index_rows.columns:
        mismatched = index_rows.filter(pl.col("Date").cast(pl.String) != date)
        if not mismatched.is_empty():
            raise DatasetContractError(f"{date}: index rows contain another Date")
        index_rows = index_rows.drop("Date")
    _assert_unique(index_rows, TICK_KEY_COLS, f"{date} event index")
    return index_rows.with_row_index("_event_order")


def _attach_tick_sources(
    date: str,
    event_rows: pl.DataFrame,
    roots: DataRoots,
    options: MergeOptions,
) -> pl.DataFrame:
    event_keys = event_rows.select(list(TICK_KEY_COLS)).lazy()
    selected_sources: dict[str, pl.DataFrame] = {}
    for source_name, display_name in (
        ("tick_data", "tickData"),
        ("tick_feature", "tickFeature"),
    ):
        path = _source_path(roots, source_name, date)
        if not path.exists():
            raise MissingDatasetSourceError(
                f"{date}: missing {display_name}: {path}"
            )
        source = pl.scan_parquet(path)
        _require_columns(source, list(TICK_KEY_COLS), f"{date} {display_name}")
        selected = source.join(
            event_keys,
            on=list(TICK_KEY_COLS),
            how="semi",
        ).collect()
        _assert_unique(
            selected, TICK_KEY_COLS, f"{date} selected {display_name}"
        )
        _assert_same_event_set(
            event_rows, selected, context=f"{date} selected {display_name}"
        )
        selected_sources[source_name] = selected

    tick_data = selected_sources["tick_data"]
    tick_feature = selected_sources["tick_feature"]

    metadata_columns = set(event_rows.columns) - set(TICK_KEY_COLS) - {"_event_order"}
    source_columns = (set(tick_data.columns) | set(tick_feature.columns)) - set(
        TICK_KEY_COLS
    )
    collisions = sorted(metadata_columns & source_columns)
    if collisions:
        raise DatasetContractError(
            f"{date}: event-index metadata collides with source columns: {collisions}; "
            "prefix selector metadata with event_"
        )

    merged = event_rows.join(
        tick_data, on=list(TICK_KEY_COLS), how="left", validate="1:1"
    ).join(
        tick_feature,
        on=list(TICK_KEY_COLS),
        how="left",
        validate="1:1",
        suffix=options.feature_suffix,
    )
    if options.strict_cardinality:
        _assert_same_event_set(event_rows, merged, context=f"{date} tick merge")
    return merged


def _attach_pre_market(
    date: str,
    merged: pl.DataFrame,
    roots: DataRoots,
    options: MergeOptions,
) -> pl.DataFrame:
    path = _source_path(roots, "pre_market", date)
    if not path.exists():
        if options.require_pre_market:
            raise MissingDatasetSourceError(f"{date}: missing preMarket: {path}")
        return merged
    pre_market = pl.read_parquet(path)
    _require_columns(pre_market, ["QuoteCode"], f"{date} preMarket")
    _assert_unique(pre_market, ("QuoteCode",), f"{date} preMarket")
    before = merged
    merged = join_pre_market_features(merged, pre_market, validate="m:1")
    if options.strict_cardinality:
        _assert_same_event_set(before, merged, context=f"{date} preMarket merge")
    return merged


def _attach_brench(
    date: str,
    merged: pl.DataFrame,
    roots: DataRoots,
    options: MergeOptions,
) -> pl.DataFrame:
    """Attach the decision-day brenchFeature file (prior-day branch features).

    Mirrors the deployed negFill loader: the file named for this day carries
    the previous session's branch flows, is keyed by QuoteCode, and joins with
    the ``_bf`` suffix on any column that already exists.
    """

    if not options.attach_brench:
        return merged
    if roots.brench_feature_dir is None:
        if options.require_brench:
            raise MissingDatasetSourceError(f"{date}: brenchFeature root is not configured")
        return merged
    path = _source_path(roots, "brench_feature", date)
    if not path.exists():
        if options.require_brench:
            raise MissingDatasetSourceError(f"{date}: missing brenchFeature: {path}")
        return merged
    brench = pl.read_parquet(path).drop("feature_date", "decision_date", strict=False)
    _require_columns(brench, ["QuoteCode"], f"{date} brenchFeature")
    _assert_unique(brench, ("QuoteCode",), f"{date} brenchFeature")
    before = merged
    merged = merged.join(brench, on="QuoteCode", how="left", suffix="_bf")
    if options.strict_cardinality:
        _assert_same_event_set(before, merged, context=f"{date} brenchFeature merge")
    return merged


def _tick_bar_columns(frame: pl.DataFrame, requested: tuple[str, ...] | None) -> list[str]:
    if requested is not None:
        missing = [column for column in requested if column not in frame.columns]
        if missing:
            raise DatasetContractError(f"tickBar: requested columns unavailable: {missing}")
        features = list(requested)
    else:
        features = [
            column
            for column in frame.columns
            if column.endswith("Rank") or column.endswith("Count")
        ]
        if "AmountRank_canDayTrade" in frame.columns:
            features.append("AmountRank_canDayTrade")
    ordered = ["QuoteCode", "TransTime", "TimeSlot", *features]
    return list(dict.fromkeys(column for column in ordered if column in frame.columns))


def _attach_tick_bar(
    date: str,
    merged: pl.DataFrame,
    roots: DataRoots,
    options: MergeOptions,
) -> pl.DataFrame:
    if not options.attach_tick_bar:
        return merged
    path = _source_path(roots, "tick_bar", date)
    if not path.exists():
        if options.require_tick_bar:
            raise MissingDatasetSourceError(f"{date}: missing tickBar: {path}")
        return merged
    tick_bar = pl.read_parquet(path)
    _require_columns(
        tick_bar, ["QuoteCode", "TransTime", "TimeSlot"], f"{date} tickBar"
    )
    tick_bar = tick_bar.select(_tick_bar_columns(tick_bar, options.tick_bar_columns))
    _require_columns(merged, ["QuoteCode", "TransTime"], f"{date} merged events")
    before = merged

    if options.tick_bar_alignment == "causal_completed_slot":
        _assert_unique(tick_bar, ("QuoteCode", "TimeSlot"), f"{date} tickBar slots")
        tick_bar = (
            tick_bar.rename({"TransTime": "tick_bar_source_time"})
            .with_columns(
                (
                    pl.lit(datetime.strptime(date, "%Y%m%d").date())
                    .dt.combine(pl.col("TimeSlot"))
                    + pl.duration(microseconds=1)
                ).alias("_tick_bar_available_at")
            )
            .sort(["QuoteCode", "_tick_bar_available_at"])
        )
        merged = merged.sort(["QuoteCode", "TransTime"]).join_asof(
            tick_bar,
            left_on="TransTime",
            right_on="_tick_bar_available_at",
            by="QuoteCode",
            strategy="backward",
        ).drop("_tick_bar_available_at")
    elif options.tick_bar_alignment == "legacy_stock_tick_time":
        # Historical behavior.  Multiple TimeSlots can share the same last
        # stock tick, so this mode is intentionally not presented as causal.
        tick_bar = (
            tick_bar.sort(["QuoteCode", "TransTime"])
            .rename({"TransTime": "TransTime_tb"})
            .with_columns(
                (pl.col("TransTime_tb") + pl.duration(microseconds=1)).alias(
                    "TransTime_tb"
                )
            )
        )
        merged = merged.sort(["QuoteCode", "TransTime"]).join_asof(
            tick_bar,
            left_on="TransTime",
            right_on="TransTime_tb",
            by="QuoteCode",
            strategy="backward",
        )
    else:  # pragma: no cover - protected by the public literal type
        raise DatasetContractError(
            f"unknown tick_bar_alignment={options.tick_bar_alignment!r}"
        )
    if options.strict_cardinality:
        _assert_same_event_set(before, merged, context=f"{date} tickBar merge")
    return merged


def materialize_event_day(
    date: str,
    index_rows: pl.DataFrame,
    *,
    roots: DataRoots,
    options: MergeOptions | None = None,
    txf_loader: TxfDataLoader | None = None,
) -> DayMergeResult:
    """Attach all configured sources to one day's immutable event keys."""

    options = options or MergeOptions()
    event_rows = _prepare_day_index(date, index_rows)
    indexed_rows = len(event_rows)
    merged = _attach_tick_sources(date, event_rows, roots, options)
    merged = _attach_pre_market(date, merged, roots, options)
    merged = _attach_brench(date, merged, roots, options)

    before_universe = merged
    proposed = options.universe.apply(merged)
    _assert_same_event_set(
        before_universe,
        proposed,
        context=f"{date} universe {options.universe.universe_id}",
        allow_subset=True,
    )
    # A universe is only authorized to choose rows. Ignore any value/schema
    # mutation performed by its callback and semi-join the selected keys back
    # to the canonical pre-filter frame.
    merged = before_universe.join(
        proposed.select(list(TICK_KEY_COLS)),
        on=list(TICK_KEY_COLS),
        how="semi",
    )
    universe_removed = len(before_universe) - len(merged)

    merged = _attach_tick_bar(date, merged, roots, options)
    tick_bar_attached = (
        options.attach_tick_bar
        and _source_path(roots, "tick_bar", date).exists()
    )
    tick_bar_match_column = (
        "tick_bar_source_time"
        if options.tick_bar_alignment == "causal_completed_slot"
        else "TransTime_tb"
    )
    tick_bar_matched = (
        merged.select(pl.col(tick_bar_match_column).is_not_null().sum()).item()
        if tick_bar_attached and tick_bar_match_column in merged.columns
        else None
    )
    if options.attach_txf:
        before_txf = merged
        loader = txf_loader or TxfDataLoader(roots.txf_tick_dir)
        merged = add_txf_event_residual_vol(
            attach_txf_to_1330_return(merged, date, loader)
        )
        if options.strict_cardinality:
            _assert_same_event_set(before_txf, merged, context=f"{date} TXF merge")
        if (
            options.require_txf
            and not merged.is_empty()
            and (
                "txf_source" not in merged.columns
                or (merged["txf_source"] == "unavailable").all()
            )
        ):
            raise MissingDatasetSourceError(
                f"{date}: TXF was required but no usable day source was available"
            )

    merged = (
        merged.with_columns(pl.lit(date).alias("Date"))
        .sort("_event_order")
        .drop("_event_order")
    )
    if options.preserve_tick_data_column_order:
        tick_order = list(
            pl.read_parquet_schema(_source_path(roots, "tick_data", date))
        )
        ordered = [column for column in tick_order if column in merged.columns]
        ordered.extend(column for column in merged.columns if column not in ordered)
        merged = merged.select(ordered)
    txf_available = None
    if "txf_to_1330_return" in merged.columns:
        txf_available = merged.select(
            pl.col("txf_to_1330_return").is_not_null().sum()
        ).item()
    audit = DayMergeAudit(
        date=date,
        indexed_rows=indexed_rows,
        materialized_rows=len(merged),
        universe_removed_rows=universe_removed,
        tick_bar_alignment=(
            options.tick_bar_alignment if tick_bar_attached else None
        ),
        tick_bar_attached=tick_bar_attached,
        tick_bar_matched_rows=tick_bar_matched,
        txf_available_rows=txf_available,
    )
    return DayMergeResult(merged, audit)


def merge_event_day(
    date: str,
    index_rows: pl.DataFrame,
    *,
    roots: DataRoots,
    options: MergeOptions | None = None,
    txf_loader: TxfDataLoader | None = None,
) -> pl.DataFrame:
    """Convenience wrapper returning only the materialized frame."""

    return materialize_event_day(
        date,
        index_rows,
        roots=roots,
        options=options,
        txf_loader=txf_loader,
    ).frame


def _event_index_contract(index_path: Path) -> dict[str, object]:
    manifest_path = index_path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        return {"status": "unmanifested", "manifest_path": str(manifest_path)}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetContractError(
            f"cannot read event-index manifest: {manifest_path}"
        ) from exc
    fields = (
        "format_version",
        "event_id",
        "selector_version",
        "key_columns",
        "key_schema",
    )
    return {
        "status": "manifested",
        "manifest_path": str(manifest_path.resolve()),
        **{field: manifest.get(field) for field in fields},
    }


def _manifest_payload(
    spec: EventDatasetSpec, roots: DataRoots
) -> dict[str, object]:
    options = spec.merge_options
    return {
        "contract": "event_dataset/v1",
        "event_id": spec.event_id,
        "dataset_version": spec.dataset_version,
        "index_path": str(spec.index_path.resolve()),
        "universe_id": options.universe.universe_id,
        "universe_version": options.universe.universe_version,
        "require_pre_market": options.require_pre_market,
        "tick_bar_alignment": options.tick_bar_alignment,
        "attach_tick_bar": options.attach_tick_bar,
        "require_tick_bar": options.require_tick_bar,
        "tick_bar_columns": (
            list(options.tick_bar_columns)
            if options.tick_bar_columns is not None
            else None
        ),
        "attach_txf": options.attach_txf,
        # Only recorded when enabled so pre-existing study manifests keep
        # their identity; the default (False) is the historical behavior.
        **({"attach_brench": True} if options.attach_brench else {}),
        "require_txf": options.require_txf,
        "strict_cardinality": options.strict_cardinality,
        "feature_suffix": options.feature_suffix,
        "schema_policy": options.schema_policy,
        "preserve_tick_data_column_order": (
            options.preserve_tick_data_column_order
        ),
        "event_index_contract": _event_index_contract(spec.index_path),
        "data_roots": {
            "tick_data_dir": str(roots.tick_data_dir.resolve()),
            "tick_feature_dir": str(roots.tick_feature_dir.resolve()),
            "pre_market_dir": str(roots.pre_market_dir.resolve()),
            "tick_bar_dir": str(roots.tick_bar_dir.resolve()),
            "txf_tick_dir": str(roots.txf_tick_dir.resolve()),
        },
    }


def _atomic_write_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
        encoding="utf-8",
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _atomic_write_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temp_path = Path(handle.name)
    handle.close()
    try:
        frame.write_parquet(temp_path, compression="zstd")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _check_or_initialize_manifest(
    spec: EventDatasetSpec, roots: DataRoots
) -> dict[str, object]:
    expected = _manifest_payload(spec, roots)
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    if spec.manifest_path.exists():
        try:
            actual = json.loads(spec.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetContractError(
                f"cannot read dataset manifest: {spec.manifest_path}"
            ) from exc
        # A legacy dataset may have been adopted before the event-index
        # manifest existed. Per-day key identities already protect its
        # partitions, so binding that newly available manifest is a safe
        # metadata upgrade rather than a contract change.
        old_index_contract = actual.get("event_index_contract", {})
        new_index_contract = expected.get("event_index_contract", {})
        safe_index_upgrade = (
            spec.allow_unmanifested_existing
            and isinstance(old_index_contract, dict)
            and old_index_contract.get("status") == "unmanifested"
            and isinstance(new_index_contract, dict)
            and new_index_contract.get("status") == "manifested"
        )
        if safe_index_upgrade:
            actual["event_index_contract"] = new_index_contract
        mismatches = {
            key: (actual.get(key), value)
            for key, value in expected.items()
            if actual.get(key) != value
        }
        if mismatches:
            raise DatasetContractError(
                f"dataset manifest does not match requested spec: {mismatches}"
            )
        actual.setdefault("partitions", {})
        if safe_index_upgrade:
            actual["event_index_lineage_upgraded"] = True
            _atomic_write_json(actual, spec.manifest_path)
        return actual
    existing = list(spec.output_dir.glob("*.parquet"))
    if existing and not spec.allow_unmanifested_existing:
        raise DatasetContractError(
            f"{spec.output_dir} contains unmanifested parquet files; choose a new "
            "versioned output directory or explicitly allow legacy data"
        )
    payload = {
        **expected,
        "created_at": datetime.now().astimezone().isoformat(),
        "partitions": {},
    }
    _atomic_write_json(payload, spec.manifest_path)
    return payload


def _schema_signature(frame: pl.DataFrame) -> list[list[str]]:
    # A strict schema includes column order; Polars vertical concat requires it.
    return [[name, str(dtype)] for name, dtype in frame.schema.items()]


def _validate_bound(value: str | None, label: str) -> None:
    if value is None:
        return
    try:
        parsed = datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise ValueError(f"{label} must be a valid YYYYMMDD date") from exc
    if parsed.strftime("%Y%m%d") != value:
        raise ValueError(f"{label} must be exactly eight YYYYMMDD digits")


def _index_day_identities(index: pl.DataFrame) -> dict[str, dict[str, int]]:
    identities = (
        index.group_by("Date").agg(
            pl.len().alias("row_count"),
            pl.struct(["QuoteCode", "ChannelSeq"])
            .hash(seed=0)
            .sum()
            .alias("key_hash_sum_0"),
            pl.struct(["QuoteCode", "ChannelSeq"])
            .hash(seed=1)
            .sum()
            .alias("key_hash_sum_1"),
        )
    )
    return {
        str(row["Date"]): {
            "row_count": int(row["row_count"]),
            "key_hash_sum_0": int(row["key_hash_sum_0"]),
            "key_hash_sum_1": int(row["key_hash_sum_1"]),
        }
        for row in identities.iter_rows(named=True)
    }


def _path_identity(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path.resolve()), "exists": False}
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _day_source_identities(
    date: str, roots: DataRoots, options: MergeOptions
) -> dict[str, object]:
    sources = {
        "tick_data": _source_path(roots, "tick_data", date),
        "tick_feature": _source_path(roots, "tick_feature", date),
    }
    if options.require_pre_market or options.universe != ALL_EVENTS_UNIVERSE:
        sources["pre_market"] = _source_path(roots, "pre_market", date)
    elif _source_path(roots, "pre_market", date).exists():
        sources["pre_market"] = _source_path(roots, "pre_market", date)
    if options.attach_tick_bar:
        sources["tick_bar"] = _source_path(roots, "tick_bar", date)
    if options.attach_txf:
        sources["txf_tick"] = roots.txf_tick_dir / f"{date}_TxfTick.parquet"
    return {name: _path_identity(path) for name, path in sources.items()}


def build_event_dataset(
    spec: EventDatasetSpec,
    *,
    roots: DataRoots,
    force: bool = False,
    start: str | None = None,
    end: str | None = None,
    refresh_latest: bool = False,
) -> DatasetBuildReport:
    """Materialize all indexed days, failing closed on any source error."""

    if not spec.index_path.exists():
        raise FileNotFoundError(f"event index not found: {spec.index_path}")
    _validate_bound(start, "start")
    _validate_bound(end, "end")
    if start is not None and end is not None and start > end:
        raise ValueError("start must be <= end")
    index = pl.read_parquet(spec.index_path)
    validate_event_index(index, require_sorted=True)
    # Only publish dataset metadata after the complete index contract passes.
    manifest = _check_or_initialize_manifest(spec, roots)
    all_index_identities = _index_day_identities(index)
    dates = index["Date"].unique().sort().to_list()
    if start is not None:
        dates = [date for date in dates if date >= start]
    if end is not None:
        dates = [date for date in dates if date <= end]
    latest_existing = max(
        (path.stem for path in spec.output_dir.glob("*.parquet")), default=None
    )
    loader = TxfDataLoader(roots.txf_tick_dir)
    saved: list[str] = []
    skipped: list[str] = []
    audits: list[DayMergeAudit] = []
    expected_schema = manifest.get("schema")
    partitions = manifest.setdefault("partitions", {})

    for date in dates:
        output_path = spec.output_dir / f"{date}.parquet"
        should_refresh = refresh_latest and date == latest_existing
        index_identity = all_index_identities[date]
        source_identities = _day_source_identities(
            date, roots, spec.merge_options
        )
        prior_partition = partitions.get(date)
        if output_path.exists() and not force and not should_refresh:
            if prior_partition is None:
                if not spec.allow_unmanifested_existing:
                    raise DatasetContractError(
                        f"{date}: output exists without partition lineage"
                    )
                try:
                    existing_schema_frame = pl.read_parquet(output_path, n_rows=0)
                    existing_rows = pl.scan_parquet(output_path).select(
                        pl.len()
                    ).collect().item()
                except Exception as exc:
                    raise DatasetContractError(
                        f"{date}: cannot adopt unreadable legacy partition"
                    ) from exc
                partitions[date] = {
                    "status": "legacy_adopted_unverified",
                    "index": index_identity,
                    "sources": source_identities,
                    "row_count": int(existing_rows),
                    "schema": _schema_signature(existing_schema_frame),
                }
                _atomic_write_json(manifest, spec.manifest_path)
                skipped.append(date)
                continue
            if (
                prior_partition.get("index") == index_identity
                and prior_partition.get("sources") == source_identities
            ):
                try:
                    existing_schema = _schema_signature(
                        pl.read_parquet(output_path, n_rows=0)
                    )
                except Exception as exc:
                    raise DatasetContractError(
                        f"{date}: existing partition is unreadable"
                    ) from exc
                if existing_schema != prior_partition.get("schema"):
                    raise DatasetContractError(
                        f"{date}: existing partition schema differs from lineage"
                    )
                skipped.append(date)
                continue
        day_rows = index.filter(pl.col("Date") == date)
        result = materialize_event_day(
            date,
            day_rows,
            roots=roots,
            options=spec.merge_options,
            txf_loader=loader,
        )
        signature = _schema_signature(result.frame)
        if expected_schema is None:
            expected_schema = signature
            manifest["schema"] = signature
        elif spec.merge_options.schema_policy == "strict" and signature != expected_schema:
            raise DatasetContractError(
                f"{date}: schema drift under strict policy; bump dataset_version"
            )
        _atomic_write_parquet(result.frame, output_path)
        partitions[date] = {
            "status": "materialized",
            "index": index_identity,
            "sources": source_identities,
            "row_count": len(result.frame),
            "schema": signature,
        }
        _atomic_write_json(manifest, spec.manifest_path)
        saved.append(date)
        audits.append(result.audit)

    return DatasetBuildReport(
        event_id=spec.event_id,
        dataset_version=spec.dataset_version,
        saved_dates=tuple(saved),
        skipped_dates=tuple(skipped),
        day_audits=tuple(audits),
    )


def read_event_dataset(
    output_dir: Path | str,
    *,
    date: str | None = None,
    dates: list[str] | tuple[str, ...] | None = None,
    columns: list[str] | tuple[str, ...] | None = None,
    strict_files: bool = True,
    schema_policy: SchemaPolicy = "strict",
) -> pl.DataFrame:
    """Read one or more versioned per-day event datasets."""

    if date is not None and dates is not None:
        raise ValueError("pass either date or dates, not both")
    root = Path(output_dir)
    manifest: dict[str, object] | None = None
    manifest_path = root / "_dataset_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetContractError(
                f"cannot read dataset manifest: {manifest_path}"
            ) from exc
        stored_policy = manifest.get("schema_policy")
        if stored_policy is not None and stored_policy != schema_policy:
            raise DatasetContractError(
                f"reader schema_policy={schema_policy!r} disagrees with "
                f"dataset manifest {stored_policy!r}"
            )
    requested = [date] if date is not None else list(dates) if dates is not None else None
    paths = (
        [root / f"{value}.parquet" for value in requested]
        if requested is not None
        else sorted(root.glob("*.parquet"))
    )
    missing = [path for path in paths if not path.exists()]
    if missing and strict_files:
        raise FileNotFoundError(f"dataset files not found: {missing}")
    paths = [path for path in paths if path.exists()]
    if not paths:
        return pl.DataFrame()
    parts = [pl.read_parquet(path, columns=columns) for path in paths]
    if manifest is not None:
        partition_lineage = manifest.get("partitions", {})
        if isinstance(partition_lineage, dict):
            for path, part in zip(paths, parts, strict=True):
                lineage = partition_lineage.get(path.stem)
                if not isinstance(lineage, dict):
                    raise DatasetContractError(
                        f"{path.name}: missing partition lineage"
                    )
                if lineage.get("row_count") != len(part):
                    raise DatasetContractError(
                        f"{path.name}: row count differs from partition lineage"
                    )
                if columns is None and lineage.get("schema") != _schema_signature(part):
                    raise DatasetContractError(
                        f"{path.name}: schema differs from partition lineage"
                    )
    if schema_policy == "strict":
        signature = _schema_signature(parts[0])
        drift = [path.name for path, part in zip(paths[1:], parts[1:], strict=True) if _schema_signature(part) != signature]
        if drift:
            raise DatasetContractError(f"schema drift while reading: {drift}")
        return pl.concat(parts, how="vertical")
    if schema_policy == "union":
        return pl.concat(parts, how="diagonal")
    raise ValueError(f"unknown schema_policy={schema_policy!r}")


__all__ = [
    "ALL_EVENTS_UNIVERSE",
    "DataRoots",
    "DatasetBuildReport",
    "DatasetContractError",
    "DayMergeAudit",
    "DayMergeResult",
    "EventDatasetSpec",
    "MergeOptions",
    "MissingDatasetSourceError",
    "TICK_KEY_COLS",
    "UniverseSpec",
    "build_event_dataset",
    "materialize_event_day",
    "merge_event_day",
    "read_event_dataset",
]
