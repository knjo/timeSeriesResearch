"""Audit sampling freshness and upstream/rebuilt high-state parity."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from src.research.intradaySignal.framework.events import EventDaySources
from src.research.intradaySignal.framework.profiles.filllots_at_high import (
    build_filllots_at_high_candidate_frame,
)

from study import ARTIFACT_DIR, EVENT_INDEX_PATH, data_roots


AUDIT_DATES = (
    "20250211",
    "20250402",
    "20250526",
    "20250715",
    "20250902",
    "20251027",
    "20251215",
    "20260204",
    "20260408",
    "20260528",
    "20260720",
    "20260824",
)
OUTPUT_DIR = ARTIFACT_DIR / "sampling_audit_v1"


def main() -> int:
    if not EVENT_INDEX_PATH.exists():
        raise FileNotFoundError(EVENT_INDEX_PATH)
    index = pl.read_parquet(EVENT_INDEX_PATH)
    roots = data_roots()
    rows: list[dict[str, object]] = []
    for date in AUDIT_DATES:
        keys = index.filter(pl.col("Date") == date).drop("Date")
        day = EventDaySources(date, roots)
        rebuilt = build_filllots_at_high_candidate_frame(day).join(
            keys, on=["QuoteCode", "ChannelSeq"], how="semi"
        )
        source = (
            day.scan_tick_feature()
            .select(
                "QuoteCode",
                "ChannelSeq",
                pl.col("FillLots_atHigh").alias("source_FillLots_atHigh"),
            )
            .join(keys.lazy(), on=["QuoteCode", "ChannelSeq"], how="semi")
            .collect()
        )
        joined = rebuilt.select(
            "QuoteCode",
            "ChannelSeq",
            "FillLots_atHigh",
            "_fresh_event",
            "_md_price_change",
            "_large_fill",
        ).join(source, on=["QuoteCode", "ChannelSeq"], how="inner", validate="1:1")
        if len(joined) != len(keys):
            raise RuntimeError(f"{date}: audit join lost event keys")
        state_diff = (
            joined["FillLots_atHigh"] - joined["source_FillLots_atHigh"]
        )
        row = {
            "Date": date,
            "events": len(joined),
            "fresh_events": int(joined["_fresh_event"].sum()),
            "md_price_change_events": int(joined["_md_price_change"].sum()),
            "large_fill_events": int(joined["_large_fill"].sum()),
            "state_mismatch_events": int((state_diff != 0).sum()),
            "state_max_abs_difference": int(state_diff.abs().max() or 0),
        }
        if row["fresh_events"] != row["events"]:
            raise RuntimeError(f"{date}: selected event without a fresh trigger")
        if row["state_mismatch_events"] != 0:
            raise RuntimeError(f"{date}: upstream/rebuilt FillLots_atHigh mismatch")
        rows.append(row)
        print(row, flush=True)

    output = pl.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output.write_parquet(OUTPUT_DIR / "daily_audit.parquet", compression="zstd")
    payload = {
        "contract": "filllots_at_high_sampling_audit/v1",
        "audited_dates": list(AUDIT_DATES),
        "events": int(output["events"].sum()),
        "fresh_events": int(output["fresh_events"].sum()),
        "md_price_change_events": int(output["md_price_change_events"].sum()),
        "large_fill_events": int(output["large_fill_events"].sum()),
        "state_mismatch_events": int(output["state_mismatch_events"].sum()),
        "result": "pass",
    }
    (OUTPUT_DIR / "audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
