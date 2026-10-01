"""Apply the frozen sparse-to-dense promotion gate without outcome edits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl

from src.research.intradaySignal.framework.profiles.filllots_at_high import (
    CANDIDATE_IDS,
)


STUDY_SPEC_PATH = Path(__file__).with_name("study_spec.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_promotions(
    period_summary: pl.DataFrame,
    study_spec: dict[str, object],
) -> tuple[pl.DataFrame, list[dict[str, object]]]:
    sparse = study_spec["sparse_screen"]
    gate = sparse["promotion_per_label"]
    cap = int(sparse["promotion_cap"])
    if cap < 1:
        raise ValueError("promotion_cap must be positive")

    full = period_summary.filter(pl.col("period") == "full")
    expected_rows = 2 * len(CANDIDATE_IDS)
    if len(full) != expected_rows:
        raise ValueError(
            f"full-period family has {len(full)} rows, expected {expected_rows}"
        )
    expected_keys = {
        (direction, candidate_id)
        for direction in ("short", "long")
        for candidate_id in CANDIDATE_IDS
    }
    actual_keys = set(full.select("direction", "candidate_id").iter_rows())
    if actual_keys != expected_keys:
        raise ValueError("period summary does not contain the frozen registry")

    yearly = (
        period_summary.filter(pl.col("period").is_in(["2025", "2026"]))
        .select(
            "direction",
            "candidate_id",
            "period",
            "equal_day_delta_z_mean",
        )
        .pivot(
            on="period",
            index=["direction", "candidate_id"],
            values="equal_day_delta_z_mean",
        )
        .rename({"2025": "effect_2025", "2026": "effect_2026"})
    )
    gated = (
        full.join(
            yearly,
            on=["direction", "candidate_id"],
            how="left",
            validate="1:1",
        )
        .with_columns(
            (
                (pl.col("session_coverage") >= float(gate["active_day_ratio_min"]))
                & (pl.col("bh_q_full") <= float(gate["bh_q_full_max"]))
                & (pl.col("equal_day_delta_z_mean") > 0.0)
                & (
                    pl.col("events_per_active_day")
                    >= float(gate["events_per_active_day_min"])
                )
                & (
                    pl.col("positive_month_ratio")
                    >= float(gate["positive_month_ratio_min"])
                )
                & (pl.col("effect_2025") > 0.0)
                & (pl.col("effect_2026") > 0.0)
            ).alias("passes_promotion_gate")
        )
        .select(
            "direction",
            "candidate_id",
            "active_days",
            "session_coverage",
            "events_per_active_day",
            "equal_day_delta_z_mean",
            "hac_p_two_sided",
            "bh_q_full",
            "joint_holm_p_full",
            "positive_month_ratio",
            "effect_2025",
            "effect_2026",
            "passes_promotion_gate",
        )
        .sort("direction", "bh_q_full", "equal_day_delta_z_mean", descending=[False, False, True])
    )

    selected: list[dict[str, object]] = []
    for direction in ("short", "long"):
        direction_rows = gated.filter(pl.col("direction") == direction)
        passing = direction_rows.filter(pl.col("passes_promotion_gate"))
        if passing.is_empty():
            fallback = direction_rows.sort("hac_p_two_sided").row(
                0,
                named=True,
            )
            selected.append(
                {
                    "direction": direction,
                    "candidate_id": fallback["candidate_id"],
                    "decision": "fallback_falsification",
                }
            )
            continue
        for row in passing.head(cap).iter_rows(named=True):
            selected.append(
                {
                    "direction": direction,
                    "candidate_id": row["candidate_id"],
                    "decision": "promoted",
                }
            )
    return gated, selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--period-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    spec = json.loads(STUDY_SPEC_PATH.read_text(encoding="utf-8"))
    period_summary = pl.read_parquet(args.period_summary)
    gate_table, selected = select_promotions(period_summary, spec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "contract": "filllots_at_high_sparse_promotion/v1",
        "study_spec": {
            "path": str(STUDY_SPEC_PATH.resolve()),
            "sha256": _sha256(STUDY_SPEC_PATH),
        },
        "period_summary": {
            "path": str(args.period_summary.resolve()),
            "sha256": _sha256(args.period_summary),
        },
        "selected": selected,
        "gate_rows": gate_table.to_dicts(),
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(selected, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
