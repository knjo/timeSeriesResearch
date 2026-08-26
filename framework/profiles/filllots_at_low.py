"""FillLots_atLow event-family selectors for bias discovery.

The family deliberately excludes negFill's Spread and BidPreMove conditions.
Every candidate is computed from the complete, chronologically ordered raw
tick stream before any study-universe filter is applied; eligibility is
handled later by the shared condition-study pool.

``FillLots_atLow`` is a cumulative signed-flow state within each ``RecordLow``
epoch.  The registry therefore distinguishes repeated state updates from
epoch-aware entry events instead of treating both as the same event.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl

from ..events import DAY_KEY_COLS, EventDaySources, EventIndexValidationError


FILLLOTS_AT_LOW_FAMILY_ID = "filllots_at_low"
FILLLOTS_AT_LOW_FAMILY_VERSION = "3"
FILLLOTS_AT_LOW_STATE_FAMILY_ID = "filllots_at_low_state"
FILLLOTS_AT_LOW_STATE_FAMILY_VERSION = "1"
STATE_CANDIDATE_IDS = ("negative_state", "positive_state")
STATE_CANDIDATE_DESCRIPTIONS = {
    "negative_state": "unmodified cumulative state F < 0",
    "positive_state": "unmodified cumulative state F > 0",
}

if tuple(STATE_CANDIDATE_DESCRIPTIONS) != STATE_CANDIDATE_IDS:  # pragma: no cover
    raise RuntimeError("FillLotsAtLow state descriptions drifted")

CANDIDATE_IDS = (
    "negative_state_update",
    "negative_sell_flow_update",
    "negative_buy_flow_update",
    "negative_epoch_start",
    "negative_same_epoch_cross",
    "negative_epoch_entry",
    "depth_lt_neg1_state_update",
    "depth_lt_neg2_state_update",
    "depth_lt_neg4_state_update",
    "depth_lt_neg8_state_update",
    "depth_lt_neg1_epoch_entry",
    "depth_lt_neg2_epoch_entry",
    "depth_lt_neg4_epoch_entry",
    "depth_lt_neg8_epoch_entry",
    "positive_state_update",
    "positive_epoch_start",
    "positive_same_epoch_cross",
    "positive_epoch_entry",
    "depth_gt_pos1_state_update",
    "depth_gt_pos2_state_update",
    "depth_gt_pos4_state_update",
    "depth_gt_pos8_state_update",
    "depth_gt_pos1_epoch_entry",
    "depth_gt_pos2_epoch_entry",
    "depth_gt_pos4_epoch_entry",
    "depth_gt_pos8_epoch_entry",
)

CANDIDATE_DESCRIPTIONS = {
    "negative_state_update": "F < 0 and F changes on a formal trade",
    "negative_sell_flow_update": (
        "negative_state_update and signed FillLots is negative"
    ),
    "negative_buy_flow_update": (
        "negative_state_update and signed FillLots is positive"
    ),
    "negative_epoch_start": (
        "a RecordLow epoch starts on a formal trade with F < 0"
    ),
    "negative_same_epoch_cross": (
        "within one RecordLow epoch, F crosses from nonnegative to negative"
    ),
    "negative_epoch_entry": (
        "negative_epoch_start or any negative_same_epoch_cross; may repeat"
    ),
    "depth_lt_neg1_state_update": "ratio < -1 and F changes on a formal trade",
    "depth_lt_neg2_state_update": "ratio < -2 and F changes on a formal trade",
    "depth_lt_neg4_state_update": "ratio < -4 and F changes on a formal trade",
    "depth_lt_neg8_state_update": "ratio < -8 and F changes on a formal trade",
    "depth_lt_neg1_epoch_entry": (
        "epoch starts below -1, or ratio crosses/re-enters below -1"
    ),
    "depth_lt_neg2_epoch_entry": (
        "epoch starts below -2, or ratio crosses/re-enters below -2"
    ),
    "depth_lt_neg4_epoch_entry": (
        "epoch starts below -4, or ratio crosses/re-enters below -4"
    ),
    "depth_lt_neg8_epoch_entry": (
        "epoch starts below -8, or ratio crosses/re-enters below -8"
    ),
    "positive_state_update": "F > 0 and F changes on a formal trade",
    "positive_epoch_start": (
        "a RecordLow epoch starts on a formal trade with F > 0"
    ),
    "positive_same_epoch_cross": (
        "within one RecordLow epoch, F crosses from nonpositive to positive"
    ),
    "positive_epoch_entry": (
        "positive_epoch_start or any positive_same_epoch_cross; may repeat"
    ),
    "depth_gt_pos1_state_update": "ratio > 1 and F changes on a formal trade",
    "depth_gt_pos2_state_update": "ratio > 2 and F changes on a formal trade",
    "depth_gt_pos4_state_update": "ratio > 4 and F changes on a formal trade",
    "depth_gt_pos8_state_update": "ratio > 8 and F changes on a formal trade",
    "depth_gt_pos1_epoch_entry": (
        "epoch starts above 1, or ratio crosses/re-enters above 1"
    ),
    "depth_gt_pos2_epoch_entry": (
        "epoch starts above 2, or ratio crosses/re-enters above 2"
    ),
    "depth_gt_pos4_epoch_entry": (
        "epoch starts above 4, or ratio crosses/re-enters above 4"
    ),
    "depth_gt_pos8_epoch_entry": (
        "epoch starts above 8, or ratio crosses/re-enters above 8"
    ),
}

if tuple(CANDIDATE_DESCRIPTIONS) != CANDIDATE_IDS:  # pragma: no cover
    raise RuntimeError("FillLotsAtLow candidate descriptions drifted")


def _require_columns(
    frame: pl.LazyFrame,
    columns: tuple[str, ...],
    *,
    source: str,
) -> None:
    available = set(frame.collect_schema().names())
    missing = [column for column in columns if column not in available]
    if missing:
        raise EventIndexValidationError(
            f"{source}: missing FillLots_atLow candidate columns {missing}"
        )


def build_filllots_at_low_candidate_frame(
    day: EventDaySources,
) -> pl.DataFrame:
    """Recompute causal state on the complete raw stream and attach prior depth."""

    tick = day.scan_tick_data()
    pre_market = day.scan_pre_market()
    _require_columns(
        tick,
        (
            *DAY_KEY_COLS,
            "FillLots",
            "InOut",
            "TrialMatch",
            "marketOpen",
            "RecordLow",
            "FillPrice",
        ),
        source=str(day.path("tick_data")),
    )
    _require_columns(
        pre_market,
        ("QuoteCode", "avg_bidLots1", "avg_askLots1"),
        source=str(day.path("pre_market")),
    )

    depth = pre_market.select(
        "QuoteCode",
        "avg_bidLots1",
        "avg_askLots1",
    )
    # Keep this formula identical to
    # src.features.definitions.price_dynamics.calc_accFillLots.  In particular,
    # trial-match rows contribute zero but remain in the chronological stream so
    # the first later formal trade has the correct prior state and epoch.
    signed_flow = (
        pl.col("FillLots")
        * pl.col("InOut")
        * (pl.col("TrialMatch") == 0)
    )
    denominator = pl.col("avg_bidLots1") + pl.col("avg_askLots1")
    frame = (
        tick.select(
            *DAY_KEY_COLS,
            "FillLots",
            "InOut",
            "TrialMatch",
            "marketOpen",
            "RecordLow",
            "FillPrice",
        )
        .sort(list(DAY_KEY_COLS))
        .with_columns(
            signed_flow.cum_sum()
            .over(["QuoteCode", "RecordLow"])
            .alias("FillLots_atLow")
        )
        .with_columns(
            pl.col("FillLots_atLow")
            .shift(1)
            .over("QuoteCode")
            .alias("_previous_filllots_at_low"),
            pl.col("RecordLow")
            .shift(1)
            .over("QuoteCode")
            .alias("_previous_record_low"),
            pl.col("marketOpen")
            .shift(1)
            .over("QuoteCode")
            .alias("_previous_market_open"),
        )
        .join(
            depth,
            on="QuoteCode",
            how="left",
            validate="m:1",
        )
        .with_columns(
            pl.when(denominator.is_finite() & (denominator > 0.0))
            .then(pl.col("FillLots_atLow") / denominator)
            .alias("filllots_at_low_depth_ratio"),
            pl.when(denominator.is_finite() & (denominator > 0.0))
            .then(pl.col("_previous_filllots_at_low") / denominator)
            .alias("_previous_filllots_at_low_depth_ratio"),
        )
        .with_columns(
            (
                pl.col("_previous_filllots_at_low").is_not_null()
                & (
                    pl.col("FillLots_atLow")
                    != pl.col("_previous_filllots_at_low")
                )
            ).alias("_state_changed"),
            (
                pl.col("marketOpen")
                & (
                    pl.col("_previous_market_open").is_null()
                    | ~pl.col("_previous_market_open").fill_null(False)
                    | (pl.col("RecordLow") < pl.col("_previous_record_low"))
                )
            ).alias("_epoch_started"),
            (
                pl.col("marketOpen")
                & pl.col("_previous_market_open").fill_null(False)
                & (pl.col("RecordLow") == pl.col("_previous_record_low"))
            ).alias("_same_epoch"),
            (
                pl.col("marketOpen")
                & (pl.col("TrialMatch") == 0)
                & (pl.col("FillLots") > 0)
                & pl.col("InOut").is_in([-1, 1])
            ).alias("_formal_trade"),
            (pl.col("FillLots") * pl.col("InOut")).alias("signed_filllots"),
            (
                pl.col("FillPrice").is_not_null()
                & (pl.col("FillPrice") > 0)
                & (pl.col("FillPrice") == pl.col("RecordLow"))
            ).alias("at_record_low"),
            (denominator.is_finite() & (denominator > 0.0)).alias(
                "depth_valid"
            ),
        )
        .sort(list(DAY_KEY_COLS))
        .collect()
    )
    duplicate_keys = (
        frame.select(DAY_KEY_COLS)
        .group_by(DAY_KEY_COLS)
        .len()
        .filter(pl.col("len") > 1)
    )
    if not duplicate_keys.is_empty():
        raise EventIndexValidationError(
            f"{day.path('tick_data')}: duplicate event keys; "
            f"sample={duplicate_keys.head(3).to_dicts()}"
        )
    return frame


def select_filllots_at_low_candidates(
    day: EventDaySources,
    *,
    candidate_ids: Sequence[str] | None = None,
) -> Mapping[str, pl.DataFrame]:
    """Return the pre-registered FillLots_atLow candidate key sets."""

    frame = build_filllots_at_low_candidate_frame(day)
    current = pl.col("FillLots_atLow")
    previous = pl.col("_previous_filllots_at_low")
    ratio = pl.col("filllots_at_low_depth_ratio")
    previous_ratio = pl.col("_previous_filllots_at_low_depth_ratio")
    changed = pl.col("_state_changed")
    epoch_started = pl.col("_epoch_started")
    same_epoch = pl.col("_same_epoch")
    formal_trade = pl.col("_formal_trade")
    sell_flow = pl.col("signed_filllots") < 0
    buy_flow = pl.col("signed_filllots") > 0

    negative_update = (current < 0) & changed & formal_trade
    positive_update = (current > 0) & changed & formal_trade
    negative_epoch_start = (current < 0) & epoch_started & formal_trade
    positive_epoch_start = (current > 0) & epoch_started & formal_trade
    negative_same_epoch_cross = (
        (current < 0) & (previous >= 0) & same_epoch & formal_trade
    )
    positive_same_epoch_cross = (
        (current > 0) & (previous <= 0) & same_epoch & formal_trade
    )

    def negative_depth_update(threshold: float) -> pl.Expr:
        return (ratio < threshold) & changed & formal_trade

    def positive_depth_update(threshold: float) -> pl.Expr:
        return (ratio > threshold) & changed & formal_trade

    def negative_depth_entry(threshold: float) -> pl.Expr:
        return (
            (ratio < threshold)
            & formal_trade
            & (
                epoch_started
                | (same_epoch & (previous_ratio >= threshold))
            )
        )

    def positive_depth_entry(threshold: float) -> pl.Expr:
        return (
            (ratio > threshold)
            & formal_trade
            & (
                epoch_started
                | (same_epoch & (previous_ratio <= threshold))
            )
        )

    predicates = {
        "negative_state_update": negative_update,
        "negative_sell_flow_update": negative_update & sell_flow,
        "negative_buy_flow_update": negative_update & buy_flow,
        "negative_epoch_start": negative_epoch_start,
        "negative_same_epoch_cross": negative_same_epoch_cross,
        "negative_epoch_entry": (
            negative_epoch_start | negative_same_epoch_cross
        ),
        "depth_lt_neg1_state_update": negative_depth_update(-1.0),
        "depth_lt_neg2_state_update": negative_depth_update(-2.0),
        "depth_lt_neg4_state_update": negative_depth_update(-4.0),
        "depth_lt_neg8_state_update": negative_depth_update(-8.0),
        "depth_lt_neg1_epoch_entry": negative_depth_entry(-1.0),
        "depth_lt_neg2_epoch_entry": negative_depth_entry(-2.0),
        "depth_lt_neg4_epoch_entry": negative_depth_entry(-4.0),
        "depth_lt_neg8_epoch_entry": negative_depth_entry(-8.0),
        "positive_state_update": positive_update,
        "positive_epoch_start": positive_epoch_start,
        "positive_same_epoch_cross": positive_same_epoch_cross,
        "positive_epoch_entry": positive_epoch_start | positive_same_epoch_cross,
        "depth_gt_pos1_state_update": positive_depth_update(1.0),
        "depth_gt_pos2_state_update": positive_depth_update(2.0),
        "depth_gt_pos4_state_update": positive_depth_update(4.0),
        "depth_gt_pos8_state_update": positive_depth_update(8.0),
        "depth_gt_pos1_epoch_entry": positive_depth_entry(1.0),
        "depth_gt_pos2_epoch_entry": positive_depth_entry(2.0),
        "depth_gt_pos4_epoch_entry": positive_depth_entry(4.0),
        "depth_gt_pos8_epoch_entry": positive_depth_entry(8.0),
    }
    if tuple(predicates) != CANDIDATE_IDS:  # pragma: no cover - import invariant
        raise RuntimeError("FillLotsAtLow candidate manifest drifted")
    requested = CANDIDATE_IDS if candidate_ids is None else tuple(candidate_ids)
    unknown = sorted(set(requested) - set(CANDIDATE_IDS))
    if unknown:
        raise ValueError(f"unknown FillLotsAtLow candidates: {unknown}")
    if len(requested) != len(set(requested)):
        raise ValueError("candidate_ids cannot contain duplicates")
    return {
        candidate_id: frame.filter(predicate).select(DAY_KEY_COLS)
        for candidate_id in requested
        for predicate in (predicates[candidate_id],)
    }


def select_filllots_at_low_state_candidates(
    day: EventDaySources,
    *,
    candidate_ids: Sequence[str] | None = None,
) -> Mapping[str, pl.DataFrame]:
    """Select the unmodified signed state partitions used for calibration.

    Unlike the exploratory state-update/epoch-entry registry, these conditions
    do not require a formal trade, a state change, an onset, or depth scaling.
    They answer the first-stage question directly: can the labeled eligible
    pool recover the known bias of ``FillLots_atLow < 0`` (or its positive
    counterpart)?  Execution debouncing belongs to a later study stage.
    """

    requested = (
        STATE_CANDIDATE_IDS if candidate_ids is None else tuple(candidate_ids)
    )
    unknown = sorted(set(requested) - set(STATE_CANDIDATE_IDS))
    if unknown:
        raise ValueError(f"unknown FillLotsAtLow state candidates: {unknown}")
    if len(requested) != len(set(requested)):
        raise ValueError("candidate_ids cannot contain duplicates")

    frame = build_filllots_at_low_candidate_frame(day)
    predicates = {
        "negative_state": pl.col("FillLots_atLow") < 0,
        "positive_state": pl.col("FillLots_atLow") > 0,
    }
    return {
        candidate_id: frame.filter(predicates[candidate_id]).select(DAY_KEY_COLS)
        for candidate_id in requested
    }


__all__ = [
    "CANDIDATE_DESCRIPTIONS",
    "CANDIDATE_IDS",
    "FILLLOTS_AT_LOW_FAMILY_ID",
    "FILLLOTS_AT_LOW_FAMILY_VERSION",
    "FILLLOTS_AT_LOW_STATE_FAMILY_ID",
    "FILLLOTS_AT_LOW_STATE_FAMILY_VERSION",
    "STATE_CANDIDATE_DESCRIPTIONS",
    "STATE_CANDIDATE_IDS",
    "build_filllots_at_low_candidate_frame",
    "select_filllots_at_low_candidates",
    "select_filllots_at_low_state_candidates",
]
