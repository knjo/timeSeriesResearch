"""negfill — the deployed negFill short, as one :class:`SignalConfig`.

Every number reproduces the deployed contract in
``src/strategy/negFill/modelUpdate.py``; only the *order* differs from the
legacy pipeline: the tick-level filters (layer 2) and the learned day selector
(layer 3) now sit before the model, so the model trains and sets thresholds on
the rows it will actually trade.  Layer 1 (envelope, condition, index) is
unchanged, so the store rows and index keys stay identical to the deployed
pre-gate set.

    layer 1  envelope   |Spread.diff|>0 & |BidPreMove|>0 & FillLots_atLow<0   (profiles.negfill)
             condition  ToRef_bid1 in (-1.5%, 5%) & amount rank <= 150        (fixed; grid empty; deployed Bid1 convention)
    layer 2  points     09:00:30 < t < 12:30, formal match, Bid1 > 0,
                        OTC | mdRate > .2, SpreadPairElapsed > .1, rank <= 100,
                        and the seven deployed day-level rules (R1, market cap, brench, price 3000, maker spread)
    layer 3  day        none (the learned selector did not beat the rules; see signals/_short_common.py)
    layer 4  model      14-feature Ridge A(60d)/M(20d), 16-cell prior-3-day thresholds, qA 0.05
"""

from __future__ import annotations

from ..framework.profiles.negfill import select_negfill_events
from ..framework.profiles.tw_equity import NEXTDAY_ALLOW_DAY_TRADE_UNIVERSE_V1
from ..framework.registry import CellThresholdSpec, GateSpec, RollingSelection, SignalConfig
from ._short_common import (
    PRICE_AMOUNT_GATE_PARAMS,
    TO_REF_BID1,
    short_labeled_pl,
    short_model_spec,
    short_point_filter_spec,
    short_price_amount_gate,
)


ST_NAME = "negfill"
VERSION = "1"


# layer 1 — condition (the notebook's PriceCondition / AmountCondition); add e.g.
# grid={"rank_max": (100, 150, 200)} to roll it daily.
CONDITION = GateSpec(
    gate_id="negfill_price_amount",
    version="3",   # v3: deployed Bid1 ToRef convention, materialized as ToRef_bid1 (source ToRef untouched)
    params={**PRICE_AMOUNT_GATE_PARAMS, "to_ref_source": "bid1"},
    apply=short_price_amount_gate,
    grid={},
    stage="condition",
)

# layer 2 — the notebook's post-threshold tick filters plus the seven deployed day-level rules,
# all applied before the model.  2026-09-30: the learned day selector (risk targets, trees,
# quantile cuts) beat "no filter" but not the rules on a risk-adjusted basis (2026 Sharpe 2.8 vs
# 3.4, MDD 692 vs 408), so negfill keeps the rules as a fixed removal and has no day selector.
POINT_FILTER = short_point_filter_spec("negfill_points", "2", day_rules=True, md_l1rate_min=0.2, elapsed_min=0.1, rank_max=100)

# layer 4 — the A threshold quantile is 0.05 in the four-layer order: the legacy 0.30 was cut on
# every index row, of which the filtered rows sat mostly above it; re-cutting 30% of the already
# filtered rows removed good trades (2026: 4,136 -> 4,799萬 at equal Sharpe).
MODEL = short_model_spec(
    version="3",
    to_ref_source="bid1",
    threshold=CellThresholdSpec(ref_col=TO_REF_BID1, q_timing=0.05),
    description="Ridge A(60d) + M(20d) on the filtered points; Bid1 ToRef/ToOpen; 16-cell thresholds with qA 0.05.",
)


NEGFILL = SignalConfig(
    st_name=ST_NAME,
    version=VERSION,
    stage="deployed",
    description="Deployed negFill short: spread move + bid pre-move + FillLots_atLow < 0.",
    envelope=select_negfill_events,
    envelope_required_sources=("tick_feature",),
    universe=NEXTDAY_ALLOW_DAY_TRADE_UNIVERSE_V1,
    gate=CONDITION,
    selection=RollingSelection(window_days=60, objective="stockday_mean_label",
                               label_col="label_residual_bp", min_rows=300, min_stock_days=30),
    selection_pool=short_labeled_pl,
    point_filter=POINT_FILTER,
    day_selector=None,
    model=MODEL,
)


__all__ = ["CONDITION", "MODEL", "NEGFILL", "POINT_FILTER"]
