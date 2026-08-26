"""Directional event-label construction for time-series research.

The core contract keeps trade direction, entry execution, exit policy, market
hedge, and volatility normalization independent.  It defines no strategy
defaults; pinned research choices live under :mod:`framework.profiles`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
import hashlib
import json
import math
from typing import TypeVar

import polars as pl


FrameT = TypeVar("FrameT", pl.DataFrame, pl.LazyFrame)


class Direction(IntEnum):
    """Signed position direction used by all PnL formulas."""

    SHORT = -1
    LONG = 1


class LiquidityRole(StrEnum):
    """Execution assumption attached to an entry price."""

    MAKER = "maker"
    TAKER = "taker"
    MARK = "mark"


class TriggerOp(StrEnum):
    """Comparison used by a barrier exit."""

    GT = "gt"
    GE = "ge"
    LT = "lt"
    LE = "le"


class VolUnit(StrEnum):
    """Unit of the supplied volatility column."""

    RETURN = "return"
    BP = "bp"


@dataclass(frozen=True)
class EntrySpec:
    """Entry execution price and its liquidity assumption."""

    price_col: str
    liquidity: LiquidityRole

    def __post_init__(self) -> None:
        if not self.price_col:
            raise ValueError("price_col cannot be empty")
        object.__setattr__(self, "liquidity", LiquidityRole(self.liquidity))


@dataclass(frozen=True)
class ExitSpec:
    """Close-price exit, optionally replaced by a single price barrier.

    ``adverse_slippage_ticks`` is direction aware.  It increases a short
    buy-to-cover price and decreases a long sell price.
    """

    fallback_price_col: str
    trigger_price_col: str | None = None
    trigger_op: TriggerOp = TriggerOp.GT
    barrier_reference_col: str | None = None
    barrier_multiplier: float | None = None
    tick_size_col: str | None = None
    adverse_slippage_ticks: float = 0.0
    horizon: str = "13:30"

    def __post_init__(self) -> None:
        if not self.fallback_price_col:
            raise ValueError("fallback_price_col cannot be empty")
        object.__setattr__(self, "trigger_op", TriggerOp(self.trigger_op))
        barrier_fields = (
            self.trigger_price_col,
            self.barrier_reference_col,
            self.barrier_multiplier,
        )
        supplied = sum(value is not None for value in barrier_fields)
        if supplied not in (0, len(barrier_fields)):
            raise ValueError(
                "trigger_price_col, barrier_reference_col, and "
                "barrier_multiplier must be supplied together"
            )
        if self.adverse_slippage_ticks < 0.0:
            raise ValueError("adverse_slippage_ticks cannot be negative")
        if not math.isfinite(self.adverse_slippage_ticks):
            raise ValueError("adverse_slippage_ticks must be finite")
        if self.adverse_slippage_ticks > 0.0 and self.tick_size_col is None:
            raise ValueError(
                "tick_size_col is required when adverse_slippage_ticks is positive"
            )
        if self.barrier_multiplier is not None and (
            not math.isfinite(self.barrier_multiplier)
            or self.barrier_multiplier <= 0.0
        ):
            raise ValueError("barrier_multiplier must be positive and finite")
        if not self.horizon:
            raise ValueError("exit horizon cannot be empty")

    @property
    def has_barrier(self) -> bool:
        return self.trigger_price_col is not None


@dataclass(frozen=True)
class HedgeSpec:
    """Prior beta and future factor return used to remove market exposure."""

    beta_col: str = "txf_beta_60d"
    factor_return_bp_col: str = "txf_to_1330_bp"
    hedge_ratio: float = 1.0
    horizon: str = "13:30"

    def __post_init__(self) -> None:
        if not math.isfinite(self.hedge_ratio) or self.hedge_ratio < 0.0:
            raise ValueError("hedge_ratio must be non-negative and finite")
        if not self.horizon:
            raise ValueError("hedge horizon cannot be empty")


@dataclass(frozen=True)
class VolSpec:
    """Volatility scale used to standardize residual PnL."""

    vol_col: str = "txf_residual_vol_to_1330"
    unit: VolUnit = VolUnit.RETURN
    min_vol_bp: float = 1e-12

    def __post_init__(self) -> None:
        object.__setattr__(self, "unit", VolUnit(self.unit))
        if not math.isfinite(self.min_vol_bp) or self.min_vol_bp < 0.0:
            raise ValueError("min_vol_bp must be non-negative and finite")


@dataclass(frozen=True)
class DirectionalLabelSpec:
    """Complete, versioned economic contract for one event label."""

    profile_id: str
    direction: Direction
    entry: EntrySpec
    exit: ExitSpec
    hedge: HedgeSpec | None = None
    normalization: VolSpec | None = None
    allow_exit_hedge_horizon_mismatch: bool = False

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("profile_id cannot be empty")
        object.__setattr__(self, "direction", Direction(self.direction))
        if (
            self.hedge is not None
            and self.exit.horizon != self.hedge.horizon
            and not self.allow_exit_hedge_horizon_mismatch
        ):
            raise ValueError(
                "exit and hedge horizons differ; set "
                "allow_exit_hedge_horizon_mismatch=True to acknowledge it"
            )

    def canonical_dict(self) -> dict[str, object]:
        """Return every output-affecting label choice in stable form."""

        hedge = None
        if self.hedge is not None:
            hedge = {
                "beta_col": self.hedge.beta_col,
                "factor_return_bp_col": self.hedge.factor_return_bp_col,
                "hedge_ratio": float(self.hedge.hedge_ratio),
                "horizon": self.hedge.horizon,
            }
        normalization = None
        if self.normalization is not None:
            normalization = {
                "vol_col": self.normalization.vol_col,
                "unit": self.normalization.unit.value,
                "min_vol_bp": float(self.normalization.min_vol_bp),
            }
        return {
            "profile_id": self.profile_id,
            "direction": int(self.direction),
            "entry": {
                "price_col": self.entry.price_col,
                "liquidity": self.entry.liquidity.value,
            },
            "exit": {
                "fallback_price_col": self.exit.fallback_price_col,
                "trigger_price_col": self.exit.trigger_price_col,
                "trigger_op": self.exit.trigger_op.value,
                "barrier_reference_col": self.exit.barrier_reference_col,
                "barrier_multiplier": self.exit.barrier_multiplier,
                "tick_size_col": self.exit.tick_size_col,
                "adverse_slippage_ticks": float(
                    self.exit.adverse_slippage_ticks
                ),
                "horizon": self.exit.horizon,
            },
            "hedge": hedge,
            "normalization": normalization,
            "allow_exit_hedge_horizon_mismatch": (
                self.allow_exit_hedge_horizon_mismatch
            ),
        }

    def stable_hash(self) -> str:
        payload = json.dumps(
            self.canonical_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def fingerprint(self) -> str:
        return self.stable_hash()


@dataclass(frozen=True)
class LabelColumns:
    """Stable output names emitted by :func:`add_directional_labels`."""

    profile_id: str = "label_profile_id"
    direction: str = "label_direction"
    entry_price: str = "label_entry_price"
    exit_price: str = "label_exit_price"
    price_move_bp: str = "label_price_move_bp"
    raw_bp: str = "label_raw_bp"
    market_component_bp: str = "label_market_component_bp"
    hedge_pnl_bp: str = "label_hedge_pnl_bp"
    residual_bp: str = "label_residual_bp"
    vol_bp: str = "label_vol_bp"
    z: str = "label_z"
    valid: str = "label_valid"


DEFAULT_LABEL_COLUMNS = LabelColumns()


@dataclass(frozen=True)
class CenterSpec:
    """One group-centered derivative of a label column."""

    output_col: str
    group_cols: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.output_col:
            raise ValueError("output_col cannot be empty")
        if not self.group_cols:
            raise ValueError("group_cols cannot be empty")


DEFAULT_CENTER_SPECS = (
    CenterSpec("label_z_within_stock_day", ("Date", "QuoteCode")),
    CenterSpec("label_z_within_day", ("Date",)),
)


def _schema_names(frame: pl.DataFrame | pl.LazyFrame) -> set[str]:
    if isinstance(frame, pl.LazyFrame):
        return set(frame.collect_schema().names())
    return set(frame.columns)


def required_label_columns(spec: DirectionalLabelSpec) -> tuple[str, ...]:
    """Return the complete, deterministic input schema for a label profile."""

    required = {spec.entry.price_col, spec.exit.fallback_price_col}
    if spec.exit.has_barrier:
        assert spec.exit.trigger_price_col is not None
        assert spec.exit.barrier_reference_col is not None
        required.update(
            {spec.exit.trigger_price_col, spec.exit.barrier_reference_col}
        )
        if spec.exit.tick_size_col is not None:
            required.add(spec.exit.tick_size_col)
    if spec.hedge is not None:
        required.update({spec.hedge.beta_col, spec.hedge.factor_return_bp_col})
    if spec.normalization is not None:
        required.add(spec.normalization.vol_col)
    return tuple(sorted(required))


def _finite(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.Float64, strict=False).is_finite()


def _trigger_expr(spec: ExitSpec, barrier: pl.Expr) -> pl.Expr:
    assert spec.trigger_price_col is not None
    trigger = pl.col(spec.trigger_price_col).cast(pl.Float64, strict=False)
    if spec.trigger_op == TriggerOp.GT:
        return trigger > barrier
    if spec.trigger_op == TriggerOp.GE:
        return trigger >= barrier
    if spec.trigger_op == TriggerOp.LT:
        return trigger < barrier
    return trigger <= barrier


def _exit_price_expr(spec: DirectionalLabelSpec) -> pl.Expr:
    exit_spec = spec.exit
    fallback = pl.col(exit_spec.fallback_price_col).cast(pl.Float64, strict=False)
    if not exit_spec.has_barrier:
        return fallback

    assert exit_spec.barrier_reference_col is not None
    assert exit_spec.barrier_multiplier is not None
    barrier = (
        pl.col(exit_spec.barrier_reference_col).cast(pl.Float64, strict=False)
        * exit_spec.barrier_multiplier
    )
    fill = barrier
    if exit_spec.adverse_slippage_ticks > 0.0:
        assert exit_spec.tick_size_col is not None
        tick = pl.col(exit_spec.tick_size_col).cast(pl.Float64, strict=False)
        fill = fill - (
            float(spec.direction)
            * exit_spec.adverse_slippage_ticks
            * tick
        )
    return pl.when(_trigger_expr(exit_spec, barrier)).then(fill).otherwise(fallback)


def add_directional_labels(
    frame: FrameT,
    *,
    spec: DirectionalLabelSpec,
    columns: LabelColumns = DEFAULT_LABEL_COLUMNS,
) -> FrameT:
    """Attach raw, hedged, residual, and volatility-normalized event labels.

    The function preserves eager/lazy Polars execution.  Invalid prices,
    missing hedge inputs, and non-positive volatility produce null derived
    values and ``label_valid=False`` rather than infinities or dropped rows.
    """

    missing = sorted(set(required_label_columns(spec)) - _schema_names(frame))
    if missing:
        raise ValueError(f"label input missing columns: {missing}")

    direction = float(spec.direction)
    output = frame.with_columns(
        pl.lit(spec.profile_id).alias(columns.profile_id),
        pl.lit(int(spec.direction), dtype=pl.Int8).alias(columns.direction),
        pl.col(spec.entry.price_col)
        .cast(pl.Float64, strict=False)
        .alias(columns.entry_price),
        _exit_price_expr(spec).alias(columns.exit_price),
    )

    prices_valid = (
        _finite(columns.entry_price)
        & _finite(columns.exit_price)
        & (pl.col(columns.entry_price) > 0.0)
        & (pl.col(columns.exit_price) > 0.0)
    )
    price_move = pl.when(prices_valid).then(
        (pl.col(columns.exit_price) - pl.col(columns.entry_price))
        / pl.col(columns.entry_price)
        * 10_000.0
    )
    output = output.with_columns(
        price_move.alias(columns.price_move_bp),
        (price_move * direction).alias(columns.raw_bp),
    )

    if spec.hedge is None:
        output = output.with_columns(
            pl.lit(0.0).alias(columns.market_component_bp),
            pl.lit(0.0).alias(columns.hedge_pnl_bp),
        )
    else:
        beta = pl.col(spec.hedge.beta_col).cast(pl.Float64, strict=False)
        factor = pl.col(spec.hedge.factor_return_bp_col).cast(
            pl.Float64, strict=False
        )
        component = pl.when(beta.is_finite() & factor.is_finite()).then(
            beta * factor
        )
        output = output.with_columns(
            component.alias(columns.market_component_bp),
            (
                component
                * (-direction * spec.hedge.hedge_ratio)
            ).alias(columns.hedge_pnl_bp),
        )

    residual_valid = (
        _finite(columns.raw_bp) & _finite(columns.hedge_pnl_bp)
    )
    output = output.with_columns(
        pl.when(residual_valid)
        .then(pl.col(columns.raw_bp) + pl.col(columns.hedge_pnl_bp))
        .alias(columns.residual_bp)
    )

    if spec.normalization is None:
        output = output.with_columns(
            pl.lit(None, dtype=pl.Float64).alias(columns.vol_bp),
            pl.lit(None, dtype=pl.Float64).alias(columns.z),
            _finite(columns.residual_bp).fill_null(False).alias(columns.valid),
        )
        return output

    raw_vol = pl.col(spec.normalization.vol_col).cast(pl.Float64, strict=False)
    vol_bp = (
        raw_vol * 10_000.0
        if spec.normalization.unit == VolUnit.RETURN
        else raw_vol
    )
    output = output.with_columns(
        pl.when(vol_bp.is_finite()).then(vol_bp).alias(columns.vol_bp)
    )
    z_valid = (
        _finite(columns.residual_bp)
        & _finite(columns.vol_bp)
        & (pl.col(columns.vol_bp) > spec.normalization.min_vol_bp)
    )
    output = output.with_columns(
        pl.when(z_valid)
        .then(pl.col(columns.residual_bp) / pl.col(columns.vol_bp))
        .alias(columns.z),
        z_valid.fill_null(False).alias(columns.valid),
    )
    return output


def add_centered_labels(
    frame: FrameT,
    *,
    source_col: str = DEFAULT_LABEL_COLUMNS.z,
    centers: tuple[CenterSpec, ...] = DEFAULT_CENTER_SPECS,
) -> FrameT:
    """Add group-demeaned label variants without changing the source unit."""

    required = {source_col}
    for center in centers:
        required.update(center.group_cols)
    missing = sorted(required - _schema_names(frame))
    if missing:
        raise ValueError(f"centered-label input missing columns: {missing}")

    clean = pl.when(
        pl.col(source_col).cast(pl.Float64, strict=False).is_finite()
    ).then(pl.col(source_col).cast(pl.Float64, strict=False))
    expressions = []
    for center in centers:
        group_mean = clean.mean().over(list(center.group_cols))
        expressions.append(
            pl.when(clean.is_not_null())
            .then(clean - group_mean)
            .alias(center.output_col)
        )
    return frame.with_columns(expressions)
