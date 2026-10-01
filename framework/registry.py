"""Signal contracts: one :class:`SignalConfig` per intraday signal, read by all four layers.

The pipeline is four layers, each a module under ``layers/`` and each with one
section of the signal's config:

``envelope`` (``layers/envelope.py``)
    ``envelope``: parameter-free selector on raw day sources; decides which
    ticks are materialized into the shared store.  ``gate`` + ``selection``:
    the *condition* — a parameterized subset whose parameters are re-picked
    daily from a frozen grid on strictly-prior store rows.  Output: the signal
    index.
``point_filter`` (``layers/point_filter.py``)
    ``point_filter``: fixed tick-level removals (session window, trial match,
    signal-specific tick filters).  No grid.  Output: the candidate points.
``day_selector`` (``layers/day_selector.py``)
    ``day_selector``: a shallow model refit every day on strictly-prior stock-
    days that removes *stocks* for the next day; ``hard_rules`` stay on
    regardless of the model.  Output: day scores.
``model`` (``layers/model.py``)
    ``model``: label, features, rolling Ridge A/M and the 16-cell thresholds,
    trained on what survived the three layers above.

Specs here are data only; no layer logic lives in this module.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd
import polars as pl

from .dataset import ALL_EVENTS_UNIVERSE, UniverseSpec
from .events import EventSelector, SourceName


GateStage = Literal["condition", "point_filter", "hard_rule"]
SignalStage = Literal["research", "shadow", "deployed"]
ParamValue = float | int | str | bool | None
Params = Mapping[str, ParamValue]
GateFn = Callable[[pl.DataFrame, Params], pl.Series]
PoolFn = Callable[[pl.DataFrame], pl.DataFrame]
PandasStep = Callable[[pd.DataFrame], pd.DataFrame]
ModelFamily = Literal["ridge", "hist_gb", "random_forest"]


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def param_set_id(params: Params) -> str:
    """Stable identifier for one parameter combination."""

    digest = hashlib.sha1(_stable_json(dict(params)).encode("utf-8")).hexdigest()
    return f"p_{digest[:10]}"


# --------------------------------------------------------------------------- #
# gates (used by the condition, the point filter and the day selector's hard rules)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GateSpec:
    """A parameterized boolean filter over materialized store rows.

    ``stage`` says which layer owns it: ``condition`` (rolled daily from
    ``grid``), ``point_filter`` (fixed, no grid) or ``hard_rule`` (fixed,
    stock-day level, applied by the day selector).
    """

    gate_id: str
    version: str
    params: Mapping[str, ParamValue]
    apply: GateFn = field(repr=False, compare=False)
    grid: Mapping[str, tuple[ParamValue, ...]] = field(default_factory=dict)
    stage: GateStage = "condition"

    def __post_init__(self) -> None:
        if not self.gate_id or not self.version:
            raise ValueError("gate_id and version are required")
        if not callable(self.apply):
            raise TypeError("GateSpec.apply must be callable")
        object.__setattr__(self, "params", dict(self.params))
        grid: dict[str, tuple[ParamValue, ...]] = {}
        for name, values in dict(self.grid).items():
            if name not in self.params:
                raise ValueError(f"grid parameter {name!r} has no default in params")
            values = tuple(values)
            if not values:
                raise ValueError(f"grid parameter {name!r} has no candidate values")
            if len(set(values)) != len(values):
                raise ValueError(f"grid parameter {name!r} repeats a value")
            grid[name] = values
        object.__setattr__(self, "grid", grid)
        if self.stage not in ("condition", "point_filter", "hard_rule"):
            raise ValueError(f"unknown gate stage {self.stage!r}")
        if self.stage != "condition" and grid:
            raise ValueError(f"gate {self.gate_id}: only a condition gate may carry a grid")

    @property
    def default_param_set_id(self) -> str:
        return param_set_id(self.params)

    def expand_grid(self) -> tuple[dict[str, ParamValue], ...]:
        """All parameter combinations; the default set is always first."""

        default = dict(self.params)
        if not self.grid:
            return (default,)
        names = list(self.grid)
        combos: list[dict[str, ParamValue]] = []
        for values in itertools.product(*(self.grid[name] for name in names)):
            candidate = dict(default)
            candidate.update(dict(zip(names, values)))
            combos.append(candidate)
        default_id = self.default_param_set_id
        combos.sort(key=lambda c: 0 if param_set_id(c) == default_id else 1)
        if param_set_id(combos[0]) != default_id:
            combos.insert(0, default)
        return tuple(combos)

    def mask(self, frame: pl.DataFrame, params: Params | None = None) -> pl.Series:
        """Evaluate the gate and enforce the boolean, aligned-length contract."""

        merged = dict(self.params)
        if params:
            unknown = sorted(set(params) - set(self.params))
            if unknown:
                raise ValueError(f"gate {self.gate_id}: unknown parameters {unknown}")
            merged.update(params)
        result = self.apply(frame, merged)
        if not isinstance(result, pl.Series):
            raise TypeError(f"gate {self.gate_id} must return a polars Series")
        if result.dtype != pl.Boolean:
            raise TypeError(f"gate {self.gate_id} must return a Boolean Series")
        if len(result) != frame.height:
            raise ValueError(f"gate {self.gate_id} returned {len(result)} rows for {frame.height}")
        return result.fill_null(False)

    def mask_pd(self, frame: pd.DataFrame, params: Params | None = None) -> pd.Series:
        """Evaluate the gate on a pandas frame; returns an aligned bool Series."""

        if frame.empty:
            return pd.Series(False, index=frame.index, dtype=bool)
        mask = self.mask(pl.from_pandas(frame.reset_index(drop=True)), params)
        return pd.Series(mask.to_numpy(), index=frame.index, dtype=bool)

    def stable_hash(self) -> str:
        payload = {
            "gate_id": self.gate_id,
            "version": self.version,
            "params": dict(self.params),
            "grid": {k: list(v) for k, v in self.grid.items()},
            "stage": self.stage,
        }
        return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RollingSelection:
    """How the condition's daily parameters are chosen from strictly-prior rows."""

    window_days: int = 60
    objective: Literal["stockday_mean_label", "row_mean_label"] = "stockday_mean_label"
    label_col: str = "label_residual_bp"
    min_rows: int = 300
    min_stock_days: int = 30
    min_stock_days_per_day: float = 10.0

    def __post_init__(self) -> None:
        if self.window_days <= 0:
            raise ValueError("window_days must be positive")
        if self.min_stock_days_per_day < 0:
            raise ValueError("min_stock_days_per_day cannot be negative")
        if self.objective not in ("stockday_mean_label", "row_mean_label"):
            raise ValueError(f"unknown objective {self.objective!r}")


# --------------------------------------------------------------------------- #
# day selector
# --------------------------------------------------------------------------- #


DEFAULT_DAY_FEATURES = (
    # every column is a D-1 value delivered by preMarket / brenchFeature; the only transform is log1p(trading_turnover)
    "MarketCap",
    "top1_net_buy_share",
    "net_buy_gini",
    "net_sell_gini",
    "sell_branch_breadth_rank",
    "buy_branch_breadth_rank",
    "d_intraday_return",
    "dt_major_count",
    "hft_strick_makerSpreadBP",
    "hft_participation",
    "OTC",
    "day_amount_rank",
    "RefPrice",
    "log_trading_turnover",
    "avg_bidLots1",
    "avg_askLots1",
    "big_sell_lots",
    "turnover_rate",
)


WeightFn = Callable[[pl.DataFrame], pl.Series]
RiskTask = Literal["regression", "classification"]
RiskFamily = Literal["ridge", "logistic", "hist_gb"]
RiskDirection = Literal["exclude_above", "exclude_below"]


@dataclass(frozen=True, kw_only=True)
class RiskModelSpec:
    """One learned stock-day risk filter: predict a ``ds_*`` target from D-1 features, cut at an absolute threshold.

    A regression model excludes a stock-day when its predicted value crosses
    ``threshold``; a classification model compares the predicted probability.
    Thresholds are human-set, like the hand rules they replace, and never rolled.
    """

    model_id: str
    target: str
    threshold: float
    task: RiskTask = "regression"
    family: RiskFamily = "ridge"
    direction: RiskDirection = "exclude_above"
    threshold_mode: Literal["absolute", "window_quantile"] = "absolute"   # window_quantile: threshold is a quantile (0-1) of the training window's predictions
    features: tuple[str, ...] = DEFAULT_DAY_FEATURES
    hyper: Mapping[str, object] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        if self.threshold_mode not in ("absolute", "window_quantile"):
            raise ValueError(f"unknown threshold_mode {self.threshold_mode!r}")
        if self.threshold_mode == "window_quantile" and not (0.0 < self.threshold < 1.0):
            raise ValueError(f"{self.model_id}: a window_quantile threshold must be in (0, 1)")
        if not self.model_id.isidentifier():
            raise ValueError(f"model_id must be an identifier, got {self.model_id!r}")
        if not self.target.startswith("ds_"):
            raise ValueError(f"{self.model_id}: day-selector targets carry the ds_ prefix, got {self.target!r}")
        if self.task not in ("regression", "classification"):
            raise ValueError(f"unknown task {self.task!r}")
        allowed = {"regression": ("ridge", "hist_gb"), "classification": ("logistic", "hist_gb")}[self.task]
        if self.family not in allowed:
            raise ValueError(f"{self.model_id}: family {self.family!r} is not a {self.task} family {allowed}")
        if self.direction not in ("exclude_above", "exclude_below"):
            raise ValueError(f"unknown direction {self.direction!r}")
        object.__setattr__(self, "features", tuple(self.features))
        object.__setattr__(self, "hyper", dict(self.hyper))

    @property
    def score_column(self) -> str:
        return f"score_{self.model_id}"

    @property
    def cut_column(self) -> str:
        return f"cut_{self.model_id}"

    def excludes(self, score, cut: float | None = None):
        """Boolean array: stock-days this model removes (``cut`` = the day's resolved threshold)."""

        cut = self.threshold if cut is None else cut
        return score > cut if self.direction == "exclude_above" else score < cut

    def payload(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id, "target": self.target, "task": self.task, "family": self.family,
            "threshold": self.threshold, "threshold_mode": self.threshold_mode, "direction": self.direction,
            "features": list(self.features), "hyper": dict(self.hyper),
        }


@dataclass(frozen=True, kw_only=True)
class DaySelectorSpec:
    """Learned stock-day gate: several simple risk models, refit daily, each adding exclusions.

    ``labels`` turns a day's candidate rows into one row per (Date, QuoteCode)
    with the ``ds_*`` targets; ``weight`` gives each stock-day a sample weight
    (the share of the position cap the book could fill there); ``hard_rules``
    stay on regardless of the models.  ``kept = hard_rules & no model excludes``.
    """

    selector_id: str
    version: str = "2"
    labels: PoolFn = field(repr=False, compare=False)
    models: tuple[RiskModelSpec, ...]
    weight: WeightFn | None = field(default=None, repr=False, compare=False)
    hard_rules: GateSpec | None = None
    baseline_rules: GateSpec | None = None   # evaluation only: the hand rules the models are compared against
    window_days: int = 60
    min_train_stock_days: int = 500
    description: str = ""

    def __post_init__(self) -> None:
        if not callable(self.labels):
            raise TypeError("DaySelectorSpec.labels must be callable")
        if self.weight is not None and not callable(self.weight):
            raise TypeError("DaySelectorSpec.weight must be callable")
        object.__setattr__(self, "models", tuple(self.models))
        if not self.models:
            raise ValueError("DaySelectorSpec needs at least one risk model")
        ids = [m.model_id for m in self.models]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate model_id in day selector: {ids}")
        for gate in (self.hard_rules, self.baseline_rules):
            if gate is not None and gate.stage != "hard_rule":
                raise ValueError("DaySelectorSpec rules must be hard_rule gates")

    @property
    def model_id(self) -> str:
        return f"{self.selector_id}_v{self.version}"

    @property
    def features(self) -> tuple[str, ...]:
        seen: list[str] = []
        for m in self.models:
            seen.extend(c for c in m.features if c not in seen)
        return tuple(seen)

    @property
    def targets(self) -> tuple[str, ...]:
        seen: list[str] = []
        for m in self.models:
            if m.target not in seen:
                seen.append(m.target)
        return tuple(seen)

    def stable_hash(self) -> str:
        payload = {
            "model_id": self.model_id,
            "models": [m.payload() for m in self.models],
            "window_days": self.window_days,
            "min_train_stock_days": self.min_train_stock_days,
            "hard_rules": self.hard_rules.stable_hash() if self.hard_rules else None,
        }
        return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CellThresholdSpec:
    """Prior-day quantile threshold on model scores inside fixed 2D cells."""

    time_col: str = "RemainSeconds"
    time_edges: tuple[float, ...] = (3300, 6900, 10500, 14100, 99999)
    ref_col: str = "ToRef"
    ref_edges: tuple[float, ...] = (0, 0.01, 0.02, 0.03, 0.05)
    n_days: int = 3
    q_timing: float = 0.3
    q_selection: float = 0.1

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_edges", tuple(self.time_edges))
        object.__setattr__(self, "ref_edges", tuple(self.ref_edges))


@dataclass(frozen=True, kw_only=True)
class ModelSpec:
    """Label / feature / rolling-Ridge / threshold contract (pandas side)."""

    version: str = "1"
    label: PandasStep = field(repr=False, compare=False)
    train_filter: PandasStep = field(repr=False, compare=False)   # label-validity guard (needs the label)
    features: PandasStep = field(repr=False, compare=False)
    feature_columns: tuple[str, ...]
    timing_target: str
    selection_target: str
    raw_label_col: str
    timing_lookback_days: int = 60
    selection_lookback_days: int = 20
    alpha: float = 10.0
    max_position_per_stock: float | None = 500
    feature_screening: bool = True
    threshold: CellThresholdSpec = field(default_factory=CellThresholdSpec)
    use_model_gate: bool = True   # False = static book: models/thresholds are diagnostics only
    round_feature_decimals: int | None = 6
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_columns", tuple(self.feature_columns))
        if not self.feature_columns:
            raise ValueError("feature_columns cannot be empty")
        for step in (self.label, self.train_filter, self.features):
            if not callable(step):
                raise TypeError("ModelSpec.label / train_filter / features must be callable")

    @property
    def timing_pred_col(self) -> str:
        return f"{self.timing_target}_pred"

    @property
    def selection_pred_col(self) -> str:
        return f"{self.selection_target}_pred"


# --------------------------------------------------------------------------- #
# signal = the four sections
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, kw_only=True)
class SignalConfig:
    """One registered intraday signal; every layer reads its own section of this object."""

    st_name: str
    version: str
    stage: SignalStage = "research"
    description: str = ""

    # layer 1: envelope + condition
    envelope: EventSelector = field(repr=False, compare=False)
    envelope_required_sources: tuple[SourceName, ...]
    envelope_columns: tuple[str, ...] = ()
    universe: UniverseSpec = ALL_EVENTS_UNIVERSE
    gate: GateSpec
    selection: RollingSelection = field(default_factory=RollingSelection)
    selection_pool: PoolFn = field(repr=False, compare=False)   # candidate rows -> labeled rows for grid scoring

    # layer 2: fixed tick-level removals
    point_filter: GateSpec | None = None

    # layer 3: learned stock-day gate
    day_selector: DaySelectorSpec | None = None

    # layer 4: model + thresholds
    model: ModelSpec | None = None

    def __post_init__(self) -> None:
        if self.stage not in ("research", "shadow", "deployed"):
            raise ValueError(f"unknown signal stage {self.stage!r}")
        if not self.st_name.isidentifier():
            raise ValueError(f"st_name must be an identifier, got {self.st_name!r}")
        if not self.version:
            raise ValueError("version is required")
        if not self.envelope_required_sources:
            raise ValueError("envelope_required_sources cannot be empty")
        if not callable(self.envelope):
            raise TypeError("envelope must be callable")
        if not callable(self.selection_pool):
            raise TypeError("selection_pool must be callable")
        object.__setattr__(self, "envelope_required_sources", tuple(self.envelope_required_sources))
        object.__setattr__(self, "envelope_columns", tuple(self.envelope_columns))
        if self.gate.stage != "condition":
            raise ValueError("SignalConfig.gate must be a condition gate")
        if self.point_filter is not None and self.point_filter.stage != "point_filter":
            raise ValueError("SignalConfig.point_filter must be a point_filter gate")

    @property
    def condition_id(self) -> str:
        return f"{self.st_name}_v{self.version}"

    @property
    def tag_column(self) -> str:
        return f"cond_{self.st_name}"

    def envelope_store_column(self, column: str) -> str:
        return f"{self.st_name}__{column}"

    def stable_hash(self) -> str:
        """Layer-1 identity (envelope + condition); the other layers hash their own specs."""

        payload = {
            "condition_id": self.condition_id,
            "envelope_required_sources": list(self.envelope_required_sources),
            "envelope_columns": list(self.envelope_columns),
            "gate": self.gate.stable_hash(),
            "selection": {
                "window_days": self.selection.window_days,
                "objective": self.selection.objective,
                "label_col": self.selection.label_col,
                "min_rows": self.selection.min_rows,
                "min_stock_days": self.selection.min_stock_days,
                "min_stock_days_per_day": self.selection.min_stock_days_per_day,
            },
            "universe": [self.universe.universe_id, self.universe.universe_version],
        }
        return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


class SignalRegistry:
    """Ordered, name-unique collection of signals."""

    def __init__(self, signals: Sequence[SignalConfig]) -> None:
        names = [c.st_name for c in signals]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate st_name in registry: {duplicates}")
        self._signals = tuple(signals)

    def __iter__(self) -> Iterator[SignalConfig]:
        return iter(self._signals)

    def __len__(self) -> int:
        return len(self._signals)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(c.st_name for c in self._signals)

    def get(self, st_name: str) -> SignalConfig:
        for signal in self._signals:
            if signal.st_name == st_name:
                return signal
        raise KeyError(f"unknown signal {st_name!r}; registered: {self.names}")

    def select(self, only: Sequence[str] | None) -> tuple[SignalConfig, ...]:
        if only is None:
            return self._signals
        unknown = sorted(set(only) - set(self.names))
        if unknown:
            raise KeyError(f"unknown signal(s) {unknown}; registered: {self.names}")
        return tuple(c for c in self._signals if c.st_name in set(only))


__all__ = [
    "CellThresholdSpec",
    "DEFAULT_DAY_FEATURES",
    "DaySelectorSpec",
    "GateSpec",
    "RiskModelSpec",
    "GateStage",
    "ModelFamily",
    "ModelSpec",
    "Params",
    "PoolFn",
    "RollingSelection",
    "SignalConfig",
    "SignalRegistry",
    "SignalStage",
    "param_set_id",
]
