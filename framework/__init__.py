"""Public contracts for generic intraday event research.

The package deliberately contains no default event condition.  Strategy
families such as negFill live in adapters and supply versioned selectors,
labels, universes, feature manifests, and thresholds to these APIs.
"""

from .condition_study import (
    ConditionComparisonSpec,
    ConditionDayResult,
    EligiblePoolSpec,
    PrimaryEstimand,
    build_eligible_labeled_pool,
    compare_condition_to_control,
    compare_conditions_to_shared_control,
    read_day_event_keys,
    summarize_cohort,
    summarize_condition_day,
)
from .dataset import (
    ALL_EVENTS_UNIVERSE,
    EventDatasetSpec,
    MergeOptions,
    UniverseSpec,
    build_event_dataset,
    materialize_event_day,
    merge_event_day,
    read_event_dataset,
)
from .events import (
    DataRoots,
    EventDaySources,
    EventIndexSpec,
    build_event_index,
    normalize_event_keys,
    validate_event_index,
)
from .features import FeatureManifest, FeatureSpec
from .labels import (
    DirectionalLabelSpec,
    add_centered_labels,
    add_directional_labels,
)
from .model import train_and_predict_causal_v2
from .prior_thresholds import (
    PriorQuantileSpec,
    WindowBasis,
    apply_prior_threshold,
    build_prior_session_thresholds,
)

__all__ = [
    "ALL_EVENTS_UNIVERSE",
    "ConditionComparisonSpec",
    "ConditionDayResult",
    "DataRoots",
    "DirectionalLabelSpec",
    "EligiblePoolSpec",
    "EventDatasetSpec",
    "EventDaySources",
    "EventIndexSpec",
    "FeatureManifest",
    "FeatureSpec",
    "MergeOptions",
    "PriorQuantileSpec",
    "PrimaryEstimand",
    "UniverseSpec",
    "WindowBasis",
    "add_centered_labels",
    "add_directional_labels",
    "apply_prior_threshold",
    "build_eligible_labeled_pool",
    "build_event_dataset",
    "build_event_index",
    "build_prior_session_thresholds",
    "compare_condition_to_control",
    "compare_conditions_to_shared_control",
    "materialize_event_day",
    "merge_event_day",
    "normalize_event_keys",
    "read_day_event_keys",
    "read_event_dataset",
    "summarize_cohort",
    "summarize_condition_day",
    "train_and_predict_causal_v2",
    "validate_event_index",
]
