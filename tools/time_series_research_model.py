"""Import-safe negFill reference training adapter.

This module intentionally performs no file discovery, label construction, or
model fitting at import time.  Generic research should assemble its own label
profile and feature manifest; these helpers simply pin the current negFill
reference choices for parity experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import polars as pl

from src.research.intradaySignal.framework.features import FeatureManifest
from src.research.intradaySignal.framework.labels import (
    DirectionalLabelSpec,
    add_centered_labels,
    add_directional_labels,
)
from src.research.intradaySignal.framework.profiles.negfill import (
    NEGFILL_V1,
    SHORT_TAKER_STOP108_V1,
)
from src.research.intradaySignal.framework.model import train_and_predict_causal_v2


@dataclass(frozen=True)
class ReferenceTrainingSpec:
    """Versioned choices for the negFill method-validation model pair."""

    label: DirectionalLabelSpec = SHORT_TAKER_STOP108_V1
    features: FeatureManifest = NEGFILL_V1
    timing_target: str = "label_z_within_stock_day"
    selection_target: str = "label_z_within_day"
    timing_lookback_days: int = 60
    selection_lookback_days: int = 20
    alpha: float = 0.1


NEGFILL_REFERENCE_TRAINING_V2 = ReferenceTrainingSpec()


def add_reference_labels(
    frame: pl.DataFrame | pl.LazyFrame,
    *,
    spec: ReferenceTrainingSpec = NEGFILL_REFERENCE_TRAINING_V2,
) -> pl.DataFrame | pl.LazyFrame:
    """Apply the pinned directional residual-z and centered target contracts."""

    return add_centered_labels(add_directional_labels(frame, spec=spec.label))


def train_reference_models(
    signal_df: pd.DataFrame,
    *,
    spec: ReferenceTrainingSpec = NEGFILL_REFERENCE_TRAINING_V2,
    common_rolling_kwargs: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Fit the 60-day timing and 20-day selection reference models.

    ``signal_df`` must already contain labels emitted by
    :func:`add_reference_labels` and all ordered manifest features. Predictions
    remain in z units; callers that need BP estimates must write a separate
    column and multiply by the event-horizon volatility.
    """

    common = dict(common_rolling_kwargs or {})
    forbidden = {"n_training_days", "alpha"} & set(common)
    if forbidden:
        raise TypeError(
            "ReferenceTrainingSpec owns these rolling arguments: "
            f"{sorted(forbidden)}"
        )
    missing_targets = [
        target
        for target in (spec.timing_target, spec.selection_target)
        if target not in signal_df.columns
    ]
    if missing_targets:
        raise ValueError(
            "reference frame is missing centered label targets: "
            f"{missing_targets}; call add_reference_labels first"
        )

    timing = train_and_predict_causal_v2(
        signal_df,
        spec.features,
        spec.timing_target,
        n_training_days=spec.timing_lookback_days,
        alpha=spec.alpha,
        **common,
    )
    selection = train_and_predict_causal_v2(
        timing["df"],
        spec.features,
        spec.selection_target,
        n_training_days=spec.selection_lookback_days,
        alpha=spec.alpha,
        **common,
    )
    return {
        "df": selection["df"],
        "timing_model": timing,
        "selection_model": selection,
        "reference_contract": {
            "label_profile_id": spec.label.profile_id,
            "label_profile_hash": spec.label.stable_hash(),
            "feature_manifest": spec.features.name,
            "feature_manifest_version": spec.features.version,
            "feature_manifest_hash": spec.features.stable_hash(),
            "timing_target": spec.timing_target,
            "selection_target": spec.selection_target,
            "timing_lookback_days": spec.timing_lookback_days,
            "selection_lookback_days": spec.selection_lookback_days,
            "alpha": spec.alpha,
        },
    }


__all__ = [
    "NEGFILL_REFERENCE_TRAINING_V2",
    "ReferenceTrainingSpec",
    "add_reference_labels",
    "train_reference_models",
]
