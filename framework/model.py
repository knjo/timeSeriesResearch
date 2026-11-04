"""Causal-v2 model entry point for time-series event research."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .features import FeatureManifest, prepare_feature_frame


def train_and_predict_causal_v2(
    signal_df: pd.DataFrame,
    manifest: FeatureManifest,
    Y_col: str,
    **rolling_kwargs: Any,
) -> dict[str, object]:
    """Run the rolling linear engine from an explicit feature contract.

    The wrapper supplies the manifest preparation as the rolling engine's one
    causal preprocessing step.  Feature roles are derived only from the
    manifest; callers cannot override them through ``rolling_kwargs``. Ridge
    remains the default. Controlled research ablations may explicitly request
    ``model_family="elastic_net"`` and its penalty contract.
    """

    reserved = {
        "X_list",
        "day_features",
        "forced_masks",
        "forced_features",
        "preprocessing_mode",
        "causal_feature_preprocessor",
    }
    overridden = sorted(reserved & set(rolling_kwargs))
    if overridden:
        raise TypeError(
            "Causal-v2 wrapper owns feature/preprocessing arguments; remove: "
            f"{overridden}"
        )
    if Y_col in manifest.ordered_names:
        raise ValueError(f"Prediction target {Y_col} cannot also be a feature.")

    def prepare_from_manifest(frame: pd.DataFrame) -> pd.DataFrame:
        return prepare_feature_frame(frame, manifest)

    # Keep package import side-effect free for event/dataset/label users.  The
    # legacy engine configures BLAS process settings at import time, so load it
    # only when training was explicitly requested.
    from src.research.rolling_model import train_and_predict_ridge_rolling

    result = train_and_predict_ridge_rolling(
        signal_df=signal_df,
        X_list=list(manifest.ordered_names),
        Y_col=Y_col,
        day_features=list(manifest.names_for_role("day")),
        forced_masks=list(manifest.names_for_role("mask")),
        forced_features=list(manifest.names_for_role("forced")),
        preprocessing_mode="causal_v2",
        causal_feature_preprocessor=prepare_from_manifest,
        **rolling_kwargs,
    )
    result["feature_manifest"] = {
        "name": manifest.name,
        "version": manifest.version,
        "hash": manifest.stable_hash(),
        "ordered_features": list(manifest.ordered_names),
    }
    return result
