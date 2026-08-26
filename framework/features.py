"""Versioned feature contracts and causal missing-value preparation.

Feature order is part of the model contract.  This module deliberately avoids
schema discovery (suffix matching, regular expressions, or dataframe order): a
model receives exactly the fields declared by its :class:`FeatureManifest`.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable, Literal

import numpy as np
import pandas as pd


FeatureRole = Literal["tick", "day", "mask", "forced"]
MissingPolicy = Literal["prior_median", "zero", "constant", "error"]
Availability = Literal["pre_market", "event"]

FEATURE_ROLES = frozenset({"tick", "day", "mask", "forced"})
MISSING_POLICIES = frozenset({"prior_median", "zero", "constant", "error"})
AVAILABILITIES = frozenset({"pre_market", "event"})

# Exact names known to contain an event outcome or information observed after
# entry.  Manifests may add more strategy-specific forbidden names, but may not
# remove these defaults.
FORBIDDEN_FUTURE_COLUMNS = frozenset(
    {
        "Close",
        "FutureHigh",
        "FutureLow",
        "txf_close_mid",
        "txf_close_quote_time",
        "txf_to_1330_return",
        "txf_to_1330_bp",
        "TakerSell_CloseBP",
        "TakerBuy_CloseBP",
        "residual_TakerSell_CloseBP",
        "residual_TakerBuy_CloseBP",
        "TakerSell_CloseBP_net",
        "TakerSell_CloseBP_netM",
        "TakerBuy_CloseBP_net",
        "TakerBuy_CloseBP_netM",
    }
)


@dataclass(frozen=True)
class FeatureSpec:
    """One ordered, event-time-available model input."""

    name: str
    role: FeatureRole = "tick"
    required: bool = True
    missing_policy: MissingPolicy | None = None
    fallback: float = 0.0
    source: str = "upstream"
    availability: Availability = "event"
    transform: str = "identity"

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Feature name must be non-empty.")
        if self.role not in FEATURE_ROLES:
            raise ValueError(f"Unsupported feature role for {self.name}: {self.role}")
        if self.effective_missing_policy not in MISSING_POLICIES:
            raise ValueError(
                f"Unsupported missing policy for {self.name}: "
                f"{self.effective_missing_policy}"
            )
        if self.availability not in AVAILABILITIES:
            raise ValueError(
                f"Unsupported availability for {self.name}: {self.availability}"
            )
        if not math.isfinite(float(self.fallback)):
            raise ValueError(f"Feature fallback must be finite: {self.name}")
        if self.role == "mask" and self.effective_missing_policy not in {
            "zero",
            "error",
        }:
            raise ValueError(
                f"Mask feature {self.name} must use zero or error missing policy."
            )

    @property
    def effective_missing_policy(self) -> MissingPolicy:
        if self.missing_policy is not None:
            return self.missing_policy
        return "zero" if self.role == "mask" else "prior_median"

    def canonical_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "role": self.role,
            "required": self.required,
            "missing_policy": self.effective_missing_policy,
            "fallback": float(self.fallback),
            "source": self.source,
            "availability": self.availability,
            "transform": self.transform,
        }


@dataclass(frozen=True)
class FeatureManifest:
    """An immutable, ordered feature contract with a stable fingerprint."""

    name: str
    version: str
    features: tuple[FeatureSpec, ...]
    forbidden_columns: frozenset[str] = FORBIDDEN_FUTURE_COLUMNS

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", tuple(self.features))
        object.__setattr__(
            self,
            "forbidden_columns",
            frozenset(self.forbidden_columns) | FORBIDDEN_FUTURE_COLUMNS,
        )
        if not self.name or not self.version:
            raise ValueError("Feature manifest name and version are required.")
        names = self.ordered_names
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate feature names: {duplicates}")
        forbidden = sorted(set(names) & self.forbidden_columns)
        if forbidden:
            raise ValueError(f"Future/outcome columns cannot be features: {forbidden}")

    @property
    def ordered_names(self) -> tuple[str, ...]:
        return tuple(feature.name for feature in self.features)

    def names_for_role(self, role: FeatureRole) -> tuple[str, ...]:
        if role not in FEATURE_ROLES:
            raise ValueError(f"Unsupported feature role: {role}")
        return tuple(feature.name for feature in self.features if feature.role == role)

    def canonical_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            # The list order is intentionally preserved and therefore affects
            # the hash even though mapping keys are sorted during serialization.
            "features": [feature.canonical_dict() for feature in self.features],
            "forbidden_columns": sorted(self.forbidden_columns),
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

    def validate_columns(self, columns: Iterable[str]) -> None:
        available = set(columns)
        missing = [
            feature.name
            for feature in self.features
            if feature.required and feature.name not in available
        ]
        if missing:
            raise ValueError(
                f"Manifest {self.name}@{self.version} is missing required "
                f"features: {missing}"
            )


def _numeric_feature(series: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    invalid = numeric.isna() & series.notna()
    if invalid.any():
        raise TypeError(f"Feature {name} contains non-numeric values.")
    return numeric.replace([np.inf, -np.inf], np.nan)


def _fill_with_strictly_prior_daily_median(
    values: pd.Series,
    dates: pd.Series,
    fallback: float,
) -> pd.Series:
    daily_median = (
        pd.DataFrame({"_date": dates, "_value": values})
        .groupby("_date", sort=True)["_value"]
        .median()
        .sort_index()
    )
    prior_median = daily_median.ffill().shift(1)
    return values.fillna(dates.map(prior_median)).fillna(float(fallback))


def prepare_feature_frame(
    df: pd.DataFrame,
    manifest: FeatureManifest,
    *,
    date_col: str = "Date",
) -> pd.DataFrame:
    """Validate and causally fill manifest features without changing ``df``.

    ``prior_median`` means the most recent finite daily median strictly before
    each row's date.  The current date's cross-section is never used to fill
    itself, and dates appended in the future cannot alter earlier rows.
    """

    manifest.validate_columns(df.columns)
    out = df.copy()
    needs_dates = any(
        feature.effective_missing_policy == "prior_median"
        for feature in manifest.features
    )
    if needs_dates and date_col not in out.columns:
        raise ValueError(f"Causal feature preparation requires {date_col}.")
    dates = (
        pd.to_datetime(out[date_col], errors="raise")
        if needs_dates
        else pd.Series(index=out.index, dtype="datetime64[ns]")
    )

    for feature in manifest.features:
        policy = feature.effective_missing_policy
        if feature.name not in out.columns:
            # Required fields were rejected above.  Optional fields get their
            # declared fixed fallback so output schema remains deterministic.
            out[feature.name] = float(feature.fallback)
        values = _numeric_feature(out[feature.name], feature.name)

        if policy == "prior_median":
            values = _fill_with_strictly_prior_daily_median(
                values, dates, feature.fallback
            )
        elif policy == "zero":
            values = values.fillna(0.0)
        elif policy == "constant":
            values = values.fillna(float(feature.fallback))
        elif policy == "error" and values.isna().any():
            raise ValueError(f"Feature {feature.name} contains missing values.")

        if values.isna().any():
            raise ValueError(f"Feature {feature.name} remains missing after fill.")
        if feature.role == "mask":
            mask_values = values.to_numpy(dtype=float)
            is_binary = np.isclose(mask_values, 0.0) | np.isclose(mask_values, 1.0)
            if not np.all(is_binary):
                bad = np.unique(mask_values[~is_binary])[:5].tolist()
                raise ValueError(
                    f"Mask feature {feature.name} must be binary after fill; "
                    f"found {bad}."
                )
            values = pd.Series(mask_values.astype(np.float32), index=out.index)

        out[feature.name] = values

    return out


