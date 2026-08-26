"""Auditable prior-session quantile thresholds.

Two window contracts are intentionally supported:

``calendar_sessions``
    Look only at the previous N sessions in the supplied trading calendar.

``bucket_observation_sessions``
    For each bucket, look at its previous N sessions with finite observations.
    This reproduces the historical notebook behavior for sparse buckets.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence, TypeVar

import numpy as np
import polars as pl


FrameT = TypeVar("FrameT", pl.DataFrame, pl.LazyFrame)


class WindowBasis(StrEnum):
    CALENDAR_SESSIONS = "calendar_sessions"
    BUCKET_OBSERVATION_SESSIONS = "bucket_observation_sessions"


@dataclass(frozen=True)
class PriorQuantileSpec:
    """Configuration for one prior-only score threshold."""

    score_col: str
    threshold_col: str
    q: float
    bucket_cols: tuple[str, ...] = ()
    lookback_sessions: int = 3
    min_sessions: int = 1
    min_observations: int = 1
    interpolation: str = "linear"
    window_basis: WindowBasis = WindowBasis.CALENDAR_SESSIONS
    score_unit: str = "model_score"

    def __post_init__(self) -> None:
        object.__setattr__(self, "window_basis", WindowBasis(self.window_basis))
        if not self.score_col:
            raise ValueError("score_col cannot be empty")
        if not self.threshold_col:
            raise ValueError("threshold_col cannot be empty")
        if self.score_col == self.threshold_col:
            raise ValueError("score_col and threshold_col must differ")
        if len(set(self.bucket_cols)) != len(self.bucket_cols):
            raise ValueError("bucket_cols must be unique")
        reserved = {
            self.threshold_col,
            "history_start",
            "history_end",
            "history_window_sessions",
            "history_sessions",
            "history_observations",
            "quantile",
            "window_basis",
            "score_col",
            "score_unit",
        }
        overlap = sorted(set(self.bucket_cols) & reserved)
        if overlap:
            raise ValueError(f"bucket_cols use reserved output names: {overlap}")
        if not 0.0 <= self.q <= 1.0:
            raise ValueError("q must be between 0 and 1")
        if self.lookback_sessions < 1:
            raise ValueError("lookback_sessions must be positive")
        if not 1 <= self.min_sessions <= self.lookback_sessions:
            raise ValueError(
                "min_sessions must be between 1 and lookback_sessions"
            )
        if self.min_observations < 1:
            raise ValueError("min_observations must be positive")
        if self.interpolation not in {
            "inverted_cdf",
            "averaged_inverted_cdf",
            "closest_observation",
            "interpolated_inverted_cdf",
            "hazen",
            "weibull",
            "linear",
            "median_unbiased",
            "normal_unbiased",
            "lower",
            "higher",
            "midpoint",
            "nearest",
        }:
            raise ValueError(f"unsupported quantile interpolation: {self.interpolation}")


AUDIT_COLUMNS = (
    "history_start",
    "history_end",
    "history_window_sessions",
    "history_sessions",
    "history_observations",
    "quantile",
    "window_basis",
    "score_col",
    "score_unit",
)


def _as_eager(frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    return frame.collect() if isinstance(frame, pl.LazyFrame) else frame


def _calendar_values(sessions: pl.Series | Sequence[object]) -> list[object]:
    values = sessions.to_list() if isinstance(sessions, pl.Series) else list(sessions)
    if not values:
        raise ValueError("sessions cannot be empty")
    if any(value is None for value in values):
        raise ValueError("sessions cannot contain null")
    if len(set(values)) != len(values):
        raise ValueError("sessions must be unique")
    try:
        ordered = sorted(values)
    except TypeError as exc:
        raise ValueError("sessions must have one sortable date type") from exc
    if values != ordered:
        raise ValueError("sessions must be in strictly increasing order")
    return values


def _validate_inputs(
    scores: pl.DataFrame,
    *,
    sessions: list[object],
    spec: PriorQuantileSpec,
    date_col: str,
) -> None:
    required = {date_col, spec.score_col, *spec.bucket_cols}
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"threshold input missing columns: {missing}")
    unknown_dates = set(scores[date_col].drop_nulls().to_list()) - set(sessions)
    if unknown_dates:
        display = sorted(str(value) for value in unknown_dates)
        raise ValueError(f"score dates absent from sessions: {display}")


def _bucket_tuple(row: dict[str, object], bucket_cols: tuple[str, ...]) -> tuple:
    return tuple(row[column] for column in bucket_cols)


def _score_store(
    scores: pl.DataFrame,
    *,
    spec: PriorQuantileSpec,
    date_col: str,
) -> dict[tuple[object, tuple], np.ndarray]:
    valid_bucket = (
        pl.all_horizontal(
            [pl.col(column).is_not_null() for column in spec.bucket_cols]
        )
        if spec.bucket_cols
        else pl.lit(True)
    )
    clean = (
        scores.select(date_col, *spec.bucket_cols, spec.score_col)
        .filter(
            pl.col(date_col).is_not_null()
            & valid_bucket
            & pl.col(spec.score_col)
            .cast(pl.Float64, strict=False)
            .is_finite()
        )
        .with_columns(pl.col(spec.score_col).cast(pl.Float64, strict=False))
        .group_by(date_col, *spec.bucket_cols, maintain_order=True)
        .agg(pl.col(spec.score_col).alias("_values"))
    )
    store: dict[tuple[object, tuple], np.ndarray] = {}
    for row in clean.iter_rows(named=True):
        key = (row[date_col], _bucket_tuple(row, spec.bucket_cols))
        store[key] = np.asarray(row["_values"], dtype=float)
    return store


def _target_buckets(
    scores: pl.DataFrame,
    *,
    sessions: list[object],
    spec: PriorQuantileSpec,
    date_col: str,
) -> dict[object, list[tuple]]:
    if not spec.bucket_cols:
        return {session: [()] for session in sessions}

    valid_bucket = pl.all_horizontal(
        [pl.col(column).is_not_null() for column in spec.bucket_cols]
    )
    distinct = (
        scores.select(date_col, *spec.bucket_cols)
        .filter(pl.col(date_col).is_not_null() & valid_bucket)
        .unique(maintain_order=True)
    )
    output: dict[object, list[tuple]] = {session: [] for session in sessions}
    for row in distinct.iter_rows(named=True):
        output[row[date_col]].append(_bucket_tuple(row, spec.bucket_cols))
    for session in output:
        output[session] = sorted(
            output[session],
            key=lambda values: tuple(map(str, values)),
        )
    return output


def _empty_threshold_table(
    scores: pl.DataFrame,
    *,
    spec: PriorQuantileSpec,
    date_col: str,
) -> pl.DataFrame:
    schema = {date_col: scores.schema[date_col]}
    schema.update({column: scores.schema[column] for column in spec.bucket_cols})
    schema.update(
        {
            spec.threshold_col: pl.Float64,
            "history_start": scores.schema[date_col],
            "history_end": scores.schema[date_col],
            "history_window_sessions": pl.Int64,
            "history_sessions": pl.Int64,
            "history_observations": pl.Int64,
            "quantile": pl.Float64,
            "window_basis": pl.String,
            "score_col": pl.String,
            "score_unit": pl.String,
        }
    )
    return pl.DataFrame(schema=schema)


def build_prior_session_thresholds(
    scores: pl.DataFrame | pl.LazyFrame,
    *,
    sessions: pl.Series | Sequence[object],
    spec: PriorQuantileSpec,
    date_col: str = "Date",
) -> pl.DataFrame:
    """Build one auditable threshold row per target session and bucket.

    Same-session scores are never used.  Bucketed output is generated for the
    buckets present on each target session; unbucketed output covers every
    supplied calendar session.  A missing threshold remains null so applying
    it fails closed.
    """

    eager = _as_eager(scores)
    calendar = _calendar_values(sessions)
    _validate_inputs(eager, sessions=calendar, spec=spec, date_col=date_col)
    store = _score_store(eager, spec=spec, date_col=date_col)
    targets = _target_buckets(
        eager,
        sessions=calendar,
        spec=spec,
        date_col=date_col,
    )
    session_index = {session: index for index, session in enumerate(calendar)}

    observed_by_bucket: dict[tuple, list[object]] = {}
    for session, bucket in store:
        observed_by_bucket.setdefault(bucket, []).append(session)
    for bucket in observed_by_bucket:
        observed_by_bucket[bucket].sort(key=session_index.__getitem__)

    rows: list[dict[str, object]] = []
    for target in calendar:
        target_index = session_index[target]
        for bucket in targets[target]:
            if spec.window_basis == WindowBasis.CALENDAR_SESSIONS:
                window = calendar[
                    max(0, target_index - spec.lookback_sessions):target_index
                ]
            else:
                observed = observed_by_bucket.get(bucket, [])
                prior = [
                    session
                    for session in observed
                    if session_index[session] < target_index
                ]
                window = prior[-spec.lookback_sessions:]

            support = [session for session in window if (session, bucket) in store]
            arrays = [store[(session, bucket)] for session in support]
            observation_count = int(sum(array.size for array in arrays))
            threshold: float | None = None
            if (
                len(support) >= spec.min_sessions
                and observation_count >= spec.min_observations
            ):
                pool = np.concatenate(arrays)
                threshold = float(
                    np.quantile(pool, spec.q, method=spec.interpolation)
                )

            row: dict[str, object] = {
                date_col: target,
                spec.threshold_col: threshold,
                "history_start": window[0] if window else None,
                "history_end": window[-1] if window else None,
                "history_window_sessions": len(window),
                "history_sessions": len(support),
                "history_observations": observation_count,
                "quantile": spec.q,
                "window_basis": spec.window_basis.value,
                "score_col": spec.score_col,
                "score_unit": spec.score_unit,
            }
            row.update(dict(zip(spec.bucket_cols, bucket, strict=True)))
            rows.append(row)

    if not rows:
        return _empty_threshold_table(
            eager,
            spec=spec,
            date_col=date_col,
        )
    order = [date_col, *spec.bucket_cols]
    schema_overrides = {
        date_col: eager.schema[date_col],
        **{column: eager.schema[column] for column in spec.bucket_cols},
        spec.threshold_col: pl.Float64,
        "history_start": eager.schema[date_col],
        "history_end": eager.schema[date_col],
        "history_window_sessions": pl.Int64,
        "history_sessions": pl.Int64,
        "history_observations": pl.Int64,
        "quantile": pl.Float64,
        "window_basis": pl.String,
        "score_col": pl.String,
        "score_unit": pl.String,
    }
    return pl.DataFrame(rows, schema_overrides=schema_overrides).sort(order)


def apply_prior_threshold(
    events: FrameT,
    thresholds: pl.DataFrame | pl.LazyFrame,
    *,
    score_col: str,
    threshold_col: str,
    bucket_cols: tuple[str, ...] = (),
    date_col: str = "Date",
    strict: bool = True,
    output_col: str = "passes_threshold",
) -> FrameT:
    """Join an audit table and apply a null-safe prior threshold gate."""

    threshold_frame = _as_eager(thresholds)
    keys = [date_col, *bucket_cols]
    event_schema = (
        set(events.collect_schema().names())
        if isinstance(events, pl.LazyFrame)
        else set(events.columns)
    )
    missing_events = sorted({date_col, score_col, *bucket_cols} - event_schema)
    if missing_events:
        raise ValueError(
            f"threshold application missing event columns: {missing_events}"
        )
    missing_thresholds = sorted(
        {date_col, threshold_col, *bucket_cols} - set(threshold_frame.columns)
    )
    if missing_thresholds:
        raise ValueError(
            f"threshold application missing threshold columns: {missing_thresholds}"
        )
    if threshold_frame.select(keys).is_duplicated().any():
        raise ValueError(f"threshold table keys are not unique: {keys}")

    joined_columns = set(threshold_frame.columns) - set(keys)
    collisions = sorted(joined_columns & event_schema)
    if output_col in event_schema and output_col not in collisions:
        collisions.append(output_col)
    if collisions:
        raise ValueError(f"threshold join would overwrite columns: {collisions}")

    right = (
        threshold_frame.lazy()
        if isinstance(events, pl.LazyFrame)
        else threshold_frame
    )
    output = events.join(right, on=keys, how="left", validate="m:1")
    score = pl.col(score_col).cast(pl.Float64, strict=False)
    threshold = pl.col(threshold_col).cast(pl.Float64, strict=False)
    comparison = score > threshold if strict else score >= threshold
    passed = (score.is_finite() & threshold.is_finite() & comparison).fill_null(
        False
    )
    return output.with_columns(passed.alias(output_col))
