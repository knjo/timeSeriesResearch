from __future__ import annotations

import unittest

import polars as pl
from polars.testing import assert_frame_equal

from src.research.timeSeries.framework.prior_thresholds import (
    PriorQuantileSpec,
    WindowBasis,
    apply_prior_threshold,
    build_prior_session_thresholds,
)


DATES = ["20260102", "20260105", "20260106", "20260107"]


def _sparse_bucket_scores() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Date": [DATES[0], DATES[1], DATES[2], DATES[3]],
            "bucket": ["A", "B", "A", "A"],
            "score": [0.0, 50.0, 10.0, 100.0],
        }
    )


def _row(
    frame: pl.DataFrame,
    date: str,
    bucket: str,
) -> dict[str, object]:
    return frame.filter(
        (pl.col("Date") == date) & (pl.col("bucket") == bucket)
    ).to_dicts()[0]


class PriorThresholdsTest(unittest.TestCase):
    def test_calendar_window_uses_global_sessions_and_excludes_today(self) -> None:
        spec = PriorQuantileSpec(
            score_col="score",
            threshold_col="threshold",
            q=0.5,
            bucket_cols=("bucket",),
            lookback_sessions=2,
            window_basis=WindowBasis.CALENDAR_SESSIONS,
        )
        result = build_prior_session_thresholds(
            _sparse_bucket_scores(), sessions=DATES, spec=spec
        )
        audit = _row(result, DATES[3], "A")

        # The window is d2,d3; only d3 contains A.  Today's score=100 is not
        # allowed to affect its own threshold.
        self.assertAlmostEqual(audit["threshold"], 10.0)
        self.assertEqual(audit["history_start"], DATES[1])
        self.assertEqual(audit["history_end"], DATES[2])
        self.assertEqual(audit["history_window_sessions"], 2)
        self.assertEqual(audit["history_sessions"], 1)
        self.assertEqual(audit["history_observations"], 1)
        self.assertEqual(audit["window_basis"], "calendar_sessions")

    def test_bucket_observation_window_reproduces_sparse_notebook_semantics(self) -> None:
        spec = PriorQuantileSpec(
            score_col="score",
            threshold_col="threshold",
            q=0.5,
            bucket_cols=("bucket",),
            lookback_sessions=2,
            window_basis=WindowBasis.BUCKET_OBSERVATION_SESSIONS,
        )
        result = build_prior_session_thresholds(
            _sparse_bucket_scores(), sessions=DATES, spec=spec
        )
        audit = _row(result, DATES[3], "A")

        # A appeared on d1 and d3, so the legacy window reaches through d2.
        self.assertAlmostEqual(audit["threshold"], 5.0)
        self.assertEqual(audit["history_start"], DATES[0])
        self.assertEqual(audit["history_end"], DATES[2])
        self.assertEqual(audit["history_window_sessions"], 2)
        self.assertEqual(audit["history_sessions"], 2)
        self.assertEqual(audit["window_basis"], "bucket_observation_sessions")

    def test_minimum_support_fails_closed_with_auditable_counts(self) -> None:
        spec = PriorQuantileSpec(
            score_col="score",
            threshold_col="threshold",
            q=0.5,
            bucket_cols=("bucket",),
            lookback_sessions=2,
            min_sessions=2,
            window_basis=WindowBasis.CALENDAR_SESSIONS,
        )
        result = build_prior_session_thresholds(
            _sparse_bucket_scores(), sessions=DATES, spec=spec
        )
        audit = _row(result, DATES[3], "A")
        self.assertIsNone(audit["threshold"])
        self.assertEqual(audit["history_sessions"], 1)
        self.assertEqual(audit["history_observations"], 1)

    def test_linear_quantile_method_is_pinned(self) -> None:
        scores = pl.DataFrame(
            {
                "Date": [DATES[0], DATES[0], DATES[1]],
                "bucket": ["A", "A", "A"],
                "score": [0.0, 10.0, 999.0],
            }
        )
        spec = PriorQuantileSpec(
            score_col="score",
            threshold_col="threshold",
            q=0.25,
            bucket_cols=("bucket",),
            lookback_sessions=1,
            interpolation="linear",
        )
        result = build_prior_session_thresholds(
            scores, sessions=DATES[:2], spec=spec
        )
        self.assertAlmostEqual(_row(result, DATES[1], "A")["threshold"], 2.5)

    def test_bucket_values_never_cross_contaminate(self) -> None:
        scores = pl.DataFrame(
            {
                "Date": [DATES[0], DATES[0], DATES[1], DATES[1]],
                "bucket": ["A", "B", "A", "B"],
                "score": [1.0, 100.0, 999.0, 999.0],
            }
        )
        spec = PriorQuantileSpec(
            score_col="score",
            threshold_col="threshold",
            q=0.5,
            bucket_cols=("bucket",),
            lookback_sessions=1,
        )
        result = build_prior_session_thresholds(
            scores, sessions=DATES[:2], spec=spec
        )
        self.assertAlmostEqual(_row(result, DATES[1], "A")["threshold"], 1.0)
        self.assertAlmostEqual(_row(result, DATES[1], "B")["threshold"], 100.0)

    def test_appending_future_data_does_not_change_prior_thresholds(self) -> None:
        base_scores = pl.DataFrame(
            {
                "Date": DATES,
                "score": [1.0, 2.0, 3.0, 4.0],
            }
        )
        future_date = "20260108"
        extended_scores = pl.concat(
            [
                base_scores,
                pl.DataFrame({"Date": [future_date], "score": [1_000_000.0]}),
            ]
        )
        spec = PriorQuantileSpec(
            score_col="score",
            threshold_col="threshold",
            q=0.5,
            lookback_sessions=3,
        )
        base = build_prior_session_thresholds(
            base_scores, sessions=DATES, spec=spec
        )
        extended = build_prior_session_thresholds(
            extended_scores, sessions=[*DATES, future_date], spec=spec
        ).filter(pl.col("Date").is_in(DATES))
        assert_frame_equal(base, extended)

    def test_strict_apply_fails_on_equal_or_missing_threshold(self) -> None:
        spec = PriorQuantileSpec(
            score_col="score",
            threshold_col="threshold",
            q=0.5,
            bucket_cols=("bucket",),
            lookback_sessions=2,
        )
        thresholds = build_prior_session_thresholds(
            _sparse_bucket_scores(), sessions=DATES, spec=spec
        )
        events = pl.DataFrame(
            {
                "Date": [DATES[3], DATES[3], DATES[3]],
                "bucket": ["A", "A", "C"],
                "event_score": [10.0, 11.0, 1_000.0],
            }
        )
        result = apply_prior_threshold(
            events,
            thresholds,
            score_col="event_score",
            threshold_col="threshold",
            bucket_cols=("bucket",),
            strict=True,
        )
        self.assertEqual(
            result["passes_threshold"].to_list(), [False, True, False]
        )
        self.assertIsNone(result["threshold"][2])

        non_strict = apply_prior_threshold(
            events.head(1).lazy(),
            thresholds,
            score_col="event_score",
            threshold_col="threshold",
            bucket_cols=("bucket",),
            strict=False,
        ).collect()
        self.assertTrue(non_strict["passes_threshold"][0])

    def test_duplicate_threshold_keys_are_rejected(self) -> None:
        thresholds = pl.DataFrame(
            {
                "Date": [DATES[0], DATES[0]],
                "threshold": [1.0, 2.0],
            }
        )
        events = pl.DataFrame({"Date": [DATES[0]], "score": [3.0]})
        with self.assertRaisesRegex(ValueError, "keys are not unique"):
            apply_prior_threshold(
                events,
                thresholds,
                score_col="score",
                threshold_col="threshold",
            )

    def test_calendar_validation_rejects_unknown_or_unsorted_dates(self) -> None:
        scores = pl.DataFrame({"Date": [DATES[0]], "score": [1.0]})
        spec = PriorQuantileSpec("score", "threshold", 0.5)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            build_prior_session_thresholds(
                scores, sessions=[DATES[1], DATES[0]], spec=spec
            )
        with self.assertRaisesRegex(ValueError, "absent from sessions"):
            build_prior_session_thresholds(
                scores, sessions=[DATES[1]], spec=spec
            )


if __name__ == "__main__":
    unittest.main()
