from __future__ import annotations

import builtins
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import src.research.rolling_model as rolling_model
import src.research.intradaySignal.framework.model as framework_model
from src.research.rolling_model import train_and_predict_ridge_rolling
from src.research.intradaySignal.framework.features import (
    FeatureManifest,
    FeatureSpec,
    prepare_feature_frame,
)


class CausalFeaturePreparationTest(unittest.TestCase):
    def test_prior_median_uses_last_finite_strictly_prior_day(self) -> None:
        frame = pd.DataFrame(
            {
                "Date": [
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-05",
                ],
                "x": [np.nan, 2.0, np.nan, np.nan, np.inf, np.nan, 8.0, np.nan],
            }
        )
        manifest = FeatureManifest(
            "causal",
            "1",
            (FeatureSpec("x", fallback=-1.0),),
        )

        result = prepare_feature_frame(frame, manifest)

        # First date has no history.  Consecutive empty dates keep using the
        # latest finite daily median (2.0), never their own cross-section.
        self.assertEqual(
            result["x"].tolist(),
            [-1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 8.0, 2.0],
        )

    def test_appending_extreme_future_rows_cannot_rewrite_prefix(self) -> None:
        prefix = pd.DataFrame(
            {
                "Date": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
                "x": [np.nan, 2.0, np.nan, np.nan],
            }
        )
        future = pd.DataFrame(
            {
                "Date": ["2024-01-04", "2024-01-04"],
                "x": [1.0e12, -1.0e12],
            }
        )
        manifest = FeatureManifest("causal", "1", (FeatureSpec("x"),))

        prepared_prefix = prepare_feature_frame(prefix, manifest)
        prepared_full = prepare_feature_frame(
            pd.concat([prefix, future], ignore_index=True), manifest
        )

        pd.testing.assert_series_equal(
            prepared_prefix["x"],
            prepared_full.loc[: len(prefix) - 1, "x"],
        )

    def test_mask_is_filled_with_zero_and_must_remain_binary(self) -> None:
        manifest = FeatureManifest(
            "mask",
            "1",
            (FeatureSpec("gate", role="mask"),),
        )
        valid = prepare_feature_frame(
            pd.DataFrame({"Date": ["2024-01-02"] * 3, "gate": [1.0, np.nan, 0.0]}),
            manifest,
        )
        self.assertEqual(valid["gate"].tolist(), [1.0, 0.0, 0.0])

        with self.assertRaisesRegex(ValueError, "must be binary"):
            prepare_feature_frame(
                pd.DataFrame({"Date": ["2024-01-02"], "gate": [0.5]}),
                manifest,
            )

    def test_missing_required_feature_fails_before_model_fit(self) -> None:
        manifest = FeatureManifest("required", "1", (FeatureSpec("x"),))
        with self.assertRaisesRegex(ValueError, "missing required features"):
            prepare_feature_frame(pd.DataFrame({"Date": ["2024-01-02"]}), manifest)


def _rolling_fixture(include_future: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2024-01-02", periods=6 if include_future else 5, freq="D")
    for day_idx, date in enumerate(dates):
        for event_idx in range(4):
            is_future = day_idx == 5
            x = (
                (1.0e9 if event_idx % 2 == 0 else -1.0e9)
                if is_future
                else float(day_idx * 2 + event_idx)
            )
            if event_idx == 0 and day_idx in {0, 2}:
                x = np.nan
            rows.append(
                {
                    "Date": date,
                    "QuoteCode": "A" if event_idx < 2 else "B",
                    "TransTime": date + pd.Timedelta(hours=9, seconds=event_idx),
                    "ChannelSeq": event_idx,
                    "x": x,
                    "gate": np.nan if event_idx == 1 else float(event_idx % 2),
                    "target": float((event_idx - 1.5) * (day_idx + 1)),
                }
            )
    return pd.DataFrame(rows)


class RollingCausalV2IntegrationTest(unittest.TestCase):
    def _run(self, frame: pd.DataFrame) -> dict[str, object]:
        return train_and_predict_ridge_rolling(
            signal_df=frame,
            X_list=["x"],
            Y_col="target",
            forced_masks=["gate"],
            n_training_days=3,
            n_jobs=1,
            feature_screening=False,
            alpha=1.0,
            max_position_per_stock=-1,
            preprocessing_mode="causal_v2",
        )

    def test_model_predictions_are_append_future_invariant(self) -> None:
        prefix_result = self._run(_rolling_fixture(include_future=False))
        full_result = self._run(_rolling_fixture(include_future=True))
        prefix_df = prefix_result["df"]
        full_prefix = full_result["df"].loc[
            full_result["df"]["Date"] <= pd.Timestamp("2024-01-06")
        ]

        key = ["Date", "QuoteCode", "TransTime", "ChannelSeq"]
        pd.testing.assert_frame_equal(
            prefix_df[key].reset_index(drop=True),
            full_prefix[key].reset_index(drop=True),
        )
        np.testing.assert_allclose(
            prefix_df["target_pred"].to_numpy(),
            full_prefix["target_pred"].to_numpy(),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )
        self.assertEqual(prefix_result["feature_names"], full_result["feature_names"])

    def test_direct_causal_mode_rejects_missing_and_nonbinary_masks(self) -> None:
        frame = _rolling_fixture(include_future=False)
        with self.assertRaisesRegex(ValueError, "missing required features"):
            train_and_predict_ridge_rolling(
                frame.drop(columns="x"),
                ["x"],
                "target",
                n_jobs=1,
                preprocessing_mode="causal_v2",
            )

        bad_mask = frame.copy()
        bad_mask.loc[0, "gate"] = 0.25
        with self.assertRaisesRegex(ValueError, "must be binary"):
            self._run(bad_mask)

    def test_direct_causal_mode_does_not_import_framework_layers(self) -> None:
        original_import = builtins.__import__

        def reject_framework_import(name, *args, **kwargs):
            if "timeSeries.framework" in name:
                raise AssertionError(f"rolling engine imported framework layer: {name}")
            return original_import(name, *args, **kwargs)

        with mock.patch.object(
            builtins,
            "__import__",
            side_effect=reject_framework_import,
        ):
            result = self._run(_rolling_fixture(include_future=False))

        self.assertIn("target_pred", result["df"].columns)

    def test_legacy_remains_default_and_mode_validation_is_explicit(self) -> None:
        default = train_and_predict_ridge_rolling.__defaults__
        self.assertIsNotNone(default)
        self.assertEqual(default[-1], "legacy_v1")
        with self.assertRaisesRegex(ValueError, "preprocessing_mode"):
            train_and_predict_ridge_rolling(
                _rolling_fixture(include_future=False),
                ["x"],
                "target",
                preprocessing_mode="unknown",
            )


class FrameworkCausalV2IntegrationTest(unittest.TestCase):
    def test_wrapper_prepares_once_with_manifest_policies_and_fallbacks(self) -> None:
        frame = _rolling_fixture(include_future=False)
        manifest = FeatureManifest(
            "custom-policies",
            "1",
            (
                FeatureSpec(
                    "x",
                    missing_policy="constant",
                    fallback=123.0,
                ),
                FeatureSpec(
                    "optional",
                    required=False,
                    missing_policy="constant",
                    fallback=-7.0,
                ),
                FeatureSpec("gate", role="mask"),
            ),
        )

        with mock.patch.object(
            framework_model,
            "prepare_feature_frame",
            wraps=prepare_feature_frame,
        ) as prepare:
            result = framework_model.train_and_predict_causal_v2(
                frame,
                manifest,
                "target",
                n_training_days=3,
                n_jobs=1,
                feature_screening=False,
                alpha=1.0,
                max_position_per_stock=-1,
            )

        prepare.assert_called_once()
        self.assertIs(prepare.call_args.args[1], manifest)

        prepared = result["df"]
        missing_x_keys = frame.loc[frame["x"].isna(), [
            "Date",
            "QuoteCode",
            "TransTime",
            "ChannelSeq",
        ]]
        filled_x = missing_x_keys.merge(
            prepared,
            on=["Date", "QuoteCode", "TransTime", "ChannelSeq"],
            how="left",
            validate="one_to_one",
        )["x"]
        np.testing.assert_allclose(filled_x.to_numpy(), 123.0)
        np.testing.assert_allclose(prepared["optional"].to_numpy(), -7.0)
        self.assertTrue((prepared.loc[prepared["ChannelSeq"] == 1, "gate"] == 0.0).all())

    def test_wrapper_preserves_manifest_error_policy(self) -> None:
        manifest = FeatureManifest(
            "strict",
            "1",
            (FeatureSpec("x", missing_policy="error"),),
        )
        with self.assertRaisesRegex(ValueError, "contains missing values"):
            framework_model.train_and_predict_causal_v2(
                _rolling_fixture(include_future=False),
                manifest,
                "target",
                n_training_days=3,
                n_jobs=1,
                feature_screening=False,
                alpha=1.0,
                max_position_per_stock=-1,
            )


if __name__ == "__main__":
    unittest.main()
