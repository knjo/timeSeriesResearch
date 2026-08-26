from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest

import src.research.timeSeries.framework as framework
import src.research.timeSeries.framework.profiles.negfill as negfill_profile
from src.research.timeSeries.framework.features import FeatureManifest, FeatureSpec
from src.research.timeSeries.framework.profiles.negfill import NEGFILL_V1


class FeatureManifestTest(unittest.TestCase):
    def test_generic_package_import_does_not_load_rolling_engine(self) -> None:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import src.research.timeSeries.framework; "
                "print('src.research.rolling_model' in sys.modules)",
            ],
            cwd=Path(__file__).resolve().parents[4],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(probe.stdout.strip(), "False")

    def test_generic_package_root_does_not_export_reference_profiles(self) -> None:
        self.assertFalse(hasattr(framework, "NEGFILL_V1"))
        self.assertFalse(
            hasattr(framework, "NEXTDAY_ALLOW_DAY_TRADE_UNIVERSE_V1")
        )

    def test_negfill_profile_exports_only_negfill_reference_family(self) -> None:
        self.assertFalse(hasattr(negfill_profile, "NEGFILL_V1_FEATURE_MANIFEST"))
        self.assertFalse(hasattr(negfill_profile, "LONG_TAKER_TP109_V1"))
        self.assertFalse(hasattr(negfill_profile, "LONG_MAKER_TP109_V1"))
        self.assertFalse(
            hasattr(negfill_profile, "NEXTDAY_ALLOW_DAY_TRADE_UNIVERSE_V1")
        )

    def test_negfill_v1_is_an_explicit_ordered_30_feature_contract(self) -> None:
        expected = (
            "ToLow",
            "ToHigh",
            "Low_High",
            "RemainSeconds",
            "ToOpen",
            "ToRef",
            "B1_A1B1",
            "B1_B1B5",
            "A1_A1A5",
            "B2_Last",
            "A2_Last",
            "B45_AB45",
            "B1_B12",
            "A1_A12",
            "QL_BidHHI",
            "QL_AskHHI",
            "LOB_BidVelocity_30_re",
            "LOB_AskVelocity_30_re",
            "FillLots_atLow_re",
            "FillLots_atHigh_re",
            "L1_SellBiggestLots_30_re",
            "L1_BuyBiggestLots_30_re",
            "MD_ElaspeTime_30_re",
            "MD_L1Rate_30_re",
            "IA_BuyImpact_30_re",
            "IA_SellImpact_30_re",
            "LOB_NetPressure_30_re",
            "QL_Asymmetry_re",
            "txf_residual_vol_to_1330",
            "txf_residual_vol_0900_1330_60d",
        )
        self.assertEqual(NEGFILL_V1.ordered_names, expected)
        self.assertEqual(len(NEGFILL_V1.features), 30)
        self.assertNotIn("txf_to_1330_return", NEGFILL_V1.ordered_names)

    def test_hash_is_stable_but_feature_order_is_material(self) -> None:
        specs = (
            FeatureSpec("alpha", role="tick"),
            FeatureSpec("regime", role="forced", availability="pre_market"),
        )
        first = FeatureManifest("example", "1", specs)
        identical = FeatureManifest("example", "1", tuple(specs))
        reversed_order = FeatureManifest("example", "1", tuple(reversed(specs)))

        self.assertEqual(first.stable_hash(), identical.stable_hash())
        self.assertEqual(first.fingerprint, identical.fingerprint)
        self.assertNotEqual(first.stable_hash(), reversed_order.stable_hash())

    def test_duplicate_and_forbidden_features_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate feature"):
            FeatureManifest(
                "duplicate",
                "1",
                (FeatureSpec("x"), FeatureSpec("x")),
            )
        with self.assertRaisesRegex(ValueError, "Future/outcome"):
            FeatureManifest(
                "leaky",
                "1",
                (FeatureSpec("txf_to_1330_return"),),
            )

    def test_required_columns_are_checked_but_extra_columns_are_ignored(self) -> None:
        manifest = FeatureManifest("required", "1", (FeatureSpec("x"),))
        manifest.validate_columns(["x", "unused_future_schema_addition_re"])
        with self.assertRaisesRegex(ValueError, "missing required features"):
            manifest.validate_columns(["unused"])

    def test_mask_missing_policy_is_role_safe(self) -> None:
        self.assertEqual(FeatureSpec("gate", role="mask").effective_missing_policy, "zero")
        with self.assertRaisesRegex(ValueError, "Mask feature"):
            FeatureSpec("gate", role="mask", missing_policy="prior_median")


if __name__ == "__main__":
    unittest.main()
