from __future__ import annotations

import unittest

import numpy as np

from src.research.rolling_model import _make_linear_model


class LinearModelFamilyTest(unittest.TestCase):
    def test_ridge_l2_equivalent_elastic_alpha(self) -> None:
        model, effective = _make_linear_model(
            model_family="elastic_net",
            alpha=0.1,
            n_samples=100,
            elastic_net_l1_ratio=0.5,
            elastic_net_alpha_mode="ridge_l2_equivalent",
            max_iter=1000,
            tol=1e-4,
        )
        self.assertAlmostEqual(effective, 0.002)
        self.assertAlmostEqual(model.alpha, effective)

    def test_elastic_net_fit_is_deterministic_and_sparse(self) -> None:
        x = np.column_stack(
            [np.arange(30.0), np.zeros(30), np.ones(30)]
        )
        y = 0.5 * x[:, 0]
        kwargs = dict(
            model_family="elastic_net",
            alpha=0.1,
            n_samples=len(x),
            elastic_net_l1_ratio=0.5,
            elastic_net_alpha_mode="ridge_l2_equivalent",
            max_iter=10000,
            tol=1e-8,
        )
        first, _ = _make_linear_model(**kwargs)
        second, _ = _make_linear_model(**kwargs)
        first.fit(x, y)
        second.fit(x, y)
        np.testing.assert_allclose(first.coef_, second.coef_, rtol=0.0, atol=0.0)
        self.assertGreaterEqual(np.count_nonzero(first.coef_ == 0.0), 2)

    def test_invalid_family_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "model_family"):
            _make_linear_model(
                model_family="lasso",
                alpha=0.1,
                n_samples=10,
                elastic_net_l1_ratio=0.5,
                elastic_net_alpha_mode="fixed",
                max_iter=100,
                tol=1e-4,
            )


if __name__ == "__main__":
    unittest.main()
