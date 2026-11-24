"""Small reproducibility checks; no instructor data required."""
import unittest
import numpy as np
import pandas as pd
from src.preprocessing import fit_preprocessor, transform_features


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({"tlmta": [0.1, 0.2, 0.4, 0.8],
                                   "sigma": [0.2, np.nan, 0.6, np.inf],
                                   "cash_at": [0.1, 0.5, np.nan, 0.3]})
        self.params = fit_preprocessor(self.train, ["tlmta", "sigma"], ["cash_at"])

    def test_fixed_bounds_and_imputation(self):
        validation = pd.DataFrame({"tlmta": [-100, 100], "sigma": [np.nan, 500],
                                   "cash_at": [np.inf, -100]}, index=[7, 9])
        before = self.params.copy(deep=True)
        x = transform_features(validation, self.params)
        pd.testing.assert_frame_equal(self.params, before)
        self.assertEqual(x.index.tolist(), [7, 9])
        self.assertTrue(np.isfinite(x.to_numpy()).all())
        for f in x:
            self.assertTrue(x[f].between(self.params.loc[f, "lower"], self.params.loc[f, "upper"]).all())
        self.assertAlmostEqual(x.loc[7, "sigma"], 0.4)
        self.assertAlmostEqual(x.loc[7, "cash_at"], 0.3)

    def test_required_feature_missing_fails(self):
        invalid = self.train.copy()
        invalid.loc[0, "tlmta"] = np.nan
        with self.assertRaises(ValueError):
            transform_features(invalid, self.params)
        with self.assertRaises(ValueError):
            fit_preprocessor(invalid, ["tlmta", "sigma"], ["cash_at"])

    def test_no_finite_training_data_fails(self):
        invalid = self.train.assign(cash_at=np.nan)
        with self.assertRaises(ValueError):
            fit_preprocessor(invalid, ["tlmta", "sigma"], ["cash_at"])


if __name__ == "__main__":
    unittest.main()
