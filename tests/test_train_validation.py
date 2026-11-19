"""Offline schema and numeric-validation regression tests."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train import ANONYMIZED_FEATURES, SCALED_FEATURES, TARGET, load_data


class LoadDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "transactions.csv"
        self.frame = pd.DataFrame(
            {name: [1.0, 2.0] for name in SCALED_FEATURES + ANONYMIZED_FEATURES}
        )
        self.frame[TARGET] = [0, 1]

    def load(self):
        self.frame.to_csv(self.path, index=False)
        return load_data(self.path)

    def test_valid_numeric_features_and_binary_target(self):
        features, labels = self.load()
        self.assertEqual(features.shape, (2, 30))
        self.assertEqual(labels.tolist(), [0, 1])

    def test_rejects_infinite_values(self):
        for value in (np.inf, -np.inf):
            with self.subTest(value=value):
                self.frame.loc[0, "Amount"] = value
                with self.assertRaisesRegex(ValueError, "Amount.*finite numeric"):
                    self.load()

    def test_rejects_non_numeric_feature(self):
        self.frame.loc[0, "V1"] = "not-a-number"
        with self.assertRaisesRegex(ValueError, "V1.*finite numeric"):
            self.load()

    def test_rejects_missing_feature(self):
        self.frame = self.frame.drop(columns=["V2"])
        with self.assertRaisesRegex(ValueError, "Missing expected columns"):
            self.load()

    def test_rejects_missing_value(self):
        self.frame.loc[0, "Time"] = np.nan
        with self.assertRaisesRegex(ValueError, "missing values"):
            self.load()

    def test_rejects_invalid_class(self):
        self.frame.loc[1, TARGET] = 2
        with self.assertRaisesRegex(ValueError, "only zero and one"):
            self.load()


if __name__ == "__main__":
    unittest.main()
