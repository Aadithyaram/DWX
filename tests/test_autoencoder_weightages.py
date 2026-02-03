import unittest

import importlib.util

HAS_DEPS = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in ("numpy", "torch", "pandas", "sklearn")
)

if HAS_DEPS:
    import numpy as np

    from autoencoder_weightages import _normalize_importances, _validate_splits
else:
    np = None
    _normalize_importances = None
    _validate_splits = None


class FeatureWeightageTests(unittest.TestCase):
    def test_feature_weightages_sorted_descending(self) -> None:
        if not HAS_DEPS:
            self.skipTest("Required ML dependencies are not available")
        importances = np.array([0.2, 0.6, 0.1])
        weightages = _normalize_importances(importances, ["a", "b", "c"])
        self.assertEqual(list(weightages.keys()), ["b", "a", "c"])
        self.assertGreaterEqual(weightages["b"], weightages["a"])
        self.assertGreaterEqual(weightages["a"], weightages["c"])

    def test_validation_and_test_split_guardrails(self) -> None:
        if not HAS_DEPS:
            self.skipTest("Required ML dependencies are not available")
        with self.assertRaises(ValueError):
            _validate_splits(0.6, 0.5)


if __name__ == "__main__":
    unittest.main()
