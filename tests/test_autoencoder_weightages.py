import unittest

import importlib.util

HAS_DEPS = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in ("numpy", "torch", "pandas", "sklearn")
)

if HAS_DEPS:
    from autoencoder_weightages import AutoEncoder, _feature_weightages, _validate_splits
else:
    AutoEncoder = None
    _feature_weightages = None
    _validate_splits = None


class FeatureWeightageTests(unittest.TestCase):
    def test_feature_weightages_sorted_descending(self) -> None:
        if not HAS_DEPS:
            self.skipTest("Required ML dependencies are not available")
        import torch

        model = AutoEncoder(input_dim=3, latent_dim=2)
        with torch.no_grad():
            model.encoder[0].weight.copy_(
                torch.tensor([[1.0, 3.0, 0.0], [2.0, 1.0, 0.0]])
            )
        weightages = _feature_weightages(model, ["a", "b", "c"])
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
