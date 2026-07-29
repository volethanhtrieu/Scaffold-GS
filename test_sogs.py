#!/usr/bin/env python3
"""Lightweight CPU tests for the optional SOGS integration.

The tests use only small synthetic tensors and never load data, initialize the
renderer extension, or launch training.  Torch-dependent cases are reported as
skipped when the active Python environment does not have PyTorch installed.
"""

import io
import importlib.util
import math
import sys
import tempfile
import types
import unittest
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

from arguments import ModelParams, validate_sogs_config

try:
    import torch

    from utils.loss_utils import selective_gradient_loss
    from utils.sogs_utils import (
        SecondOrderFeatureAugmentor,
        compute_second_order_statistics,
    )

    TORCH_AVAILABLE = True
except (ImportError, OSError):
    torch = None
    TORCH_AVAILABLE = False

def install_extension_import_stubs():
    """Stub unused CUDA/PLY imports so CPU-only model tests can instantiate."""

    if importlib.util.find_spec("torch_scatter") is None:
        module = types.ModuleType("torch_scatter")

        def unavailable_scatter_max(*args, **kwargs):
            raise RuntimeError("torch_scatter is unavailable in this CPU-only test")

        module.scatter_max = unavailable_scatter_max
        sys.modules["torch_scatter"] = module
    if importlib.util.find_spec("plyfile") is None:
        module = types.ModuleType("plyfile")

        class UnavailablePly:
            @staticmethod
            def read(*args, **kwargs):
                raise RuntimeError("plyfile is unavailable in this CPU-only test")

        module.PlyData = UnavailablePly
        module.PlyElement = UnavailablePly
        sys.modules["plyfile"] = module
    if importlib.util.find_spec("simple_knn") is None:
        package = types.ModuleType("simple_knn")
        package.__path__ = []
        extension = types.ModuleType("simple_knn._C")

        def unavailable_distance(*args, **kwargs):
            raise RuntimeError("simple_knn is unavailable in this CPU-only test")

        extension.distCUDA2 = unavailable_distance
        sys.modules["simple_knn"] = package
        sys.modules["simple_knn._C"] = extension
    if importlib.util.find_spec("colorama") is None:
        module = types.ModuleType("colorama")

        class EmptyCodes:
            def __getattr__(self, name):
                return ""

        module.Fore = EmptyCodes()
        module.Style = EmptyCodes()
        module.init = lambda *args, **kwargs: None
        sys.modules["colorama"] = module
    if importlib.util.find_spec("jaxtyping") is None:
        module = types.ModuleType("jaxtyping")

        class Shaped:
            def __class_getitem__(cls, item):
                return item[0] if isinstance(item, tuple) else item

        module.Shaped = Shaped
        sys.modules["jaxtyping"] = module


try:
    if TORCH_AVAILABLE:
        install_extension_import_stubs()
        from scene.gaussian_model import GaussianModel
        MODEL_IMPORT_ERROR = None
    else:
        GaussianModel = None
        MODEL_IMPORT_ERROR = "PyTorch is not installed"
except (ImportError, OSError) as error:
    GaussianModel = None
    MODEL_IMPORT_ERROR = str(error)


def optimization_args():
    """Return the small set of optimizer fields GaussianModel consumes."""

    return SimpleNamespace(
        percent_dense=0.01,
        position_lr_init=0.0,
        position_lr_final=0.0,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=10,
        offset_lr_init=0.01,
        offset_lr_final=0.001,
        offset_lr_delay_mult=0.01,
        offset_lr_max_steps=10,
        feature_lr=0.0075,
        opacity_lr=0.02,
        scaling_lr=0.007,
        rotation_lr=0.002,
        mlp_opacity_lr_init=0.002,
        mlp_opacity_lr_final=0.00002,
        mlp_opacity_lr_delay_mult=0.01,
        mlp_opacity_lr_max_steps=10,
        mlp_cov_lr_init=0.004,
        mlp_cov_lr_final=0.004,
        mlp_cov_lr_delay_mult=0.01,
        mlp_cov_lr_max_steps=10,
        mlp_color_lr_init=0.008,
        mlp_color_lr_final=0.00005,
        mlp_color_lr_delay_mult=0.01,
        mlp_color_lr_max_steps=10,
        mlp_featurebank_lr_init=0.01,
        mlp_featurebank_lr_final=0.00001,
        mlp_featurebank_lr_delay_mult=0.01,
        mlp_featurebank_lr_max_steps=10,
        appearance_lr_init=0.05,
        appearance_lr_final=0.0005,
        appearance_lr_delay_mult=0.01,
        appearance_lr_max_steps=10,
    )


class ConfigurationTests(unittest.TestCase):
    def parse_model(self, values):
        parser = ArgumentParser()
        params = ModelParams(parser)
        parsed = parser.parse_args(["--source_path", "."] + values)
        return params.extract(parsed)

    def test_sogs_enabled(self):
        config = self.parse_model(
            [
                "--use_second_order",
                "True",
                "--feat_dim",
                "16",
                "--num_eigenvectors",
                "2",
                "--lambda_sgl",
                "0.01",
            ]
        )
        self.assertTrue(config.use_second_order)
        self.assertEqual(config.feat_dim, 16)
        self.assertEqual(config.num_eigenvectors, 2)

    def test_sogs_disabled_explicit_false(self):
        config = self.parse_model(["--use_second_order", "False"])
        self.assertFalse(config.use_second_order)

    def test_invalid_feature_dimension(self):
        with self.assertRaisesRegex(ValueError, "feat_dim"):
            validate_sogs_config(0, True, 1, 0.01)

    def test_invalid_eigenvector_count(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            validate_sogs_config(4, True, 0, 0.01)
        with self.assertRaisesRegex(ValueError, "exceed"):
            validate_sogs_config(4, True, 5, 0.01)

    def test_zero_selective_gradient_weight(self):
        values = validate_sogs_config(4, True, 1, 0)
        self.assertEqual(values[-1], 0.0)
        with self.assertRaisesRegex(ValueError, "lambda_sgl"):
            validate_sogs_config(4, True, 1, -0.01)
        with self.assertRaisesRegex(ValueError, "lambda_sgl"):
            validate_sogs_config(4, True, 1, math.nan)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class SecondOrderShapeAndNumericalTests(unittest.TestCase):
    def test_original_and_augmented_feature_dimensions(self):
        features = torch.randn(5, 4)
        self.assertEqual(tuple(features.shape), (5, 4))
        augmentor = SecondOrderFeatureAugmentor(4, 2)
        augmented = augmentor(features)
        self.assertEqual(tuple(augmented.shape), (5, 4 * (1 + 2)))

    def test_random_features_are_finite_and_differentiable(self):
        features = torch.randn(6, 4, requires_grad=True)
        augmentor = SecondOrderFeatureAugmentor(4, 2)
        augmented = augmentor(features)
        augmented.square().mean().backward()
        self.assertTrue(torch.isfinite(augmented).all().item())
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all().item())

    def test_constant_features(self):
        features = torch.ones(4, 3)
        statistics = compute_second_order_statistics(features, 2)
        self.assertTrue(torch.isfinite(statistics.covariance).all().item())
        self.assertTrue(torch.isfinite(statistics.correlation).all().item())
        self.assertTrue(torch.allclose(statistics.covariance, torch.zeros(3, 3)))

    def test_single_anchor(self):
        features = torch.tensor([[1.0, 2.0, 3.0]])
        augmentor = SecondOrderFeatureAugmentor(3, 1)
        augmented = augmentor(features)
        self.assertEqual(tuple(augmented.shape), (1, 6))
        self.assertTrue(torch.isfinite(augmented).all().item())

    def test_half_precision_statistics_and_augmentation(self):
        features = torch.randn(4, 3, dtype=torch.float16)
        augmentor = SecondOrderFeatureAugmentor(3, 1)
        augmented = augmentor(features)
        self.assertEqual(augmented.dtype, torch.float16)
        self.assertTrue(torch.isfinite(augmented).all().item())

    def test_empty_anchor_matrix(self):
        features = torch.empty(0, 3)
        augmentor = SecondOrderFeatureAugmentor(3, 1)
        self.assertEqual(tuple(augmentor(features).shape), (0, 6))

    def test_nonfinite_anchor_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            compute_second_order_statistics(
                torch.tensor([[math.nan, 0.0], [1.0, 2.0]]), 1
            )
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            compute_second_order_statistics(
                torch.tensor([[math.inf, 0.0], [1.0, 2.0]]), 1
            )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class SelectiveGradientLossTests(unittest.TestCase):
    def test_unbatched_and_batched_layouts(self):
        prediction = torch.zeros(3, 8, 8)
        target = torch.zeros_like(prediction)
        self.assertEqual(selective_gradient_loss(prediction, target).ndim, 0)
        self.assertEqual(
            selective_gradient_loss(
                prediction.unsqueeze(0), target.unsqueeze(0)
            ).ndim,
            0,
        )

    def test_half_precision_input(self):
        prediction = torch.zeros(3, 8, 8, dtype=torch.float16)
        target = prediction.clone()
        target[:, :, 4:] = 1
        loss = selective_gradient_loss(prediction, target)
        self.assertTrue(torch.isfinite(loss).item())

    def test_identical_and_constant_images(self):
        image = torch.full((3, 8, 8), 0.5)
        loss = selective_gradient_loss(image, image.clone())
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(torch.isfinite(loss).item())

    def test_sharp_synthetic_edge(self):
        prediction = torch.zeros(3, 8, 8)
        target = torch.zeros_like(prediction)
        target[:, :, 4:] = 1.0
        self.assertGreater(
            selective_gradient_loss(prediction, target).item(), 0.0
        )

    def test_shape_and_nonfinite_validation(self):
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            selective_gradient_loss(
                torch.zeros(3, 8, 8), torch.zeros(3, 7, 8)
            )
        bad = torch.zeros(3, 8, 8)
        bad[0, 0, 0] = math.nan
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            selective_gradient_loss(bad, torch.zeros_like(bad))
        bad = torch.zeros(3, 8, 8)
        bad[0, 0, 0] = math.inf
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            selective_gradient_loss(bad, torch.zeros_like(bad))


@unittest.skipUnless(
    TORCH_AVAILABLE and GaussianModel is not None,
    "GaussianModel dependencies are not installed: " + str(MODEL_IMPORT_ERROR),
)
class ModelCompatibilityTests(unittest.TestCase):
    def make_model(self, enabled):
        model = GaussianModel(
            feat_dim=4,
            n_offsets=2,
            appearance_dim=0,
            use_second_order=enabled,
            num_eigenvectors=2,
            lambda_sgl=0.01,
            device="cpu",
        )
        model._anchor = torch.nn.Parameter(torch.randn(3, 3))
        model._offset = torch.nn.Parameter(torch.zeros(3, 2, 3))
        model._anchor_feat = torch.nn.Parameter(torch.randn(3, 4))
        model._scaling = torch.nn.Parameter(torch.zeros(3, 6))
        model._rotation = torch.nn.Parameter(torch.randn(3, 4))
        model._opacity = torch.nn.Parameter(torch.zeros(3, 1))
        model.max_radii2D = torch.zeros(3)
        return model

    def test_renderer_mlp_input_dimensions(self):
        baseline = self.make_model(False)
        sogs = self.make_model(True)
        self.assertEqual(baseline.mlp_opacity[0].in_features, 4 + 3)
        self.assertEqual(sogs.mlp_opacity[0].in_features, 12 + 3)
        self.assertEqual(sogs.mlp_cov[0].in_features, 12 + 3)
        self.assertEqual(sogs.mlp_color[0].in_features, 12 + 3)
        sogs_with_extras = GaussianModel(
            feat_dim=4,
            n_offsets=2,
            appearance_dim=5,
            add_opacity_dist=True,
            add_cov_dist=True,
            add_color_dist=True,
            use_second_order=True,
            num_eigenvectors=2,
            device="cpu",
        )
        self.assertEqual(sogs_with_extras.mlp_opacity[0].in_features, 12 + 4)
        self.assertEqual(sogs_with_extras.mlp_cov[0].in_features, 12 + 4)
        self.assertEqual(
            sogs_with_extras.mlp_color[0].in_features, 12 + 4 + 5
        )

    def test_baseline_does_not_call_augmentor(self):
        baseline = self.make_model(False)
        self.assertIsNone(baseline.second_order_augmentor)
        self.assertIs(baseline.get_render_features(), baseline._anchor_feat)

    def test_optimizer_contains_second_order_parameters(self):
        model = self.make_model(True)
        model.training_setup(optimization_args())
        groups = {group["name"]: group for group in model.optimizer.param_groups}
        self.assertIn("mlp_second_order", groups)
        expected = {id(parameter) for parameter in model.second_order_augmentor.parameters()}
        actual = {id(parameter) for parameter in groups["mlp_second_order"]["params"]}
        self.assertEqual(actual, expected)

    def test_capture_restore_and_config_round_trip(self):
        model = self.make_model(True)
        model.training_setup(optimization_args())
        state = model.capture()
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, map_location="cpu")
        restored = self.make_model(True)
        restored.restore(loaded, optimization_args())
        self.assertEqual(restored.get_sogs_config(), model.get_sogs_config())
        self.assertTrue(
            torch.allclose(restored.get_render_features(), model.get_render_features())
        )

    def test_split_mlp_save_load_round_trip(self):
        model = self.make_model(True)
        with tempfile.TemporaryDirectory() as directory:
            model.save_mlp_checkpoints(directory)
            self.assertTrue(
                (Path(directory) / "sogs_config.json").is_file()
            )
            self.assertTrue(
                (Path(directory) / "second_order_mlp_0.pt").is_file()
            )
            restored = self.make_model(True)
            restored._anchor_feat = torch.nn.Parameter(
                model._anchor_feat.detach().clone()
            )
            restored.load_mlp_checkpoints(directory)
            self.assertTrue(
                torch.allclose(
                    restored.get_render_features(),
                    model.get_render_features(),
                )
            )

    def test_united_mlp_save_load_round_trip(self):
        model = self.make_model(True)
        with tempfile.TemporaryDirectory() as directory:
            model.save_mlp_checkpoints(directory, mode="unite")
            restored = self.make_model(True)
            restored._anchor_feat = torch.nn.Parameter(
                model._anchor_feat.detach().clone()
            )
            restored.load_mlp_checkpoints(directory, mode="unite")
            self.assertTrue(
                torch.allclose(
                    restored.get_render_features(),
                    model.get_render_features(),
                )
            )

    def test_sogs_rejects_missing_metadata(self):
        model = self.make_model(True)
        with tempfile.TemporaryDirectory() as directory:
            model.save_mlp_checkpoints(directory)
            (Path(directory) / "sogs_config.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "missing SOGS metadata"):
                self.make_model(True).load_mlp_checkpoints(directory)


class StaticIntegrationTests(unittest.TestCase):
    def test_renderer_uses_single_model_feature_accessor(self):
        root = Path(__file__).resolve().parent
        renderer = (root / "gaussian_renderer" / "__init__.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("pc.get_render_features()", renderer)
        self.assertNotIn("feat = pc._anchor_feat[visible_mask]", renderer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
