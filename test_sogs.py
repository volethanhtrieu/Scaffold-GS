#!/usr/bin/env python3
"""Lightweight CPU tests for the optional SOGS integration.

The tests use only small synthetic tensors and never load data, initialize the
renderer extension, or launch training.  Torch-dependent cases are reported as
skipped when the active Python environment does not have PyTorch installed.
"""

import io
import importlib.util
import math
import subprocess
import sys
import tempfile
import types
import unittest
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

from arguments import (
    ModelParams,
    validate_optimizer_backend,
    validate_positive_int,
    validate_sogs_chunk_size,
    validate_sogs_config,
    validate_tf32_mode,
)
from utils.training_budget import (
    TrainingBudget,
    should_checkpoint_iteration,
    validate_checkpoint_interval,
    validate_max_runtime_minutes,
)

try:
    import torch

    from utils.loss_utils import (
        _sobel_gradient_maps,
        scaling_volume_regularization,
        selective_gradient_loss,
        ssim,
    )
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
    if importlib.util.find_spec("diff_gaussian_rasterization") is None:
        module = types.ModuleType("diff_gaussian_rasterization")

        class UnavailableRasterizer:
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    "the CUDA rasterizer is unavailable in this CPU-only test"
                )

        module.GaussianRasterizationSettings = UnavailableRasterizer
        module.GaussianRasterizer = UnavailableRasterizer
        sys.modules["diff_gaussian_rasterization"] = module
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
        from gaussian_renderer import generate_neural_gaussians
        MODEL_IMPORT_ERROR = None
    else:
        GaussianModel = None
        generate_neural_gaussians = None
        MODEL_IMPORT_ERROR = "PyTorch is not installed"
except (ImportError, OSError) as error:
    GaussianModel = None
    generate_neural_gaussians = None
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
        optimizer_backend="default",
        densification_chunk_size=4096,
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

    def test_sogs_chunk_size_validation(self):
        self.assertEqual(validate_sogs_chunk_size("1024"), 1024)
        self.assertEqual(validate_sogs_chunk_size(4096), 4096)
        for invalid in (0, -1, True, 1.5, "not-an-int"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "sogs_chunk_size"):
                    validate_sogs_chunk_size(invalid)

    def test_a100_runtime_controls_parse_safely(self):
        config = self.parse_model(
            [
                "--sogs_checkpointing",
                "False",
                "--sogs_validate_numerics",
                "False",
                "--sogs_cache_render_features",
                "True",
                "--tf32_mode",
                "enabled",
                "--cudnn_benchmark",
                "True",
            ]
        )
        self.assertFalse(config.sogs_checkpointing)
        self.assertFalse(config.sogs_validate_numerics)
        self.assertTrue(config.sogs_cache_render_features)
        self.assertEqual(config.tf32_mode, "enabled")
        self.assertTrue(config.cudnn_benchmark)

    def test_runtime_setting_validation(self):
        self.assertEqual(validate_tf32_mode("enabled"), "enabled")
        self.assertEqual(validate_optimizer_backend("auto"), "auto")
        self.assertEqual(
            validate_positive_int("8192", name="densification_chunk_size"),
            8192,
        )
        with self.assertRaisesRegex(ValueError, "tf32_mode"):
            validate_tf32_mode("fastest")
        with self.assertRaisesRegex(ValueError, "optimizer_backend"):
            validate_optimizer_backend("magic")
        with self.assertRaisesRegex(ValueError, "log_interval"):
            validate_positive_int(0, name="log_interval")


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

    def test_chunked_checkpoint_path_keeps_gradients_finite(self):
        features = torch.nn.Parameter(torch.randn(9, 4))
        augmentor = SecondOrderFeatureAugmentor(4, 2, chunk_size=2)
        loss = augmentor(features).square().mean()
        loss.backward()
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all().item())
        for parameter in augmentor.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())

    def test_checkpointed_and_retained_paths_match(self):
        torch.manual_seed(7)
        checkpointed = SecondOrderFeatureAugmentor(
            4,
            2,
            chunk_size=2,
            checkpointing=True,
        )
        retained = SecondOrderFeatureAugmentor(
            4,
            2,
            chunk_size=64,
            checkpointing=False,
        )
        retained.load_state_dict(checkpointed.state_dict())
        features_checkpointed = torch.randn(7, 4, requires_grad=True)
        features_retained = (
            features_checkpointed.detach().clone().requires_grad_(True)
        )

        output_checkpointed = checkpointed(features_checkpointed)
        output_retained = retained(features_retained)
        self.assertTrue(
            torch.allclose(
                output_checkpointed,
                output_retained,
                atol=1e-6,
                rtol=1e-6,
            )
        )
        output_checkpointed.square().mean().backward()
        output_retained.square().mean().backward()
        self.assertTrue(
            torch.allclose(
                features_checkpointed.grad,
                features_retained.grad,
                atol=1e-6,
                rtol=1e-5,
            )
        )
        for checkpointed_parameter, retained_parameter in zip(
            checkpointed.parameters(),
            retained.parameters(),
        ):
            self.assertTrue(
                torch.allclose(
                    checkpointed_parameter.grad,
                    retained_parameter.grad,
                    atol=1e-6,
                    rtol=1e-5,
                )
            )

    def test_constant_features(self):
        features = torch.ones(4, 3, requires_grad=True)
        statistics = compute_second_order_statistics(features, 2)
        self.assertTrue(torch.isfinite(statistics.covariance).all().item())
        self.assertTrue(torch.isfinite(statistics.correlation).all().item())
        self.assertTrue(torch.allclose(statistics.covariance, torch.zeros(3, 3)))
        statistics.correlation.square().sum().backward()
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all().item())

    def test_zero_initialized_features_remain_finite_across_steps(self):
        # COMPATIBILITY: Scaffold-GS initializes every anchor feature to zero.
        # Exercise a non-uniform downstream signal so the test covers the
        # eigendecomposition gradient used during actual rendering.
        features = torch.nn.Parameter(torch.zeros(8, 4))
        augmentor = SecondOrderFeatureAugmentor(4, 2)
        target = torch.linspace(-1.0, 1.0, steps=8 * 12).reshape(8, 12)
        optimizer = torch.optim.Adam(
            [features] + list(augmentor.parameters()),
            lr=0.0075,
            eps=1e-15,
        )
        for _ in range(3):
            loss = (augmentor(features) - target).square().mean()
            loss.backward()
            self.assertIsNotNone(features.grad)
            self.assertTrue(torch.isfinite(features.grad).all().item())
            for parameter in augmentor.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
            optimizer.step()
            self.assertTrue(torch.isfinite(features).all().item())
            optimizer.zero_grad(set_to_none=True)

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
class VolumeRegularizationCompatibilityTests(unittest.TestCase):
    def test_value_and_gradient_match_xyz_volume(self):
        scaling = torch.tensor(
            [[2.0, 3.0, 4.0], [1.0, 5.0, 2.0]],
            requires_grad=True,
        )
        loss = scaling_volume_regularization(scaling)
        self.assertEqual(loss.item(), 17.0)

        loss.backward()
        expected_gradient = torch.tensor(
            [[6.0, 4.0, 3.0], [5.0, 1.0, 2.5]]
        )
        self.assertTrue(torch.equal(scaling.grad, expected_gradient))

    def test_shape_validation(self):
        with self.assertRaisesRegex(ValueError, "K x 3"):
            scaling_volume_regularization(torch.ones(2, 4))
        with self.assertRaisesRegex(ValueError, "K x 3"):
            scaling_volume_regularization(torch.ones(3))

    def test_empty_scaling_is_finite_zero(self):
        scaling = torch.empty(0, 3, requires_grad=True)
        loss = scaling_volume_regularization(scaling)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(scaling.grad)


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

    def test_cached_loss_kernels_preserve_values(self):
        prediction = torch.rand(3, 12, 12)
        target = torch.rand_like(prediction)
        first_sgl = selective_gradient_loss(prediction, target)
        second_sgl = selective_gradient_loss(prediction, target)
        self.assertTrue(torch.equal(first_sgl, second_sgl))
        first_ssim = ssim(prediction, target)
        second_ssim = ssim(prediction, target)
        self.assertTrue(torch.equal(first_ssim, second_ssim))

        image = prediction.unsqueeze(0)
        fused_x, fused_y = _sobel_gradient_maps(image)
        reference_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3).expand(3, 1, 3, 3)
        reference_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3).expand(3, 1, 3, 3)
        expected_x = torch.nn.functional.conv2d(
            image, reference_x, padding=1, groups=3
        )
        expected_y = torch.nn.functional.conv2d(
            image, reference_y, padding=1, groups=3
        )
        # COMPATIBILITY: older PyTorch convolution backends may accumulate a
        # grouped convolution in a different order than separate x/y calls.
        # The two paths must agree numerically, but need not be bit-identical.
        self.assertTrue(
            torch.allclose(fused_x, expected_x, rtol=1e-5, atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(fused_y, expected_y, rtol=1e-5, atol=1e-6)
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
    def make_model(
        self,
        enabled,
        *,
        chunk_size=2048,
        checkpointing=True,
        validate_numerics=True,
        cache_render_features=False,
    ):
        model = GaussianModel(
            feat_dim=4,
            n_offsets=2,
            appearance_dim=0,
            use_second_order=enabled,
            num_eigenvectors=2,
            lambda_sgl=0.01,
            sogs_chunk_size=chunk_size,
            sogs_checkpointing=checkpointing,
            sogs_validate_numerics=validate_numerics,
            sogs_cache_render_features=cache_render_features,
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

    def test_visible_feature_accessor_only_augments_selected_rows(self):
        model = self.make_model(True)
        visible = torch.tensor([True, False, True])
        all_features = model.get_render_features()
        visible_features = model.get_render_features(visible_mask=visible)
        self.assertEqual(tuple(visible_features.shape), (2, 12))
        self.assertTrue(torch.allclose(visible_features, all_features[visible]))

    def test_chunk_size_is_saved_in_model_configuration(self):
        model = self.make_model(True, chunk_size=1024)
        self.assertEqual(model.get_sogs_config()["sogs_chunk_size"], 1024)

    def test_chunk_size_can_change_when_loading_weights(self):
        source = self.make_model(True, chunk_size=1024)
        state = source.capture()
        restored = self.make_model(True, chunk_size=2048)
        restored.restore(state, optimization_args())
        self.assertEqual(restored.sogs_chunk_size, 2048)

    def test_runtime_memory_controls_can_change_when_loading_weights(self):
        source = self.make_model(
            True,
            checkpointing=True,
            validate_numerics=True,
            cache_render_features=False,
        )
        state = source.capture()
        restored = self.make_model(
            True,
            checkpointing=False,
            validate_numerics=False,
            cache_render_features=True,
        )
        restored.restore(state, optimization_args())
        self.assertFalse(restored.sogs_checkpointing)
        self.assertFalse(restored.sogs_validate_numerics)
        self.assertTrue(restored.sogs_cache_render_features)

    def test_eval_render_feature_cache_is_reused_and_invalidated(self):
        model = self.make_model(
            True,
            checkpointing=False,
            cache_render_features=True,
        )
        model.eval()
        with torch.no_grad():
            first = model.get_render_features()
            second = model.get_render_features()
            visible = model.get_render_features(
                visible_mask=torch.tensor([True, False, True])
            )
        self.assertEqual(first.data_ptr(), second.data_ptr())
        self.assertTrue(torch.equal(visible, first[[0, 2]]))
        self.assertIsNotNone(model._render_features_cache)
        model.train()
        self.assertIsNone(model._render_features_cache)

    def test_memory_efficient_renderer_forward_and_backward(self):
        model = self.make_model(True, chunk_size=1)
        with torch.no_grad():
            model.mlp_opacity[2].weight.zero_()
            model.mlp_opacity[2].bias.copy_(torch.tensor([1.0, -1.0]))
        camera = SimpleNamespace(
            camera_center=torch.zeros(3),
            uid=0,
        )
        visible = torch.tensor([True, False, True])
        (
            xyz,
            color,
            opacity,
            scaling,
            rotation,
            neural_opacity,
            selection_mask,
        ) = generate_neural_gaussians(
            camera,
            model,
            visible_mask=visible,
            is_training=True,
        )
        self.assertEqual(tuple(xyz.shape), (2, 3))
        self.assertEqual(tuple(color.shape), (2, 3))
        self.assertEqual(tuple(opacity.shape), (2, 1))
        self.assertEqual(tuple(scaling.shape), (2, 3))
        self.assertEqual(tuple(rotation.shape), (2, 4))
        self.assertEqual(tuple(neural_opacity.shape), (4, 1))
        self.assertEqual(tuple(selection_mask.shape), (4,))
        self.assertEqual(
            selection_mask.tolist(),
            [True, False, True, False],
        )
        loss = (
            xyz.square().mean()
            + color.square().mean()
            + opacity.square().mean()
            + scaling.square().mean()
            + rotation.square().mean()
        )
        loss.backward()
        self.assertIsNotNone(model._anchor_feat.grad)
        self.assertTrue(torch.isfinite(model._anchor_feat.grad).all().item())

    def test_memory_efficient_training_statistics_index_mapping(self):
        model = self.make_model(True)
        model.opacity_accum = torch.zeros(3, 1)
        model.anchor_demon = torch.zeros(3, 1)
        model.offset_gradient_accum = torch.zeros(6, 1)
        model.offset_denom = torch.zeros(6, 1)
        viewspace = torch.zeros(2, 3, requires_grad=True)
        viewspace.grad = torch.tensor(
            [[3.0, 4.0, 0.0], [6.0, 8.0, 0.0]]
        )
        model.training_statis(
            viewspace_point_tensor=viewspace,
            opacity=torch.tensor([[0.1], [-0.2], [0.3], [0.4]]),
            update_filter=torch.tensor([True, False]),
            offset_selection_mask=torch.tensor([True, False, False, True]),
            anchor_visible_mask=torch.tensor([True, False, True]),
        )
        self.assertTrue(
            torch.allclose(
                model.opacity_accum,
                torch.tensor([[0.1], [0.0], [0.7]]),
            )
        )
        self.assertTrue(
            torch.equal(model.anchor_demon, torch.tensor([[1.0], [0.0], [1.0]]))
        )
        self.assertEqual(model.offset_gradient_accum[0].item(), 5.0)
        self.assertEqual(model.offset_denom[0].item(), 1.0)
        self.assertEqual(model.offset_gradient_accum[5].item(), 0.0)

    def test_optimizer_contains_second_order_parameters(self):
        model = self.make_model(True)
        model.training_setup(optimization_args())
        self.assertEqual(model.optimizer_backend, "default")
        self.assertEqual(model.densification_chunk_size, 4096)
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


class TrainingBudgetTests(unittest.TestCase):
    def test_disabled_budget_never_expires(self):
        budget = TrainingBudget(0)
        self.assertFalse(budget.enabled)
        self.assertFalse(budget.expired())
        self.assertIsNone(budget.remaining_seconds())

    def test_budget_expires_at_monotonic_deadline(self):
        now = [100.0]
        budget = TrainingBudget(1.0, clock=lambda: now[0])
        self.assertTrue(budget.enabled)
        self.assertEqual(budget.remaining_seconds(), 60.0)
        now[0] = 159.5
        self.assertFalse(budget.expired())
        now[0] = 160.0
        self.assertTrue(budget.expired())
        self.assertEqual(budget.remaining_seconds(), 0.0)

    def test_invalid_runtime_budgets_are_rejected(self):
        for value in (-1, float("nan"), float("inf"), True, "bad"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_max_runtime_minutes(value)

    def test_periodic_and_explicit_checkpoint_selection(self):
        self.assertEqual(validate_checkpoint_interval(500), 500)
        self.assertTrue(should_checkpoint_iteration(500, [], 500))
        self.assertTrue(should_checkpoint_iteration(75, [75], 0))
        self.assertFalse(should_checkpoint_iteration(76, [75], 500))
        for value in (-1, 1.5, True, "bad"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_checkpoint_interval(value)


class StaticIntegrationTests(unittest.TestCase):
    def test_renderer_uses_single_model_feature_accessor(self):
        root = Path(__file__).resolve().parent
        renderer = (root / "gaussian_renderer" / "__init__.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("pc.get_render_features(visible_mask=visible_mask)", renderer)
        self.assertIn("_run_attribute_mlp_chunked", renderer)
        self.assertNotIn("concatenated_all", renderer)
        self.assertNotIn("feat = pc._anchor_feat[visible_mask]", renderer)

    def test_training_avoids_nvrtc_prod_reduction(self):
        root = Path(__file__).resolve().parent
        training = (root / "train.py").read_text(encoding="utf-8")
        self.assertIn("scaling_volume_regularization(scaling)", training)
        self.assertNotIn("scaling.prod(dim=1)", training)

    def test_sogs_chunk_budget_reaches_densification(self):
        root = Path(__file__).resolve().parent
        model = (root / "scene" / "gaussian_model.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("chunk_size = self.densification_chunk_size", model)
        self.assertIn("remove_duplicates.logical_or_", model)
        self.assertNotIn("remove_duplicates_list", model)

    def test_runtime_budget_saves_renderable_and_resumable_state(self):
        root = Path(__file__).resolve().parent
        training = (root / "train.py").read_text(encoding="utf-8")
        self.assertIn("runtime_budget.expired()", training)
        self.assertIn("scene.save(iteration)", training)
        self.assertIn("_save_training_checkpoint(", training)

    def test_training_records_resolved_runtime_and_peak_memory(self):
        root = Path(__file__).resolve().parent
        training = (root / "train.py").read_text(encoding="utf-8")
        self.assertIn('"runtime_config.json"', training)
        self.assertIn('"resolved_torch_runtime"', training)
        self.assertIn("config_snapshot=args", training)
        self.assertIn("torch.cuda.max_memory_allocated", training)

    def test_one_hour_profile_preserves_full_image_resolution(self):
        root = Path(__file__).resolve().parent
        launcher = (
            root / "scripts" / "train_sogs_one_hour.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--max_runtime_minutes", launcher)
        self.assertIn("--checkpoint_interval 500", launcher)
        self.assertNotIn("--resolution 2", launcher)

    def test_a100_profiles_are_explicit_and_do_not_launch_implicitly(self):
        root = Path(__file__).resolve().parent
        launcher = (
            root / "scripts" / "train_sogs_a100.sh"
        ).read_text(encoding="utf-8")
        throughput = (
            root / "configs" / "sogs_a100_throughput.yaml"
        ).read_text(encoding="utf-8")
        quality = (
            root / "configs" / "sogs_a100_quality.yaml"
        ).read_text(encoding="utf-8")
        ultra = (
            root / "configs" / "sogs_a100_ultra.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn("--sogs_checkpointing False", launcher)
        self.assertIn("--sogs_cache_render_features True", launcher)
        self.assertIn("--tf32_mode enabled", launcher)
        self.assertIn("--tf32_mode disabled", launcher)
        self.assertIn("--appearance_dim 0", launcher)
        self.assertIn("--resolution 1", launcher)
        self.assertIn("--resolution 4", launcher)
        self.assertIn("--update_until 7500", launcher)
        self.assertIn("DRY_RUN", launcher)
        self.assertIn("feat_dim: 16", throughput)
        self.assertIn("feat_dim: 32", quality)
        self.assertIn("resolution: 1", throughput)
        self.assertIn("resolution: 1", quality)
        for expected in (
            "feat_dim: 8",
            "num_eigenvectors: 1",
            "lambda_sgl: 0",
            "n_offsets: 5",
            "appearance_dim: 0",
            "resolution: 4",
            "ratio: 2",
            "voxel_size: 0.001",
            "iterations: 30000",
            "update_until: 7500",
            "sogs_chunk_size: 262144",
            "sogs_checkpointing: false",
            "sogs_validate_numerics: false",
            "sogs_cache_render_features: true",
            "densification_chunk_size: 16384",
            "tf32_mode: enabled",
            "cudnn_benchmark: true",
            "optimizer_backend: auto",
            "log_interval: 1000",
        ):
            self.assertIn(expected, ultra)

    def test_a100_max_speed_runner_dry_run_is_non_mutating(self):
        root = Path(__file__).resolve().parent
        launcher = root / "scripts" / "run_sogs_a100_max_speed.sh"
        launcher_text = launcher.read_text(encoding="utf-8")
        self.assertIn("--query-compute-apps", launcher_text)
        self.assertIn("No process was killed", launcher_text)

        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            data_root = temporary_root / "data"
            output_root = temporary_root / "models"
            log_root = temporary_root / "logs"
            (data_root / "synthetic" / "train").mkdir(parents=True)

            result = subprocess.run(
                [
                    "bash",
                    str(launcher),
                    "--dry-run",
                    "--data-root",
                    str(data_root),
                    "--output-root",
                    str(output_root),
                    "--log-root",
                    str(log_root),
                    "synthetic",
                ],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("--test_iterations 1000000000", result.stdout)
            self.assertIn("--skip_postprocess", result.stdout)
            self.assertIn("--tf32_mode enabled", result.stdout)
            self.assertIn("--log_interval 250", result.stdout)
            self.assertIn("Profile: A100 throughput", result.stdout)
            for expected in (
                "--feat_dim 16",
                "--num_eigenvectors 2",
                "--lambda_sgl 0.01",
                "--n_offsets 10",
                "--resolution 1",
                "--ratio 1",
                "--update_until 15000",
                "--sogs_chunk_size 65536",
                "--densification_chunk_size 8192",
            ):
                self.assertIn(expected, result.stdout)
            self.assertIn("training was not started", result.stdout)
            self.assertFalse(output_root.exists())
            self.assertFalse(log_root.exists())

            ultra_result = subprocess.run(
                [
                    "bash",
                    str(launcher),
                    "--profile",
                    "ultra",
                    "--dry-run",
                    "--data-root",
                    str(data_root),
                    "--output-root",
                    str(output_root),
                    "--log-root",
                    str(log_root),
                    "synthetic",
                ],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("Profile: A100 ultra speed", ultra_result.stdout)
            for expected in (
                "--use_second_order True",
                "--feat_dim 8",
                "--num_eigenvectors 1",
                "--lambda_sgl 0",
                "--n_offsets 5",
                "--appearance_dim 0",
                "--resolution 4",
                "--ratio 2",
                "--voxel_size 0.001",
                "--iterations 30000",
                "--update_until 7500",
                "--sogs_chunk_size 262144",
                "--sogs_checkpointing False",
                "--sogs_validate_numerics False",
                "--sogs_cache_render_features True",
                "--densification_chunk_size 16384",
                "--cudnn_benchmark True",
                "--tf32_mode enabled",
                "--optimizer_backend auto",
                "--log_interval 1000",
            ):
                self.assertIn(expected, ultra_result.stdout)
            for option in (
                "--feat_dim",
                "--num_eigenvectors",
                "--lambda_sgl",
                "--n_offsets",
                "--resolution",
                "--ratio",
                "--update_until",
                "--sogs_chunk_size",
                "--sogs_validate_numerics",
                "--densification_chunk_size",
                "--tf32_mode",
                "--optimizer_backend",
                "--log_interval",
            ):
                self.assertEqual(ultra_result.stdout.count(option), 1)
            self.assertIn(
                "training was not started",
                ultra_result.stdout,
            )
            self.assertFalse(output_root.exists())
            self.assertFalse(log_root.exists())

            quality_result = subprocess.run(
                [
                    "bash",
                    str(launcher),
                    "--profile",
                    "quality",
                    "--dry-run",
                    "--data-root",
                    str(data_root),
                    "--output-root",
                    str(output_root),
                    "--log-root",
                    str(log_root),
                    "synthetic",
                ],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn(
                "Profile: A100 quality control",
                quality_result.stdout,
            )
            for expected in (
                "--feat_dim 32",
                "--num_eigenvectors 2",
                "--lambda_sgl 0.01",
                "--n_offsets 10",
                "--resolution 1",
                "--sogs_validate_numerics True",
                "--tf32_mode disabled",
                "--optimizer_backend default",
                "--log_interval 250",
            ):
                self.assertIn(expected, quality_result.stdout)
            self.assertFalse(output_root.exists())
            self.assertFalse(log_root.exists())

            invalid_result = subprocess.run(
                [
                    "bash",
                    str(launcher),
                    "--profile",
                    "bogus",
                    "--dry-run",
                ],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(invalid_result.returncode, 0)
            self.assertIn("unknown profile", invalid_result.stderr)
            self.assertFalse(output_root.exists())
            self.assertFalse(log_root.exists())

            missing_result = subprocess.run(
                ["bash", str(launcher), "--profile"],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(missing_result.returncode, 0)
            self.assertIn(
                "--profile requires a value",
                missing_result.stderr,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
