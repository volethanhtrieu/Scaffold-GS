#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
import json
from torch_scatter import scatter_max
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.embedding import Embedding
from utils.sogs_utils import (
    SecondOrderFeatureAugmentor,
    validate_second_order_dimensions,
)
from arguments import validate_sogs_chunk_size, validate_sogs_config

    
class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, 
                 feat_dim: int=32, 
                 n_offsets: int=5, 
                 voxel_size: float=0.01,
                 update_depth: int=3, 
                 update_init_factor: int=100,
                 update_hierachy_factor: int=4,
                 use_feat_bank : bool = False,
                 appearance_dim : int = 32,
                 ratio : int = 1,
                 add_opacity_dist : bool = False,
                 add_cov_dist : bool = False,
                 add_color_dist : bool = False,
                 use_second_order: bool = False,
                 num_eigenvectors: int = 2,
                 lambda_sgl: float = 0.01,
                 device=None,
                 sogs_chunk_size: int = 2048,
                 ):

        # COMPATIBILITY: append SOGS settings after the original positional
        # arguments so existing Scaffold-GS callers remain valid.
        (
            self.feat_dim,
            self.use_second_order,
            self.num_eigenvectors,
            self.lambda_sgl,
        ) = validate_sogs_config(
            feat_dim,
            use_second_order,
            num_eigenvectors,
            lambda_sgl,
        )
        validate_second_order_dimensions(
            self.feat_dim,
            self.num_eigenvectors,
            enabled=self.use_second_order,
        )
        self.sogs_chunk_size = validate_sogs_chunk_size(sogs_chunk_size)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.n_offsets = n_offsets
        self.voxel_size = voxel_size
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.use_feat_bank = use_feat_bank

        self.appearance_dim = appearance_dim
        self.embedding_appearance = None
        self.ratio = ratio
        self.add_opacity_dist = add_opacity_dist
        self.add_cov_dist = add_cov_dist
        self.add_color_dist = add_color_dist

        self._anchor = torch.empty(0, device=self.device)
        self._offset = torch.empty(0, device=self.device)
        self._anchor_feat = torch.empty(0, device=self.device)
        
        self.opacity_accum = torch.empty(0, device=self.device)

        self._scaling = torch.empty(0, device=self.device)
        self._rotation = torch.empty(0, device=self.device)
        self._opacity = torch.empty(0, device=self.device)
        self.max_radii2D = torch.empty(0, device=self.device)
        
        self.offset_gradient_accum = torch.empty(0, device=self.device)
        self.offset_denom = torch.empty(0, device=self.device)

        self.anchor_demon = torch.empty(0, device=self.device)
        # COMPATIBILITY: legacy capture() referenced these names without
        # initializing them in this local fork.
        self._local = torch.empty(0, device=self.device)
        self.denom = torch.empty(0, device=self.device)
        self.active_sh_degree = 0
                
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        self.render_feat_dim = self.feat_dim * (
            1 + self.num_eigenvectors if self.use_second_order else 1
        )
        # PAPER: learn one two-layer branch per selected eigenvector and
        # concatenate branch outputs with the original anchor feature.
        self.second_order_augmentor = (
            SecondOrderFeatureAugmentor(
                self.feat_dim,
                self.num_eigenvectors,
                chunk_size=self.sogs_chunk_size,
            ).to(self.device)
            if self.use_second_order
            else None
        )

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3+1, self.feat_dim),
                nn.ReLU(True),
                nn.Linear(self.feat_dim, 3),
                nn.Softmax(dim=1)
            ).to(self.device)

        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
        self.mlp_opacity = nn.Sequential(
            nn.Linear(
                self.render_feat_dim+3+self.opacity_dist_dim, self.feat_dim
            ),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, n_offsets),
            nn.Tanh()
        ).to(self.device)

        self.add_cov_dist = add_cov_dist
        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        self.mlp_cov = nn.Sequential(
            nn.Linear(
                self.render_feat_dim+3+self.cov_dist_dim, self.feat_dim
            ),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 7*self.n_offsets),
        ).to(self.device)

        self.color_dist_dim = 1 if self.add_color_dist else 0
        self.mlp_color = nn.Sequential(
            nn.Linear(
                self.render_feat_dim+3+self.color_dist_dim+self.appearance_dim,
                self.feat_dim,
            ),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).to(self.device)


    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        if self.second_order_augmentor is not None:
            self.second_order_augmentor.eval()
        if self.appearance_dim > 0 and self.embedding_appearance is not None:
            self.embedding_appearance.eval()
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        if self.second_order_augmentor is not None:
            self.second_order_augmentor.train()
        if self.appearance_dim > 0 and self.embedding_appearance is not None:
            self.embedding_appearance.train()
        if self.use_feat_bank:                   
            self.mlp_feature_bank.train()

    @property
    def get_render_feature_dim(self):
        """Return the feature width consumed by the attribute MLPs."""

        return self.render_feat_dim

    def get_render_features(self, visible_mask=None):
        """Return base or second-order-augmented anchor features.

        The returned tensor has shape ``[N, D]`` for Scaffold-GS and
        ``[N, D * (1 + M)]`` for SOGS.  Statistics are intentionally computed
        from all anchors before optional renderer visibility masking so the
        correlation patterns remain scene-global rather than view-dependent.
        When ``visible_mask`` is supplied, only visible rows are augmented.
        """

        features = self._anchor_feat
        if features.numel() == 0:
            return features.reshape(0, self.render_feat_dim)
        if features.ndim != 2 or features.shape[1] != self.feat_dim:
            raise RuntimeError(
                "anchor feature tensor must have shape "
                f"[N, {self.feat_dim}], received {tuple(features.shape)}"
            )
        if visible_mask is not None:
            if not isinstance(visible_mask, torch.Tensor):
                raise TypeError("visible_mask must be a torch.Tensor")
            if (
                visible_mask.ndim != 1
                or visible_mask.shape[0] != features.shape[0]
            ):
                raise ValueError(
                    "visible_mask must have shape [N] matching anchor features"
                )
            if visible_mask.dtype != torch.bool:
                raise ValueError("visible_mask must use torch.bool dtype")
            if visible_mask.device != features.device:
                raise ValueError("visible_mask and anchor features must share a device")
        if not self.use_second_order:
            expected_dim = self.feat_dim
            assert features.shape[-1] == expected_dim
            return features if visible_mask is None else features[visible_mask]
        if self.second_order_augmentor is None:
            raise RuntimeError(
                "SOGS is enabled but its second-order augmentor is not initialized"
            )
        # PAPER: expose one feature path to every renderer consumer.
        augmented_features = self.second_order_augmentor(
            features, output_mask=visible_mask
        )
        expected_dim = self.feat_dim * (1 + self.num_eigenvectors)
        assert augmented_features.shape[-1] == expected_dim
        return augmented_features

    def get_sogs_config(self):
        """Return serializable SOGS settings for configs/checkpoint metadata."""

        return {
            "use_second_order": bool(self.use_second_order),
            "num_eigenvectors": int(self.num_eigenvectors),
            "lambda_sgl": float(self.lambda_sgl),
            "feat_dim": int(self.feat_dim),
            "render_feat_dim": int(self.render_feat_dim),
            "sogs_chunk_size": int(self.sogs_chunk_size),
        }

    def _assert_checkpoint_compatible(self, checkpoint_config, *, source="checkpoint"):
        """Reject architecture/configuration mismatches instead of random state."""

        if checkpoint_config is None:
            if self.use_second_order:
                raise RuntimeError(
                    f"{source} does not record SOGS configuration; "
                    "cannot safely load it with use_second_order=True"
                )
            return
        if not isinstance(checkpoint_config, dict):
            raise RuntimeError(
                f"{source} has an invalid SOGS configuration; expected an object"
            )
        if not checkpoint_config:
            if self.use_second_order:
                raise RuntimeError(
                    f"{source} does not record SOGS configuration; "
                    "cannot safely load it with use_second_order=True"
                )
            return
        expected = self.get_sogs_config()
        for key in (
            "use_second_order",
            "num_eigenvectors",
            "lambda_sgl",
            "feat_dim",
            "render_feat_dim",
            "sogs_chunk_size",
        ):
            if key not in checkpoint_config:
                # COMPATIBILITY: older SOGS checkpoints predate the runtime
                # memory knob; the default chunk size is safe for loading.
                if self.use_second_order and key != "sogs_chunk_size":
                    raise RuntimeError(
                        f"{source} is missing required SOGS setting {key!r}"
                    )
                continue
            actual = checkpoint_config[key]
            if key == "sogs_chunk_size":
                # INFERENCE: this only controls activation tiling, so a
                # checkpoint can safely be rendered with a different VRAM
                # budget without changing its learned architecture.
                try:
                    validate_sogs_chunk_size(actual)
                except ValueError as error:
                    raise RuntimeError(
                        f"invalid {source} setting sogs_chunk_size={actual!r}"
                    ) from error
                continue
            if key == "lambda_sgl":
                try:
                    matches = abs(float(actual) - expected[key]) <= 1e-12
                except (TypeError, ValueError) as error:
                    raise RuntimeError(
                        f"invalid {source} setting lambda_sgl={actual!r}"
                    ) from error
            else:
                matches = actual == expected[key]
            if not matches:
                raise RuntimeError(
                    f"incompatible {source} setting {key}: "
                    f"checkpoint={actual!r}, requested={expected[key]!r}"
                )

    @staticmethod
    def _load_module_state(module, state, name):
        if state is None:
            raise RuntimeError(f"checkpoint is missing state for {name}")
        try:
            module.load_state_dict(state, strict=True)
        except (RuntimeError, KeyError) as error:
            raise RuntimeError(f"incompatible state for {name}: {error}") from error

    def _mlp_state(self):
        state = {
            "opacity_mlp": self.mlp_opacity.state_dict(),
            "cov_mlp": self.mlp_cov.state_dict(),
            "color_mlp": self.mlp_color.state_dict(),
        }
        if self.use_feat_bank:
            state["feature_bank_mlp"] = self.mlp_feature_bank.state_dict()
        if self.appearance_dim > 0 and self.embedding_appearance is not None:
            state["appearance"] = self.embedding_appearance.state_dict()
        if self.second_order_augmentor is not None:
            state["second_order"] = self.second_order_augmentor.state_dict()
        return state

    def capture(self):
        """Capture model, SOGS modules, optimizer, and compatibility metadata.

        A dictionary is used for new checkpoints.  ``restore`` still accepts
        the original Scaffold-GS positional tuple so old checkpoints can be
        loaded in baseline mode.
        """

        return {
            "format_version": 2,
            "config": self.get_sogs_config(),
            "active_sh_degree": int(self.active_sh_degree),
            "tensors": {
                "anchor": self._anchor,
                "offset": self._offset,
                "anchor_feat": self._anchor_feat,
                "local": self._local,
                "scaling": self._scaling,
                "rotation": self._rotation,
                "opacity": self._opacity,
                "max_radii2D": self.max_radii2D,
                "denom": self.denom,
            },
            "parameter_requires_grad": {
                name: bool(getattr(self, "_" + name).requires_grad)
                for name in (
                    "anchor",
                    "offset",
                    "anchor_feat",
                    "scaling",
                    "rotation",
                    "opacity",
                )
            },
            "mlp": self._mlp_state(),
            "optimizer": (
                self.optimizer.state_dict() if self.optimizer is not None else None
            ),
            "training_statistics": {
                "opacity_accum": self.opacity_accum,
                "offset_gradient_accum": self.offset_gradient_accum,
                "offset_denom": self.offset_denom,
                "anchor_demon": self.anchor_demon,
            },
            "spatial_lr_scale": self.spatial_lr_scale,
        }
    
    def restore(self, model_args, training_args):
        """Restore a new structured checkpoint or a legacy Scaffold-GS tuple."""

        if isinstance(model_args, dict) and "tensors" in model_args:
            if model_args.get("format_version") != 2:
                raise RuntimeError(
                    "unsupported structured checkpoint format; "
                    "expected format_version=2"
                )
            self._assert_checkpoint_compatible(
                model_args.get("config"), source="model checkpoint"
            )
            tensors = model_args["tensors"]
            if not isinstance(tensors, dict):
                raise RuntimeError(
                    "model checkpoint has invalid tensor state; expected an object"
                )
            required = (
                "anchor",
                "offset",
                "anchor_feat",
                "scaling",
                "rotation",
                "opacity",
                "max_radii2D",
                "denom",
            )
            missing = [name for name in required if name not in tensors]
            if missing:
                raise RuntimeError(
                    "model checkpoint is missing tensor state: " + ", ".join(missing)
                )
            invalid = [
                name
                for name in required
                if not isinstance(tensors.get(name), torch.Tensor)
            ]
            if invalid:
                raise RuntimeError(
                    "model checkpoint has non-tensor state for: "
                    + ", ".join(invalid)
                )
            parameter_names = {
                "anchor",
                "offset",
                "anchor_feat",
                "scaling",
                "rotation",
                "opacity",
            }
            requires_grad = model_args.get("parameter_requires_grad", {})
            if requires_grad is None:
                requires_grad = {}
            if not isinstance(requires_grad, dict):
                raise RuntimeError(
                    "model checkpoint has invalid parameter_requires_grad metadata"
                )
            self.active_sh_degree = int(
                model_args.get("active_sh_degree", self.active_sh_degree)
            )
            for name, tensor in tensors.items():
                if tensor is None:
                    continue
                value = tensor.to(self.device)
                if name in parameter_names:
                    value = nn.Parameter(
                        value.requires_grad_(
                            bool(
                                requires_grad.get(
                                    name, name not in {"rotation", "opacity"}
                                )
                            )
                        )
                    )
                attribute = (
                    "_" + name
                    if name in parameter_names
                    else "_local"
                    if name == "local"
                    else name
                )
                setattr(self, attribute, value)
            mlp_state = model_args.get("mlp", {})
            if not isinstance(mlp_state, dict):
                raise RuntimeError(
                    "model checkpoint has invalid MLP state; expected an object"
                )
            self._load_module_state(
                self.mlp_opacity, mlp_state.get("opacity_mlp"), "opacity_mlp"
            )
            self._load_module_state(
                self.mlp_cov, mlp_state.get("cov_mlp"), "cov_mlp"
            )
            self._load_module_state(
                self.mlp_color, mlp_state.get("color_mlp"), "color_mlp"
            )
            if self.second_order_augmentor is not None:
                self._load_module_state(
                    self.second_order_augmentor,
                    mlp_state.get("second_order"),
                    "second_order_augmentor",
                )
            if self.use_feat_bank:
                self._load_module_state(
                    self.mlp_feature_bank,
                    mlp_state.get("feature_bank_mlp"),
                    "feature_bank_mlp",
                )
            if self.appearance_dim > 0:
                if self.embedding_appearance is None:
                    raise RuntimeError(
                        "appearance embedding must be initialized before restoring "
                        "a checkpoint with appearance_dim > 0"
                    )
                self._load_module_state(
                    self.embedding_appearance,
                    mlp_state.get("appearance"),
                    "appearance_embedding",
                )
            self._local = tensors.get("local", self._local).to(self.device)
            self.denom = tensors.get("denom", self.denom).to(self.device)
            self.spatial_lr_scale = model_args.get(
                "spatial_lr_scale", self.spatial_lr_scale
            )
            self.training_setup(training_args)
            for name, tensor in model_args.get(
                "training_statistics", {}
            ).items():
                if isinstance(tensor, torch.Tensor):
                    setattr(self, name, tensor.to(self.device))
            optimizer_state = model_args.get("optimizer")
            if optimizer_state is not None:
                try:
                    self.optimizer.load_state_dict(optimizer_state)
                except (RuntimeError, ValueError) as error:
                    raise RuntimeError(
                        f"incompatible optimizer state in model checkpoint: {error}"
                    ) from error
            return

        # COMPATIBILITY: legacy local checkpoints were a positional tuple.
        if not isinstance(model_args, (tuple, list)) or len(model_args) < 10:
            raise RuntimeError("unsupported Scaffold-GS checkpoint format")
        if len(model_args) >= 11:
            (
                self.active_sh_degree,
                self._anchor,
                self._offset,
                self._local,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                denom,
                opt_dict,
                self.spatial_lr_scale,
            ) = model_args[:11]
        else:
            (
                self._anchor,
                self._offset,
                self._local,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                denom,
                opt_dict,
                self.spatial_lr_scale,
            ) = model_args
        if self.use_second_order:
            raise RuntimeError(
                "legacy Scaffold-GS checkpoints do not contain SOGS modules; "
                "reload with use_second_order=False or retrain"
            )
        self._anchor = nn.Parameter(
            self._anchor.to(self.device).requires_grad_(True)
        )
        self._offset = nn.Parameter(
            self._offset.to(self.device).requires_grad_(True)
        )
        legacy_local = self._local.to(self.device)
        if (
            legacy_local.ndim == 2
            and legacy_local.shape[0] == self._anchor.shape[0]
            and legacy_local.shape[1] == self.feat_dim
        ):
            # COMPATIBILITY: upstream/local legacy variants used the third
            # tuple slot for the anchor feature under different names.
            self._anchor_feat = nn.Parameter(legacy_local.requires_grad_(True))
            self._local = torch.empty(0, device=self.device)
        elif self._anchor_feat.shape != (
            self._anchor.shape[0],
            self.feat_dim,
        ):
            raise RuntimeError(
                "legacy checkpoint has no restorable anchor feature tensor"
            )
        else:
            self._local = legacy_local
        self._scaling = nn.Parameter(
            self._scaling.to(self.device).requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            self._rotation.to(self.device).requires_grad_(False)
        )
        self._opacity = nn.Parameter(
            self._opacity.to(self.device).requires_grad_(False)
        )
        self.max_radii2D = self.max_radii2D.to(self.device)
        self.denom = denom.to(self.device)
        self.training_setup(training_args)
        if opt_dict is not None:
            self.optimizer.load_state_dict(opt_dict)

    def set_appearance(self, num_cameras):
        if self.appearance_dim > 0:
            self.embedding_appearance = Embedding(
                num_cameras, self.appearance_dim
            ).to(self.device)

    @property
    def get_appearance(self):
        return self.embedding_appearance

    @property
    def get_scaling(self):
        return 1.0*self.scaling_activation(self._scaling)
    
    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank
    
    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity
    
    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_anchor(self):
        return self._anchor
    
    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size
        
        return data

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        points = pcd.points[::self.ratio]

        if self.voxel_size <= 0:
            init_points = torch.tensor(points, device=self.device).float()
            init_dist = distCUDA2(init_points).float().to(self.device)
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        print(f'Initial voxel_size: {self.voxel_size}')
        
        
        points = self.voxelize_sample(points, voxel_size=self.voxel_size)
        fused_point_cloud = torch.tensor(
            np.asarray(points), device=self.device
        ).float()
        offsets = torch.zeros(
            (fused_point_cloud.shape[0], self.n_offsets, 3), device=self.device
        ).float()
        anchors_feat = torch.zeros(
            (fused_point_cloud.shape[0], self.feat_dim), device=self.device
        ).float()
        
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(
            distCUDA2(fused_point_cloud).float().to(self.device), 0.0000001
        )
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)
        
        rots = torch.zeros(
            (fused_point_cloud.shape[0], 4), device=self.device
        )
        rots[:, 0] = 1

        opacities = inverse_sigmoid(
            0.1
            * torch.ones(
                (fused_point_cloud.shape[0], 1),
                dtype=torch.float,
                device=self.device,
            )
        )

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros(
            (self.get_anchor.shape[0]), device=self.device
        )


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.opacity_accum = torch.zeros(
            (self.get_anchor.shape[0], 1), device=self.device
        )

        self.offset_gradient_accum = torch.zeros(
            (self.get_anchor.shape[0] * self.n_offsets, 1), device=self.device
        )
        self.offset_denom = torch.zeros(
            (self.get_anchor.shape[0] * self.n_offsets, 1), device=self.device
        )
        self.anchor_demon = torch.zeros(
            (self.get_anchor.shape[0], 1), device=self.device
        )

        l = [
            {
                "params": [self._anchor],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "anchor",
            },
            {
                "params": [self._offset],
                "lr": training_args.offset_lr_init * self.spatial_lr_scale,
                "name": "offset",
            },
            {
                "params": [self._anchor_feat],
                "lr": training_args.feature_lr,
                "name": "anchor_feat",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
            {
                "params": self.mlp_opacity.parameters(),
                "lr": training_args.mlp_opacity_lr_init,
                "name": "mlp_opacity",
            },
        ]
        if self.use_feat_bank:
            l.append(
                {
                    "params": self.mlp_feature_bank.parameters(),
                    "lr": training_args.mlp_featurebank_lr_init,
                    "name": "mlp_featurebank",
                }
            )
        l.extend(
            [
                {
                    "params": self.mlp_cov.parameters(),
                    "lr": training_args.mlp_cov_lr_init,
                    "name": "mlp_cov",
                },
                {
                    "params": self.mlp_color.parameters(),
                    "lr": training_args.mlp_color_lr_init,
                    "name": "mlp_color",
                },
            ]
        )
        # PAPER: Fi modules are trainable and therefore must have an optimizer
        # parameter group.  The paper does not specify a separate schedule;
        # reuse the anchor feature learning rate.
        if self.second_order_augmentor is not None:
            l.append(
                {
                    "params": self.second_order_augmentor.parameters(),
                    "lr": training_args.feature_lr,
                    "name": "mlp_second_order",
                }
            )
        if self.appearance_dim > 0:
            if self.embedding_appearance is None:
                raise RuntimeError(
                    "appearance_dim > 0 requires set_appearance() before training_setup()"
                )
            l.append(
                {
                    "params": self.embedding_appearance.parameters(),
                    "lr": training_args.appearance_lr_init,
                    "name": "embedding_appearance",
                }
            )

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)
        
        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)
        
        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)
        if self.appearance_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(lr_init=training_args.appearance_lr_init,
                                                        lr_final=training_args.appearance_lr_final,
                                                        lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                        max_steps=training_args.appearance_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.appearance_dim > 0 and param_group["name"] == "embedding_appearance":
                lr = self.appearance_scheduler_args(iteration)
                param_group['lr'] = lr
            
            
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, normals, offset, anchor_feat, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        if len(anchor_feat_names) != self.feat_dim:
            raise RuntimeError(
                "incompatible PLY anchor feature dimension: "
                f"checkpoint={len(anchor_feat_names)}, requested={self.feat_dim}"
            )
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))
        
        self._anchor_feat = nn.Parameter(
            torch.tensor(
                anchor_feats, dtype=torch.float, device=self.device
            ).requires_grad_(True)
        )

        self._offset = nn.Parameter(
            torch.tensor(
                offsets, dtype=torch.float, device=self.device
            ).transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._anchor = nn.Parameter(
            torch.tensor(
                anchor, dtype=torch.float, device=self.device
            ).requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(
                opacities, dtype=torch.float, device=self.device
            ).requires_grad_(True)
        )
        self._scaling = nn.Parameter(
            torch.tensor(
                scales, dtype=torch.float, device=self.device
            ).requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(
                rots, dtype=torch.float, device=self.device
            ).requires_grad_(True)
        )


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors


    # statis grad information to guide liftting. 
    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
        # update opacity stats
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        
        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        
        # update anchor visiting statis
        self.anchor_demon[anchor_visible_mask] += 1

        # update neural gaussian statis
        # INFERENCE: map visible local offsets directly back to their global
        # indices. This avoids two full-size boolean masks and a repeated
        # anchor-visible mask during every densification-statistics update.
        visible_anchor_indices = anchor_visible_mask.nonzero(
            as_tuple=False
        ).squeeze(dim=1)
        local_offsets = torch.arange(
            self.n_offsets,
            dtype=visible_anchor_indices.dtype,
            device=visible_anchor_indices.device,
        )
        global_offset_indices = (
            visible_anchor_indices[:, None] * self.n_offsets
            + local_offsets[None, :]
        ).reshape(-1)
        selected_global_indices = global_offset_indices[offset_selection_mask]
        updated_global_indices = selected_global_indices[update_filter]

        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.offset_gradient_accum[updated_global_indices] += grad_norm
        self.offset_denom[updated_global_indices] += 1

        

        
    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            
            
        return optimizable_tensors

    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

    
    def anchor_growing(self, grads, threshold, offset_mask):
        ## 
        init_length = self.get_anchor.shape[0]*self.n_offsets
        for i in range(self.update_depth):
            # update threshold
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)
            
            # random pick
            rand_mask = torch.rand_like(candidate_mask.float())>(0.5**(i+1))
            rand_mask = rand_mask.to(candidate_mask.device)
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)
            
            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat(
                    [
                        candidate_mask,
                        torch.zeros(
                            length_inc,
                            dtype=torch.bool,
                            device=candidate_mask.device,
                        ),
                    ],
                    dim=0,
                )

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:3].unsqueeze(dim=1)
            
            # assert self.update_init_factor // (self.update_hierachy_factor**i) > 0
            # size_factor = min(self.update_init_factor // (self.update_hierachy_factor**i), 1)
            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size*size_factor
            
            grid_coords = torch.round(self.get_anchor / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)


            ## split data for reducing peak memory calling
            use_chunk = True
            if use_chunk:
                # INFERENCE: reuse the SOGS runtime budget for the large
                # densification duplicate-check tensor and cap it at 1024 for
                # 24-GiB cards. Preserve Scaffold-GS's original 4096-row chunk
                # when second-order mode is disabled.
                chunk_size = (
                    min(1024, self.sogs_chunk_size)
                    if self.use_second_order
                    else 4096
                )
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                # INFERENCE: accumulate the reduction in-place instead of
                # retaining one [candidate_count] boolean tensor per chunk.
                remove_duplicates = torch.zeros(
                    selected_grid_coords_unique.shape[0],
                    dtype=torch.bool,
                    device=selected_grid_coords_unique.device,
                )
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates.logical_or_(cur_remove_duplicates)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size

            
            if candidate_anchor.shape[0] > 0:
                new_scaling = (
                    torch.ones_like(candidate_anchor)
                    .repeat([1, 2])
                    .float()
                    * cur_size
                )  # *0.05
                new_scaling = torch.log(new_scaling)
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:,0] = 1.0

                new_opacities = inverse_sigmoid(
                    0.1
                    * torch.ones(
                        (candidate_anchor.shape[0], 1),
                        dtype=torch.float,
                        device=candidate_anchor.device,
                    )
                )

                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]

                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

                new_offsets = (
                    torch.zeros_like(candidate_anchor)
                    .unsqueeze(dim=1)
                    .repeat([1, self.n_offsets, 1])
                    .float()
                )

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                }
                

                temp_anchor_demon = torch.cat(
                    [
                        self.anchor_demon,
                        torch.zeros(
                            [new_opacities.shape[0], 1],
                            device=self.anchor_demon.device,
                        ).float(),
                    ],
                    dim=0,
                )
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat(
                    [
                        self.opacity_accum,
                        torch.zeros(
                            [new_opacities.shape[0], 1],
                            device=self.opacity_accum.device,
                        ).float(),
                    ],
                    dim=0,
                )
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()
                
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]
                


    def adjust_anchor(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom # [N*k, 1]
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1)
        
        self.anchor_growing(grads_norm, grad_threshold, offset_mask)
        
        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
        
        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N] 
        
        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum
        
        # update opacity accum 
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros(
                [anchors_mask.sum(), 1], device=self.opacity_accum.device
            ).float()
            self.anchor_demon[anchors_mask] = torch.zeros(
                [anchors_mask.sum(), 1], device=self.anchor_demon.device
            ).float()
        
        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)
        
        self.max_radii2D = torch.zeros(
            (self.get_anchor.shape[0]), device=self.get_anchor.device
        )

    def _write_sogs_metadata(self, path):
        """Write architecture/configuration metadata beside every saved model."""

        metadata_path = os.path.join(path, "sogs_config.json")
        metadata = {
            "format_version": 2,
            "sogs": self.get_sogs_config(),
        }
        with open(metadata_path, "w") as metadata_file:
            json.dump(metadata, metadata_file, indent=2, sort_keys=True)

    def _read_sogs_metadata(self, path):
        metadata_path = os.path.join(path, "sogs_config.json")
        if not os.path.isfile(metadata_path):
            if self.use_second_order:
                raise RuntimeError(
                    f"missing SOGS metadata at {metadata_path}; "
                    "the checkpoint cannot be loaded in SOGS mode"
                )
            return {}
        try:
            with open(metadata_path, "r") as metadata_file:
                metadata = json.load(metadata_file)
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"could not read checkpoint metadata {metadata_path}: {error}"
            ) from error
        if not isinstance(metadata, dict) or metadata.get("format_version") != 2:
            raise RuntimeError(
                f"unsupported checkpoint metadata format in {metadata_path}; "
                "expected format_version=2"
            )
        config = metadata.get("sogs", metadata.get("config"))
        if not isinstance(config, dict):
            raise RuntimeError(
                f"checkpoint metadata {metadata_path} has no SOGS configuration"
            )
        self._assert_checkpoint_compatible(config, source=metadata_path)
        return config

    def save_mlp_checkpoints(self, path, mode="split"):
        """Save attribute and SOGS MLPs with explicit configuration metadata."""

        mkdir_p(path)
        self._write_sogs_metadata(path)
        if mode == "split":
            # COMPATIBILITY: retain the local split TorchScript layout while adding
            # one file per second-order branch.
            module_specs = (
                (
                    self.mlp_opacity,
                    "opacity_mlp.pt",
                    self.render_feat_dim + 3 + self.opacity_dist_dim,
                ),
                (
                    self.mlp_cov,
                    "cov_mlp.pt",
                    self.render_feat_dim + 3 + self.cov_dist_dim,
                ),
                (
                    self.mlp_color,
                    "color_mlp.pt",
                    self.render_feat_dim + 3 + self.color_dist_dim + self.appearance_dim,
                ),
            )
            for module, filename, input_dim in module_specs:
                was_training = module.training
                module.eval()
                parameter = next(module.parameters(), None)
                dtype = parameter.dtype if parameter is not None else torch.float32
                device = parameter.device if parameter is not None else self.device
                traced = torch.jit.trace(
                    module,
                    (torch.rand(1, input_dim, dtype=dtype, device=device),),
                )
                traced.save(os.path.join(path, filename))
                module.train(was_training)

            if self.use_feat_bank:
                was_training = self.mlp_feature_bank.training
                self.mlp_feature_bank.eval()
                parameter = next(self.mlp_feature_bank.parameters())
                traced = torch.jit.trace(
                    self.mlp_feature_bank,
                    (
                        torch.rand(
                            1, 4, dtype=parameter.dtype, device=parameter.device
                        ),
                    ),
                )
                traced.save(os.path.join(path, "feature_bank_mlp.pt"))
                self.mlp_feature_bank.train(was_training)

            if self.appearance_dim:
                if self.embedding_appearance is None:
                    raise RuntimeError(
                        "cannot save appearance MLP before set_appearance()"
                    )
                was_training = self.embedding_appearance.training
                self.embedding_appearance.eval()
                traced = torch.jit.trace(
                    self.embedding_appearance,
                    (
                        torch.zeros(
                            (1,), dtype=torch.long, device=self.device
                        ),
                    ),
                )
                traced.save(os.path.join(path, "embedding_appearance.pt"))
                self.embedding_appearance.train(was_training)

            if self.second_order_augmentor is not None:
                for index, branch in enumerate(self.second_order_augmentor.branches):
                    was_training = branch.training
                    branch.eval()
                    parameter = next(branch.parameters())
                    traced = torch.jit.trace(
                        branch,
                        (
                            torch.rand(
                                1,
                                2 * self.feat_dim,
                                dtype=parameter.dtype,
                                device=parameter.device,
                            ),
                        ),
                    )
                    traced.save(
                        os.path.join(path, f"second_order_mlp_{index}.pt")
                    )
                    branch.train(was_training)

        elif mode == "unite":
            state = {
                "format_version": 2,
                "sogs_config": self.get_sogs_config(),
                "opacity_mlp": self.mlp_opacity.state_dict(),
                "cov_mlp": self.mlp_cov.state_dict(),
                "color_mlp": self.mlp_color.state_dict(),
            }
            if self.use_feat_bank:
                state["feature_bank_mlp"] = self.mlp_feature_bank.state_dict()
            if self.appearance_dim > 0:
                if self.embedding_appearance is None:
                    raise RuntimeError(
                        "cannot save appearance MLP before set_appearance()"
                    )
                state["appearance"] = self.embedding_appearance.state_dict()
            if self.second_order_augmentor is not None:
                state["second_order"] = self.second_order_augmentor.state_dict()
            torch.save(state, os.path.join(path, "checkpoints.pth"))
        else:
            raise NotImplementedError(f"unsupported checkpoint mode: {mode}")

    def load_mlp_checkpoints(self, path, mode="split"):
        """Load MLP state and reject missing/incompatible SOGS components."""

        self._read_sogs_metadata(path)
        if mode == "split":
            required = ["opacity_mlp.pt", "cov_mlp.pt", "color_mlp.pt"]
            if self.use_feat_bank:
                required.append("feature_bank_mlp.pt")
            if self.appearance_dim > 0:
                required.append("embedding_appearance.pt")
            if self.second_order_augmentor is not None:
                required.extend(
                    f"second_order_mlp_{index}.pt"
                    for index in range(self.num_eigenvectors)
                )
            missing = [
                filename
                for filename in required
                if not os.path.isfile(os.path.join(path, filename))
            ]
            if missing:
                raise RuntimeError(
                    "checkpoint is missing required MLP files: "
                    + ", ".join(missing)
                )

            self.mlp_opacity = torch.jit.load(
                os.path.join(path, "opacity_mlp.pt"), map_location=self.device
            ).to(self.device)
            self.mlp_cov = torch.jit.load(
                os.path.join(path, "cov_mlp.pt"), map_location=self.device
            ).to(self.device)
            self.mlp_color = torch.jit.load(
                os.path.join(path, "color_mlp.pt"), map_location=self.device
            ).to(self.device)
            if self.use_feat_bank:
                self.mlp_feature_bank = torch.jit.load(
                    os.path.join(path, "feature_bank_mlp.pt"),
                    map_location=self.device,
                ).to(self.device)
            if self.appearance_dim > 0:
                self.embedding_appearance = torch.jit.load(
                    os.path.join(path, "embedding_appearance.pt"),
                    map_location=self.device,
                ).to(self.device)
            if self.second_order_augmentor is not None:
                for index in range(self.num_eigenvectors):
                    self.second_order_augmentor.branches[index] = torch.jit.load(
                        os.path.join(path, f"second_order_mlp_{index}.pt"),
                        map_location=self.device,
                    ).to(self.device)
        elif mode == "unite":
            checkpoint_path = os.path.join(path, "checkpoints.pth")
            if not os.path.isfile(checkpoint_path):
                raise RuntimeError(f"missing united checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if not isinstance(checkpoint, dict):
                raise RuntimeError(
                    f"unsupported united checkpoint format in {checkpoint_path}; "
                    "expected an object"
                )
            if checkpoint.get("format_version") != 2 and self.use_second_order:
                raise RuntimeError(
                    f"united checkpoint {checkpoint_path} has no SOGS format "
                    "metadata; cannot load it in SOGS mode"
                )
            self._assert_checkpoint_compatible(
                checkpoint.get("sogs_config"), source=checkpoint_path
            )
            required = {"opacity_mlp", "cov_mlp", "color_mlp"}
            if self.use_feat_bank:
                required.add("feature_bank_mlp")
            if self.appearance_dim > 0:
                required.add("appearance")
            if self.second_order_augmentor is not None:
                required.add("second_order")
            missing = sorted(required.difference(checkpoint.keys()))
            if missing:
                raise RuntimeError(
                    "united checkpoint is missing required state: "
                    + ", ".join(missing)
                )
            self._load_module_state(
                self.mlp_opacity, checkpoint["opacity_mlp"], "opacity_mlp"
            )
            self._load_module_state(self.mlp_cov, checkpoint["cov_mlp"], "cov_mlp")
            self._load_module_state(
                self.mlp_color, checkpoint["color_mlp"], "color_mlp"
            )
            if self.use_feat_bank:
                self._load_module_state(
                    self.mlp_feature_bank,
                    checkpoint["feature_bank_mlp"],
                    "feature_bank_mlp",
                )
            if self.appearance_dim > 0:
                self._load_module_state(
                    self.embedding_appearance,
                    checkpoint["appearance"],
                    "appearance_embedding",
                )
            if self.second_order_augmentor is not None:
                self._load_module_state(
                    self.second_order_augmentor,
                    checkpoint["second_order"],
                    "second_order_augmentor",
                )
        else:
            raise NotImplementedError(f"unsupported checkpoint mode: {mode}")
