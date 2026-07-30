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

from argparse import ArgumentParser, Namespace
import sys
import os
import math
import numbers


def str2bool(value):
    """Parse a boolean safely from CLI/config-friendly spellings.

    ``type=bool`` is intentionally avoided because ``bool("False")`` is
    ``True``.  The optional argument value also lets existing flags such as
    ``--eval`` continue to mean ``True`` while accepting
    ``--use_second_order False``.
    """

    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(
        f"invalid boolean value {value!r}; expected True/False, 1/0, yes/no, or on/off"
    )


def validate_sogs_config(
    feat_dim,
    use_second_order=False,
    num_eigenvectors=2,
    lambda_sgl=0.01,
):
    """Validate explicit SOGS settings and return normalized values.

    The function is deliberately independent of the parser so checkpoint and
    renderer entry points can validate values loaded from ``cfg_args`` too.
    """

    if isinstance(feat_dim, bool) or (
        not isinstance(feat_dim, (str, numbers.Integral))
    ):
        raise ValueError("feat_dim must be a positive integer")
    try:
        normalized_feat_dim = int(feat_dim)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("feat_dim must be a positive integer") from error
    if normalized_feat_dim <= 0:
        raise ValueError("feat_dim must be greater than zero")

    try:
        normalized_enabled = str2bool(use_second_order)
    except (TypeError, ValueError) as error:
        raise ValueError("use_second_order must be a boolean") from error

    if isinstance(num_eigenvectors, bool) or (
        not isinstance(num_eigenvectors, (str, numbers.Integral))
    ):
        raise ValueError("num_eigenvectors must be an integer")
    try:
        normalized_eigenvectors = int(num_eigenvectors)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("num_eigenvectors must be an integer") from error
    if normalized_eigenvectors < 0:
        raise ValueError("num_eigenvectors must be non-negative")
    if normalized_enabled and normalized_eigenvectors < 1:
        raise ValueError(
            "num_eigenvectors must be at least 1 when use_second_order is True"
        )
    if normalized_eigenvectors > normalized_feat_dim:
        raise ValueError(
            "num_eigenvectors must not exceed feat_dim "
            f"({normalized_eigenvectors} > {normalized_feat_dim})"
        )

    if isinstance(lambda_sgl, bool):
        raise ValueError("lambda_sgl must be a non-negative number")
    try:
        normalized_lambda = float(lambda_sgl)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("lambda_sgl must be a non-negative number") from error
    if not math.isfinite(normalized_lambda) or normalized_lambda < 0:
        raise ValueError("lambda_sgl must be non-negative")

    return (
        normalized_feat_dim,
        normalized_enabled,
        normalized_eigenvectors,
        normalized_lambda,
    )


def validate_sogs_chunk_size(value=2048):
    """Validate the maximum anchor chunk used by memory-efficient SOGS."""

    if isinstance(value, bool) or not isinstance(value, (str, numbers.Integral)):
        raise ValueError("sogs_chunk_size must be a positive integer")
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("sogs_chunk_size must be a positive integer") from error
    if normalized <= 0:
        raise ValueError("sogs_chunk_size must be a positive integer")
    return normalized


def validate_positive_int(value, *, name):
    """Return a positive integer with a field-specific error message."""

    if isinstance(value, bool) or not isinstance(value, (str, numbers.Integral)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if normalized <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return normalized


def validate_tf32_mode(value="default"):
    """Validate the opt-in CUDA float32 matmul policy."""

    normalized = str(value).strip().lower()
    if normalized not in {"default", "enabled", "disabled"}:
        raise ValueError(
            "tf32_mode must be one of: default, enabled, disabled"
        )
    return normalized


def validate_optimizer_backend(value="default"):
    """Validate the requested Adam implementation."""

    normalized = str(value).strip().lower()
    if normalized not in {"default", "auto", "foreach", "fused", "single"}:
        raise ValueError(
            "optimizer_backend must be one of: "
            "default, auto, foreach, fused, single"
        )
    return normalized


class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument(
                        "--" + key,
                        ("-" + key[0:1]),
                        default=value,
                        nargs="?",
                        const=True,
                        type=str2bool,
                    )
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument(
                        "--" + key,
                        default=value,
                        nargs="?",
                        const=True,
                        type=str2bool,
                    )
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self.feat_dim = 32
        # PAPER: SOGS is opt-in so the original Scaffold-GS path remains the
        # default.  M=2 and lambda=.01 are the paper's reference settings.
        self.use_second_order = False
        self.num_eigenvectors = 2
        self.lambda_sgl = 0.01
        # INFERENCE: 2048 bounds SOGS branch activation memory on 24-GiB GPUs.
        self.sogs_chunk_size = 2048
        # INFERENCE: checkpoint recomputation and finite-value checks are safe
        # defaults. Large-memory profiles may disable them explicitly.
        self.sogs_checkpointing = True
        self.sogs_validate_numerics = True
        # INFERENCE: full augmented-feature caching is useful for multi-view
        # inference but is opt-in because its memory scales with N*D*(1+M).
        self.sogs_cache_render_features = False
        # COMPATIBILITY: runtime math defaults remain owned by the installed
        # PyTorch build unless an experiment explicitly selects a policy.
        self.tf32_mode = "default"
        self.cudnn_benchmark = False
        self.n_offsets = 10
        self.voxel_size =  0.001 # if voxel_size<=0, using 1nn dist
        self.update_depth = 3
        self.update_init_factor = 16
        self.update_hierachy_factor = 4

        self.use_feat_bank = False
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.lod = 0

        self.appearance_dim = 32
        self.lowpoly = False
        self.ds = 1
        self.ratio = 1 # sampling the input point cloud
        self.undistorted = False 
        
        # In the Bungeenerf dataset, we propose to set the following three parameters to True,
        # Because there are enough dist variations.
        self.add_opacity_dist = False
        self.add_cov_dist = False
        self.add_color_dist = False
        
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        # ``sentinel=True`` is used by render.py; fill missing fields from
        # stable baseline/reference defaults when loading an old cfg_args.
        if getattr(g, "feat_dim", None) is None:
            g.feat_dim = 32
        if getattr(g, "use_second_order", None) is None:
            g.use_second_order = False
        if getattr(g, "num_eigenvectors", None) is None:
            g.num_eigenvectors = 2
        if getattr(g, "lambda_sgl", None) is None:
            g.lambda_sgl = 0.01
        if getattr(g, "sogs_chunk_size", None) is None:
            g.sogs_chunk_size = 2048
        if getattr(g, "sogs_checkpointing", None) is None:
            g.sogs_checkpointing = True
        if getattr(g, "sogs_validate_numerics", None) is None:
            g.sogs_validate_numerics = True
        if getattr(g, "sogs_cache_render_features", None) is None:
            g.sogs_cache_render_features = False
        if getattr(g, "tf32_mode", None) is None:
            g.tf32_mode = "default"
        if getattr(g, "cudnn_benchmark", None) is None:
            g.cudnn_benchmark = False
        (
            g.feat_dim,
            g.use_second_order,
            g.num_eigenvectors,
            g.lambda_sgl,
        ) = validate_sogs_config(
            g.feat_dim,
            g.use_second_order,
            g.num_eigenvectors,
            g.lambda_sgl,
        )
        g.sogs_chunk_size = validate_sogs_chunk_size(g.sogs_chunk_size)
        g.sogs_checkpointing = str2bool(g.sogs_checkpointing)
        g.sogs_validate_numerics = str2bool(g.sogs_validate_numerics)
        g.sogs_cache_render_features = str2bool(
            g.sogs_cache_render_features
        )
        g.tf32_mode = validate_tf32_mode(g.tf32_mode)
        g.cudnn_benchmark = str2bool(g.cudnn_benchmark)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.0
        self.position_lr_final = 0.0
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        
        self.offset_lr_init = 0.01
        self.offset_lr_final = 0.0001
        self.offset_lr_delay_mult = 0.01
        self.offset_lr_max_steps = 30_000

        self.feature_lr = 0.0075
        self.opacity_lr = 0.02
        self.scaling_lr = 0.007
        self.rotation_lr = 0.002
        
        
        self.mlp_opacity_lr_init = 0.002
        self.mlp_opacity_lr_final = 0.00002  
        self.mlp_opacity_lr_delay_mult = 0.01
        self.mlp_opacity_lr_max_steps = 30_000

        self.mlp_cov_lr_init = 0.004
        self.mlp_cov_lr_final = 0.004
        self.mlp_cov_lr_delay_mult = 0.01
        self.mlp_cov_lr_max_steps = 30_000
        
        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000

        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000
        
        self.mlp_featurebank_lr_init = 0.01
        self.mlp_featurebank_lr_final = 0.00001
        self.mlp_featurebank_lr_delay_mult = 0.01
        self.mlp_featurebank_lr_max_steps = 30_000

        self.appearance_lr_init = 0.05
        self.appearance_lr_final = 0.0005
        self.appearance_lr_delay_mult = 0.01
        self.appearance_lr_max_steps = 30_000

        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        
        # for anchor densification
        self.start_stat = 500
        self.update_from = 1500
        self.update_interval = 100
        self.update_until = 15_000
        
        self.min_opacity = 0.005
        self.success_threshold = 0.8
        self.densify_grad_threshold = 0.0002
        # COMPATIBILITY: "default" constructs Adam exactly as before. A100
        # profiles may request a supported fused/foreach implementation.
        self.optimizer_backend = "default"
        # INFERENCE: duplicate checks are tiled independently from SOGS feature
        # activations because their temporary memory scales differently.
        self.densification_chunk_size = 4096

        super().__init__(parser, "Optimization Parameters")

    def extract(self, args):
        g = super().extract(args)
        g.optimizer_backend = validate_optimizer_backend(
            g.optimizer_backend
        )
        g.densification_chunk_size = validate_positive_int(
            g.densification_chunk_size,
            name="densification_chunk_size",
        )
        return g

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
