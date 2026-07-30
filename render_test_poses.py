#!/usr/bin/env python3
"""Render each scene's CSV test poses from a trained Scaffold-GS model."""

from __future__ import annotations

import argparse
import ast
import csv
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from arguments import str2bool
from prepare_data import CSV_COLUMNS, discover_scenes, locate_test_pose_csv


def configure_cuda_device(argv: Sequence[str]) -> None:
    """Set a requested GPU before importing PyTorch."""
    requested_gpu: Optional[str] = None
    for index, argument in enumerate(argv):
        if argument == "--gpu" and index + 1 < len(argv):
            requested_gpu = argv[index + 1]
        elif argument.startswith("--gpu="):
            requested_gpu = argument.split("=", maxsplit=1)[1]
    if requested_gpu and requested_gpu != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = requested_gpu


def read_config_namespace(path: Path) -> Dict[str, Any]:
    """Read the trusted ``cfg_args`` Namespace written by train.py safely."""
    expression = ast.parse(path.read_text(encoding="utf-8"), mode="eval").body
    if not isinstance(expression, ast.Call) or not isinstance(
        expression.func, ast.Name
    ) or expression.func.id != "Namespace":
        raise ValueError(f"Unsupported cfg_args format: {path}")
    if expression.args or any(keyword.arg is None for keyword in expression.keywords):
        raise ValueError(f"Unsupported positional cfg_args value: {path}")

    values: Dict[str, Any] = {}
    for keyword in expression.keywords:
        try:
            values[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, TypeError) as error:
            raise ValueError(
                f"Non-literal value in cfg_args ({keyword.arg}): {path}"
            ) from error
    return values


def find_iteration(model_dir: Path, requested: int) -> Tuple[int, Path]:
    """Resolve a requested iteration or the latest saved iteration."""
    point_cloud_dir = model_dir / "point_cloud"
    candidates: Dict[int, Path] = {}
    for path in point_cloud_dir.glob("iteration_*"):
        try:
            iteration = int(path.name.split("_", maxsplit=1)[1])
        except (IndexError, ValueError):
            continue
        if path.is_dir():
            candidates[iteration] = path
    if not candidates:
        raise FileNotFoundError(f"No saved point-cloud iterations in {point_cloud_dir}")
    iteration = max(candidates) if requested == -1 else requested
    if iteration not in candidates:
        raise FileNotFoundError(
            f"Iteration {iteration} is not present in {point_cloud_dir}"
        )
    return iteration, candidates[iteration]


def load_pose_rows(
    csv_path: Path,
) -> List[Dict[str, Union[float, int, str]]]:
    """Read and validate all numeric fields needed to render test cameras."""
    rows: List[Dict[str, Union[float, int, str]]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            raise ValueError(
                f"{csv_path} header must be exactly {','.join(CSV_COLUMNS)}"
            )
        for line_number, raw_row in enumerate(reader, start=2):
            if None in raw_row:
                raise ValueError(f"{csv_path}:{line_number} has extra columns")
            row: Dict[str, Union[float, int, str]] = {}
            image_name = (raw_row.get("image_name") or "").strip()
            if (
                not image_name
                or Path(image_name).name != image_name
                or "\\" in image_name
            ):
                raise ValueError(
                    f"{csv_path}:{line_number} has an invalid image_name"
                )
            row["image_name"] = image_name
            for column in CSV_COLUMNS[1:12]:
                try:
                    value = float(raw_row[column])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"{csv_path}:{line_number} has invalid {column}"
                    ) from error
                if not math.isfinite(value):
                    raise ValueError(
                        f"{csv_path}:{line_number} has non-finite {column}"
                    )
                row[column] = value
            for column in ("width", "height"):
                try:
                    value = float(raw_row[column])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"{csv_path}:{line_number} has invalid {column}"
                    ) from error
                if not value.is_integer() or value <= 0:
                    raise ValueError(
                        f"{csv_path}:{line_number} has invalid {column}"
                    )
                row[column] = int(value)

            quaternion_norm = math.sqrt(
                sum(float(row[key]) ** 2 for key in ("qw", "qx", "qy", "qz"))
            )
            if abs(quaternion_norm - 1.0) > 1e-3:
                raise ValueError(
                    f"{csv_path}:{line_number} quaternion norm is "
                    f"{quaternion_norm:.6f}"
                )
            if float(row["fx"]) <= 0 or float(row["fy"]) <= 0:
                raise ValueError(f"{csv_path}:{line_number} focal lengths must be positive")
            if not 0 <= float(row["cx"]) <= int(row["width"]):
                raise ValueError(f"{csv_path}:{line_number} cx is outside the image")
            if not 0 <= float(row["cy"]) <= int(row["height"]):
                raise ValueError(f"{csv_path}:{line_number} cy is outside the image")
            rows.append(row)

    output_names = [f"{Path(str(row['image_name'])).stem}.png" for row in rows]
    if len({name.casefold() for name in output_names}) != len(output_names):
        raise ValueError(f"{csv_path} contains duplicate normalized PNG names")
    if not rows:
        raise ValueError(f"{csv_path} contains no test poses")
    return rows


def make_test_camera(
    row: Mapping[str, Union[float, int, str]],
    *,
    torch: Any,
    np: Any,
    MiniCam: Any,
    qvec2rotmat: Any,
    getWorld2View2: Any,
    focal2fov: Any,
    appearance_index: int,
) -> Any:
    """Construct a Scaffold-GS MiniCam, including off-center intrinsics."""
    width = int(row["width"])
    height = int(row["height"])
    fx = float(row["fx"])
    fy = float(row["fy"])
    cx = float(row["cx"])
    cy = float(row["cy"])

    quaternion = np.asarray(
        [float(row[key]) for key in ("qw", "qx", "qy", "qz")],
        dtype=np.float64,
    )
    quaternion /= np.linalg.norm(quaternion)
    rotation = qvec2rotmat(quaternion).T
    translation = np.asarray(
        [float(row[key]) for key in ("tx", "ty", "tz")], dtype=np.float64
    )

    world_view_transform = torch.tensor(
        getWorld2View2(rotation, translation),
        dtype=torch.float32,
        device="cuda",
    ).transpose(0, 1)

    # This is the same perspective matrix used by Scaffold-GS, with the
    # principal-point offsets retained instead of assuming a centered camera.
    znear, zfar = 0.01, 100.0
    projection = torch.zeros((4, 4), dtype=torch.float32, device="cuda")
    projection[0, 0] = 2.0 * fx / width
    projection[1, 1] = 2.0 * fy / height
    projection[0, 2] = 2.0 * cx / width - 1.0
    projection[1, 2] = 2.0 * cy / height - 1.0
    projection[2, 2] = zfar / (zfar - znear)
    projection[2, 3] = -(zfar * znear) / (zfar - znear)
    projection[3, 2] = 1.0
    projection = projection.transpose(0, 1)
    full_projection = (
        world_view_transform.unsqueeze(0)
        .bmm(projection.unsqueeze(0))
        .squeeze(0)
    )

    camera = MiniCam(
        width=width,
        height=height,
        fovy=focal2fov(fy, height),
        fovx=focal2fov(fx, width),
        znear=znear,
        zfar=zfar,
        world_view_transform=world_view_transform,
        full_proj_transform=full_projection,
    )
    # These attributes are only needed when appearance embeddings are enabled.
    camera.uid = appearance_index
    camera.image_name = str(row["image_name"])
    return camera


def render_scene(
    scene_dir: Path,
    model_dir: Path,
    prediction_dir: Path,
    *,
    iteration: int,
    white_background: bool,
    appearance_index: int,
) -> int:
    """Load one checkpoint and render all of its requested poses."""
    import numpy as np
    import torch
    import torchvision
    from gaussian_renderer import GaussianModel, prefilter_voxel, render
    from scene.cameras import MiniCam
    from scene.colmap_loader import qvec2rotmat
    from utils.graphics_utils import focal2fov, getWorld2View2
    from utils.runtime_utils import configure_torch_runtime

    cfg_path = model_dir / "cfg_args"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing training config: {cfg_path}")
    config = read_config_namespace(cfg_path)
    configure_torch_runtime(
        tf32_mode=str(config.get("tf32_mode", "default")),
        cudnn_benchmark=str2bool(config.get("cudnn_benchmark", False)),
    )
    selected_iteration, iteration_dir = find_iteration(model_dir, iteration)

    model = GaussianModel(
        feat_dim=int(config.get("feat_dim", 32)),
        n_offsets=int(config.get("n_offsets", 10)),
        voxel_size=float(config.get("voxel_size", 0.001)),
        update_depth=int(config.get("update_depth", 3)),
        update_init_factor=int(config.get("update_init_factor", 16)),
        update_hierachy_factor=int(config.get("update_hierachy_factor", 4)),
        use_feat_bank=bool(config.get("use_feat_bank", False)),
        appearance_dim=int(config.get("appearance_dim", 0)),
        ratio=int(config.get("ratio", 1)),
        add_opacity_dist=bool(config.get("add_opacity_dist", False)),
        add_cov_dist=bool(config.get("add_cov_dist", False)),
        add_color_dist=bool(config.get("add_color_dist", False)),
        use_second_order=str2bool(config.get("use_second_order", False)),
        num_eigenvectors=int(config.get("num_eigenvectors", 2)),
        lambda_sgl=float(config.get("lambda_sgl", 0.01)),
        sogs_chunk_size=int(config.get("sogs_chunk_size", 2048)),
        sogs_checkpointing=str2bool(
            config.get("sogs_checkpointing", True)
        ),
        sogs_validate_numerics=str2bool(
            config.get("sogs_validate_numerics", True)
        ),
        sogs_cache_render_features=str2bool(
            config.get("sogs_cache_render_features", False)
        ),
    )
    model.load_ply_sparse_gaussian(str(iteration_dir / "point_cloud.ply"))
    model.load_mlp_checkpoints(str(iteration_dir))
    model.eval()

    if model.appearance_dim > 0:
        embedding = model.get_appearance
        if appearance_index < 0:
            raise ValueError("--appearance-index must be non-negative")
        try:
            embedding_size = int(
                embedding(
                    torch.zeros(1, dtype=torch.long, device="cuda")
                ).shape[1]
            )
        except (AttributeError, IndexError, RuntimeError) as error:
            raise ValueError(
                "Could not inspect the appearance embedding; use "
                "--appearance-dim 0 during training or provide a valid index"
            ) from error
        if appearance_index >= embedding_size:
            raise ValueError(
                f"--appearance-index {appearance_index} is outside the "
                f"embedding range [0, {embedding_size})"
            )

    pipeline = SimpleNamespace(
        debug=bool(config.get("debug", False)),
        compute_cov3D_python=bool(config.get("compute_cov3D_python", False)),
        convert_SHs_python=bool(config.get("convert_SHs_python", False)),
    )
    use_white_background = white_background or bool(
        config.get("white_background", False)
    )
    background = torch.tensor(
        [1.0, 1.0, 1.0] if use_white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )

    rows = load_pose_rows(locate_test_pose_csv(scene_dir))
    prediction_dir.mkdir(parents=True, exist_ok=True)
    rendered_count = 0
    with torch.no_grad():
        for row in rows:
            camera = make_test_camera(
                row,
                torch=torch,
                np=np,
                MiniCam=MiniCam,
                qvec2rotmat=qvec2rotmat,
                getWorld2View2=getWorld2View2,
                focal2fov=focal2fov,
                appearance_index=appearance_index,
            )
            visible_mask = prefilter_voxel(camera, model, pipeline, background)
            rendered = render(
                camera,
                model,
                pipeline,
                background,
                visible_mask=visible_mask,
            )["render"].clamp(0.0, 1.0)
            output_name = f"{Path(str(row['image_name'])).stem}.png"
            torchvision.utils.save_image(
                rendered, str(prediction_dir / output_name)
            )
            rendered_count += 1

    print(
        f"[{scene_dir.name}] rendered {rendered_count} pose(s) at "
        f"iteration {selected_iteration} -> {prediction_dir}"
    )
    return rendered_count


def build_parser() -> argparse.ArgumentParser:
    """Build the rendering command-line parser."""
    parser = argparse.ArgumentParser(
        description="Render CSV test poses from trained Scaffold-GS checkpoints."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("outputs") / "round1",
        help="Directory containing one trained model directory per scene.",
    )
    parser.add_argument(
        "--predictions-root",
        type=Path,
        default=Path("predictions") / "round1",
    )
    parser.add_argument("--scenes", nargs="*", default=())
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--gpu", default="-1")
    parser.add_argument("--appearance-index", type=int, default=0)
    parser.add_argument("--white-background", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Render all requested scenes."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    configure_cuda_device(raw_argv)
    args = build_parser().parse_args(raw_argv)
    try:
        scene_dirs = discover_scenes(args.data_root.resolve(), args.scenes)
    except (FileNotFoundError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    succeeded = True
    for scene_dir in scene_dirs:
        try:
            render_scene(
                scene_dir=scene_dir.resolve(),
                model_dir=(args.model_root / scene_dir.name).resolve(),
                prediction_dir=(args.predictions_root / scene_dir.name).resolve(),
                iteration=args.iteration,
                white_background=args.white_background,
                appearance_index=args.appearance_index,
            )
        except Exception as error:  # one bad scene should be reported clearly
            succeeded = False
            print(f"[{scene_dir.name}] render failed: {error}", file=sys.stderr)
            if not args.continue_on_error:
                return 1
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
