#!/usr/bin/env python3
"""Launch Scaffold-GS training once for each Viettel AI Race scene.

The launcher intentionally does not copy or move data.  The competition's
``train`` directory is passed directly as Scaffold-GS's source path.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from arguments import (
    str2bool,
    validate_optimizer_backend,
    validate_positive_int,
    validate_sogs_chunk_size,
    validate_sogs_config,
    validate_tf32_mode,
)
from prepare_data import discover_scenes, validate_scene
from utils.training_budget import (
    validate_checkpoint_interval,
    validate_max_runtime_minutes,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the launcher command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate and train each scene sequentially. Training is started "
            "only when this script is explicitly executed."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root containing scene directories (default: data).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs") / "round1",
        help="Directory receiving one model directory per scene.",
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=(),
        help="Optional scene names; all scenes are used when omitted.",
    )
    parser.add_argument("--iterations", type=int, default=30_000)
    parser.add_argument(
        "--max-runtime-minutes",
        type=float,
        default=0.0,
        help=(
            "Per-scene training budget. The child process saves a model and "
            "resumable checkpoint before stopping; zero disables the limit."
        ),
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=0,
        help=(
            "Save a resumable checkpoint every N iterations; zero disables "
            "periodic checkpoints."
        ),
    )
    parser.add_argument("--voxel-size", type=float, default=0.001)
    parser.add_argument("--update-init-factor", type=int, default=16)
    parser.add_argument("--feat-dim", type=int, default=32)
    parser.add_argument(
        "--n-offsets",
        type=int,
        default=10,
        help="Offsets predicted per anchor; lower values reduce renderer VRAM.",
    )
    parser.add_argument(
        "--use-second-order",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable SOGS second-order anchor features.",
    )
    parser.add_argument("--num-eigenvectors", type=int, default=2)
    parser.add_argument("--lambda-sgl", type=float, default=0.01)
    parser.add_argument(
        "--sogs-chunk-size",
        type=int,
        default=2048,
        help=(
            "Maximum anchors per SOGS/renderer activation chunk "
            "(lower uses less VRAM)."
        ),
    )
    parser.add_argument(
        "--sogs-checkpointing",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Recompute SOGS/attribute MLP activations during backward.",
    )
    parser.add_argument(
        "--sogs-validate-numerics",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Run full finite-value reductions inside SOGS and SGL.",
    )
    parser.add_argument(
        "--sogs-cache-render-features",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Cache all augmented anchors during no-grad evaluation.",
    )
    parser.add_argument(
        "--tf32-mode",
        default="default",
        choices=("default", "enabled", "disabled"),
        help="CUDA float32 matmul policy.",
    )
    parser.add_argument(
        "--cudnn-benchmark",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable the cuDNN convolution autotuner.",
    )
    parser.add_argument(
        "--optimizer-backend",
        default="default",
        choices=("default", "auto", "foreach", "fused", "single"),
        help="Adam implementation requested from the installed PyTorch.",
    )
    parser.add_argument(
        "--densification-chunk-size",
        type=int,
        default=4096,
        help="Anchor rows per duplicate-check tile during densification.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="Write synchronized scalar logs every N iterations.",
    )
    parser.add_argument(
        "--disable-gui",
        action="store_true",
        help="Disable the training GUI connection check.",
    )
    parser.add_argument(
        "--appearance-dim",
        type=int,
        default=0,
        help="Use 0 for a pose-only test camera with no unknown appearance ID.",
    )
    parser.add_argument("--ratio", type=int, default=1)
    parser.add_argument("--gpu", default="-1", help="Physical GPU id, or -1.")
    parser.add_argument(
        "--white-background",
        action="store_true",
        help="Train with a white background and use it during rendering.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run the repository's optional second warmup pass.",
    )
    parser.add_argument("--port-base", type=int, default=6009)
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help=(
            "Permit a non-empty model directory. Existing files are retained "
            "but may be overwritten by train.py."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print commands without starting training.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to later scenes if one subprocess fails.",
    )
    return parser


def training_command(
    scene_dir: Path,
    model_dir: Path,
    args: argparse.Namespace,
    port: int,
) -> List[str]:
    """Build one reproducible ``train.py`` command."""
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "train.py"),
        "--source_path",
        str((scene_dir / "train").resolve()),
        "--model_path",
        str(model_dir.resolve()),
        "--voxel_size",
        str(args.voxel_size),
        "--update_init_factor",
        str(args.update_init_factor),
        "--feat_dim",
        str(args.feat_dim),
        "--n_offsets",
        str(args.n_offsets),
        "--use_second_order",
        str(args.use_second_order),
        "--num_eigenvectors",
        str(args.num_eigenvectors),
        "--lambda_sgl",
        str(args.lambda_sgl),
        "--sogs_chunk_size",
        str(args.sogs_chunk_size),
        "--sogs_checkpointing",
        str(args.sogs_checkpointing),
        "--sogs_validate_numerics",
        str(args.sogs_validate_numerics),
        "--sogs_cache_render_features",
        str(args.sogs_cache_render_features),
        "--tf32_mode",
        str(args.tf32_mode),
        "--cudnn_benchmark",
        str(args.cudnn_benchmark),
        "--optimizer_backend",
        str(args.optimizer_backend),
        "--densification_chunk_size",
        str(args.densification_chunk_size),
        "--log_interval",
        str(args.log_interval),
        "--appearance_dim",
        str(args.appearance_dim),
        "--ratio",
        str(args.ratio),
        "--iterations",
        str(args.iterations),
        "--test_iterations",
        str(args.iterations),
        "--save_iterations",
        str(args.iterations),
        "--max_runtime_minutes",
        str(args.max_runtime_minutes),
        "--checkpoint_interval",
        str(args.checkpoint_interval),
        "--port",
        str(port),
        "--gpu",
        str(args.gpu),
        "--skip_postprocess",
    ]
    if args.warmup:
        command.append("--warmup")
    if args.white_background:
        command.append("--white_background")
    if args.disable_gui:
        command.append("--disable_gui")
    return command


def validate_selected_scenes(
    scene_dirs: Sequence[Path],
) -> bool:
    """Print validation results and return whether all scenes are valid."""
    all_valid = True
    for scene_dir in scene_dirs:
        report = validate_scene(scene_dir)
        if report.warnings:
            for warning in report.warnings:
                print(f"[{scene_dir.name}] warning: {warning}")
        if report.errors:
            all_valid = False
            for error in report.errors:
                print(f"[{scene_dir.name}] error: {error}")
        else:
            print(
                f"[{scene_dir.name}] validated: "
                f"{report.train_images} train images, "
                f"{report.test_poses} test poses"
            )
    return all_valid


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Validate scenes and optionally launch their training subprocesses."""
    args = build_parser().parse_args(argv)
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()

    if args.iterations <= 0:
        print("--iterations must be positive", file=sys.stderr)
        return 2
    if args.appearance_dim < 0 or args.ratio <= 0 or args.n_offsets <= 0:
        print(
            "--appearance-dim must be non-negative, --ratio positive, "
            "and --n-offsets positive",
            file=sys.stderr,
        )
        return 2
    try:
        validate_sogs_config(
            args.feat_dim,
            args.use_second_order,
            args.num_eigenvectors,
            args.lambda_sgl,
        )
        validate_sogs_chunk_size(args.sogs_chunk_size)
        validate_tf32_mode(args.tf32_mode)
        validate_optimizer_backend(args.optimizer_backend)
        args.densification_chunk_size = validate_positive_int(
            args.densification_chunk_size,
            name="densification_chunk_size",
        )
        args.log_interval = validate_positive_int(
            args.log_interval,
            name="log_interval",
        )
        args.max_runtime_minutes = validate_max_runtime_minutes(
            args.max_runtime_minutes
        )
        args.checkpoint_interval = validate_checkpoint_interval(
            args.checkpoint_interval
        )
    except ValueError as error:
        print(f"invalid training configuration: {error}", file=sys.stderr)
        return 2

    try:
        scene_dirs = discover_scenes(data_root, args.scenes)
    except (FileNotFoundError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    if not validate_selected_scenes(scene_dirs):
        print("Validation failed; no training was started.", file=sys.stderr)
        return 1

    print(
        "Data policy: using each scene's existing train/ directory directly; "
        "no files will be moved or copied."
    )
    if args.dry_run:
        print("Dry run requested; no training was started.")

    for index, scene_dir in enumerate(scene_dirs):
        model_dir = output_root / scene_dir.name
        try:
            if model_dir.exists() and any(model_dir.iterdir()):
                if not args.allow_existing_output:
                    raise FileExistsError(
                        f"non-empty output exists: {model_dir}; choose another "
                        "output root or pass --allow-existing-output"
                    )
            elif not args.dry_run:
                model_dir.mkdir(parents=True, exist_ok=True)

            command = training_command(
                scene_dir=scene_dir,
                model_dir=model_dir,
                args=args,
                port=args.port_base + index,
            )
            print("\n[" + scene_dir.name + "] " + " ".join(command))
            if args.dry_run:
                continue

            subprocess.run(command, check=True, cwd=Path(__file__).resolve().parent)
            print(f"[{scene_dir.name}] training completed")
        except (OSError, subprocess.CalledProcessError) as error:
            print(f"[{scene_dir.name}] training failed: {error}", file=sys.stderr)
            if not args.continue_on_error:
                return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
