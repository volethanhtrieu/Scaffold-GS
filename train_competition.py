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

from prepare_data import discover_scenes, validate_scene


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
    parser.add_argument("--voxel-size", type=float, default=0.001)
    parser.add_argument("--update-init-factor", type=int, default=16)
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
    if args.appearance_dim < 0 or args.ratio <= 0:
        print("--appearance-dim must be non-negative and --ratio positive", file=sys.stderr)
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
