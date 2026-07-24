#!/usr/bin/env python3
"""Validate rendered predictions and build the competition ZIP archive."""

from __future__ import annotations

import argparse
import csv
import os
import struct
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from prepare_data import (
    CSV_COLUMNS,
    SceneReport,
    discover_scenes,
    locate_test_pose_csv,
    read_and_validate_test_poses,
)


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_START = b"\xff\xd8"
JPEG_SOF_MARKERS = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}


class SubmissionError(ValueError):
    """Raised when predictions cannot satisfy the ZIP contract."""


def read_png_size(path: Path) -> Tuple[int, int]:
    """Read width and height from a PNG IHDR header."""
    with path.open("rb") as handle:
        if handle.read(8) != PNG_SIGNATURE:
            raise SubmissionError(f"{path} is not a PNG file")
        length_bytes = handle.read(4)
        chunk_type = handle.read(4)
        if len(length_bytes) != 4 or chunk_type != b"IHDR":
            raise SubmissionError(f"{path} has no PNG IHDR header")
        (length,) = struct.unpack(">I", length_bytes)
        if length != 13:
            raise SubmissionError(f"{path} has an invalid PNG IHDR length")
        ihdr = handle.read(13)
        if len(ihdr) != 13:
            raise SubmissionError(f"{path} has a truncated PNG IHDR header")
        width, height, bit_depth, color_type = struct.unpack(">IIBB", ihdr[:10])
        if width <= 0 or height <= 0:
            raise SubmissionError(f"{path} has invalid PNG dimensions")
        if bit_depth != 8 or color_type not in (2, 6):
            raise SubmissionError(
                f"{path} is not an 8-bit RGB/RGBA PNG (color type {color_type})"
            )
        return width, height


def read_jpeg_size(path: Path) -> Tuple[int, int]:
    """Read width and height from a JPEG SOF marker."""
    with path.open("rb") as handle:
        if handle.read(2) != JPEG_START:
            raise SubmissionError(f"{path} is not a JPEG file")
        while True:
            byte = handle.read(1)
            if not byte:
                break
            if byte != b"\xff":
                continue
            marker_byte = handle.read(1)
            while marker_byte == b"\xff":
                marker_byte = handle.read(1)
            if not marker_byte:
                break
            marker = marker_byte[0]
            if marker in (0xD8, 0xD9):
                continue
            length_bytes = handle.read(2)
            if len(length_bytes) != 2:
                break
            (segment_length,) = struct.unpack(">H", length_bytes)
            if segment_length < 2:
                raise SubmissionError(f"{path} has an invalid JPEG segment")
            if marker in JPEG_SOF_MARKERS:
                payload = handle.read(5)
                if len(payload) != 5:
                    break
                height, width = struct.unpack(">HH", payload[1:5])
                return width, height
            handle.seek(segment_length - 2, 1)
    raise SubmissionError(f"Could not find JPEG dimensions in {path}")


def read_image_size(path: Path) -> Tuple[int, int]:
    """Read dimensions for the formats accepted by the submission helper."""
    with path.open("rb") as handle:
        signature = handle.read(8)
    if signature == PNG_SIGNATURE:
        return read_png_size(path)
    if signature[:2] == JPEG_START:
        return read_jpeg_size(path)
    raise SubmissionError(
        f"{path} is neither PNG nor JPEG; predictions must be RGB images"
    )


def read_pose_manifest(
    csv_path: Path,
) -> List[Tuple[str, str, int, int]]:
    """Return (CSV name, archive name, width, height) entries."""
    entries: List[Tuple[str, str, int, int]] = []
    seen: Set[str] = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            raise SubmissionError(
                f"{csv_path} header must be exactly {','.join(CSV_COLUMNS)}"
            )
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise SubmissionError(f"{csv_path}:{line_number} has extra columns")
            image_name = (row.get("image_name") or "").strip()
            if (
                not image_name
                or Path(image_name).name != image_name
                or "\\" in image_name
            ):
                raise SubmissionError(
                    f"{csv_path}:{line_number} has an invalid image_name"
                )
            try:
                width_value = float(row["width"])
                height_value = float(row["height"])
            except (KeyError, TypeError, ValueError) as error:
                raise SubmissionError(
                    f"{csv_path}:{line_number} has invalid dimensions"
                ) from error
            if (
                not width_value.is_integer()
                or not height_value.is_integer()
                or width_value <= 0
                or height_value <= 0
            ):
                raise SubmissionError(
                    f"{csv_path}:{line_number} has non-positive integer dimensions"
                )
            width, height = int(width_value), int(height_value)
            archive_name = f"{Path(image_name).stem}.png"
            folded = archive_name.casefold()
            if folded in seen:
                raise SubmissionError(
                    f"{csv_path}:{line_number} collides after PNG normalization"
                )
            seen.add(folded)
            entries.append((image_name, archive_name, width, height))
    if not entries:
        raise SubmissionError(f"{csv_path} contains no target poses")
    return entries


def candidate_prediction_paths(
    prediction_dir: Path, image_name: str, archive_name: str
) -> List[Path]:
    """Find likely prediction files without following paths outside a scene."""
    names = {
        image_name.casefold(),
        archive_name.casefold(),
        Path(image_name).stem.casefold() + ".jpg",
        Path(image_name).stem.casefold() + ".jpeg",
        Path(image_name).stem.casefold() + ".png",
    }
    matches = [
        path
        for path in prediction_dir.rglob("*")
        if path.is_file() and path.name.casefold() in names
    ]
    # Prefer the direct output produced by render_test_poses.py.
    direct = [
        path
        for path in matches
        if path.parent == prediction_dir and path.name.casefold() == archive_name.casefold()
    ]
    return direct + [path for path in matches if path not in direct]


def validate_predictions(
    scene_dirs: Sequence[Path],
    predictions_root: Path,
    filename_mode: str,
) -> List[Tuple[Path, str, str]]:
    """Validate all target files and return (source, scene, archive name)."""
    selected: List[Tuple[Path, str, str]] = []
    errors: List[str] = []
    for scene_dir in scene_dirs:
        prediction_dir = predictions_root / scene_dir.name
        try:
            audit = SceneReport(name=scene_dir.name)
            pose_csv = locate_test_pose_csv(scene_dir)
            read_and_validate_test_poses(pose_csv, audit)
            if audit.errors:
                raise SubmissionError(
                    f"{pose_csv}: "
                    + "; ".join(audit.errors)
                )
            manifest = read_pose_manifest(pose_csv)
        except (
            OSError,
            UnicodeError,
            csv.Error,
            ValueError,
        ) as error:
            errors.append(str(error))
            continue
        for image_name, png_name, width, height in manifest:
            candidates = candidate_prediction_paths(
                prediction_dir, image_name, png_name
            )
            if not candidates:
                errors.append(
                    f"[{scene_dir.name}] missing prediction for {image_name} "
                    f"(looked under {prediction_dir})"
                )
                continue
            if len(candidates) > 1:
                errors.append(
                    f"[{scene_dir.name}] ambiguous predictions for {image_name}: "
                    + ", ".join(str(path) for path in candidates)
                )
                continue
            source = candidates[0]
            try:
                actual_width, actual_height = read_image_size(source)
                if (actual_width, actual_height) != (width, height):
                    raise SubmissionError(
                        f"{source} is {actual_width}x{actual_height}; "
                        f"expected {width}x{height}"
                    )
                if filename_mode == "png":
                    with source.open("rb") as handle:
                        if handle.read(8) != PNG_SIGNATURE:
                            raise SubmissionError(
                                f"{source} is not PNG; use a PNG renderer or "
                                "--filename-mode image-name"
                            )
                archive_name = png_name if filename_mode == "png" else image_name
                selected.append((source, scene_dir.name, archive_name))
            except (OSError, SubmissionError) as error:
                errors.append(f"[{scene_dir.name}] {error}")

    if errors:
        raise SubmissionError("\n".join(errors))
    return selected


def write_submission(
    selected: Sequence[Tuple[Path, str, str]], output_path: Path
) -> None:
    """Write the archive atomically and verify its entries."""
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)

        archive_names: Set[str] = set()
        with zipfile.ZipFile(
            temporary_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for source, scene_name, archive_name in selected:
                member_name = f"{scene_name}/{archive_name}"
                if member_name in archive_names:
                    raise SubmissionError(f"duplicate archive member: {member_name}")
                archive_names.add(member_name)
                archive.write(source, arcname=member_name)

        with zipfile.ZipFile(temporary_path, "r") as archive:
            broken_member = archive.testzip()
            if broken_member is not None:
                raise SubmissionError(
                    f"archive integrity check failed at {broken_member}"
                )
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass


def build_parser() -> argparse.ArgumentParser:
    """Build the submission command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate every requested pose and create submission_round1.zip. "
            "No prediction file is resized or silently omitted."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--predictions-root",
        type=Path,
        default=Path("predictions") / "round1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission_round1.zip"),
    )
    parser.add_argument("--scenes", nargs="*", default=())
    parser.add_argument(
        "--filename-mode",
        choices=("png", "image-name"),
        default="png",
        help=(
            "png (default) follows the PDF's .png archive example and uses "
            "the CSV stem; image-name preserves the CSV extension literally."
        ),
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Build the requested submission archive."""
    args = build_parser().parse_args(argv)
    try:
        scene_dirs = discover_scenes(args.data_root.resolve(), args.scenes)
        selected = validate_predictions(
            scene_dirs,
            args.predictions_root.resolve(),
            args.filename_mode,
        )
        write_submission(selected, args.output)
    except (FileNotFoundError, OSError, SubmissionError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(
        f"Wrote {args.output.resolve()} with {len(selected)} image(s) "
        f"across {len(scene_dirs)} scene(s)."
    )
    if args.filename_mode == "image-name":
        print(
            "Warning: image-name mode preserves CSV extensions; use the "
            "default png mode for the PDF's PNG archive layout."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
