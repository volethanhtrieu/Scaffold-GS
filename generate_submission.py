#!/usr/bin/env python3
"""Validate rendered predictions and build the competition ZIP archive."""

from __future__ import annotations

import argparse
import csv
import io
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
    selected: Sequence[Tuple[Path, str, str]],
    output_path: Path,
    jpeg_quality: Optional[int] = None,
    jpeg_subsampling: int = 2,
    max_size_bytes: Optional[int] = None,
) -> None:
    """Write the archive atomically, optionally transcoding images to JPEG."""
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    image_module = None
    unidentified_image_error = OSError
    if jpeg_quality is not None:
        try:
            from PIL import Image, UnidentifiedImageError
        except ImportError as error:
            raise SubmissionError(
                "JPEG transcoding requires Pillow in the active environment"
            ) from error
        image_module = Image
        unidentified_image_error = UnidentifiedImageError

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
        compression = (
            zipfile.ZIP_STORED
            if jpeg_quality is not None
            else zipfile.ZIP_DEFLATED
        )
        with zipfile.ZipFile(temporary_path, "w", compression=compression) as archive:
            for source, scene_name, archive_name in selected:
                member_name = f"{scene_name}/{archive_name}"
                if member_name in archive_names:
                    raise SubmissionError(f"duplicate archive member: {member_name}")
                archive_names.add(member_name)
                if jpeg_quality is None:
                    archive.write(source, arcname=member_name)
                    continue

                if Path(archive_name).suffix.casefold() not in (".jpg", ".jpeg"):
                    raise SubmissionError(
                        "JPEG transcoding requires every CSV image_name to end "
                        f"in .jpg or .jpeg; got {archive_name}"
                    )
                try:
                    assert image_module is not None
                    with image_module.open(source) as image:
                        rgb_image = image.convert("RGB")
                        encoded = io.BytesIO()
                        rgb_image.save(
                            encoded,
                            format="JPEG",
                            quality=jpeg_quality,
                            subsampling=jpeg_subsampling,
                            optimize=True,
                        )
                except (OSError, unidentified_image_error) as error:
                    raise SubmissionError(
                        f"could not transcode {source} to JPEG: {error}"
                    ) from error
                archive.writestr(
                    member_name,
                    encoded.getvalue(),
                    compress_type=zipfile.ZIP_STORED,
                )

        with zipfile.ZipFile(temporary_path, "r") as archive:
            broken_member = archive.testzip()
            if broken_member is not None:
                raise SubmissionError(
                    f"archive integrity check failed at {broken_member}"
                )
        archive_size = temporary_path.stat().st_size
        if max_size_bytes is not None and archive_size > max_size_bytes:
            raise SubmissionError(
                f"archive is {archive_size / 1_000_000:.2f} MB, exceeding "
                f"the configured {max_size_bytes / 1_000_000:.2f} MB limit; "
                "lower --jpeg-quality and try again"
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
    parser.add_argument(
        "--transcode-jpeg",
        action="store_true",
        help=(
            "Transcode predictions to RGB JPEG while preserving literal CSV "
            "image_name values. Requires --filename-mode image-name."
        ),
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=92,
        help="JPEG quality from 1 to 100 (default: 92).",
    )
    parser.add_argument(
        "--jpeg-subsampling",
        choices=("444", "422", "420"),
        default="420",
        help="JPEG chroma subsampling (default: 420 for smaller files).",
    )
    parser.add_argument(
        "--max-size-mb",
        type=float,
        default=None,
        help=(
            "Reject the archive instead of replacing the output when it "
            "exceeds this decimal-megabyte limit."
        ),
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Build the requested submission archive."""
    args = build_parser().parse_args(argv)
    if args.transcode_jpeg and args.filename_mode != "image-name":
        print(
            "ERROR: --transcode-jpeg requires --filename-mode image-name",
            file=sys.stderr,
        )
        return 2
    if not 1 <= args.jpeg_quality <= 100:
        print("ERROR: --jpeg-quality must be between 1 and 100", file=sys.stderr)
        return 2
    if args.max_size_mb is not None and args.max_size_mb <= 0:
        print("ERROR: --max-size-mb must be positive", file=sys.stderr)
        return 2

    subsampling_values = {"444": 0, "422": 1, "420": 2}
    max_size_bytes = (
        int(args.max_size_mb * 1_000_000)
        if args.max_size_mb is not None
        else None
    )
    try:
        scene_dirs = discover_scenes(args.data_root.resolve(), args.scenes)
        selected = validate_predictions(
            scene_dirs,
            args.predictions_root.resolve(),
            args.filename_mode,
        )
        write_submission(
            selected,
            args.output,
            jpeg_quality=args.jpeg_quality if args.transcode_jpeg else None,
            jpeg_subsampling=subsampling_values[args.jpeg_subsampling],
            max_size_bytes=max_size_bytes,
        )
    except (FileNotFoundError, OSError, SubmissionError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    output_size_mb = args.output.resolve().stat().st_size / 1_000_000
    print(
        f"Wrote {args.output.resolve()} with {len(selected)} image(s) "
        f"across {len(scene_dirs)} scene(s), size {output_size_mb:.2f} MB."
    )
    if args.transcode_jpeg:
        print(
            f"Images were encoded as RGB JPEG at quality {args.jpeg_quality}, "
            f"subsampling {args.jpeg_subsampling}, using literal CSV names."
        )
    elif args.filename_mode == "image-name":
        print(
            "Warning: image-name mode changes archive names only; it does not "
            "transcode PNG bytes. Add --transcode-jpeg for a real JPEG archive."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
