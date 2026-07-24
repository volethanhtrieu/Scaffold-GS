#!/usr/bin/env python3
"""Validate Viettel AI Race scenes without modifying the source dataset."""

from __future__ import annotations

import argparse
import csv
import math
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Iterable, List, Sequence, Set, Optional


CSV_COLUMNS = (
    "image_name",
    "qw",
    "qx",
    "qy",
    "qz",
    "tx",
    "ty",
    "tz",
    "fx",
    "fy",
    "cx",
    "cy",
    "width",
    "height",
)
FLOAT_COLUMNS = CSV_COLUMNS[1:12]
REQUIRED_COLMAP_FILES = ("cameras.bin", "images.bin", "points3D.bin")
TEST_POSE_FILENAMES = ("test_poses.csv", "test_pose.csv")
SUPPORTED_CAMERA_MODELS = {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "PINHOLE"}
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass
class SceneReport:
    """Validation result for one scene."""

    name: str
    train_images: int = 0
    test_poses: int = 0
    colmap_images: int = 0
    withheld_colmap_images: int = 0
    camera_models: Set[str] = field(default_factory=set)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Return whether this scene has no validation errors."""
        return not self.errors


def read_exact(handle: BinaryIO, size: int, path: Path) -> bytes:
    """Read exactly ``size`` bytes or raise a useful format error."""
    data = handle.read(size)
    if len(data) != size:
        raise ValueError(f"Unexpected end of COLMAP file: {path}")
    return data


def read_colmap_image_names(path: Path) -> List[str]:
    """Read registered image names from a COLMAP ``images.bin`` file."""
    names: List[str] = []
    with path.open("rb") as handle:
        (image_count,) = struct.unpack("<Q", read_exact(handle, 8, path))
        for _ in range(image_count):
            # IMAGE_ID, QVEC[4], TVEC[3], CAMERA_ID.
            read_exact(handle, 64, path)
            name_bytes = bytearray()
            while True:
                character = read_exact(handle, 1, path)
                if character == bytes([0]):
                    break
                name_bytes.extend(character)
            names.append(name_bytes.decode("utf-8"))
            (point_count,) = struct.unpack("<Q", read_exact(handle, 8, path))
            handle.seek(24 * point_count, 1)
    return names


def read_colmap_camera_models(path: Path) -> Set[str]:
    """Read camera model names from a COLMAP ``cameras.bin`` file."""
    models: Set[str] = set()
    with path.open("rb") as handle:
        (camera_count,) = struct.unpack("<Q", read_exact(handle, 8, path))
        for _ in range(camera_count):
            _, model_id, _, _ = struct.unpack(
                "<iiQQ", read_exact(handle, 24, path)
            )
            if model_id not in CAMERA_MODELS:
                raise ValueError(
                    f"Unknown COLMAP camera model id {model_id} in {path}"
                )
            model_name, parameter_count = CAMERA_MODELS[model_id]
            models.add(model_name)
            read_exact(handle, 8 * parameter_count, path)
    return models


def validate_positive_point_cloud(path: Path) -> None:
    """Check that ``points3D.bin`` contains at least one point."""
    with path.open("rb") as handle:
        (point_count,) = struct.unpack("<Q", read_exact(handle, 8, path))
    if point_count == 0:
        raise ValueError(f"COLMAP point cloud is empty: {path}")


def parse_dimension(value: str, column: str, line_number: int) -> int:
    """Parse a positive integer dimension from a CSV value."""
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or not numeric_value.is_integer():
        raise ValueError(
            f"line {line_number}: {column} must be a finite integer"
        )
    dimension = int(numeric_value)
    if dimension <= 0:
        raise ValueError(f"line {line_number}: {column} must be positive")
    return dimension


def read_and_validate_test_poses(
    csv_path: Path, report: SceneReport
) -> Set[str]:
    """Validate ``test_poses.csv`` and return its requested image names."""
    requested_names: Set[str] = set()
    png_names: Set[str] = set()

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            report.errors.append(
                "test_poses.csv header must be exactly: "
                + ",".join(CSV_COLUMNS)
            )
            return requested_names

        for line_number, row in enumerate(reader, start=2):
            try:
                if None in row or any(
                    row[column] is None or not row[column].strip()
                    for column in CSV_COLUMNS
                ):
                    raise ValueError(
                        f"line {line_number}: missing or extra CSV value"
                    )

                image_name = row["image_name"].strip()
                if (
                    Path(image_name).name != image_name
                    or "\\" in image_name
                    or image_name in {".", ".."}
                ):
                    raise ValueError(
                        f"line {line_number}: image_name must be a plain filename"
                    )
                if image_name in requested_names:
                    raise ValueError(
                        f"line {line_number}: duplicate image_name {image_name!r}"
                    )

                numeric = {
                    column: float(row[column]) for column in FLOAT_COLUMNS
                }
                if not all(math.isfinite(value) for value in numeric.values()):
                    raise ValueError(
                        f"line {line_number}: pose values must be finite"
                    )

                width = parse_dimension(row["width"], "width", line_number)
                height = parse_dimension(row["height"], "height", line_number)
                if numeric["fx"] <= 0 or numeric["fy"] <= 0:
                    raise ValueError(
                        f"line {line_number}: fx and fy must be positive"
                    )
                if not 0 <= numeric["cx"] <= width:
                    raise ValueError(
                        f"line {line_number}: cx lies outside the image"
                    )
                if not 0 <= numeric["cy"] <= height:
                    raise ValueError(
                        f"line {line_number}: cy lies outside the image"
                    )

                quaternion_norm = math.sqrt(
                    sum(numeric[key] ** 2 for key in ("qw", "qx", "qy", "qz"))
                )
                if abs(quaternion_norm - 1.0) > 1e-3:
                    raise ValueError(
                        f"line {line_number}: quaternion norm is "
                        f"{quaternion_norm:.6f}, expected 1"
                    )

                png_name = f"{Path(image_name).stem}.png"
                if png_name.casefold() in png_names:
                    raise ValueError(
                        f"line {line_number}: duplicate normalized PNG name "
                        f"{png_name!r}"
                    )

                requested_names.add(image_name)
                png_names.add(png_name.casefold())
            except (KeyError, TypeError, ValueError) as error:
                report.errors.append(str(error))

    report.test_poses = len(requested_names)
    if not requested_names:
        report.errors.append("test_poses.csv contains no valid target poses")
    return requested_names


def validate_scene(scene_dir: Path) -> SceneReport:
    """Validate one competition scene."""
    report = SceneReport(name=scene_dir.name)
    train_images_dir = scene_dir / "train" / "images"
    sparse_dir = scene_dir / "train" / "sparse" / "0"

    try:
        csv_path = locate_test_pose_csv(scene_dir)
        if not train_images_dir.is_dir():
            report.errors.append(f"Missing directory: {train_images_dir}")
            return report
        if not sparse_dir.is_dir():
            report.errors.append(f"Missing directory: {sparse_dir}")
            return report
        if not csv_path.is_file():
            report.errors.append(f"Missing file: {csv_path}")
            return report

        for filename in REQUIRED_COLMAP_FILES:
            path = sparse_dir / filename
            if not path.is_file() or path.stat().st_size == 0:
                report.errors.append(f"Missing or empty COLMAP file: {path}")
        if report.errors:
            return report

        image_files = {
            path.name
            for path in train_images_dir.iterdir()
            if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
        }
        ignored_files = [
            path.name
            for path in train_images_dir.iterdir()
            if path.is_file() and path.suffix.casefold() not in IMAGE_SUFFIXES
        ]
        report.train_images = len(image_files)
        if not image_files:
            report.errors.append("train/images contains no supported images")
        if ignored_files:
            report.warnings.append(
                f"ignored {len(ignored_files)} non-image file(s) in train/images"
            )

        requested_names = read_and_validate_test_poses(csv_path, report)
        registered_list = read_colmap_image_names(sparse_dir / "images.bin")
        registered_names = set(registered_list)
        report.colmap_images = len(registered_names)
        if not registered_names:
            report.errors.append("COLMAP images.bin contains no registered images")
        if len(registered_names) != len(registered_list):
            report.errors.append("COLMAP images.bin contains duplicate names")

        unregistered_train = sorted(image_files - registered_names)
        if unregistered_train:
            report.errors.append(
                f"{len(unregistered_train)} training image(s) are not "
                f"registered in COLMAP; first: {unregistered_train[0]}"
            )

        missing_test_poses = sorted(requested_names - registered_names)
        if missing_test_poses:
            report.errors.append(
                f"{len(missing_test_poses)} test pose name(s) are not "
                f"registered in COLMAP; first: {missing_test_poses[0]}"
            )

        overlap = sorted(image_files & requested_names)
        if overlap:
            report.warnings.append(
                f"{len(overlap)} test target(s) also exist in train/images"
            )

        report.withheld_colmap_images = len(registered_names - image_files)
        report.camera_models = read_colmap_camera_models(
            sparse_dir / "cameras.bin"
        )
        if not report.camera_models:
            report.errors.append("COLMAP cameras.bin contains no cameras")
        unsupported = report.camera_models - SUPPORTED_CAMERA_MODELS
        if unsupported:
            report.errors.append(
                "Scaffold-GS does not support camera model(s): "
                + ", ".join(sorted(unsupported))
            )
        validate_positive_point_cloud(sparse_dir / "points3D.bin")
    except (OSError, UnicodeError, csv.Error, struct.error, ValueError) as error:
        report.errors.append(str(error))

    return report


def locate_test_pose_csv(scene_dir: Path) -> Path:
    """Return the scene's pose CSV, accepting the PDF's singular typo too."""
    candidates = [
        scene_dir / "test" / filename
        for filename in TEST_POSE_FILENAMES
        if (scene_dir / "test" / filename).is_file()
    ]
    if len(candidates) > 1:
        raise ValueError(
            f"Both test pose CSV spellings exist in {scene_dir / 'test'}; "
            "keep only test_poses.csv"
        )
    if candidates:
        return candidates[0]
    return scene_dir / "test" / TEST_POSE_FILENAMES[0]


def discover_scenes(data_root: Path, requested: Sequence[str]) -> List[Path]:
    """Resolve requested scene directories and reject unknown scene names."""
    if requested:
        scene_dirs = [data_root / name for name in requested]
        missing = [path.name for path in scene_dirs if not path.is_dir()]
        if missing:
            raise FileNotFoundError(
                "Unknown scene(s): " + ", ".join(sorted(missing))
            )
        return scene_dirs
    return sorted(path for path in data_root.iterdir() if path.is_dir())


def print_contract_summary() -> None:
    """Print the PDF-derived input and output contract."""
    print("PDF input contract:")
    print("  data/<scene>/train/images/*")
    print(
        "  data/<scene>/train/sparse/0/"
        "{cameras.bin,images.bin,points3D.bin}"
    )
    print("  data/<scene>/test/test_poses.csv")
    print("PDF output contract:")
    print(
        "  submission_round1.zip -> <scene>/<image_name stem>.png "
        "at each CSV width x height"
    )
    print(
        "  Note: the round PDF also says literal image_name; "
        "generate_submission.py supports --filename-mode image-name."
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate Viettel AI Race data in place. This command never moves, "
            "copies, renames, or deletes dataset files."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root containing one directory per scene (default: data).",
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=(),
        help="Optional scene names; validates every scene when omitted.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Validate all selected scenes and print a concise audit."""
    args = build_parser().parse_args(argv)
    data_root = args.data_root.resolve()
    print_contract_summary()

    try:
        if not data_root.is_dir():
            raise FileNotFoundError(f"Data root does not exist: {data_root}")
        scene_dirs = discover_scenes(data_root, args.scenes)
        if not scene_dirs:
            raise FileNotFoundError(f"No scenes found under: {data_root}")
    except OSError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print(f"\nValidating {len(scene_dirs)} scene(s) under {data_root}:")
    reports = [validate_scene(scene_dir) for scene_dir in scene_dirs]
    for report in reports:
        status = "OK" if report.valid else "ERROR"
        models = ",".join(sorted(report.camera_models)) or "unknown"
        print(
            f"  [{status}] {report.name}: train={report.train_images}, "
            f"test={report.test_poses}, COLMAP={report.colmap_images}, "
            f"withheld={report.withheld_colmap_images}, models={models}"
        )
        for warning in report.warnings:
            print(f"    warning: {warning}")
        for error in report.errors:
            print(f"    error: {error}")

    valid_count = sum(report.valid for report in reports)
    print(f"\nResult: {valid_count}/{len(reports)} scene(s) valid.")
    if valid_count == len(reports):
        print(
            "No data reorganization is needed. Train with "
            "--source_path data/<scene>/train."
        )
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
