import argparse
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


DEFAULT_SOURCE = Path("data/Data 2/mri_segmentation")
DEFAULT_OUTPUT = Path("data/Data 2/yolo_segmentation")
VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MRI tumor segmentation masks to YOLO segmentation labels."
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE),
        help="Folder that contains the LGG MRI segmentation dataset.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output folder for YOLO segmentation data.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Include images with empty masks as negative examples.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy image files instead of hard-linking them. Uses much more disk space.",
    )
    return parser.parse_args()


def find_mask_files(source: Path) -> list[Path]:
    return sorted(
        path
        for path in source.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VALID_IMAGE_SUFFIXES
        and path.stem.endswith("_mask")
    )


def paired_image_path(mask_path: Path) -> Path | None:
    image_stem = mask_path.stem[: -len("_mask")]
    for suffix in VALID_IMAGE_SUFFIXES:
        candidate = mask_path.with_name(image_stem + suffix)
        if candidate.exists():
            return candidate
    return None


def read_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")
    _, binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    return binary


def mask_to_yolo_segments(mask: np.ndarray) -> list[list[float]]:
    height, width = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments: list[list[float]] = []

    for contour in contours:
        if cv2.contourArea(contour) < 4:
            continue

        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(approx) < 3:
            continue

        segment: list[float] = []
        for x, y in approx:
            segment.extend(
                [
                    min(max(float(x) / width, 0.0), 1.0),
                    min(max(float(y) / height, 0.0), 1.0),
                ]
            )
        segments.append(segment)

    return segments


def image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def split_name(index: int, total: int, train_ratio: float, val_ratio: float) -> str:
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    if index < train_end:
        return "train"
    if index < val_end:
        return "val"
    return "test"


def safe_sample_stem(image_path: Path, source: Path) -> str:
    relative = image_path.relative_to(source).with_suffix("")
    return "__".join(relative.parts)


def write_data_yaml(output_dir: Path) -> None:
    yaml_text = "\n".join(
        [
            f"path: {output_dir.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "",
            "names:",
            "  0: tumor",
            "",
        ]
    )
    (output_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")


def place_image(source: Path, target: Path, copy_images: bool) -> str:
    if copy_images:
        shutil.copy2(source, target)
        return "copy"

    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def prepare_dataset(args: argparse.Namespace) -> None:
    source = Path(args.source)
    output = Path(args.output)

    if not source.exists():
        raise FileNotFoundError(
            f"Source folder not found: {source}. Run data/Data 2/download_mri_data.py first."
        )

    samples = []
    missing_images = 0
    empty_masks = 0

    for mask_path in find_mask_files(source):
        image_path = paired_image_path(mask_path)
        if image_path is None:
            missing_images += 1
            continue

        mask = read_mask(mask_path)
        segments = mask_to_yolo_segments(mask)
        if not segments:
            empty_masks += 1
            if not args.include_empty:
                continue

        samples.append((image_path, segments))

    if not samples:
        raise ValueError("No image/mask pairs were found for YOLO segmentation.")

    random.Random(args.seed).shuffle(samples)

    if output.exists():
        shutil.rmtree(output)

    for split in ["train", "val", "test"]:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    split_counts = {"train": 0, "val": 0, "test": 0}
    image_place_methods = {"hardlink": 0, "copy": 0}
    for index, (image_path, segments) in enumerate(samples):
        split = split_name(index, len(samples), args.train_ratio, args.val_ratio)
        split_counts[split] += 1

        suffix = image_path.suffix.lower()
        sample_stem = safe_sample_stem(image_path, source)
        target_image = output / "images" / split / f"{sample_stem}{suffix}"
        target_label = output / "labels" / split / f"{sample_stem}.txt"

        method = place_image(image_path, target_image, args.copy_images)
        image_place_methods[method] += 1
        with target_label.open("w", encoding="utf-8") as label_file:
            for segment in segments:
                points = " ".join(f"{value:.6f}" for value in segment)
                label_file.write(f"0 {points}\n")

    write_data_yaml(output)

    summary = [
        "YOLO Segmentation Dataset Summary",
        "",
        f"source: {source.resolve()}",
        f"output: {output.resolve()}",
        f"total used images: {len(samples)}",
        f"missing image pairs skipped: {missing_images}",
        f"empty masks skipped: {0 if args.include_empty else empty_masks}",
        f"empty masks included: {empty_masks if args.include_empty else 0}",
        "",
        f"train: {split_counts['train']}",
        f"val: {split_counts['val']}",
        f"test: {split_counts['test']}",
        f"image hardlinks: {image_place_methods['hardlink']}",
        f"image copies: {image_place_methods['copy']}",
        "",
        "class 0: tumor",
        "",
    ]
    (output / "summary.txt").write_text("\n".join(summary), encoding="utf-8")

    print("Saved YOLO segmentation dataset to:", output.resolve())
    print("Saved config to:", (output / "data.yaml").resolve())


def main() -> None:
    prepare_dataset(parse_args())


if __name__ == "__main__":
    main()
