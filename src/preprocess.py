import argparse
import random
import shutil
import sys
from pathlib import Path

from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


CLASS_NAMES = {"yes": "tumor", "no": "no_tumor"}
IMAGE_SIZE = (224, 224)
VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess brain MRI images.")
    parser.add_argument("--input", required=True, help="Path to downloaded Kaggle dataset.")
    parser.add_argument("--output", default="data/processed", help="Output directory.")
    parser.add_argument("--test-size", type=float, default=0.15, help="Test split size.")
    parser.add_argument("--val-size", type=float, default=0.15, help="Validation split size.")
    return parser.parse_args()


def find_class_dir(dataset_dir: Path, class_name: str) -> Path:
    matches = [path for path in dataset_dir.rglob("*") if path.is_dir() and path.name.lower() == class_name]
    if not matches:
        raise FileNotFoundError(f"Could not find class folder named '{class_name}' in {dataset_dir}")
    return matches[0]


def collect_images(dataset_dir: Path) -> list[tuple[Path, str]]:
    images: list[tuple[Path, str]] = []
    for source_name, target_name in CLASS_NAMES.items():
        class_dir = find_class_dir(dataset_dir, source_name)
        for image_path in class_dir.iterdir():
            if image_path.is_file() and image_path.suffix.lower() in VALID_SUFFIXES:
                images.append((image_path, target_name))
    return images


def save_resized_image(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE)
        image.save(target, quality=95)


def copy_split(items: list[tuple[Path, str]], split_name: str, output_dir: Path) -> None:
    for index, (source, label) in enumerate(items):
        target = output_dir / split_name / label / f"{source.stem}_{index}.jpg"
        save_resized_image(source, target)


def stratified_split(
    images: list[tuple[Path, str]],
    test_size: float,
    val_size: float,
    seed: int = 42,
) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]], list[tuple[Path, str]]]:
    rng = random.Random(seed)
    by_label: dict[str, list[tuple[Path, str]]] = {}
    for item in images:
        by_label.setdefault(item[1], []).append(item)

    train: list[tuple[Path, str]] = []
    val: list[tuple[Path, str]] = []
    test: list[tuple[Path, str]] = []

    for label_items in by_label.values():
        rng.shuffle(label_items)
        test_count = max(1, round(len(label_items) * test_size))
        val_count = max(1, round(len(label_items) * val_size))

        test.extend(label_items[:test_count])
        val.extend(label_items[test_count : test_count + val_count])
        train.extend(label_items[test_count + val_count :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).resolve()

    if output_dir.exists():
        shutil.rmtree(output_dir)

    images = collect_images(dataset_dir)
    train, val, test = stratified_split(images, args.test_size, args.val_size)

    copy_split(train, "train", output_dir)
    copy_split(val, "val", output_dir)
    copy_split(test, "test", output_dir)

    print(f"Processed {len(train)} train, {len(val)} val, {len(test)} test images.")
    print("Output:", output_dir)


if __name__ == "__main__":
    main()
