import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_angles(value: str) -> list[float]:
    angles: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if item:
            angles.append(float(item))
    if not angles:
        raise argparse.ArgumentTypeError("At least one rotation angle is required.")
    return angles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a bigger brain MRI dataset by rotating training images."
    )
    parser.add_argument(
        "--input",
        default="data/Data 1",
        help="Processed dataset folder that contains train, val, and test.",
    )
    parser.add_argument(
        "--output",
        default="data/Data 1 augmented",
        help="Output folder for the augmented dataset.",
    )
    parser.add_argument(
        "--angles",
        type=parse_angles,
        default=parse_angles("-20,-10,10,20"),
        help="Comma-separated rotation angles, for example: -20,-10,10,20.",
    )
    parser.add_argument(
        "--keep-original",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy the original train images in addition to rotated images.",
    )
    return parser.parse_args()


def image_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
    )


def copy_tree(source: Path, target: Path) -> None:
    if source.exists():
        shutil.copytree(source, target)


def save_rotated_image(source: Path, target: Path, angle: float) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = image.convert("RGB")
        rotated = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(0, 0, 0))
        rotated.save(target, quality=95)


def augment_train_split(input_train: Path, output_train: Path, angles: list[float], keep_original: bool) -> tuple[int, int]:
    original_count = 0
    augmented_count = 0

    for class_dir in sorted(path for path in input_train.iterdir() if path.is_dir()):
        output_class_dir = output_train / class_dir.name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        for image_path in image_files(class_dir):
            original_count += 1
            if keep_original:
                shutil.copy2(image_path, output_class_dir / image_path.name)

            for angle in angles:
                angle_name = str(angle).replace("-", "neg").replace(".", "p")
                target_name = f"{image_path.stem}_rot_{angle_name}{image_path.suffix.lower()}"
                save_rotated_image(image_path, output_class_dir / target_name, angle)
                augmented_count += 1

    return original_count, augmented_count


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    input_train = input_dir / "train"

    if not input_train.exists():
        raise FileNotFoundError(f"Train folder not found: {input_train}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    copy_tree(input_dir / "val", output_dir / "val")
    copy_tree(input_dir / "test", output_dir / "test")

    original_count, augmented_count = augment_train_split(
        input_train,
        output_dir / "train",
        args.angles,
        args.keep_original,
    )

    total_train = augmented_count + (original_count if args.keep_original else 0)
    print(f"Original train images: {original_count}")
    print(f"Rotated train images:  {augmented_count}")
    print(f"New train total:       {total_train}")
    print("Output:", output_dir.resolve())


if __name__ == "__main__":
    main()
