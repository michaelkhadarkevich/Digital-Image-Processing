import argparse
import importlib.util
import shutil
from pathlib import Path

from PIL import Image


DEFAULT_DISTORTED_ROOT = Path("data/Data 2 with distortion")
DEFAULT_OUTPUT_ROOT = Path("data/Data 2 with distortion/restoration")
RESTORATION_MODULE_PATH = Path("task/restoration/image_restoration.py")
METHODS = [
    "gaussian_noise_snr_5db",
    "gaussian_noise_snr_15db",
    "gaussian_noise_snr_30db",
    "gaussian_blur_sigma_0_5",
    "gaussian_blur_sigma_1_5",
    "gaussian_blur_sigma_3_0",
    "brightness_contrast_b1_15_c1_25",
    "brightness_contrast_b1_35_c1_55",
    "brightness_contrast_b1_55_c1_80",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create YOLO segmentation datasets from distorted images after restoration."
    )
    parser.add_argument(
        "--distorted-root",
        default=str(DEFAULT_DISTORTED_ROOT),
        help="Folder containing yolo_segmentation_<method> distorted datasets.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Folder where restored YOLO datasets are written.",
    )
    parser.add_argument("--methods", nargs="+", default=METHODS)
    return parser.parse_args()


def load_restoration_module():
    spec = importlib.util.spec_from_file_location(
        "image_restoration_runtime",
        RESTORATION_MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {RESTORATION_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def restoration_for_method(method: str, restoration_module):
    if method.startswith("gaussian_noise_snr_"):
        return restoration_module.denoise_gaussian
    if method.startswith("gaussian_blur_sigma_"):
        return restoration_module.deblur_sharpen
    if method.startswith("brightness_contrast_"):
        return restoration_module.normalize_intensity
    raise KeyError(f"No restoration function for method: {method}")


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


def restore_image(source_image: Path, target_image: Path, restore_function) -> None:
    with Image.open(source_image) as image:
        restored = restore_function(image.convert("RGB"))
        restored.save(target_image, format="JPEG", quality=95)


def prepare_method(
    distorted_root: Path,
    output_root: Path,
    method: str,
    restoration_module,
) -> None:
    source = distorted_root / f"yolo_segmentation_{method}"
    output = output_root / f"yolo_segmentation_{method}_restored"
    if not source.exists():
        raise FileNotFoundError(source)
    if output.exists():
        shutil.rmtree(output)

    restore_function = restoration_for_method(method, restoration_module)
    total_images = 0
    total_labels = 0

    for split in ["train", "val", "test"]:
        source_images = source / "images" / split
        source_labels = source / "labels" / split
        target_images = output / "images" / split
        target_labels = output / "labels" / split
        target_images.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)

        for image_path in sorted(path for path in source_images.iterdir() if path.is_file()):
            target_image = target_images / f"{image_path.stem}.jpg"
            restore_image(image_path, target_image, restore_function)
            total_images += 1

            label_path = source_labels / f"{image_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, target_labels / label_path.name)
                total_labels += 1

    write_data_yaml(output)
    summary = [
        "Restored YOLO Segmentation Dataset Summary",
        "",
        f"method: {method}",
        f"source: {source.resolve()}",
        f"output: {output.resolve()}",
        f"images: {total_images}",
        f"labels: {total_labels}",
        "",
        "Images are restored after distortion. Labels are reused because these",
        "restoration methods do not change the image geometry.",
        "",
    ]
    (output / "summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print(f"Saved restored {method} YOLO segmentation dataset to: {output}")


def main() -> None:
    args = parse_args()
    distorted_root = Path(args.distorted_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    restoration_module = load_restoration_module()
    for method in args.methods:
        prepare_method(distorted_root, output_root, method, restoration_module)


if __name__ == "__main__":
    main()
