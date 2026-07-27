import argparse
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


DEFAULT_SOURCE = Path("data/Data 2/yolo_segmentation")
DEFAULT_OUTPUT_ROOT = Path("data/Data 2 with distortion")
DEFAULT_METHODS = [
    "gaussian_noise",
    "gaussian_blur",
    "brightness_contrast",
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
        description="Create YOLO segmentation datasets with distorted MRI images."
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE),
        help="Prepared clean YOLO segmentation dataset folder.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Folder where distorted YOLO datasets are written.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        choices=sorted(DISTORTION_METHODS.keys()) if "DISTORTION_METHODS" in globals() else None,
        help="Distortion methods to generate.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def gaussian_noise(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    array = np.asarray(image).astype(np.float32)
    noise = rng.normal(0, 24.0, array.shape)
    noisy = np.clip(array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def gaussian_noise_snr(image: Image.Image, rng: np.random.Generator, snr_db: float) -> Image.Image:
    array = np.asarray(image).astype(np.float32)
    signal_power = np.mean(array * array)
    if signal_power <= 0:
        sigma = 0.0
    else:
        noise_power = signal_power / (10 ** (snr_db / 10))
        sigma = float(np.sqrt(noise_power))
    noise = rng.normal(0, sigma, array.shape)
    noisy = np.clip(array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def gaussian_blur(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    del rng
    return image.filter(ImageFilter.GaussianBlur(radius=3))


def gaussian_blur_sigma(image: Image.Image, rng: np.random.Generator, sigma: float) -> Image.Image:
    del rng
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def brightness_contrast(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    del rng
    image = ImageEnhance.Brightness(image).enhance(1.35)
    return ImageEnhance.Contrast(image).enhance(1.55)


def brightness_contrast_values(
    image: Image.Image,
    rng: np.random.Generator,
    brightness: float,
    contrast: float,
) -> Image.Image:
    del rng
    image = ImageEnhance.Brightness(image).enhance(brightness)
    return ImageEnhance.Contrast(image).enhance(contrast)


DISTORTION_METHODS = {
    "gaussian_noise": gaussian_noise,
    "gaussian_blur": gaussian_blur,
    "brightness_contrast": brightness_contrast,
    "gaussian_noise_snr_5db": lambda image, rng: gaussian_noise_snr(image, rng, 5),
    "gaussian_noise_snr_15db": lambda image, rng: gaussian_noise_snr(image, rng, 15),
    "gaussian_noise_snr_30db": lambda image, rng: gaussian_noise_snr(image, rng, 30),
    "gaussian_blur_sigma_0_5": lambda image, rng: gaussian_blur_sigma(image, rng, 0.5),
    "gaussian_blur_sigma_1_5": lambda image, rng: gaussian_blur_sigma(image, rng, 1.5),
    "gaussian_blur_sigma_3_0": lambda image, rng: gaussian_blur_sigma(image, rng, 3.0),
    "brightness_contrast_b1_15_c1_25": lambda image, rng: brightness_contrast_values(image, rng, 1.15, 1.25),
    "brightness_contrast_b1_35_c1_55": lambda image, rng: brightness_contrast_values(image, rng, 1.35, 1.55),
    "brightness_contrast_b1_55_c1_80": lambda image, rng: brightness_contrast_values(image, rng, 1.55, 1.80),
}


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


def distort_image(source_image: Path, target_image: Path, method: str, rng: np.random.Generator) -> None:
    with Image.open(source_image) as image:
        distorted = DISTORTION_METHODS[method](image.convert("RGB"), rng)
        distorted.save(target_image, format="JPEG", quality=95)


def prepare_method(source: Path, output_root: Path, method: str, seed: int) -> None:
    output = output_root / f"yolo_segmentation_{method}"
    if output.exists():
        shutil.rmtree(output)

    total_images = 0
    total_labels = 0
    rng = np.random.default_rng(seed)

    for split in ["train", "val", "test"]:
        source_images = source / "images" / split
        source_labels = source / "labels" / split
        target_images = output / "images" / split
        target_labels = output / "labels" / split
        target_images.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(path for path in source_images.iterdir() if path.is_file())
        random.Random(seed).shuffle(image_paths)

        for image_path in image_paths:
            target_image = target_images / f"{image_path.stem}.jpg"
            distort_image(image_path, target_image, method, rng)
            total_images += 1

            label_path = source_labels / f"{image_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, target_labels / label_path.name)
                total_labels += 1

    write_data_yaml(output)
    summary = [
        "Distorted YOLO Segmentation Dataset Summary",
        "",
        f"method: {method}",
        f"source: {source.resolve()}",
        f"output: {output.resolve()}",
        f"images: {total_images}",
        f"labels: {total_labels}",
        "",
        "The labels are reused from the clean segmentation dataset.",
        "Only image-intensity distortions are used here, so tumor masks remain aligned.",
        "",
    ]
    (output / "summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print(f"Saved {method} YOLO segmentation dataset to: {output}")


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    output_root = Path(args.output_root)

    if not source.exists():
        raise FileNotFoundError(
            f"Clean YOLO segmentation dataset not found: {source}. "
            'Run: python "task/task 3 yolo/prepare_yolo_segmentation.py"'
        )

    output_root.mkdir(parents=True, exist_ok=True)
    for method in args.methods:
        prepare_method(source, output_root, method, args.seed)


if __name__ == "__main__":
    main()
