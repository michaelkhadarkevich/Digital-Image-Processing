import argparse
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


DEFAULT_INPUT_DIR = Path("data/Data 1/test")
OUTPUT_DIR = Path("data/Data 1 with distortion/distortions")
IMAGE_SIZE = (224, 224)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply common image distortion methods.")
    parser.add_argument(
        "--image",
        help="Path to one input image. If omitted, a test image is chosen automatically.",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR),
        help="Directory where distorted images are saved.",
    )
    return parser.parse_args()


def find_default_image() -> Path:
    valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    candidates = [
        path
        for path in DEFAULT_INPUT_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in valid_suffixes
    ]
    if not candidates:
        raise FileNotFoundError(
            "No default image found. Run preprocess.py first or pass --image."
        )
    return sorted(candidates)[0]


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").resize(IMAGE_SIZE)


def gaussian_noise(image: Image.Image, sigma: float = 24.0) -> Image.Image:
    array = np.asarray(image).astype(np.float32)
    noise = np.random.default_rng(42).normal(0, sigma, array.shape)
    noisy = np.clip(array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def gaussian_noise_snr(image: Image.Image, snr_db: float) -> Image.Image:
    array = np.asarray(image).astype(np.float32)
    signal_power = np.mean(array * array)
    if signal_power <= 0:
        sigma = 0.0
    else:
        noise_power = signal_power / (10 ** (snr_db / 10))
        sigma = float(np.sqrt(noise_power))
    noise = np.random.default_rng(42).normal(0, sigma, array.shape)
    noisy = np.clip(array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def salt_and_pepper_noise(image: Image.Image, amount: float = 0.04) -> Image.Image:
    array = np.asarray(image).copy()
    rng = np.random.default_rng(42)
    mask = rng.random(array.shape[:2])
    array[mask < amount / 2] = 0
    array[mask > 1 - amount / 2] = 255
    return Image.fromarray(array)


def blur(image: Image.Image) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=3))


def gaussian_blur_sigma(image: Image.Image, sigma: float) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def brightness_contrast(image: Image.Image) -> Image.Image:
    image = ImageEnhance.Brightness(image).enhance(1.35)
    return ImageEnhance.Contrast(image).enhance(1.55)


def rotate(image: Image.Image) -> Image.Image:
    return image.rotate(18, resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0))


def perspective_warp(image: Image.Image) -> Image.Image:
    width, height = image.size
    source = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )
    target = np.array(
        [
            [20, 8],
            [width - 24, 24],
            [width - 6, height - 18],
            [10, height - 8],
        ],
        dtype=np.float32,
    )
    coefficients = find_perspective_coefficients(source, target)
    return image.transform(
        image.size,
        Image.Transform.PERSPECTIVE,
        coefficients,
        Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0),
    )


def find_perspective_coefficients(source: np.ndarray, target: np.ndarray) -> list[float]:
    matrix = []
    for (x, y), (u, v) in zip(target, source):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    values = source.flatten()
    return np.linalg.solve(np.asarray(matrix), values).tolist()


def barrel_distortion(image: Image.Image, strength: float = 0.000015) -> Image.Image:
    array = np.asarray(image)
    height, width = array.shape[:2]
    center_x = width / 2
    center_y = height / 2

    y_indices, x_indices = np.indices((height, width))
    x = x_indices - center_x
    y = y_indices - center_y
    radius_squared = x * x + y * y

    factor = 1 + strength * radius_squared
    source_x = np.clip(center_x + x * factor, 0, width - 1).astype(np.int32)
    source_y = np.clip(center_y + y * factor, 0, height - 1).astype(np.int32)

    distorted = array[source_y, source_x]
    return Image.fromarray(distorted)


def pixelate(image: Image.Image, block_size: int = 12) -> Image.Image:
    small_size = (IMAGE_SIZE[0] // block_size, IMAGE_SIZE[1] // block_size)
    small = image.resize(small_size, Image.Resampling.BILINEAR)
    return small.resize(IMAGE_SIZE, Image.Resampling.NEAREST)


def cnn_level_distortions(original: Image.Image) -> dict[str, Image.Image]:
    return {
        "original": original,
        "gaussian_noise": gaussian_noise(original),
        "salt_and_pepper": salt_and_pepper_noise(original),
        "gaussian_blur": blur(original),
        "brightness_contrast": brightness_contrast(original),
        "rotation": rotate(original),
        "perspective_warp": perspective_warp(original),
        "barrel_distortion": barrel_distortion(original),
        "pixelation": pixelate(original),
        "gaussian_noise_snr_5db": gaussian_noise_snr(original, 5),
        "gaussian_noise_snr_10db": gaussian_noise_snr(original, 10),
        "gaussian_noise_snr_15db": gaussian_noise_snr(original, 15),
        "gaussian_noise_snr_20db": gaussian_noise_snr(original, 20),
        "gaussian_noise_snr_30db": gaussian_noise_snr(original, 30),
        "salt_and_pepper_density_0_01": salt_and_pepper_noise(original, 0.01),
        "salt_and_pepper_density_0_03": salt_and_pepper_noise(original, 0.03),
        "salt_and_pepper_density_0_05": salt_and_pepper_noise(original, 0.05),
        "salt_and_pepper_density_0_10": salt_and_pepper_noise(original, 0.10),
        "gaussian_blur_sigma_0_5": gaussian_blur_sigma(original, 0.5),
        "gaussian_blur_sigma_1_0": gaussian_blur_sigma(original, 1.0),
        "gaussian_blur_sigma_1_5": gaussian_blur_sigma(original, 1.5),
        "gaussian_blur_sigma_2_0": gaussian_blur_sigma(original, 2.0),
        "gaussian_blur_sigma_3_0": gaussian_blur_sigma(original, 3.0),
    }


def save_comparison_grid(images: dict[str, Image.Image], output_path: Path) -> None:
    names = list(images.keys())
    columns = 3
    rows = int(np.ceil(len(names) / columns))

    plt.figure(figsize=(10, 3.4 * rows))
    for index, name in enumerate(names, start=1):
        plt.subplot(rows, columns, index)
        plt.imshow(images[name])
        plt.title(name)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image) if args.image else find_default_image()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    original = load_image(image_path)
    distortions = {
        "clean": original,
        "gaussian_noise": gaussian_noise(original),
        "salt_and_pepper": salt_and_pepper_noise(original),
        "gaussian_blur": blur(original),
        "brightness_contrast": brightness_contrast(original),
        "rotation": rotate(original),
        "perspective_warp": perspective_warp(original),
        "barrel_distortion": barrel_distortion(original),
        "pixelation": pixelate(original),
    }

    for name, image in distortions.items():
        image.save(output_dir / f"{name}.png")

    save_comparison_grid(distortions, output_dir / "distortion_grid.png")
    (output_dir / "methods.txt").write_text(
        "\n".join(
            [
                "Distortion Methods",
                "",
                f"Input image: {image_path.resolve()}",
                "",
                "1. Gaussian noise: adds normally distributed random pixel noise.",
                "2. Salt and pepper noise: randomly turns pixels black or white.",
                "3. Gaussian blur: smooths local detail and edges.",
                "4. Brightness/contrast: changes image intensity and contrast.",
                "5. Rotation: applies geometric rotation.",
                "6. Perspective warp: simulates viewpoint distortion.",
                "7. Barrel distortion: bends the image outward from the center.",
                "8. Pixelation: reduces spatial detail by enlarging blocks.",
            ]
        ),
        encoding="utf-8",
    )

    print("Input image:", image_path.resolve())
    print("Saved distortion results to:", output_dir.resolve())


if __name__ == "__main__":
    main()
