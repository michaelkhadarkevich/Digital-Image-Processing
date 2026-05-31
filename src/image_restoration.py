import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


DISTORTION_DIR = Path("results/distortions")
OUTPUT_DIR = Path("results/restoration")
IMAGE_SIZE = (224, 224)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply image restoration techniques.")
    parser.add_argument(
        "--distortion-dir",
        default=str(DISTORTION_DIR),
        help="Directory containing distortion method outputs.",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR),
        help="Directory where restoration results are saved.",
    )
    return parser.parse_args()


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").resize(IMAGE_SIZE)


def denoise_gaussian(image: Image.Image) -> Image.Image:
    smoothed = image.filter(ImageFilter.GaussianBlur(radius=1.2))
    return smoothed.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=4))


def remove_salt_and_pepper(image: Image.Image) -> Image.Image:
    return image.filter(ImageFilter.MedianFilter(size=3))


def deblur_sharpen(image: Image.Image) -> Image.Image:
    return image.filter(ImageFilter.UnsharpMask(radius=2.5, percent=220, threshold=3))


def normalize_intensity(image: Image.Image) -> Image.Image:
    autocontrasted = ImageOps.autocontrast(image, cutoff=1)
    return ImageEnhance.Contrast(autocontrasted).enhance(0.85)


def restore_rotation(image: Image.Image) -> Image.Image:
    return image.rotate(-18, resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0))


def perspective_restore(image: Image.Image) -> Image.Image:
    width, height = image.size
    original = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )
    warped = np.array(
        [
            [20, 8],
            [width - 24, 24],
            [width - 6, height - 18],
            [10, height - 8],
        ],
        dtype=np.float32,
    )
    coefficients = find_perspective_coefficients(warped, original)
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


def pincushion_correction(image: Image.Image, strength: float = -0.000011) -> Image.Image:
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

    corrected = array[source_y, source_x]
    return Image.fromarray(corrected)


def smooth_pixelation(image: Image.Image) -> Image.Image:
    smoothed = image.resize(IMAGE_SIZE, Image.Resampling.BICUBIC)
    return smoothed.filter(ImageFilter.UnsharpMask(radius=1.4, percent=120, threshold=2))


def save_restoration_grid(
    original: Image.Image,
    restored: dict[str, tuple[Image.Image, Image.Image]],
    output_path: Path,
) -> None:
    rows = len(restored)
    plt.figure(figsize=(9, 2.7 * rows))

    for row_index, (name, (distorted_image, restored_image)) in enumerate(
        restored.items(),
        start=1,
    ):
        base_index = (row_index - 1) * 3

        plt.subplot(rows, 3, base_index + 1)
        plt.imshow(original)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(rows, 3, base_index + 2)
        plt.imshow(distorted_image)
        plt.title(name)
        plt.axis("off")

        plt.subplot(rows, 3, base_index + 3)
        plt.imshow(restored_image)
        plt.title("Restored")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def safe_filename(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def main() -> None:
    args = parse_args()
    distortion_dir = Path(args.distortion_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_files = {
        "original": distortion_dir / "original.png",
        "Gaussian noise": distortion_dir / "gaussian_noise.png",
        "Salt and pepper": distortion_dir / "salt_and_pepper.png",
        "Gaussian blur": distortion_dir / "gaussian_blur.png",
        "Brightness/contrast": distortion_dir / "brightness_contrast.png",
        "Rotation": distortion_dir / "rotation.png",
        "Perspective warp": distortion_dir / "perspective_warp.png",
        "Barrel distortion": distortion_dir / "barrel_distortion.png",
        "Pixelation": distortion_dir / "pixelation.png",
    }

    missing = [str(path) for path in required_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing distortion files. Run distortion_methods.py first.\n"
            + "\n".join(missing)
        )

    original = load_image(required_files["original"])
    restoration_steps = {
        "Gaussian noise": denoise_gaussian,
        "Salt and pepper": remove_salt_and_pepper,
        "Gaussian blur": deblur_sharpen,
        "Brightness/contrast": normalize_intensity,
        "Rotation": restore_rotation,
        "Perspective warp": perspective_restore,
        "Barrel distortion": pincushion_correction,
        "Pixelation": smooth_pixelation,
    }

    restored: dict[str, tuple[Image.Image, Image.Image]] = {}
    for name, restore_function in restoration_steps.items():
        distorted = load_image(required_files[name])
        restored_image = restore_function(distorted)
        restored[name] = (distorted, restored_image)
        restored_image.save(output_dir / f"{safe_filename(name)}_restored.png")

    save_restoration_grid(original, restored, output_dir / "restoration_grid.png")
    (output_dir / "methods.txt").write_text(
        "\n".join(
            [
                "Image Restoration Techniques",
                "",
                "These techniques reduce or partially reverse distortions. Some lost",
                "information cannot be fully recovered, especially blur and pixelation.",
                "",
                "1. Gaussian noise: smoothing plus sharpening.",
                "2. Salt and pepper noise: median filtering.",
                "3. Gaussian blur: unsharp masking.",
                "4. Brightness/contrast: autocontrast and contrast normalization.",
                "5. Rotation: inverse rotation.",
                "6. Perspective warp: inverse perspective transform.",
                "7. Barrel distortion: pincushion-style geometric correction.",
                "8. Pixelation: interpolation and mild sharpening.",
            ]
        ),
        encoding="utf-8",
    )

    print("Saved restoration results to:", output_dir.resolve())


if __name__ == "__main__":
    main()
