import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


DEFAULT_INPUT_DIR = Path("data/processed/test")
DISTORTION_DIR = Path("results/distortions")
RESTORATION_DIR = Path("results/restoration")
OUTPUT_DIR = Path("results/super_resolution")
IMAGE_SIZE = (224, 224)
DOWNSCALE_FACTOR = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Super-resolution task with x2 downsampling.")
    parser.add_argument(
        "--image",
        help="Path to one input image. If omitted, a test image is chosen automatically.",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR),
        help="Directory where super-resolution results are saved.",
    )
    parser.add_argument(
        "--include-derived",
        action="store_true",
        help="Also run super-resolution on distortion and restoration result images.",
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


def downsample_x2(image: Image.Image) -> Image.Image:
    low_resolution_size = (
        image.size[0] // DOWNSCALE_FACTOR,
        image.size[1] // DOWNSCALE_FACTOR,
    )
    return image.resize(low_resolution_size, Image.Resampling.BICUBIC)


def nearest_reconstruction(low_resolution: Image.Image) -> Image.Image:
    return low_resolution.resize(IMAGE_SIZE, Image.Resampling.NEAREST)


def bilinear_reconstruction(low_resolution: Image.Image) -> Image.Image:
    return low_resolution.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)


def bicubic_reconstruction(low_resolution: Image.Image) -> Image.Image:
    return low_resolution.resize(IMAGE_SIZE, Image.Resampling.BICUBIC)


def lanczos_reconstruction(low_resolution: Image.Image) -> Image.Image:
    return low_resolution.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)


def sharpened_lanczos_reconstruction(low_resolution: Image.Image) -> Image.Image:
    upscaled = lanczos_reconstruction(low_resolution)
    return upscaled.filter(ImageFilter.UnsharpMask(radius=1.2, percent=160, threshold=2))


def mse(original: Image.Image, reconstructed: Image.Image) -> float:
    original_array = np.asarray(original, dtype=np.float32)
    reconstructed_array = np.asarray(reconstructed, dtype=np.float32)
    return float(np.mean((original_array - reconstructed_array) ** 2))


def psnr(original: Image.Image, reconstructed: Image.Image) -> float:
    error = mse(original, reconstructed)
    if error == 0:
        return math.inf
    return 20 * math.log10(255.0 / math.sqrt(error))


def save_grid(images: dict[str, Image.Image], output_path: Path) -> None:
    names = list(images.keys())
    columns = 3
    rows = int(math.ceil(len(names) / columns))

    plt.figure(figsize=(10, 3.5 * rows))
    for index, name in enumerate(names, start=1):
        plt.subplot(rows, columns, index)
        plt.imshow(images[name])
        plt.title(name)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def collect_pngs(directory: Path) -> list[Path]:
    excluded = {
        "distortion_grid.png",
        "restoration_grid.png",
        "super_resolution_grid.png",
        "downsampled_x2_preview.png",
    }
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.glob("*.png")
        if path.is_file() and path.name not in excluded
    )


def super_resolve_image(image_path: Path, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    original = load_image(image_path)
    low_resolution = downsample_x2(original)
    low_resolution_preview = low_resolution.resize(IMAGE_SIZE, Image.Resampling.NEAREST)

    reconstructions = {
        "nearest": nearest_reconstruction(low_resolution),
        "bilinear": bilinear_reconstruction(low_resolution),
        "bicubic": bicubic_reconstruction(low_resolution),
        "lanczos": lanczos_reconstruction(low_resolution),
        "sharpened_lanczos": sharpened_lanczos_reconstruction(low_resolution),
    }

    original.save(output_dir / "original.png")
    low_resolution.save(output_dir / "downsampled_x2.png")
    low_resolution_preview.save(output_dir / "downsampled_x2_preview.png")

    grid_images = {
        "Original 224x224": original,
        "Downsampled x2 preview": low_resolution_preview,
        **{name.replace("_", " ").title(): image for name, image in reconstructions.items()},
    }
    save_grid(grid_images, output_dir / "super_resolution_grid.png")

    metric_lines = [
        "Super-Resolution Results",
        "",
        f"Input image: {image_path.resolve()}",
        f"Original size: {original.size[0]}x{original.size[1]}",
        f"Downsampled x2 size: {low_resolution.size[0]}x{low_resolution.size[1]}",
        "",
        "method,mse,psnr",
    ]

    for name, image in reconstructions.items():
        image.save(output_dir / f"{name}.png")
        metric_lines.append(f"{name},{mse(original, image):.4f},{psnr(original, image):.4f}")

    (output_dir / "metrics.csv").write_text("\n".join(metric_lines), encoding="utf-8")
    return metric_lines


def run_single_image(image_path: Path, output_dir: Path) -> None:
    super_resolve_image(image_path, output_dir)
    (output_dir / "methods.txt").write_text(
        "\n".join(
            [
                "Super-Resolution Task",
                "",
                "Step 1: Downsample the image by x2.",
                "Step 2: Upscale the low-resolution image back to the original size.",
                "",
                "Methods:",
                "1. Nearest-neighbor interpolation.",
                "2. Bilinear interpolation.",
                "3. Bicubic interpolation.",
                "4. Lanczos interpolation.",
                "5. Lanczos interpolation followed by unsharp masking.",
                "",
                "PSNR is higher when the reconstructed image is closer to the original.",
            ]
        ),
        encoding="utf-8",
    )

    print("Input image:", image_path.resolve())
    print("Saved single-image super-resolution results to:", output_dir.resolve())


def run_collection(group_name: str, image_paths: list[Path], output_dir: Path) -> None:
    group_dir = output_dir / group_name
    group_dir.mkdir(parents=True, exist_ok=True)

    summary_lines = [
        f"Super-Resolution on {group_name.title()} Images",
        "",
        "image,method,mse,psnr",
    ]

    for image_path in image_paths:
        image_output_dir = group_dir / image_path.stem
        metric_lines = super_resolve_image(image_path, image_output_dir)
        for line in metric_lines:
            if not line or line.startswith(("Super-", "Input", "Original", "Downsampled", "method")):
                continue
            summary_lines.append(f"{image_path.name},{line}")

    (group_dir / "summary_metrics.csv").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )


def write_methods(output_dir: Path) -> None:
    (output_dir / "methods.txt").write_text(
        "\n".join(
            [
                "Super-Resolution Task",
                "",
                "Step 1: Downsample the image by x2.",
                "Step 2: Upscale the low-resolution image back to the original size.",
                "",
                "Methods:",
                "1. Nearest-neighbor interpolation.",
                "2. Bilinear interpolation.",
                "3. Bicubic interpolation.",
                "4. Lanczos interpolation.",
                "5. Lanczos interpolation followed by unsharp masking.",
                "",
                "The same process can be run on original, distorted, and restored images.",
                "PSNR is higher when the reconstructed image is closer to the input before downsampling.",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    image_path = Path(args.image) if args.image else find_default_image()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_single_image(image_path, output_dir)
    write_methods(output_dir)

    if args.include_derived:
        distortion_images = collect_pngs(DISTORTION_DIR)
        restoration_images = collect_pngs(RESTORATION_DIR)
        if not distortion_images:
            raise FileNotFoundError("No distortion images found. Run distortion_methods.py first.")
        if not restoration_images:
            raise FileNotFoundError("No restoration images found. Run image_restoration.py first.")

        run_collection("distortions", distortion_images, output_dir)
        run_collection("restoration", restoration_images, output_dir)
        print("Saved distortion super-resolution results to:", (output_dir / "distortions").resolve())
        print("Saved restoration super-resolution results to:", (output_dir / "restoration").resolve())


if __name__ == "__main__":
    main()
