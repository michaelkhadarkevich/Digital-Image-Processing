import argparse
import csv
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


DEFAULT_INPUT_DIR = Path("data/Data 1/test")
DISTORTION_DIR = Path("data/Data 1 with distortion/distortions")
RESTORATION_DIR = Path("data/Data 1 with distortion/restoration")
OUTPUT_DIR = Path("result/results task 2/super_resolution")
IMAGE_SIZE = (224, 224)
DOWNSCALE_FACTOR = 2
RECONSTRUCTION_METHODS = [
    "nearest",
    "bilinear",
    "bicubic",
    "lanczos",
    "sharpened_lanczos",
]


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


def mae(original: Image.Image, reconstructed: Image.Image) -> float:
    original_array = np.asarray(original, dtype=np.float32)
    reconstructed_array = np.asarray(reconstructed, dtype=np.float32)
    return float(np.mean(np.abs(original_array - reconstructed_array)))


def rmse(original: Image.Image, reconstructed: Image.Image) -> float:
    return math.sqrt(mse(original, reconstructed))


def psnr(original: Image.Image, reconstructed: Image.Image) -> float:
    error = mse(original, reconstructed)
    if error == 0:
        return math.inf
    return 20 * math.log10(255.0 / math.sqrt(error))


def ssim(original: Image.Image, reconstructed: Image.Image) -> float:
    original_gray = np.asarray(original.convert("L"), dtype=np.float64)
    reconstructed_gray = np.asarray(reconstructed.convert("L"), dtype=np.float64)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mean_original = original_gray.mean()
    mean_reconstructed = reconstructed_gray.mean()
    variance_original = original_gray.var()
    variance_reconstructed = reconstructed_gray.var()
    covariance = ((original_gray - mean_original) * (reconstructed_gray - mean_reconstructed)).mean()

    numerator = (2 * mean_original * mean_reconstructed + c1) * (2 * covariance + c2)
    denominator = (mean_original**2 + mean_reconstructed**2 + c1) * (
        variance_original + variance_reconstructed + c2
    )
    return float(numerator / denominator)


def reconstruction_score(original: Image.Image, reconstructed: Image.Image) -> float:
    score = 100 * (1 - mae(original, reconstructed) / 255)
    return max(0.0, min(100.0, score))


def calculate_metrics(original: Image.Image, reconstructed: Image.Image) -> dict[str, float]:
    return {
        "mse": mse(original, reconstructed),
        "rmse": rmse(original, reconstructed),
        "mae": mae(original, reconstructed),
        "psnr": psnr(original, reconstructed),
        "ssim": ssim(original, reconstructed),
        "reconstruction_score": reconstruction_score(original, reconstructed),
    }


def metric_csv_line(name: str, metrics: dict[str, float]) -> str:
    return (
        f"{name},"
        f"{metrics['mse']:.4f},"
        f"{metrics['rmse']:.4f},"
        f"{metrics['mae']:.4f},"
        f"{metrics['psnr']:.4f},"
        f"{metrics['ssim']:.6f},"
        f"{metrics['reconstruction_score']:.2f}"
    )


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
        "method,mse,rmse,mae,psnr,ssim,reconstruction_score",
    ]

    for name, image in reconstructions.items():
        image.save(output_dir / f"{name}.png")
        metrics = calculate_metrics(original, image)
        metric_lines.append(metric_csv_line(name, metrics))

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
                "Success metrics:",
                "1. MSE/RMSE/MAE: lower is better.",
                "2. PSNR: higher is better.",
                "3. SSIM: closer to 1 is better.",
                "4. Reconstruction score: closer to 100 is better.",
            ]
        ),
        encoding="utf-8",
    )

    print("Input image:", image_path.resolve())
    print("Saved single-image super-resolution results to:", output_dir.resolve())


def run_collection(
    group_name: str,
    image_paths: list[Path],
    output_dir: Path,
    reference_image: Image.Image | None = None,
) -> None:
    group_dir = output_dir / group_name
    group_dir.mkdir(parents=True, exist_ok=True)

    summary_lines = [
        f"Super-Resolution on {group_name.title()} Images",
        "",
        "image,method,mse,rmse,mae,psnr,ssim,reconstruction_score",
    ]
    vs_original_lines = [
        f"Super-Resolution on {group_name.title()} Images Compared To Original Clean Image",
        "",
        "image,method,mse,rmse,mae,psnr,ssim,reconstruction_score",
    ]

    for image_path in image_paths:
        image_output_dir = group_dir / image_path.stem
        metric_lines = super_resolve_image(image_path, image_output_dir)
        for line in metric_lines:
            if not line or line.startswith(("Super-", "Input", "Original", "Downsampled", "method")):
                continue
            summary_lines.append(f"{image_path.name},{line}")
        if reference_image is not None:
            for method in RECONSTRUCTION_METHODS:
                reconstructed_path = image_output_dir / f"{method}.png"
                if not reconstructed_path.exists():
                    continue
                reconstructed = load_image(reconstructed_path)
                metrics = calculate_metrics(reference_image, reconstructed)
                vs_original_lines.append(f"{image_path.name},{metric_csv_line(method, metrics)}")

    (group_dir / "summary_metrics.csv").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )
    if reference_image is not None:
        (group_dir / "summary_metrics_vs_original.csv").write_text(
            "\n".join(vs_original_lines),
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
                "",
                "Success metrics:",
                "1. MSE/RMSE/MAE: lower is better.",
                "2. PSNR: higher is better.",
                "3. SSIM: closer to 1 is better.",
                "4. Reconstruction score: closer to 100 is better.",
            ]
        ),
        encoding="utf-8",
    )


def read_metrics_rows(path: Path, group_name: str) -> list[dict[str, str]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = None
        for row in reader:
            if not row:
                continue
            if row[0] == "method" or row[0] == "image":
                header = row
                continue
            if header is None:
                continue
            values = dict(zip(header, row))
            values["group"] = group_name
            rows.append(values)
    return rows


def average_metrics(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    metric_names = ["mse", "rmse", "mae", "psnr", "ssim", "reconstruction_score"]
    sums: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}

    for row in rows:
        method = row["method"]
        sums.setdefault(method, {metric: 0.0 for metric in metric_names})
        counts[method] = counts.get(method, 0) + 1
        for metric in metric_names:
            sums[method][metric] += float(row[metric])

    return {
        method: {
            metric: totals[metric] / counts[method]
            for metric in metric_names
        }
        for method, totals in sums.items()
    }


def safe_write_text(path: Path, text: str) -> Path:
    try:
        path.write_text(text, encoding="utf-8")
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_new{path.suffix}")
        fallback.write_text(text, encoding="utf-8")
        return fallback


def safe_savefig(fig: plt.Figure, path: Path) -> Path:
    try:
        fig.savefig(path, dpi=160)
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_new{path.suffix}")
        fig.savefig(fallback, dpi=160)
        return fallback


def paired_source_order(rows: list[dict[str, str]]) -> list[str]:
    sources = []
    for row in rows:
        source = row["source"]
        if source not in sources:
            sources.append(source)

    order = []
    if "Clean" in sources:
        order.append("Clean")

    distortions = [
        source
        for source in sources
        if source != "Clean" and not source.endswith("_restored")
    ]
    restorations = [
        source
        for source in sources
        if source.endswith("_restored")
    ]

    for distortion in distortions:
        if distortion not in order:
            order.append(distortion)
        restored = f"{distortion}_restored"
        if restored in restorations and restored not in order:
            order.append(restored)

    for source in sources:
        if source not in order:
            order.append(source)

    return order


def sort_rows_for_graph(
    rows: list[dict[str, str]],
    sources: list[str],
    methods: list[str],
) -> list[dict[str, str]]:
    source_index = {source: index for index, source in enumerate(sources)}
    method_index = {method: index for index, method in enumerate(methods)}
    return sorted(
        rows,
        key=lambda row: (
            source_index.get(row["source"], len(source_index)),
            method_index.get(row["method"], len(method_index)),
        ),
    )


def write_comparison_outputs(output_dir: Path) -> None:
    clean_rows = read_metrics_rows(output_dir / "Clean" / "metrics.csv", "Clean")
    distortion_rows = read_metrics_rows(
        output_dir / "distortions" / "summary_metrics.csv",
        "Distortions",
    )
    restoration_rows = read_metrics_rows(
        output_dir / "restoration" / "summary_metrics.csv",
        "Restoration",
    )
    distortion_vs_original_rows = read_metrics_rows(
        output_dir / "distortions" / "summary_metrics_vs_original.csv",
        "Distortions vs Original",
    )
    restoration_vs_original_rows = read_metrics_rows(
        output_dir / "restoration" / "summary_metrics_vs_original.csv",
        "Restoration vs Original",
    )

    detailed_rows = []
    for row in clean_rows:
        detailed_rows.append({"source": "Clean", **row})
    for row in distortion_rows:
        detailed_rows.append({"source": Path(row["image"]).stem, **row})
    for row in restoration_rows:
        detailed_rows.append({"source": Path(row["image"]).stem, **row})

    methods = []
    for row in detailed_rows:
        if row["method"] not in methods:
            methods.append(row["method"])
    labels = [method.replace("_", " ").title() for method in methods]
    sources = paired_source_order(detailed_rows)
    detailed_rows = sort_rows_for_graph(detailed_rows, sources, methods)

    detailed_lines = [
        "source,group,image,method,mse,rmse,mae,psnr,ssim,reconstruction_score",
    ]
    for row in detailed_rows:
        detailed_lines.append(
            f"{row['source']},"
            f"{row['group']},"
            f"{row.get('image', '')},"
            f"{row['method']},"
            f"{float(row['mse']):.4f},"
            f"{float(row['rmse']):.4f},"
            f"{float(row['mae']):.4f},"
            f"{float(row['psnr']):.4f},"
            f"{float(row['ssim']):.6f},"
            f"{float(row['reconstruction_score']):.2f}"
        )
    safe_write_text(output_dir / "detailed_comparison_metrics.csv", "\n".join(detailed_lines))

    graph_files = {
        "psnr": "detailed_psnr_graph.png",
        "ssim": "detailed_ssim_graph.png",
        "reconstruction_score": "detailed_reconstruction_score_graph.png",
    }
    for metric, filename in graph_files.items():
        matrix = np.full((len(sources), len(methods)), np.nan)
        for row in detailed_rows:
            row_index = sources.index(row["source"])
            column_index = methods.index(row["method"])
            matrix[row_index, column_index] = float(row[metric])

        fig, axis = plt.subplots(figsize=(10, max(7, len(sources) * 0.42)))
        image = axis.imshow(matrix, aspect="auto", cmap="viridis")
        axis.set_title(f"{metric.replace('_', ' ').upper()} by Distortion and Method")
        axis.set_xticks(np.arange(len(methods)))
        axis.set_xticklabels(labels, rotation=35, ha="right")
        axis.set_yticks(np.arange(len(sources)))
        axis.set_yticklabels([source.replace("_", " ") for source in sources])
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        fig.tight_layout()
        safe_savefig(fig, output_dir / filename)
        plt.close(fig)

    vs_original_rows = []
    for row in clean_rows:
        vs_original_rows.append({"source": "Clean", **row})
    for row in distortion_vs_original_rows:
        vs_original_rows.append({"source": Path(row["image"]).stem, **row})
    for row in restoration_vs_original_rows:
        vs_original_rows.append({"source": Path(row["image"]).stem, **row})

    vs_original_sources = paired_source_order(vs_original_rows)
    vs_original_rows = sort_rows_for_graph(vs_original_rows, vs_original_sources, methods)

    vs_original_lines = [
        "source,group,image,method,mse,rmse,mae,psnr,ssim,reconstruction_score",
    ]
    for row in vs_original_rows:
        vs_original_lines.append(
            f"{row['source']},"
            f"{row['group']},"
            f"{row.get('image', '')},"
            f"{row['method']},"
            f"{float(row['mse']):.4f},"
            f"{float(row['rmse']):.4f},"
            f"{float(row['mae']):.4f},"
            f"{float(row['psnr']):.4f},"
            f"{float(row['ssim']):.6f},"
            f"{float(row['reconstruction_score']):.2f}"
        )
    safe_write_text(
        output_dir / "detailed_comparison_metrics_vs_original.csv",
        "\n".join(vs_original_lines),
    )

    vs_original_graph_files = {
        "psnr": "vs_original_psnr_graph.png",
        "ssim": "vs_original_ssim_graph.png",
        "reconstruction_score": "vs_original_reconstruction_score_graph.png",
    }
    for metric, filename in vs_original_graph_files.items():
        matrix = np.full((len(vs_original_sources), len(methods)), np.nan)
        for row in vs_original_rows:
            row_index = vs_original_sources.index(row["source"])
            column_index = methods.index(row["method"])
            matrix[row_index, column_index] = float(row[metric])

        fig, axis = plt.subplots(figsize=(10, max(7, len(vs_original_sources) * 0.42)))
        image = axis.imshow(matrix, aspect="auto", cmap="plasma")
        axis.set_title(f"{metric.replace('_', ' ').upper()} Compared To Original Clean Image")
        axis.set_xticks(np.arange(len(methods)))
        axis.set_xticklabels(labels, rotation=35, ha="right")
        axis.set_yticks(np.arange(len(vs_original_sources)))
        axis.set_yticklabels([source.replace("_", " ") for source in vs_original_sources])
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        fig.tight_layout()
        safe_savefig(fig, output_dir / filename)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    image_path = Path(args.image) if args.image else find_default_image()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_dir = output_dir / "Clean"

    run_single_image(image_path, clean_dir)
    write_methods(clean_dir)

    if args.include_derived:
        distortion_images = collect_pngs(DISTORTION_DIR)
        restoration_images = collect_pngs(RESTORATION_DIR)
        if not distortion_images:
            raise FileNotFoundError("No distortion images found. Run distortion_methods.py first.")
        if not restoration_images:
            raise FileNotFoundError("No restoration images found. Run image_restoration.py first.")

        reference_image = load_image(image_path)
        run_collection("distortions", distortion_images, output_dir, reference_image)
        run_collection("restoration", restoration_images, output_dir, reference_image)
        write_comparison_outputs(output_dir)
        print("Saved distortion super-resolution results to:", (output_dir / "distortions").resolve())
        print("Saved restoration super-resolution results to:", (output_dir / "restoration").resolve())
        print("Saved PSNR graph to:", (output_dir / "detailed_psnr_graph.png").resolve())
        print("Saved SSIM graph to:", (output_dir / "detailed_ssim_graph.png").resolve())
        print("Saved reconstruction score graph to:", (output_dir / "detailed_reconstruction_score_graph.png").resolve())
        print("Saved original-reference PSNR graph to:", (output_dir / "vs_original_psnr_graph.png").resolve())
        print("Saved original-reference SSIM graph to:", (output_dir / "vs_original_ssim_graph.png").resolve())
        print("Saved original-reference reconstruction score graph to:", (output_dir / "vs_original_reconstruction_score_graph.png").resolve())


if __name__ == "__main__":
    main()
