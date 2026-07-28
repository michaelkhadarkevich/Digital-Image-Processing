import csv
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

import super_resolution


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


OUTPUT_DIR = Path("result/results task 1/super_resolution_snr")
RESTORATION_MODULE_PATH = Path("task/restoration/image_restoration.py")
LEVEL_GROUPS = {
    "gaussian_noise_snr": {
        "title": "Super-Resolution By Gaussian Noise SNR",
        "levels": [
            ("5db", "gaussian_noise_snr_5db"),
            ("10db", "gaussian_noise_snr_10db"),
            ("15db", "gaussian_noise_snr_15db"),
            ("20db", "gaussian_noise_snr_20db"),
            ("30db", "gaussian_noise_snr_30db"),
        ],
    },
    "salt_and_pepper_density": {
        "title": "Super-Resolution By Salt And Pepper Density",
        "levels": [
            ("0.01", "salt_and_pepper_density_0_01"),
            ("0.03", "salt_and_pepper_density_0_03"),
            ("0.05", "salt_and_pepper_density_0_05"),
            ("0.10", "salt_and_pepper_density_0_10"),
        ],
    },
    "gaussian_blur_sigma": {
        "title": "Super-Resolution By Gaussian Blur Sigma",
        "levels": [
            ("0.5", "gaussian_blur_sigma_0_5"),
            ("1.0", "gaussian_blur_sigma_1_0"),
            ("1.5", "gaussian_blur_sigma_1_5"),
            ("2.0", "gaussian_blur_sigma_2_0"),
            ("3.0", "gaussian_blur_sigma_3_0"),
        ],
    },
}


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


def find_default_image() -> Path:
    return super_resolution.find_default_image()


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").resize(super_resolution.IMAGE_SIZE)


def gaussian_noise_snr(image: Image.Image, snr_db: float) -> Image.Image:
    array = np.asarray(image).astype(np.float32)
    signal_power = np.mean(array * array)
    sigma = 0.0 if signal_power <= 0 else float(np.sqrt(signal_power / (10 ** (snr_db / 10))))
    noise = np.random.default_rng(42).normal(0, sigma, array.shape)
    noisy = np.clip(array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def salt_and_pepper_noise(image: Image.Image, amount: float) -> Image.Image:
    array = np.asarray(image).copy()
    rng = np.random.default_rng(42)
    mask = rng.random(array.shape[:2])
    array[mask < amount / 2] = 0
    array[mask > 1 - amount / 2] = 255
    return Image.fromarray(array)


def gaussian_blur_sigma(image: Image.Image, sigma: float) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def distort_image(image: Image.Image, method: str) -> Image.Image:
    if method.startswith("gaussian_noise_snr_") and method.endswith("db"):
        snr_db = float(method.removeprefix("gaussian_noise_snr_").removesuffix("db"))
        return gaussian_noise_snr(image, snr_db)
    if method.startswith("salt_and_pepper_density_"):
        density = float(method.removeprefix("salt_and_pepper_density_").replace("_", "."))
        return salt_and_pepper_noise(image, density)
    if method.startswith("gaussian_blur_sigma_"):
        sigma = float(method.removeprefix("gaussian_blur_sigma_").replace("_", "."))
        return gaussian_blur_sigma(image, sigma)
    raise KeyError(f"No distortion function for method: {method}")


def restore_image(image: Image.Image, method: str, restoration_module) -> Image.Image:
    if method.startswith("gaussian_noise_snr_"):
        return restoration_module.denoise_gaussian(image)
    if method.startswith("salt_and_pepper_density_"):
        return restoration_module.remove_salt_and_pepper(image)
    if method.startswith("gaussian_blur_sigma_"):
        return restoration_module.deblur_sharpen(image)
    raise KeyError(f"No restoration function for method: {method}")


def read_method_metrics(path: Path) -> dict[str, dict[str, float]]:
    rows = {}
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = None
        for row in reader:
            if not row:
                continue
            if row[0] == "method":
                header = row
                continue
            if header is None or row[0].startswith(("Super-", "Input", "Original", "Downsampled")):
                continue
            values = dict(zip(header, row))
            rows[values["method"]] = {
                "psnr": float(values["psnr"]),
                "ssim": float(values["ssim"]),
                "reconstruction_score": float(values["reconstruction_score"]),
            }
    return rows


def write_summary(rows: list[dict[str, str | float]]) -> None:
    output_path = OUTPUT_DIR / "snr_super_resolution_metrics.csv"
    fieldnames = [
        "distortion_type",
        "level",
        "method",
        "image_type",
        "reconstruction_method",
        "psnr",
        "ssim",
        "reconstruction_score",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_metric_graph(
    labels: list[str],
    distorted_values: list[float],
    restored_values: list[float],
    title: str,
    metric: str,
    output_path: Path,
) -> None:
    x_positions = np.arange(len(labels))
    width = 0.34

    plt.figure(figsize=(9, 5))
    distorted_bars = plt.bar(
        x_positions - width / 2,
        distorted_values,
        width,
        label="Distorted",
        color="#4E79A7",
    )
    restored_bars = plt.bar(
        x_positions + width / 2,
        restored_values,
        width,
        label="Restored",
        color="#59A14F",
    )

    plt.xticks(x_positions, labels)
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(
        f"{title} - {metric.replace('_', ' ').title()} "
        "(sharpened_lanczos)"
    )
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend()

    for bars in [distorted_bars, restored_bars]:
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}" if metric != "ssim" else f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    distortion_dir = OUTPUT_DIR / "distortions"
    restoration_dir = OUTPUT_DIR / "restoration"
    results_dir = OUTPUT_DIR / "super_resolution_results"
    distortion_dir.mkdir(parents=True, exist_ok=True)
    restoration_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    image_path = find_default_image()
    original = load_image(image_path)
    restoration_module = load_restoration_module()
    rows: list[dict[str, str | float]] = []
    graph_values: dict[str, dict[str, dict[str, list[float] | list[str]]]] = {}

    for distortion_type, group in LEVEL_GROUPS.items():
        graph_values[distortion_type] = {
            "psnr": {"labels": [], "distorted": [], "restored": []},
            "ssim": {"labels": [], "distorted": [], "restored": []},
            "reconstruction_score": {"labels": [], "distorted": [], "restored": []},
        }
        for label, method in group["levels"]:
            distorted = distort_image(original, method)
            restored = restore_image(distorted, method, restoration_module)

            distorted_path = distortion_dir / f"{method}.png"
            restored_path = restoration_dir / f"{method}_restored.png"
            distorted.save(distorted_path)
            restored.save(restored_path)

            for image_type, source_path in [
                ("distorted", distorted_path),
                ("restored", restored_path),
            ]:
                image_output_dir = results_dir / image_type / method
                super_resolution.super_resolve_image(source_path, image_output_dir)
                metrics_by_reconstruction = read_method_metrics(image_output_dir / "metrics.csv")

                for reconstruction_method, metrics in metrics_by_reconstruction.items():
                    rows.append(
                        {
                            "distortion_type": distortion_type,
                            "level": label,
                            "method": method,
                            "image_type": image_type,
                            "reconstruction_method": reconstruction_method,
                            "psnr": metrics["psnr"],
                            "ssim": metrics["ssim"],
                            "reconstruction_score": metrics["reconstruction_score"],
                        }
                    )

                best_metrics = metrics_by_reconstruction["sharpened_lanczos"]
                for metric in ["psnr", "ssim", "reconstruction_score"]:
                    if image_type == "distorted":
                        graph_values[distortion_type][metric]["labels"].append(label)
                    graph_values[distortion_type][metric][image_type].append(best_metrics[metric])

    write_summary(rows)

    for distortion_type, group in LEVEL_GROUPS.items():
        for metric in ["psnr", "ssim", "reconstruction_score"]:
            values = graph_values[distortion_type][metric]
            save_metric_graph(
                values["labels"],
                values["distorted"],
                values["restored"],
                group["title"],
                metric,
                OUTPUT_DIR / f"{distortion_type}_{metric}_graph.png",
            )

    (OUTPUT_DIR / "methods.txt").write_text(
        "\n".join(
            [
                "Task 1 Super-Resolution SNR/Level Results",
                "",
                f"Input image: {image_path.resolve()}",
                "",
                "This folder runs super-resolution on the same parameterized",
                "distortion levels used in Task 2.",
                "",
                "Graphs use the sharpened_lanczos reconstruction method.",
                "The CSV stores all reconstruction methods.",
            ]
        ),
        encoding="utf-8",
    )
    print("Saved Task 1 SNR super-resolution results to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
