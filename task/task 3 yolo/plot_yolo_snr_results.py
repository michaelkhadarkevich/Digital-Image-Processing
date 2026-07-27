import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


RESULTS_ROOT = Path("result/results task 3/yolo_fine_tuned")
OUTPUT_DIR = Path("result/results task 3/yolo_snr_results")
BASIC_METRICS_PATH = OUTPUT_DIR / "yolo_basic_distortion_level_metrics.csv"
BASIC_RESTORED_METRICS_PATH = OUTPUT_DIR / "yolo_basic_restored_distortion_level_metrics.csv"
DISTORTION_GROUPS = {
    "gaussian_noise_snr": {
        "title": "YOLO Segmentation Performance By Gaussian Noise SNR",
        "map50_output": "yolo_gaussian_noise_snr_map50_graph.png",
        "map50_95_output": "yolo_gaussian_noise_snr_map50_95_graph.png",
        "levels": [
            ("5db", "gaussian_noise_snr_5db"),
            ("15db", "gaussian_noise_snr_15db"),
            ("30db", "gaussian_noise_snr_30db"),
        ],
    },
    "gaussian_blur_sigma": {
        "title": "YOLO Segmentation Performance By Gaussian Blur Sigma",
        "map50_output": "yolo_gaussian_blur_sigma_map50_graph.png",
        "map50_95_output": "yolo_gaussian_blur_sigma_map50_95_graph.png",
        "levels": [
            ("0.5", "gaussian_blur_sigma_0_5"),
            ("1.5", "gaussian_blur_sigma_1_5"),
            ("3.0", "gaussian_blur_sigma_3_0"),
        ],
    },
    "brightness_contrast": {
        "title": "YOLO Segmentation Performance By Brightness/Contrast Level",
        "map50_output": "yolo_brightness_contrast_level_map50_graph.png",
        "map50_95_output": "yolo_brightness_contrast_level_map50_95_graph.png",
        "levels": [
            ("B1.15 C1.25", "brightness_contrast_b1_15_c1_25"),
            ("B1.35 C1.55", "brightness_contrast_b1_35_c1_55"),
            ("B1.55 C1.80", "brightness_contrast_b1_55_c1_80"),
        ],
    },
}


def read_last_metrics(method: str) -> dict[str, float]:
    results_path = (
        RESULTS_ROOT
        / f"yolo_segmentation_{method}_10epochs"
        / "train"
        / "results.csv"
    )
    if not results_path.exists():
        raise FileNotFoundError(results_path)

    with results_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"No training rows found in {results_path}")

    row = rows[-1]
    return {
        "mask_map50": float(row["metrics/mAP50(M)"]),
        "mask_map50_95": float(row["metrics/mAP50-95(M)"]),
    }


def read_basic_metrics() -> dict[str, dict[str, float]]:
    if not BASIC_METRICS_PATH.exists():
        raise FileNotFoundError(BASIC_METRICS_PATH)

    metrics_by_method = {}
    with BASIC_METRICS_PATH.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            metrics_by_method[row["method"]] = {
                "mask_map50": float(row["basic_mask_map50"]),
                "mask_map50_95": float(row["basic_mask_map50_95"]),
            }

    return metrics_by_method


def read_basic_restored_metrics() -> dict[str, dict[str, float]]:
    if not BASIC_RESTORED_METRICS_PATH.exists():
        raise FileNotFoundError(BASIC_RESTORED_METRICS_PATH)

    metrics_by_method = {}
    with BASIC_RESTORED_METRICS_PATH.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            metrics_by_method[row["method"]] = {
                "mask_map50": float(row["basic_restored_mask_map50"]),
                "mask_map50_95": float(row["basic_restored_mask_map50_95"]),
            }

    return metrics_by_method


def save_metrics_csv(rows: list[dict[str, str | float]]) -> None:
    output_path = OUTPUT_DIR / "yolo_distortion_level_metrics.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "distortion_type",
                "level",
                "method",
                "basic_mask_map50",
                "basic_restored_mask_map50",
                "fine_tuned_mask_map50",
                "basic_mask_map50_95",
                "basic_restored_mask_map50_95",
                "fine_tuned_mask_map50_95",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_metric_graph(
    labels: list[str],
    basic_values: list[float],
    basic_restored_values: list[float],
    fine_tuned_values: list[float],
    title: str,
    metric_name: str,
    output_path: Path,
) -> None:
    x_positions = np.arange(len(labels))
    width = 0.26

    plt.figure(figsize=(9, 5))
    basic_bars = plt.bar(
        x_positions - width,
        basic_values,
        width,
        label=f"Basic YOLO on distorted {metric_name}",
        color="#4E79A7",
    )
    basic_restored_bars = plt.bar(
        x_positions,
        basic_restored_values,
        width,
        label=f"Basic YOLO on restored {metric_name}",
        color="#F28E2B",
    )
    fine_tuned_bars = plt.bar(
        x_positions + width,
        fine_tuned_values,
        width,
        label=f"Fine-tuned YOLO {metric_name}",
        color="#59A14F",
    )

    plt.xticks(x_positions, labels)
    plt.ylabel(metric_name)
    plt.ylim(0, 1.0)
    plt.title(f"{title} - {metric_name}")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend()

    for bars in [basic_bars, basic_restored_bars, fine_tuned_bars]:
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.015,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    basic_metrics_by_method = read_basic_metrics()
    basic_restored_metrics_by_method = read_basic_restored_metrics()
    rows = []

    for distortion_type, group in DISTORTION_GROUPS.items():
        labels = []
        basic_map50_values = []
        basic_restored_map50_values = []
        fine_tuned_map50_values = []
        basic_map50_95_values = []
        basic_restored_map50_95_values = []
        fine_tuned_map50_95_values = []
        for label, method in group["levels"]:
            fine_tuned_metrics = read_last_metrics(method)
            basic_metrics = basic_metrics_by_method[method]
            basic_restored_metrics = basic_restored_metrics_by_method[method]
            labels.append(label)
            basic_map50_values.append(basic_metrics["mask_map50"])
            basic_restored_map50_values.append(basic_restored_metrics["mask_map50"])
            fine_tuned_map50_values.append(fine_tuned_metrics["mask_map50"])
            basic_map50_95_values.append(basic_metrics["mask_map50_95"])
            basic_restored_map50_95_values.append(basic_restored_metrics["mask_map50_95"])
            fine_tuned_map50_95_values.append(fine_tuned_metrics["mask_map50_95"])
            rows.append(
                {
                    "distortion_type": distortion_type,
                    "level": label,
                    "method": method,
                    "basic_mask_map50": basic_metrics["mask_map50"],
                    "basic_restored_mask_map50": basic_restored_metrics["mask_map50"],
                    "fine_tuned_mask_map50": fine_tuned_metrics["mask_map50"],
                    "basic_mask_map50_95": basic_metrics["mask_map50_95"],
                    "basic_restored_mask_map50_95": basic_restored_metrics["mask_map50_95"],
                    "fine_tuned_mask_map50_95": fine_tuned_metrics["mask_map50_95"],
                }
            )

        save_metric_graph(
            labels,
            basic_map50_values,
            basic_restored_map50_values,
            fine_tuned_map50_values,
            group["title"],
            "mAP50",
            OUTPUT_DIR / group["map50_output"],
        )
        save_metric_graph(
            labels,
            basic_map50_95_values,
            basic_restored_map50_95_values,
            fine_tuned_map50_95_values,
            group["title"],
            "mAP50-95",
            OUTPUT_DIR / group["map50_95_output"],
        )

    save_metrics_csv(rows)
    print("Saved YOLO distortion-level graphs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
