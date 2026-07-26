import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path("result/results task 3/yolo_basic_vs_fine_tuned")
FINE_TUNED_ROOT = Path("result/results task 3/yolo_fine_tuned")


BASIC_RESULTS = {
    "gaussian_noise": {
        "mask_map50": 0.669,
        "mask_map50_95": 0.376,
    },
    "gaussian_blur": {
        "mask_map50": 0.787,
        "mask_map50_95": 0.443,
    },
    "brightness_contrast": {
        "mask_map50": 0.835,
        "mask_map50_95": 0.507,
    },
}

CLEAN_BASIC_RESULTS = {
    "mask_map50": 0.85081,
    "mask_map50_95": 0.52480,
}


FINE_TUNED_RESULTS_CSV = {
    "gaussian_noise": FINE_TUNED_ROOT
    / "yolo_segmentation_gaussian_noise_10epochs"
    / "train"
    / "results.csv",
    "gaussian_blur": FINE_TUNED_ROOT
    / "yolo_segmentation_gaussian_blur_10epochs"
    / "train"
    / "results.csv",
    "brightness_contrast": FINE_TUNED_ROOT
    / "yolo_segmentation_brightness_contrast_10epochs"
    / "train"
    / "results.csv",
}


def read_last_training_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"Missing fine-tuned YOLO results file: {path}")

    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"No rows found in: {path}")

    row = rows[-1]
    return {
        "mask_map50": float(row["metrics/mAP50(M)"]),
        "mask_map50_95": float(row["metrics/mAP50-95(M)"]),
    }


def save_metrics_csv(rows: list[dict[str, str | float]]) -> None:
    output_path = OUTPUT_DIR / "yolo_basic_vs_fine_tuned_metrics.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "distortion",
                "basic_mask_map50",
                "fine_tuned_mask_map50",
                "basic_mask_map50_95",
                "fine_tuned_mask_map50_95",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_grouped_bars(
    distortions: list[str],
    basic_values: list[float],
    fine_tuned_values: list[float],
    clean_baseline: float,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    x = np.arange(len(distortions))
    width = 0.36

    plt.figure(figsize=(10, 5.8))
    basic_bars = plt.bar(x - width / 2, basic_values, width, label="Basic YOLO")
    fine_bars = plt.bar(x + width / 2, fine_tuned_values, width, label="Fine-tuned YOLO")
    plt.axhline(
        clean_baseline,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Basic YOLO on clean data",
    )

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(x, [name.replace("_", " ") for name in distortions], rotation=12, ha="right")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend()

    for bars in [basic_bars, fine_bars]:
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

    distortions = ["gaussian_noise", "gaussian_blur", "brightness_contrast"]
    rows = []
    basic_map50 = []
    fine_map50 = []
    basic_map50_95 = []
    fine_map50_95 = []

    for distortion in distortions:
        basic = BASIC_RESULTS[distortion]
        fine_tuned = read_last_training_metrics(FINE_TUNED_RESULTS_CSV[distortion])

        basic_map50.append(basic["mask_map50"])
        fine_map50.append(fine_tuned["mask_map50"])
        basic_map50_95.append(basic["mask_map50_95"])
        fine_map50_95.append(fine_tuned["mask_map50_95"])

        rows.append(
            {
                "distortion": distortion,
                "basic_mask_map50": basic["mask_map50"],
                "fine_tuned_mask_map50": fine_tuned["mask_map50"],
                "basic_mask_map50_95": basic["mask_map50_95"],
                "fine_tuned_mask_map50_95": fine_tuned["mask_map50_95"],
            }
        )

    save_metrics_csv(rows)

    plot_grouped_bars(
        distortions,
        basic_map50,
        fine_map50,
        CLEAN_BASIC_RESULTS["mask_map50"],
        "Basic YOLO vs Fine-tuned YOLO on Distorted MRI Masks",
        "Mask mAP50",
        OUTPUT_DIR / "mask_map50_comparison.png",
    )
    plot_grouped_bars(
        distortions,
        basic_map50_95,
        fine_map50_95,
        CLEAN_BASIC_RESULTS["mask_map50_95"],
        "Basic YOLO vs Fine-tuned YOLO on Distorted MRI Masks",
        "Mask mAP50-95",
        OUTPUT_DIR / "mask_map50_95_comparison.png",
    )

    print("Saved comparison graphs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
