from pathlib import Path
import csv

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


RESULTS_TASK_1 = Path("result/results task 1")
DISTORTION_RESTORATION_CSV = (
    RESULTS_TASK_1 / "cnn_on_distortion_restoration" / "confusion_percentages.csv"
)
FINE_TUNE_CSV = RESULTS_TASK_1 / "cnn_fine_tune_distortion" / "summary.csv"
OUTPUT_PATH = (
    RESULTS_TASK_1
    / "cnn_compare_snr_distortion_restoration_fine_tune_total_true_percent_graph.png"
)
SALT_PEPPER_OUTPUT_PATH = (
    RESULTS_TASK_1
    / "cnn_compare_salt_and_pepper_distortion_restoration_fine_tune_total_true_percent_graph.png"
)
BLUR_OUTPUT_PATH = (
    RESULTS_TASK_1
    / "cnn_compare_gaussian_blur_distortion_restoration_fine_tune_total_true_percent_graph.png"
)

SNR_METHODS = [
    ("5db", "gaussian_noise_snr_5db"),
    ("10db", "gaussian_noise_snr_10db"),
    ("15db", "gaussian_noise_snr_15db"),
    ("20db", "gaussian_noise_snr_20db"),
    ("30db", "gaussian_noise_snr_30db"),
]
SALT_PEPPER_METHODS = [
    ("0.01", "salt_and_pepper_density_0_01"),
    ("0.03", "salt_and_pepper_density_0_03"),
    ("0.05", "salt_and_pepper_density_0_05"),
    ("0.10", "salt_and_pepper_density_0_10"),
]
BLUR_METHODS = [
    ("0.5", "gaussian_blur_sigma_0_5"),
    ("1.0", "gaussian_blur_sigma_1_0"),
    ("1.5", "gaussian_blur_sigma_1_5"),
    ("2.0", "gaussian_blur_sigma_2_0"),
    ("3.0", "gaussian_blur_sigma_3_0"),
]


def total_true_percent(row: dict[str, str]) -> float:
    return float(row["TT_percent"]) + float(row["FF_percent"])


def read_distortion_restoration() -> tuple[dict[str, float], dict[str, float]]:
    distorted = {}
    restored = {}
    with DISTORTION_RESTORATION_CSV.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            source = row["source"]
            if source.endswith(" distorted"):
                method = source.removesuffix(" distorted")
                distorted[method] = total_true_percent(row)
            elif source.endswith(" restored"):
                method = source.removesuffix(" restored")
                restored[method] = total_true_percent(row)
    return distorted, restored


def read_fine_tune() -> dict[str, float]:
    fine_tuned = {}
    with FINE_TUNE_CSV.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            fine_tuned[row["method"]] = float(row["test_accuracy"]) * 100
    return fine_tuned


def save_comparison_graph(
    methods: list[tuple[str, str]],
    distorted: dict[str, float],
    restored: dict[str, float],
    fine_tuned: dict[str, float],
    title: str,
    output_path: Path,
) -> None:
    labels = []
    distorted_values = []
    restored_values = []
    fine_tuned_values = []
    for label, method in methods:
        labels.append(label)
        distorted_values.append(distorted.get(method, 0.0))
        restored_values.append(restored.get(method, 0.0))
        fine_tuned_values.append(fine_tuned.get(method, 0.0))

    x_positions = np.arange(len(labels))
    width = 0.26

    plt.figure(figsize=(9, 5))
    plt.bar(
        x_positions - width,
        distorted_values,
        width,
        label="CNN on distortion",
        color="#4E79A7",
    )
    plt.bar(
        x_positions,
        restored_values,
        width,
        label="CNN on restoration",
        color="#F28E2B",
    )
    plt.bar(
        x_positions + width,
        fine_tuned_values,
        width,
        label="CNN fine tuned",
        color="#59A14F",
    )

    plt.xticks(x_positions, labels)
    plt.ylabel("Total true percent")
    plt.ylim(0, 100)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    if not DISTORTION_RESTORATION_CSV.exists():
        raise FileNotFoundError(DISTORTION_RESTORATION_CSV)
    if not FINE_TUNE_CSV.exists():
        raise FileNotFoundError(FINE_TUNE_CSV)

    distorted, restored = read_distortion_restoration()
    fine_tuned = read_fine_tune()

    save_comparison_graph(
        SNR_METHODS,
        distorted,
        restored,
        fine_tuned,
        "CNN Total True Percent By Gaussian Noise SNR",
        OUTPUT_PATH,
    )
    save_comparison_graph(
        SALT_PEPPER_METHODS,
        distorted,
        restored,
        fine_tuned,
        "CNN Total True Percent By Salt And Pepper Density",
        SALT_PEPPER_OUTPUT_PATH,
    )
    save_comparison_graph(
        BLUR_METHODS,
        distorted,
        restored,
        fine_tuned,
        "CNN Total True Percent By Gaussian Blur Sigma",
        BLUR_OUTPUT_PATH,
    )

    print("Saved comparison graphs to:", RESULTS_TASK_1)


if __name__ == "__main__":
    main()
