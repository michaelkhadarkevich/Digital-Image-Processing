import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


MODEL_PATH = Path("models/cnn_basic.keras")
DEFAULT_INPUT_DIR = Path("data/Data 1/test")
ALL_IMAGES_DATA_DIR = Path("data/Data 1 with distortion/all_images")
OUTPUT_DIR = Path("result/results task 2/cnn_on_distortion")
IMAGE_SIZE = (224, 224)
VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
SNR_METHODS = [
    "gaussian_noise_snr_5db",
    "gaussian_noise_snr_10db",
    "gaussian_noise_snr_15db",
    "gaussian_noise_snr_20db",
    "gaussian_noise_snr_30db",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the trained CNN on distorted images only.")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Trained CNN model path.")
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR),
        help="Directory where distortion-only CNN results are saved.",
    )
    return parser.parse_args()


def load_image_for_model(image_path: Path):
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize(IMAGE_SIZE)
    import numpy as np

    return np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_input_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
    )


def true_label_from_path(image_path: Path) -> str:
    parts = {part.lower() for part in image_path.parts}
    if "tumor" in parts:
        return "tumor"
    if "no_tumor" in parts:
        return "no_tumor"
    raise ValueError(f"Could not infer true label from path: {image_path}")


def predict_image(model: tf.keras.Model, image_path: Path) -> tuple[str, float]:
    tumor_probability = float(model.predict(load_image_for_model(image_path), verbose=0)[0][0])
    label = "tumor" if tumor_probability >= 0.5 else "no_tumor"
    return label, tumor_probability


def confusion_type(true_label: str, prediction: str) -> str:
    actual = "T" if true_label == "tumor" else "F"
    predicted = "T" if prediction == "tumor" else "F"
    return actual + predicted


def safe_image_dir_name(image_path: Path) -> str:
    relative = image_path.relative_to(DEFAULT_INPUT_DIR)
    return "__".join(relative.with_suffix("").parts)


def append_prediction_row(
    rows: list[dict[str, str]],
    model: tf.keras.Model,
    image_path: Path,
    group: str,
    method: str,
    true_label: str,
) -> None:
    prediction, probability = predict_image(model, image_path)
    rows.append(
        {
            "group": group,
            "method": method,
            "image_path": str(image_path.resolve()),
            "true_label": true_label,
            "prediction": prediction,
            "is_correct": str(prediction == true_label),
            "confusion_type": confusion_type(true_label, prediction),
            "tumor_probability": f"{probability:.6f}",
        }
    )


def ordered_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    methods = [
        "original",
        "gaussian_noise",
        "salt_and_pepper",
        "gaussian_blur",
        "brightness_contrast",
        "rotation",
        "perspective_warp",
        "barrel_distortion",
        "pixelation",
        "gaussian_noise_snr_5db",
        "gaussian_noise_snr_10db",
        "gaussian_noise_snr_15db",
        "gaussian_noise_snr_20db",
        "gaussian_noise_snr_30db",
        "salt_and_pepper_density_0_01",
        "salt_and_pepper_density_0_03",
        "salt_and_pepper_density_0_05",
        "salt_and_pepper_density_0_10",
        "gaussian_blur_sigma_0_5",
        "gaussian_blur_sigma_1_0",
        "gaussian_blur_sigma_1_5",
        "gaussian_blur_sigma_2_0",
        "gaussian_blur_sigma_3_0",
    ]
    order = {method: index for index, method in enumerate(methods)}
    return sorted(rows, key=lambda row: (order.get(row["method"], 999), row["image_path"]))


def metric_key(row: dict[str, str]) -> str:
    if row["group"] == "clean":
        return "clean original"
    return f'{row["method"]} distorted'


def save_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    header = "group,method,image_path,true_label,prediction,is_correct,confusion_type,tumor_probability"
    lines = [header]
    for row in rows:
        lines.append(
            f'{row["group"]},{row["method"]},"{row["image_path"]}",'
            f'{row["true_label"]},{row["prediction"]},{row["is_correct"]},'
            f'{row["confusion_type"]},{row["tumor_probability"]}'
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_summary(rows: list[dict[str, str]], output_path: Path) -> None:
    lines = [
        "CNN Evaluation on Distorted Images",
        "",
        "The tumor_probability column is the CNN output probability for the tumor class.",
        "Probability >= 0.5 is predicted as tumor; otherwise it is predicted as no_tumor.",
        "",
    ]
    for row in rows:
        lines.append(
            f'{row["group"]:10} | {row["method"]:24} | '
            f'true={row["true_label"]:8} | predicted={row["prediction"]:8} | '
            f'correct={row["is_correct"]:5} | type={row["confusion_type"]} | '
            f'tumor_probability={row["tumor_probability"]}'
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_confusion_percentages(rows: list[dict[str, str]], output_dir: Path) -> None:
    keys = []
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        key = metric_key(row)
        if key not in counts:
            keys.append(key)
            counts[key] = {"TT": 0, "TF": 0, "FT": 0, "FF": 0, "total": 0}
        counts[key][row["confusion_type"]] += 1
        counts[key]["total"] += 1

    lines = ["source,TT_percent,TF_percent,FT_percent,FF_percent,total"]
    true_percentages = []
    for key in keys:
        total = counts[key]["total"]
        values = {
            name: (counts[key][name] / total * 100) if total else 0.0
            for name in ["TT", "TF", "FT", "FF"]
        }
        true_percentages.append(values["TT"] + values["FF"])
        lines.append(
            f"{key},"
            f"{values['TT']:.2f},"
            f"{values['TF']:.2f},"
            f"{values['FT']:.2f},"
            f"{values['FF']:.2f},"
            f"{total}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "confusion_percentages.csv").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    plt.figure(figsize=(max(12, len(keys) * 0.7), 5))
    plt.bar(range(len(keys)), true_percentages, color="#59A14F")
    plt.xticks(range(len(keys)), keys, rotation=65, ha="right")
    plt.ylabel("Percent")
    plt.ylim(0, 100)
    plt.title("Total True Percent By Distortion")
    plt.tight_layout()
    plt.savefig(output_dir / "total_true_percent_graph.png", dpi=160)
    plt.close()

    snr_keys = [f"{method} distorted" for method in SNR_METHODS if f"{method} distorted" in counts]
    if snr_keys:
        snr_values = [
            (counts[key]["TT"] + counts[key]["FF"]) / counts[key]["total"] * 100
            for key in snr_keys
        ]
        snr_labels = [key.removeprefix("gaussian_noise_snr_").removesuffix(" distorted") for key in snr_keys]
        plt.figure(figsize=(8, 5))
        plt.bar(range(len(snr_keys)), snr_values, color="#4E79A7")
        plt.xticks(range(len(snr_keys)), snr_labels)
        plt.ylabel("Percent")
        plt.ylim(0, 100)
        plt.title("Total True Percent By Gaussian Noise SNR")
        plt.tight_layout()
        plt.savefig(output_dir / "snr_total_true_percent_graph.png", dpi=160)
        plt.close()


def run_evaluation(model: tf.keras.Model, output_dir: Path) -> None:
    distortion_module = load_module(
        Path("task/distortion methods/distortion_methods.py"),
        "distortion_methods_runtime",
    )
    image_paths = collect_input_images(DEFAULT_INPUT_DIR)
    if not image_paths:
        raise FileNotFoundError("No test images found under data/Data 1/test.")

    rows: list[dict[str, str]] = []
    for image_path in image_paths:
        true_label = true_label_from_path(image_path)
        image_dir = ALL_IMAGES_DATA_DIR / safe_image_dir_name(image_path)
        distortion_dir = image_dir / "distortions"
        distortion_dir.mkdir(parents=True, exist_ok=True)

        original = distortion_module.load_image(image_path)
        distortions = distortion_module.cnn_level_distortions(original)

        for method, image in distortions.items():
            distorted_path = distortion_dir / f"{method}.png"
            image.save(distorted_path)
            group = "clean" if method == "original" else "distorted"
            append_prediction_row(rows, model, distorted_path, group, method, true_label)

    rows = ordered_rows(rows)
    save_csv(rows, output_dir / "cnn_predictions_all_images.csv")
    save_summary(rows, output_dir / "summary_all_images.txt")
    save_confusion_percentages(rows, output_dir)
    print("Saved distortion-only CNN data to:", str(ALL_IMAGES_DATA_DIR.resolve()))
    print("Saved distortion-only CNN evaluation to:", str(output_dir.resolve()))


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    output_dir = Path(args.output)

    if not model_path.exists():
        raise FileNotFoundError("Model not found. Run train_cnn.py first.")

    output_dir.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(model_path)
    run_evaluation(model, output_dir)


if __name__ == "__main__":
    main()
