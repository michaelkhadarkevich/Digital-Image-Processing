import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


MODEL_PATH = Path("models/brain_mri_classifier.keras")
DEFAULT_INPUT_DIR = Path("data/Data 1/test")
DISTORTION_DIR = Path("data/Data 1 with distortion/distortions")
RESTORATION_DIR = Path("data/Data 1 with distortion/restoration")
ALL_IMAGES_DATA_DIR = Path("data/Data 1 with distortion/all_images")
OUTPUT_DIR = Path("result/results task 1/cnn_on_distortion_restoration")
IMAGE_SIZE = (224, 224)
TRUE_LABEL = "no_tumor"
VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the trained CNN on distorted and restored images."
    )
    parser.add_argument("--model", default=str(MODEL_PATH), help="Trained CNN model path.")
    parser.add_argument(
        "--distortion-dir",
        default=str(DISTORTION_DIR),
        help="Directory with distortion output images.",
    )
    parser.add_argument(
        "--restoration-dir",
        default=str(RESTORATION_DIR),
        help="Directory with restoration output images.",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR),
        help="Directory where evaluation results are saved.",
    )
    parser.add_argument(
        "--all-images",
        action="store_true",
        help="Evaluate clean, distorted, and restored versions of every test image.",
    )
    return parser.parse_args()


def load_image(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize(IMAGE_SIZE)
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
    tumor_probability = float(model.predict(load_image(image_path), verbose=0)[0][0])
    label = "tumor" if tumor_probability >= 0.5 else "no_tumor"
    return label, tumor_probability


def confusion_type(true_label: str, prediction: str) -> str:
    actual = "T" if true_label == "tumor" else "F"
    predicted = "T" if prediction == "tumor" else "F"
    return actual + predicted


def collect_pngs(directory: Path) -> list[Path]:
    excluded = {"distortion_grid.png", "restoration_grid.png"}
    return sorted(
        path
        for path in directory.glob("*.png")
        if path.name not in excluded and path.is_file()
    )


def method_name_from_file(path: Path, suffix: str = "") -> str:
    name = path.stem
    if suffix and name.endswith(suffix):
        name = name[: -len(suffix)]
    return name


def save_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    header = "group,method,image_path,true_label,prediction,is_correct,confusion_type,tumor_probability"
    lines = [header]
    for row in rows:
        lines.append(
            f'{row["group"]},{row["method"]},"{row["image_path"]}",'
            f'{row["true_label"]},{row["prediction"]},{row["is_correct"]},'
            f'{row["confusion_type"]},'
            f'{row["tumor_probability"]}'
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def paired_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    for row in rows:
        if row["group"] == "distorted" and row["method"] == "clean":
            row["group"] = "clean"
        if row["group"] == "clean" and row["method"] == "clean":
            row["method"] = "original"

    distorted_methods = sorted({
        row["method"]
        for row in rows
        if row["group"] == "distorted"
    })
    restored_methods = sorted({
        row["method"]
        for row in rows
        if row["group"] == "restored"
    })

    ordered = []
    ordered.extend(row for row in rows if row["group"] == "clean")
    for method in distorted_methods:
        ordered.extend(
            row
            for row in rows
            if row["group"] == "distorted" and row["method"] == method
        )
        if method in restored_methods:
            ordered.extend(
                row
                for row in rows
                if row["group"] == "restored" and row["method"] == method
            )

    for method in restored_methods:
        if method not in distorted_methods:
            ordered.extend(
                row
                for row in rows
                if row["group"] == "restored" and row["method"] == method
            )

    return ordered


def save_summary(rows: list[dict[str, str]], output_path: Path) -> None:
    lines = [
        "CNN Evaluation on Distorted and Restored Images",
        "",
        "The tumor_probability column is the CNN output probability for the tumor class.",
        "Probability >= 0.5 is predicted as tumor; otherwise it is predicted as no_tumor.",
        "",
    ]
    for row in rows:
        lines.append(
            f'{row["group"]:12} | {row["method"]:28} | '
            f'true={row["true_label"]:8} | predicted={row["prediction"]:8} | '
            f'correct={row["is_correct"]:5} | type={row["confusion_type"]} | '
            f'tumor_probability={row["tumor_probability"]}'
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_probability_chart(rows: list[dict[str, str]], output_path: Path) -> None:
    labels = [f'{row["group"]}\n{row["method"]}' for row in rows]
    probabilities = [float(row["tumor_probability"]) for row in rows]
    colors = ["#4C78A8" if row["group"] == "distorted" else "#59A14F" for row in rows]

    plt.figure(figsize=(max(12, len(rows) * 0.65), 5))
    plt.bar(range(len(rows)), probabilities, color=colors)
    plt.axhline(0.5, color="#D62728", linestyle="--", linewidth=1.5, label="Decision threshold")
    plt.xticks(range(len(rows)), labels, rotation=65, ha="right")
    plt.ylabel("Tumor probability")
    plt.ylim(0, 1)
    plt.title("CNN Predictions on Distorted and Restored Images")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def metric_key(row: dict[str, str]) -> str:
    if row["group"] == "clean":
        return "clean original"
    return f'{row["method"]} {row["group"]}'


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
    for key in keys:
        total = counts[key]["total"]
        values = {
            name: (counts[key][name] / total * 100) if total else 0.0
            for name in ["TT", "TF", "FT", "FF"]
        }
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

    true_percentages = [
        ((counts[key]["TT"] + counts[key]["FF"]) / counts[key]["total"] * 100)
        if counts[key]["total"]
        else 0.0
        for key in keys
    ]
    plt.figure(figsize=(max(12, len(keys) * 0.7), 5))
    plt.bar(range(len(keys)), true_percentages, color="#59A14F")
    plt.xticks(range(len(keys)), keys, rotation=65, ha="right")
    plt.ylabel("Percent")
    plt.ylim(0, 100)
    plt.title("Total True Percent By Image Type")
    plt.tight_layout()
    plt.savefig(output_dir / "total_true_percent_graph.png", dpi=160)
    plt.close()


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


def run_all_images_evaluation(model: tf.keras.Model, output_dir: Path) -> None:
    distortion_module = load_module(
        Path("task/distortion methods/distortion_methods.py"),
        "distortion_methods_runtime",
    )
    restoration_module = load_module(
        Path("task/restoration/image_restoration.py"),
        "image_restoration_runtime",
    )

    image_paths = collect_input_images(DEFAULT_INPUT_DIR)
    if not image_paths:
        raise FileNotFoundError("No test images found under data/Data 1/test.")

    rows: list[dict[str, str]] = []
    for image_path in image_paths:
        true_label = true_label_from_path(image_path)
        image_dir = ALL_IMAGES_DATA_DIR / safe_image_dir_name(image_path)
        distortion_dir = image_dir / "distortions"
        restoration_dir = image_dir / "restoration"
        distortion_dir.mkdir(parents=True, exist_ok=True)
        restoration_dir.mkdir(parents=True, exist_ok=True)

        original = distortion_module.load_image(image_path)
        distortions = {
            "clean": original,
            "gaussian_noise": distortion_module.gaussian_noise(original),
            "salt_and_pepper": distortion_module.salt_and_pepper_noise(original),
            "gaussian_blur": distortion_module.blur(original),
            "brightness_contrast": distortion_module.brightness_contrast(original),
            "rotation": distortion_module.rotate(original),
            "perspective_warp": distortion_module.perspective_warp(original),
            "barrel_distortion": distortion_module.barrel_distortion(original),
            "pixelation": distortion_module.pixelate(original),
        }

        for method, image in distortions.items():
            distorted_path = distortion_dir / f"{method}.png"
            image.save(distorted_path)
            group = "clean" if method == "clean" else "distorted"
            label_method = "original" if method == "clean" else method
            append_prediction_row(rows, model, distorted_path, group, label_method, true_label)

        restoration_steps = {
            "gaussian_noise": restoration_module.denoise_gaussian,
            "salt_and_pepper": restoration_module.remove_salt_and_pepper,
            "gaussian_blur": restoration_module.deblur_sharpen,
            "brightness_contrast": restoration_module.normalize_intensity,
            "rotation": restoration_module.restore_rotation,
            "perspective_warp": restoration_module.perspective_restore,
            "barrel_distortion": restoration_module.pincushion_correction,
            "pixelation": restoration_module.smooth_pixelation,
        }
        for method, restore_function in restoration_steps.items():
            restored_image = restore_function(distortions[method])
            restored_path = restoration_dir / f"{method}_restored.png"
            restored_image.save(restored_path)
            append_prediction_row(rows, model, restored_path, "restored", method, true_label)

    rows = paired_rows(rows)
    save_csv(rows, output_dir / "cnn_predictions_all_images.csv")
    save_summary(rows, output_dir / "summary_all_images.txt")
    save_confusion_percentages(rows, output_dir)
    print("Saved all-image CNN distortion/restoration data to:", ALL_IMAGES_DATA_DIR.resolve())
    print("Saved all-image CNN evaluation to:", output_dir.resolve())


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    distortion_dir = Path(args.distortion_dir)
    restoration_dir = Path(args.restoration_dir)
    output_dir = Path(args.output)

    if not model_path.exists():
        raise FileNotFoundError("Model not found. Run train_cnn.py first.")
    if not distortion_dir.exists():
        raise FileNotFoundError("Distortion folder not found. Run distortion_methods.py first.")
    if not restoration_dir.exists():
        raise FileNotFoundError("Restoration folder not found. Run image_restoration.py first.")

    model = tf.keras.models.load_model(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all_images:
        run_all_images_evaluation(model, output_dir)
        return

    rows: list[dict[str, str]] = []
    for image_path in collect_pngs(distortion_dir):
        prediction, probability = predict_image(model, image_path)
        method = method_name_from_file(image_path)
        group = "clean" if method == "clean" else "distorted"
        label_method = "original" if method == "clean" else method
        rows.append(
            {
                "group": group,
                "method": label_method,
                "image_path": str(image_path.resolve()),
                "true_label": TRUE_LABEL,
                "prediction": prediction,
                "is_correct": str(prediction == TRUE_LABEL),
                "confusion_type": confusion_type(TRUE_LABEL, prediction),
                "tumor_probability": f"{probability:.6f}",
            }
        )

    for image_path in collect_pngs(restoration_dir):
        prediction, probability = predict_image(model, image_path)
        rows.append(
            {
                "group": "restored",
                "method": method_name_from_file(image_path, "_restored"),
                "image_path": str(image_path.resolve()),
                "true_label": TRUE_LABEL,
                "prediction": prediction,
                "is_correct": str(prediction == TRUE_LABEL),
                "confusion_type": confusion_type(TRUE_LABEL, prediction),
                "tumor_probability": f"{probability:.6f}",
            }
        )

    rows = paired_rows(rows)
    save_csv(rows, output_dir / "cnn_predictions.csv")
    save_summary(rows, output_dir / "summary.txt")
    save_probability_chart(rows, output_dir / "tumor_probability_chart.png")

    print("Saved CNN distortion/restoration evaluation to:", output_dir.resolve())


if __name__ == "__main__":
    main()
