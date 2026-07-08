import argparse
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
DISTORTION_DIR = Path("results/distortions")
RESTORATION_DIR = Path("results/restoration")
OUTPUT_DIR = Path("results/cnn_on_distortion_restoration")
IMAGE_SIZE = (224, 224)


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
    return parser.parse_args()


def load_image(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize(IMAGE_SIZE)
    return np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)


def predict_image(model: tf.keras.Model, image_path: Path) -> tuple[str, float]:
    tumor_probability = float(model.predict(load_image(image_path), verbose=0)[0][0])
    label = "tumor" if tumor_probability >= 0.5 else "no_tumor"
    return label, tumor_probability


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
    header = "group,method,image_path,prediction,tumor_probability"
    lines = [header]
    for row in rows:
        lines.append(
            f'{row["group"]},{row["method"]},"{row["image_path"]}",'
            f'{row["prediction"]},{row["tumor_probability"]}'
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


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
            f'{row["prediction"]:8} | tumor_probability={row["tumor_probability"]}'
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

    rows: list[dict[str, str]] = []
    for image_path in collect_pngs(distortion_dir):
        prediction, probability = predict_image(model, image_path)
        rows.append(
            {
                "group": "distorted",
                "method": method_name_from_file(image_path),
                "image_path": str(image_path.resolve()),
                "prediction": prediction,
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
                "prediction": prediction,
                "tumor_probability": f"{probability:.6f}",
            }
        )

    save_csv(rows, output_dir / "cnn_predictions.csv")
    save_summary(rows, output_dir / "summary.txt")
    save_probability_chart(rows, output_dir / "tumor_probability_chart.png")

    print("Saved CNN distortion/restoration evaluation to:", output_dir.resolve())


if __name__ == "__main__":
    main()
