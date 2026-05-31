import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


MODEL_PATH = Path("models/brain_mri_classifier.keras")
IMAGE_SIZE = (224, 224)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict tumor/no-tumor from one MRI image.")
    parser.add_argument("--image", required=True, help="Image path to classify.")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Trained model path.")
    return parser.parse_args()


def load_image(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.array(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    image_path = Path(args.image)

    if not model_path.exists():
        raise FileNotFoundError("Model not found. Run train.py first.")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = tf.keras.models.load_model(model_path)
    probability = float(model.predict(load_image(image_path), verbose=0)[0][0])
    label = "tumor" if probability >= 0.5 else "no_tumor"
    confidence = probability if label == "tumor" else 1 - probability

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
