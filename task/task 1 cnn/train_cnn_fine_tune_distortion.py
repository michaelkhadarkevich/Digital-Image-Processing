import argparse
import gc
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance, ImageFilter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


DATA_DIR = Path("data/Data 1")
DISTORTED_DATA_DIR = Path("data/Data 1 with distortion/fine_tune")
BASE_MODEL_PATH = Path("models/cnn_basic.keras")
MODEL_OUTPUT_DIR = Path("models/cnn_fine_tune_distortion")
RESULTS_DIR = Path("result/results task 1/cnn_fine_tune_distortion")
BASE_DISTORTION_RESULTS_DIR = Path("result/results task 1/cnn_on_distortion_restoration")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 1
DEFAULT_EPOCHS = 5
VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
DISTORTION_METHODS = [
    "gaussian_noise",
    "salt_and_pepper",
    "gaussian_blur",
    "brightness_contrast",
    "rotation",
    "perspective_warp",
    "barrel_distortion",
    "pixelation",
]


class DistortedImageSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        samples: list[tuple[Path, int]],
        method: str,
        batch_size: int = BATCH_SIZE,
        shuffle: bool = False,
    ) -> None:
        super().__init__()
        self.samples = samples
        self.method = method
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(samples))
        self.labels = np.asarray([label for _, label in samples], dtype=np.int32)
        self.file_paths = [str(path) for path, _ in samples]
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.samples) / self.batch_size))

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        batch_indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        images = []
        labels = []

        for sample_index in batch_indexes:
            image_path, label = self.samples[int(sample_index)]
            image = apply_distortion(load_image(image_path), self.method)
            images.append(np.asarray(image, dtype=np.float32))
            labels.append(label)

        return np.stack(images), np.asarray(labels, dtype=np.float32).reshape(-1, 1)

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.default_rng().shuffle(self.indexes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune one CNN model for each distortion method."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DISTORTION_METHODS,
        choices=DISTORTION_METHODS,
        help="Distortion methods to train. Default: all methods.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Fine-tuning epochs per distortion. Default: {DEFAULT_EPOCHS}.",
    )
    parser.add_argument(
        "--force-data",
        action="store_true",
        help="Regenerate distorted datasets even if they already exist.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only generate distorted datasets; do not train models.",
    )
    parser.add_argument(
        "--graph-only",
        action="store_true",
        help="Only create total_true_percent_graph.png from summary.csv.",
    )
    return parser.parse_args()


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").resize(IMAGE_SIZE)


def gaussian_noise(image: Image.Image, sigma: float = 24.0) -> Image.Image:
    array = np.asarray(image).astype(np.float32)
    noise = np.random.default_rng(42).normal(0, sigma, array.shape)
    noisy = np.clip(array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def salt_and_pepper_noise(image: Image.Image, amount: float = 0.04) -> Image.Image:
    array = np.asarray(image).copy()
    rng = np.random.default_rng(42)
    mask = rng.random(array.shape[:2])
    array[mask < amount / 2] = 0
    array[mask > 1 - amount / 2] = 255
    return Image.fromarray(array)


def blur(image: Image.Image) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=3))


def brightness_contrast(image: Image.Image) -> Image.Image:
    image = ImageEnhance.Brightness(image).enhance(1.35)
    return ImageEnhance.Contrast(image).enhance(1.55)


def rotate(image: Image.Image) -> Image.Image:
    return image.rotate(18, resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0))


def find_perspective_coefficients(source: np.ndarray, target: np.ndarray) -> list[float]:
    matrix = []
    for (x, y), (u, v) in zip(target, source):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    values = source.flatten()
    return np.linalg.solve(np.asarray(matrix), values).tolist()


def perspective_warp(image: Image.Image) -> Image.Image:
    width, height = image.size
    source = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    target = np.array(
        [[20, 8], [width - 24, 24], [width - 6, height - 18], [10, height - 8]],
        dtype=np.float32,
    )
    coefficients = find_perspective_coefficients(source, target)
    return image.transform(
        image.size,
        Image.Transform.PERSPECTIVE,
        coefficients,
        Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0),
    )


def barrel_distortion(image: Image.Image, strength: float = 0.000015) -> Image.Image:
    array = np.asarray(image)
    height, width = array.shape[:2]
    center_x = width / 2
    center_y = height / 2
    y_indices, x_indices = np.indices((height, width))
    x = x_indices - center_x
    y = y_indices - center_y
    radius_squared = x * x + y * y
    factor = 1 + strength * radius_squared
    source_x = np.clip(center_x + x * factor, 0, width - 1).astype(np.int32)
    source_y = np.clip(center_y + y * factor, 0, height - 1).astype(np.int32)
    return Image.fromarray(array[source_y, source_x])


def pixelate(image: Image.Image, block_size: int = 12) -> Image.Image:
    small_size = (IMAGE_SIZE[0] // block_size, IMAGE_SIZE[1] // block_size)
    small = image.resize(small_size, Image.Resampling.BILINEAR)
    return small.resize(IMAGE_SIZE, Image.Resampling.NEAREST)


def apply_distortion(image: Image.Image, method: str) -> Image.Image:
    functions = {
        "gaussian_noise": gaussian_noise,
        "salt_and_pepper": salt_and_pepper_noise,
        "gaussian_blur": blur,
        "brightness_contrast": brightness_contrast,
        "rotation": rotate,
        "perspective_warp": perspective_warp,
        "barrel_distortion": barrel_distortion,
        "pixelation": pixelate,
    }
    return functions[method](image)


def source_images(split: str) -> list[Path]:
    split_dir = DATA_DIR / split
    return sorted(
        path
        for path in split_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
    )


def class_names() -> list[str]:
    train_dir = DATA_DIR / "train"
    return sorted(path.name for path in train_dir.iterdir() if path.is_dir())


def split_samples(split: str, names: list[str]) -> list[tuple[Path, int]]:
    samples = []
    label_by_class = {class_name: index for index, class_name in enumerate(names)}
    for image_path in source_images(split):
        samples.append((image_path, label_by_class[image_path.parent.name]))
    return samples


def prepare_distorted_dataset(method: str, force: bool) -> Path:
    output_root = DISTORTED_DATA_DIR / method
    output_root.mkdir(parents=True, exist_ok=True)
    note_path = output_root / "README.txt"
    note_path.write_text(
        "This fine-tune run applies the distortion in memory while training.\n"
        "Full distorted image copies are not saved here, so the disk does not fill up.\n",
        encoding="utf-8",
    )
    print("Prepared in-memory distorted dataset marker:", output_root.resolve())
    return output_root


def plot_training_history(history: tf.keras.callbacks.History, output_path: Path) -> None:
    epochs = range(1, len(history.history["loss"]) + 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history["accuracy"], label="Train")
    plt.plot(epochs, history.history["val_accuracy"], label="Validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history["loss"], label="Train")
    plt.plot(epochs, history.history["val_loss"], label="Validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str], output_path: Path) -> None:
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=25, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            plt.text(col, row, str(matrix[row, col]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_predictions(
    image_paths: list[str],
    actual_labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    lines = ["image_path,actual,predicted,tumor_probability"]
    for image_path, actual, probability in zip(image_paths, actual_labels, probabilities):
        predicted = 1 if probability >= 0.5 else 0
        lines.append(
            f'"{image_path}",{class_names[int(actual)]},{class_names[predicted]},{probability:.6f}'
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_metrics(
    output_path: Path,
    method: str,
    test_loss: float,
    test_accuracy: float,
    confusion_matrix: np.ndarray,
    class_names: list[str],
) -> None:
    lines = [
        "Brain MRI CNN Fine-Tune Distortion Results",
        "",
        f"Distortion method: {method}",
        f"Base model: {BASE_MODEL_PATH}",
        f"Test loss: {test_loss:.4f}",
        f"Test accuracy: {test_accuracy:.2%}",
        "",
        "Confusion matrix rows are actual labels; columns are predicted labels.",
        f"Classes: {', '.join(class_names)}",
        str(confusion_matrix),
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def train_method(method: str, epochs: int, force_data: bool) -> tuple[str, float, float]:
    if not BASE_MODEL_PATH.exists():
        raise FileNotFoundError("Base model not found. Run train_cnn.py first.")

    prepare_distorted_dataset(method, force_data)
    names = class_names()
    train_ds = DistortedImageSequence(split_samples("train", names), method, shuffle=True)
    val_ds = DistortedImageSequence(split_samples("val", names), method)
    test_ds = DistortedImageSequence(split_samples("test", names), method)
    model = tf.keras.models.load_model(BASE_MODEL_PATH)
    for layer in model.layers[:-3]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    result_dir = RESULTS_DIR / method
    result_dir.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_OUTPUT_DIR / f"brain_mri_classifier_{method}.keras"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ]
    print("Fine-tuning method:", method)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    probabilities = model.predict(test_ds, verbose=0).reshape(-1)
    predicted_labels = (probabilities >= 0.5).astype(int)
    actual_labels = test_ds.labels
    matrix = np.zeros((2, 2), dtype=int)
    for actual, predicted in zip(actual_labels, predicted_labels):
        matrix[actual, predicted] += 1

    plot_training_history(history, result_dir / "training_history.png")
    plot_confusion_matrix(matrix, names, result_dir / "confusion_matrix.png")
    save_predictions(test_ds.file_paths, actual_labels, probabilities, names, result_dir / "predictions.csv")
    save_metrics(result_dir / "metrics.txt", method, test_loss, test_accuracy, matrix, names)
    try:
        model.save(model_path)
        print("Saved fine-tuned model to:", model_path.resolve())
    except OSError as error:
        print("Could not save fine-tuned model:", error)
    print("Saved results to:", result_dir.resolve())
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    return method, test_loss, test_accuracy


def save_total_true_graph(summary_path: Path, output_path: Path) -> None:
    lines = summary_path.read_text(encoding="utf-8").splitlines()
    rows = [line.split(",") for line in lines[1:] if line.strip()]
    methods = [row[0] for row in rows]
    fine_tuned_percentages = [float(row[2]) * 100 for row in rows]

    base_percentages = {}
    base_path = BASE_DISTORTION_RESULTS_DIR / "confusion_percentages.csv"
    if base_path.exists():
        for line in base_path.read_text(encoding="utf-8").splitlines()[1:]:
            parts = line.split(",")
            if len(parts) < 5 or not parts[0].endswith(" distorted"):
                continue
            method = parts[0].removesuffix(" distorted")
            base_percentages[method] = float(parts[1]) + float(parts[4])

    x_positions = np.arange(len(methods))
    width = 0.36
    base_values = [base_percentages.get(method, 0.0) for method in methods]

    plt.figure(figsize=(max(12, len(methods) * 0.9), 5))
    plt.bar(
        x_positions - width / 2,
        base_values,
        width,
        label="CNN not fine tuned",
        color="#4E79A7",
    )
    plt.bar(
        x_positions + width / 2,
        fine_tuned_percentages,
        width,
        label="CNN fine tuned",
        color="#59A14F",
    )
    plt.xticks(x_positions, methods, rotation=65, ha="right")
    plt.ylabel("Percent")
    plt.ylim(0, 100)
    plt.title("Total True Percent By Distortion")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    if args.graph_only:
        save_total_true_graph(
            RESULTS_DIR / "summary.csv",
            RESULTS_DIR / "total_true_percent_graph.png",
        )
        print("Saved total true graph to:", (RESULTS_DIR / "total_true_percent_graph.png").resolve())
        return

    for method in args.methods:
        prepare_distorted_dataset(method, args.force_data)

    if args.prepare_only:
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_lines = ["method,test_loss,test_accuracy"]
    for method in args.methods:
        method_name, loss, accuracy = train_method(method, args.epochs, False)
        summary_lines.append(f"{method_name},{loss:.4f},{accuracy:.6f}")
        tf.keras.backend.clear_session()
        gc.collect()

    (RESULTS_DIR / "summary.csv").write_text("\n".join(summary_lines), encoding="utf-8")
    print("Saved fine-tune summary to:", (RESULTS_DIR / "summary.csv").resolve())
    save_total_true_graph(
        RESULTS_DIR / "summary.csv",
        RESULTS_DIR / "total_true_percent_graph.png",
    )
    print("Saved total true graph to:", (RESULTS_DIR / "total_true_percent_graph.png").resolve())


if __name__ == "__main__":
    main()
