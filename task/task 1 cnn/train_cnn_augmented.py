import shutil
from pathlib import Path
import sys
from datetime import datetime
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


DATA_DIR = Path("data/Data 1 augmented")
MODEL_PATH = Path("models/cnn_augmented.keras")
RESULTS_DIR = Path("result/results task 1/cnn_on_augmented")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
ACCURACY_TOLERANCE = 0.001


def backup_existing_model() -> Path | None:
    if not MODEL_PATH.exists():
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = MODEL_PATH.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{MODEL_PATH.stem}_{timestamp}{MODEL_PATH.suffix}"
    shutil.copy2(MODEL_PATH, backup_path)
    print("Backed up existing model to:", backup_path.resolve())
    return backup_path


def read_saved_score(metrics_path: Path) -> tuple[float, float]:
    if not metrics_path.exists():
        return float("inf"), -1.0

    text = metrics_path.read_text(encoding="utf-8")
    loss_match = re.search(r"Test loss:\s*([0-9.]+)", text)
    accuracy_match = re.search(r"Test accuracy:\s*([0-9.]+)%", text)
    loss = float(loss_match.group(1)) if loss_match else float("inf")
    accuracy = float(accuracy_match.group(1)) / 100.0 if accuracy_match else -1.0
    return loss, accuracy


def is_better_score(loss: float, accuracy: float, best_loss: float, best_accuracy: float) -> bool:
    if accuracy > best_accuracy + ACCURACY_TOLERANCE:
        return True
    return abs(accuracy - best_accuracy) <= ACCURACY_TOLERANCE and loss < best_loss


class BestTestCheckpoint(tf.keras.callbacks.Callback):
    def __init__(
        self,
        test_ds: tf.data.Dataset,
        output_path: Path,
        best_loss: float,
        best_accuracy: float,
    ) -> None:
        super().__init__()
        self.test_ds = test_ds
        self.output_path = output_path
        self.best_loss = best_loss
        self.best_accuracy = best_accuracy
        self.best_epoch = 0
        self.improved = False

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        test_loss, test_accuracy = self.model.evaluate(self.test_ds, verbose=0)
        print(
            f"Epoch {epoch + 1} test loss: {test_loss:.4f} - "
            f"test accuracy: {test_accuracy:.2%}"
        )

        if not is_better_score(test_loss, test_accuracy, self.best_loss, self.best_accuracy):
            return

        self.best_loss = test_loss
        self.best_accuracy = test_accuracy
        self.best_epoch = epoch + 1
        self.improved = True
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(self.output_path)
        print("Saved new best model from epoch", self.best_epoch)


def build_model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))

    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
    x = tf.keras.layers.RandomFlip("horizontal")(x)
    x = tf.keras.layers.RandomRotation(0.05)(x)
    x = tf.keras.layers.RandomZoom(0.1)(x)

    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


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
    test_loss: float,
    test_accuracy: float,
    confusion_matrix: np.ndarray,
    class_names: list[str],
) -> None:
    lines = [
        "Brain MRI CNN Results",
        "",
        f"Test loss: {test_loss:.4f}",
        f"Test accuracy: {test_accuracy:.2%}",
        "",
        "Confusion matrix rows are actual labels; columns are predicted labels.",
        f"Classes: {', '.join(class_names)}",
        str(confusion_matrix),
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    backup_path = backup_existing_model()
    previous_loss, previous_accuracy = read_saved_score(RESULTS_DIR / "metrics.txt")

    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"
    test_dir = DATA_DIR / "test"
    if not train_dir.exists() or not val_dir.exists() or not test_dir.exists():
        raise FileNotFoundError("Run preprocess.py before training.")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        seed=42,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False,
    )
    class_names = test_ds.class_names
    test_image_paths = test_ds.file_paths

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(500).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)
    test_ds = test_ds.cache().prefetch(buffer_size=autotune)

    if MODEL_PATH.exists():
        print("Loading existing model from:", MODEL_PATH.resolve())
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        model = build_model()

    best_test_checkpoint = BestTestCheckpoint(
        test_ds,
        MODEL_PATH,
        previous_loss,
        previous_accuracy,
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        best_test_checkpoint,
    ]

    model.summary()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    if best_test_checkpoint.improved:
        print("Using best test epoch:", best_test_checkpoint.best_epoch)
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        print(
            "No epoch beat the saved best "
            f"({previous_accuracy:.2%}, loss {previous_loss:.4f})."
        )
        if backup_path is not None:
            shutil.copy2(backup_path, MODEL_PATH)
            print("Restored previous best model from:", backup_path.resolve())
        return

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    probabilities = model.predict(test_ds, verbose=0).reshape(-1)
    predicted_labels = (probabilities >= 0.5).astype(int)
    actual_labels = np.concatenate([labels.numpy().reshape(-1).astype(int) for _, labels in test_ds])
    confusion_matrix = np.zeros((2, 2), dtype=int)
    for actual, predicted in zip(actual_labels, predicted_labels):
        confusion_matrix[actual, predicted] += 1

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.2%}")

    print("Saved model to:", MODEL_PATH.resolve())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_training_history(history, RESULTS_DIR / "training_history.png")
    plot_confusion_matrix(confusion_matrix, class_names, RESULTS_DIR / "confusion_matrix.png")
    save_predictions(
        test_image_paths,
        actual_labels,
        probabilities,
        class_names,
        RESULTS_DIR / "predictions.csv",
    )
    save_metrics(
        RESULTS_DIR / "metrics.txt",
        test_loss,
        test_accuracy,
        confusion_matrix,
        class_names,
    )
    print("Saved results to:", RESULTS_DIR.resolve())


if __name__ == "__main__":
    main()
