import argparse
import csv
from pathlib import Path


DEFAULT_MODEL = Path("models/yolo/yolo_tumor_seg.pt")
DEFAULT_INPUT = Path("data/Data 1/test")
DEFAULT_OUTPUT = Path("result/results task 3/yolo_tumor_segmentation")
VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trained YOLO tumor segmentation.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Trained YOLO segmentation model.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Image or folder to segment.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output folder.")
    parser.add_argument("--confidence", type=float, default=0.25)
    return parser.parse_args()


def import_yolo():
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Ultralytics YOLO is not installed. Install it with:\n"
            "pip install ultralytics"
        ) from exc
    return YOLO


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in VALID_SUFFIXES else []
    if not input_path.exists():
        return []
    return sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
    )


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    input_path = Path(args.input)
    output_dir = Path(args.output)
    annotated_dir = output_dir / "annotated"
    labels_dir = output_dir / "labels"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained YOLO segmentation model not found: {model_path}\n"
            'Run: python "task/task 3 yolo/train_yolo_segmentation.py"'
        )

    image_paths = collect_images(input_path)
    if not image_paths:
        raise FileNotFoundError(f"No images found under: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    YOLO = import_yolo()
    model = YOLO(str(model_path))

    rows = []
    for image_path in image_paths:
        results = model.predict(
            source=str(image_path),
            conf=args.confidence,
            save=False,
            verbose=False,
        )
        result = results[0]
        annotated_path = annotated_dir / f"{image_path.stem}_tumor_seg.jpg"
        label_path = labels_dir / f"{image_path.stem}.txt"

        result.save(filename=str(annotated_path))

        detections = 0
        top_confidence = 0.0
        if result.boxes is not None and len(result.boxes) > 0:
            detections = len(result.boxes)
            top_confidence = max(float(box.conf[0]) for box in result.boxes)

        with label_path.open("w", encoding="utf-8") as label_file:
            if result.masks is not None and result.boxes is not None:
                for box, segment in zip(result.boxes, result.masks.xyn):
                    confidence = float(box.conf[0])
                    points = " ".join(f"{value:.6f}" for xy in segment for value in xy)
                    label_file.write(f"tumor {confidence:.6f} {points}\n")

        rows.append(
            {
                "image": str(image_path),
                "annotated_image": str(annotated_path),
                "detections": detections,
                "top_class": "tumor" if detections else "none",
                "top_confidence": f"{top_confidence:.6f}",
            }
        )

    with (output_dir / "tumor_segmentation_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image", "annotated_image", "detections", "top_class", "top_confidence"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Saved tumor segmentation results to:", output_dir.resolve())


if __name__ == "__main__":
    main()
