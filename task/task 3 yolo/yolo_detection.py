import argparse
import csv
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


CLEAR_DIR = Path("data/Data 1/test")
DISTORTION_DIR = Path("data/Data 1 with distortion/distortions")
RESTORATION_DIR = Path("data/Data 1 with distortion/restoration")
OUTPUT_DIR = Path("result/results task 3/yolo_detection")
VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO on clear, distorted, and restored images."
    )
    parser.add_argument(
        "--model",
        default="models/yolo/yolo11n.pt",
        help="YOLO model path/name. Example: models/yolo/yolo11n.pt, models/yolo/yolov8n.pt, or a custom .pt file.",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR),
        help="Directory where YOLO results are saved.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=12,
        help="Maximum clear images to evaluate. Distortion/restoration images are all evaluated.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="YOLO confidence threshold.",
    )
    return parser.parse_args()


def import_yolo():
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Ultralytics YOLO is not installed. Install it with:\n"
            "pip install ultralytics\n\n"
            "Then run:\n"
            'python "task/task 3 yolo\\yolo_detection.py"'
        ) from exc
    return YOLO


def collect_images(directory: Path, limit: int | None = None) -> list[Path]:
    excluded = {
        "distortion_grid.png",
        "restoration_grid.png",
        "tumor_probability_chart.png",
        "super_resolution_grid.png",
    }
    if not directory.exists():
        return []

    images = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VALID_SUFFIXES
        and path.name not in excluded
    )
    if limit is not None:
        return images[:limit]
    return images


def evaluate_group(
    model,
    group_name: str,
    image_paths: list[Path],
    output_dir: Path,
    confidence: float,
) -> list[dict[str, str]]:
    group_dir = output_dir / group_name
    annotated_dir = group_dir / "annotated"
    labels_dir = group_dir / "labels"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for image_path in image_paths:
        results = model.predict(
            source=str(image_path),
            conf=confidence,
            save=False,
            verbose=False,
        )
        result = results[0]
        annotated_path = annotated_dir / f"{image_path.stem}_yolo.jpg"
        label_path = labels_dir / f"{image_path.stem}.txt"

        result.save(filename=str(annotated_path))

        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                score = float(box.conf[0])
                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0]]
                detections.append((class_name, score, x1, y1, x2, y2))

        with label_path.open("w", encoding="utf-8") as file:
            for class_name, score, x1, y1, x2, y2 in detections:
                file.write(f"{class_name} {score:.6f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")

        rows.append(
            {
                "group": group_name,
                "image": str(image_path),
                "annotated_image": str(annotated_path),
                "detections": str(len(detections)),
                "top_detection": detections[0][0] if detections else "none",
                "top_confidence": f"{detections[0][1]:.6f}" if detections else "0.000000",
            }
        )

    return rows


def save_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    fieldnames = [
        "group",
        "image",
        "annotated_image",
        "detections",
        "top_detection",
        "top_confidence",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary(rows: list[dict[str, str]], output_path: Path) -> None:
    lines = [
        "YOLO Detection Summary",
        "",
        "This task runs the same YOLO model on clear, distorted, and restored images.",
        "For MRI images, pretrained COCO YOLO detections may not be medically meaningful.",
        "The value is comparing how image processing changes detector behavior.",
        "",
    ]

    for group_name in ["clear", "distortions", "restoration"]:
        group_rows = [row for row in rows if row["group"] == group_name]
        images_count = len(group_rows)
        detections_count = sum(int(row["detections"]) for row in group_rows)
        images_with_detection = sum(1 for row in group_rows if int(row["detections"]) > 0)
        lines.extend(
            [
                f"{group_name}:",
                f"  images: {images_count}",
                f"  total detections: {detections_count}",
                f"  images with detections: {images_with_detection}",
                "",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    YOLO = import_yolo()
    model = YOLO(args.model)

    groups = {
        "clear": collect_images(CLEAR_DIR, limit=args.max_images),
        "distortions": collect_images(DISTORTION_DIR),
        "restoration": collect_images(RESTORATION_DIR),
    }

    missing_groups = [name for name, images in groups.items() if not images]
    if missing_groups:
        raise FileNotFoundError(
            "No images found for: "
            + ", ".join(missing_groups)
            + ". Run preprocess.py, distortion_methods.py, and image_restoration.py first."
        )

    all_rows: list[dict[str, str]] = []
    for group_name, image_paths in groups.items():
        all_rows.extend(
            evaluate_group(
                model,
                group_name,
                image_paths,
                output_dir,
                args.confidence,
            )
        )

    save_csv(all_rows, output_dir / "yolo_predictions.csv")
    save_summary(all_rows, output_dir / "summary.txt")
    print("Saved YOLO results to:", output_dir.resolve())


if __name__ == "__main__":
    main()
