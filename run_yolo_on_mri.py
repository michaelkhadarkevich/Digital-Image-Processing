#!/usr/bin/env python3
"""Run YOLO detection on MRI Segmentation dataset."""

import argparse
import csv
import sys
from pathlib import Path
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO on MRI Segmentation dataset."
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO model. Example: yolo11n.pt, yolov8n.pt",
    )
    parser.add_argument(
        "--dataset",
        default="data/mri_segmentation",
        help="Dataset directory path",
    )
    parser.add_argument(
        "--output",
        default="results/results_augmented",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Max images to process (None = all)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="YOLO confidence threshold",
    )
    return parser.parse_args()


def import_yolo():
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Ultralytics YOLO is not installed. Install it with:\n"
            "pip install ultralytics\n"
        ) from exc
    return YOLO


def collect_images(directory: Path, limit: int | None = None) -> list[Path]:
    """Collect image files from directory."""
    valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    
    if not directory.exists():
        print(f"⚠️  Directory not found: {directory}")
        return []

    # Skip mask files - they are grayscale and will cause YOLO errors
    images = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() 
        and path.suffix.lower() in valid_suffixes
        and "_mask" not in path.name.lower()
    )
    
    if limit is not None:
        images = images[:limit]
    
    return images


def evaluate_dataset(
    model,
    dataset_dir: Path,
    output_dir: Path,
    confidence: float,
    max_images: int | None = None,
) -> list[dict]:
    """Run YOLO on dataset and save results."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = output_dir / "annotated"
    labels_dir = output_dir / "labels"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Collect images
    image_paths = collect_images(dataset_dir, max_images)
    if not image_paths:
        print(f"❌ No images found in {dataset_dir}")
        return []

    print(f"📊 Found {len(image_paths)} images")
    
    rows = []
    for idx, image_path in enumerate(image_paths, 1):
        print(f"[{idx}/{len(image_paths)}] Processing: {image_path.name}...", end=" ")
        
        try:
            results = model.predict(
                source=str(image_path),
                conf=confidence,
                save=False,
                verbose=False,
            )

            result = results[0]
            boxes = result.boxes
            
            # Save annotated image
            ann_path = annotated_dir / f"{image_path.stem}_annotated.png"
            if result.plot() is not None:
                from PIL import Image
                Image.fromarray(result.plot()).save(ann_path)
            
            # Save YOLO labels (COCO format)
            label_path = labels_dir / f"{image_path.stem}.txt"
            if boxes is not None and len(boxes) > 0:
                with open(label_path, "w") as f:
                    for box in boxes:
                        # YOLO format: class x_center y_center width height (normalized)
                        cls_id = int(box.cls[0].item())
                        coords = box.xywhn[0]  # normalized coordinates
                        f.write(f"{cls_id} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}\n")
            else:
                label_path.touch()  # Empty file
            
            detection_count = len(boxes) if boxes is not None else 0
            rows.append({
                "image_path": str(image_path),
                "filename": image_path.name,
                "detections": str(detection_count),
                "confidence_threshold": str(confidence),
            })
            
            print(f"✅ {detection_count} detections")
            
        except Exception as e:
            print(f"❌ Skipped (error: {str(e)[:50]}...)")
            # Skip rows with errors, just log them

    return rows


def main():
    args = parse_args()
    
    print("=" * 70)
    print("🔍 YOLO Detection on MRI Segmentation Dataset")
    print("=" * 70)
    print(f"📁 Dataset: {args.dataset}")
    print(f"🤖 Model: {args.model}")
    print(f"📤 Output: {args.output}")
    print(f"🎯 Confidence: {args.confidence}")
    print()

    # Import YOLO
    try:
        YOLO = import_yolo()
    except ModuleNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)

    # Load model
    print(f"⏳ Loading YOLO model: {args.model}...")
    try:
        model = YOLO(args.model)
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    # Run detection
    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output)
    
    print(f"\n⏳ Running YOLO detection...")
    rows = evaluate_dataset(
        model,
        dataset_dir,
        output_dir,
        args.confidence,
        args.max_images,
    )

    # Save results to CSV
    if rows:
        csv_path = output_dir / "yolo_detections.csv"
        # Get all unique fieldnames from all rows
        all_fieldnames = set()
        for row in rows:
            all_fieldnames.update(row.keys())
        fieldnames = sorted(list(all_fieldnames))
        
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n💾 Results saved to: {csv_path}")
    
    # Summary
    total = len(rows)
    print(f"\n📊 Summary:")
    print(f"   Total images processed: {total}")
    print(f"   Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
