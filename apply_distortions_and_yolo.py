#!/usr/bin/env python3
"""Apply distortions to MRI images and run YOLO detection."""

import argparse
import csv
import sys
from pathlib import Path
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


# ============= Distortion Methods =============
IMAGE_SIZE = (224, 224)

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


def perspective_warp(image: Image.Image) -> Image.Image:
    width, height = image.size
    source = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )
    target = np.array(
        [
            [20, 8],
            [width - 24, 24],
            [width - 6, height - 18],
            [10, height - 8],
        ],
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


def find_perspective_coefficients(source: np.ndarray, target: np.ndarray) -> list[float]:
    matrix = []
    for (x, y), (u, v) in zip(target, source):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    values = source.flatten()
    return np.linalg.solve(np.asarray(matrix), values).tolist()


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

    distorted = array[source_y, source_x]
    return Image.fromarray(distorted)


def pixelate(image: Image.Image, block_size: int = 12) -> Image.Image:
    small_size = (IMAGE_SIZE[0] // block_size, IMAGE_SIZE[1] // block_size)
    small = image.resize(small_size, Image.Resampling.BILINEAR)
    return small.resize(IMAGE_SIZE, Image.Resampling.NEAREST)


DISTORTION_METHODS = {
    "original": lambda img: img,
    "gaussian_noise": gaussian_noise,
    "salt_and_pepper": salt_and_pepper_noise,
    "gaussian_blur": blur,
    "brightness_contrast": brightness_contrast,
    "rotation": rotate,
    "perspective_warp": perspective_warp,
    "barrel_distortion": barrel_distortion,
    "pixelation": pixelate,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply distortions to MRI images and run YOLO detection."
    )
    parser.add_argument(
        "--dataset",
        default="C:/Users/renan/.cache/kagglehub/datasets/mateuszbuda/lgg-mri-segmentation/versions/2",
        help="Dataset directory path",
    )
    parser.add_argument(
        "--output",
        default="results/results_augmented",
        help="Output directory for distorted images and results",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=50,
        help="Max images to process (None = all)",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO model",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="YOLO confidence threshold",
    )
    return parser.parse_args()


def load_image(path: Path) -> Image.Image | None:
    """Load and convert image to RGB."""
    try:
        with Image.open(path) as img:
            return img.convert("RGB").resize(IMAGE_SIZE)
    except Exception:
        return None


def collect_images(directory: Path, limit: int | None = None) -> list[Path]:
    """Collect image files from directory, skip mask files."""
    valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    
    if not directory.exists():
        print(f"⚠️  Directory not found: {directory}")
        return []

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


def import_yolo():
    """Import YOLO with error handling."""
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Ultralytics YOLO is not installed. Install it with:\n"
            "pip install ultralytics\n"
        ) from exc
    return YOLO


def main():
    args = parse_args()
    
    print("=" * 80)
    print("🔍 Applying Distortions & Running YOLO Detection on MRI Dataset")
    print("=" * 80)
    print(f"📁 Dataset: {args.dataset}")
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

    # Collect images
    dataset_dir = Path(args.dataset)
    image_paths = collect_images(dataset_dir, args.max_images)
    if not image_paths:
        print(f"❌ No images found in {dataset_dir}")
        sys.exit(1)

    print(f"📊 Found {len(image_paths)} images")
    print(f"🎨 Will apply {len(DISTORTION_METHODS)} distortion methods")
    print(f"📸 Total: {len(image_paths) * len(DISTORTION_METHODS)} images to process\n")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    distorted_dir = output_dir / "distorted_images"
    distorted_dir.mkdir(parents=True, exist_ok=True)
    
    results_rows = []
    total_images = len(image_paths) * len(DISTORTION_METHODS)
    current = 0

    # Process each image with each distortion
    for img_idx, image_path in enumerate(image_paths, 1):
        original = load_image(image_path)
        if original is None:
            print(f"⚠️  Skipped (cannot load): {image_path.name}")
            continue

        print(f"\n[{img_idx}/{len(image_paths)}] {image_path.name}")
        
        for dist_idx, (dist_name, dist_func) in enumerate(DISTORTION_METHODS.items(), 1):
            current += 1
            progress = f"[{current}/{total_images}]"
            
            try:
                # Apply distortion
                distorted = dist_func(original)
                
                # Save distorted image
                base_name = image_path.stem
                distorted_filename = f"{base_name}_{dist_name}.png"
                distorted_path = distorted_dir / distorted_filename
                distorted.save(distorted_path)
                
                # Run YOLO
                results = model.predict(
                    source=str(distorted_path),
                    conf=args.confidence,
                    save=False,
                    verbose=False,
                )
                
                result = results[0]
                boxes = result.boxes
                detection_count = len(boxes) if boxes is not None else 0
                
                print(f"  {dist_name:20} {progress:15} ✅ {detection_count} detections")
                
                results_rows.append({
                    "image": base_name,
                    "distortion": dist_name,
                    "filename": distorted_filename,
                    "path": str(distorted_path),
                    "detections": detection_count,
                    "confidence_threshold": args.confidence,
                })
                
            except Exception as e:
                print(f"  {dist_name:20} {progress:15} ❌ Error: {str(e)[:40]}")

    # Save results to CSV
    if results_rows:
        csv_path = output_dir / "distorted_yolo_results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results_rows[0].keys())
            writer.writeheader()
            writer.writerows(results_rows)
        print(f"\n💾 Results saved to: {csv_path}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"Total original images: {len(image_paths)}")
    print(f"Distortion methods: {len(DISTORTION_METHODS)}")
    print(f"Total images processed: {len(results_rows)}")
    print(f"Total detections: {sum(r['detections'] for r in results_rows)}")
    
    if results_rows:
        avg_detections = sum(r['detections'] for r in results_rows) / len(results_rows)
        print(f"Average detections per image: {avg_detections:.2f}")
        
        # Stats by distortion
        print(f"\nDetections by distortion method:")
        from collections import defaultdict
        dist_stats = defaultdict(lambda: {"count": 0, "detections": 0})
        for row in results_rows:
            dist_stats[row['distortion']]['count'] += 1
            dist_stats[row['distortion']]['detections'] += row['detections']
        
        for dist_name in DISTORTION_METHODS.keys():
            stats = dist_stats[dist_name]
            if stats['count'] > 0:
                avg = stats['detections'] / stats['count']
                print(f"  {dist_name:20} {stats['detections']:4} total, {avg:.2f} avg")
    
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
