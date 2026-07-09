# YOLO Models Directory

This directory contains YOLO models used for object detection on MRI images.

## Model: yolov8n.pt

- **Name**: YOLOv8 Nano (Smallest & Fastest)
- **Size**: ~6.2 MB (compressed)
- **Dataset Trained On**: COCO (Common Objects in Context)
- **Classes**: 80 object classes
- **Speed**: Fast inference on CPU/GPU
- **Accuracy**: Lower than larger variants, but sufficient for real-time detection

### Download

The model is automatically downloaded on first use by Ultralytics:

```python
from ultralytics import YOLO
model = YOLO('models/yolo/yolov8n.pt')  # Auto-downloads if missing
```

Or manually download:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Usage

```python
from ultralytics import YOLO

# Load model
model = YOLO('models/yolo/yolov8n.pt')

# Predict on image
results = model.predict(
    source='path/to/image.jpg',
    conf=0.25,  # Confidence threshold
    verbose=False
)

# Access results
for result in results:
    boxes = result.boxes
    print(f"Detections: {len(boxes)}")
```

## Alternative Models

You can also use:
- `yolov8s.pt` - Small (22MB)
- `yolov8m.pt` - Medium (49MB)
- `yolov8l.pt` - Large (103MB)
- `yolo11n.pt` - YOLOv11 Nano (latest, ~5.3MB)

Simply update the model path in scripts or pass as argument:

```bash
python apply_distortions_and_yolo.py --model models/yolo/yolo11n.pt
```

## References

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [COCO Dataset](https://cocodataset.org/)
- [YOLO Documentation](https://docs.ultralytics.com/)
