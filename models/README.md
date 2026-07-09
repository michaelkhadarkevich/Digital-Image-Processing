# Models Directory

This directory contains all trained and pre-trained models used in the project.

## Structure

### `models/yolo/`
- **yolov8n.pt** - Pre-trained YOLOv8 Nano model (COCO dataset)
  - Used for: Object detection on MRI images
  - Size: ~6.2 MB
  - Source: Ultralytics (automatically downloaded)

### `models/cnn/`
- **Train CNN models** (if applicable)
- Stored after training with timestamps
- Example: `cnn_mri_segmentation_20240709.h5`

## Usage

```python
from ultralytics import YOLO

# Load model
model = YOLO('models/yolo/yolov8n.pt')

# Run inference
results = model.predict(source='image.jpg', conf=0.25)
```

## Model Download

If `models/yolo/yolov8n.pt` is missing, it will be automatically downloaded when you run:

```bash
python apply_distortions_and_yolo.py
python run_yolo_on_mri.py
```

The Ultralytics library automatically downloads the model to this directory on first use.
