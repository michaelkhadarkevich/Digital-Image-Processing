# 📁 Digital Image Processing - Project Structure

## Overview
All project files are organized into three main directories for clarity and maintainability.

## Directory Structure

```
Digital-Image-Processing/
│
├── 📦 models/
│   ├── README.md
│   ├── yolo/
│   │   ├── README.md
│   │   └── yolov8n.pt (6.2 MB pre-trained model)
│   └── cnn/
│       └── (CNN models stored here after training)
│
├── 🎓 training/
│   ├── README.md
│   ├── logs/
│   │   └── (Training logs, metrics, CSV files)
│   ├── checkpoints/
│   │   └── (Model checkpoints during training - intermediate weights)
│   └── history/
│       ├── training_history.png (Accuracy/Loss curves)
│       └── confusion_matrix.png (Classification results)
│
├── 📊 results/
│   ├── README.md
│   ├── results_augmented/
│   │   ├── yolo_detections.csv (100 original images)
│   │   ├── distorted_yolo_results.csv (450 distorted images)
│   │   └── distorted_images/ (PNG files for each distortion)
│   ├── results_segmentation/
│   ├── yolo_detection/
│   ├── cnn_on_distortion_restoration/
│   ├── distortions/
│   ├── restoration/
│   └── super_resolution/
│
├── 📚 src/
│   ├── train_cnn.py
│   ├── train_cnn_augmented.py
│   ├── yolo_detection.py
│   ├── distortion_methods.py
│   ├── image_restoration.py
│   ├── super_resolution.py
│   └── ... (other utilities)
│
├── 🔧 Main Scripts
│   ├── apply_distortions_and_yolo.py (Core pipeline)
│   ├── run_yolo_on_mri.py (YOLO on originals)
│   ├── download_mri_data.py (Kaggle download)
│   └── analyze_results.py (Statistics)
│
├── 📋 Configuration
│   ├── requirements.txt
│   ├── README.md
│   ├── ENHANCEMENTS.md
│   ├── PROJECT_STRUCTURE.md (this file)
│   └── .gitignore
│
└── 💾 data/
    └── (MRI images from Kaggle - stored locally, not in Git)
```

## What Goes Where

### 🏆 Models Directory (`models/`)
**Store:** Pre-trained models and trained models
- **models/yolo/** - YOLO models (yolov8n.pt, yolo11n.pt, etc.)
- **models/cnn/** - Trained CNN models (after running train_cnn.py)

### 📖 Training Directory (`training/`)
**Store:** Everything related to the training process
- **logs/** - Training metrics, CSV files with epoch stats
- **checkpoints/** - Intermediate model weights saved during training
- **history/** - Visualizations (loss/accuracy curves, confusion matrix)

### 📈 Results Directory (`results/`)
**Store:** All output predictions and evaluations
- **results_augmented/** - YOLO predictions on original and distorted images
- **results_segmentation/** - Segmentation model outputs
- **yolo_detection/** - YOLO detection analysis
- **cnn_on_distortion_restoration/** - CNN predictions
- **super_resolution/** - Super-resolution metrics and results
- **distortions/** - Information about distortion methods used
- **restoration/** - Image restoration technique results

## File Organization Rules

| File Type | Location |
|-----------|----------|
| Pre-trained model (`.pt`, `.h5`) | `models/` |
| Trained model checkpoint | `training/checkpoints/` |
| Training logs (`.csv`, `.json`) | `training/logs/` |
| Training visualizations (`.png`) | `training/history/` |
| Predictions (`.csv`) | `results/[method]/` |
| Result images (`.png`) | `results/[method]/` |
| Source code | `src/` or root directory |
| Raw dataset | `data/` |

## Key Statistics

- **Total Images Processed**: 450 (50 original × 9 distortion methods)
- **Total YOLO Detections**: 289
- **Model Size**: yolov8n.pt = 6.2 MB
- **Training History**: stored in `training/history/`
- **Results CSV**: 450 rows with detection data

## Usage Examples

```bash
# Download dataset
python download_mri_data.py

# Run YOLO on original images
python run_yolo_on_mri.py --dataset "path/to/data" --max-images 100

# Apply distortions and run YOLO
python apply_distortions_and_yolo.py --max-images 50 --confidence 0.25

# Analyze results
python analyze_results.py
```

## Distortion Methods (9 Total)

1. **Original** - No distortion
2. **Gaussian Noise** - σ=24.0
3. **Salt & Pepper** - 4% noise
4. **Gaussian Blur** - radius=3
5. **Brightness & Contrast** - 1.35× brightness, 1.55× contrast
6. **Rotation** - 18°
7. **Perspective Warp** - 4-point transform
8. **Barrel Distortion** - Radial lens effect
9. **Pixelation** - 12×12 blocks

## Git Ignore Rules

The following are NOT committed to GitHub:
- `data/` - MRI dataset (too large, cached locally)
- `models/cnn/` - Generated CNN models
- `training/checkpoints/` - Training checkpoints
- `training/logs/` - Log files
- `results/**/distorted_images/` - Generated images
- `*.pt` files in other locations (but models/yolo/yolov8n.pt is tracked for reference)

## Last Updated
- **Date**: July 9, 2026
- **Reorganization**: Separated models, training, and results into dedicated directories
- **Commit**: 31cc23d
