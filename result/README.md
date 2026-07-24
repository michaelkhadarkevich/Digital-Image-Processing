# Results Directory

This directory contains all output results from the project experiments and pipelines.

## Structure

### `result/results task 3/yolo_distortion_results/`
- **YOLO Detection Results on Distorted & Original Images**
- `yolo_detections.csv` - YOLO predictions on 100 original MRI images
- `distorted_yolo_results.csv` - YOLO predictions on 450 distorted images (50 images × 9 distortion methods)
- `distorted_images/` - (450 PNG files) Distorted versions of MRI images
- Columns: image, distortion, filename, path, detections, confidence_threshold

### `results/results_segmentation/`
- **MRI Segmentation Results**
- Model predictions on segmentation tasks
- Ground truth comparisons

### `results/yolo_detection/`
- **YOLO Detection Analysis**
- `summary.txt` - Summary statistics
- `yolo_predictions.csv` - Detailed predictions
- Subdirectories: `clear/`, `distortions/`, `restoration/`

### `results/cnn_on_distortion_restoration/`
- **CNN Model Predictions**
- `cnn_predictions.csv` - CNN classification results
- `summary.txt` - Model performance metrics

### `results/distortions/`
- **Distortion Methods Applied**
- `methods.txt` - List of 9 distortion methods used
- 1. Original (no distortion)
- 2. Gaussian Noise (σ=24.0)
- 3. Salt & Pepper (4%)
- 4. Gaussian Blur (radius=3)
- 5. Brightness & Contrast (1.35× brightness, 1.55× contrast)
- 6. Rotation (18°)
- 7. Perspective Warp
- 8. Barrel Distortion
- 9. Pixelation (12×12 blocks)

### `results/restoration/`
- **Image Restoration Techniques**
- Results from restoration methods applied to distorted images
- `methods.txt` - List of restoration techniques

### `results/super_resolution/`
- **Super-Resolution Results**
- `metrics.csv` - PSNR, SSIM, and other quality metrics
- Subdirectories: `distortions/`, `restoration/`

## Key CSV Formats

### `distorted_yolo_results.csv`
```csv
image,distortion,filename,path,detections,confidence_threshold
TCGA_CS_4941_19960909_1,original,TCGA_CS_4941_19960909_1_original.png,result\results task 3\yolo_distortion_results\distorted_images\...,0,0.25
TCGA_CS_4941_19960909_1,gaussian_noise,TCGA_CS_4941_19960909_1_gaussian_noise.png,result\results task 3\yolo_distortion_results\distorted_images\...,1,0.25
```

## Statistics Summary

- **Total YOLO Runs**: 450 (100 original + 350 distorted)
- **Total Detections**: 289 across all runs
- **Average Detections per Image**: 0.64

### Best Performing Distortions (by detection count):
1. **Gaussian Noise**: 55 detections (1.10 avg per image)
2. **Salt & Pepper**: 48 detections (0.96 avg)
3. **Perspective Warp**: 40 detections (0.80 avg)
4. **Barrel Distortion**: 37 detections (0.74 avg)
5. **Original**: 35 detections (0.70 avg)

### Worst Performing Distortions:
- **Pixelation**: 0 detections (completely obscures detectable features)
- **Gaussian Blur**: 16 detections (0.32 avg)
- **Rotation**: 25 detections (0.50 avg)

## Data Analysis Scripts

- `analyze_results.py` - Aggregates statistics from CSV files
- Calculates totals, averages, and max detections by method

## Notes

- Images in `distorted_images/` are PNGs (lossy distortions applied)
- All CSV files use UTF-8 encoding
- Confidence threshold: 0.25 (YOLO default)
- MRI images resized to 224×224 pixels during processing
