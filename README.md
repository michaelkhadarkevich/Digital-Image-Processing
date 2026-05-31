# Digital-Image-Processing

## Brain MRI Tumor Detection

Python project for downloading the Kaggle brain MRI dataset, preprocessing images,
training a CNN binary classifier, and predicting whether a new MRI image shows a tumor.

Dataset: `navoneel/brain-mri-images-for-brain-tumor-detection`

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Download Dataset

```bash
python src\download_dataset.py
```

This downloads the dataset with KaggleHub and prints the local dataset path.

## 2. Preprocess Images

Use the dataset path printed by the download step:

```bash
python src\preprocess.py --input "PASTE_DATASET_PATH_HERE"
```

Processed files are saved into:

```text
data\processed\
  train\
  val\
  test\
```

## 3. Train CNN

```bash
python src\train.py
```

The trained model is saved to:

```text
models\brain_mri_classifier.keras
```

Training also saves result files to:

```text
results\
  training_history.png
  confusion_matrix.png
  metrics.txt
  predictions.csv
```

## 4. Predict One Image

```bash
python src\predict.py --image "path\to\image.jpg"
```

## 5. Distortion Methods

Apply common digital image distortion methods to one image:

```bash
python src\distortion_methods.py --image "path\to\image.jpg"
```

If `--image` is not provided, the script uses one image from `data\processed\test`.

Distortion outputs are saved to:

```text
results\distortions\
  distortion_grid.png
  methods.txt
  gaussian_noise.png
  salt_and_pepper.png
  gaussian_blur.png
  brightness_contrast.png
  rotation.png
  perspective_warp.png
  barrel_distortion.png
  pixelation.png
```

## 6. Image Restoration

Apply restoration techniques to the distortion outputs:

```bash
python src\image_restoration.py
```

Run `distortion_methods.py` first, because this script uses files from
`results\distortions`.

Restoration outputs are saved to:

```text
results\restoration\
  restoration_grid.png
  methods.txt
  gaussian_noise_restored.png
  salt_and_pepper_restored.png
  gaussian_blur_restored.png
  brightness_contrast_restored.png
  rotation_restored.png
  perspective_warp_restored.png
  barrel_distortion_restored.png
  pixelation_restored.png
```

## 7. CNN on Distorted and Restored Images

Run the trained CNN on the distortion images and the restored images:

```bash
python src\evaluate_distortion_restoration.py
```

Run these first:

```bash
python src\distortion_methods.py
python src\image_restoration.py
```

The CNN comparison outputs are saved to:

```text
results\cnn_on_distortion_restoration\
  cnn_predictions.csv
  summary.txt
  tumor_probability_chart.png
```

## 8. Super-Resolution

Start by downsampling an image by x2, then reconstruct it back to the original
size with interpolation-based super-resolution techniques:

```bash
python src\super_resolution.py
```

Or choose a specific image:

```bash
python src\super_resolution.py --image "path\to\image.jpg"
```

Super-resolution outputs are saved to:

```text
results\super_resolution\
  original.png
  downsampled_x2.png
  downsampled_x2_preview.png
  nearest.png
  bilinear.png
  bicubic.png
  lanczos.png
  sharpened_lanczos.png
  super_resolution_grid.png
  metrics.csv
  methods.txt
```

To also run super-resolution on all distortion and restoration images:

```bash
python src\super_resolution.py --include-derived
```

Additional outputs are saved to:

```text
results\super_resolution\distortions\
results\super_resolution\restoration\
```

## Notes

This project is for learning and experimentation. It is not a medical diagnostic
tool and should not be used for clinical decisions.
