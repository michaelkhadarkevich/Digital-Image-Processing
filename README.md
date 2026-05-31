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

## Notes

This project is for learning and experimentation. It is not a medical diagnostic
tool and should not be used for clinical decisions.
