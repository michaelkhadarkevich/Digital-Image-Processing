# Digital-Image-Processing

## Brain MRI Project

This project works with the Kaggle brain MRI dataset and is organized into
three tasks:

```text
Task 1: CNN tumor classification
Task 2: Super-resolution
Task 3: YOLO detection
```

Dataset: `navoneel/brain-mri-images-for-brain-tumor-detection`

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Shared Data Preparation

Download the dataset:

```bash
python "data\Data 1\download_dataset.py"
```

This downloads the dataset with KaggleHub and prints the local dataset path.

Preprocess the images:

```bash
python "data\Data 1\preprocess.py" --input "PASTE_DATASET_PATH_HERE"
```

Processed files are saved into:

```text
data\Data 1\
  train\
  val\
  test\
```

## Task 2: Super-Resolution

Task 2 downsamples an MRI image by x2 and reconstructs it back to the original
size using interpolation-based super-resolution techniques.

Run on the default image:

```bash
python "task\task 2 super resalution\super_resolution.py"
```

Or choose a specific image:

```bash
python "task\task 2 super resalution\super_resolution.py" --image "path\to\image.jpg"
```

Outputs are saved to:

```text
result\results task 2\super_resolution\
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

To also run super-resolution on distortion and restoration images:

```bash
python "task\task 2 super resalution\super_resolution.py" --include-derived
```

Additional outputs are saved to:

```text
result\results task 2\super_resolution\distortions\
result\results task 2\super_resolution\restoration\
```

Super-resolution success is measured without the CNN:

```text
MSE / RMSE / MAE: lower is better
PSNR: higher is better
SSIM: closer to 1 is better
Reconstruction score: closer to 100 is better
```

## Shared Distortion and Restoration

Distortion and restoration are shared experiments. They can be used by all
tasks:

```text
Task 2 can run super-resolution on distorted/restored images.
Task 1 can run the CNN on distorted/restored images.
Task 3 can run YOLO on distorted/restored images.
```

Apply common digital image distortion methods to one image:

```bash
python "task\distortion methods\distortion_methods.py" --image "path\to\image.jpg"
```

If `--image` is not provided, the script uses one image from
`data\Data 1\test`.

Distortion outputs are saved to:

```text
data\Data 1 with distortion\distortions\
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

Apply restoration techniques to the distortion outputs:

```bash
python "task\restoration\image_restoration.py"
```

Run `distortion_methods.py` first, because this script uses files from
`data\Data 1 with distortion\distortions`.

Restoration outputs are saved to:

```text
data\Data 1 with distortion\restoration\
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

## Task 1: CNN Tumor Classification

Task 1 trains a CNN binary classifier that predicts whether an MRI image has a
tumor.

Train the normal CNN:

```bash
python "task\task 1 cnn\train_cnn.py"
```

The trained model is saved to:

```text
models\brain_mri_classifier.keras
```

Results are saved to:

```text
result\results task 1\
  training_history.png
  confusion_matrix.png
  metrics.txt
  predictions.csv
```

Create a larger training set with rotated copies:

```bash
python "task\augmentation\augment_train_rotate.py"
```

This writes:

```text
data\Data 1 augmented\
```

Train the CNN on the augmented dataset:

```bash
python "task\task 1 cnn\train_cnn_augmented.py"
```

## data\Data 2: YOLO Data

`data\Data 2` is for the YOLO MRI segmentation dataset.

Download the YOLO dataset:

```bash
python "data\Data 2\download_mri_data.py"
```

The YOLO dataset is saved under:

```text
data\Data 2\mri_segmentation\
```

The augmented model is saved to:

```text
models\brain_mri_classifier_augmented.keras
```

Augmented results are saved to:

```text
result\results task 1\augmented\
  training_history.png
  confusion_matrix.png
  metrics.txt
  predictions.csv
```

Predict one image:

```bash
python "task\task 1 cnn\predict.py" --image "path\to\image.jpg"
```

Run the trained CNN on shared distortion and restoration images:

```bash
python "task\task 1 cnn\evaluate_distortion_restoration.py"
```

Run these first:

```bash
python "task\distortion methods\distortion_methods.py"
python "task\restoration\image_restoration.py"
```

The CNN comparison outputs are saved to:

```text
result\results task 1\cnn_on_distortion_restoration\
  cnn_predictions.csv
  summary.txt
  tumor_probability_chart.png
```

## Task 3: YOLO Detection

Task 3 runs YOLO on clear images, shared distortion images, and shared
restoration images.

Run YOLO:

```bash
python "task\task 3 yolo\yolo_detection.py"
```

This uses `yolo11n.pt` by default. You can also pass a custom YOLO model:

```bash
python "task\task 3 yolo\yolo_detection.py" --model "path\to\custom_model.pt"
```

YOLO outputs are saved to:

```text
result\results task 3\yolo_detection\
  summary.txt
  yolo_predictions.csv
  clear\annotated\
  distortions\annotated\
  restoration\annotated\
```

Note: pretrained YOLO models are trained on natural images, not MRI scans.
For medical tumor localization, train YOLO on MRI bounding-box annotations.

## Future Enhancements

Planned improvements are tracked in:

```text
ENHANCEMENTS.md
```

GitHub issue templates are included for:

```text
CNN enhancement
Super-resolution enhancement
YOLO enhancement
```

## Notes

This project is for learning and experimentation. It is not a medical diagnostic
tool and should not be used for clinical decisions.
