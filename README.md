# Digital-Image-Processing

## Brain MRI Project

This project works with two brain MRI datasets and is organized into three
tasks:

```text
Task 1: CNN tumor classification
Task 2: Super-resolution
Task 3: YOLO detection
```

Datasets:

```text
Data 1: Kaggle brain MRI tumor/no-tumor classification dataset
        navoneel/brain-mri-images-for-brain-tumor-detection

Data 2: MRI tumor segmentation dataset with masks, converted to YOLO format
```

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

Run the Task 1-style SNR/level super-resolution experiment:

```bash
python "task\task 2 super resalution\super_resolution_snr.py"
```

Outputs are saved to:

```text
result\results task 2\super_resolution_snr\
  distortions\
  restoration\
  super_resolution_results\
  snr_super_resolution_metrics.csv
  gaussian_noise_snr_psnr_graph.png
  gaussian_noise_snr_ssim_graph.png
  gaussian_noise_snr_reconstruction_score_graph.png
  salt_and_pepper_density_psnr_graph.png
  salt_and_pepper_density_ssim_graph.png
  salt_and_pepper_density_reconstruction_score_graph.png
  gaussian_blur_sigma_psnr_graph.png
  gaussian_blur_sigma_ssim_graph.png
  gaussian_blur_sigma_reconstruction_score_graph.png
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
models\cnn_basic.keras
```

Results are saved to:

```text
result\results task 1\cnn_on_basic\
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

Augmentation note:

The basic CNN was improved by trying data augmentation. The reason for this
choice was that Data 1 is not very large, so adding augmented training images
could give the CNN more examples to learn from. After many runs, augmentation
did improve the model, but only by a small amount. The main conclusion was that
changing the position or rotation of the head does not help very much for this
dataset.

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
models\cnn_augmented.keras
```

Augmented results are saved to:

```text
result\results task 1\cnn_on_augmented\
  training_history.png
  confusion_matrix.png
  metrics.txt
  predictions.csv
```

Fine-tuned distortion CNN results should be saved to:

```text
result\results task 1\cnn_fine_tune_distortion\
```

Run the trained basic CNN on clean and distorted images only:

```bash
python "task\task 1 cnn\evaluate_distortion.py"
```

The distortion-only CNN outputs are saved to:

```text
result\results task 1\cnn_on_distortion\
  cnn_predictions_all_images.csv
  summary_all_images.txt
  confusion_percentages.csv
  total_true_percent_graph.png
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
  cnn_predictions_all_images.csv
  summary_all_images.txt
  confusion_percentages.csv
  total_true_percent_graph.png
  snr_total_true_percent_graph.png
  salt_and_pepper_total_true_percent_graph.png
  gaussian_blur_total_true_percent_graph.png
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

Note: pretrained YOLO detection models are trained on natural images, not MRI
scans. They are useful for an image-processing comparison, but they are not
tumor detectors.

## Task 3: YOLO Tumor Segmentation

YOLO can mark tumor regions if it is trained as a segmentation model. Data 2
uses the LGG MRI segmentation dataset masks and converts them into YOLO
segmentation labels.

Download Data 2:

```bash
python "data\Data 2\download_mri_data.py"
```

Convert the segmentation masks into YOLO segmentation format:

```bash
python "task\task 3 yolo\prepare_yolo_segmentation.py"
```

This creates:

```text
data\Data 2\yolo_segmentation\
  data.yaml
  images\train\
  images\val\
  images\test\
  labels\train\
  labels\val\
  labels\test\
```

Train YOLO segmentation:

```bash
python "task\task 3 yolo\train_yolo_segmentation.py"
```

The trained tumor segmentation model is copied to:

```text
models\yolo\yolo_tumor_seg.pt
```

Use the trained model to mark tumors:

```bash
python "task\task 3 yolo\run_yolo_segmentation.py"
```

Tumor segmentation outputs are saved to:

```text
result\results task 3\yolo_tumor_segmentation\
  tumor_segmentation_predictions.csv
  annotated\
  labels\
```

Run YOLO distortion-level experiments:

```bash
python "task\task 3 yolo\prepare_yolo_distorted_segmentation.py"
python "task\task 3 yolo\prepare_yolo_restored_segmentation.py"
python "task\task 3 yolo\validate_yolo_basic_on_distortion_levels.py"
python "task\task 3 yolo\validate_yolo_basic_on_restored_distortion_levels.py"
python "task\task 3 yolo\plot_yolo_snr_results.py"
```

YOLO distortion-level results are saved to:

```text
result\results task 3\yolo_snr_results\
  yolo_distortion_level_metrics.csv
  yolo_basic_distortion_level_metrics.csv
  yolo_basic_restored_distortion_level_metrics.csv
  yolo_gaussian_noise_snr_map50_graph.png
  yolo_gaussian_noise_snr_map50_95_graph.png
  yolo_gaussian_blur_sigma_map50_graph.png
  yolo_gaussian_blur_sigma_map50_95_graph.png
  yolo_brightness_contrast_level_map50_graph.png
  yolo_brightness_contrast_level_map50_95_graph.png
```

## Documentation Of Choices, Processing, And Results

### Project Choices

| Area | Choice |
| --- | --- |
| Data 1 | Brain MRI tumor/no-tumor image classification data. |
| Data 2 | MRI segmentation data converted into YOLO segmentation format with images, masks, and YOLO labels. |
| Task 1 | CNN binary classifier for tumor vs no tumor. |
| Task 2 | x2 downsample then reconstruct using nearest, bilinear, bicubic, lanczos, and sharpened_lanczos. |
| Task 3 | YOLO segmentation for marking tumor regions. |
| Distortion levels | Gaussian noise SNR, salt-and-pepper density, gaussian blur sigma, and YOLO brightness/contrast levels. |
| Restoration | Noise uses smoothing/sharpening, salt-and-pepper uses median filtering, blur uses unsharp masking, brightness/contrast uses intensity normalization. |

### Input And Output Processing Steps

1. Data 1 images are downloaded and split into `train`, `val`, and `test`.
2. CNN inputs are resized and classified as tumor or no tumor.
3. Distorted images are generated from clean MRI images.
4. Restored images are generated from distorted images.
5. Super-resolution downsamples each input by x2, then reconstructs it back to 224x224.
6. Data 2 masks are converted to YOLO segmentation labels.
7. YOLO outputs annotated images with predicted tumor masks.

Important before/after and annotation outputs:

```text
data\Data 1 with distortion\distortions\distortion_grid.png
data\Data 1 with distortion\restoration\restoration_grid.png
result\results task 2\super_resolution\Clean\super_resolution_grid.png
result\results task 2\super_resolution_snr\super_resolution_results\
result\results task 3\yolo_tumor_segmentation\annotated\
```

Example distortion and restoration images:

![Distortion grid](data/Data%201%20with%20distortion/distortions/distortion_grid.png)

![Restoration grid](data/Data%201%20with%20distortion/restoration/restoration_grid.png)

### Task 1 CNN Measurements

CNN correctness is measured with four percentages:

```text
TT: tumor image predicted as tumor
TF: tumor image predicted as no tumor
FT: no-tumor image predicted as tumor
FF: no-tumor image predicted as no tumor
```

Selected results from `result\results task 1\cnn_on_distortion_restoration\confusion_percentages.csv`:

| Source | TT % | TF % | FT % | FF % | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean original | 52.63 | 7.89 | 13.16 | 26.32 | 38 |
| gaussian_noise distorted | 57.89 | 2.63 | 26.32 | 13.16 | 38 |
| gaussian_noise restored | 60.53 | 0.00 | 31.58 | 7.89 | 38 |
| salt_and_pepper distorted | 5.26 | 55.26 | 0.00 | 39.47 | 38 |
| salt_and_pepper restored | 57.89 | 2.63 | 23.68 | 15.79 | 38 |
| gaussian_blur distorted | 60.53 | 0.00 | 39.47 | 0.00 | 38 |
| gaussian_blur restored | 60.53 | 0.00 | 39.47 | 0.00 | 38 |

CNN visualization files:

```text
result\results task 1\cnn_on_basic\training_history.png
result\results task 1\cnn_on_basic\confusion_matrix.png
result\results task 1\cnn_on_distortion\total_true_percent_graph.png
result\results task 1\cnn_on_distortion_restoration\total_true_percent_graph.png
result\results task 1\cnn_fine_tune_distortion\total_true_percent_graph.png
result\results task 1\cnn_compare_snr_distortion_restoration_fine_tune_total_true_percent_graph.png
result\results task 1\cnn_compare_salt_and_pepper_distortion_restoration_fine_tune_total_true_percent_graph.png
result\results task 1\cnn_compare_gaussian_blur_distortion_restoration_fine_tune_total_true_percent_graph.png
```

CNN result visualizations:

![CNN basic confusion matrix](result/results%20task%201/cnn_on_basic/confusion_matrix.png)

![CNN SNR comparison](result/results%20task%201/cnn_compare_snr_distortion_restoration_fine_tune_total_true_percent_graph.png)

![CNN salt and pepper comparison](result/results%20task%201/cnn_compare_salt_and_pepper_distortion_restoration_fine_tune_total_true_percent_graph.png)

![CNN gaussian blur comparison](result/results%20task%201/cnn_compare_gaussian_blur_distortion_restoration_fine_tune_total_true_percent_graph.png)

### Task 2 Super-Resolution Measurements

Task 2 metrics:

```text
PSNR: higher is better.
SSIM: closer to 1 is better.
Reconstruction score: closer to 100 is better.
MSE, RMSE, MAE: lower is better.
```

In `super_resolution_snr`, distorted images are compared to the distorted input, and restored images are compared to the restored input. The table below uses the `sharpened_lanczos` reconstruction method.

| Distortion | Level | Image type | PSNR | SSIM | Reconstruction score |
| --- | --- | --- | ---: | ---: | ---: |
| gaussian_noise_snr | 5db | distorted | 17.25 | 0.930 | 89.83 |
| gaussian_noise_snr | 5db | restored | 27.35 | 0.986 | 97.26 |
| gaussian_noise_snr | 30db | distorted | 26.74 | 0.987 | 97.31 |
| gaussian_noise_snr | 30db | restored | 28.14 | 0.990 | 97.88 |
| salt_and_pepper_density | 0.01 | distorted | 22.68 | 0.966 | 96.53 |
| salt_and_pepper_density | 0.01 | restored | 27.79 | 0.989 | 97.80 |
| salt_and_pepper_density | 0.10 | distorted | 14.95 | 0.820 | 89.87 |
| salt_and_pepper_density | 0.10 | restored | 27.31 | 0.988 | 97.71 |
| gaussian_blur_sigma | 0.5 | distorted | 27.50 | 0.988 | 97.62 |
| gaussian_blur_sigma | 3.0 | distorted | 43.95 | 1.000 | 99.74 |

Task 2 visualization files:

```text
result\results task 2\super_resolution\detailed_psnr_graph.png
result\results task 2\super_resolution\detailed_ssim_graph.png
result\results task 2\super_resolution\detailed_reconstruction_score_graph.png
result\results task 2\super_resolution\vs_original_psnr_graph.png
result\results task 2\super_resolution\vs_original_ssim_graph.png
result\results task 2\super_resolution\vs_original_reconstruction_score_graph.png
result\results task 2\super_resolution_snr\gaussian_noise_snr_psnr_graph.png
result\results task 2\super_resolution_snr\salt_and_pepper_density_psnr_graph.png
result\results task 2\super_resolution_snr\gaussian_blur_sigma_psnr_graph.png
```

Super-resolution visualizations:

![Clean super-resolution grid](result/results%20task%202/super_resolution/Clean/super_resolution_grid.png)

![Super-resolution gaussian noise PSNR](result/results%20task%202/super_resolution_snr/gaussian_noise_snr_psnr_graph.png)

![Super-resolution gaussian noise SSIM](result/results%20task%202/super_resolution_snr/gaussian_noise_snr_ssim_graph.png)

![Super-resolution salt and pepper PSNR](result/results%20task%202/super_resolution_snr/salt_and_pepper_density_psnr_graph.png)

![Super-resolution gaussian blur reconstruction score](result/results%20task%202/super_resolution_snr/gaussian_blur_sigma_reconstruction_score_graph.png)

### Task 3 YOLO Measurements

YOLO segmentation is measured with mask mAP:

```text
mAP50: mask average precision at IoU 0.50.
mAP50-95: average mask precision across IoU thresholds 0.50 to 0.95.
```

Selected results from `result\results task 3\yolo_snr_results\yolo_distortion_level_metrics.csv`:

| Distortion | Level | Basic mAP50 | Basic restored mAP50 | Fine-tuned mAP50 | Basic mAP50-95 | Basic restored mAP50-95 | Fine-tuned mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gaussian_noise_snr | 5db | 0.696 | 0.774 | 0.782 | 0.393 | 0.442 | 0.443 |
| gaussian_noise_snr | 15db | 0.828 | 0.829 | 0.839 | 0.490 | 0.497 | 0.497 |
| gaussian_noise_snr | 30db | 0.847 | 0.838 | 0.849 | 0.522 | 0.505 | 0.519 |
| gaussian_blur_sigma | 0.5 | 0.839 | 0.818 | 0.859 | 0.518 | 0.494 | 0.518 |
| gaussian_blur_sigma | 3.0 | 0.787 | 0.794 | 0.831 | 0.443 | 0.450 | 0.481 |
| brightness_contrast | B1.15 C1.25 | 0.843 | 0.837 | 0.857 | 0.517 | 0.505 | 0.513 |
| brightness_contrast | B1.55 C1.80 | 0.815 | 0.813 | 0.850 | 0.468 | 0.474 | 0.497 |

YOLO visualization files:

```text
result\results task 3\yolo_basic\train\confusion_matrix.png
result\results task 3\yolo_basic\train\results.png
result\results task 3\yolo_snr_results\yolo_gaussian_noise_snr_map50_graph.png
result\results task 3\yolo_snr_results\yolo_gaussian_noise_snr_map50_95_graph.png
result\results task 3\yolo_snr_results\yolo_gaussian_blur_sigma_map50_graph.png
result\results task 3\yolo_snr_results\yolo_gaussian_blur_sigma_map50_95_graph.png
result\results task 3\yolo_snr_results\yolo_brightness_contrast_level_map50_graph.png
result\results task 3\yolo_snr_results\yolo_brightness_contrast_level_map50_95_graph.png
```

YOLO result visualizations:

![YOLO basic training results](result/results%20task%203/yolo_basic/train/results.png)

![YOLO gaussian noise mAP50](result/results%20task%203/yolo_snr_results/yolo_gaussian_noise_snr_map50_graph.png)

![YOLO gaussian noise mAP50-95](result/results%20task%203/yolo_snr_results/yolo_gaussian_noise_snr_map50_95_graph.png)

![YOLO gaussian blur mAP50](result/results%20task%203/yolo_snr_results/yolo_gaussian_blur_sigma_map50_graph.png)

![YOLO brightness contrast mAP50-95](result/results%20task%203/yolo_snr_results/yolo_brightness_contrast_level_map50_95_graph.png)

### Measurements Summary

| Task | Measurements | Main plots |
| --- | --- | --- |
| Task 1 CNN | TT, TF, FT, FF, total true percent, confusion matrix | bar plots comparing clean, distorted, restored, and fine-tuned CNN results |
| Task 2 Super-resolution | MSE, RMSE, MAE, PSNR, SSIM, reconstruction score | PSNR, SSIM, reconstruction-score heatmaps and bar plots |
| Task 3 YOLO | mask mAP50 and mask mAP50-95 | mAP50 and mAP50-95 bar plots comparing basic, restored, and fine-tuned YOLO |

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
