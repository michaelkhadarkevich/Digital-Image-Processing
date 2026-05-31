# Future Enhancements

Use this file as the project roadmap for future improvements.

## CNN Enhancements

- Improve classification accuracy with transfer learning.
- Add precision, recall, F1-score, and a richer confusion matrix report.
- Add class balancing for tumor and no-tumor images.
- Add stronger data augmentation.
- Add Grad-CAM heatmaps to explain model predictions.
- Compare CNN predictions on clear, distorted, restored, and super-resolution images.

## Super-Resolution Enhancements

- Add zoomed crop comparisons.
- Add more reconstruction metrics and charts.
- Compare nearest, bilinear, bicubic, Lanczos, and deep-learning methods.
- Add a deep-learning super-resolution model such as SRCNN or ESRGAN.
- Evaluate how super-resolution affects CNN predictions.

## YOLO Enhancements

- Train YOLO on MRI tumor bounding-box annotations.
- Add mAP, precision, recall, and IoU metrics.
- Compare YOLO detections on clear, distorted, restored, and super-resolution images.
- Add a custom dataset format guide for YOLO training.
- Save more visual detection comparison grids.

## Notes

For real medical object detection, YOLO needs MRI images with bounding-box labels.
The current pretrained YOLO task is for image-processing comparison only.
