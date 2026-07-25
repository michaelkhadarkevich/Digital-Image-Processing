# Project Structure

The project is organized into three main folders:

```text
Brain_Mri/
|-- data/
|   |-- Data 1/
|   |   |-- train/
|   |   |-- val/
|   |   |-- test/
|   |   |-- download_dataset.py
|   |   `-- preprocess.py
|   |-- Data 1 augmented/
|   |-- Data 1 with distortion/
|   |-- Data 2/
|   |   |-- download_mri_data.py
|   |   `-- mri_segmentation/
|   `-- Data 2 with distortion/
|-- task/
|   |-- task 1 cnn/
|   |-- augmentation/
|   |-- task 2 super resalution/
|   |-- task 3 yolo/
|   |-- distortion methods/
|   `-- restoration/
|-- result/
|   |-- results task 1/
|   |-- results task 2/
|   `-- results task 3/
|-- models/
|-- src/
`-- requirements.txt
```

`src/` now only contains a README. The runnable code is under `data/` and `task/`.

## Main Commands

```bash
python "data\Data 1\preprocess.py" --input "PASTE_DATASET_PATH_HERE"
python "task\task 1 cnn\train_cnn.py"
python "task\augmentation\augment_train_rotate.py"
python "task\distortion methods\distortion_methods.py"
python "task\restoration\image_restoration.py"
python "task\task 2 super resalution\super_resolution.py"
python "task\task 3 yolo\yolo_detection.py"
```

## Folder Meaning

- `data`: all datasets and distorted data.
- `task`: all runnable task code.
- `result`: all generated result folders.
- `data\Data 1`: normal preprocessed MRI dataset.
- `data\Data 1 augmented`: augmented CNN dataset.
- `data\Data 1 with distortion`: distortion/restoration images made from Data 1.
- `data\Data 2`: YOLO MRI segmentation dataset.
- `data\Data 2 with distortion`: reserved for distorted YOLO data.
- `task\task 1 cnn`: CNN training, prediction, and CNN evaluation scripts.
- `task\augmentation`: augmentation scripts.
- `task\task 2 super resalution`: super-resolution scripts.
- `task\task 3 yolo`: YOLO detection and YOLO analysis scripts.
- `task\distortion methods`: distortion scripts.
- `task\restoration`: restoration scripts.
- `result\results task 1`: CNN outputs grouped as `cnn_on_basic`, `cnn_on_augmented`, `cnn_on_distortion`, `cnn_on_distortion_restoration`, and `cnn_fine_tune_distortion`.
- `result\results task 2`: super-resolution outputs.
- `result\results task 3`: YOLO outputs.
