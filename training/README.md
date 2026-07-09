# Training Directory

This directory contains all training-related files, logs, and checkpoints.

## Structure

### `training/logs/`
- **Training logs and metrics**
- CSV files with epoch-by-epoch statistics
- Terminal output from training runs
- Named by date/time: `train_20240709_143022.log`

### `training/checkpoints/`
- **Model checkpoints saved during training**
- Intermediate model weights at each epoch
- Used for resuming training or early stopping
- Formats: `.h5`, `.keras`, `.pth`

### `training/history/`
- **Training history visualizations**
- `training_history.png` - Accuracy/Loss curves
- `confusion_matrix.png` - Classification confusion matrix
- Other diagnostic plots

## Files in this Directory

- **training_history.png** - Plot of training and validation loss/accuracy over epochs
- **confusion_matrix.png** - Confusion matrix for classification results

## Usage

```python
import json

# Load training history
with open('training/logs/train_latest.json', 'r') as f:
    history = json.load(f)

print(f"Final accuracy: {history['accuracy'][-1]}")
```

## Training Scripts

Training is orchestrated by:
- `src/train_cnn.py` - Train CNN on original images
- `src/train_cnn_augmented.py` - Train CNN on augmented/distorted images
- Logs are automatically saved to this directory
