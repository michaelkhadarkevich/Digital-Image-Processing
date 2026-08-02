"""Download LGG MRI Segmentation dataset from Kaggle."""

import sys
import os
from pathlib import Path

os.chdir(Path(__file__).parent)

try:
    import kagglehub
    
    data_dir = Path("mri_segmentation")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading dataset: mateuszbuda/lgg-mri-segmentation...")
    dataset_path = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")
    print(f"Dataset successfully saved to: {dataset_path}")

except ImportError:
    print("Error: 'kagglehub' package is not installed. Run: pip install kagglehub", file=sys.stderr)
    sys.exit(1)

except Exception as e:
    print(f"Download failed: {e}", file=sys.stderr)
    print("\nKaggle API Credentials Required:", file=sys.stderr)
    print("1. Go to https://www.kaggle.com/settings/account", file=sys.stderr)
    print("2. Select 'Create New API Token' to download kaggle.json", file=sys.stderr)
    print("3. Place 'kaggle.json' in your local ~/.kaggle/ directory", file=sys.stderr)
    sys.exit(1)
