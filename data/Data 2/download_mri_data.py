#!/usr/bin/env python3
"""Download MRI Segmentation dataset from Kaggle and run YOLO detection."""

import sys
import os
from pathlib import Path

# שמור מסלול עבודה עדכני
os.chdir(Path(__file__).parent)

print("=" * 60)
print("הורדת דאטא MRI Segmentation מ-Kaggle")
print("=" * 60)

try:
    import kagglehub
    
    # יצירת תיקייה לדאטא
    data_dir = Path("mri_segmentation")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📥 מורידים את דאטא MRI Segmentation...")
    dataset_path = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")
    print(f"✅ הדאטא הורד בהצלחה ל: {dataset_path}")
    
    # בדיקה של התוכן
    dataset_path = Path(dataset_path)
    if dataset_path.exists():
        files = list(dataset_path.rglob("*.tif")) + list(dataset_path.rglob("*.png"))
        print(f"📊 נמצאו {len(files)} קבצי תמונה בדאטא")
    
except ImportError:
    print("❌ שגיאה: kagglehub לא מותקן")
    print("הקש: pip install kagglehub")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ שגיאה בהורדה: {e}")
    print("\n" + "=" * 60)
    print("דרוש הרשמה ל-Kaggle:")
    print("=" * 60)
    print("1. לך ל: https://www.kaggle.com/settings/account")
    print("2. לחץ על 'Create New API Token'")
    print("3. זה יוריד kaggle.json")
    print("4. העתק אותו ל: C:\\Users\\renan\\.kaggle\\kaggle.json")
    print("5. הרץ את הסקריפט שוב")
    sys.exit(1)

print("\n✅ הכנה הסתיימה בהצלחה!")
