from pathlib import Path
import os
import sys

import certifi

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

certificate_path = certifi.where()
os.environ.setdefault("REQUESTS_CA_BUNDLE", certificate_path)
os.environ.setdefault("SSL_CERT_FILE", certificate_path)

import kagglehub


DATASET = "navoneel/brain-mri-images-for-brain-tumor-detection"


def main() -> None:
    path = kagglehub.dataset_download(DATASET)
    print("Path to dataset files:", Path(path).resolve())


if __name__ == "__main__":
    main()
