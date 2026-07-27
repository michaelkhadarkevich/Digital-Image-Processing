import csv
import os
from pathlib import Path


MODEL_PATH = Path("models/yolo/yolo_tumor_seg_10epochs.pt")
DATA_ROOT = Path("data/Data 2 with distortion/restoration")
RESULTS_ROOT = Path("result/results task 3/yolo_basic_on_restored_distortion_levels")
METRICS_PATH = Path(
    "result/results task 3/yolo_snr_results/yolo_basic_restored_distortion_level_metrics.csv"
)
METHODS = [
    "gaussian_noise_snr_5db",
    "gaussian_noise_snr_15db",
    "gaussian_noise_snr_30db",
    "gaussian_blur_sigma_0_5",
    "gaussian_blur_sigma_1_5",
    "gaussian_blur_sigma_3_0",
    "brightness_contrast_b1_15_c1_25",
    "brightness_contrast_b1_35_c1_55",
    "brightness_contrast_b1_55_c1_80",
]


def import_yolo():
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Ultralytics YOLO is not installed. Install it with:\n"
            "pip install ultralytics"
        ) from exc
    return YOLO


def metric_value(metrics, group_name: str, attr_name: str) -> float:
    group = getattr(metrics, group_name, None)
    if group is None:
        raise AttributeError(f"YOLO metrics object has no {group_name!r} group.")
    return float(getattr(group, attr_name))


def validate_method(model, method: str) -> dict[str, str | float]:
    data_yaml = DATA_ROOT / f"yolo_segmentation_{method}_restored" / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(data_yaml)

    metrics = model.val(
        data=str(data_yaml),
        imgsz=256,
        batch=4,
        project=str(RESULTS_ROOT.resolve()),
        name=method,
        exist_ok=True,
        task="segment",
        workers=0,
        plots=True,
    )
    return {
        "method": method,
        "basic_restored_mask_map50": metric_value(metrics, "seg", "map50"),
        "basic_restored_mask_map50_95": metric_value(metrics, "seg", "map"),
    }


def main() -> None:
    local_yolo_config = Path(".ultralytics")
    if local_yolo_config.exists():
        os.environ.setdefault("YOLO_CONFIG_DIR", str(local_yolo_config.resolve()))

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    YOLO = import_yolo()
    model = YOLO(str(MODEL_PATH))

    rows = []
    for method in METHODS:
        print("Validating basic YOLO on restored:", method)
        rows.append(validate_method(model, method))

    with METRICS_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "method",
                "basic_restored_mask_map50",
                "basic_restored_mask_map50_95",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Saved basic YOLO restored distortion-level metrics to:", METRICS_PATH)


if __name__ == "__main__":
    main()
