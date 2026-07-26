import argparse
import shutil
from pathlib import Path


DEFAULT_DATA_YAML = Path("data/Data 2/yolo_segmentation/data.yaml")
DEFAULT_OUTPUT = Path("result/results task 3/yolo_segmentation")
DEFAULT_MODEL_OUTPUT = Path("models/yolo/yolo_tumor_seg.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO segmentation for MRI tumor masks.")
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA_YAML),
        help="YOLO segmentation data.yaml path.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n-seg.pt",
        help="Base YOLO segmentation model. Example: yolov8n-seg.pt or yolo11n-seg.pt.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Training result folder.",
    )
    parser.add_argument(
        "--save-model",
        default=str(DEFAULT_MODEL_OUTPUT),
        help="Where to copy the trained best.pt model.",
    )
    return parser.parse_args()


def import_yolo():
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Ultralytics YOLO is not installed. Install it with:\n"
            "pip install ultralytics"
        ) from exc
    return YOLO


def main() -> None:
    args = parse_args()
    data_yaml = Path(args.data)
    output_dir = Path(args.output)
    saved_model_path = Path(args.save_model)

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"YOLO segmentation data config not found: {data_yaml}\n"
            'Run: python "task/task 3 yolo/prepare_yolo_segmentation.py"'
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_model_path.parent.mkdir(parents=True, exist_ok=True)

    YOLO = import_yolo()
    model = YOLO(args.model)

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(output_dir.resolve()),
        name="train",
        exist_ok=True,
        task="segment",
        workers=0,
    )

    save_dir = Path(getattr(results, "save_dir", output_dir / "train"))
    best_model = save_dir / "weights" / "best.pt"
    if best_model.exists():
        shutil.copy2(best_model, saved_model_path)
        print("Saved trained tumor segmentation model to:", saved_model_path.resolve())
    else:
        print("Training finished, but best.pt was not found at:", best_model.resolve())

    print("Training results saved to:", save_dir.resolve())


if __name__ == "__main__":
    main()
