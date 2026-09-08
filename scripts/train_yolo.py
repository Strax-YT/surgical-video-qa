#!/usr/bin/env python3
"""Fine-tune YOLO on a surgical-instrument detection dataset.

Usage:
    python scripts/train_yolo.py --data path/to/data.yaml --epochs 50
    python scripts/train_yolo.py --make-data-yaml path/to/dataset

Dataset layout expected by ultralytics (Roboflow exports this shape directly):

    dataset/
      data.yaml
      train/{images,labels}/
      valid/{images,labels}/
      test/{images,labels}/

Notes on choices that matter for this dataset rather than in general:

  * `yolov8n` is the default because a weekend project on a laptop or a free
    Colab GPU trains it in minutes and the accuracy gap to `yolov8s` on a
    7-class problem is small. Pass --model yolov8s.pt when you have the GPU.
  * Mosaic augmentation is disabled for the last 10 epochs (`close_mosaic`).
    Mosaic helps early but hurts final localisation accuracy, and instrument
    tips are exactly where localisation matters.
  * Horizontal flip is kept, vertical flip is off: laparoscopic footage has a
    consistent gravity direction and flipping it vertically teaches the model
    an orientation that never occurs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from svqa.vision.detector import INSTRUMENT_CLASSES


def make_data_yaml(dataset_dir: Path, classes: tuple[str, ...]) -> Path:
    """Write a data.yaml matching INSTRUMENT_CLASSES.

    Class order here must match the order the detector reports, or every
    prediction is silently mislabelled — the single easiest way to lose a day.
    """
    payload = {
        "path": str(dataset_dir.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": dict(enumerate(classes)),
    }
    target = dataset_dir / "data.yaml"
    target.write_text(yaml.safe_dump(payload, sort_keys=False))
    print(f"wrote {target}")
    print(f"classes: {list(classes)}")
    return target


def train(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        close_mosaic=10,
        fliplr=0.5,
        flipud=0.0,
        hsv_v=0.4,      # surgical lighting varies a lot; brightness jitter helps
        degrees=10.0,
        project=args.project,
        name=args.name,
        seed=args.seed,
        val=True,
    )
    print(results)

    metrics = model.val(data=args.data, split="test" if args.test else "val")
    print(
        f"mAP50-95: {metrics.box.map:.4f}  mAP50: {metrics.box.map50:.4f}"
    )
    # Per-class AP is the number that matters: a headline mAP can look fine
    # while a rarely-used instrument sits near zero.
    for i, ap in enumerate(getattr(metrics.box, "maps", [])):
        name = INSTRUMENT_CLASSES[i] if i < len(INSTRUMENT_CLASSES) else str(i)
        print(f"  {name:<15} AP50-95 = {ap:.4f}")

    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    best = Path(model.trainer.best) if getattr(model, "trainer", None) else None
    if best and best.exists():
        destination.write_bytes(best.read_bytes())
        print(f"best weights -> {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="path to data.yaml")
    parser.add_argument("--make-data-yaml", metavar="DIR",
                        help="write data.yaml for a dataset directory and exit")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="surgical")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test", action="store_true",
                        help="evaluate on the test split instead of val")
    parser.add_argument("--out", default="data/artifacts/yolov8n-surgical.pt")
    args = parser.parse_args()

    if args.make_data_yaml:
        make_data_yaml(Path(args.make_data_yaml), INSTRUMENT_CLASSES)
        return
    if not args.data:
        parser.error("--data is required unless --make-data-yaml is given")
    train(args)


if __name__ == "__main__":
    main()
