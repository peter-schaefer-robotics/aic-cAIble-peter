#!/usr/bin/env python3
"""Prepare dataset from recorded snapshots and train YOLOv8.

Usage:
  pixi run pip install ultralytics
  pixi run python3 aic_foundationpose/scripts/train_yolov8.py \
      --snapshots rgbd_snapshots \
      --dataset   yolo_dataset \
      --task      detect \
      --epochs    100 \
      --model     yolov8n.pt

Options:
  --task   detect   →  use {cam}_labels.txt   (bounding boxes)
           segment  →  use {cam}_seg.txt       (polygon masks)
  --val-split  fraction of data held out for validation (default 0.2)
"""

import argparse
import random
import shutil
import yaml
from pathlib import Path


CAMERAS = ["left_camera", "center_camera", "right_camera"]


def collect_samples(snapshots_dir: Path, task: str) -> list[tuple[Path, Path]]:
    """Return list of (image_path, label_path) pairs with non-empty labels."""
    label_suffix = "_labels.txt" if task == "detect" else "_seg.txt"
    samples = []
    for snap in sorted(snapshots_dir.iterdir()):
        if not snap.is_dir():
            continue
        for cam in CAMERAS:
            short  = cam.replace("_camera", "")
            img    = snap / f"{short}_rgb.png"
            labels = snap / f"{short}{label_suffix}"
            if img.exists() and labels.exists() and labels.stat().st_size > 0:
                samples.append((img, labels))
    return samples


def filter_labels(src: Path, dst: Path, exclude: set[int], remap: dict[int,int]):
    """Copy label file, dropping excluded classes and remapping IDs."""
    lines_out = []
    for line in src.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        cls = int(parts[0])
        if cls in exclude:
            continue
        parts[0] = str(remap.get(cls, cls))
        lines_out.append(" ".join(parts))
    dst.write_text("\n".join(lines_out) + ("\n" if lines_out else ""))


def build_dataset(samples: list, dataset_dir: Path, val_split: float,
                  exclude_classes: set[int]):
    """Copy images + labels into YOLOv8 directory structure."""
    random.shuffle(samples)
    # Drop samples whose labels become empty after filtering
    filtered = []
    for img, lbl in samples:
        keeps = [l for l in lbl.read_text().splitlines()
                 if l and int(l.split()[0]) not in exclude_classes]
        if keeps:
            filtered.append((img, lbl))
    samples = filtered

    n_val  = max(1, int(len(samples) * val_split))
    splits = {"val": samples[:n_val], "train": samples[n_val:]}

    # Remap class IDs to fill gaps left by excluded classes
    all_classes = set()
    for _, lbl in samples:
        for line in lbl.read_text().splitlines():
            if line:
                all_classes.add(int(line.split()[0]))
    kept_sorted = sorted(all_classes - exclude_classes)
    remap = {old: new for new, old in enumerate(kept_sorted)}

    for split, pairs in splits.items():
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img, lbl in pairs:
            stem = f"{img.parent.name}_{img.stem}"
            shutil.copy2(img, dataset_dir / "images" / split / f"{stem}.png")
            filter_labels(lbl,
                          dataset_dir / "labels" / split / f"{stem}.txt",
                          exclude_classes, remap)
    return remap

    print(f"  train: {len(splits['train'])} images")
    print(f"  val:   {len(splits['val'])}   images")


def write_yaml(dataset_dir: Path, classes_yaml: Path, task: str,
               remap: dict[int,int] = None) -> Path:
    """Write dataset.yaml for YOLOv8."""
    # Load class names from the recorder's classes.yaml
    if classes_yaml.exists():
        with open(classes_yaml) as f:
            classes = yaml.safe_load(f)
        orig_names = classes.get("names", {0: "object"})
    else:
        orig_names = {0: "nic_card", 1: "sc_port", 2: "nic_port"}

    if remap:
        names = {new_id: orig_names[old_id]
                 for old_id, new_id in remap.items()
                 if old_id in orig_names}
    else:
        names = orig_names
    nc = len(names)

    cfg = {
        "path":  str(dataset_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    nc,
        "names": names,
    }
    out = dataset_dir / "dataset.yaml"
    with open(out, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return out


def train(yaml_path: Path, task: str, model: str, epochs: int,
          imgsz: int, batch: int, device: str):
    from ultralytics import YOLO

    # For segment task, use a seg model
    if task == "segment" and not model.endswith("-seg.pt"):
        model = model.replace(".pt", "-seg.pt")
        print(f"  Auto-switched to segmentation model: {model}")

    net = YOLO(model)
    net.train(
        data    = str(yaml_path),
        task    = task,
        epochs  = epochs,
        imgsz   = imgsz,
        batch   = batch,
        device  = device,
        project = "yolov8_runs",
        name    = f"aic_{task}",
        exist_ok= True,
    )
    print(f"\nTraining complete. Results in yolov8_runs/aic_{task}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots",  default="rgbd_snapshots",
                    help="Root snapshot directory")
    ap.add_argument("--dataset",    default="yolo_dataset",
                    help="Output dataset directory")
    ap.add_argument("--task",       default="detect",
                    choices=["detect", "segment"])
    ap.add_argument("--model",      default="yolov8n.pt",
                    help="YOLOv8 model variant (n/s/m/l/x)")
    ap.add_argument("--epochs",     type=int,   default=100)
    ap.add_argument("--imgsz",      type=int,   default=640)
    ap.add_argument("--batch",      type=int,   default=16)
    ap.add_argument("--device",     default="0",
                    help="GPU id or 'cpu'")
    ap.add_argument("--val-split",  type=float, default=0.2)
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--exclude", type=int, nargs="+", default=[],
                    metavar="CLASS_ID",
                    help="Class IDs to exclude (e.g. --exclude 0  to drop nic_card)")
    ap.add_argument("--prepare-only", action="store_true",
                    help="Only prepare dataset, skip training")
    args = ap.parse_args()

    random.seed(args.seed)
    snapshots_dir = Path(args.snapshots)
    dataset_dir   = Path(args.dataset)
    classes_yaml  = snapshots_dir / "classes.yaml"

    print(f"=== Collecting samples from {snapshots_dir} ===")
    samples = collect_samples(snapshots_dir, args.task)
    if not samples:
        print("ERROR: no samples found. Run record_snapshots.py first.")
        return
    print(f"Found {len(samples)} labelled images across all cameras.")

    exclude = set(args.exclude)
    if exclude:
        print(f"Excluding class IDs: {exclude}")

    print(f"\n=== Building dataset in {dataset_dir} ===")
    remap = build_dataset(samples, dataset_dir, args.val_split, exclude)

    yaml_path = write_yaml(dataset_dir, classes_yaml, args.task, remap)
    print(f"Wrote {yaml_path}")

    if args.prepare_only:
        print("\nDataset ready. Run training manually with:")
        print(f"  yolo {args.task} train data={yaml_path} model={args.model} "
              f"epochs={args.epochs} imgsz={args.imgsz}")
        return

    print(f"\n=== Training YOLOv8 ({args.task}, model={args.model}) ===")
    train(yaml_path, args.task, args.model, args.epochs,
          args.imgsz, args.batch, args.device)


if __name__ == "__main__":
    main()
