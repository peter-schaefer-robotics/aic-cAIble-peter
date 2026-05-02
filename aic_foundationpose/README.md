# aic_foundationpose

Perception and manipulation scripts for the AIC task board — SC port detection,
cable insertion, and supporting data-collection / training utilities.

## Scripts

### `plug_sc_port.py` — autonomous SC cable insertion
Detects the closest SC port with YOLOv8 + depth cameras, aligns the gripper
using the Intrinsic logo orientation, and executes a compliant insertion motion.

```bash
pixi run python3 aic_foundationpose/scripts/plug_sc_port.py
pixi run python3 aic_foundationpose/scripts/plug_sc_port.py --dry-run   # detect only
```

Key options:
| Flag | Default | Description |
|---|---|---|
| `--model` | `runs/detect/…/best.pt` | YOLOv8 weights |
| `--conf` | 0.5 | Detection confidence threshold |
| `--approach-offset` | 0.10 m | Hover height above port |
| `--insert-depth` | 0.025 m | Push depth past entrance |
| `--speed` | 0.02 m/s | Approach speed (insertion is 0.15×) |
| `--hover-wait` | 3.0 s | Settle time before re-detection |
| `--z-rot-offset` | 0° | Fine-tune trim on top of auto logo yaw |

**Motion sequence:**
1. YOLO + depth → detect SC port XY position
2. Logo detection → board orientation → plug Z rotation
3. Approach hover position (position only, no rotation yet)
4. Hold firmly, re-detect with fresh YOLO frame
5. Rotate gripper into insertion orientation
6. Descend slowly into port
7. Wiggle ±2° around insertion axis to seat plug

**Ground truth vs evaluation:**
- With `ground_truth:=true`: uses live `cable_0/sc_tip_link` TF for precise tip orientation
- Without ground truth: uses hardcoded `T_tcp_tip` measured on 2026-05-02

---

### `record_snapshots.py` — RGBD training data collection
Saves synchronised RGB, depth, segmentation, and ground-truth YOLO labels from
all three cameras every N seconds.

```bash
pixi run python3 aic_foundationpose/scripts/record_snapshots.py --interval 5
pixi run python3 aic_foundationpose/scripts/record_snapshots.py --once
```

Saves to `rgbd_snapshots/<timestamp>/` per snapshot.

---

### `train_yolov8.py` — train detector from snapshots
Builds a YOLOv8 dataset from recorded snapshots and trains the model.

```bash
pixi run python3 aic_foundationpose/scripts/train_yolov8.py \
    --snapshots rgbd_snapshots \
    --dataset   yolo_dataset \
    --epochs    100 \
    --device    cpu
```

---

### `augment_dataset.py` — synthetic data augmentation
Multiplies the training set with flips, rotation, scale, colour jitter,
Gaussian noise, cutout (simulates cable occlusion), and mosaic.

```bash
pixi run python3 aic_foundationpose/scripts/augment_dataset.py \
    --dataset yolo_dataset --factor 8
```

---

### `find_logo.py` — Intrinsic logo 6-DoF pose detection
Detects the magenta Intrinsic logo across all three cameras using multi-view
triangulation and flat-plane prior to recover position + yaw.

```bash
pixi run python3 aic_foundationpose/scripts/find_logo.py \
    --width 0.06 --height 0.03
```

Saves `logo_pose.json` and `logo_detection.png`.

---

### `sim_gui.py` — Tkinter robot control GUI
Interactive sliders for Cartesian XYZ + RPY control and task board configuration.

```bash
pixi run python3 aic_foundationpose/scripts/sim_gui.py
```

---

### `plot_training.py` — live training progress plot
Reads `results.csv` and shows loss / mAP curves, refreshing every 30 s.

```bash
pixi run python3 aic_foundationpose/scripts/plot_training.py \
    runs/detect/yolov8_runs/aic_detect/results.csv
```

---

## Data layout (not committed — see .gitignore)

```
rgbd_snapshots/          raw training snapshots
yolo_dataset/            processed YOLOv8 dataset (images + labels)
runs/                    YOLOv8 training outputs and weights
```

## Classes

| ID | Name | Description |
|---|---|---|
| 0 | `sc_port` | SC fibre optic port socket |
| 1 | `nic_port` | SFP port slot entrance on NIC card |
