#!/usr/bin/env python3
"""Live training progress plot — refreshes every 30s until training ends."""
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else \
      Path("runs/detect/yolov8_runs/aic_detect/results.csv")

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("YOLOv8 Training Progress", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])


def update(_frame):
    if not CSV.exists():
        return
    try:
        df = pd.read_csv(CSV)
        df.columns = df.columns.str.strip()
    except Exception:
        return

    metrics = [
        ("train/box_loss",   axes[0, 0], "Box Loss",      "tab:blue"),
        ("train/cls_loss",   axes[0, 1], "Class Loss",    "tab:orange"),
        ("train/dfl_loss",   axes[0, 2], "DFL Loss",      "tab:red"),
        ("metrics/mAP50(B)",      axes[1, 0], "mAP@50",        "tab:green"),
        ("metrics/mAP50-95(B)",   axes[1, 1], "mAP@50-95",     "tab:purple"),
        ("metrics/precision(B)",  axes[1, 2], "Precision / Recall", "tab:cyan"),
    ]

    epoch_col = "epoch" if "epoch" in df.columns else df.columns[0]
    epochs = df[epoch_col]

    for col, ax, title, color in metrics:
        ax.cla()
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        if col in df.columns:
            ax.plot(epochs, df[col], color=color, linewidth=1.5)
        elif title == "Precision / Recall":
            for c, lbl in [("metrics/precision(B)", "Prec"),
                           ("metrics/recall(B)",    "Recall")]:
                if c in df.columns:
                    ax.plot(epochs, df[c], label=lbl, linewidth=1.5)
            ax.legend(fontsize=8)

    ep = int(epochs.iloc[-1]) if len(epochs) else 0
    total = 100
    fig.suptitle(f"YOLOv8 Training — epoch {ep}/{total}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])


ani = animation.FuncAnimation(fig, update, interval=30_000, cache_frame_data=False)
update(None)   # draw immediately
plt.show()
