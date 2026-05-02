#!/usr/bin/env python3
"""Synthetically augment a YOLO detection dataset.

Reads images + labels from yolo_dataset/images/{split} and
writes augmented copies alongside them, multiplying dataset size.

Augmentations applied (all preserve/adjust bounding boxes correctly):
  - Horizontal / vertical flip
  - Rotation ±20°
  - Scale zoom 0.75 – 1.25
  - Brightness / contrast / saturation jitter
  - Gaussian noise
  - Cutout (random rectangular occlusion — simulates cables)
  - Mosaic (4-image tile)

Usage:
  pixi run python3 aic_foundationpose/scripts/augment_dataset.py \
      --dataset yolo_dataset \
      --factor  8           \   # produce 8x augmented copies per original
      --splits  train       \   # only augment train, leave val clean
      --seed    42
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


# ─── YOLO bbox helpers ────────────────────────────────────────────────────────

def yolo_to_corners(cx, cy, w, h, W, H):
    """YOLO normalised → pixel (x1,y1,x2,y2)."""
    x1 = (cx - w/2) * W;  y1 = (cy - h/2) * H
    x2 = (cx + w/2) * W;  y2 = (cy + h/2) * H
    return x1, y1, x2, y2


def corners_to_yolo(x1, y1, x2, y2, W, H):
    """Pixel (x1,y1,x2,y2) → YOLO normalised (cx,cy,w,h)."""
    cx = (x1 + x2) / 2 / W;  cy = (y1 + y2) / 2 / H
    w  = (x2 - x1) / W;       h  = (y2 - y1) / H
    return cx, cy, w, h


def clip_boxes(boxes, min_wh=0.01):
    """Clip boxes to [0,1] and drop tiny ones."""
    kept = []
    for cls, cx, cy, bw, bh in boxes:
        x1 = max(0, cx - bw/2);  y1 = max(0, cy - bh/2)
        x2 = min(1, cx + bw/2);  y2 = min(1, cy + bh/2)
        nw = x2 - x1;  nh = y2 - y1
        if nw >= min_wh and nh >= min_wh:
            kept.append((cls, (x1+x2)/2, (y1+y2)/2, nw, nh))
    return kept


def read_labels(path: Path):
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 5:
            boxes.append((int(parts[0]), *map(float, parts[1:])))
    return boxes


def write_labels(path: Path, boxes):
    path.write_text(
        "".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
                for c, cx, cy, w, h in boxes)
    )


# ─── individual augmentations ─────────────────────────────────────────────────

def flip_h(img, boxes):
    img2 = cv2.flip(img, 1)
    boxes2 = [(c, 1-cx, cy, w, h) for c, cx, cy, w, h in boxes]
    return img2, boxes2


def flip_v(img, boxes):
    img2 = cv2.flip(img, 0)
    boxes2 = [(c, cx, 1-cy, w, h) for c, cx, cy, w, h in boxes]
    return img2, boxes2


def rotate(img, boxes, angle):
    H, W = img.shape[:2]
    M    = cv2.getRotationMatrix2D((W/2, H/2), angle, 1.0)
    img2 = cv2.warpAffine(img, M, (W, H),
                           borderMode=cv2.BORDER_REFLECT_101)
    boxes2 = []
    for c, cx, cy, bw, bh in boxes:
        # Rotate all 4 corners
        x1, y1, x2, y2 = yolo_to_corners(cx, cy, bw, bh, W, H)
        pts = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32)
        pts = cv2.transform(pts.reshape(-1,1,2), M).reshape(-1,2)
        rx1,ry1 = pts[:,0].min(), pts[:,1].min()
        rx2,ry2 = pts[:,0].max(), pts[:,1].max()
        boxes2.append((c, *corners_to_yolo(rx1,ry1,rx2,ry2,W,H)))
    return img2, clip_boxes(boxes2)


def scale(img, boxes, factor):
    H, W = img.shape[:2]
    nW, nH = int(W*factor), int(H*factor)
    img2   = cv2.resize(img, (nW, nH))
    if factor > 1:            # crop back to original size
        x0 = (nW-W)//2;  y0 = (nH-H)//2
        img2 = img2[y0:y0+H, x0:x0+W]
        # boxes: scale then re-centre
        boxes2 = []
        for c, cx, cy, bw, bh in boxes:
            ncx = (cx*nW - x0) / W
            ncy = (cy*nH - y0) / H
            boxes2.append((c, ncx, ncy, bw*factor, bh*factor))
    else:                     # pad to original size
        canvas = np.zeros_like(img)
        x0 = (W-nW)//2;  y0 = (H-nH)//2
        canvas[y0:y0+nH, x0:x0+nW] = img2
        img2 = canvas
        boxes2 = [(c, cx*factor+x0/W, cy*factor+y0/H, bw*factor, bh*factor)
                  for c, cx, cy, bw, bh in boxes]
    return img2, clip_boxes(boxes2)


def color_jitter(img, boxes,
                 brightness=0.3, contrast=0.3, saturation=0.3):
    img2 = img.astype(np.float32)
    # brightness
    img2 += random.uniform(-brightness, brightness) * 255
    # contrast
    mean = img2.mean()
    img2 = (img2 - mean) * random.uniform(1-contrast, 1+contrast) + mean
    img2 = np.clip(img2, 0, 255).astype(np.uint8)
    # saturation (HSV)
    hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] *= random.uniform(1-saturation, 1+saturation)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), boxes


def gaussian_noise(img, boxes, std=15):
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    img2  = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img2, boxes


def cutout(img, boxes, n_holes=3, max_frac=0.15):
    """Random rectangular occlusions — simulates cable obstruction."""
    H, W  = img.shape[:2]
    img2  = img.copy()
    for _ in range(n_holes):
        hw = int(random.uniform(0.02, max_frac) * W)
        hh = int(random.uniform(0.02, max_frac) * H)
        cx = random.randint(0, W);  cy = random.randint(0, H)
        x1,y1 = max(0,cx-hw//2), max(0,cy-hh//2)
        x2,y2 = min(W,cx+hw//2), min(H,cy+hh//2)
        img2[y1:y2, x1:x2] = random.randint(0, 50)   # dark cable colour
    return img2, boxes


def mosaic(imgs_boxes, out_size=640):
    """Combine 4 random images into one mosaic."""
    H, W  = out_size, out_size
    cx, cy = W//2, H//2
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    all_boxes = []
    positions = [(0,0,cx,cy), (cx,0,W,cy), (0,cy,cx,H), (cx,cy,W,H)]

    for (x0,y0,x1,y1), (img, boxes) in zip(positions, imgs_boxes):
        tile_w, tile_h = x1-x0, y1-y0
        tile = cv2.resize(img, (tile_w, tile_h))
        canvas[y0:y1, x0:x1] = tile
        for c, bcx, bcy, bw, bh in boxes:
            ncx = (bcx*tile_w + x0) / W
            ncy = (bcy*tile_h + y0) / H
            all_boxes.append((c, ncx, ncy, bw*tile_w/W, bh*tile_h/H))

    return canvas, clip_boxes(all_boxes)


# ─── augmentation pipeline ────────────────────────────────────────────────────

def augment_once(img, boxes):
    ops = []
    if random.random() < 0.5:
        ops.append(('flip_h', {}))
    if random.random() < 0.2:
        ops.append(('flip_v', {}))
    if random.random() < 0.5:
        ops.append(('rotate', {'angle': random.uniform(-20, 20)}))
    if random.random() < 0.5:
        ops.append(('scale', {'factor': random.uniform(0.75, 1.25)}))
    if random.random() < 0.8:
        ops.append(('color_jitter', {}))
    if random.random() < 0.4:
        ops.append(('gaussian_noise', {'std': random.uniform(5, 20)}))
    if random.random() < 0.5:
        ops.append(('cutout', {'n_holes': random.randint(1, 4)}))

    for op, kw in ops:
        if op == 'flip_h':      img, boxes = flip_h(img, boxes)
        elif op == 'flip_v':    img, boxes = flip_v(img, boxes)
        elif op == 'rotate':    img, boxes = rotate(img, boxes, **kw)
        elif op == 'scale':     img, boxes = scale(img, boxes, **kw)
        elif op == 'color_jitter': img, boxes = color_jitter(img, boxes)
        elif op == 'gaussian_noise': img, boxes = gaussian_noise(img, boxes, **kw)
        elif op == 'cutout':    img, boxes = cutout(img, boxes, **kw)
        if not boxes:
            return None, None
    return img, boxes


# ─── main ─────────────────────────────────────────────────────────────────────

def augment_split(images_dir: Path, labels_dir: Path,
                  factor: int, use_mosaic: bool):
    pairs = []
    for img_path in sorted(images_dir.glob("*.png")):
        if "_aug" in img_path.stem:
            continue   # skip previously generated augmentations
        lbl_path = labels_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))

    if not pairs:
        print(f"  No original images found in {images_dir}")
        return

    print(f"  {len(pairs)} originals → generating {len(pairs)*factor} augmented copies")
    generated = 0

    for img_path, lbl_path in pairs:
        img   = cv2.imread(str(img_path))
        boxes = read_labels(lbl_path)
        if img is None or not boxes:
            continue

        for k in range(factor):
            # Every 4th augmentation: mosaic from random other images
            if use_mosaic and k % 4 == 3 and len(pairs) >= 4:
                four = random.sample(pairs, 4)
                imgs_boxes = []
                for ip, lp in four:
                    i2 = cv2.imread(str(ip));  b2 = read_labels(lp)
                    if i2 is not None and b2:
                        imgs_boxes.append((i2, b2))
                if len(imgs_boxes) == 4:
                    aug_img, aug_boxes = mosaic(imgs_boxes)
                else:
                    aug_img, aug_boxes = augment_once(img.copy(), list(boxes))
            else:
                aug_img, aug_boxes = augment_once(img.copy(), list(boxes))

            if aug_img is None or not aug_boxes:
                continue

            stem = f"{img_path.stem}_aug{k:03d}"
            cv2.imwrite(str(images_dir / f"{stem}.png"), aug_img)
            write_labels(labels_dir / f"{stem}.txt", aug_boxes)
            generated += 1

    print(f"  Generated {generated} augmented images")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",  default="yolo_dataset")
    ap.add_argument("--factor",   type=int, default=8,
                    help="Augmented copies per original image")
    ap.add_argument("--splits",   nargs="+", default=["train"],
                    help="Which splits to augment (leave val clean)")
    ap.add_argument("--no-mosaic", action="store_true")
    ap.add_argument("--seed",     type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    dataset = Path(args.dataset)

    for split in args.splits:
        images_dir = dataset / "images" / split
        labels_dir = dataset / "labels" / split
        if not images_dir.exists():
            print(f"Skipping {split} — not found")
            continue
        print(f"=== Augmenting {split} (×{args.factor}) ===")
        augment_split(images_dir, labels_dir,
                      args.factor, not args.no_mosaic)

    print("\nDone. Re-run training to use the augmented dataset.")


if __name__ == "__main__":
    main()
