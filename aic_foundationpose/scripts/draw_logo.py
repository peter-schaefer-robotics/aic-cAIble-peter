#!/usr/bin/env python3
"""Draw the Intrinsic logo as understood. Uses only OpenCV (no matplotlib)."""
import numpy as np
import cv2

W, H    = 300, 300   # pixels (1px ≈ 0.2mm at 300px=60mm)
t_thin  = 20         # thin border ~4mm
t_thick = 70         # thick side ~14mm (right side)
slit_h  = 50         # slit height ~10mm
mg      = (180, 0, 180)   # magenta BGR
bg      = (220, 220, 220) # light grey

img = np.full((H + 60, W + 120, 3), bg, dtype=np.uint8)

def r(x, y, w, h, c):
    cv2.rectangle(img, (x, y), (x+w, y+h), c, -1)

# ── logo frame ──────────────────────────────────────────────────────────
r(0,      0,      W,       t_thin,  mg)           # bottom border
r(0,      H-t_thin, W,     t_thin,  mg)           # top border
r(0,      t_thin, t_thin,  H-2*t_thin, mg)        # left border

slit_y_center = H // 4
slit_y_bot    = slit_y_center - slit_h // 2
slit_y_top    = slit_y_center + slit_h // 2

# right thick border: upper part (above slit)
r(W-t_thick, slit_y_top,  t_thick, H-t_thin-slit_y_top, mg)
# right thick border: lower part (below slit)
r(W-t_thick, t_thin,      t_thick, slit_y_bot-t_thin,   mg)
# hollow interior
r(t_thin, t_thin, W-t_thin-t_thick, H-2*t_thin, (240, 240, 240))

# ── slit outline ─────────────────────────────────────────────────────────
cv2.rectangle(img, (W-t_thick, slit_y_bot), (W, slit_y_top), (255,128,0), 2)
cv2.putText(img, "slit (lower half)", (W+5, slit_y_center+5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 100, 0), 1)

# ── coordinate frame at centroid ─────────────────────────────────────────
cx, cy = W//2, H//2
ax_len = 55
# +X right (toward thick side)
cv2.arrowedLine(img, (cx, cy), (cx+ax_len, cy), (0,0,220), 3, tipLength=0.2)
cv2.putText(img, "+X", (cx+ax_len+3, cy+5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,200), 2)
# +Y up (away from slit = toward top of logo)
cv2.arrowedLine(img, (cx, cy), (cx, cy+ax_len), (0,180,0), 3, tipLength=0.2)
cv2.putText(img, "+Y", (cx-30, cy+ax_len+15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,160,0), 2)
cv2.circle(img, (cx, cy), 7, (0, 0, 0), -1)
cv2.putText(img, "centroid", (cx+8, cy-8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)

# ── labels ────────────────────────────────────────────────────────────────
cv2.putText(img, "thick side (~14mm)", (W-t_thick, H+20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100,0,100), 1)
cv2.putText(img, "thin sides (~4mm)", (0, H+40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100,0,100), 1)
cv2.putText(img, "Intrinsic logo – as understood",
            (0, H+58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

# flip Y for display (image Y increases downward, we drew logo with Y up)
img = cv2.flip(img, 0)
cv2.imwrite("logo_understanding.png", img)
print("Saved logo_understanding.png")
