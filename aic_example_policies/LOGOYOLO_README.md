# LogoYoloPolicy

Policy for autonomous SFP and SC cable insertion. Detects the magenta logo on the task board to determine board pose, then uses YOLO to refine the target port position before inserting.

## How to run

```bash
# SFP port
pixi run python3 aic_foundationpose/scripts/test_logo_port_policy.py \
  --module nic_card_mount_0 --port sfp_port_0 --plug-type sfp

# SC port
pixi run python3 aic_foundationpose/scripts/test_logo_port_policy.py \
  --module sc_port_0 --port sc_port_base --plug-type sc

# Eval mode
ros2 run aic_model aic_model --ros-args -p policy:=aic_example_policies.ros.LogoYoloPolicy
```

## Execution steps

### 1. Logo search
If the logo is not immediately visible, the robot rises **15 cm** and sweeps **Y from +45 cm to −45 cm**, checking for magenta pixels (center camera, ≥800 px) every 5 commands. The sweep loops continuously until the logo is found.

### 2. Logo detection (two-phase)
- **Phase 1** — center camera only: detects the magenta logo and moves directly above it in XY.
- **Phase 2** — all three cameras: moves closer in Z until any two cameras agree on yaw within 2°. Applies a **4.5° systematic yaw correction**.

Debug images are saved to `/tmp/logo_yolo_debug/`.

### 3. Static entrance estimate
Looks up the target port's logo-relative XYZ offset from a hardcoded table and transforms it to `base_link` using the detected logo pose.

### 4. Descend to hover height
If the robot is above hover height after logo detection, it descends to hover Z first (keeping current XY) so the camera is closer to the board for YOLO.

### 5. Two-phase hover approach
To prevent the cable snagging on NIC cards:
- **Phase A** — move XY to above the port at current Z (cable clears the cards, yaw rotates to `logo_yaw + 90°`).
- **Phase B** — descend vertically to hover height (6 cm above the entrance).

### 6. YOLO detection
Runs YOLOv8 (`models/best_richard.pt`, conf=0.4) on all three cameras.

- **SFP**: nudges **+40 mm** along the logo X-axis for a better view, detects both ports of the NIC card, picks port_0 (higher Y) or port_1 (lower Y) by relative Y in logo frame. Updates hover XY and EEF yaw from the port-to-port vector.
- **SC**: detects from hover position (no nudge), prefers center camera. Updates hover XY only — yaw stays unchanged.

Detections outside ±30 mm of the expected logo-frame X for the target NIC mount are rejected.

### 7. Insertion orientation
- **SFP**: SLERP to orientation where the tip Z points straight down, using the stored `gripper/tcp → sfp_tip_link` transform.
- **SC**: keeps current yaw, applies only the pitch component of the `gripper/tcp → sc_tip_link` transform.

### 8. Insertion
Pushes straight down at **10 mm/s** with position commands (stiffness 200 N/m, damping 200 Ns/m):
- **SFP**: 13 cm depth
- **SC**: 35 cm depth

Waits **5 s** after reaching target depth.

## Key parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `APPROACH_OFFSET_M` | 0.06 m | Hover height above port entrance |
| `_LOGO_YAW_CORR_DEG` | 4.5° | Systematic logo yaw correction |
| `_X_TOL` | 30 mm | NIC mount identification tolerance |
| `yolo_conf` | 0.4 | YOLO confidence threshold |
| SFP insert depth | 13 cm | |
| SC insert depth | 35 cm | |

## YOLO models

| File | Description |
|------|-------------|
| `models/best_richard.pt` | Primary model (default) |
| `models/best_paul.pt` | Alternative model |

YOLO classes: `0 = sc_port`, `1 = nic_port (SFP)`
