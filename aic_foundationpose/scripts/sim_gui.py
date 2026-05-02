#!/usr/bin/env python3
"""Simulation control GUI — Robot mover + Task board launcher.

Usage:
  pixi run python3 aic_foundationpose/scripts/sim_gui.py
"""

import math
import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

# ─── ROS backend (runs in background thread) ──────────────────────────────────

import rclpy
from rclpy.node import Node
import tf2_ros
from aic_control_interfaces.srv import ChangeTargetMode
from aic_control_interfaces.msg import MotionUpdate, TargetMode, TrajectoryGenerationMode


class RosBackend:
    """Persistent ROS node: TF listener for pose readback + pose publisher."""

    def __init__(self):
        rclpy.init()
        self._node = rclpy.create_node("sim_gui")
        self._tf_buf = tf2_ros.Buffer()
        self._tf_lst = tf2_ros.TransformListener(self._tf_buf, self._node)
        self._mode_client = self._node.create_client(
            ChangeTargetMode, "/aic_controller/change_target_mode"
        )
        self._pose_pub = self._node.create_publisher(
            MotionUpdate, "/aic_controller/pose_commands", 1
        )
        self._mode_ready = False
        threading.Thread(target=rclpy.spin, args=(self._node,), daemon=True).start()

    # ── mode switching ────────────────────────────────────────────────────────

    def ensure_cartesian_mode(self):
        if self._mode_ready:
            return
        threading.Thread(target=self._set_mode_bg, daemon=True).start()

    def _set_mode_bg(self):
        if self._mode_ready:
            return
        if not self._mode_client.wait_for_service(timeout_sec=5.0):
            return
        req = ChangeTargetMode.Request()
        req.target_mode.mode = TargetMode.MODE_CARTESIAN
        future = self._mode_client.call_async(req)
        # wait without blocking the spin thread — poll until done
        deadline = 3.0
        import time
        t0 = time.monotonic()
        while not future.done() and (time.monotonic() - t0) < deadline:
            time.sleep(0.05)
        if future.done():
            self._mode_ready = True

    # ── EEF pose via TF ───────────────────────────────────────────────────────

    def get_eef_pose(self):
        """Return (x,y,z, qx,qy,qz,qw) of tool0 in base_link, or None."""
        try:
            tf = self._tf_buf.lookup_transform(
                "base_link", "tool0", rclpy.time.Time()
            )
            t = tf.transform.translation
            r = tf.transform.rotation
            return t.x, t.y, t.z, r.x, r.y, r.z, r.w
        except Exception:
            return None

    # ── publish ───────────────────────────────────────────────────────────────

    def publish_pose(self, x, y, z, qx, qy, qz, qw,
                     stiffness: float = 85.0, damping: float = 75.0):
        self.ensure_cartesian_mode()
        s, d = stiffness, damping
        diag = [s,0,0,0,0,0, 0,s,0,0,0,0, 0,0,s,0,0,0,
                0,0,0,s,0,0, 0,0,0,0,s,0, 0,0,0,0,0,s]
        damp = [d,0,0,0,0,0, 0,d,0,0,0,0, 0,0,d,0,0,0,
                0,0,0,d,0,0, 0,0,0,0,d,0, 0,0,0,0,0,d]
        msg = MotionUpdate()
        msg.header.frame_id = "base_link"
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        msg.pose.orientation.x = float(qx)
        msg.pose.orientation.y = float(qy)
        msg.pose.orientation.z = float(qz)
        msg.pose.orientation.w = float(qw)
        msg.target_stiffness = diag
        msg.target_damping = damp
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_POSITION
        self._pose_pub.publish(msg)

    def shutdown(self):
        self._node.destroy_node()
        rclpy.shutdown()


# ─── math helpers ─────────────────────────────────────────────────────────────

def euler_to_quat(roll_deg, pitch_deg, yaw_deg):
    r, p, y = (math.radians(v) for v in (roll_deg, pitch_deg, yaw_deg))
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def quat_to_euler_deg(qx, qy, qz, qw):
    """Quaternion → (roll, pitch, yaw) in degrees."""
    sinr = 2 * (qw * qx + qy * qz)
    cosr = 1 - 2 * (qx * qx + qy * qy)
    roll = math.atan2(sinr, cosr)

    sinp = 2 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)

    siny = 2 * (qw * qz + qx * qy)
    cosy = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny, cosy)

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


# ─── subprocess helper (for ros2 launch commands) ─────────────────────────────

_WS = Path(__file__).parents[2]


def _ros_env() -> str:
    candidates = [
        _WS / ".pixi" / "envs" / "default" / "setup.bash",
        _WS / "install" / "setup.bash",
        _WS / "aic_interfaces" / "install" / "setup.bash",
    ]
    lines = [f"source {p}" for p in candidates if p.exists()]
    return " && ".join(lines) if lines else "true"


def run_cmd(cmd: list[str], log: tk.Text | None = None):
    if cmd and cmd[0] == "ros2":
        shell_cmd = _ros_env() + " && " + " ".join(cmd)
        actual_cmd = ["bash", "-c", shell_cmd]
    else:
        actual_cmd = cmd

    def _target():
        try:
            proc = subprocess.Popen(
                actual_cmd, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, env=os.environ.copy(),
            )
            if log:
                for line in proc.stdout:
                    log.insert(tk.END, line)
                    log.see(tk.END)
            proc.wait()
        except Exception as e:
            if log:
                log.insert(tk.END, f"ERROR: {e}\n")

    threading.Thread(target=_target, daemon=True).start()


# ─── scrollable frame ─────────────────────────────────────────────────────────

class ScrollFrame(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        sb = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.inner = tk.Frame(canvas)
        self.inner.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>",
            lambda e: canvas.yview_scroll(-1 * (e.delta // 120), "units"))


# ─── labelled entry ───────────────────────────────────────────────────────────

def labelled_entry(parent, label, default, row, col=0, width=9):
    tk.Label(parent, text=label, anchor="w").grid(
        row=row, column=col, sticky="w", padx=4, pady=2)
    e = tk.Entry(parent, width=width)
    e.insert(0, default)
    e.grid(row=row, column=col + 1, padx=4, pady=2)
    return e


# ─── component row ────────────────────────────────────────────────────────────

class ComponentRow:
    def __init__(self, parent, label, default_present, default_translation, row):
        self.present_var = tk.BooleanVar(value=default_present)
        ttk.Checkbutton(parent, text=label, variable=self.present_var,
                        width=22).grid(row=row, column=0, sticky="w", padx=4, pady=1)
        tk.Label(parent, text="translation (m)").grid(row=row, column=1, sticky="e", padx=2)
        self.trans_entry = tk.Entry(parent, width=8)
        self.trans_entry.insert(0, str(default_translation))
        self.trans_entry.grid(row=row, column=2, padx=4, pady=1)

    @property
    def present(self):
        return "true" if self.present_var.get() else "false"

    @property
    def translation(self):
        return self.trans_entry.get()


# ─── slider + entry row ───────────────────────────────────────────────────────

class SliderRow:
    def __init__(self, parent, label, from_, to, default, row, resolution=0.001):
        self._var = tk.DoubleVar(value=default)
        self._updating = False

        tk.Label(parent, text=label, width=10, anchor="w").grid(
            row=row, column=0, padx=(6, 2), pady=3, sticky="w")
        self._slider = tk.Scale(
            parent, variable=self._var, from_=from_, to=to,
            resolution=resolution, orient="horizontal",
            length=260, showvalue=False, command=self._on_slider,
        )
        self._slider.grid(row=row, column=1, padx=4, pady=3)
        self._entry = tk.Entry(parent, width=8)
        self._entry.insert(0, f"{default:.3f}")
        self._entry.grid(row=row, column=2, padx=4, pady=3)
        self._entry.bind("<Return>",   self._on_entry)
        self._entry.bind("<FocusOut>", self._on_entry)

    def _on_slider(self, _=None):
        if self._updating:
            return
        self._updating = True
        self._entry.delete(0, tk.END)
        self._entry.insert(0, f"{self._var.get():.3f}")
        self._updating = False

    def _on_entry(self, _=None):
        if self._updating:
            return
        try:
            self._updating = True
            self._var.set(float(self._entry.get()))
            self._updating = False
        except ValueError:
            pass

    def get(self):
        return self._var.get()

    def set(self, v):
        self._var.set(v)
        self._on_slider()


# ─── Robot Control Tab ────────────────────────────────────────────────────────

class RobotTab(ttk.Frame):
    def __init__(self, notebook, log: tk.Text, ros: RosBackend,
                 snapshot_dir: tk.StringVar):
        super().__init__(notebook)
        self._log = log
        self._ros = ros
        self._snapshot_dir = snapshot_dir
        self._pub_after_id = None
        self._live_var = tk.BooleanVar(value=True)

        # ── Position ──────────────────────────────────────────────────────────
        pfrm = ttk.LabelFrame(self, text="Position")
        pfrm.pack(fill="x", padx=10, pady=6)
        self._x = SliderRow(pfrm, "X (m)", -0.8,  0.8,  0.0, 0)
        self._y = SliderRow(pfrm, "Y (m)", -0.8,  0.8,  0.0, 1)
        self._z = SliderRow(pfrm, "Z (m)",  0.0,  1.0,  0.5, 2)

        # ── Rotation ──────────────────────────────────────────────────────────
        rfrm = ttk.LabelFrame(self, text="Rotation")
        rfrm.pack(fill="x", padx=10, pady=4)
        self._roll  = SliderRow(rfrm, "Roll (°)",  -180, 180, 0.0, 0, resolution=0.5)
        self._pitch = SliderRow(rfrm, "Pitch (°)", -180, 180, 0.0, 1, resolution=0.5)
        self._yaw   = SliderRow(rfrm, "Yaw (°)",   -180, 180, 0.0, 2, resolution=0.5)

        # ── Quaternion display ────────────────────────────────────────────────
        qfrm = ttk.LabelFrame(self, text="Quaternion (live)")
        qfrm.pack(fill="x", padx=10, pady=4)
        self._qvars = [tk.StringVar(value="—") for _ in range(4)]
        for i, (lbl, var) in enumerate(zip(["qx", "qy", "qz", "qw"], self._qvars)):
            tk.Label(qfrm, text=lbl).grid(row=0, column=i * 2, padx=6)
            tk.Label(qfrm, textvariable=var, width=10, relief="sunken").grid(
                row=0, column=i * 2 + 1, padx=2, pady=4)

        # ── Controller params ─────────────────────────────────────────────────
        cfrm = ttk.LabelFrame(self, text="Controller")
        cfrm.pack(fill="x", padx=10, pady=4)
        self._stiff = labelled_entry(cfrm, "Stiffness", "85.0", 0)
        self._damp  = labelled_entry(cfrm, "Damping",   "75.0", 1)
        ttk.Checkbutton(cfrm, text="Live publish on slider move",
                        variable=self._live_var).grid(
            row=2, column=0, columnspan=2, sticky="w", padx=4, pady=4)

        # ── Status bar ────────────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Waiting for robot TF…")
        tk.Label(self, textvariable=self._status_var, anchor="w",
                 fg="gray").pack(fill="x", padx=10)

        # ── Buttons ───────────────────────────────────────────────────────────
        bfrm = tk.Frame(self)
        bfrm.pack(pady=8)
        ttk.Button(bfrm, text="Send pose", command=self._send,
                   style="Accent.TButton").pack(side="left", padx=6)
        ttk.Separator(bfrm, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Button(bfrm, text="📷  Take snapshot",
                   command=self._snapshot).pack(side="left", padx=6)

        # ── Snapshot dir ──────────────────────────────────────────────────────
        dfrm = tk.Frame(self)
        dfrm.pack(fill="x", padx=10, pady=(0, 6))
        tk.Label(dfrm, text="Snapshot dir:").pack(side="left")
        tk.Entry(dfrm, textvariable=self._snapshot_dir, width=30).pack(
            side="left", padx=4)

        # Wire all sliders → quat display + live publish
        for s in (self._x, self._y, self._z,
                  self._roll, self._pitch, self._yaw):
            s._var.trace_add("write", self._on_slider_change)

        # Read initial pose after GUI is up
        self.after(800, self._try_fetch_pose)

    # ── slider change ─────────────────────────────────────────────────────────

    def _on_slider_change(self, *_):
        self._update_quat()
        if not self._live_var.get():
            return
        if self._pub_after_id is not None:
            self.after_cancel(self._pub_after_id)
        self._pub_after_id = self.after(80, self._do_publish)

    def _update_quat(self):
        try:
            qx, qy, qz, qw = euler_to_quat(
                self._roll.get(), self._pitch.get(), self._yaw.get())
            for var, v in zip(self._qvars, [qx, qy, qz, qw]):
                var.set(f"{v:.5f}")
        except Exception:
            pass

    # ── publish ───────────────────────────────────────────────────────────────

    def _do_publish(self):
        try:
            qx, qy, qz, qw = euler_to_quat(
                self._roll.get(), self._pitch.get(), self._yaw.get())
            self._ros.publish_pose(
                self._x.get(), self._y.get(), self._z.get(),
                qx, qy, qz, qw,
                float(self._stiff.get()), float(self._damp.get()),
            )
            self._status_var.set(
                f"Published  xyz=({self._x.get():.3f}, {self._y.get():.3f}, "
                f"{self._z.get():.3f})"
            )
        except Exception as e:
            self._status_var.set(f"Publish error: {e}")

    def _send(self):
        self._do_publish()

    # ── read robot pose ───────────────────────────────────────────────────────

    def _try_fetch_pose(self):
        if not self._fetch_pose():
            self.after(500, self._try_fetch_pose)  # retry until TF available

    def _fetch_pose(self) -> bool:
        pose = self._ros.get_eef_pose()
        if pose is None:
            self._status_var.set("Waiting for robot TF (tool0 → base_link)…")
            return False
        x, y, z, qx, qy, qz, qw = pose
        roll, pitch, yaw = quat_to_euler_deg(qx, qy, qz, qw)
        # Temporarily disable live publish while we set sliders
        live = self._live_var.get()
        self._live_var.set(False)
        self._x.set(x);     self._y.set(y);     self._z.set(z)
        self._roll.set(roll); self._pitch.set(pitch); self._yaw.set(yaw)
        self._live_var.set(live)
        self._update_quat()
        self._status_var.set(
            f"Robot pose loaded  xyz=({x:.3f}, {y:.3f}, {z:.3f})  "
            f"rpy=({roll:.1f}°, {pitch:.1f}°, {yaw:.1f}°)"
        )
        return True

    # ── snapshot ──────────────────────────────────────────────────────────────

    def _snapshot(self):
        cmd = [sys.executable,
               "aic_foundationpose/scripts/record_snapshots.py",
               "--output", self._snapshot_dir.get(), "--once"]
        self._log.insert(tk.END, "📷 Taking snapshot…\n")
        self._log.see(tk.END)
        run_cmd(cmd, self._log)


# ─── Task Board Tab ───────────────────────────────────────────────────────────

class TaskBoardTab(ttk.Frame):
    def __init__(self, notebook, log: tk.Text):
        super().__init__(notebook)
        self._log = log

        sf = ScrollFrame(self)
        sf.pack(fill="both", expand=True, padx=4, pady=4)
        p = sf.inner

        pfrm = ttk.LabelFrame(p, text="Task Board Pose")
        pfrm.pack(fill="x", padx=8, pady=4)
        self._bx     = labelled_entry(pfrm, "X (m)",     "0.3",  0)
        self._by     = labelled_entry(pfrm, "Y (m)",     "-0.1", 1)
        self._bz     = labelled_entry(pfrm, "Z (m)",     "1.2",  2)
        self._broll  = labelled_entry(pfrm, "Roll (°)",  "0.0",  3)
        self._bpitch = labelled_entry(pfrm, "Pitch (°)", "0.0",  4)
        self._byaw   = labelled_entry(pfrm, "Yaw (°)",   "45.0", 5)

        r0frm = ttk.LabelFrame(p, text="Rail 0 Mounts")
        r0frm.pack(fill="x", padx=8, pady=4)
        self._sfp0 = ComponentRow(r0frm, "SFP Mount Rail 0", True,  -0.08, 0)
        self._sc0  = ComponentRow(r0frm, "SC Mount Rail 0",  True,  -0.09, 1)
        self._lc0  = ComponentRow(r0frm, "LC Mount Rail 0",  False,  0.0,  2)

        r1frm = ttk.LabelFrame(p, text="Rail 1 Mounts")
        r1frm.pack(fill="x", padx=8, pady=4)
        self._sfp1 = ComponentRow(r1frm, "SFP Mount Rail 1", False, 0.0, 0)
        self._sc1  = ComponentRow(r1frm, "SC Mount Rail 1",  False, 0.0, 1)
        self._lc1  = ComponentRow(r1frm, "LC Mount Rail 1",  False, 0.0, 2)

        spfrm = ttk.LabelFrame(p, text="SC Ports")
        spfrm.pack(fill="x", padx=8, pady=4)
        self._scp0 = ComponentRow(spfrm, "SC Port 0", True,  -0.04, 0)
        self._scp1 = ComponentRow(spfrm, "SC Port 1", False,  0.0,  1)

        nfrm = ttk.LabelFrame(p, text="NIC Card Mounts")
        nfrm.pack(fill="x", padx=8, pady=4)
        self._nic = [
            ComponentRow(nfrm, f"NIC Card Mount {i}",
                         i == 0, 0.005 if i == 0 else 0.0, i)
            for i in range(5)
        ]

        cfrm = ttk.LabelFrame(p, text="Cable")
        cfrm.pack(fill="x", padx=8, pady=4)
        self._spawn_cable  = tk.BooleanVar(value=True)
        self._attach_cable = tk.BooleanVar(value=True)
        ttk.Checkbutton(cfrm, text="Spawn cable",
                        variable=self._spawn_cable).grid(row=0, column=0, padx=4)
        tk.Label(cfrm, text="Cable type").grid(row=0, column=1, padx=8)
        self._cable_type = ttk.Combobox(
            cfrm, width=18,
            values=["sfp_sc_cable", "sfp_sc_cable_reversed",
                    "lc_sc_cable", "lc_cable", "sc_cable"],
        )
        self._cable_type.set("sfp_sc_cable")
        self._cable_type.grid(row=0, column=2, padx=4)
        ttk.Checkbutton(cfrm, text="Attach to gripper",
                        variable=self._attach_cable).grid(row=0, column=3, padx=8)

        ofrm = ttk.LabelFrame(p, text="Options")
        ofrm.pack(fill="x", padx=8, pady=4)
        self._ground_truth = tk.BooleanVar(value=True)
        self._start_engine = tk.BooleanVar(value=False)
        ttk.Checkbutton(ofrm, text="Ground truth",
                        variable=self._ground_truth).grid(row=0, column=0, padx=10)
        ttk.Checkbutton(ofrm, text="Start AIC engine",
                        variable=self._start_engine).grid(row=0, column=1, padx=10)

        # ── Scenario save ─────────────────────────────────────────────────────
        sfrm = ttk.LabelFrame(p, text="Save Scenario SDF")
        sfrm.pack(fill="x", padx=8, pady=4)
        tk.Label(sfrm, text="Name:").grid(row=0, column=0, padx=4, pady=4)
        self._scenario_name = tk.Entry(sfrm, width=24)
        self._scenario_name.insert(0, "nic_slot_0")
        self._scenario_name.grid(row=0, column=1, padx=4, pady=4)
        tk.Label(sfrm, text="Dir:").grid(row=0, column=2, padx=4)
        self._scenario_dir = tk.Entry(sfrm, width=28)
        self._scenario_dir.insert(0, str(Path.home() / "training_scenarios"))
        self._scenario_dir.grid(row=0, column=3, padx=4, pady=4)
        ttk.Button(sfrm, text="Save SDF  (/tmp/aic.sdf → …)",
                   command=self._save_sdf).grid(row=0, column=4, padx=8)

        # ── Buttons ───────────────────────────────────────────────────────────
        bfrm = tk.Frame(p)
        bfrm.pack(pady=8)
        ttk.Button(bfrm, text="Preview command",
                   command=self._preview).pack(side="left", padx=6)
        self._launch_btn = ttk.Button(bfrm, text="Launch via /entrypoint.sh",
                                      command=self._launch,
                                      style="Accent.TButton")
        self._launch_btn.pack(side="left", padx=6)
        self._stop_btn = ttk.Button(bfrm, text="Stop simulation",
                                    command=self._stop, state="disabled")
        self._stop_btn.pack(side="left", padx=6)
        self._sim_proc = None

    def _bstr(self, bv):
        return "true" if bv.get() else "false"

    def _all_args(self) -> list[str]:
        """All args for one /entrypoint.sh call (spawn_task_board included)."""
        yaw_rad   = math.radians(float(self._byaw.get()))
        roll_rad  = math.radians(float(self._broll.get()))
        pitch_rad = math.radians(float(self._bpitch.get()))
        args = [
            "spawn_task_board:=true",
            f"task_board_x:={self._bx.get()}",
            f"task_board_y:={self._by.get()}",
            f"task_board_z:={self._bz.get()}",
            f"task_board_roll:={roll_rad:.6f}",
            f"task_board_pitch:={pitch_rad:.6f}",
            f"task_board_yaw:={yaw_rad:.6f}",
            f"sfp_mount_rail_0_present:={self._sfp0.present}",
            f"sfp_mount_rail_0_translation:={self._sfp0.translation}",
            f"sc_mount_rail_0_present:={self._sc0.present}",
            f"sc_mount_rail_0_translation:={self._sc0.translation}",
            f"lc_mount_rail_0_present:={self._lc0.present}",
            f"lc_mount_rail_0_translation:={self._lc0.translation}",
            f"sfp_mount_rail_1_present:={self._sfp1.present}",
            f"sfp_mount_rail_1_translation:={self._sfp1.translation}",
            f"sc_mount_rail_1_present:={self._sc1.present}",
            f"sc_mount_rail_1_translation:={self._sc1.translation}",
            f"lc_mount_rail_1_present:={self._lc1.present}",
            f"lc_mount_rail_1_translation:={self._lc1.translation}",
            f"sc_port_0_present:={self._scp0.present}",
            f"sc_port_0_translation:={self._scp0.translation}",
            f"sc_port_1_present:={self._scp1.present}",
            f"sc_port_1_translation:={self._scp1.translation}",
        ]
        for i, nic in enumerate(self._nic):
            args += [f"nic_card_mount_{i}_present:={nic.present}",
                     f"nic_card_mount_{i}_translation:={nic.translation}"]
        args += [
            f"spawn_cable:={self._bstr(self._spawn_cable)}",
            f"cable_type:={self._cable_type.get()}",
            f"attach_cable_to_gripper:={self._bstr(self._attach_cable)}",
            f"ground_truth:={self._bstr(self._ground_truth)}",
            f"start_aic_engine:={self._bstr(self._start_engine)}",
        ]
        return args

    def _preview(self):
        full = "/entrypoint.sh \\\n  " + " \\\n  ".join(self._all_args())
        win = tk.Toplevel()
        win.title("Command preview")
        txt = tk.Text(win, width=90, height=30, font=("Monospace", 10))
        txt.pack(fill="both", expand=True, padx=8, pady=8)
        txt.insert(tk.END, full)
        ttk.Button(win, text="Copy to clipboard",
                   command=lambda: (win.clipboard_clear(),
                                    win.clipboard_append(full))).pack(pady=4)

    def _launch(self):
        if self._sim_proc and self._sim_proc.poll() is None:
            messagebox.showwarning("Already running",
                                   "Simulation is already running. Stop it first.")
            return
        cmd = ["/entrypoint.sh"] + self._all_args()
        self._log.insert(tk.END, "→ " + " ".join(cmd) + "\n")
        self._log.see(tk.END)

        import os
        self._sim_proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=os.environ.copy(),
        )
        self._launch_btn.config(state="disabled")
        self._stop_btn.config(state="normal")

        def _stream():
            for line in self._sim_proc.stdout:
                self._log.insert(tk.END, line)
                self._log.see(tk.END)
            # process ended
            self._log.insert(tk.END, "[ simulation stopped ]\n")
            self._log.see(tk.END)
            self._launch_btn.config(state="normal")
            self._stop_btn.config(state="disabled")

        threading.Thread(target=_stream, daemon=True).start()

    def _stop(self):
        if self._sim_proc and self._sim_proc.poll() is None:
            self._sim_proc.terminate()
            self._log.insert(tk.END, "Stopping simulation (SIGTERM)…\n")
            self._log.see(tk.END)
        self._stop_btn.config(state="disabled")

    def _save_sdf(self):
        src = Path("/tmp/aic.sdf")
        if not src.exists():
            messagebox.showerror("Not found",
                                 "/tmp/aic.sdf does not exist.\n"
                                 "Run the simulation first.")
            return
        dest_dir = Path(self._scenario_dir.get())
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{self._scenario_name.get()}.sdf"
        import shutil
        shutil.copy2(src, dest)
        self._log.insert(tk.END, f"✓ Saved SDF → {dest}\n")
        self._log.see(tk.END)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ros = RosBackend()

    root = tk.Tk()
    root.title("AIC Simulation GUI")
    root.resizable(True, True)
    root.protocol("WM_DELETE_WINDOW", lambda: (ros.shutdown(), root.destroy()))

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=6, pady=6)

    log_frm = ttk.LabelFrame(root, text="Output log")
    log_frm.pack(fill="x", padx=6, pady=(0, 6))
    log = tk.Text(log_frm, height=6, font=("Monospace", 9))
    log_sb = ttk.Scrollbar(log_frm, command=log.yview)
    log.configure(yscrollcommand=log_sb.set)
    log_sb.pack(side="right", fill="y")
    log.pack(fill="x")
    ttk.Button(log_frm, text="Clear",
               command=lambda: log.delete("1.0", tk.END)).pack(anchor="e")

    snapshot_dir = tk.StringVar(value="rgbd_snapshots")
    robot_tab = RobotTab(notebook, log, ros, snapshot_dir)
    task_tab  = TaskBoardTab(notebook, log)
    notebook.add(robot_tab, text="  Robot Control  ")
    notebook.add(task_tab,  text="  Task Board  ")

    root.mainloop()


if __name__ == "__main__":
    main()
