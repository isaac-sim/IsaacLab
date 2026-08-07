# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Dual-arm recording script for OpenArm tasks.

Extends the standard record_demos.py with TAB key arm switching so the operator
can control either the left or the right arm during data collection.

Uses OpenArmKeyboard instead of Se3Keyboard to avoid Isaac Sim viewport conflicts.
W/A/S/D/Q/E are viewport gizmo shortcuts — this script uses arrow keys + I/O instead.

Keyboard controls:
  W/S        forward / backward  (EE +x/-x)
  A/D        left / right        (EE +y/-y)
  PgUp/PgDn  up / down           (EE +z/-z)
  ↑/↓        pitch ±
  ←/→        yaw ±
  [/]        roll ±
  K          toggle gripper open/close
  TAB        switch active arm (left ↔ right)
  R          reset / discard current episode
  N          save current episode as successful

Action space (14D flat):
  [0:6]   left arm IK delta pose (dx dy dz drx dry drz)
  [6:7]   left gripper command (±1.0)
  [7:13]  right arm IK delta pose
  [13:14] right gripper command (±1.0)

Usage:
  ./isaaclab.sh -p scripts/tools/record_demos_openarm.py \\
      --task Isaac-Reach-RedCube-OpenArm-IK-Abs-v0 \\
      --dataset_file logs/demos/openarm_reach.hdf5 \\
      --enable_cameras

  ./isaaclab.sh -p scripts/tools/record_demos_openarm.py \\
      --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0 \\
      --dataset_file logs/demos/pickup.hdf5 \\
      --enable_cameras --num_demos 1 \\
      --teleop_device vr_ros2 \\
      --vr_udp_host 127.0.0.1 --vr_udp_port 5800
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import atexit
import signal
import contextlib

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record dual-arm OpenArm demonstrations with arm switching.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    choices=["keyboard", "vr_ros2"],
    help=(
        "Teleop device. 'keyboard' uses OpenArmKeyboard (single active arm, TAB to switch)."
        " 'vr_ros2' drives BOTH arms simultaneously from a UDP JSON side-channel fed by"
        " nodes/dora-openarm-ros2-bridge/bridge.py's --vr-udp-port (see --vr_udp_host/"
        " --vr_udp_port below) -- only valid on dual-arm (14D action) tasks. R/N/T keyboard"
        " shortcuts (reset/save/ramp-test) still work in 'vr_ros2' mode; TAB/arm-switch does not"
        " (both arms are always live)."
    ),
)
parser.add_argument(
    "--vr_udp_host",
    type=str,
    default="127.0.0.1",
    help="Host to bind for --teleop_device vr_ros2's UDP JSON listener.",
)
parser.add_argument(
    "--vr_udp_port",
    type=int,
    default=5800,
    help="Port to bind for --teleop_device vr_ros2's UDP JSON listener.",
)
parser.add_argument(
    "--vr_max_pos_step",
    type=float,
    default=0.01,
    help=(
        "--teleop_device vr_ros2 only: per-step clamp (meters) on the position delta sent to"
        " the IK controller. Since the delta is recomputed from the live VR target every step"
        " (not accumulated), clamping just makes the arm converge to the target over a few"
        " steps instead of possibly snapping -- e.g. right after a reset, or if a UDP packet"
        " is stale/lost for a moment."
    ),
)
parser.add_argument(
    "--vr_max_rot_step",
    type=float,
    default=0.05,
    help="--teleop_device vr_ros2 only: per-step clamp (radians) on the axis-angle rotation delta. See --vr_max_pos_step.",
)
parser.add_argument(
    "--vr_quat_offset",
    type=float,
    nargs=4,
    default=[1.0, 0.0, 0.0, 0.0],
    metavar=("QW", "QX", "QY", "QZ"),
    help=(
        "--teleop_device vr_ros2 only: fixed wxyz rotation left-multiplied onto every incoming"
        " VR target orientation (in the robot base frame) before it's used for IK. dora-openarm-vr's"
        " quest_receiver.py maps raw controller poses into the robot 'arm_origin' frame using a"
        " _FRAME_ROT/r_fix pair that was hand-tuned against the MuJoCo viewer -- if Isaac Sim's robot"
        " base frame uses a different axis convention than MuJoCo's arm_origin, the same UDP pose"
        " stream will produce a rotated-looking arm here. Default is identity (no correction, matches"
        " prior behavior). To determine the actual offset: hold the VR controller still, let the arm"
        " converge in both sims, read off openarm_*_ee_tcp's orientation here vs. MuJoCo's for that"
        " same held pose, and solve for the rotation that maps one to the other."
    ),
)
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos."
)
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument("--num_demos", type=int, default=0, help="Number of demonstrations to record (0 = infinite).")
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Number of consecutive success steps to conclude a demo.",
)
parser.add_argument(
    "--mirror_udp_port",
    type=int,
    default=0,
    help=(
        "If nonzero, broadcast the robot's current joint positions (by name, radians) as a UDP JSON packet"
        " to 127.0.0.1:<port> after every env.step(). Off by default. Intended to feed a separate,"
        " out-of-process real-robot bridge (see lerobot_openarm/mirror_bridge.py) -- this process never"
        " talks to hardware directly."
    ),
)
parser.add_argument(
    "--mirror_udp_host",
    type=str,
    default="127.0.0.1",
    help="Destination host for --mirror_udp_port. Defaults to loopback; only change this if you know why.",
)
parser.add_argument(
    "--mirror_feedback_port",
    type=int,
    default=0,
    help=(
        "If nonzero, listen on 127.0.0.1:<port> for UDP JSON feedback packets from the real-robot"
        " bridge (see lerobot_openarm/mirror_bridge.py's --feedback-port) carrying its ACTUAL current"
        " joint positions (already inverse-mapped back to sim joint names/radians). If set, a"
        " sim-vs-real comparison plot is saved to the current directory when this script exits."
        " Off by default -- this process still never talks to hardware directly, it only listens for"
        " numbers the bridge process chooses to send back."
    ),
)
parser.add_argument(
    "--dump_joint_order",
    type=str,
    default=None,
    help=(
        "If set, write the OpenArm joint name ordering (matching the column order of"
        " states/articulation/robot/joint_position in the exported HDF5) to this JSON path once at"
        " startup. Needed to replay a recorded episode on real hardware without requiring IsaacLab"
        " installed in the hardware-control environment -- see lerobot_openarm/replay_sim_dataset.py."
    ),
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import logging
import os
import re
import socket
import threading
import time

import gymnasium as gym
import torch

import matplotlib
matplotlib.use("Agg")  # headless -- this process only ever saves a PNG, never shows a window
import matplotlib.pyplot as plt

import omni.ui as ui

from isaaclab.envs.ui import EmptyWindow
from isaaclab.managers import DatasetExportMode

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


# ─── Keyboard controller ──────────────────────────────────────────────────────
# Linux XIM fires a secondary character-input RELEASE (no .name, str='w') for
# every letter key.  The try/except AttributeError in _on_event skips these so
# the delta is only modified once per physical key press/release.
class OpenArmKeyboard:
    """Robot keyboard controller for OpenArm recording.

    Translation:  W/S = ±X  |  A/D = ±Y  |  PgUp/PgDn = ±Z
    Rotation:     ↑/↓ = pitch ±Y  |  ←/→ = yaw ±Z  |  [/] = roll ±X
    Gripper:      K (toggle)
    Save/Reset:   N / R
    Arm switch:   TAB  (used by record_demos_openarm.py)
    """

    _POS_KEYS = {
        "W":           ( 1.0,  0.0,  0.0),   # EE forward
        "S":           (-1.0,  0.0,  0.0),   # EE backward
        "A":           ( 0.0,  1.0,  0.0),   # EE left
        "D":           ( 0.0, -1.0,  0.0),   # EE right
        "PAGE_UP":     ( 0.0,  0.0,  1.0),   # EE up
        "PAGE_DOWN":   ( 0.0,  0.0, -1.0),   # EE down
    }
    _ROT_KEYS = {
        "UP":            ( 0.0,  1.0,  0.0),   # pitch +
        "DOWN":          ( 0.0, -1.0,  0.0),   # pitch -
        "LEFT":          ( 0.0,  0.0,  1.0),   # yaw +
        "RIGHT":         ( 0.0,  0.0, -1.0),   # yaw -
        "LEFT_BRACKET":  ( 1.0,  0.0,  0.0),   # roll +
        "RIGHT_BRACKET": (-1.0,  0.0,  0.0),   # roll -
    }

    def __init__(self, pos_sensitivity: float = 0.05, rot_sensitivity: float = 0.1, sim_device: str = "cpu"):
        import carb.input as ci
        import omni.appwindow
        import numpy as np

        self._pos_sensitivity = pos_sensitivity
        self._rot_sensitivity = rot_sensitivity
        self._sim_device = sim_device
        self._np = np

        self._delta_pos = np.zeros(3, dtype=np.float64)
        self._delta_rot = np.zeros(3, dtype=np.float64)
        self._close_gripper = False
        self._additional_callbacks: dict = {}

        import weakref
        appwindow = omni.appwindow.get_default_app_window()
        self._keyboard = appwindow.get_keyboard()
        self._ci = ci.acquire_input_interface()
        self._sub = self._ci.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *_, obj=weakref.proxy(self): obj._on_event(event),
        )

    def __del__(self):
        try:
            self._ci.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
        except Exception:
            pass

    def reset(self):
        self._delta_pos[:] = 0.0
        self._delta_rot[:] = 0.0
        self._close_gripper = False

    def clear_deltas(self):
        """Like `reset()` but leaves `_close_gripper` untouched -- for clearing stray
        position/rotation deltas accumulated while keys were held during a non-teleop
        interlude (e.g. the ramp-to-rest test), without forcing the gripper open."""
        self._delta_pos[:] = 0.0
        self._delta_rot[:] = 0.0

    def add_callback(self, key: str, func):
        self._additional_callbacks[key] = func

    def advance(self) -> torch.Tensor:
        from scipy.spatial.transform import Rotation
        rot_vec = Rotation.from_euler("XYZ", self._delta_rot).as_rotvec()
        cmd = self._np.concatenate([self._delta_pos, rot_vec])
        cmd = self._np.append(cmd, -1.0 if self._close_gripper else 1.0)
        return torch.tensor(cmd, dtype=torch.float32, device=self._sim_device)

    def _on_event(self, event) -> bool:
        import carb.input as ci

        # Use event.input.name directly — same as Se3Keyboard.
        # carb.input.KeyboardInput enum members have .name == "UP", "PAGE_UP", etc.
        try:
            name = event.input.name
        except AttributeError:
            return True

        if event.type == ci.KeyboardEventType.KEY_PRESS:
            if name == "K":
                self._close_gripper = not self._close_gripper
            elif name in self._POS_KEYS:
                self._delta_pos += self._np.array(self._POS_KEYS[name]) * self._pos_sensitivity
            elif name in self._ROT_KEYS:
                self._delta_rot += self._np.array(self._ROT_KEYS[name]) * self._rot_sensitivity

            if name in self._additional_callbacks:
                self._additional_callbacks[name]()

        elif event.type == ci.KeyboardEventType.KEY_RELEASE:
            if name in self._POS_KEYS:
                self._delta_pos -= self._np.array(self._POS_KEYS[name]) * self._pos_sensitivity
            elif name in self._ROT_KEYS:
                self._delta_rot -= self._np.array(self._ROT_KEYS[name]) * self._rot_sensitivity

        return True

try:
    import isaaclab_mimic.envs  # noqa: F401
    from isaaclab_mimic.ui.instruction_display import InstructionDisplay, show_subtask_instructions
    HAS_MIMIC = True
except ImportError:
    HAS_MIMIC = False

    class InstructionDisplay:
        def __init__(self, xr=False):
            pass
        def show_demo(self, text):
            pass
        def set_labels(self, *args):
            pass

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg

logger = logging.getLogger(__name__)

# ─── Action space layout ──────────────────────────────────────────────────────
# Must match the order fields are inserted into the env's ActionsCfg:
#   arm_action        (left IK,  6D) → indices [0:6]
#   gripper_action    (left bin, 1D) → index  [6]
#   right_arm_action  (right IK, 6D) → indices [7:13]
#   right_gripper_action (right bin, 1D) → index [13]
LEFT_IK_SLICE    = slice(0, 6)
LEFT_GRP_IDX     = 6
RIGHT_IK_SLICE   = slice(7, 13)
RIGHT_GRP_IDX    = 13
TOTAL_ACTION_DIM = 14


class VRDualArmTeleop:
    """Bimanual VR teleop device fed by a UDP JSON side-channel from the Dora ROS 2
    bridge (nodes/dora-openarm-ros2-bridge/bridge.py's --vr-udp-port in the
    dora-openarm-data-collection repo).

    Isaac Lab's rclpy can't be imported here directly: this conda env is Python 3.11,
    but ROS 2 Humble's rclpy C extension is only built for the system Python 3.10, so
    the bridge process (which already runs under the matching 3.10 venv to publish the
    ROS topics) also fire-and-forget UDP-broadcasts the same data as JSON -- this class
    is the receiving end, not a ROS node itself.

    Wire format per packet (see bridge.py's VrUdpBroadcaster):
        {"t": float,
         "pose_right": [x,y,z,qw,qx,qy,qz] | null, "pose_left": [...] | null,
         "pose_reference": [...] | null,
         "gripper_right": float | null, "gripper_left": float | null}
    A null field means "no VR data received yet for this field" -- distinct from a
    real zero pose -- so a stale/never-populated arm/gripper is left at its previous
    commanded state rather than snapping to the origin.

    Unlike OpenArmKeyboard (which accumulates small deltas from held keys), the task's
    DifferentialIKControllerCfg is relative-mode (use_relative_mode=True) but the VR
    source gives an ABSOLUTE target pose. So every step this recomputes
    delta = target_pose_from_VR - current_live_EE_pose (in the robot base frame, via
    isaaclab.utils.math.compute_pose_error -- the exact inverse of the IK controller's
    own apply_delta_pose) rather than accumulating anything itself. That makes the arm
    continuously track the VR controller's live absolute pose.

    `quat_offset` (wxyz, default identity) is left-multiplied onto every incoming target
    orientation, in the robot base frame, before the pose error is computed. It exists
    because dora-openarm-vr's quest_receiver.py maps raw controller poses into the robot
    "arm_origin" frame using a _FRAME_ROT/r_fix pair that was hand-tuned against the
    MuJoCo viewer's axis convention -- if Isaac Sim's robot base frame differs from
    MuJoCo's arm_origin convention, this corrects the residual fixed rotation without
    touching the shared dora pipeline (which must keep working for MuJoCo too).

    Both arms are driven every step (true bimanual) -- there is no "active arm"/TAB
    here, unlike OpenArmKeyboard's single-arm scheme.
    """

    # Raw gripper values arrive as dora-openarm-kinematics' trigger-mapped joint angle
    # (_map_trigger_to_gripper): right in [-0.785, 0] rad, left in [0, 0.785] rad, with
    # trigger=0 (released) at the extreme and trigger=1 (fully squeezed) at 0. Assumed
    # here that "released" = open and "squeezed" = closed (typical VR grip ergonomics)
    # -- unverified against the real hardware convention. If the sim gripper opens/closes
    # backwards from what you're doing with the trigger, flip the sign in _gripper_raw_to_cmd.
    GRIPPER_RAW_RANGE = 0.785

    def __init__(
        self,
        robot,
        udp_host: str,
        udp_port: int,
        sim_device: str,
        max_pos_step: float = 0.01,
        max_rot_step: float = 0.05,
        quat_offset: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    ):
        self._robot = robot
        self._sim_device = sim_device
        self._max_pos_step = max_pos_step
        self._max_rot_step = max_rot_step
        self._quat_offset = torch.tensor([list(quat_offset)], dtype=torch.float32, device=sim_device)

        right_ids, right_names = robot.find_bodies("openarm_right_ee_tcp")
        left_ids, left_names = robot.find_bodies("openarm_left_ee_tcp")
        if len(right_ids) != 1 or len(left_ids) != 1:
            raise RuntimeError(
                "VRDualArmTeleop expected exactly one match each for"
                f" 'openarm_right_ee_tcp'/'openarm_left_ee_tcp', got right={right_names},"
                f" left={left_names}."
            )
        self._right_body_idx = right_ids[0]
        self._left_body_idx = left_ids[0]

        self._lock = threading.Lock()
        self._latest: dict = {}
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((udp_host, udp_port))
        self._sock.settimeout(0.5)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        self._additional_callbacks: dict = {}
        self._setup_keyboard()  # R/N/T only -- no WASD, no arm-switch

        print(f"[VR TELEOP] Listening for Dora bridge UDP JSON on {udp_host}:{udp_port}")
        if not torch.allclose(self._quat_offset, torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=sim_device)):
            print(f"[VR TELEOP] Applying quat_offset (wxyz) = {quat_offset} to every incoming target orientation")

    def _run(self):
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                packet = json.loads(data.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            with self._lock:
                self._latest = packet

    def _setup_keyboard(self):
        import carb.input as ci
        import omni.appwindow
        import weakref

        appwindow = omni.appwindow.get_default_app_window()
        self._keyboard = appwindow.get_keyboard()
        self._ci = ci.acquire_input_interface()
        self._sub = self._ci.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *_, obj=weakref.proxy(self): obj._on_key_event(event),
        )

    def _on_key_event(self, event) -> bool:
        import carb.input as ci

        try:
            name = event.input.name
        except AttributeError:
            return True
        if event.type == ci.KeyboardEventType.KEY_PRESS and name in self._additional_callbacks:
            self._additional_callbacks[name]()
        return True

    def __del__(self):
        self._stop.set()
        try:
            self._sock.close()
        except Exception:
            pass
        try:
            self._ci.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
        except Exception:
            pass

    def add_callback(self, key: str, func):
        self._additional_callbacks[key] = func

    def reset(self):
        pass

    def clear_deltas(self):
        pass

    def _gripper_raw_to_cmd(self, raw: float) -> float:
        cmd = (2.0 * abs(raw) / self.GRIPPER_RAW_RANGE) - 1.0
        return max(-1.0, min(1.0, cmd))

    def _current_ee_pose_b(self, body_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Live EE (pos, quat wxyz) in the robot base/root frame, batch size 1."""
        from isaaclab.utils.math import subtract_frame_transforms

        ee_pos_w = self._robot.data.body_pos_w[:1, body_idx]
        ee_quat_w = self._robot.data.body_quat_w[:1, body_idx]
        root_pos_w = self._robot.data.root_pos_w[:1]
        root_quat_w = self._robot.data.root_quat_w[:1]
        return subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)

    def _arm_delta(self, body_idx: int, target_pose: list | None) -> tuple[torch.Tensor, bool]:
        """Returns (delta6, valid) for one arm -- delta6 is [dx,dy,dz, axis-angle rx,ry,rz]."""
        from isaaclab.utils.math import compute_pose_error, quat_mul

        if target_pose is None:
            return torch.zeros(6, dtype=torch.float32, device=self._sim_device), False

        ee_pos_b, ee_quat_b = self._current_ee_pose_b(body_idx)
        target_pos_b = torch.tensor([target_pose[0:3]], dtype=torch.float32, device=self._sim_device)
        target_quat_b = torch.tensor(
            [[target_pose[3], target_pose[4], target_pose[5], target_pose[6]]],
            dtype=torch.float32,
            device=self._sim_device,
        )
        target_quat_b = quat_mul(self._quat_offset, target_quat_b)

        pos_err, rot_err = compute_pose_error(
            ee_pos_b, ee_quat_b, target_pos_b, target_quat_b, rot_error_type="axis_angle"
        )
        pos_err = torch.clamp(pos_err[0], -self._max_pos_step, self._max_pos_step)
        rot_err = torch.clamp(rot_err[0], -self._max_rot_step, self._max_rot_step)

        return torch.cat([pos_err, rot_err]), True

    def build_dual_action(
        self, left_gripper_state: float, right_gripper_state: float
    ) -> tuple[torch.Tensor, float, float]:
        """Build the full 14D action vector directly from the latest VR packet, driving
        both arms simultaneously. Any arm/gripper missing fresh VR data keeps its
        previous commanded state (zero IK delta -- i.e. hold position -- for a missing
        arm; unchanged binary command for a missing gripper).
        """
        with self._lock:
            packet = dict(self._latest)

        left_delta, left_valid = self._arm_delta(self._left_body_idx, packet.get("pose_left"))
        right_delta, right_valid = self._arm_delta(self._right_body_idx, packet.get("pose_right"))

        gripper_left_raw = packet.get("gripper_left")
        gripper_right_raw = packet.get("gripper_right")
        if gripper_left_raw is not None:
            left_gripper_state = self._gripper_raw_to_cmd(gripper_left_raw)
        if gripper_right_raw is not None:
            right_gripper_state = self._gripper_raw_to_cmd(gripper_right_raw)

        full = torch.zeros(TOTAL_ACTION_DIM, dtype=torch.float32, device=self._sim_device)
        if left_valid:
            full[LEFT_IK_SLICE] = left_delta
        full[LEFT_GRP_IDX] = left_gripper_state
        if right_valid:
            full[RIGHT_IK_SLICE] = right_delta
        full[RIGHT_GRP_IDX] = right_gripper_state

        return full, left_gripper_state, right_gripper_state


class JointMirrorBroadcaster:
    """Best-effort UDP broadcaster of the robot's current joint positions.

    Fire-and-forget by design: never blocks and never raises into the sim loop.
    A separate, out-of-process bridge (e.g. lerobot_openarm/mirror_bridge.py) is
    responsible for everything hardware-related, including deciding what to do
    about stale/missing packets. This class does not know a real robot exists.
    """

    JOINT_NAME_PATTERNS = [
        r"openarm_left_joint[1-7]",
        r"openarm_right_joint[1-7]",
        r"openarm_left_finger_joint.*",
        r"openarm_right_finger_joint.*",
    ]

    @classmethod
    def resolve_mirror_joint_indices(cls, robot) -> tuple[list[int], list[str]]:
        """Return (indices, names) of the OpenArm joints within robot.data.joint_names,
        in the SAME order they appear in robot.data.joint_pos -- i.e. the same column
        order recorded into the HDF5 dataset's states/articulation/robot/joint_position.
        """
        all_names = robot.data.joint_names
        pattern = re.compile("|".join(f"(?:{p})" for p in cls.JOINT_NAME_PATTERNS))
        indices = [i for i, name in enumerate(all_names) if pattern.fullmatch(name)]
        names = [all_names[i] for i in indices]
        return indices, names

    WIDTH_PRINT_PERIOD_S = 0.5  # throttle -- printing every step at 30Hz would flood the console

    # Matches BinaryJointPositionActionCfg's open_command_expr/close_command_expr for the
    # finger joints in stack_joint_pos_env_cfg.py (both arms use the same values).
    GRIPPER_OPEN_VAL = 0.044
    GRIPPER_CLOSED_VAL = 0.0

    def __init__(self, robot, host: str, port: int):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._addr = (host, port)
        self._indices, self._names = self.resolve_mirror_joint_indices(robot)
        self._robot = robot
        self._seq = 0
        # (t, joints, eef_left, eef_right) per sample -- eef_* is [x,y,z,qw,qx,qy,qz] in the
        # robot base frame (same convention VRDualArmTeleop._current_ee_pose_b uses), or None
        # if the ee_tcp bodies aren't found. No gripper-width ("mm") field: MuJoCo's
        # openarm_{side}_finger_joint1 is a revolute joint in radians in this model, not the
        # 0..0.044m prismatic joint Isaac's OpenArm gripper uses, so there's no valid shared mm
        # conversion across the two -- finger_joint1's raw qpos is already compared correctly,
        # in each side's own native units, via the `joints` dict above.
        self._history: list[tuple[float, dict, list | None, list | None]] = []
        self._last_width_print_t = 0.0

        right_ids, _ = robot.find_bodies("openarm_right_ee_tcp")
        left_ids, _ = robot.find_bodies("openarm_left_ee_tcp")
        self._right_ee_body_idx = right_ids[0] if len(right_ids) == 1 else None
        self._left_ee_body_idx = left_ids[0] if len(left_ids) == 1 else None

        print(f"[MIRROR] Broadcasting {len(self._names)} joints to {host}:{port} -> {self._names}")

    def _ee_pose_b(self, body_idx: int | None) -> list[float] | None:
        """Live EE (pos, quat wxyz) in the robot base/root frame -- same convention
        VRDualArmTeleop._current_ee_pose_b uses, so this is directly comparable to
        the MuJoCo-side IsaacCompareBridge's arm_origin-relative EEF pose."""
        if body_idx is None:
            return None
        from isaaclab.utils.math import subtract_frame_transforms

        ee_pos_w = self._robot.data.body_pos_w[:1, body_idx]
        ee_quat_w = self._robot.data.body_quat_w[:1, body_idx]
        root_pos_w = self._robot.data.root_pos_w[:1]
        root_quat_w = self._robot.data.root_quat_w[:1]
        pos_b, quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        return [*pos_b[0].tolist(), *quat_b[0].tolist()]

    def broadcast(self, left_gripper_state: float | None = None, right_gripper_state: float | None = None):
        """Broadcast the robot's mirrored joint state.

        Arm joints use the actual/measured position (`robot.data.joint_pos`) -- that's the real
        kinematic pose and should be mirrored as-is. Finger joints, if a `*_gripper_state` is
        given, are overridden to the fixed commanded open/close target instead of the measured
        position.

        Why: sim's cube and gripper fingers are rigid bodies, so once they contact each other the
        measured finger joint position stalls almost exactly at the object's geometric surface --
        that's correct rigid-body physics, not a bug. The real gripper's fingertip pads are
        compliant and keep squeezing past that same contact point to reach a firm grip (observed:
        sim/real agreed closely at first contact, ~52.6/53.4mm, but the real gripper needed to
        reach ~43.4mm for a grip that actually holds). Mirroring the *measured* sim position was
        capping the real robot's target at sim's rigid stopping point, never letting it command
        the real gripper to keep squeezing further. Mirroring the fixed open/close *target*
        instead lets the real hardware's own compliant pads decide how far they actually close,
        independent of whatever sim's specific rigid object happened to stop the fingers at.
        """
        joint_pos = self._robot.data.joint_pos[0, self._indices].tolist()
        joints = dict(zip(self._names, joint_pos))

        if left_gripper_state is not None:
            left_val = self.GRIPPER_OPEN_VAL if left_gripper_state > 0 else self.GRIPPER_CLOSED_VAL
            for name in joints:
                if name.startswith("openarm_left_finger_joint"):
                    joints[name] = left_val
        if right_gripper_state is not None:
            right_val = self.GRIPPER_OPEN_VAL if right_gripper_state > 0 else self.GRIPPER_CLOSED_VAL
            for name in joints:
                if name.startswith("openarm_right_finger_joint"):
                    joints[name] = right_val

        eef_left = self._ee_pose_b(self._left_ee_body_idx)
        eef_right = self._ee_pose_b(self._right_ee_body_idx)

        packet = {
            "seq": self._seq,
            "t": time.time(),
            "joints": joints,
            "eef_left": eef_left,
            "eef_right": eef_right,
        }
        self._seq += 1
        self._history.append((packet["t"], joints, eef_left, eef_right))
        try:
            self._sock.sendto(json.dumps(packet).encode("utf-8"), self._addr)
        except OSError:
            pass  # best-effort only -- never let a networking hiccup break recording

        if packet["t"] - self._last_width_print_t >= self.WIDTH_PRINT_PERIOD_S:
            self._last_width_print_t = packet["t"]
            # Both finger joints are prismatic, 0.0 (closed) .. 0.044m (open), moving symmetrically
            # outward -- see openarm_description.urdf finger_joint1/2 limits and mimic tag. Total
            # gripper opening width is the sum of both fingers' travel from the closed position.
            # Isaac-only console printout (MuJoCo's finger_joint1 isn't the same joint type/units,
            # see the comment on self._history above -- this mm figure isn't sent for comparison).
            left_mm = joints.get("openarm_left_finger_joint1", 0.0) * 2000.0
            right_mm = joints.get("openarm_right_finger_joint1", 0.0) * 2000.0
            print(f"[SIM GRIPPER]  left={left_mm:5.1f}mm  right={right_mm:5.1f}mm")

    def history(self) -> list[tuple[float, dict, list | None, list | None]]:
        return self._history


class JointFeedbackReceiver:
    """Best-effort UDP listener for ACTUAL state broadcast back by another process --
    either a real-robot bridge (e.g. lerobot_openarm/mirror_bridge.py's --feedback-port)
    or dora-openarm-mujoco's IsaacCompareBridge (its --isaac-feedback-port). Both send
    the same packet shape: {"t": float, "joints": {name: rad, ...}, "eef_left"/"eef_right":
    [x,y,z,qw,qx,qy,qz] | null}. Logs the full history (not just the latest packet) for a
    sim-vs-real (or sim-vs-sim) comparison plot when this script exits. Runs in a background
    thread; never blocks the sim loop. This process still never talks to hardware directly --
    it only listens for numbers a separate, out-of-process bridge chooses to send back.
    """

    def __init__(self, host: str, port: int):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((host, port))
        self._sock.settimeout(0.5)
        self._lock = threading.Lock()
        self._log: list[tuple[float, dict, list | None, list | None]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[MIRROR] Listening for real-robot/MuJoCo feedback on {host}:{port}")

    def _run(self):
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                packet = json.loads(data.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            with self._lock:
                self._log.append((
                    packet["t"],
                    packet["joints"],
                    packet.get("eef_left"),
                    packet.get("eef_right"),
                ))

    def history(self) -> list[tuple[float, dict, list | None, list | None]]:
        with self._lock:
            return list(self._log)

    def stop(self):
        self._stop.set()
        self._sock.close()


class RateLimiter:
    def __init__(self, hz):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env):
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()
        self.last_time += self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def run_ramp_to_rest_test(env, duration: float = 4.0, rate_hz: float = 50.0, stop_requested: dict | None = None):
    """Replay reset_to_rest_pose.py's exact real-robot ramp (straight-line joint-space
    interpolation from the current pose to the default rest pose) inside Isaac Sim, so you can
    watch the viewport for the arm/gripper intersecting the pad -- without needing the real
    hardware to test it.

    This bypasses the env's normal action interface (IK deltas + binary gripper) entirely and
    drives the robot's joints directly, matching lerobot_openarm/sim_bridge_common.py's
    `ramp_to()`:  cmd = start + alpha * (end - start), alpha going 0->1 over `duration` seconds.
    `env.step()` is not called during the ramp (nothing is recorded, no observations/rewards/
    terminations run) -- this is a visual diagnostic only. Whatever pose the arm is in when you
    trigger this becomes the ramp's start; the target is the task's own default joint pose
    (`robot.data.default_joint_pos`), i.e. what a normal `env.reset()` would put it in.
    """
    robot = env.unwrapped.scene["robot"]
    joint_ids, names = JointMirrorBroadcaster.resolve_mirror_joint_indices(robot)

    start_pos = robot.data.joint_pos[0, joint_ids].clone()
    end_pos = robot.data.default_joint_pos[0, joint_ids].clone()

    print(f"[RAMP TEST] Ramping {len(names)} joints to the default rest pose over {duration}s...")
    print("[RAMP TEST] Watch the viewport -- Ctrl+C aborts and leaves the arm wherever it stopped.")

    steps = max(1, int(duration * rate_hz))
    dt = 1.0 / rate_hz
    sim_dt = env.unwrapped.sim.get_physics_dt()
    for i in range(1, steps + 1):
        if stop_requested is not None and stop_requested.get("flag"):
            print("[RAMP TEST] Aborted.")
            break
        step_start = time.time()
        alpha = i / steps
        target = start_pos + alpha * (end_pos - start_pos)
        target_batched = target.unsqueeze(0).expand(env.unwrapped.num_envs, -1)
        robot.set_joint_position_target(target_batched, joint_ids=joint_ids)
        robot.write_data_to_sim()
        env.unwrapped.sim.step()
        robot.update(sim_dt)
        env.unwrapped.sim.render()
        remaining = dt - (time.time() - step_start)
        if remaining > 0:
            time.sleep(remaining)
    else:
        print("[RAMP TEST] Done -- arm should now be at the default rest pose.")


def build_single_arm_action(
    teleop_7d: torch.Tensor,
    gripper_state: float,
    device: str,
) -> tuple[torch.Tensor, float]:
    """Pass-through for 7D single-arm tasks (IK 6D + gripper 1D)."""
    full = torch.zeros(7, dtype=torch.float32, device=device)
    full[:6] = teleop_7d[:6]
    full[6] = teleop_7d[6].item()
    return full, teleop_7d[6].item()


def build_dual_arm_action(
    teleop_7d: torch.Tensor,
    active_arm: str,
    left_gripper_state: float,
    right_gripper_state: float,
    device: str,
) -> tuple[torch.Tensor, float, float]:
    """Route 7D teleop output to the correct arm in the 14D action vector.

    Args:
        teleop_7d: Shape (7,) tensor [dx,dy,dz,drx,dry,drz,gripper].
        active_arm: "left" or "right".
        left_gripper_state: Last gripper command for left arm (±1.0).
        right_gripper_state: Last gripper command for right arm (±1.0).
        device: Torch device string.

    Returns:
        (full_action_14d, updated_left_gripper_state, updated_right_gripper_state)
    """
    full = torch.zeros(TOTAL_ACTION_DIM, dtype=torch.float32, device=device)
    gripper_cmd = teleop_7d[6].item()

    if active_arm == "left":
        full[LEFT_IK_SLICE] = teleop_7d[:6]
        full[LEFT_GRP_IDX] = gripper_cmd
        full[RIGHT_GRP_IDX] = right_gripper_state  # keep right gripper unchanged
        return full, gripper_cmd, right_gripper_state
    else:
        full[LEFT_GRP_IDX] = left_gripper_state    # keep left gripper unchanged
        full[RIGHT_IK_SLICE] = teleop_7d[:6]
        full[RIGHT_GRP_IDX] = gripper_cmd
        return full, left_gripper_state, gripper_cmd


def main():
    rate_limiter = RateLimiter(args_cli.step_hz)

    # ── Output dirs ──────────────────────────────────────────────────────────
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # ── Env config ────────────────────────────────────────────────────────────
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env_cfg.env_name = args_cli.task.split(":")[-1]
    except Exception as e:
        logger.error(f"Failed to parse env config: {e}")
        return

    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None

    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir if output_dir else "."
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    # ── Create env ────────────────────────────────────────────────────────────
    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        return

    sim_device = args_cli.device

    # ── Real-robot mirror broadcaster (opt-in, off by default) ─────────────────
    mirror_broadcaster = None
    if args_cli.mirror_udp_port:
        mirror_broadcaster = JointMirrorBroadcaster(
            robot=env.scene["robot"], host=args_cli.mirror_udp_host, port=args_cli.mirror_udp_port
        )

    # ── Real-robot feedback listener, for a sim-vs-real plot on exit (opt-in) ──
    feedback_receiver = None
    if args_cli.mirror_feedback_port:
        feedback_receiver = JointFeedbackReceiver(host=args_cli.mirror_udp_host, port=args_cli.mirror_feedback_port)

        # Ctrl+C in this app appears to tear the process down directly rather than
        # raising a normal Python KeyboardInterrupt that falls through to code after
        # the main loop -- contextlib.suppress(KeyboardInterrupt) below never got a
        # chance to matter, the code after env.close() never ran, and no plot was
        # ever saved. atexit hooks Python's actual interpreter shutdown instead, which
        # survives that. Guarded so it can't double-run if normal flow ALSO reaches
        # the equivalent call later.
        plot_state = {"saved": False}

        def _save_plot_once():
            if plot_state["saved"]:
                return
            plot_state["saved"] = True
            feedback_receiver.stop()
            save_sim_vs_real_plot(mirror_broadcaster, feedback_receiver)

        atexit.register(_save_plot_once)

    # ── Joint-order manifest for offline replay-on-hardware (opt-in, off by default) ──
    if args_cli.dump_joint_order:
        _, joint_order_names = JointMirrorBroadcaster.resolve_mirror_joint_indices(env.scene["robot"])
        with open(args_cli.dump_joint_order, "w") as f:
            json.dump({"joint_order": joint_order_names}, f, indent=2)
        print(f"[INFO] Wrote joint order ({len(joint_order_names)} joints) to {args_cli.dump_joint_order}")

    # ── Detect action space ────────────────────────────────────────────────────
    total_action_dim = env.action_manager.total_action_dim
    is_dual_arm = (total_action_dim == TOTAL_ACTION_DIM)
    print(f"[INFO] Action dim: {total_action_dim}  ({'dual-arm' if is_dual_arm else 'single-arm'})")

    use_vr_teleop = args_cli.teleop_device == "vr_ros2"
    if use_vr_teleop and not is_dual_arm:
        logger.error("--teleop_device vr_ros2 requires a dual-arm (14D action) task.")
        return

    # ── Teleop device ─────────────────────────────────────────────────────────
    if use_vr_teleop:
        teleop = VRDualArmTeleop(
            robot=env.scene["robot"],
            udp_host=args_cli.vr_udp_host,
            udp_port=args_cli.vr_udp_port,
            sim_device=sim_device,
            max_pos_step=args_cli.vr_max_pos_step,
            max_rot_step=args_cli.vr_max_rot_step,
            quat_offset=tuple(args_cli.vr_quat_offset),
        )
    else:
        # Use OpenArmKeyboard (arrow keys + I/O) to avoid Isaac Sim viewport
        # gizmo conflicts with W/A/S/D/Q/E.
        teleop = OpenArmKeyboard(pos_sensitivity=0.05, rot_sensitivity=0.1, sim_device=sim_device)

    # ── State ─────────────────────────────────────────────────────────────────
    active_arm = "left"
    left_gripper_state = 1.0   # 1.0 = open, -1.0 = close
    right_gripper_state = 1.0
    should_reset = False
    running = True
    demo_count = 0
    success_step_count = 0
    ramp_test_requested = False

    def reset_episode():
        nonlocal should_reset
        should_reset = True
        print("Reset requested")

    def request_ramp_test():
        nonlocal ramp_test_requested
        ramp_test_requested = True

    def save_episode():
        nonlocal should_reset
        env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
        env.recorder_manager.set_success_to_episodes(
            [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
        )
        env.recorder_manager.export_episodes([0])
        print("Episode saved!")
        should_reset = True

    def toggle_arm():
        nonlocal active_arm
        active_arm = "right" if active_arm == "left" else "left"
        print(f"[ARM SWITCH] Active arm: {active_arm.upper()}")
        # Update the on-screen label
        _refresh_label()

    def _refresh_label():
        if use_vr_teleop:
            label_text = f"Bimanual VR teleop  |  Demos: {demo_count}"
        else:
            arm_indicator = "◄ LEFT" if active_arm == "left" else "RIGHT ►"
            label_text = f"Active arm: {arm_indicator}  |  Demos: {demo_count}"
        try:
            instruction_display.show_demo(label_text)
        except Exception:
            pass

    teleop.add_callback("R", reset_episode)
    teleop.add_callback("N", save_episode)
    if not use_vr_teleop:
        teleop.add_callback("TAB", toggle_arm)
    teleop.add_callback("T", request_ramp_test)

    # ── UI ────────────────────────────────────────────────────────────────────
    instruction_display = InstructionDisplay(xr=False)
    if HAS_MIMIC:
        window = EmptyWindow(env, "Instruction")
        with window.ui_window_elements["main_vstack"]:
            demo_label = ui.Label(f"Active arm: LEFT  |  Demos: {demo_count}")
            arm_label = ui.Label("")
            instruction_display.set_labels(arm_label, demo_label)

    # ── Initial reset ─────────────────────────────────────────────────────────
    env.sim.reset()
    env.reset()
    teleop.reset()

    mode_str = "Dual-Arm" if is_dual_arm else "Single-Arm"
    print(f"\n=== OpenArm {mode_str} Recording ===")
    if use_vr_teleop:
        print("  VR (bimanual) — both arms + grippers driven live from the Dora UDP bridge")
    else:
        if is_dual_arm:
            print("  TAB        — switch active arm (left ↔ right)")
        print("  K          — toggle gripper open/close")
        print("  W/S        — EE forward / backward  (+x/-x)")
        print("  A/D        — EE left / right         (+y/-y)")
        print("  PgUp/PgDn  — EE up / down            (+z/-z)")
        print("  ↑/↓        — pitch ±  |  ←/→ — yaw ±  |  [/] — roll ±")
    print("  N          — save episode as success")
    print("  R          — discard & reset episode")
    print("  T          — ramp current pose to rest (matches reset_to_rest_pose.py) and watch")
    print("               the viewport for a pad collision -- doesn't record, doesn't reset")
    if is_dual_arm and not use_vr_teleop:
        print(f"\nActive arm: LEFT\n")
    else:
        print()

    # Installed here (after AppLauncher/simulation_app setup, which may install its own
    # SIGINT handler) so this one takes effect for Ctrl+C from here on. It only sets a
    # flag rather than doing any work itself -- signal handlers can fire between any two
    # bytecode instructions, so anything more (file I/O, plotting) belongs in the main
    # loop's normal execution, not here. This exists because Ctrl+C was observed to tear
    # this app down before normal Python cleanup code (even atexit) got a chance to run.
    stop_requested = {"flag": False}
    signal.signal(signal.SIGINT, lambda signum, frame: stop_requested.__setitem__("flag", True))

    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        while simulation_app.is_running() and not stop_requested["flag"]:
            if ramp_test_requested:
                ramp_test_requested = False
                run_ramp_to_rest_test(env, stop_requested=stop_requested)
                teleop.clear_deltas()  # clear deltas accumulated while keys were held during the ramp
                continue

            # Build action vector sized to match the task's action space
            if use_vr_teleop:
                full_action, left_gripper_state, right_gripper_state = teleop.build_dual_action(
                    left_gripper_state=left_gripper_state,
                    right_gripper_state=right_gripper_state,
                )
            elif is_dual_arm:
                teleop_7d = teleop.advance()
                full_action, left_gripper_state, right_gripper_state = build_dual_arm_action(
                    teleop_7d=teleop_7d,
                    active_arm=active_arm,
                    left_gripper_state=left_gripper_state,
                    right_gripper_state=right_gripper_state,
                    device=sim_device,
                )
            else:
                teleop_7d = teleop.advance()
                full_action, left_gripper_state = build_single_arm_action(
                    teleop_7d=teleop_7d,
                    gripper_state=left_gripper_state,
                    device=sim_device,
                )
            actions = full_action.unsqueeze(0).expand(env.num_envs, -1)

            if running:
                obs, *_ = env.step(actions)
                if mirror_broadcaster is not None:
                    mirror_broadcaster.broadcast(
                        left_gripper_state=left_gripper_state,
                        right_gripper_state=right_gripper_state if is_dual_arm else None,
                    )

            # Success check
            if success_term is not None:
                if bool(success_term.func(env, **success_term.params)[0]):
                    success_step_count += 1
                    if success_step_count >= args_cli.num_success_steps:
                        env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                        env.recorder_manager.set_success_to_episodes(
                            [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                        )
                        env.recorder_manager.export_episodes([0])
                        print("Auto-success condition met!")
                        should_reset = True
                else:
                    success_step_count = 0

            # Update demo counter label
            if env.recorder_manager.exported_successful_episode_count > demo_count:
                demo_count = env.recorder_manager.exported_successful_episode_count
                print(f"Total demos recorded: {demo_count}")
                _refresh_label()

            # Check exit condition
            if args_cli.num_demos > 0 and demo_count >= args_cli.num_demos:
                print(f"All {demo_count} demos recorded. Exiting.")
                break

            # Handle reset
            if should_reset:
                print("Resetting environment...")
                env.sim.reset()
                env.recorder_manager.reset()
                env.reset()
                teleop.reset()
                success_step_count = 0
                should_reset = False
                # Reset gripper states to open
                left_gripper_state = 1.0
                right_gripper_state = 1.0
                _refresh_label()
                print(f"Ready. Active arm: {active_arm.upper()}")

            if env.sim.is_stopped():
                break

            rate_limiter.sleep(env)

    env.close()
    print(f"\nRecording done. {demo_count} successful demos saved to: {args_cli.dataset_file}")

    if feedback_receiver is not None:
        _save_plot_once()


def save_sim_vs_real_plot(mirror_broadcaster, feedback_receiver) -> None:
    """Compare the sim (Isaac) state this process broadcast against the other side's
    actual state received back via --mirror_feedback_port -- either a real-robot bridge
    (lerobot_openarm/mirror_bridge.py) or MuJoCo (dora-openarm-mujoco's
    IsaacCompareBridge, --isaac-feedback-port). Saves one PNG (per-joint angles, including
    each side's own raw gripper joint, and EEF pose tracking error) to the current working
    directory.

    History tuples from both sides are (t, joints, eef_left, eef_right) -- see
    JointMirrorBroadcaster.broadcast / JointFeedbackReceiver._run.
    """
    sim_history = mirror_broadcaster.history() if mirror_broadcaster is not None else []
    real_history = feedback_receiver.history()
    if not sim_history or not real_history:
        print("[MIRROR] No data collected for a sim-vs-real comparison plot (one or both histories are"
              " empty) -- skipping. Did the other side's --feedback-port/--isaac-feedback-port match?")
        return

    import numpy as np
    from scipy.spatial.transform import Rotation

    # Minimum y-axis half-range per category -- prevents matplotlib's auto-scaling from
    # zooming into sensor noise or a small steady-state offset and making it look like a
    # large gap. Chosen relative to each category's own meaningful scale: 0.1 rad matches
    # the arm joints' own handshake tolerance elsewhere in this pipeline, while the gripper's
    # entire physical range is only ~0.044 rad so it gets a proportionally smaller floor.
    ARM_AXIS_TOLERANCE = 0.1
    GRIPPER_AXIS_TOLERANCE = 0.005

    t0 = sim_history[0][0]
    joint_names = list(sim_history[0][1].keys())

    def _nearest_paired_errors(eef_idx: int, max_gap_s: float = 0.2):
        """Pair each sim EEF sample with real_history's nearest-in-time sample (both
        timestamped with the same wall clock, time.time()) and return
        (t_rel, pos_err_mm, rot_err_deg) for pairs within max_gap_s of each other."""
        real_pts = [(s[0], s[eef_idx]) for s in real_history if s[eef_idx] is not None]
        if not real_pts:
            return [], [], []
        real_t = np.array([p[0] for p in real_pts])

        t_out, pos_err_mm, rot_err_deg = [], [], []
        for s in sim_history:
            sim_eef = s[eef_idx]
            if sim_eef is None:
                continue
            j = int(np.argmin(np.abs(real_t - s[0])))
            if abs(real_t[j] - s[0]) > max_gap_s:
                continue
            real_eef = real_pts[j][1]
            pos_err_mm.append(
                float(np.linalg.norm(np.array(sim_eef[:3]) - np.array(real_eef[:3]))) * 1000.0
            )
            r_sim = Rotation.from_quat([sim_eef[4], sim_eef[5], sim_eef[6], sim_eef[3]])
            r_real = Rotation.from_quat([real_eef[4], real_eef[5], real_eef[6], real_eef[3]])
            rot_err_deg.append(float(np.degrees((r_sim.inv() * r_real).magnitude())))
            t_out.append(s[0] - t0)
        return t_out, pos_err_mm, rot_err_deg

    ncols = 4
    n_joint_rows = (len(joint_names) + ncols - 1) // ncols
    nrows = n_joint_rows + 1  # + 1 row EEF error (4 panels)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for idx, name in enumerate(joint_names):
        ax = axes[idx // ncols][idx % ncols]
        sim_t = [s[0] - t0 for s in sim_history if name in s[1]]
        sim_v = [s[1][name] for s in sim_history if name in s[1]]
        real_t = [s[0] - t0 for s in real_history if name in s[1]]
        real_v = [s[1][name] for s in real_history if name in s[1]]
        ax.plot(sim_t, sim_v, label="sim", linewidth=1)
        ax.plot(real_t, real_v, label="real", linewidth=1, linestyle="--")
        ax.set_title(name, fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlabel("s", fontsize=7)
        ax.set_ylabel("rad", fontsize=7)

        all_v = sim_v + real_v
        if all_v:
            tolerance = GRIPPER_AXIS_TOLERANCE if "finger" in name else ARM_AXIS_TOLERANCE
            data_min, data_max = min(all_v), max(all_v)
            center = (data_min + data_max) / 2
            half_range = max((data_max - data_min) / 2 * 1.1, tolerance)
            ax.set_ylim(center - half_range, center + half_range)

    for idx in range(len(joint_names), n_joint_rows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    # ── EEF pose tracking error, nearest-timestamp paired (not a raw overlay --
    # position/orientation only make sense as a scalar error, unlike joint angles).
    # Gripper width isn't plotted separately here: MuJoCo's finger_joint1 in this model is a
    # revolute joint in radians, not Isaac's 0..0.044m prismatic joint, so there's no valid
    # shared "mm" conversion across the two -- finger_joint1's raw qpos is already compared
    # correctly, in each side's own native units, in the per-joint grid above. ──────────────
    err_row = n_joint_rows
    eef_panels = (
        ("eef_left pos err", 2, "pos"),
        ("eef_left rot err", 2, "rot"),
        ("eef_right pos err", 3, "pos"),
        ("eef_right rot err", 3, "rot"),
    )
    for col, (title, eef_idx, kind) in enumerate(eef_panels):
        ax = axes[err_row][col]
        t_out, pos_err_mm, rot_err_deg = _nearest_paired_errors(eef_idx)
        values = pos_err_mm if kind == "pos" else rot_err_deg
        ax.plot(t_out, values, linewidth=1, color="tab:red")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("s", fontsize=7)
        ax.set_ylabel("mm" if kind == "pos" else "deg", fontsize=7)
        if values:
            ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1.0)

    fig.suptitle("Isaac Sim vs MuJoCo/real: joints, gripper width, EEF tracking error (live session)")
    fig.tight_layout()
    out_path = os.path.join(os.getcwd(), f"sim_vs_real_{time.strftime('%Y%m%d_%H%M%S')}.png")
    fig.savefig(out_path, dpi=120)
    print(f"[MIRROR] Saved sim-vs-real comparison plot to {out_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
