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
    help="Teleop device (keyboard or spacemouse).",
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

    def __init__(self, robot, host: str, port: int):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._addr = (host, port)
        self._indices, self._names = self.resolve_mirror_joint_indices(robot)
        self._robot = robot
        self._seq = 0
        self._history: list[tuple[float, dict]] = []
        print(f"[MIRROR] Broadcasting {len(self._names)} joints to {host}:{port} -> {self._names}")

    def broadcast(self):
        joint_pos = self._robot.data.joint_pos[0, self._indices].tolist()
        joints = dict(zip(self._names, joint_pos))
        packet = {
            "seq": self._seq,
            "t": time.time(),
            "joints": joints,
        }
        self._seq += 1
        self._history.append((packet["t"], joints))
        try:
            self._sock.sendto(json.dumps(packet).encode("utf-8"), self._addr)
        except OSError:
            pass  # best-effort only -- never let a networking hiccup break recording

    def history(self) -> list[tuple[float, dict]]:
        return self._history


class JointFeedbackReceiver:
    """Best-effort UDP listener for ACTUAL joint feedback broadcast back by a
    real-robot bridge (e.g. lerobot_openarm/mirror_bridge.py's --feedback-port).
    Logs the full history (not just the latest packet) for a sim-vs-real comparison
    plot when this script exits. Runs in a background thread; never blocks the sim
    loop. This process still never talks to hardware -- it only listens for numbers
    a separate, out-of-process bridge chooses to send back.
    """

    def __init__(self, host: str, port: int):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((host, port))
        self._sock.settimeout(0.5)
        self._lock = threading.Lock()
        self._log: list[tuple[float, dict]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[MIRROR] Listening for real-robot feedback on {host}:{port}")

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
                self._log.append((packet["t"], packet["joints"]))

    def history(self) -> list[tuple[float, dict]]:
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

    # ── Teleop device ─────────────────────────────────────────────────────────
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

    def reset_episode():
        nonlocal should_reset
        should_reset = True
        print("Reset requested")

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
        arm_indicator = "◄ LEFT" if active_arm == "left" else "RIGHT ►"
        label_text = f"Active arm: {arm_indicator}  |  Demos: {demo_count}"
        try:
            instruction_display.show_demo(label_text)
        except Exception:
            pass

    teleop.add_callback("R", reset_episode)
    teleop.add_callback("N", save_episode)
    teleop.add_callback("TAB", toggle_arm)

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
    if is_dual_arm:
        print("  TAB        — switch active arm (left ↔ right)")
    print("  K          — toggle gripper open/close")
    print("  W/S        — EE forward / backward  (+x/-x)")
    print("  A/D        — EE left / right         (+y/-y)")
    print("  PgUp/PgDn  — EE up / down            (+z/-z)")
    print("  ↑/↓        — pitch ±  |  ←/→ — yaw ±  |  [/] — roll ±")
    print("  N          — save episode as success")
    print("  R          — discard & reset episode")
    if is_dual_arm:
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
            # Get 7D teleop output
            teleop_7d = teleop.advance()

            # Build action vector sized to match the task's action space
            if is_dual_arm:
                full_action, left_gripper_state, right_gripper_state = build_dual_arm_action(
                    teleop_7d=teleop_7d,
                    active_arm=active_arm,
                    left_gripper_state=left_gripper_state,
                    right_gripper_state=right_gripper_state,
                    device=sim_device,
                )
            else:
                full_action, left_gripper_state = build_single_arm_action(
                    teleop_7d=teleop_7d,
                    gripper_state=left_gripper_state,
                    device=sim_device,
                )
            actions = full_action.unsqueeze(0).expand(env.num_envs, -1)

            if running:
                obs, *_ = env.step(actions)
                if mirror_broadcaster is not None:
                    mirror_broadcaster.broadcast()

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
    """Compare the sim joint positions this process broadcast against the real
    robot's actual joint feedback received back from mirror_bridge.py, and save a
    per-joint time-series plot (PNG) to the current working directory."""
    sim_history = mirror_broadcaster.history() if mirror_broadcaster is not None else []
    real_history = feedback_receiver.history()
    if not sim_history or not real_history:
        print("[MIRROR] No data collected for a sim-vs-real comparison plot (one or both histories are"
              " empty) -- skipping. Did mirror_bridge.py have --feedback-port set to match?")
        return

    t0 = sim_history[0][0]
    joint_names = list(sim_history[0][1].keys())
    ncols = 4
    nrows = (len(joint_names) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for idx, name in enumerate(joint_names):
        ax = axes[idx // ncols][idx % ncols]
        sim_t = [t - t0 for t, j in sim_history if name in j]
        sim_v = [j[name] for t, j in sim_history if name in j]
        real_t = [t - t0 for t, j in real_history if name in j]
        real_v = [j[name] for t, j in real_history if name in j]
        ax.plot(sim_t, sim_v, label="sim", linewidth=1)
        ax.plot(real_t, real_v, label="real", linewidth=1, linestyle="--")
        ax.set_title(name, fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlabel("s", fontsize=7)
        ax.set_ylabel("rad", fontsize=7)

    for idx in range(len(joint_names), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("Sim vs real joint positions (live mirroring session)")
    fig.tight_layout()
    out_path = os.path.join(os.getcwd(), f"sim_vs_real_{time.strftime('%Y%m%d_%H%M%S')}.png")
    fig.savefig(out_path, dpi=120)
    print(f"[MIRROR] Saved sim-vs-real comparison plot to {out_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
