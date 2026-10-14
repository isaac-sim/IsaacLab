# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Insert, release, regrasp and extract an Ethernet cable in one continuous simulation.

The non-RL manager-based environment owns physics and reset. This example returns ordinary TCP
and finger targets. Measured poses, finger opening and latch clearance gate the
sequence. The managed dependencies include the Newton hydroelastic contact correction.

.. code-block:: bash

    uv run isaaclab example sharpa-rj45
    uv run --extra isaacsim --extra video isaaclab example sharpa-rj45 --viz kit --video
    uv run isaaclab example sharpa-rj45 --viz none

"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

_TUCKED_FINGERS_RAD = (1.1, 0.0, 1.4, 1.0, 0.0, 1.1, 0.0, 1.4, 1.0)
_OPEN_HAND_RAD = (
    0.70216371,
    -0.13133197,
    0.77240945,
    -0.17230791,
    0.00004912,
    0.78591669,
    -0.03844880,
    0.86681288,
    0.12026159,
    1.22023412,
    -0.13107619,
    0.37545449,
    0.70664729,
    *_TUCKED_FINGERS_RAD,
)
_SUPPORT_HAND_RAD = (
    0.84772129,
    -0.10509452,
    0.67820725,
    -0.10205317,
    0.00001000,
    0.84205546,
    -0.09530088,
    0.77374883,
    0.15160327,
    1.23633955,
    -0.17374855,
    0.41702808,
    0.50194808,
    *_TUCKED_FINGERS_RAD,
)


def main() -> int:
    """Run the bounded physical cycle; return success only after extraction."""
    from isaaclab.app import add_launcher_args

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max_steps", "--num_steps", dest="num_steps", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--settle_duration_s", type=float, default=0.0, help="Initial physical grasp hold [s].")
    parser.add_argument("--video", action="store_true", help="Record with the standard task video recorder.")
    parser.add_argument(
        "--failure_tail_s", type=float, default=0.0, help="Simulate with held targets after a fault [s]."
    )
    parser.add_argument("--output_dir", type=Path, default=Path("logs/demos/sharpa_rj45"))
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["newton_gl"])
    args = parser.parse_args()
    if args.num_steps <= 0:
        parser.error("--num_steps must be positive")
    for name in ("settle_duration_s", "failure_tail_s"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name} must be finite and nonnegative")
    return _run(args)


@dataclass(frozen=True)
class RemovalPostures:
    """Calibrated thumb-over/index-and-middle-under grasp and clip press.

    Finger positions follow the canonical action order [rad]. The palm pose is
    relative to the canonical seated connector orientation [m, XYZW].
    """

    removal_tcp_offset_w: tuple[float, float, float] = (0.06986748, -0.11711917, 0.03615930)
    removal_tcp_orientation_w: tuple[float, float, float, float] = (0.64710128, 0.39442167, -0.52808549, -0.38316730)
    removal_hand_open: tuple[float, ...] = _OPEN_HAND_RAD
    removal_hand_support: tuple[float, ...] = _SUPPORT_HAND_RAD
    removal_hand_press: tuple[float, ...] = (
        0.9432912606,
        -0.0816916769,
        0.5411537581,
        -0.0139203356,
        0.0584947575,
        *_SUPPORT_HAND_RAD[5:],
    )
    removal_index_grasp: tuple[float, ...] = (0.85330546, *_SUPPORT_HAND_RAD[6:9])
    removal_middle_grasp: tuple[float, ...] = (1.24758955, -0.18499855, *_SUPPORT_HAND_RAD[11:13])
    removal_thumb_clearance_reserve_m: float = 0.0001
    """Extra measured clip clearance required before pulling [m]."""


class CableCycleStateMachine:
    """Complete one physical cycle using TCP and finger action targets only.

    Relative poses check grasp slip. Measured capture gates insertion completion,
    measured finger opening establishes release, and clip clearance gates removal.
    The seated palm remains stationary through release and retention verification.

    Args:
        tcp_pose_w: Initial measured TCP pose [m, XYZW].
        goal_pose_w: Canonical seated connector pose [m, XYZW].
        dt: Controller interval [s].
        closed_hand: Canonical 22-joint insertion posture [rad].
        postures: Calibrated removal palm and finger targets.
        settle_duration_s: Initial stationary physical grasp hold [s].
    """

    def __init__(
        self,
        tcp_pose_w: np.ndarray,
        goal_pose_w: np.ndarray,
        dt: float,
        closed_hand: tuple[float, ...],
        *,
        postures: RemovalPostures = RemovalPostures(),
        settle_duration_s: float = 0.0,
    ):
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive.")
        if not np.isfinite(settle_duration_s) or settle_duration_s < 0.0:
            raise ValueError("settle_duration_s must be finite and nonnegative.")
        self.command = _pose(tcp_pose_w)
        self.goal = _pose(goal_pose_w)
        self.target = self.command.copy()
        self.closed_hand = np.asarray(closed_hand, dtype=np.float64)
        self.open_hand = np.asarray(postures.removal_hand_open, dtype=np.float64)
        self.support_hand = np.asarray(postures.removal_hand_support, dtype=np.float64)
        self.press_hand = np.asarray(postures.removal_hand_press, dtype=np.float64)
        self.housing_hand = np.asarray((*postures.removal_index_grasp, *postures.removal_middle_grasp))
        if any(
            hand.shape != (22,) or not np.isfinite(hand).all()
            for hand in (self.closed_hand, self.open_hand, self.support_hand, self.press_hand)
        ):
            raise ValueError("Hand targets must contain 22 finite canonical joint positions.")
        self.postures = postures
        self.dt = dt
        self.settle_duration_s = settle_duration_s
        self.phase = "SETTLE" if settle_duration_s > 0.0 else "INSERTION/HOLD_GRASP"
        self.elapsed = self.dwell = self.width = 0.0
        self.insertion_completed = False
        self.failure: str | None = None
        self.grasp_offset_m: np.ndarray | None = None
        self.grasp_rotation: Rotation | None = None
        self.carry_target: np.ndarray | None = None
        self.seated_head: np.ndarray | None = None
        self.grasp: np.ndarray | None = None
        self.settle_pose: np.ndarray | None = None
        self.press_tcp: np.ndarray | None = None
        self.withdrawal_tcp: np.ndarray | None = None
        self.withdrawal_hold_tcp: np.ndarray | None = None
        self.hand_blends = (0.0, 0.0)
        self.grip_blend = self.measured_withdrawal_m = 0.0
        self.withdrawal_progress_s = self.withdrawal_start_distance_m = self.latch_loss_s = 0.0

    @property
    def demonstration(self) -> str:
        """Active half of the sequence, including its final terminal state."""
        return "removal" if self.insertion_completed else "insertion"

    def update(
        self,
        tcp_pose_w: np.ndarray,
        head_pose_w: np.ndarray,
        *,
        latch_engaged: bool,
        latch_angle_rad: float,
        capture_margins_m: np.ndarray,
        keeper_release_margin_m: float,
        keeper_exit_margin_m: float,
        hand_joint_pos: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return TCP [m, XYZW] and canonical finger targets [rad].

        Args:
            tcp_pose_w: Measured world TCP pose [m, XYZW].
            head_pose_w: Measured world connector pose [m, XYZW].
            latch_engaged: Whether the clip is captured by both socket keepers.
            latch_angle_rad: Measured clip deflection [rad]; inward is negative.
            capture_margins_m: Axial, lateral and vertical keeper overlap [m].
            keeper_release_margin_m: Worst vertical overlap of both barbs [m].
            keeper_exit_margin_m: Axial overlap with the keeper front plane [m].
            hand_joint_pos: Measured canonical finger positions [rad]. Their actual
                opening, rather than commanded opening, establishes release.
        """
        if self.phase in ("DONE", "FAILED"):
            return self.command.copy(), self._hand_targets()
        self.elapsed += self.dt
        if self.phase == "SETTLE":
            if self.elapsed < self.settle_duration_s:
                return self.command.copy(), self.closed_hand.copy()
            self._transition("INSERTION/HOLD_GRASP")
            self.elapsed += self.dt
        tcp, head = _pose(tcp_pose_w), _pose(head_pose_w)
        phase = self.phase.split("/")[-1]
        timeout = (
            30.0
            if self.phase in ("INSERTION/RAISE", "INSERTION/ORIENT", "INSERTION/ALIGN", "INSERTION/INSERT")
            else 15.0
        )
        if self.elapsed > timeout:
            self._fail(f"{phase} timed out")
            return self.command.copy(), self._hand_targets()
        if not self.insertion_completed:
            self._insert(tcp, head, latch_engaged, hand_joint_pos)
            fingers = self._hand_targets()
            if self.phase == "INSERTION/DONE":
                self._begin_removal(tcp, head)
        else:
            margins = np.asarray(capture_margins_m, dtype=np.float64)
            if (
                margins.shape != (3,)
                or not np.isfinite(margins).all()
                or not np.isfinite((latch_angle_rad, keeper_release_margin_m, keeper_exit_margin_m)).all()
            ):
                self._fail("Non-finite or invalid measured latch state")
            else:
                self._remove(tcp, head, latch_angle_rad, margins, keeper_release_margin_m, keeper_exit_margin_m)
            fingers = self._hand_targets()
        return self.command.copy(), fingers

    def _insert(self, tcp: np.ndarray, head: np.ndarray, captured: bool, hand: np.ndarray | None) -> None:
        inverse = Rotation.from_quat(tcp[3:]).inv()
        grip = head[:3] + Rotation.from_quat(head[3:]).apply((0.0, 0.0, 0.001))
        if self.phase == "INSERTION/HOLD_GRASP":
            self.dwell += self.dt
            if self.elapsed > 1.0 and self.dwell > 0.5:
                self.grasp_offset_m = inverse.apply(grip - tcp[:3])
                self.grasp_rotation = inverse * Rotation.from_quat(head[3:])
                self.carry_target = head.copy()
                self.carry_target[[0, 2]] = self.goal[[0, 2]]
                self._transition("INSERTION/RAISE")
        if self.phase in ("INSERTION/RAISE", "INSERTION/ORIENT", "INSERTION/ALIGN", "INSERTION/INSERT"):
            if np.linalg.norm(inverse.apply(grip - tcp[:3]) - self.grasp_offset_m) > 0.008:
                self._fail("Connector grasp lost during transport")
                return
            relative_rotation = inverse * Rotation.from_quat(head[3:])
            if (self.grasp_rotation.inv() * relative_rotation).magnitude() > 0.6:
                self._fail("Connector rolled between the fingers during transport")
                return
            angular_error = self._transport_target(tcp, head)
            phase = self.phase.split("/")[-1]
            seat_position = self.goal[:3] if phase == "INSERT" else self.carry_target[:3]
            position_limit, angular_limit = {"ALIGN": (0.0002, 0.02), "INSERT": (0.0005, 0.01)}.get(
                phase, (0.003, 0.25 if phase == "RAISE" else 0.035)
            )
            ready = np.linalg.norm(head[:3] - seat_position) < position_limit and angular_error < angular_limit
            ready &= phase != "INSERT" or (head[1] >= self.goal[1] + 0.0003 and captured)
            self.dwell = self.dwell + self.dt if ready else 0.0
            if self.dwell > 0.5:
                next_phase = {"RAISE": "ORIENT", "ORIENT": "ALIGN", "ALIGN": "INSERT", "INSERT": "RELEASE"}[phase]
                self._transition(f"INSERTION/{next_phase}")
                if next_phase == "ORIENT":
                    self.carry_target[3:] = self.goal[3:]
                elif next_phase in ("ALIGN", "INSERT"):
                    self.carry_target = self.goal.copy()
                    self.carry_target[1] += -0.05 if next_phase == "ALIGN" else 0.0004
                else:
                    # Cancel insertion preload before opening beside the seated plug.
                    self.target = tcp.copy()
                    self.command = tcp.copy()
        if self.phase in ("INSERTION/RELEASE", "INSERTION/RETREAT"):
            self.width = min(0.04, self.width + 0.01 * self.dt)
        if self.phase == "INSERTION/RELEASE":
            opened = hand is not None and np.all(np.abs(hand[:13] - self.open_hand[:13]) < 0.03)
            if self.width >= 0.025 and opened:
                self.target = tcp.copy()
                self._transition("INSERTION/RETREAT")
        if self.phase == "INSERTION/RETREAT" and self.elapsed > 5.0:
            if _near(head, self.goal, 0.003, 0.08) and captured:
                self._transition("INSERTION/DONE")
            else:
                self._fail("Plug did not remain seated after release")
        self._advance_command(tcp, head)

    def _transport_target(self, tcp: np.ndarray, head: np.ndarray) -> float:
        inverse = Rotation.from_quat(tcp[3:]).inv()
        head_rotation = Rotation.from_quat(head[3:])
        rotation = Rotation.from_quat(self.carry_target[3:]) * (inverse * head_rotation).inv()
        desired_head = self.carry_target[:3].copy()
        angular_error = (head_rotation.inv() * Rotation.from_quat(self.carry_target[3:])).magnitude()
        if self.phase == "INSERTION/INSERT":
            if np.linalg.norm(head[[0, 2]] - self.goal[[0, 2]]) > 0.0003 or angular_error > 0.02:
                desired_head[1] = head[1]
        self.target = np.r_[desired_head - rotation.apply(inverse.apply(head[:3] - tcp[:3])), rotation.as_quat()]
        return angular_error

    def _begin_removal(self, tcp: np.ndarray, head: np.ndarray) -> None:
        # Retarget once while open; closing-finger motion must not steer the palm.
        rotation = Rotation.from_quat(head[3:]) * Rotation.from_quat(self.goal[3:]).inv()
        self.grasp = np.r_[
            head[:3] + rotation.apply(self.postures.removal_tcp_offset_w),
            (rotation * Rotation.from_quat(self.postures.removal_tcp_orientation_w)).as_quat(),
        ]
        self.command = tcp.copy()
        self.settle_pose = tcp.copy()
        self.seated_head = head.copy()
        self.insertion_completed = True
        self._transition("REMOVAL/SETTLE")

    def _remove(
        self, tcp: np.ndarray, head: np.ndarray, angle: float, margins: np.ndarray, release: float, exit_margin: float
    ) -> None:
        self.measured_withdrawal_m = self.seated_head[1] - head[1]
        self.target = self.settle_pose.copy()
        clip_clear = angle < 0.0 and release < -0.00002
        if self.phase == "REMOVAL/SETTLE":
            captured = margins[0] > -0.00005 and np.all(margins[1:] > 0.0)
            if self._dwelled(_near(head, self.seated_head, 0.003, 0.08) and captured, 0.3):
                self._transition("REMOVAL/ORIENT")
        elif self.phase in ("REMOVAL/ORIENT", "REMOVAL/APPROACH"):
            self.target = self.grasp.copy()
            if self.phase == "REMOVAL/APPROACH":
                self.target[:3] += head[:3] - self.seated_head[:3]
            hand_ready = self.phase != "REMOVAL/ORIENT" or self.elapsed >= 1.0
            if self._dwelled(_near(tcp, self.target, 0.002, 0.03), 0.3) and hand_ready:
                self._transition("REMOVAL/APPROACH" if self.phase == "REMOVAL/ORIENT" else "REMOVAL/SUPPORT")
        elif self.phase == "REMOVAL/SUPPORT":
            if self.press_tcp is None:
                self.press_tcp = tcp.copy()
            self.target = self.press_tcp
            self.hand_blends = (_smooth(self.elapsed / 2.0), 0.0)
            if self.elapsed >= 2.0 and self._dwelled(True, 0.3):
                self._transition("REMOVAL/PRESS")
        elif self.phase in ("REMOVAL/PRESS", "REMOVAL/PINCH"):
            self.target = self.press_tcp
            self._press_thumb(angle, release)
            if self.phase == "REMOVAL/PRESS":
                if self.elapsed >= 1.0 and self._dwelled(clip_clear, 0.2):
                    self._transition("REMOVAL/PINCH")
            else:
                self.grip_blend = _smooth(self.elapsed)
                reserved = release + self.postures.removal_thumb_clearance_reserve_m < -0.00002
                if self._dwelled(self.grip_blend == 1.0 and clip_clear and reserved, 0.2):
                    inverse = Rotation.from_quat(tcp[3:]).inv()
                    self.grasp_offset_m = inverse.apply(head[:3] - tcp[:3])
                    self.grasp_rotation = inverse * Rotation.from_quat(head[3:])
                    self.withdrawal_tcp = tcp.copy()
                    self.withdrawal_start_distance_m = self.measured_withdrawal_m
                    self._transition("REMOVAL/WITHDRAW")
        elif self.phase in ("REMOVAL/WITHDRAW", "REMOVAL/HOLD"):
            clear = exit_margin < -0.0003
            inverse = Rotation.from_quat(tcp[3:]).inv()
            slip = np.linalg.norm(inverse.apply(head[:3] - tcp[:3]) - self.grasp_offset_m)
            roll = (self.grasp_rotation.inv() * inverse * Rotation.from_quat(head[3:])).magnitude()
            if slip > 0.005 or roll > 0.15:
                self._fail("Physical housing purchase or connector grasp lost during removal")
                return
            if not clear:
                self._press_thumb(angle, release)
            if self.phase == "REMOVAL/WITHDRAW":
                reserved = clip_clear and release + self.postures.removal_thumb_clearance_reserve_m <= -0.00005
                self.target = self._withdrawal_target(tcp, clear, reserved)
                if self.phase == "FAILED":
                    return
                if (
                    self.withdrawal_hold_tcp is None
                    and self.withdrawal_progress_s >= 3.0
                    and self._dwelled(self.measured_withdrawal_m >= 0.057 and _near(tcp, self.target, 0.003, 0.05), 0.3)
                ):
                    self._transition("REMOVAL/HOLD")
            else:
                self.target = self.withdrawal_tcp.copy()
                self.target[1] -= max(0.0, 0.06 - self.withdrawal_start_distance_m)
                if self._dwelled(clear and self.measured_withdrawal_m >= 0.057, 2.0):
                    self._transition("DONE")
        self._advance_command(tcp, head)

    def _withdrawal_target(self, tcp: np.ndarray, clear: bool, reserved: bool) -> np.ndarray:
        if not clear and not reserved and self.withdrawal_hold_tcp is None:
            # Cancel tracking lead while retaining the physical grasp reference.
            self.withdrawal_hold_tcp = tcp.copy()
            self.command = tcp.copy()
            self.dwell = self.latch_loss_s = 0.0
        if self.withdrawal_hold_tcp is not None:
            self.latch_loss_s += self.dt
            if self._dwelled(clear or reserved, 0.2):
                self.withdrawal_tcp = tcp.copy()
                self.withdrawal_start_distance_m = self.measured_withdrawal_m
                self.withdrawal_progress_s = self.latch_loss_s = self.dwell = 0.0
                self.withdrawal_hold_tcp = None
            else:
                if self.latch_loss_s > 2.0:
                    self._fail("Clip rebounded before clearing the socket keepers")
                return self.withdrawal_hold_tcp.copy()
        else:
            self.withdrawal_progress_s += self.dt
        target = self.withdrawal_tcp.copy()
        target[1] -= max(0.0, 0.06 - self.withdrawal_start_distance_m) * _smooth(self.withdrawal_progress_s / 3.0)
        return target

    def _press_thumb(self, angle: float, release: float) -> None:
        clearance = -0.00005 if self.phase == "REMOVAL/WITHDRAW" else -0.00002
        released = angle < 0.0 and release + self.postures.removal_thumb_clearance_reserve_m <= clearance
        thumb = self.hand_blends[1]
        if angle <= -0.23:
            thumb = max(0.0, thumb - 0.15 * self.dt)
        elif not released:
            target = _smooth(self.elapsed / 2.0) if self.phase == "REMOVAL/PRESS" else 1.0
            thumb = max(thumb, min(target, thumb + 0.3 * self.dt))
        self.hand_blends = (1.0, thumb)

    def _hand_targets(self) -> np.ndarray:
        if not self.insertion_completed:
            return self.closed_hand + (self.width / 0.04) * (self.open_hand - self.closed_hand)
        fingers = self.open_hand + self.hand_blends[0] * (self.support_hand - self.open_hand)
        fingers += self.hand_blends[1] * (self.press_hand - fingers)
        fingers[5:13] += self.grip_blend * (self.housing_hand - fingers[5:13])
        return fingers

    def _transition(self, phase: str) -> None:
        self.phase = phase
        self.elapsed = self.dwell = self.latch_loss_s = 0.0

    def _fail(self, reason: str) -> None:
        self.failure = reason
        self._transition("FAILED")

    def _dwelled(self, condition: bool, duration_s: float) -> bool:
        self.dwell = self.dwell + self.dt if condition else 0.0
        return self.dwell >= duration_s

    def _advance_command(self, tcp: np.ndarray, head: np.ndarray) -> None:
        if self.phase == "REMOVAL/PINCH" and self.grip_blend == 1.0:
            # Closing the fingers changes the grasp frame; align the measured plug before pulling.
            self.carry_target = self.goal.copy()
            self._transport_target(tcp, head)
        desired = self.target.copy()
        inserting = self.phase == "INSERTION/INSERT"
        lead = self.target[:3] - self.command[:3]
        close = np.abs(lead) < 0.02 if inserting else np.linalg.norm(lead) < 0.02
        desired[:3] = np.where(
            close, self.command[:3] + min(1.0, 4.0 * self.dt) * (self.target[:3] - tcp[:3]), desired[:3]
        )
        delta = desired[:3] - self.command[:3]
        if inserting:
            speed = np.interp(self.goal[1] - head[1], (0.015, 0.030), (0.006, 0.040))
            limits = np.array((0.01, speed, 0.01)) * self.dt
            self.command[:3] += np.clip(delta, -limits, limits)
        else:
            self.command[:3] += delta * min(1.0, 0.04 * self.dt / max(np.linalg.norm(delta), 1e-9))
        rotation = Rotation.from_quat(self.command[3:])
        desired_rotation = Rotation.from_quat(self.target[3:])
        if (rotation.inv() * desired_rotation).magnitude() < 0.1:
            error = (desired_rotation * Rotation.from_quat(tcp[3:]).inv()).as_rotvec()
            desired_rotation = Rotation.from_rotvec(min(1.0, 4.0 * self.dt) * error) * rotation
        angle = (rotation.inv() * desired_rotation).as_rotvec()
        self.command[3:] = (
            rotation * Rotation.from_rotvec(angle * min(1.0, 0.5 * self.dt / max(np.linalg.norm(angle), 1e-9)))
        ).as_quat()


def _run(args: argparse.Namespace) -> int:
    import torch

    from isaaclab.app import launch_simulation
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
    from isaaclab.utils import instantiate
    from isaaclab.utils.math import subtract_frame_transforms

    from isaaclab_tasks.contrib.gb300_sharpa_insertion import mdp
    from isaaclab_tasks.contrib.gb300_sharpa_insertion.assets import RIZON_SHARPA_RIGHT_HAND_JOINT_NAMES
    from isaaclab_tasks.contrib.gb300_sharpa_insertion.env_cfg import GB300SharpaInsertionEnvCfg
    from isaaclab_tasks.contrib.gb300_sharpa_insertion.robot_cfg import CLOSE_HAND_RAD
    from isaaclab_tasks.contrib.gb300_sharpa_insertion.visual_scene import add_gl_visuals, configure_visual_scene
    from isaaclab_tasks.contrib.gb300_sharpa_insertion.workcell import head_goal_for_socket

    cfg = GB300SharpaInsertionEnvCfg()
    cfg.seed = args.seed
    cfg.sim.device = args.device
    if args.video:
        cfg.video_recorders = [
            VideoRecorderCfg(
                output_dir=str(args.output_dir),
                output_filename_prefix=f"cycle_{datetime.now():%Y%m%d_%H%M%S_%f}",
                video_length=args.num_steps,
            )
        ]
    with launch_simulation(cfg, args):
        rack_path = configure_visual_scene(cfg)
        env = instantiate(cfg)
        try:
            observations, _ = env.reset(seed=args.seed)
            add_gl_visuals(env, rack_path)
            if env.action_manager.total_action_dim != 29:
                raise RuntimeError("The example must provide absolute TCP and 22 finger targets.")
            robot = env.scene["robot"]
            hand_ids, _ = robot.find_joints(RIZON_SHARPA_RIGHT_HAND_JOINT_NAMES, preserve_order=True)

            def controller_state(status: torch.Tensor) -> np.ndarray:
                # One readback of poses, clip geometry, finger opening and state validity.
                captured, margins, release, exit_margin = mdp.latch_metrics(env)
                return (
                    torch.cat(
                        (
                            mdp.tcp_pose(env)[0],
                            mdp.head_pose(env)[0],
                            mdp.latch_state(env)[0, :1],
                            captured[0, None],
                            margins[0],
                            release[0, None],
                            exit_margin[0, None],
                            robot.data.joint_pos.torch[0, hand_ids],
                            status,
                        )
                    )
                    .cpu()
                    .numpy()
                )

            state = controller_state(torch.isfinite(observations["state"]).all()[None])
            if not state[-1]:
                raise RuntimeError("Cable cycle encountered a non-finite reset state.")
            goal_position = np.asarray(head_goal_for_socket(cfg.target_socket_position_e))
            goal_position += env.scene.env_origins[0].cpu().numpy()
            machine = CableCycleStateMachine(
                state[:7],
                np.r_[goal_position, 2**-0.5, 0.0, 0.0, 2**-0.5],
                env.step_dt,
                CLOSE_HAND_RAD,
                settle_duration_s=args.settle_duration_s,
            )
            initial_hand = robot.data.joint_pos.torch[:, hand_ids].clone()
            closed_hand = initial_hand.new_tensor(CLOSE_HAND_RAD)[None]
            step = 0
            failure_step: int | None = None
            failure_tail_steps = math.ceil(min(args.failure_tail_s / env.step_dt, args.num_steps))
            reason = "Simulation closed or step budget exhausted before the complete cycle."
            prior_phase = machine.phase
            with torch.inference_mode():
                while env.sim.is_running() and step < args.num_steps:
                    if not env.sim.is_playing():
                        env.sim.render()
                        continue
                    if failure_step is None:
                        command, fingers = machine.update(
                            state[:7],
                            state[7:14],
                            latch_engaged=bool(state[15]),
                            latch_angle_rad=float(state[14]),
                            capture_margins_m=state[16:19],
                            keeper_release_margin_m=float(state[19]),
                            keeper_exit_margin_m=float(state[20]),
                            hand_joint_pos=state[21:43],
                        )
                    if machine.phase != prior_phase:
                        logger.info("[GB300 cycle] step=%d phase=%s", step, machine.phase)
                        prior_phase = machine.phase
                    if machine.phase == "DONE":
                        reason = None
                        break
                    if machine.phase == "FAILED":
                        if failure_step is None:
                            failure_step = step
                            reason = machine.failure
                            logger.warning(
                                "[GB300 cycle] fault=%s; holding targets for %g s", reason, args.failure_tail_s
                            )
                        if step - failure_step >= failure_tail_steps:
                            break
                    targets = torch.as_tensor(np.r_[command, fingers], dtype=torch.float32, device=env.device)[None]
                    root = robot.data.root_pose_w.torch
                    position, rotation = subtract_frame_transforms(
                        root[:, :3], root[:, 3:], targets[:, :3], targets[:, 3:7]
                    )
                    hand_targets = targets[:, 7:]
                    if failure_step is None and step * env.step_dt < 0.2:
                        fraction = min((step + 1) * env.step_dt / 0.2, 1.0)
                        hand_targets = torch.lerp(initial_hand, closed_hand, fraction)
                    observations, _ = env.step(torch.cat((position, rotation, hand_targets), dim=-1))
                    step += 1
                    state = controller_state(torch.isfinite(observations["state"]).all()[None])
                    if not state[-1]:
                        raise RuntimeError("Cable cycle encountered a non-finite simulation state.")
            success = machine.phase == "DONE" and machine.insertion_completed
            print(
                json.dumps(
                    {
                        "success": success,
                        "phase": machine.phase,
                        "reason": reason,
                        "insertion_completed": machine.insertion_completed,
                        "steps": step,
                        "failure_step": failure_step,
                        "post_failure_simulation_s": 0.0
                        if failure_step is None
                        else (step - failure_step) * env.step_dt,
                        "measured_withdrawal_m": machine.measured_withdrawal_m,
                    },
                    indent=2,
                ),
                flush=True,
            )
            return 0 if success else 1
        finally:
            env.close()


def _near(pose: np.ndarray, target: np.ndarray, position_m: float, angle_rad: float) -> bool:
    return (
        np.linalg.norm(pose[:3] - target[:3]) < position_m
        and (Rotation.from_quat(pose[3:]).inv() * Rotation.from_quat(target[3:])).magnitude() < angle_rad
    )


def _smooth(progress: float) -> float:
    progress = float(np.clip(progress, 0.0, 1.0))
    return progress**3 * (10.0 + progress * (-15.0 + 6.0 * progress))


def _pose(value: np.ndarray) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64).copy()
    if pose.shape != (7,) or not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) < 1e-9:
        raise ValueError("TCP and connector poses must contain finite [m, XYZW] values and a nonzero quaternion.")
    return pose


if __name__ == "__main__":
    raise SystemExit(main())
