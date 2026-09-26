# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measured robot actions for fruit, visual tap filling, lid, docking and button.

Only robot actions are commanded. Cup and basket transforms are measured grasp
references, never physical attachments. The environment owns tap state, fill
timers, contact measurements and task success.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .assembly_controller import OPTIONS as ASSEMBLY_OPTIONS
from .assembly_controller import AssemblyTrajectory, bounded_joint_posture_step

OPTIONS = {
    "tap_cup_position": [0.38, -0.30, 0.013],
    "tap_button_position": [0.28, -0.30, 0.10],
    "tap_button_direction": [0.0, 0.0, -1.0],
    "tap_button_rotation": [1.0, 0.0, 0.0, 0.0],
    "motor_button_position": [0.64, 0.177, 0.09],
    "motor_button_direction": [0.0, 1.0, 0.0],
    "motor_button_rotation": [-(2**-0.5), 0.0, 0.0, 2**-0.5],
    "button_prepress_distance_m": 0.045,
    "button_press_distance_m": 0.004,
    # The pinned closed finger pads extend 4.9 mm beyond the measured TCP.
    "finger_pad_forward_offset_m": 0.0049,
    "button_contact_overtravel_m": 0.0005,
    "cup_lift_m": 0.17,
    "tap_clearance_lift_m": 0.03,
    "tap_front_offset_m": [0.0, -0.14, 0.0],
    "cup_pregrasp_distance_m": 0.05,
    "position_tolerance_m": 0.008,
    "rotation_tolerance_rad": 0.08,
    "endpoint_hold_s": 0.20,
    "support_hold_s": 0.35,
    "tap_wait_s": 2.0,
    "maximum_stage_s": 60.0,
}

TAP_STAGES = (
    ("raise_empty", 4.0),
    ("cup_joint_posture", 12.0),
    ("cup_pregrasp", 4.0),
    ("cup_approach", 3.0),
    ("cup_close", 2.5),
    ("cup_lift", 3.0),
    ("cup_carry_corner", 3.0),
    ("cup_carry_tap", 4.0),
    ("cup_lower_tap_front", 3.0),
    ("cup_enter_tap", 3.0),
    ("cup_lower_tap", 3.0),
    ("cup_support_tap", 0.5),
    ("cup_release_tap", 2.5),
    ("cup_retreat_tap", 2.0),
    ("tap_raise", 2.0),
    ("tap_approach", 3.0),
    ("tap_press_on", 1.0),
    ("tap_retract_on", 1.2),
    ("tap_wait", 0.0),
    ("tap_press_off", 1.0),
    ("tap_retract_off", 1.2),
    ("return_raise", 2.0),
    ("return_pregrasp", 4.0),
    ("return_approach", 3.0),
    ("return_close", 2.5),
    ("return_lift", 3.0),
    ("return_leave_tap", 3.0),
    ("return_raise_cup", 3.0),
    ("return_carry_corner", 3.0),
    ("return_carry_home", 4.0),
    ("return_lower_home", 3.0),
    ("return_support_home", 0.5),
    ("return_release_home", 2.5),
    ("return_retreat_home", 2.0),
    ("complete", 0.25),
)
BUTTON_STAGES = (("raise", 3.0), ("approach", 3.0), ("press", 1.0), ("retract", 1.5), ("complete", 0.25))


def _one(value: Any) -> Any:
    return value.item() if hasattr(value, "numel") and value.numel() == 1 else value


from .basket_controller import BasketController


def _pose(position: np.ndarray, rotation: Rotation) -> np.ndarray:
    return np.r_[position, rotation.as_quat()]


def _interpolate(start: np.ndarray, goal: np.ndarray, fraction: float) -> np.ndarray:
    fraction = float(np.clip(fraction, 0.0, 1.0))
    blend = fraction * fraction * (3.0 - 2.0 * fraction)
    rotation = Slerp([0.0, 1.0], Rotation.from_quat([start[3:], goal[3:]]))([blend])[0]
    return _pose(start[:3] + blend * (goal[:3] - start[:3]), rotation)


class _SmoothieTrajectory:
    """Track physical cup and button targets using measured completion gates."""

    def __init__(self, options: dict[str, Any], *, button_only: bool = False):
        self.options = options
        self.button_only = button_only
        self.stages = BUTTON_STAGES if button_only else TAP_STAGES
        self.index = 0
        self.elapsed_s = self.stage_wall_s = self.hold_s = self.support_s = 0.0
        self.start: np.ndarray | None = None
        self.goal: np.ndarray | None = None
        self.joint_start: np.ndarray | None = None
        self.home: np.ndarray | None = None
        self.grasp_position: np.ndarray | None = None
        self.grasp_rotation: Rotation | None = None
        self.close_target: np.ndarray | None = None
        self.release_authorized = False
        self.complete = False
        self.placement = AssemblyTrajectory()

    @property
    def stage(self) -> str:
        return self.stages[self.index][0]

    def _grasp(self, cup: np.ndarray, *, backoff: bool = False) -> np.ndarray:
        rotation = Rotation.from_quat(cup[3:])
        local = np.asarray(ASSEMBLY_OPTIONS["cup_grasp_local_m"]).copy()
        if backoff:
            local[1] -= self.options["cup_pregrasp_distance_m"]
        return _pose(
            cup[:3] + rotation.apply(local), rotation * Rotation.from_quat(ASSEMBLY_OPTIONS["cup_hand_local_xyzw"])
        )

    def _held(self, cup: np.ndarray) -> np.ndarray:
        if self.grasp_position is None or self.grasp_rotation is None:
            raise RuntimeError("Cup motion requires a measured closed grasp.")
        rotation = Rotation.from_quat(cup[3:])
        return _pose(cup[:3] + rotation.apply(self.grasp_position), rotation * self.grasp_rotation)

    def _button(self, *, pressed: bool) -> np.ndarray:
        prefix = "motor" if self.button_only else "tap"
        direction = np.asarray(self.options[prefix + "_button_direction"], dtype=float)
        direction /= np.linalg.norm(direction)
        distance = (
            self.options["button_press_distance_m"]
            - self.options["finger_pad_forward_offset_m"]
            + self.options["button_contact_overtravel_m"]
            if pressed
            else -self.options["button_prepress_distance_m"]
        )
        return _pose(
            np.asarray(self.options[prefix + "_button_position"]) + distance * direction,
            Rotation.from_quat(self.options[prefix + "_button_rotation"]),
        )

    def _enter(self, hand: np.ndarray, cup: np.ndarray, joints: np.ndarray) -> None:
        stage = self.stage
        self.start, self.goal = hand.copy(), hand.copy()
        self.release_authorized = False
        self.support_s = 0.0
        if self.home is None:
            self.home = cup.copy()
        if self.button_only:
            if stage == "raise":
                self.goal[2] = max(hand[2], 0.40)
            elif stage in ("approach", "retract"):
                self.goal = self._button(pressed=False)
            elif stage == "press":
                self.goal = self._button(pressed=True)
            return
        if stage in ("raise_empty", "tap_raise", "return_raise"):
            self.goal[2] = max(hand[2], 0.60 if stage == "raise_empty" else 0.42)
        elif stage == "cup_joint_posture":
            self.joint_start = joints.copy()
        elif stage in ("cup_pregrasp", "return_pregrasp"):
            self.goal = self._grasp(cup, backoff=True)
            self.placement = AssemblyTrajectory()
        elif stage in ("cup_approach", "return_approach"):
            self.goal = self._grasp(cup)
        elif stage in ("cup_close", "return_close"):
            self.goal = self.close_target.copy() if self.close_target is not None else hand.copy()
        elif stage in ("cup_lift", "return_lift"):
            rotation = Rotation.from_quat(cup[3:])
            self.grasp_position = rotation.inv().apply(hand[:3] - cup[:3])
            self.grasp_rotation = rotation.inv() * Rotation.from_quat(hand[3:])
            target = cup.copy()
            target[2] += self.options["tap_clearance_lift_m" if stage == "return_lift" else "cup_lift_m"]
            self.goal = self._held(target)
        elif stage in (
            "cup_carry_corner",
            "cup_carry_tap",
            "cup_lower_tap_front",
            "cup_enter_tap",
            "cup_lower_tap",
            "return_leave_tap",
            "return_raise_cup",
            "return_carry_corner",
            "return_carry_home",
            "return_lower_home",
        ):
            target = self.home.copy()
            if stage not in ("cup_carry_corner", "return_carry_corner", "return_carry_home", "return_lower_home"):
                target[:3] = self.options["tap_cup_position"]
            if stage in ("cup_carry_corner", "return_carry_corner"):
                target[1] = self.options["tap_cup_position"][1] + self.options["tap_front_offset_m"][1]
            # Stay in front of the nozzle until the cup rim is below its outlet.
            if stage in ("cup_carry_tap", "cup_lower_tap_front", "return_leave_tap", "return_raise_cup"):
                target[:3] += self.options["tap_front_offset_m"]
            if stage in (
                "cup_carry_corner",
                "cup_carry_tap",
                "return_raise_cup",
                "return_carry_corner",
                "return_carry_home",
            ):
                target[2] += self.options["cup_lift_m"]
            elif stage in ("cup_lower_tap_front", "cup_enter_tap", "return_leave_tap"):
                target[2] += self.options["tap_clearance_lift_m"]
            self.goal = self._held(target)
        elif stage in ("cup_retreat_tap", "return_retreat_home"):
            self.goal[:3] += Rotation.from_quat(cup[3:]).apply([0.0, -0.11, 0.03])
        elif stage in ("tap_approach", "tap_retract_on", "tap_retract_off", "tap_wait"):
            self.goal = self._button(pressed=False)
        elif stage in ("tap_press_on", "tap_press_off"):
            self.goal = self._button(pressed=True)

    def _gates(self, m: dict, endpoint: bool, dt: float) -> tuple[bool, bool]:
        def get(name: str) -> bool:
            return bool(m.get(name, False))

        stage = self.stage
        if self.button_only:
            gate = get("assembly_complete")
            if stage == "press" and endpoint:
                gate &= get("motor_on") or get("motor_button_pressed")
            if stage in ("retract", "complete") and endpoint:
                gate &= get("motor_on")
            return gate, True
        held_stages = {
            "cup_lift",
            "cup_carry_corner",
            "cup_carry_tap",
            "cup_lower_tap_front",
            "cup_enter_tap",
            "cup_lower_tap",
            "cup_support_tap",
            "return_lift",
            "return_leave_tap",
            "return_raise_cup",
            "return_carry_corner",
            "return_carry_home",
            "return_lower_home",
            "return_support_home",
        }
        close = stage in held_stages or stage in ("cup_close", "return_close") or stage.startswith("tap_")
        gate = get("cup_held") if stage in held_stages else True
        if stage in ("cup_close", "return_close") and endpoint:
            gate &= get("cup_held")
        if stage in ("cup_support_tap", "cup_release_tap", "return_support_home", "return_release_home"):
            positioned = get("cup_under_tap") if stage.startswith("cup_") else get("cup_home")
            supported = get("cup_supported") and positioned
            self.support_s = self.support_s + dt if supported else 0.0
            if "release" in stage:
                self.release_authorized |= self.support_s >= self.options["support_hold_s"] - 1e-9
                close = not self.release_authorized
                gate = self.release_authorized and supported
                if endpoint:
                    gate &= get("fingers_open")
            elif endpoint:
                gate &= supported and self.support_s >= self.options["support_hold_s"] - 1e-9
        if stage.startswith("tap_"):
            gate &= get("cup_under_tap") and get("cup_supported")
            if stage == "tap_press_on" and endpoint:
                gate &= get("tap_on")
            elif stage in ("tap_retract_on", "tap_wait"):
                gate &= get("tap_on")
                if endpoint:
                    gate &= not get("tap_button_pressed")
            elif stage in ("tap_press_off", "tap_retract_off") and endpoint:
                gate &= not get("tap_on")
            if stage == "tap_wait":
                gate &= float(m.get("tap_fill_time_s", 0.0)) >= self.options["tap_wait_s"]
        if stage.startswith("return_"):
            gate &= not get("tap_on")
        if stage in (
            "raise_empty",
            "cup_joint_posture",
            "cup_pregrasp",
            "cup_approach",
            "return_raise",
            "return_pregrasp",
            "return_approach",
        ):
            gate &= get("fingers_open")
        if stage in ("return_retreat_home", "complete"):
            gate &= get("cup_home") and get("cup_supported") and get("fingers_open") and not get("tap_on")
        return gate, close

    def step(self, state: dict, dt: float) -> dict:
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Policy interval must be positive and finite [s].")
        hand, cup = np.asarray(state["hand_pose"]), np.asarray(state["cup_pose"])
        joints, m = np.asarray(state["joints"]), state["metrics"]
        if not bool(m.get("finite", m.get("valid", False))) or bool(m.get("failed", False)):
            raise RuntimeError("Tap sequence encountered invalid physical state.")
        if not all(np.isfinite(a).all() for a in (hand, cup, joints)):
            raise RuntimeError("Nonfinite measured robot or cup state.")
        if self.start is None:
            self._enter(hand, cup, joints)
        duration = self.stages[self.index][1]
        endpoint = self.elapsed_s >= duration - 1e-9
        fraction = min(self.elapsed_s / max(duration, dt), 1.0)
        sample = _interpolate(self.start, self.goal, fraction)
        joint_target = None
        placement = {}
        position_error = float(np.linalg.norm(sample[:3] - hand[:3]))
        rotation_error = float((Rotation.from_quat(sample[3:]) * Rotation.from_quat(hand[3:]).inv()).magnitude())
        tracking = (
            position_error <= self.options["position_tolerance_m"]
            and rotation_error <= self.options["rotation_tolerance_rad"]
        )
        if self.stage == "cup_joint_posture":
            blend = fraction * fraction * (3.0 - 2.0 * fraction)
            joint_target = self.joint_start + blend * (
                np.asarray(ASSEMBLY_OPTIONS["cup_empty_joint_waypoint_rad"]) - self.joint_start
            )
            tracking = np.max(np.abs(joint_target - joints)) <= (0.008 if endpoint else 0.020)
        if self.stage in ("cup_approach", "return_approach") and endpoint:
            sample, tracking, placement = self.placement._cup_approach_endpoint(
                hand, cup, dt, bool(m.get("fingers_open", False))
            )
            self.close_target = sample.copy()
        gate, close = self._gates(m, endpoint, dt)
        self.stage_wall_s += dt
        if self.stage_wall_s > self.options["maximum_stage_s"]:
            raise RuntimeError(f"Tap substage {self.stage} exceeded its bounded physical time.")
        self.hold_s = self.hold_s + dt if endpoint and tracking and gate else 0.0
        result = {
            "stage": self.stage,
            "tcp_position": sample[:3],
            "hand_xyzw": sample[3:],
            "close": close,
            "joint_position_target_rad": joint_target,
            "elapsed_s": self.elapsed_s,
            "stage_wall_s": self.stage_wall_s,
            "position_error_m": position_error,
            "rotation_error_rad": rotation_error,
            "tracking": bool(tracking),
            "gate": bool(gate),
            "hold_s": self.hold_s,
            "complete": self.complete,
            **placement,
        }
        if tracking and gate:
            self.elapsed_s = min(duration, self.elapsed_s + dt)
        if self.hold_s >= self.options["endpoint_hold_s"] - 1e-9:
            if self.index == len(self.stages) - 1:
                self.complete = True
                result["complete"] = True
            else:
                self.index += 1
                self.elapsed_s = self.stage_wall_s = self.hold_s = 0.0
                self.start = self.goal = None
        return result


class SmoothieSequenceController:
    """Compose basket, cup/tap, lid/dock and physical blender-button actions.

    The one-world environment supplies world poses [m, XYZW], a 30 Hz policy
    interval, arm EMA history, ``basket_measurements``, ``assembly_measurements``
    and ``tap_measurements``. The tap measurements include physical grasp/support,
    cup home/tap alignment, switch state, fill time [s], and motor button state.
    """

    def __init__(self, env: Any, *, trace_path: Path | None = None):
        from .bounded_return_pose import BoundedReturnPoseController

        self.env = env
        self.basket = BasketController(env)
        self.pose_controller = BoundedReturnPoseController(env)
        options = {name: getattr(env.cfg, name, value) for name, value in OPTIONS.items()}
        self.tap = _SmoothieTrajectory(options)
        self.button = _SmoothieTrajectory(options, button_only=True)
        self.assembly = None
        self.stage = "basket"
        self.diagnostics: dict[str, Any] = {}
        self.previous_step: int | None = None
        # The runner closes this owned stream through close_trace in its finally block.
        self.trace = None if trace_path is None else Path(trace_path).open("x", buffering=1)  # noqa: SIM115
        self.last_trace_stage: str | None = None

    @property
    def tap_complete(self) -> bool:
        """Whether the cup was filled, returned, released and cleared by the hand."""
        return self.tap.complete

    @property
    def complete(self) -> bool:
        """Whether the final physical button action and hand retreat completed."""
        return self.button.complete

    def _measured_state(self) -> dict:
        env = self.env
        origin = env.scene.env_origins[0].detach().cpu().numpy()
        hand = env.robot.data.body_link_pose_w.torch[0, env.hand_id].detach().cpu().numpy().copy()
        hand[:3] = env.tcp()[0].detach().cpu().numpy() - origin
        cup = env.pose("cup")[0].detach().cpu().numpy().copy()
        cup[:3] -= origin
        metrics = {key: _one(value) for key, value in env.tap_measurements().items()}
        if self.tap.complete:
            metrics.update({key: _one(value) for key, value in env.assembly_measurements().items()})
        return {
            "hand_pose": hand,
            "cup_pose": cup,
            "metrics": metrics,
            "joints": env.robot.data.joint_pos.torch[0, self.pose_controller.joint_ids].detach().cpu().numpy(),
        }

    def _actions(self, sample: dict):
        import torch

        env = self.env
        target = torch.as_tensor(sample["tcp_position"], device=env.device, dtype=env.tcp().dtype)[None]
        target = target + env.scene.env_origins
        rotation = target.new_tensor(sample["hand_xyzw"])[None]
        if sample["joint_position_target_rad"] is None:
            return self.pose_controller.compute(target, rotation, sample["close"])
        joints = env.robot.data.joint_pos.torch[0, self.pose_controller.joint_ids]
        limits = env.robot.data.joint_pos_limits.torch[0, self.pose_controller.joint_ids]
        previous = env.action_manager.get_term("arm_action").processed_actions[0]
        raw, diagnostics = bounded_joint_posture_step(
            sample["joint_position_target_rad"],
            joints.detach().cpu().numpy(),
            limits.detach().cpu().numpy(),
            previous.detach().cpu().numpy(),
        )
        sample["joint_control"] = diagnostics
        actions = torch.zeros((1, 8), device=env.device, dtype=joints.dtype)
        actions[0, :7] = actions.new_tensor(raw)
        actions[0, 7] = -1.0 if sample["close"] else 1.0
        return actions

    def compute(self, step: int):
        """Return consecutive raw robot actions, shape [1, 8], without state writes."""
        import torch

        from .assembly_controller import AssemblyController

        if type(step) is not int or step < 0 or (self.previous_step is not None and step != self.previous_step + 1):
            raise ValueError("Controller calls require consecutive nonnegative policy steps.")
        self.previous_step = step
        phase = int(_one(self.env.phase))
        if phase == 0:
            actions = self.basket.compute(step)
            self.stage, self.diagnostics = "basket/" + self.basket.stage, dict(self.basket.diagnostics)
        elif not self.tap.complete:
            sample = self.tap.step(self._measured_state(), self.env.step_dt)
            actions = self._actions(sample)
            self.stage, self.diagnostics = "tap/" + sample["stage"], sample
        elif self.assembly is None or not self.assembly.trajectory.complete:
            if self.assembly is None:
                self.assembly = AssemblyController(self.env)
            actions = self.assembly.compute(step)
            self.stage, self.diagnostics = "assembly/" + self.assembly.stage, dict(self.assembly.diagnostics)
        else:
            sample = self.button.step(self._measured_state(), self.env.step_dt)
            actions = self._actions(sample)
            self.stage, self.diagnostics = "button/" + sample["stage"], sample
        if actions.shape != (1, 8) or not torch.isfinite(actions).all() or (actions.abs() > 1).any():
            raise RuntimeError("Tap controller produced invalid raw actions.")
        if self.trace is not None and (self.stage != self.last_trace_stage or step % 30 == 0):

            def convert(value):
                if isinstance(value, np.ndarray):
                    return value.tolist()
                if isinstance(value, np.generic):
                    return value.item()
                if torch.is_tensor(value):
                    return value.detach().cpu().tolist()
                raise TypeError(type(value).__name__)

            self.trace.write(
                json.dumps(
                    {
                        "step": step,
                        "time_s": step * self.env.step_dt,
                        "stage": self.stage,
                        "diagnostics": self.diagnostics,
                    },
                    default=convert,
                )
                + "\n"
            )
            self.last_trace_stage = self.stage
        return actions

    def close_trace(self) -> None:
        """Close the optional controller event trace."""
        if self.trace is not None:
            self.trace.close()
