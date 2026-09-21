# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted pickup followed by a measured fruit pour and gated basket return.

The maintained basket expert commands raw robot actions and retains live EMA
history. The environment owns receipt/support measurements and all task latches;
physical success requires a complete simulated recording.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .basket_pose_controller import OPTIONS as POSE_OPTIONS
from .basket_pose_controller import BasketPoseController
from .controlled_fruit_pour import OPTIONS as POUR_OPTIONS
from .controlled_fruit_pour import ControlledFruitPour
from .pose_control import BasketExpert

POLICY_DT_S = 1.0 / 30.0
ARM_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))
PICKUP_RECIPE = {
    "controller": "BasketExpert",
    "reference": "configured live basket rim grasp pose; actual TCP captured when closing starts",
    "grasp_frame": "continuous_env.make_cfg; exact basket offset/quaternion and rationale recorded in profile",
    "approach_above_grasp_m": 0.15,
    "descent_start_s": 3.0,
    "descent_duration_s": 3.5,
    "close_start_s": 7.5,
    "lift_start_s": 11.0,
    "lift_duration_s": 3.5,
    "lift_height_m": 0.15,
    "after_lift": "hold the raised target until the measured pickup handoff qualifies",
}
OPTIONS = {
    "pickup_recipe": PICKUP_RECIPE,
    "basket_pose_adapter": POSE_OPTIONS,
    "clock": "basket-local policy steps at 30 Hz; measured gates own phase transitions",
    "arm_scale": 0.03,
    "arm_alpha": 0.2,
    "gripper_alpha": 0.04,
    "return_gate": "delivery_complete and current receipt and held and handoff_hold_s >= 0.25",
    "minimum_fruit_count": 16,
    "pickup_handoff": "valid held grasp and measured strict upright lift complete",
    "pour": POUR_OPTIONS,
    "held_handoff_s": 0.25,
    "support_hold_s": 0.5,
    "tracking_position_tolerance_m": 0.015,
    "tracking_rotation_tolerance_rad": 0.15,
    "return_nominal_duration_s": 42.0,
    "return_grasp": "measured once at qualified handoff; no object attachment",
    "return_clock": "pauses on tracking/grasp/support/open gates",
    "global_timeout": "owned by supervisor; no controller timeout or environment reset",
    "phase0_gripper_open_m": 0.015,
    "phase0_gripper_close_m": 0.0,
    "physical_transfer_validated": False,
}
from .return_path import ReturnCandidate


def _one(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Basket controller requires one-world scalar measurements.")
        return value.item()
    return value


class BasketController:
    """Compute one world's eight raw actions using measured state, without state writes.

    Args:
        env: Combined environment exposing basket-specific measurements/latches,
            actual named joints, arm EMA history, TCP and rigid-body poses [m, XYZW].
    """

    def __init__(self, env: Any):
        arm = env.cfg.actions.arm_action
        if (
            arm.scale != OPTIONS["arm_scale"]
            or arm.alpha != OPTIONS["arm_alpha"]
            or not arm.use_zero_offset
            or arm.clip is not None
        ):
            raise ValueError("Named-joint arm scale, EMA, zero offset and clipping must be preserved.")
        gripper = env.cfg.actions.gripper_action
        if (
            gripper.open_positions_by_phase != {0: 0.015, 1: 0.04, 2: 0.04, 3: 0.04, 4: 0.04, 5: 0.04}
            or gripper.close_positions_by_phase != {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
            or gripper.default_position != 0.015
            or gripper.alpha != OPTIONS["gripper_alpha"]
        ):
            raise ValueError("The combined phase gripper must preserve original basket targets and EMA.")
        ids, names = env.robot.find_joints(list(ARM_NAMES), preserve_order=True)
        if tuple(names) != ARM_NAMES or len(set(ids)) != 7:
            raise ValueError("Expected seven distinct, correctly ordered named Panda arm joints.")
        self.env, self.joint_ids = env, ids
        self.pickup = BasketExpert(env)
        self.pose_controller = BasketPoseController(env)
        self.stage = "scripted_pickup"
        self.diagnostics: dict[str, Any] = {}
        self._return: Any | None = None
        self._pour: ControlledFruitPour | None = None
        self._return_elapsed = 0.0
        self._open_started = False
        self._last_step = -1

    @torch.no_grad()
    def compute(self, local_step: int) -> torch.Tensor:
        """Return raw actions, shape [1, 8], at a basket-local policy step.

        A new local step zero clears only this controller's phase state. It never
        resets joints, objects, action filters or the environment's task latches.
        """
        if type(local_step) is not int or local_step < 0:
            raise ValueError("Require a nonnegative integer component-local step.")
        if local_step == 0:
            self._return, self._return_elapsed, self._open_started = None, 0.0, False
            self._pour = None
        elif local_step <= self._last_step:
            raise ValueError("Local steps must advance, except for an explicit component reset to zero.")
        if int(_one(self.env.phase)) != 0:
            raise ValueError("BasketController may execute only in basket phase zero.")
        self._last_step = local_step
        m, state = self.env.basket_measurements(), self.env.basket_state
        valid = bool(_one(m["finite"])) and not bool(_one(m["failed"]))
        receipt = int(_one(m["fruit_count"])) >= OPTIONS["minimum_fruit_count"] and bool(_one(m["all_types"]))
        held = bool(_one(m["held"]))
        eligible = (
            valid
            and receipt
            and held
            and bool(_one(state["delivery_complete"]))
            and float(_one(state["handoff_hold_s"])) >= OPTIONS["held_handoff_s"] - 1e-6
        )
        self.diagnostics = {
            "local_step": local_step,
            "local_time_s": local_step * POLICY_DT_S,
            "valid": valid,
            "held": held,
            "strict_held": bool(_one(m.get("strict_held", m["held"]))),
            "grasp_continuation": bool(_one(m.get("grasp_continuation", False))),
            "receipt_current": receipt,
            "return_handoff_eligible": eligible,
            "pickup_recipe": PICKUP_RECIPE,
        }
        if self._return is None and eligible:
            basket, tcp, hand, cup = self._measured_pose()
            self._return = ReturnCandidate(basket, tcp, hand, cup)
            self._return_elapsed = 0.0
            self.diagnostics["captured_grip_b_m"] = self._return.grip_b.tolist()
            self.diagnostics["captured_hand_b_xyzw"] = self._return.hand_b.as_quat().tolist()
        if (
            self._return is None
            and self._pour is None
            and valid
            and held
            and bool(_one(self.env.basket_strict_lift_complete))
        ):
            basket, tcp, hand, cup = self._measured_pose()
            self._pour = ControlledFruitPour(basket[:3], basket[3:], tcp, hand.as_quat(), cup[:3], cup[3:], POLICY_DT_S)
        if self._return is None and self._pour is None:
            self.stage = "scripted_pickup"
            actions = self._scripted_actions(local_step)
        elif self._return is None:
            actions = self._pour_actions(valid, held, int(_one(m["fruit_count"])))
        else:
            actions = self._return_actions(m, state, valid, held)
        if actions.shape != (1, 8) or not torch.isfinite(actions).all() or (actions.abs() > 1).any():
            raise ValueError("Basket controller produced invalid raw actions.")
        self.diagnostics["stage"] = self.stage
        return actions

    def _scripted_actions(self, local_step: int) -> torch.Tensor:
        actions = self.pickup.compute(local_step)
        elapsed = local_step * POLICY_DT_S
        if elapsed < PICKUP_RECIPE["close_start_s"]:
            phase = "approach"
        elif elapsed < PICKUP_RECIPE["lift_start_s"]:
            phase = "close_dwell"
        elif elapsed < PICKUP_RECIPE["lift_start_s"] + PICKUP_RECIPE["lift_duration_s"]:
            phase = "lift"
        else:
            phase = "hold_raised"
        self.diagnostics.update(
            pickup_phase=phase,
            raw_gripper_sign=float(actions[0, 7]),
        )
        return actions

    def _measured_pose(self) -> tuple[np.ndarray, np.ndarray, Rotation, np.ndarray]:
        origin = self.env.scene.env_origins[0].detach().cpu().numpy()
        basket = self.env.pose("basket")[0].detach().cpu().numpy().copy()
        cup = self.env.pose("cup")[0].detach().cpu().numpy().copy()
        basket[:3] -= origin
        cup[:3] -= origin
        tcp = self.env.tcp()[0].detach().cpu().numpy() - origin
        quat = self.env.robot.data.body_link_pose_w.torch[0, self.env.hand_id, 3:].detach().cpu().numpy()
        if not all(np.isfinite(a).all() for a in (basket, cup, tcp, quat)):
            raise ValueError("Nonfinite measured basket handoff geometry.")
        return basket, tcp, Rotation.from_quat(quat), cup

    def _pour_actions(self, valid: bool, held: bool, fruit_count: int) -> torch.Tensor:
        """Command the measured pouring reference without changing physical state."""
        basket, tcp, hand, cup = self._measured_pose()
        sample = self._pour.step(
            basket_position=basket[:3],
            basket_quaternion=basket[3:],
            tcp_position=tcp,
            hand_quaternion=hand.as_quat(),
            cup_position=cup[:3],
            cup_quaternion=cup[3:],
            held=held,
            valid=valid,
            delivered_count=fruit_count,
        )
        target = self.env.tcp().clone()
        target[0] = target.new_tensor(sample["tcp"]) + self.env.scene.env_origins[0]
        rotation = target.new_tensor(sample["hand_quat"])[None]
        self.stage = "measured_pour_" + sample["diagnostics"]["phase"]
        self.diagnostics.update(sample["diagnostics"])
        actions = self.pose_controller.compute(target, rotation, True)
        self.diagnostics["raw_gripper_sign"] = float(actions[0, 7])
        return actions

    def _return_actions(self, m: dict, state: dict, valid: bool, held: bool) -> torch.Tensor:
        _, tcp, hand, _ = self._measured_pose()
        t = self._return_elapsed
        supported = bool(_one(m["supported"]))
        support_ready = (
            valid
            and bool(_one(state["controlled_descent"]))
            and supported
            and float(_one(state["support_hold_s"])) >= OPTIONS["support_hold_s"] - 1e-6
        )
        paused = False
        if t >= 34.0 and not self._open_started:
            if support_ready:
                self._open_started = True
            else:
                t, paused = 34.0 - 1e-6, True
        if t >= 37.0 and not (valid and bool(_one(m["open"])) and supported):
            t, paused = 37.0 - 1e-6, True
        sample = self._return.sample(t)
        error_p = float(np.linalg.norm(sample["tcp_position"] - tcp))
        error_r = float((Rotation.from_quat(sample["hand_xyzw"]) * hand.inv()).magnitude())
        tracking = (
            error_p < OPTIONS["tracking_position_tolerance_m"] and error_r < OPTIONS["tracking_rotation_tolerance_rad"]
        )
        advance = valid and tracking and (t >= 29.0 or held) and not paused
        self._return_elapsed = min(t + POLICY_DT_S, OPTIONS["return_nominal_duration_s"]) if advance else t
        self.stage = "return_" + sample["phase"]
        if paused:
            self.stage = "await_measured_open" if self._open_started else "await_supported_setdown"
        if (
            bool(_one(state["finished"]))
            and valid
            and supported
            and bool(_one(m["open"]))
            and bool(_one(m["separated"]))
        ):
            self.stage = "basket_released"
        target = self.env.tcp().clone()
        target[0] = target.new_tensor(sample["tcp_position"]) + self.env.scene.env_origins[0]
        rotation = target.new_tensor(sample["hand_xyzw"])[None]
        actions = self.pose_controller.compute(target, rotation, not self._open_started)
        self.diagnostics.update(
            return_elapsed_s=t,
            return_clock_advanced=advance,
            support_ready=support_ready,
            open_started=self._open_started,
            measured_open=bool(_one(m["open"])),
            measured_separated=bool(_one(m["separated"])),
            tracking_position_error_m=error_p,
            tracking_rotation_error_rad=error_r,
            raw_gripper_sign=float(actions[0, 7]),
        )
        return actions
