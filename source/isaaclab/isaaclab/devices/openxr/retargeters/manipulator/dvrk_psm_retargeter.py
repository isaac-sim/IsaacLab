# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bimanual dVRK PSM retargeting from OpenXR motion controllers.

This adapter owns four simulator-independent Isaac Teleop state machines: one
Cartesian clutch and one paired-jaw intent machine for each PSM. It emits a
stable 18-dimensional command in left-pose, left-jaws, right-pose, right-jaws
order. A held command only freezes the target emitted by this adapter; it does
not disable downstream actuator drives or latch measured robot state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg


@dataclass(frozen=True)
class DVRKPSMSideRetargeterCfg:
    """World-frame control configuration for one dVRK PSM."""

    home_position: tuple[float, float, float]
    """World-frame tool-tip home position."""

    home_orientation: tuple[float, float, float, float]
    """World-frame tool-tip home orientation as a scalar-last ``xyzw`` quaternion."""

    workspace_lower: tuple[float, float, float]
    """Inclusive world-frame lower workspace bound."""

    workspace_upper: tuple[float, float, float]
    """Inclusive world-frame upper workspace bound."""

    jaw_open: tuple[float, float]
    """Ordered open targets for gripper joints one and two, in radians."""

    jaw_closed: tuple[float, float]
    """Ordered closed targets for gripper joints one and two, in radians."""

    translation_scale: float = 1.0
    orientation_offset: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    initial_closedness: float = 0.0
    """Exact per-side jaw closedness restored by construction and RESET."""

    clutch_threshold: float = 0.5
    trigger_deadband: float = 0.05
    opening_intent_duration_s: float = 0.1


@dataclass
class _SideKernels:
    pose: Any
    jaw: Any


class DVRKPSMRetargeter(RetargeterBase):
    """Retarget two OpenXR controllers to an ordered 18-D dVRK PSM command.

    The left and right sides are validated and advanced independently. Tracking
    loss, malformed input, or clutch release atomically holds that side's full
    nine-dimensional pose-and-jaw command. Session lifecycle is explicit:
    :meth:`start` enables fresh controller-origin capture, :meth:`stop` holds
    and disengages all kernels, and :meth:`reset` restores configured home
    targets without changing session activity.
    """

    _POSE_LENGTH = 7
    _JAW_LENGTH = 2
    _CONTROLLER_SHAPE = (2, 7)
    _MIN_QUATERNION_NORM = 1.0e-6

    def __init__(self, cfg: DVRKPSMRetargeterCfg):
        """Initialise the adapter and lazily load the Isaac Teleop kernels."""
        super().__init__(cfg)
        if cfg.left is None or cfg.right is None:
            raise ValueError("DVRKPSMRetargeterCfg requires explicit left and right side configurations")

        try:
            from isaacteleop.retargeters.DVRK.control import (
                DVRKPSMCartesianClutchConfig,
                DVRKPSMCartesianClutchStateMachine,
                DVRKPSMJawIntentConfig,
                DVRKPSMJawIntentStateMachine,
            )
        except ImportError as error:
            raise ModuleNotFoundError(
                "DVRKPSMRetargeter requires an Isaac Teleop release containing the dVRK control kernels from "
                "NVIDIA/IsaacTeleop PR 769"
            ) from error

        def build_side(side_cfg: DVRKPSMSideRetargeterCfg) -> _SideKernels:
            pose = DVRKPSMCartesianClutchStateMachine(
                DVRKPSMCartesianClutchConfig(
                    home_position=side_cfg.home_position,
                    home_orientation=side_cfg.home_orientation,
                    workspace_lower=side_cfg.workspace_lower,
                    workspace_upper=side_cfg.workspace_upper,
                    translation_scale=side_cfg.translation_scale,
                    orientation_offset=side_cfg.orientation_offset,
                    clutch_threshold=side_cfg.clutch_threshold,
                )
            )
            jaw = DVRKPSMJawIntentStateMachine(
                DVRKPSMJawIntentConfig(
                    jaw_open=side_cfg.jaw_open,
                    jaw_closed=side_cfg.jaw_closed,
                    initial_closedness=side_cfg.initial_closedness,
                    clutch_threshold=side_cfg.clutch_threshold,
                    trigger_deadband=side_cfg.trigger_deadband,
                    opening_intent_duration_s=side_cfg.opening_intent_duration_s,
                )
            )
            return _SideKernels(pose=pose, jaw=jaw)

        self._left = build_side(cfg.left)
        self._right = build_side(cfg.right)
        self._session_active = False
        self._last_jaw_time_ns: dict[str, int | None] = {"left": None, "right": None}
        self._left_pose = self._left.pose.reset()
        self._left_jaws = self._left.jaw.reset()
        self._right_pose = self._right.pose.reset()
        self._right_jaws = self._right.jaw.reset()

    @property
    def session_active(self) -> bool:
        """Whether controller samples may currently change command targets."""
        return self._session_active

    @property
    def action(self) -> torch.Tensor:
        """Return the currently held contiguous 18-D command."""
        return self._as_tensor()

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        """Request only motion-controller data from the OpenXR device."""
        return [RetargeterBase.Requirement.MOTION_CONTROLLER]

    def start(self) -> None:
        """Activate the session; the next squeezed sample captures a fresh origin."""
        self._session_active = True

    def stop(self) -> None:
        """Deactivate the session, disengage every kernel, and hold its command."""
        self._session_active = False
        self._left_pose = self._inactive_pose_step(self._left)
        self._left_jaws = self._inactive_jaw_step(self._left)
        self._right_pose = self._inactive_pose_step(self._right)
        self._right_jaws = self._inactive_jaw_step(self._right)
        self._clear_jaw_clocks()

    def reset(self) -> None:
        """Restore configured home targets while preserving session activity."""
        self._left_pose = self._left.pose.reset()
        self._left_jaws = self._left.jaw.reset()
        self._right_pose = self._right.pose.reset()
        self._right_jaws = self._right.jaw.reset()
        self._clear_jaw_clocks()

    def retarget(self, data: Any) -> torch.Tensor:
        """Advance both independent PSM sides and return the ordered 18-D command."""
        now_ns = time.monotonic_ns()
        self._left_pose, self._left_jaws = self._retarget_side(
            data,
            DeviceBase.TrackingTarget.CONTROLLER_LEFT,
            self._left,
            self._elapsed_seconds("left", now_ns),
        )
        self._right_pose, self._right_jaws = self._retarget_side(
            data,
            DeviceBase.TrackingTarget.CONTROLLER_RIGHT,
            self._right,
            self._elapsed_seconds("right", now_ns),
        )
        return self._as_tensor()

    def _retarget_side(
        self,
        data: Any,
        target: DeviceBase.TrackingTarget,
        kernels: _SideKernels,
        dt_seconds: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        valid, position, orientation, trigger, squeeze = self._parse_controller(data, target)
        pose = kernels.pose.step(
            controller_position=position if valid else None,
            controller_orientation=orientation if valid else None,
            squeeze=squeeze if valid else None,
            tracking_valid=valid,
            session_active=self._session_active,
        )
        jaws = kernels.jaw.step(
            trigger=trigger if valid else None,
            squeeze=squeeze if valid else None,
            tracking_valid=valid,
            session_active=self._session_active,
            dt_seconds=dt_seconds,
        )
        return pose, jaws

    @classmethod
    def _parse_controller(
        cls, data: Any, target: DeviceBase.TrackingTarget
    ) -> tuple[bool, np.ndarray | None, np.ndarray | None, float | None, float | None]:
        try:
            controller = np.asarray(data[target], dtype=np.float64)
        except (KeyError, OverflowError, TypeError, ValueError):
            return False, None, None, None, None
        if controller.shape != cls._CONTROLLER_SHAPE:
            return False, None, None, None, None

        pose = controller[DeviceBase.MotionControllerDataRowIndex.POSE.value]
        inputs = controller[DeviceBase.MotionControllerDataRowIndex.INPUTS.value]
        trigger = inputs[DeviceBase.MotionControllerInputIndex.TRIGGER.value]
        squeeze = inputs[DeviceBase.MotionControllerInputIndex.SQUEEZE.value]
        pose_valid = inputs[DeviceBase.MotionControllerInputIndex.POSE_VALID.value]
        consumed = np.concatenate((pose, (trigger, squeeze, pose_valid)))
        if not np.all(np.isfinite(consumed)) or pose_valid != 1.0:
            return False, None, None, None, None

        # IsaacLab receives scalar-first wxyz; Isaac Teleop consumes scalar-last xyzw.
        orientation = np.asarray((pose[4], pose[5], pose[6], pose[3]), dtype=np.float64)
        scale = float(np.max(np.abs(orientation)))
        if scale == 0.0:
            return False, None, None, None, None
        scaled_norm = float(np.linalg.norm(orientation / scale))
        if not np.isfinite(scaled_norm) or scale < cls._MIN_QUATERNION_NORM / scaled_norm:
            return False, None, None, None, None
        orientation /= scale * scaled_norm

        return True, pose[:3].copy(), orientation, float(trigger), float(squeeze)

    @staticmethod
    def _inactive_pose_step(kernels: _SideKernels) -> np.ndarray:
        return kernels.pose.step(
            controller_position=None,
            controller_orientation=None,
            squeeze=None,
            tracking_valid=False,
            session_active=False,
        )

    @staticmethod
    def _inactive_jaw_step(kernels: _SideKernels) -> np.ndarray:
        return kernels.jaw.step(
            trigger=None,
            squeeze=None,
            tracking_valid=False,
            session_active=False,
            dt_seconds=0.0,
        )

    def _elapsed_seconds(self, side: str, now_ns: int) -> float:
        previous_ns = self._last_jaw_time_ns[side]
        if previous_ns is None:
            self._last_jaw_time_ns[side] = now_ns
            return 0.0
        if now_ns <= previous_ns:
            return 0.0
        self._last_jaw_time_ns[side] = now_ns
        return (now_ns - previous_ns) / 1_000_000_000.0

    def _clear_jaw_clocks(self) -> None:
        self._last_jaw_time_ns["left"] = None
        self._last_jaw_time_ns["right"] = None

    def _as_tensor(self) -> torch.Tensor:
        action = np.concatenate((self._left_pose, self._left_jaws, self._right_pose, self._right_jaws))
        if action.shape != (18,) or not np.all(np.isfinite(action)):
            raise RuntimeError("dVRK retargeting kernels emitted an invalid command")
        return torch.tensor(action, dtype=torch.float32, device=self._sim_device).contiguous()


@dataclass
class DVRKPSMRetargeterCfg(RetargeterCfg):
    """Configuration for the bimanual dVRK PSM retargeter."""

    left: DVRKPSMSideRetargeterCfg | None = None
    right: DVRKPSMSideRetargeterCfg | None = None
    retargeter_type: type[RetargeterBase] = DVRKPSMRetargeter


__all__ = ["DVRKPSMRetargeter", "DVRKPSMRetargeterCfg", "DVRKPSMSideRetargeterCfg"]
