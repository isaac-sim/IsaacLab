# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Apple Vision Pro wrist control and geometric DexPilot finger retargeting.

The frame calibration and pinned DexPilot assets come from the workstation's
Rizon/Sharpa RJ45 task. See ``assets/NOTICE`` for upstream asset provenance.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.utils import math as math_utils

from .robot import HAND_HOLD_POSITIONS, HAND_JOINT_LIMITS, HAND_JOINT_NAMES

if TYPE_CHECKING:
    from isaacteleop.retargeting_engine.interface import OutputCombiner

HANDTRACKING_TO_BASELINK = (0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0)
"""Official NVIDIA OpenXR-to-Sharpa kinematic hand basis, flattened row-major."""

OPENXR_TO_PALM_RPY_DEG = (90.28118057, 2.77868426, -101.70219973)
"""OpenXR wrist to Fabrics-Sim ``r_palm_ctrl`` intrinsic XYZ offset [deg]."""

SHARPA_URDF_REVISION = "3e953f588ba9954cebaa720aaa4cee06a43a068e"
SHARPA_URDF_SHA256 = "7a9ab7f824482d23765b2da40b7e96fc605e7e70eda4615a5ca51fea88afb845"


def right_hand_urdf() -> Path:
    """Resolve pinned Sharpa kinematics for DexPilot without visual meshes.

    Set ``ISAACLAB_SHARPA_RIGHT_URDF`` to override the default asset cache. See
    the task README for download instructions. A derived cache removes unused
    visual/collision elements so the optimizer needs no STL bundle. Joint
    origins, limits, inertias and frames are retained; the source is unchanged.
    Neither model is redistributed.
    """
    default = Path.home() / ".cache/isaaclab/sharpa" / SHARPA_URDF_REVISION / "right_sharpa_wave.urdf"
    path = Path(os.environ.get("ISAACLAB_SHARPA_RIGHT_URDF", default)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Sharpa retargeting URDF not found at {path}. Follow the coffee task README's asset setup or set"
            " ISAACLAB_SHARPA_RIGHT_URDF. Scripted operation does not require this model."
        )
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHARPA_URDF_SHA256:
        raise ValueError(f"Sharpa retargeting URDF at {path} does not match pinned revision {SHARPA_URDF_REVISION}.")
    model = ET.parse(path).getroot()
    for link in model.findall("link"):
        for geometry in link.findall("visual") + link.findall("collision"):
            link.remove(geometry)
    content = ET.tostring(model, encoding="utf-8", xml_declaration=True)
    kinematics = default.with_name("right_sharpa_wave_kinematic.urdf")
    if not kinematics.is_file() or kinematics.read_bytes() != content:
        kinematics.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=kinematics.parent, suffix=".urdf", delete=False) as temporary:
            temporary.write(content)
        os.replace(temporary.name, kinematics)
    return kinematics


class CoffeeTeleopController:
    """Recenter absolute XR input and limit commanded palm speed.

    The first valid sample anchors to the measured simulated palm. Translation
    and orientation deltas then use the calibrated robot-base axes. Missing
    wrist tracking holds the last target; explicit reset establishes a new
    anchor. Limits are 0.25 m/s translation and 1.2 rad/s rotation.
    """

    def __init__(self, env):
        if env.num_envs != 1:
            raise ValueError("Vision Pro operation requires one coffee scene.")
        self._env = env
        self._limits = torch.tensor(HAND_JOINT_LIMITS, device=env.device)
        self.reset()

    def reset(self) -> None:
        """Re-clutch to the measured palm on the next valid XR sample."""
        from .grasp_reset import configured_hand_targets

        self._operator_pose = None
        self._robot_pose = self._env.palm_pose().clone()
        hand = torch.tensor(configured_hand_targets(self._env.cfg), device=self._env.device).reshape(1, -1)
        self._held = torch.cat((self._robot_pose, hand), dim=-1)

    def compute(self, sample: torch.Tensor) -> torch.Tensor:
        """Return one bounded, recentered palm pose and 22 hand joint targets."""
        sample = sample.reshape(1, -1).to(self._env.device)
        if sample.shape[1] != 29:
            raise ValueError("Expected a palm pose and 22 Sharpa finger angles.")
        fingers = sample[:, 7:]
        self._held[:, 7:] = torch.where(
            torch.isfinite(fingers), fingers.clamp(self._limits[:, 0], self._limits[:, 1]), self._held[:, 7:]
        )
        pose = sample[:, :7].clone()
        valid = torch.isfinite(pose).all() & (pose[:, :3].norm() > 1.0e-6) & (pose[:, 3:].norm() > 1.0e-6)
        if not bool(valid):
            return self._held.clone()
        pose[:, 3:] /= pose[:, 3:].norm(dim=-1, keepdim=True)
        if self._operator_pose is None:
            self._operator_pose = pose
            return self._held.clone()
        target_pos = self._robot_pose[:, :3] + pose[:, :3] - self._operator_pose[:, :3]
        displacement = target_pos - self._held[:, :3]
        distance = displacement.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        self._held[:, :3] += displacement * (0.25 * self._env.step_dt / distance).clamp(max=1.0)
        delta = math_utils.quat_mul(pose[:, 3:], math_utils.quat_inv(self._operator_pose[:, 3:]))
        target_quat = math_utils.quat_mul(delta, self._robot_pose[:, 3:])
        error = math_utils.quat_unique(math_utils.quat_mul(target_quat, math_utils.quat_inv(self._held[:, 3:7])))
        axis_angle = math_utils.axis_angle_from_quat(error)
        angle = axis_angle.norm(dim=-1)
        increment = math_utils.quat_from_angle_axis(
            angle.clamp(max=1.2 * self._env.step_dt), axis_angle / angle[:, None].clamp_min(1.0e-8)
        )
        self._held[:, 3:7] = math_utils.quat_mul(increment, self._held[:, 3:7])
        return self._held.clone()


def build_teleop_pipeline() -> OutputCombiner:
    """Return ``palm xyz/xyzw + 22 finger radians`` from a tracked right hand.

    The arm pose uses the XR anchor transform; DexPilot consumes the raw hand
    skeleton and removes wrist motion itself. Reapplying the XR transform to
    fingers would mix coordinate systems. A missing/invalid wrist produces a
    zero-position sentinel, which :class:`CoffeeTeleopController` interprets as hold.
    Missing finger joints hold the last complete hand command.
    """
    from isaacteleop.retargeters import (
        DexHandRetargeter,
        DexHandRetargeterConfig,
        Se3AbsRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import HandsSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import HandInputIndex, HandJointIndex, TransformMatrix

    def valid_tracking(tracked, indices) -> bool:
        if tracked.is_none:
            return False
        flags = np.from_dlpack(tracked[HandInputIndex.JOINT_VALID])
        positions = np.from_dlpack(tracked[HandInputIndex.JOINT_POSITIONS])
        wrist_rotation = np.from_dlpack(tracked[HandInputIndex.JOINT_ORIENTATIONS])[HandJointIndex.WRIST]
        return bool(
            np.all(flags[indices] > 0)
            and np.isfinite(positions[indices]).all()
            and np.isfinite(wrist_rotation).all()
            and np.linalg.norm(wrist_rotation) > 1.0e-6
        )

    class TrackedPalmRetargeter(Se3AbsRetargeter):
        """Expose tracking loss instead of retaining a pre-reset wrist target."""

        def _compute_fn(self, inputs, outputs, context) -> None:
            tracked = inputs[self._config.input_device]
            if not valid_tracking(tracked, [HandJointIndex.WRIST]):
                outputs["ee_pose"][0] = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
                return
            super()._compute_fn(inputs, outputs, context)

    class TrackedSharpaRetargeter(DexHandRetargeter):
        """Retain a complete hand command through optical occlusion."""

        def __init__(self, config, name: str) -> None:
            super().__init__(config, name)
            self._held_command = np.asarray(HAND_HOLD_POSITIONS, dtype=np.float64).copy()
            self._limits = np.asarray(HAND_JOINT_LIMITS)

        def _compute_fn(self, inputs, outputs, context) -> None:
            if context.execution_events.reset:
                self._held_command[:] = HAND_HOLD_POSITIONS
            tracked = inputs[HandsSource.RIGHT]
            if valid_tracking(tracked, np.asarray(self._hand_joints_index, dtype=np.int64)):
                super()._compute_fn(inputs, outputs, context)
                candidate = np.asarray([outputs["hand_joints"][i] for i in range(len(HAND_JOINT_NAMES))])
                if np.isfinite(candidate).all():
                    self._held_command[:] = np.clip(candidate, self._limits[:, 0], self._limits[:, 1])
            for index, angle in enumerate(self._held_command):
                outputs["hand_joints"][index] = float(angle)

    assets = Path(__file__).with_name("assets")
    hands = HandsSource(name="hands")
    anchor = ValueInput("world_T_anchor", TransformMatrix())
    transformed_hands = hands.transformed(anchor.output(ValueInput.VALUE))
    palm = TrackedPalmRetargeter(
        Se3RetargeterConfig(
            input_device=HandsSource.RIGHT,
            zero_out_xy_rotation=False,
            use_wrist_rotation=True,
            use_wrist_position=True,
            target_offset_roll=OPENXR_TO_PALM_RPY_DEG[0],
            target_offset_pitch=OPENXR_TO_PALM_RPY_DEG[1],
            target_offset_yaw=OPENXR_TO_PALM_RPY_DEG[2],
        ),
        name="right_palm",
    )
    palm_output = palm.connect({HandsSource.RIGHT: transformed_hands.output(HandsSource.RIGHT)}).output("ee_pose")
    hand = TrackedSharpaRetargeter(
        DexHandRetargeterConfig(
            hand_retargeting_config=str(assets / "sharpa_wave_right_dexpilot.yml"),
            hand_urdf=str(right_hand_urdf()),
            hand_joint_names=list(HAND_JOINT_NAMES),
            handtracking_to_baselink_frame_transform=HANDTRACKING_TO_BASELINK,
            hand_side="right",
        ),
        name="right_sharpa_dexpilot",
    )
    hand_output = hand.connect({HandsSource.RIGHT: hands.output(HandsSource.RIGHT)}).output("hand_joints")
    pose_elements = ["pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w"]
    reorderer = TensorReorderer(
        input_config={"right_palm": pose_elements, "right_hand": list(HAND_JOINT_NAMES)},
        output_order=pose_elements + list(HAND_JOINT_NAMES),
        name="coffee_action",
        input_types={"right_palm": "array", "right_hand": "scalar"},
    )
    output = reorderer.connect({"right_palm": palm_output, "right_hand": hand_output})
    return OutputCombiner({"action": output.output("output")})
