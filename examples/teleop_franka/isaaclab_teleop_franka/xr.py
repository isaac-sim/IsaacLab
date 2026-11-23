# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Map Quest controller poses to Franka Reach's existing absolute IK action.

The Capture pipeline supplies tracked world poses and calibration boundaries.
The action input reads robot state on the simulation thread and establishes a
measured end-effector reference when control begins. Subsequent controller motion
changes an absolute target, so holding the controller still preserves the target
while the robot converges. Both Capture and Isaac Lab use XYZW quaternions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
from isaaccapture.retargeting_engine.deviceio_source_nodes import ControllersSource
from isaaccapture.retargeting_engine.interface import (
    BaseRetargeter,
    ComputeContext,
    ExecutionState,
    OptionalTensorGroup,
    OptionalType,
    OutputCombiner,
    RetargeterIO,
    RetargeterIOType,
    TensorGroupType,
    ValueInput,
)
from isaaccapture.retargeting_engine.tensor_types import (
    ControllerInput,
    ControllerInputIndex,
    DLDataType,
    NDArrayType,
    TransformMatrix,
)
from isaaclab_teleop.input import TeleopInput

from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab_teleop import TeleopInputCfg

    from isaaclab.envs import ManagerBasedEnv


def controller_pose_pipeline() -> OutputCombiner:
    """Read the right controller's grip pose in the simulation world frame."""
    source = ControllersSource("xr_tracking")
    anchor = ValueInput("world_T_anchor", TransformMatrix())
    transformed = source.transformed(anchor.output(ValueInput.VALUE))
    pose = _ControllerPoseSource().connect({ControllersSource.RIGHT: transformed.output(ControllersSource.RIGHT)})
    return OutputCombiner({"action": pose.output("pose")})


class _ControllerPoseSource(BaseRetargeter):
    """Expose poses and persistent calibration epochs without accessing the robot.

    Output is ``[x, y, z, qx, qy, qz, qw, valid, epoch]`` as float32. A pose is
    valid only while tracking and execution are active. The epoch changes on
    reset, execution transitions, and tracking loss or recovery. Persisting the
    epoch lets the simulation detect these boundaries even when asynchronous
    sampling skips an intervening invalid frame.
    """

    def __init__(self) -> None:
        self._was_running = False
        self._was_tracked = False
        self._epoch = 0
        super().__init__(name="right_absolute_pose")

    def input_spec(self) -> RetargeterIOType:
        return {ControllersSource.RIGHT: OptionalType(ControllerInput())}

    def output_spec(self) -> RetargeterIOType:
        return {
            "pose": TensorGroupType(
                "tracked_pose",
                [NDArrayType("pose_with_status", shape=(9,), dtype=DLDataType.FLOAT, dtype_bits=32)],
            )
        }

    def _read_pose(self, sample: OptionalTensorGroup) -> np.ndarray | None:
        if sample.is_none or not bool(sample[ControllerInputIndex.GRIP_IS_VALID]):
            return None
        position = np.from_dlpack(sample[ControllerInputIndex.GRIP_POSITION])
        orientation = np.from_dlpack(sample[ControllerInputIndex.GRIP_ORIENTATION]).astype(np.float64)
        # Normalize in float64 to avoid overflow from finite float32 components.
        norm = float(np.linalg.norm(orientation))
        if not np.isfinite(position).all() or not np.isfinite(norm) or norm <= 1.0e-8:
            return None
        return np.concatenate((position, orientation / norm)).astype(np.float32)

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context: ComputeContext) -> None:
        pose = self._read_pose(inputs[ControllersSource.RIGHT])
        tracked = pose is not None
        running = context.execution_events.execution_state == ExecutionState.RUNNING
        if context.execution_events.reset or running != self._was_running or tracked != self._was_tracked:
            # Preserve exact integer representation in the float32 action vector.
            self._epoch = (self._epoch + 1) % (1 << 24)
        self._was_running = running
        self._was_tracked = tracked

        result = np.zeros(9, dtype=np.float32)
        result[6] = 1.0
        if pose is not None:
            result[:7] = pose
        result[7] = float(tracked and running)
        result[8] = self._epoch
        outputs["pose"][0] = result


class _AbsolutePoseMapping:
    """Keep controller and measured robot references for one engagement."""

    def __init__(self, position_scale: float = 1.0) -> None:
        self.position_scale = position_scale
        self.reset()

    def reset(self) -> None:
        """Latch fresh references on the next valid sample."""
        self._epoch: int | None = None
        self._controller_reference: torch.Tensor | None = None
        self._robot_reference: torch.Tensor | None = None

    def target(self, controller_pose: torch.Tensor, robot_pose: torch.Tensor, epoch: int) -> torch.Tensor:
        """Map controller displacement from engagement to an absolute base-frame pose."""
        if self._epoch != epoch:
            self._epoch = epoch
            self._controller_reference = controller_pose.clone()
            self._robot_reference = robot_pose.clone()
        assert self._controller_reference is not None and self._robot_reference is not None

        position = self._robot_reference[:, :3] + self.position_scale * (
            controller_pose[:, :3] - self._controller_reference[:, :3]
        )
        rotation = math_utils.quat_mul(controller_pose[:, 3:], math_utils.quat_inv(self._controller_reference[:, 3:]))
        orientation = math_utils.normalize(math_utils.quat_mul(rotation, self._robot_reference[:, 3:]))
        return torch.cat((position, orientation), dim=-1)


class CalibratedAbsolutePoseInput(TeleopInput):
    """Calibrate tracked poses against the task's measured end-effector pose.

    Controller displacement maps to end-effector displacement at 1:1 scale;
    controller rotation left-composes with the engagement orientation. The
    original action term retains its controller, scale, offset, and IK settings.

    Pausing, resetting, or losing tracking clears calibration and returns the
    caller's actions. Checkpoint-free playback supplies measured-pose hold
    commands for this absolute IK term. Policy playback instead resumes its
    policy. Robot state is read only here, on the simulation thread.
    """

    def __init__(self, cfg: TeleopInputCfg, env: ManagerBasedEnv, *, action_dim: int) -> None:
        super().__init__(cfg, env, action_dim=action_dim)
        self._mapping = _AbsolutePoseMapping()
        action_cfg = env.action_manager.get_term("arm_action").cfg
        if (
            not isinstance(action_cfg, DifferentialInverseKinematicsActionCfg)
            or action_dim != 7
            or action_cfg.controller.use_relative_mode
            or len(cfg.env_ids) != 1
        ):
            raise ValueError("Franka XR input requires one absolute differential IK pose action.")
        if action_cfg.body_offset is not None:
            raise ValueError("Franka XR input maps the target body directly and requires no body offset.")
        self._asset = env.scene[action_cfg.asset_name]
        bodies, _ = self._asset.find_bodies(action_cfg.body_name)
        if len(bodies) != 1:
            raise ValueError("Franka XR input requires exactly one target body.")
        self._body = bodies[0]
        self._scale = torch.as_tensor(action_cfg.scale, device=env.device)
        self._offset = torch.as_tensor(action_cfg.offset, device=env.device)
        if torch.any(self._scale == 0):
            raise ValueError("Franka XR input cannot invert a zero action scale.")

    def resolve_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Replace the selected row with a calibrated absolute pose while active."""
        sample = self.sample
        if sample is None or sample.reset or not sample.active or sample.actions is None:
            self._mapping.reset()
            return actions
        tracked = sample.actions.reshape(-1)
        if tracked.shape != (9,) or not torch.isfinite(tracked).all():
            raise ValueError("Expected a finite Capture pose, validity flag, and calibration epoch.")
        if not bool(tracked[7]):
            self._mapping.reset()
            return actions

        ids = self._env_ids
        root_position = self._asset.data.root_pos_w.torch[ids]
        root_orientation = self._asset.data.root_quat_w.torch[ids]
        position, orientation = math_utils.subtract_frame_transforms(
            root_position, root_orientation, tracked[None, :3], tracked[None, 3:7]
        )
        robot_position, robot_orientation = math_utils.subtract_frame_transforms(
            root_position,
            root_orientation,
            self._asset.data.body_pos_w.torch[ids, self._body],
            self._asset.data.body_quat_w.torch[ids, self._body],
        )
        target = self._mapping.target(
            torch.cat((position, orientation), dim=-1),
            torch.cat((robot_position, robot_orientation), dim=-1),
            epoch=int(tracked[8]),
        )
        result = actions.clone()
        result[ids] = ((target - self._offset) / self._scale).to(actions)
        return result

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Clear calibration when the environment resets."""
        super().reset(env_ids)
        self._mapping.reset()
