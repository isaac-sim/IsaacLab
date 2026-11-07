# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment and motion replay for the coffee kitchen camera benchmark."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import newton
import torch
from isaaclab_newton.cloner import newton_builder_world_hook

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ActionTerm
from isaaclab.utils import math as math_utils

from isaaclab_tasks.benchmark.benchmark_mode import (
    BENCHMARK_MODES,
    check_direct_pose_cfg,
    check_direct_pose_rendering,
)
from isaaclab_tasks.contrib.rizon_sharpa_coffee.attachment import MountedSpoonAttachment
from isaaclab_tasks.contrib.rizon_sharpa_coffee.robot import (
    DEFAULT_SPOON_OFFSET_POS,
    DEFAULT_SPOON_OFFSET_ROT,
    PALM_BODY_NAME,
)

if TYPE_CHECKING:
    from .coffee_kitchen_camera_env_cfg import CoffeeKitchenCameraEnvCfg, CoffeeMotionPlaybackCfg


def spread_offsets(num_envs: int, length: int, seed: int) -> torch.Tensor:
    """Start frames spread evenly over a recording of ``length`` frames, assigned in a seeded random order.

    A dedicated generator keeps the order reproducible, whatever else used the global one.
    """
    offsets = torch.arange(num_envs) * length // num_envs
    return offsets[torch.randperm(num_envs, generator=torch.Generator().manual_seed(seed))]


def frame_indices(offsets: torch.Tensor, step: int, length: int) -> torch.Tensor:
    """The recording frame each environment shows at a step counter: its offset plus the step, looped."""
    return (offsets + step) % length


class CoffeeMotionPlayback(ActionTerm):
    """Replay the demo's recorded arm motion in a loop, from a different time offset per environment.

    The term takes no actions. At step counter ``n`` an environment shows recording frame ``(offset + n) % L``. The
    offsets are spread evenly over the recording and assigned to the environments in a random order. The hand keeps
    the demo's grip, its default pose.

    * ``"render"`` mode writes each frame's recorded pose after physics (:func:`pose_from_recording`); physics holds
      the pose with matching actuator targets.
    * ``"physics_render"`` mode sends the demo's recorded actuator targets, one per physics substep, so MJWarp moves
      the arm as it did in the demo. An environment that wraps from the last frame to the first is placed on the first
      frame directly, as a reset would.
    """

    cfg: CoffeeMotionPlaybackCfg

    def __init__(self, cfg: CoffeeMotionPlaybackCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        motion = json.loads(Path(cfg.motion_file).read_text(encoding="utf-8"))
        if motion["decimation"] != env.cfg.decimation or round(motion["rate_hz"] * env.step_dt, 6) != 1.0:
            raise ValueError(
                f"The motion is recorded at {motion['rate_hz']} Hz with decimation {motion['decimation']}; the"
                f" environment steps every {env.step_dt} s with decimation {env.cfg.decimation}."
            )
        # The recorded targets are per substep, so each substep must reach apply_actions.
        if env.sim.backend_manager.handles_decimation():
            raise ValueError("Physics that runs the decimation loop itself cannot replay per-substep targets.")
        self._positions = torch.tensor(motion["positions"], device=self.device)
        self._velocities = torch.tensor(motion["velocities"], device=self.device)
        self._targets = torch.tensor(motion["targets"], device=self.device)
        self._length = len(self._positions)
        self._arm_ids = torch.tensor(
            [self._asset.joint_names.index(name) for name in motion["joint_names"]], device=self.device
        )
        self._offsets_host = spread_offsets(self.num_envs, self._length, env.cfg.seed or 0)
        self._offsets = self._offsets_host.to(self.device)
        self._default_pose = self._asset.data.default_joint_pos.torch.clone()
        self._physics_render = env.cfg.benchmark_mode == "physics_render"
        self._step_targets: torch.Tensor | None = None
        self._wrapping_ids: list[int] = []
        self._substep = 0
        self._actions = torch.zeros(self.num_envs, 0, device=self.device)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._actions

    def process_actions(self, actions: torch.Tensor) -> None:
        """Prepare this step's per-substep targets; runs once per environment step, before physics."""
        self._substep = 0
        if not self._physics_render:
            return
        step = self._env.common_step_counter
        # The last frame has no recorded target: those environments restart at the first frame instead.
        last_frame = frame_indices(self._offsets_host, step, self._length) == self._length - 1
        self._wrapping_ids = last_frame.nonzero().flatten().tolist()
        frames = frame_indices(self._offsets, step, self._length).clamp(max=self._length - 2)
        self._step_targets = self._targets[frames]
        if self._wrapping_ids:
            self._step_targets[self._wrapping_ids] = self._positions[0]

    def apply_actions(self) -> None:
        """Send this substep's recorded targets in ``"physics_render"`` mode; runs before every physics substep."""
        if not self._physics_render:
            return
        self._asset.actuators.target_command.set_position_index(
            value=self._step_targets[:, self._substep], joint_ids=self._arm_ids
        )
        if self._substep == 0 and self._wrapping_ids:
            self.write_pose(self._wrapping_ids, frame=0)
            # Update the kinematics here, so the physics step's timer does not include it.
            self._env.sim.forward()
        self._substep += 1

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        self.write_pose(env_ids)

    def write_pose(self, env_ids: Sequence[int] | torch.Tensor | None = None, frame: int | None = None) -> None:
        """Write a recorded frame's arm pose and hold it with matching actuator targets.

        Args:
            env_ids: Environments to write, or ``None`` for all.
            frame: Recording frame for every selected environment; by default, each one's frame at the current step.
        """
        selected = slice(None) if env_ids is None else env_ids
        if frame is None:
            frames = frame_indices(self._offsets[selected], self._env.common_step_counter, self._length)
        else:
            frames = torch.full_like(self._offsets[selected], frame)
        position = self._default_pose[selected].clone()
        position[:, self._arm_ids] = self._positions[frames]
        velocity = torch.zeros_like(position)
        # The demo's arm is moving at most frames; physics_render continues from that motion, render does not.
        if self._physics_render:
            velocity[:, self._arm_ids] = self._velocities[frames]
        self._asset.write_joint_position_to_sim_index(position=position, env_ids=env_ids)
        self._asset.write_joint_velocity_to_sim_index(velocity=velocity, env_ids=env_ids)
        self._asset.actuators.target_command.set_position_index(value=position, env_ids=env_ids)


def pose_from_recording(env: CoffeeKitchenCameraEnv, env_ids: torch.Tensor | None) -> None:
    """Interval event that poses the arm from the recording after physics in ``"render"`` mode."""
    if env.cfg.benchmark_mode == "render":
        env.pose_from_recording()


class CoffeeKitchenCameraEnv(ManagerBasedRLEnv):
    """The demo's environment without its powder: no contacts but the robot's, and the spoon mounted on the palm."""

    cfg: CoffeeKitchenCameraEnvCfg

    def __init__(self, cfg: CoffeeKitchenCameraEnvCfg, render_mode: str | None = None, **kwargs):
        # Any other mode would leave the arm frozen while the numbers still look plausible.
        if cfg.benchmark_mode not in BENCHMARK_MODES:
            raise ValueError(f"benchmark_mode must be one of {list(BENCHMARK_MODES)}, not {cfg.benchmark_mode!r}.")
        check_direct_pose_cfg(cfg.benchmark_mode, cfg.scene.lazy_sensor_update)
        self._attachment = None
        self._next_shape = 0
        with newton_builder_world_hook(self._disable_contacts):
            super().__init__(cfg, render_mode, **kwargs)
        check_direct_pose_rendering(self, cfg.benchmark_mode)

    def _disable_contacts(self, builder, env_id: int, _position, _quaternion) -> None:
        """Turn off shape contacts for everything but the robot, as the demo's ``_configure_colliders`` does."""
        for index in range(self._next_shape, builder.shape_count):
            if builder.shape_world[index] == env_id and "/Robot/" not in str(builder.shape_label[index]):
                builder.shape_flags[index] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
        self._next_shape = builder.shape_count

    def load_managers(self) -> None:
        # As in the demo, the spoon is placed and mounted before the managers.
        self._palm_id = self.scene["robot"].find_bodies(PALM_BODY_NAME)[0][0]
        self._tool_position = torch.tensor(DEFAULT_SPOON_OFFSET_POS, device=self.device).expand(self.num_envs, -1)
        self._tool_quaternion = torch.tensor(DEFAULT_SPOON_OFFSET_ROT, device=self.device).expand(self.num_envs, -1)
        self.sync_spoon()
        self._attachment = MountedSpoonAttachment(
            self.scene["robot"], self.scene["spoon"], DEFAULT_SPOON_OFFSET_POS, DEFAULT_SPOON_OFFSET_ROT
        )
        super().load_managers()

    def _reset_idx(self, env_ids: torch.Tensor | slice):
        super()._reset_idx(env_ids)
        # The mount moves the spoon only during physics steps; the frame rendered at reset needs it on the palm.
        self.sync_spoon(env_ids)

    def pose_from_recording(self) -> None:
        """Pose every arm from the recording after physics, carry the spoon along, and update the transforms."""
        self.action_manager.get_term("arm_motion").write_pose()
        self.sync_spoon()
        self.sim.forward()

    def sync_spoon(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Place the spoon on the palm, as the demo's ``sync_spoon`` does on construction and reset."""
        selected = slice(None) if env_ids is None else env_ids
        data = self.scene["robot"].data
        palm = data.body_link_pose_w.torch[selected, self._palm_id]
        pos, quat = math_utils.combine_frame_transforms(
            palm[:, :3], palm[:, 3:], self._tool_position[selected], self._tool_quaternion[selected]
        )
        velocity = data.body_link_vel_w.torch[selected, self._palm_id].clone()
        offset = pos - palm[:, :3]
        velocity[:, :3] += torch.cross(velocity[:, 3:], offset, dim=-1)
        spoon = self.scene["spoon"]
        spoon.write_root_link_pose_to_sim_index(root_pose=torch.cat((pos, quat), dim=-1), env_ids=env_ids)
        spoon.write_root_link_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)

    def close(self) -> None:
        if self._attachment is not None:
            self._attachment.close()
            self._attachment = None
        super().close()
