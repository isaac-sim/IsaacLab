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

from isaaclab_tasks.contrib.rizon_sharpa_coffee.attachment import MountedSpoonAttachment
from isaaclab_tasks.contrib.rizon_sharpa_coffee.robot import DEFAULT_SPOON_OFFSET_POS, DEFAULT_SPOON_OFFSET_ROT

if TYPE_CHECKING:
    from .coffee_kitchen_camera_env_cfg import CoffeeKitchenCameraEnvCfg, CoffeeMotionPlaybackCfg


class CoffeeMotionPlayback(ActionTerm):
    """Replay the demo's recorded arm motion from a random time offset per environment.

    The term takes no actions. Before every physics step it writes the recorded pose with zero velocity and holds
    it with the joint position targets; the hand keeps the demo's grip, its default pose. The recording plays
    forward, then backward, so it loops without a jump.
    """

    cfg: CoffeeMotionPlaybackCfg

    def __init__(self, cfg: CoffeeMotionPlaybackCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        motion = json.loads(Path(cfg.motion_file).read_text(encoding="utf-8"))
        if round(motion["rate_hz"] * env.step_dt, 6) != 1.0:
            raise ValueError(
                f"The motion is recorded at {motion['rate_hz']} Hz; the environment steps every {env.step_dt} s"
            )
        recorded = torch.tensor(motion["positions"], device=self.device)
        self._frames = torch.cat((recorded, recorded[1:-1].flip(0)))
        self._arm_ids = [self._asset.joint_names.index(name) for name in motion["joint_names"]]
        # A dedicated generator keeps the offsets reproducible, whatever else used the global one.
        generator = torch.Generator().manual_seed(env.cfg.seed or 0)
        self._offsets = torch.randint(len(self._frames), (self.num_envs,), generator=generator).to(self.device)
        self._default_pose = self._asset.data.default_joint_pos.torch.clone()
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
        pass

    def apply_actions(self) -> None:
        self._write_pose(None)

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        self._write_pose(env_ids)

    def _write_pose(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        selected = slice(None) if env_ids is None else env_ids
        frames = (self._offsets[selected] + self._env.common_step_counter) % len(self._frames)
        pose = self._default_pose[selected].clone()
        pose[:, self._arm_ids] = self._frames[frames]
        self._asset.write_joint_state_to_sim_index(position=pose, velocity=torch.zeros_like(pose), env_ids=env_ids)
        self._asset.actuators.target_command.set_position_index(value=pose, env_ids=env_ids)


class CoffeeKitchenCameraEnv(ManagerBasedRLEnv):
    """The demo's environment without its powder: no contacts but the robot's, and the spoon mounted on the palm."""

    cfg: CoffeeKitchenCameraEnvCfg

    def __init__(self, cfg: CoffeeKitchenCameraEnvCfg, render_mode: str | None = None, **kwargs):
        self._attachment = None
        self._next_shape = 0
        with newton_builder_world_hook(self._disable_contacts):
            super().__init__(cfg, render_mode, **kwargs)

    def _disable_contacts(self, builder, env_id: int, _position, _quaternion) -> None:
        """Turn off shape contacts for everything but the robot, as the demo's ``_configure_colliders`` does."""
        prefix = f"/World/envs/env_{env_id}/"
        for index in range(self._next_shape, builder.shape_count):
            label = str(builder.shape_label[index])
            if label.startswith(prefix) and "/Robot/" not in label:
                builder.shape_flags[index] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
        self._next_shape = builder.shape_count

    def load_managers(self) -> None:
        # As in the demo, the mount is registered before the managers.
        self._attachment = MountedSpoonAttachment(
            self.scene["robot"], self.scene["spoon"], DEFAULT_SPOON_OFFSET_POS, DEFAULT_SPOON_OFFSET_ROT
        )
        super().load_managers()

    def close(self) -> None:
        if self._attachment is not None:
            self._attachment.close()
            self._attachment = None
        super().close()
