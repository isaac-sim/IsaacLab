# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The existing analytic render benchmark with the coffee scene's mounted spoon."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import newton
import numpy as np
import torch
from isaaclab_newton.cloner import newton_builder_world_hook

from isaaclab_tasks.benchmark.render_benchmark.render_benchmark_env import RenderBenchmarkEnv
from isaaclab_tasks.contrib.rizon_sharpa_coffee.attachment import MountedSpoonAttachment
from isaaclab_tasks.contrib.rizon_sharpa_coffee.robot import (
    DEFAULT_SPOON_OFFSET_POS,
    DEFAULT_SPOON_OFFSET_ROT,
    HAND_JOINT_NAMES,
)

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

    from .coffee_kitchen_camera_env_cfg import CoffeeKitchenCameraEnvCfg


class CoffeeKitchenCameraEnv(RenderBenchmarkEnv):
    """Animate the arm analytically while retaining the demo's hand pose and mounted utensil."""

    cfg: CoffeeKitchenCameraEnvCfg

    def __init__(self, cfg: CoffeeKitchenCameraEnvCfg, render_mode: str | None = None, **kwargs):
        self._attachment: MountedSpoonAttachment | None = None
        self._next_shape = 0
        with newton_builder_world_hook(self._disable_prop_contacts):
            super().__init__(cfg, render_mode, **kwargs)
        robot = self.scene["robot"]
        self._hand_ids = robot.find_joints(HAND_JOINT_NAMES, preserve_order=True)[0]
        self._attachment = MountedSpoonAttachment(
            robot,
            self.scene["spoon"],
            DEFAULT_SPOON_OFFSET_POS,
            DEFAULT_SPOON_OFFSET_ROT,
            backend_manager=self.sim.backend_manager,
        )
        self._attachment.sync_to_sim()
        self.sim.forward()

    def _disable_prop_contacts(
        self, builder: newton.ModelBuilder, env_id: int, position: np.ndarray, quaternion: np.ndarray
    ) -> None:
        for index in range(self._next_shape, builder.shape_count):
            if builder.shape_world[index] == env_id and "/Robot/" not in str(builder.shape_label[index]):
                builder.shape_flags[index] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
        self._next_shape = builder.shape_count

    def _animation_targets(self) -> Iterator[tuple[Articulation, torch.Tensor]]:
        for articulation, target in super()._animation_targets():
            target[:, self._hand_ids] = articulation.data.default_joint_pos.torch[:, self._hand_ids]
            yield articulation, target

    def _pose_joints_directly(self) -> None:
        super()._pose_joints_directly()
        # Evaluate the new palm pose before the mount; the inherited final forward publishes the spoon.
        self.sim.forward()
        self._attachment.sync()

    def _reset_idx(self, env_ids: torch.Tensor | slice | None) -> None:
        super()._reset_idx(env_ids)
        if self._attachment is not None:
            self.sim.forward()
            self._attachment.sync_to_sim(env_ids)

    def close(self) -> None:
        if self._attachment is not None:
            self._attachment.close()
            self._attachment = None
        super().close()
