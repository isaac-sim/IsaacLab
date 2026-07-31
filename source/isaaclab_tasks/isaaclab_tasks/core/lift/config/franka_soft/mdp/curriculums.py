# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum functions for the deformable lift tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class modify_gravity_linear(ManagerTermBase):
    """Curriculum that linearly ramps the vertical gravity toward its target value.

    The vertical gravity component is interpolated from :paramref:`start_gravity_z` to
    :paramref:`end_gravity_z` [m/s^2] as the global step counter advances from
    :paramref:`start_step` to :paramref:`end_step`, then held constant. This lets the policy
    first learn under near-weightless dynamics before full gravity is applied.

    The active physics backend is detected automatically (PhysX or Newton).
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        manager_name = env.sim.physics_manager.__name__.lower()
        if "newton" in manager_name:
            self._backend = "newton"
            import isaaclab_newton.physics.newton_manager as newton_manager_module  # noqa: PLC0415
            from newton import ModelFlags  # noqa: PLC0415

            self._newton_manager = newton_manager_module.NewtonManager
            self._notify_model_properties = ModelFlags.MODEL_PROPERTIES
        else:
            self._backend = "physx"
            import carb  # noqa: PLC0415

            self._carb = carb
            self._physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        start_gravity_z: float,
        end_gravity_z: float,
        start_step: int,
        end_step: int,
    ) -> float:
        # linearly interpolate the vertical gravity based on training progress
        alpha = (env.common_step_counter - start_step) / max(end_step - start_step, 1)
        alpha = min(max(alpha, 0.0), 1.0)
        gravity_z = start_gravity_z + alpha * (end_gravity_z - start_gravity_z)

        if self._backend == "newton":
            import warp as wp  # noqa: PLC0415

            model = self._newton_manager.get_model()
            if model is None or model.gravity is None:
                raise RuntimeError("Newton model is not initialized. Cannot modify gravity.")
            # write to all worlds so gravity stays consistent regardless of per-env reset timing
            wp.to_torch(model.gravity)[:, 2] = gravity_z
            self._newton_manager.add_model_change(self._notify_model_properties)
        else:
            self._physics_sim_view.set_gravity(self._carb.Float3(0.0, 0.0, gravity_z))

        return gravity_z
