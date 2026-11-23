# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based control environment with scoped Newton connector construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_newton.cloner import newton_builder_world_hook

from isaaclab.envs import ManagerBasedEnv

from .assets import configured_rizon_sharpa_usd

if TYPE_CHECKING:
    import newton
    import numpy as np

    from .env_cfg import GB300SharpaInsertionEnvCfg


class GB300SharpaInsertionEnv(ManagerBasedEnv):
    """Insert a physically attached Ethernet connector into a GB300 switch."""

    def __init__(self, cfg: GB300SharpaInsertionEnvCfg):
        from .sharpa_contacts import spawn_sharpa_contact_meshes

        cfg.scene.robot.spawn.usd_path = configured_rizon_sharpa_usd()
        cfg.scene.robot.spawn.func = spawn_sharpa_contact_meshes
        super().__init__(cfg)

    def _init_sim(self) -> None:
        from .cable import add_connectors

        def build_connectors(
            builder: newton.ModelBuilder, world_idx: int, env_position: np.ndarray, env_rotation: np.ndarray
        ) -> None:
            add_connectors(
                builder,
                world_idx,
                env_position,
                env_rotation,
                grasp_body_patterns=self.cfg.grasp_body_patterns,
                rigid_connector_contact_stiffness=self.cfg.rigid_connector_contact_stiffness,
            )
            from .hydroelastic import configure_hydroelastic_shapes

            configure_hydroelastic_shapes(
                builder,
                self.cfg.grasp_body_patterns,
                pad_pressure_gradient=self.cfg.hydroelastic_pad_pressure_gradient,
            )

        with newton_builder_world_hook(build_connectors):
            super()._init_sim()

    def load_managers(self) -> None:
        self.tcp_body_id = self.scene["robot"].find_bodies(self.cfg.tcp_body_name)[0][0]
        self.head_body_id = self.scene["connector"].find_bodies("head")[0][0]
        self.latch_body_id = self.scene["connector"].find_bodies("latch")[0][0]
        super().load_managers()
