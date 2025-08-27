# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warp as wp

from .kernels import add_forces_and_torques_at_position


class WrenchComposer:
    def __init__(self, num_envs: int, num_bodies: int, device: str):
        self.num_envs = num_envs
        self.num_bodies = num_bodies
        self.device = device

        self._composed_force_b = wp.zeros((num_envs, num_bodies), dtype=wp.vec3f, device=device)
        self._composed_torque_b = wp.zeros((num_envs, num_bodies), dtype=wp.vec3f, device=device)

    @property
    def composed_force(self):
        return self._composed_force_b

    @property
    def composed_torque(self):
        return self._composed_torque_b

    @property
    def composed_force_as_numpy(self):
        return self._composed_force_b.numpy()

    @property
    def composed_torque_as_numpy(self):
        return self._composed_torque_b.numpy()

    @property
    def composed_force_as_torch(self):
        return wp.to_torch(self._composed_force_b)

    @property
    def composed_torque_as_torch(self):
        return wp.to_torch(self._composed_torque_b)

    def add_forces_and_torques(
        self,
        env_ids: wp.array(dtype=wp.int32),
        body_ids: wp.array,
        forces: wp.array | None = None,
        torques: wp.array | None = None,
        positions: wp.array | None = None,
    ):
        if forces is None:
            forces = wp.empty((0, 0), dtype=wp.vec3f, device=self.device)
        if torques is None:
            torques = wp.empty((0, 0), dtype=wp.vec3f, device=self.device)
        if positions is None:
            positions = wp.empty((0, 0), dtype=wp.vec3f, device=self.device)

        wp.launch(
            add_forces_and_torques_at_position,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[env_ids, body_ids, forces, torques, positions, self._composed_force_b, self._composed_torque_b],
            device=self.device,
        )

    def reset(self):
        self._composed_force_b.zero_()
        self._composed_torque_b.zero_()
