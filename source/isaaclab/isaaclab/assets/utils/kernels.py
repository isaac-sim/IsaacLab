# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warp as wp


@wp.func
def skew_symetric_matrix(v: wp.vec3f) -> wp.mat33f:
    return wp.mat33f(0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0)


@wp.kernel
def add_forces_and_torques_at_position(
    env_ids: wp.array(dtype=wp.int32),
    body_ids: wp.array(dtype=wp.int32),
    forces: wp.array2d(dtype=wp.vec3f),
    torques: wp.array2d(dtype=wp.vec3f),
    positions: wp.array2d(dtype=wp.vec3f),
    composed_forces_b: wp.array2d(dtype=wp.vec3f),
    composed_torques_b: wp.array2d(dtype=wp.vec3f),
):
    tid_env, tid_body = wp.tid()
    if forces.shape[0] > 0:
        composed_forces_b[env_ids[tid_env], body_ids[tid_body]] += forces[tid_env, tid_body]
    if (positions.shape[0] > 0) and (forces.shape[0] > 0):
        composed_torques_b[env_ids[tid_env], body_ids[tid_body]] += (
            skew_symetric_matrix(positions[tid_env, tid_body]) @ forces[tid_env, tid_body]
        )
    if torques.shape[0] > 0:
        composed_torques_b[env_ids[tid_env], body_ids[tid_body]] += torques[tid_env, tid_body]


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
