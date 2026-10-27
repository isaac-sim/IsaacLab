# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One-way, measured-plug boundary conditions for the native VBD cable."""

from __future__ import annotations

import warp as wp
from isaaclab_newton.physics import VBDSolver
from newton import BodyFlags, Contacts, Control, Model, State, StateFlags
from newton.solvers import SolverVBD


class EndpointVBD(SolverVBD):
    """VBD whose proxy lifecycle prescribes the loose cable endpoint."""

    def __init__(self, model: Model, **kwargs):
        # Override the destination view only; the source rigid bodies stay dynamic.
        body_flags = model.body_flags.numpy()
        proxies = [index for index, flags in enumerate(body_flags) if int(flags) & int(BodyFlags.PROXY)]
        for body in proxies:
            body_flags[body] = (int(body_flags[body]) & ~int(BodyFlags.DYNAMIC)) | int(BodyFlags.KINEMATIC)
        model.body_flags = wp.array(body_flags, dtype=wp.int32, device=model.device)
        super().__init__(model, **kwargs)
        labels = {str(label): index for index, label in enumerate(model.body_label)}
        pairs = []
        offsets = []
        poses = model.body_q.numpy()
        for label, endpoint in labels.items():
            if not label.endswith("/Cable/geometry/plug_endpoint"):
                continue
            plug_label = label.removesuffix("/geometry/plug_endpoint") + "/head"
            if plug_label not in labels:
                raise ValueError(f"The endpoint requires a visible plug proxy at {plug_label}.")
            plug = labels[plug_label]
            pairs.append((plug, endpoint))
            plug_pose = wp.transform(poses[plug, :3], poses[plug, 3:])
            endpoint_pose = wp.transform(poses[endpoint, :3], poses[endpoint, 3:])
            offsets.append(wp.transform_multiply(wp.transform_inverse(plug_pose), endpoint_pose))
        if not pairs:
            raise ValueError("The one-way cable requires a kinematic plug_endpoint body.")
        self._endpoint_pairs = wp.array(pairs, dtype=wp.vec2i, device=model.device)
        self._endpoint_offsets = wp.array(offsets, dtype=wp.transform, device=model.device)
        self._prescribed_bodies = wp.array([*proxies, *(pair[1] for pair in pairs)], dtype=int, device=model.device)
        self._prescribed_velocities = wp.zeros(
            len(self._prescribed_bodies), dtype=wp.spatial_vector, device=model.device
        )

    def coupling_notify_input_state_update(
        self, state: State, flags: StateFlags | int, *, iteration_restart: bool = False, dt: float = 0.0
    ) -> None:
        """Prescribe the measured cable pose [m, XYZW] and COM twist [m/s, rad/s]."""
        if int(flags) & int(StateFlags.BODY_Q):
            wp.launch(
                _follow_endpoint,
                dim=len(self._endpoint_pairs),
                inputs=[
                    self._endpoint_pairs,
                    self._endpoint_offsets,
                    self.model.body_com,
                    state.body_q,
                    state.body_qd,
                ],
                device=self.device,
            )
            wp.launch(
                _save_prescribed_velocity,
                dim=len(self._prescribed_bodies),
                inputs=[self._prescribed_bodies, state.body_qd, self._prescribed_velocities],
                device=self.device,
            )
        super().coupling_notify_input_state_update(state, flags, iteration_restart=iteration_restart, dt=dt)

    def coupling_harvest_proxy_wrenches(
        self,
        body_local_to_proxy_global: wp.array[int],
        out_body_f: wp.array[wp.spatial_vector],
        *,
        body_qd_before: wp.array[wp.spatial_vector],
        state: State,
        state_out: State,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Discard feedback: prescribed cable boundaries exert no source force."""
        out_body_f.zero_()

    def step(self, state_in: State, state_out: State, control: Control, contacts: Contacts | None, dt: float) -> None:
        """Advance the interior cable, retaining the prescribed endpoint velocity [m/s, rad/s]."""
        super().step(state_in, state_out, control, contacts, dt)
        # VBD's finite-difference finalization also overwrites kinematic velocities.
        wp.launch(
            _restore_prescribed_velocity,
            dim=len(self._prescribed_bodies),
            inputs=[self._prescribed_bodies, self._prescribed_velocities, state_in.body_qd, state_out.body_qd],
            device=self.device,
        )


class EndpointVBDSolver(VBDSolver):
    """VBD solver that builds :class:`EndpointVBD`."""

    solver_class = EndpointVBD


@wp.kernel
def _follow_endpoint(
    pairs: wp.array(dtype=wp.vec2i),
    offsets: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    index = wp.tid()
    plug, endpoint = pairs[index][0], pairs[index][1]
    plug_pose = body_q[plug]
    endpoint_pose = wp.transform_multiply(plug_pose, offsets[index])
    angular = wp.spatial_bottom(body_qd[plug])
    com_offset = wp.transform_point(endpoint_pose, body_com[endpoint]) - wp.transform_point(plug_pose, body_com[plug])
    velocity = wp.spatial_vector(wp.spatial_top(body_qd[plug]) + wp.cross(angular, com_offset), angular)
    body_q[endpoint] = endpoint_pose
    body_qd[endpoint] = velocity


@wp.kernel
def _save_prescribed_velocity(
    bodies: wp.array(dtype=int),
    body_qd: wp.array(dtype=wp.spatial_vector),
    velocities: wp.array(dtype=wp.spatial_vector),
):
    index = wp.tid()
    velocities[index] = body_qd[bodies[index]]


@wp.kernel
def _restore_prescribed_velocity(
    bodies: wp.array(dtype=int),
    velocities: wp.array(dtype=wp.spatial_vector),
    body_qd_in: wp.array(dtype=wp.spatial_vector),
    body_qd_out: wp.array(dtype=wp.spatial_vector),
):
    index = wp.tid()
    body = bodies[index]
    body_qd_in[body] = velocities[index]
    body_qd_out[body] = velocities[index]
