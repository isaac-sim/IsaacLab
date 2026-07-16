# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime data for a Newton cable."""

import warp as wp
from isaaclab_newton.physics import NewtonManager as SimulationManager

from isaaclab.utils.warp import ProxyArray


@wp.kernel
def _gather_segment_state(
    body_q: wp.array(dtype=wp.transformf),
    body_qd: wp.array(dtype=wp.spatial_vectorf),
    body_indices: wp.array2d(dtype=wp.int32),
    segment_pose: wp.array2d(dtype=wp.transformf),
    segment_velocity: wp.array2d(dtype=wp.spatial_vectorf),
):
    env_id, segment_id = wp.tid()
    body_id = body_indices[env_id, segment_id]
    segment_pose[env_id, segment_id] = body_q[body_id]
    segment_velocity[env_id, segment_id] = body_qd[body_id]


class CableData:
    """Current cable segment state."""

    def __init__(self, body_indices: wp.array2d(dtype=wp.int32), device: str):
        """Initialize the cable data.

        Args:
            body_indices: Newton body indices with shape ``(num_instances, num_segments)``.
            device: The device used for processing.
        """
        self._device = device
        self._bind(body_indices)

    @property
    def segment_pose_w(self) -> ProxyArray:
        """Segment actor-frame poses in the world frame.

        Warp shape is ``(num_instances, num_segments)`` with ``wp.transformf`` dtype. The Torch view has
        shape ``(num_instances, num_segments, 7)`` with position [m] and ``(x, y, z, w)`` quaternion.
        """
        return self._segment_pose_w

    @property
    def segment_velocity_w(self) -> ProxyArray:
        """Segment COM velocities in the world frame.

        Warp shape is ``(num_instances, num_segments)`` with ``wp.spatial_vectorf`` dtype. The Torch view has
        shape ``(num_instances, num_segments, 6)`` with linear [m/s] then angular [rad/s] velocity.
        """
        return self._segment_velocity_w

    def update(self, dt: float) -> None:
        """Gather state from Newton's current state buffer."""
        del dt
        state = SimulationManager.get_state_0()
        if state is None or state.body_q is None or state.body_qd is None:
            raise RuntimeError("Newton body state is unavailable for the cable.")
        wp.launch(
            _gather_segment_state,
            dim=self._body_indices.shape,
            inputs=[state.body_q, state.body_qd, self._body_indices],
            outputs=[self._segment_pose_w.warp, self._segment_velocity_w.warp],
            device=self._device,
        )

    def _bind(self, body_indices: wp.array2d(dtype=wp.int32)) -> None:
        """Bind a recreated model's cable body indices."""
        self._body_indices = body_indices
        shape = body_indices.shape
        if not hasattr(self, "_segment_pose_w") or self._segment_pose_w.shape != shape:
            self._segment_pose_w = ProxyArray(wp.empty(shape, dtype=wp.transformf, device=self._device))
            self._segment_velocity_w = ProxyArray(wp.empty(shape, dtype=wp.spatial_vectorf, device=self._device))
