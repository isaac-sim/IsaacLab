# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A graph-safe, one-way mounted spoon constraint at the physics tick rate."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp
from newton import JointType, Model
from newton.selection import ArticulationView

from isaaclab.sim import SimulationContext

from .robot import PALM_BODY_NAME

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject


class MountedSpoonAttachment:
    """Follow measured native palm state inside every Newton decimation iteration.

    Newton body transforms refer to link origins; spatial velocities contain
    world-frame COM linear velocity followed by world-frame angular velocity.
    This attachment transfers the twist at the spoon COM, including both body
    COM offsets. It applies no reaction force to the robot.

    Construct after the scene and solver are initialized, before the first
    physics graph capture. The task uses one solver substep per physics tick.
    The manager clears callback registrations when the environment closes.
    """

    def __init__(
        self,
        robot: Articulation,
        spoon: RigidObject,
        offset_pos: tuple[float, float, float],
        offset_rot: tuple[float, float, float, float],
    ):
        self._backend_manager = SimulationContext.instance().backend_manager
        self._model = self._backend_manager.backend.model
        if spoon.root_view.root_joint_type != JointType.FREE:
            raise ValueError("The mounted spoon must have a free root joint.")
        palm_id = robot.backend_body_names.index(PALM_BODY_NAME)
        frequency = Model.AttributeFrequency
        self._palm_ids = _model_indices(robot.root_view, frequency.BODY, palm_id)
        self._spoon_ids = _model_indices(spoon.root_view, frequency.BODY)
        self._q_starts = _model_indices(spoon.root_view, frequency.JOINT_COORD)
        self._qd_starts = _model_indices(spoon.root_view, frequency.JOINT_DOF)
        self._offset = wp.transform(wp.vec3(offset_pos), wp.quat(offset_rot))
        self._enabled = wp.ones(1, dtype=int, device=self._model.device)
        self._backend_manager.register_post_actuator_callback(self.sync)

    def sync(self) -> None:
        """Write the mounted pose and COM twist to both native state buffers."""
        state_0, state_1 = self._backend_manager.backend.state_0, self._backend_manager.backend.state_1
        wp.launch(
            _sync_mounted_spoon,
            dim=self._palm_ids.size,
            inputs=[
                self._palm_ids,
                self._spoon_ids,
                self._q_starts,
                self._qd_starts,
                self._offset,
                self._model.body_com,
                self._enabled,
                state_0.body_q,
                state_0.body_qd,
                state_1.body_q,
                state_1.body_qd,
                state_0.joint_q,
                state_0.joint_qd,
                state_1.joint_q,
                state_1.joint_qd,
            ],
            device=self._model.device,
        )

    def close(self) -> None:
        """Disable captured writes until the manager clears its callback list."""
        self._enabled.zero_()


def _model_indices(view: ArticulationView, frequency: Model.AttributeFrequency, index: int = 0) -> wp.array:
    """Return one absolute model index per instance in the asset's own ordering."""
    layout = view.frequency_layouts[frequency]
    rows = layout.get_absolute_indices(view.world_count, view.count_per_world)
    return wp.array(rows[..., index].reshape(-1), dtype=int, device=view.device)


@wp.kernel
def _sync_mounted_spoon(
    palm_ids: wp.array[int],
    spoon_ids: wp.array[int],
    q_starts: wp.array[int],
    qd_starts: wp.array[int],
    offset: wp.transform,
    body_com: wp.array[wp.vec3],
    enabled: wp.array[int],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    next_body_q: wp.array[wp.transform],
    next_body_qd: wp.array[wp.spatial_vector],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    next_joint_q: wp.array[float],
    next_joint_qd: wp.array[float],
):
    i = wp.tid()
    if enabled[0] == 0:
        return
    palm = palm_ids[i]
    spoon = spoon_ids[i]
    palm_pose = body_q[palm]
    spoon_pose = palm_pose * offset
    palm_com = wp.transform_point(palm_pose, body_com[palm])
    spoon_com = wp.transform_point(spoon_pose, body_com[spoon])
    angular = wp.spatial_bottom(body_qd[palm])
    linear = wp.spatial_top(body_qd[palm]) + wp.cross(angular, spoon_com - palm_com)
    velocity = wp.spatial_vector(linear, angular)
    body_q[spoon] = spoon_pose
    body_qd[spoon] = velocity
    next_body_q[spoon] = spoon_pose
    next_body_qd[spoon] = velocity
    # Root asset views read free-joint coordinates; FK also consumes them.
    for component in range(7):
        joint_q[q_starts[i] + component] = spoon_pose[component]
        next_joint_q[q_starts[i] + component] = spoon_pose[component]
    for component in range(6):
        joint_qd[qd_starts[i] + component] = velocity[component]
        next_joint_qd[qd_starts[i] + component] = velocity[component]
