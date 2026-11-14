# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A graph-safe, one-way mounted spoon constraint at the physics tick rate."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING

import torch
import warp as wp
from newton import JointType

from isaaclab.sim import SimulationContext
from isaaclab.utils import math as math_utils

from .robot import PALM_BODY_NAME

if TYPE_CHECKING:
    from isaaclab_newton.physics import NewtonManager

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
    Use :meth:`sync_to_sim` after construction, partial resets, or direct robot
    pose writes to initialize the asset state before rendering.

    Args:
        robot: Robot carrying the palm frame.
        spoon: Free-root tool attached to the palm.
        offset_pos: Spoon link position relative to the palm [m].
        offset_rot: Spoon link orientation relative to the palm, XYZW.
        backend_manager: Initialized Newton manager. When omitted, use the
            current simulation context for compatibility with existing callers.
    """

    def __init__(
        self,
        robot: Articulation,
        spoon: RigidObject,
        offset_pos: tuple[float, float, float],
        offset_rot: tuple[float, float, float, float],
        *,
        backend_manager: NewtonManager | None = None,
    ) -> None:
        self._robot = robot
        self._spoon = spoon
        self._palm_id = robot.find_bodies(PALM_BODY_NAME)[0][0]
        self._tool_position = torch.tensor(offset_pos, device=robot.device).expand(robot.num_instances, -1)
        self._tool_quaternion = torch.tensor(offset_rot, device=robot.device).expand(robot.num_instances, -1)
        self._backend_manager = (
            backend_manager if backend_manager is not None else SimulationContext.instance().backend_manager
        )
        self._model = self._backend_manager.backend.model
        palm_pattern = re.compile(robot.cfg.prim_path + "/" + re.escape(PALM_BODY_NAME))
        spoon_pattern = re.compile(spoon.cfg.prim_path)
        bodies = defaultdict(lambda: ([], []))
        for index, (label, world) in enumerate(
            zip(self._model.body_label, self._model.body_world.numpy(), strict=True)
        ):
            if palm_pattern.fullmatch(label):
                bodies[world][0].append(index)
            if spoon_pattern.fullmatch(label):
                bodies[world][1].append(index)
        palm_ids, spoon_ids = [], []
        for env_index in range(robot.num_instances):
            palms, spoons = bodies[env_index]
            if len(palms) != 1 or len(spoons) != 1:
                raise ValueError(f"Expected one native palm and spoon in env_{env_index}; got {palms}, {spoons}.")
            palm_ids.append(palms[0])
            spoon_ids.append(spoons[0])
        self._palm_ids = wp.array(palm_ids, dtype=int, device=self._model.device)
        self._spoon_ids = wp.array(spoon_ids, dtype=int, device=self._model.device)
        free_joints = defaultdict(list)
        for index, (child, joint_type) in enumerate(
            zip(self._model.joint_child.numpy(), self._model.joint_type.numpy(), strict=True)
        ):
            if joint_type == JointType.FREE:
                free_joints[child].append(index)
        q_starts = self._model.joint_q_start.numpy()
        qd_starts = self._model.joint_qd_start.numpy()
        spoon_joints = []
        for body in spoon_ids:
            joints = free_joints[body]
            if len(joints) != 1:
                raise ValueError(f"Expected one free root joint for mounted spoon body {body}; got {joints}.")
            spoon_joints.append(joints[0])
        self._q_starts = wp.array(q_starts[spoon_joints], dtype=int, device=self._model.device)
        self._qd_starts = wp.array(qd_starts[spoon_joints], dtype=int, device=self._model.device)
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

    def sync_to_sim(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Publish the mounted link pose and twist through the asset APIs.

        This is the construction/reset/direct-render path. Physics ticks use
        :meth:`sync`, which writes native COM velocities in both solver buffers.

        Args:
            env_ids: Selected environment indices on the asset device, or
                ``None`` for all environments. Unselected states stay unchanged.
        """
        selected = slice(None) if env_ids is None else env_ids
        data = self._robot.data
        palm = data.body_link_pose_w.torch[selected, self._palm_id]
        pos, quat = math_utils.combine_frame_transforms(
            palm[:, :3], palm[:, 3:], self._tool_position[selected], self._tool_quaternion[selected]
        )
        velocity = data.body_link_vel_w.torch[selected, self._palm_id].clone()
        velocity[:, :3] += torch.cross(velocity[:, 3:], pos - palm[:, :3], dim=-1)
        self._spoon.write_root_link_pose_to_sim_index(root_pose=torch.cat((pos, quat), dim=-1), env_ids=env_ids)
        self._spoon.write_root_link_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)

    def close(self) -> None:
        """Disable captured writes until the manager clears its callback list."""
        self._enabled.zero_()


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
