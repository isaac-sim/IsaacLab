# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 velocity task with left-right mirrored world-model transitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab_tasks.core.velocity.mdp.symmetry.g1 import action_mirror_map, body_mirror_permutation, joint_mirror_map
from isaaclab_tasks.core.velocity.supermimic import VelocityTask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class G1VelocityTask(VelocityTask):
    """Velocity task that reflects G1 transitions and the terrain across the line ``y = terrain.mirror_y``.

    Beyond that line the heightfield returns the mirrored terrain, so reflected positions keep their terrain heights.
    """

    def __init__(self, env: ManagerBasedRLEnv, device: str = "cpu") -> None:
        super().__init__(env, device)
        owner = env.unwrapped
        robot = owner.scene["robot"]
        sensor = owner.scene.sensors[owner.cfg.rewards.feet_air_time.params["sensor_cfg"].name]
        joint_perm, joint_sign = joint_mirror_map(owner)
        action_perm, action_sign = action_mirror_map(owner)
        self.joint_perm, self.action_perm = joint_perm.to(device), action_perm.to(device)
        # The shared G1 symmetry gives every elbow joint the sign of a pitch joint, but the elbow roll joints rotate
        # about the forward axis, so their reflection negates them like the other roll joints
        elbow_roll = torch.tensor(["elbow_roll" in name for name in robot.joint_names], device=device)
        self.joint_sign = torch.where(elbow_roll, -1.0, joint_sign.to(device))
        self.action_sign = torch.where(elbow_roll[self.action_joints], -1.0, action_sign.to(device))
        self.body_perm = torch.tensor(body_mirror_permutation(list(robot.body_names)), device=device)
        self.contact_perm = torch.tensor(body_mirror_permutation(list(sensor.body_names)), device=device)
        feet = [sensor.body_names[index] for index in self.sensor_foot_ids]
        self.foot_perm = torch.tensor(body_mirror_permutation(feet), device=device)
        # y negation of vectors; xyzw quaternions and angular velocities keep only y
        self.vector_sign = torch.tensor([1.0, -1.0, 1.0], device=device)
        self.quat_sign = torch.tensor([-1.0, 1.0, -1.0, 1.0], device=device)
        self.axial_sign = torch.tensor([-1.0, 1.0, -1.0], device=device)
        self.position_offset = torch.tensor([0.0, 2 * self.terrain.mirror_y, 0.0], device=device)

    def mirror_state(self, state: torch.Tensor) -> torch.Tensor:
        """Reflect the base motion and position, and swap left and right joints and foot clocks."""
        lin, ang, quat, pos, q, dq, *clocks = self.representation.unflatten(state)
        return torch.cat(
            (
                lin * self.vector_sign,
                ang * self.axial_sign,
                quat * self.quat_sign,
                pos * self.vector_sign + self.position_offset,
                q[..., self.joint_perm] * self.joint_sign,
                dq[..., self.joint_perm] * self.joint_sign,
                *(clock[..., self.foot_perm] for clock in clocks),
            ),
            dim=-1,
        )

    def mirror_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Swap left and right joint actions."""
        return actions[..., self.action_perm] * self.action_sign

    def mirror_context(self, context: torch.Tensor) -> torch.Tensor:
        """Swap left and right body masses."""
        return context[..., self.body_perm]

    def mirror_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Swap left and right foot contacts, body contacts and first contacts."""
        feet, contacts, first = labels.split((self.num_feet, self.num_bodies, self.num_feet), dim=-1)
        return torch.cat((feet[..., self.foot_perm], contacts[..., self.contact_perm], first[..., self.foot_perm]), -1)
