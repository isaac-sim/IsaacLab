# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 motion tracking task with left-right mirrored world-model transitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab_tasks.core.motion_tracking.supermimic import MotionTrackingTask
from isaaclab_tasks.core.velocity.mdp.symmetry.g1 import action_mirror_map, body_mirror_permutation, joint_mirror_map

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class G1MotionTrackingTask(MotionTrackingTask):
    """Motion tracking task that reflects G1 transitions across the sagittal plane."""

    def __init__(self, env: ManagerBasedRLEnv, device: str = "cpu") -> None:
        super().__init__(env, device)
        owner = env.unwrapped
        self.joint_perm, self.joint_sign = (tensor.to(device) for tensor in joint_mirror_map(owner))
        self.action_perm, self.action_sign = (tensor.to(device) for tensor in action_mirror_map(owner))
        self.body_perm = torch.tensor(body_mirror_permutation(list(owner.scene["robot"].body_names)), device=device)
        # y negation for positions and linear velocities; xyzw quaternions and angular velocities keep only y
        self.vector_sign = torch.tensor([1.0, -1.0, 1.0], device=device)
        self.quat_sign = torch.tensor([-1.0, 1.0, -1.0, 1.0], device=device)
        self.axial_sign = torch.tensor([-1.0, 1.0, -1.0], device=device)

    def mirror_state(self, state: torch.Tensor) -> torch.Tensor:
        """Reflect the root pose and velocity and swap left and right joints."""
        pos, quat, lin, ang, q, dq = self.representation.unflatten(state)
        return torch.cat(
            (
                pos * self.vector_sign,
                quat * self.quat_sign,
                lin * self.vector_sign,
                ang * self.axial_sign,
                q[..., self.joint_perm] * self.joint_sign,
                dq[..., self.joint_perm] * self.joint_sign,
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
        """Keep the self-collision labels, which the reflection leaves unchanged."""
        return labels
