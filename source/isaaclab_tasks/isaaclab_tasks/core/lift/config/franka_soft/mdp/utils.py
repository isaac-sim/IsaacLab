# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sanitized state accessors shared by the deformable lift MDP terms.

The coupled rigid/soft solve can diverge and turn a whole environment's state non-finite. Measured
behaviour is a single-step event in one environment out of thousands: the robot joint state, the
robot body poses, the end-effector frame and every deformable node all become ``NaN`` at once, with
no growth in the preceding steps. Every reward or observation reading that state then returns
``NaN``, and RL libraries check the returned rewards and observations, so one diverged environment
aborts the whole run.

Terminating on the divergence is not enough on its own. In
:meth:`~isaaclab.envs.ManagerBasedRLEnv.step` the reward manager runs after the termination manager
but *before* the environments are reset, so rewards for the terminating step are still computed from
the diverged state. Observations are computed after the reset and are normally clean, but the
pre-reset paths (an active recorder term, or ``compute_final_obs``) also read the diverged state.

Reward terms, and the deformable observation terms, therefore read state through the helpers below,
which replace non-finite entries with ``0.0``. This places a diverged body at the world origin,
yielding a finite but meaningless value for exactly one step. That is intentional and acceptable,
because :func:`~isaaclab_tasks.core.lift.config.franka_soft.mdp.deformable_state_invalid` flags the
same step from the raw state and the environment is reset immediately.

The robot's root pose is deliberately left raw: the Franka is fixed-base, so body 0 is welded and
its transform has no joint-state dependence, keeping it finite while every descendant body goes
non-finite. A floating-base variant of this task would have to sanitize it too.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, DeformableObject
    from isaaclab.sensors import FrameTransformer


def _finite(value: torch.Tensor) -> torch.Tensor:
    """Copy of ``value`` with every non-finite entry replaced by ``0.0``."""
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def _com_w(asset: DeformableObject) -> torch.Tensor:
    """Sanitized world-frame center of mass of a deformable object [m].

    Args:
        asset: The deformable object entity.

    Returns:
        Tensor of shape ``(num_envs, 3)`` with non-finite entries replaced by ``0.0``.
    """
    return _finite(asset.data.root_pos_w.torch)


def _nodal_pos_w(asset: DeformableObject) -> torch.Tensor:
    """Sanitized world-frame nodal positions of a deformable object [m].

    Args:
        asset: The deformable object entity.

    Returns:
        Tensor of shape ``(num_envs, num_nodes, 3)`` with non-finite entries replaced by ``0.0``.
    """
    return _finite(asset.data.nodal_pos_w.torch)


def _body_pos_w(asset: Articulation, body_ids: slice | list[int]) -> torch.Tensor:
    """Sanitized world-frame positions of the selected robot bodies [m].

    Args:
        asset: The articulation entity.
        body_ids: Indices of the bodies to read.

    Returns:
        Tensor of shape ``(num_envs, num_bodies, 3)`` with non-finite entries replaced by ``0.0``.
    """
    return _finite(asset.data.body_pos_w.torch[:, body_ids])


def _ee_pos_w(sensor: FrameTransformer) -> torch.Tensor:
    """Sanitized world-frame position of the first target frame of a frame transformer [m].

    Args:
        sensor: The frame transformer sensor.

    Returns:
        Tensor of shape ``(num_envs, 3)`` with non-finite entries replaced by ``0.0``.
    """
    return _finite(sensor.data.target_pos_w.torch[..., 0, :])
