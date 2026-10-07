# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action source shared by the ANYmal-D recording and deployment examples."""

from collections.abc import Callable
from pathlib import Path

import torch
from isaaclab_physicsnemo import SmoothRandomActions

from isaaclab.envs import ManagerBasedRLEnv


def make_action_source(
    env: ManagerBasedRLEnv, *, policy_path: Path | None, scale: float, seed: int
) -> Callable[[torch.Tensor | None], torch.Tensor]:
    """Return exploratory actions, optionally added to the actions of an exported locomotion policy.

    Args:
        env: Environment after its first reset.
        policy_path: TorchScript policy exported by the RSL-RL ``play.py`` script, or ``None``.
        scale: Standard deviation of the smooth exploration noise, in action units. The ANYmal-D joint-position
            action scales actions by 0.5 rad.
        seed: Seed of the exploration noise.

    Returns:
        A callable that takes the previous step's dones and returns the next actions.
    """
    noise = SmoothRandomActions(
        env.num_envs, env.action_manager.total_action_dim, scale=scale, device=env.device, seed=seed
    )
    if policy_path is None:
        return noise
    policy = torch.jit.load(str(policy_path), map_location=env.device).eval()

    def actions(dones: torch.Tensor | None) -> torch.Tensor:
        return policy(env.obs_buf["policy"]) + noise(dones)

    return actions
