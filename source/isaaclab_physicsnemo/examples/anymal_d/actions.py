# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action source shared by the ANYmal-D recording and deployment examples."""

from collections.abc import Callable
from pathlib import Path

import torch
from isaaclab_physicsnemo import UniformRandomActions

from isaaclab.envs import ManagerBasedRLEnv

# The ANYmal-D joint-position action scales by 0.5 rad, so 1.5 gives targets within 0.75 rad of the default pose.
ACTION_BOUND = 1.5


def make_action_source(
    env: ManagerBasedRLEnv, *, policy_path: Path | None, high: float, seed: int, hold_steps: int = 1
) -> Callable[[torch.Tensor | None], torch.Tensor]:
    """Return uniform joint-target actions, optionally added to the actions of an exported locomotion policy.

    Args:
        env: Environment after its first reset.
        policy_path: TorchScript policy exported by the RSL-RL ``play.py`` script, or ``None``.
        high: Bound of the uniform actions, in action units.
        seed: Seed of the actions.
        hold_steps: Number of environment steps that keep one target.

    Returns:
        A callable that takes the previous step's dones and returns the next actions.
    """
    noise = UniformRandomActions(
        env.num_envs,
        env.action_manager.total_action_dim,
        high=high,
        hold_steps=hold_steps,
        device=env.device,
        seed=seed,
    )
    if policy_path is None:
        return noise
    policy = torch.jit.load(str(policy_path), map_location=env.device).eval()

    def actions(dones: torch.Tensor | None) -> torch.Tensor:
        return policy(env.obs_buf["policy"]) + noise(dones)

    return actions
