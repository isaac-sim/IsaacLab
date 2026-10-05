# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Agent configuration overrides shared by the Stable-Baselines3 entrypoints."""

from __future__ import annotations

from ..common import resolve_seed


def update_sb3_cfg(agent_cfg: dict, device: str, *, seed: int | None = None) -> dict:
    """Override a Stable-Baselines3 agent configuration with the command-line arguments.

    Args:
        agent_cfg: The configuration for Stable-Baselines3 agent.
        device: Device the agent runs on, the simulation's device once :func:`~isaaclab.app.launch_simulation`
            has resolved it. Pass ``agent_cfg["device"]`` to ``PPO.load`` as well.
        seed: Seed override; ``-1`` draws a random seed.

    Returns:
        The updated Stable-Baselines3 agent configuration.
    """
    agent_cfg["device"] = device
    if seed is not None:
        agent_cfg["seed"] = resolve_seed(seed)
    return agent_cfg
