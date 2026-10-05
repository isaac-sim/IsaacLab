# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Agent configuration overrides shared by the RL-Games entrypoints."""

from __future__ import annotations

from ..common import resolve_seed


def update_rl_games_cfg(agent_cfg: dict, device: str, *, seed: int | None = None) -> dict:
    """Override an RL-Games agent configuration with the command-line arguments.

    Args:
        agent_cfg: The configuration for RL-Games agent.
        device: Device the agent runs on, the simulation's device once :func:`~isaaclab.app.launch_simulation`
            has resolved it.
        seed: Seed override; ``-1`` draws a random seed.

    Returns:
        The updated RL-Games agent configuration.
    """
    config = agent_cfg["params"]["config"]
    config["device"] = config["device_name"] = device
    if seed is not None:
        agent_cfg["params"]["seed"] = resolve_seed(seed)
    return agent_cfg
