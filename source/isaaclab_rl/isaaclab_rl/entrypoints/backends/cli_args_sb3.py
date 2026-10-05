# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Agent configuration overrides shared by the Stable-Baselines3 entrypoints."""

from __future__ import annotations

import argparse

from ..common import resolve_seed


def update_sb3_cfg(agent_cfg: dict, args_cli: argparse.Namespace) -> dict:
    """Override a Stable-Baselines3 agent configuration with the command-line arguments.

    Call it inside :func:`~isaaclab.app.launch_simulation`, which resolves ``args_cli.device``: the agent runs on
    the simulation's device. Loading a checkpoint still needs that device passed to ``PPO.load``.

    Args:
        agent_cfg: The configuration for Stable-Baselines3 agent.
        args_cli: The command line arguments.

    Returns:
        The updated Stable-Baselines3 agent configuration.
    """
    agent_cfg["device"] = args_cli.device
    if getattr(args_cli, "seed", None) is not None:
        args_cli.seed = resolve_seed(args_cli.seed)
        agent_cfg["seed"] = args_cli.seed
    return agent_cfg
