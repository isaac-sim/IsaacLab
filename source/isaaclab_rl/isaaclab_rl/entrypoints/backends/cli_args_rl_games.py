# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Agent configuration overrides shared by the RL-Games entrypoints."""

from __future__ import annotations

import argparse

from ..common import resolve_seed


def update_rl_games_cfg(agent_cfg: dict, args_cli: argparse.Namespace) -> dict:
    """Override an RL-Games agent configuration with the command-line arguments.

    Args:
        agent_cfg: The configuration for RL-Games agent.
        args_cli: The command line arguments.

    Returns:
        The updated RL-Games agent configuration.
    """
    config = agent_cfg["params"]["config"]
    config["device"] = config["device_name"] = args_cli.device
    if args_cli.seed is not None:
        args_cli.seed = resolve_seed(args_cli.seed)
        agent_cfg["params"]["seed"] = args_cli.seed
    return agent_cfg
