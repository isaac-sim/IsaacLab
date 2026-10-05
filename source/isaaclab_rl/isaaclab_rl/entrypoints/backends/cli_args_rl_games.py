# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Agent configuration overrides shared by the RL-Games entrypoints."""

from __future__ import annotations

import argparse
import os

from ..common import resolve_seed


def update_rl_games_cfg(agent_cfg: dict, args_cli: argparse.Namespace) -> dict:
    """Override an RL-Games agent configuration with the command-line arguments.

    Call it inside :func:`~isaaclab.app.launch_simulation`, which resolves ``args_cli.device``: the agent runs on
    the simulation's device.

    Args:
        agent_cfg: The configuration for RL-Games agent.
        args_cli: The command line arguments.

    Returns:
        The updated RL-Games agent configuration.
    """
    params = agent_cfg["params"]
    config = params["config"]
    config["device"] = config["device_name"] = args_cli.device
    if getattr(args_cli, "seed", None) is not None:
        args_cli.seed = resolve_seed(args_cli.seed)
        params["seed"] = args_cli.seed
    if getattr(args_cli, "max_iterations", None) is not None:
        config["max_epochs"] = args_cli.max_iterations
    if getattr(args_cli, "distributed", False):
        # offsetting the seed by the rank decorrelates exploration across ranks
        params["seed"] += int(os.getenv("RANK", "0"))
        config["multi_gpu"] = True
    return agent_cfg
