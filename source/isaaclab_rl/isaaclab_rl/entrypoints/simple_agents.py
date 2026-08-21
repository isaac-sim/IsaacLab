# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checkpoint-free playback workflows for Isaac Lab environments.

The zero and random agents are variations of playback that need no trained checkpoint:
the policy either emits constant zero actions or samples uniform random actions.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from typing import Literal

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import (
    resolve_task_config,
    setup_preset_cli,
)

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

PolicyName = Literal["zero", "random"]
"""Action policies supported by the checkpoint-free agents."""

_DESCRIPTIONS: dict[str, str] = {
    "zero": "Zero agent for Isaac Lab environments.",
    "random": "Random agent for Isaac Lab environments.",
}

_SEED = 42


def run(argv: list[str] | None = None, *, policy: PolicyName) -> None:
    """Run an Isaac Lab environment with a checkpoint-free policy.

    Args:
        argv: Command-line arguments excluding the executable name. Reads ``sys.argv`` when omitted.
        policy: Action policy to apply, either constant zero actions or uniform random actions.

    Raises:
        ValueError: If the requested policy is not supported.
    """
    if policy not in _DESCRIPTIONS:
        raise ValueError(f"Unsupported policy {policy!r}. Expected one of: {sorted(_DESCRIPTIONS)}.")

    args_cli = _parse_args(argv, policy)

    torch.manual_seed(_SEED)

    # parse configuration via Hydra (supports preset selection, e.g. env.sim.physics=newton_mjwarp)
    env_cfg, _ = resolve_task_config(args_cli.task, "")

    with launch_simulation(env_cfg, args_cli):
        # override with CLI arguments
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        if args_cli.disable_fabric:
            env_cfg.sim.use_fabric = False

        # create environment
        env = gym.make(args_cli.task, cfg=env_cfg)

        # print info (this is vectorized environment)
        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")
        # reset environment
        env.reset()
        # simulate environment
        # keep running while any visualizer is open, and until the step budget is exhausted
        sim = env.unwrapped.sim
        device = env.unwrapped.device
        zero_actions = torch.zeros(env.action_space.shape, device=device)
        step = 0
        while sim.is_headless_or_exist_active_visualizer():
            if args_cli.max_steps is not None and step >= args_cli.max_steps:
                break
            step += 1
            # run everything in inference mode
            with torch.inference_mode():
                if policy == "zero":
                    actions = zero_actions
                else:
                    # sample actions from -1 to 1
                    actions = 2 * torch.rand(env.action_space.shape, device=device) - 1
                # apply actions
                env.step(actions)
        # close the simulator
        env.close()


def _parse_args(argv: list[str] | None, policy: PolicyName) -> argparse.Namespace:
    """Parse the command line of a checkpoint-free agent and hand the remainder to Hydra."""
    parser = argparse.ArgumentParser(description=_DESCRIPTIONS[policy])
    parser.add_argument(
        "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
    )
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument(
        "--max_steps", type=int, default=None, help="Number of environment steps to run. Runs unbounded when omitted."
    )
    # append AppLauncher cli args
    add_launcher_args(parser)
    # simple agents should open Kit visualizer by default
    parser.set_defaults(visualizer=["kit"])
    args_cli, hydra_args = setup_preset_cli(parser, argv)
    sys.argv = [sys.argv[0]] + hydra_args
    return args_cli
