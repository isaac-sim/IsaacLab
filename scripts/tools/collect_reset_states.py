# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect the start states of an MDP2 task into a file, to fill its reset bank in later runs.

Building the task fills the bank of its ``reset_from_bank`` event from the event's fill, typically a candidates task.
The script saves that bank. Fill later runs from it with ``fill=mdp.FillFromFile(path)``, in the same number of worlds
unless the bank is shared. For example::

    uv run python scripts/tools/collect_reset_states.py --task Isaac-Lift-KukaAllegro-MDP2 --num_envs 4096 \\
        --bank_size 64 --out kuka_lift_bank.pt physics=newton_mjwarp
"""

import argparse
import dataclasses
import sys

import gymnasium as gym
from isaaclab_experimental.mdp2 import mdp

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", required=True, help="Name of an MDP2 task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of worlds; default: the task's.")
parser.add_argument("--bank_size", type=int, default=None, help="States per bank, for a candidates fill.")
parser.add_argument("--event", default="reset", help="Name of the reset_from_bank event.")
parser.add_argument("--out", required=True, help="File to save the states to.")
add_launcher_args(parser)
parser.set_defaults(device=None)  # let the task select the device
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

env_cfg, _ = resolve_task_config(args_cli.task, "")
if args_cli.num_envs is not None:
    env_cfg.scene.num_envs = args_cli.num_envs
params = env_cfg.task.events[args_cli.event].params
if args_cli.bank_size is not None:
    if not isinstance(params["fill"], mdp.FillFromCandidates):
        raise SystemExit(f"--bank_size needs a candidates fill, but the event fills with {params['fill']!r}.")
    params["fill"] = dataclasses.replace(params["fill"], bank_size=args_cli.bank_size)
params.pop("record", None)  # only the filled states

with launch_simulation(env_cfg, args_cli):
    env = gym.make(args_cli.task, cfg=env_cfg)
    bank = mdp.reset_bank(env.unwrapped.state, f"events.{args_cli.event}")
    bank.save(args_cli.out)
    worlds = "one shared bank" if bank.shared else f"{bank.num_worlds} worlds"
    print(f"Saved {bank.size} states for {worlds} to {args_cli.out}.")
    env.close()
