# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate the reset states of an MDP2 task into a file, to load them in later runs.

Building the task builds the reset states of its ``reset_to_states`` event from the event's source, typically
``mdp.FromTask`` with a generator task. The script saves them. Later runs load them with
``source=mdp.FromFile(path)``, in the same number of worlds unless the states are shared. For example::

    uv run python scripts/tools/generate_reset_states.py --task Isaac-Lift-KukaAllegro-MDP2 --num_envs 4096 \\
        --num_states 64 --out kuka_lift_reset_states.pt physics=newton_mjwarp
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
parser.add_argument("--num_states", type=int, default=None, help="States per world, for a FromTask source.")
parser.add_argument("--event", default="reset", help="Name of the reset_to_states event.")
parser.add_argument("--out", required=True, help="File to save the states to.")
add_launcher_args(parser)
parser.set_defaults(device=None)  # let the task select the device
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

env_cfg, _ = resolve_task_config(args_cli.task, "")
if args_cli.num_envs is not None:
    env_cfg.scene.num_envs = args_cli.num_envs
params = env_cfg.task.events[args_cli.event].params
if args_cli.num_states is not None:
    if not isinstance(params["source"], mdp.FromTask):
        raise SystemExit(f"--num_states needs a FromTask source, but the event's source is {params['source']!r}.")
    params["source"] = dataclasses.replace(params["source"], num_states=args_cli.num_states)
params.pop("record", None)  # only the generated states

with launch_simulation(env_cfg, args_cli):
    env = gym.make(args_cli.task, cfg=env_cfg)
    reset_states = mdp.get_reset_states(env.unwrapped.state, f"events.{args_cli.event}")
    reset_states.save(args_cli.out)
    worlds = "all worlds, shared" if reset_states.shared else f"each of {reset_states.num_worlds} worlds"
    print(f"Saved {reset_states.size} states for {worlds} to {args_cli.out}.")
    env.close()
