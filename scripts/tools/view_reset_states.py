# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""View the reset states of an MDP2 task: browse them, roll its generator task, and play episodes from them.

The panel opens in each Viser or Newton viewer, for example::

    uv run --extra viser python scripts/tools/view_reset_states.py --task Isaac-Lift-KukaAllegro-MDP2 \\
        --num_envs 16 --viz viser physics=newton_mjwarp

then open the printed URL. ``--states`` shows saved reset states instead of generating new ones.
"""

import argparse
import sys

import gymnasium as gym
import torch
from isaaclab_experimental.envs import ResetStateViewer
from isaaclab_experimental.mdp2 import mdp

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", required=True, help="Name of an MDP2 task.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of worlds.")
parser.add_argument("--event", default="reset", help="Name of the reset_to_states event.")
parser.add_argument(
    "--states", default=None, help="Reset states saved by ResetStates.save or generate_reset_states.py."
)
parser.add_argument("--policy", choices=("zero", "random"), default="zero", help="Actions of played episodes.")
add_launcher_args(parser)
parser.set_defaults(device=None)  # let the task select the device
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

env_cfg, _ = resolve_task_config(args_cli.task, "")
env_cfg.scene.num_envs = args_cli.num_envs
if args_cli.states is not None:
    env_cfg.task.events[args_cli.event].params["source"] = mdp.FromFile(args_cli.states)

with launch_simulation(env_cfg, args_cli):
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    unwrapped = env.unwrapped
    shape = (unwrapped.num_envs, unwrapped.mdp.action_dim)
    policy = None
    if args_cli.policy == "random":
        policy = lambda observations: 2.0 * torch.rand(shape, device=unwrapped.device) - 1.0  # noqa: E731
    viewer = ResetStateViewer(env, args_cli.event, policy)
    if not viewer.attach():
        print("No Viser or Newton viewer is open: pass --viz viser or --viz newton_gl.")
    while unwrapped.sim.is_running():
        viewer.update()
    env.close()
