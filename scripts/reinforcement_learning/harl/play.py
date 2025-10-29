# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train an algorithm."""

import argparse

# import numpy as np
import sys
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train an RL agent with HARL.")
parser.add_argument(
    "--algorithm",
    type=str,
    default="happo",
    choices=[
        "happo",
        "hatrpo",
        "haa2c",
        "haddpg",
        "hatd3",
        "hasac",
        "had3qn",
        "maddpg",
        "matd3",
        "mappo",
        "happo_adv",
    ],
    help="Algorithm name. Choose from: happo, hatrpo, haa2c, haddpg, hatd3, hasac, had3qn, maddpg, matd3, mappo.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--num_env_steps", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--dir", type=str, default=None, help="folder with trained models")
parser.add_argument("--debug", action="store_true", help="whether to run in debug mode for visualization")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from harl.runners import RUNNER_REGISTRY

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = f"harl_{algorithm}_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    args = args_cli.__dict__

    args["env"] = "isaaclab"
    args["algo"] = args["algorithm"]
    args["exp_name"] = "play"

    algo_args = agent_cfg

    algo_args["eval"]["use_eval"] = False
    algo_args["render"]["use_render"] = True
    algo_args["train"]["model_dir"] = args["dir"]

    env_args = {}
    env_cfg.scene.num_envs = args["num_envs"]
    env_args["task"] = args["task"]
    env_args["config"] = env_cfg
    env_args["video_settings"] = {}
    env_args["video_settings"]["video"] = False
    env_args["headless"] = args["headless"]
    env_args["debug"] = args["debug"]

    # create runner
    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)

    obs, _, _ = runner.env.reset()

    max_action_space = 0

    for agent_id, obs_space in runner.env.action_space.items():
        if obs_space.shape[0] > max_action_space:
            max_action_space = obs_space.shape[0]

    actions = torch.zeros((args["num_envs"], runner.num_agents, max_action_space), dtype=torch.float32, device="cuda:0")
    rnn_states = torch.zeros(
        (
            args["num_envs"],
            runner.num_agents,
            runner.recurrent_n,
            runner.rnn_hidden_size,
        ),
        dtype=torch.float32,
        device="cuda:0",
    )
    masks = torch.ones(
        (args["num_envs"], runner.num_agents, 1),
        dtype=torch.float32,
    )

    while simulation_app.is_running():
        with torch.inference_mode():
            for agent_id, (_, agent_obs) in enumerate(obs.items()):
                action, _, rnn_state = runner.actor[agent_id].get_actions(
                    agent_obs, rnn_states[:, agent_id, :], masks[:, agent_id, :], None, None
                )
                action_space = action.shape[1]
                actions[:, agent_id, :action_space] = action
                rnn_states[:, agent_id, :] = rnn_state

            obs, _, _, dones, _, _ = runner.env.step(actions)

            dones_env = torch.all(dones, dim=1)
            masks = torch.ones((args["num_envs"], runner.num_agents, 1), dtype=torch.float32, device="cuda:0")
            masks[dones_env] = 0.0
            rnn_states[dones_env] = torch.zeros(
                ((dones_env).sum(), runner.num_agents, runner.recurrent_n, runner.rnn_hidden_size),
                dtype=torch.float32,
                device="cuda:0",
            )

            if runner.env.unwrapped.sim._number_of_steps >= args["num_env_steps"]:
                break

    runner.env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
