# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train an algorithm."""

import argparse
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train an RL agent with HARL.")
parser.add_argument("--video", action="store_true", help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=20000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=1, help="Seed used for the environment")
parser.add_argument("--save_interval", type=int, default=None, help="How often to save the model")
parser.add_argument("--save_checkpoints", action="store_true", default=False, help="Whether or not to save checkpoints")
parser.add_argument("--checkpoint_interval", type=int, default=200, help="How often to save a model checkpoint")
parser.add_argument("--log_interval", type=int, default=None, help="How often to log outputs")
parser.add_argument("--exp_name", type=str, default="test", help="Name of the Experiment")
parser.add_argument("--num_env_steps", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--dir", type=str, default=None, help="folder with trained models")
parser.add_argument("--debug", action="store_true", help="whether to run in debug mode for visualization")
parser.add_argument("--adversarial_training_mode", default="parallel", choices=["parallel", "ladder", "leapfrog"], help="the mode type for adversarial training,\
                     note on ladder training with teams that are composed of heterogenous agents, the two teams must place the robots in the same order in their environment \
                    for ladder to work")
parser.add_argument("--adversarial_training_iterations", default=50_000_000, type=int,help="the number of iterations to swap training for adversarial modes like ladder and leapfrog")

parser.add_argument(
    "--algorithm",
    type=str,
    default="happo",
    choices=[
        "happo",
        "hatrpo",
        "haa2c",
        "mappo",
        "mappo_unshare",
        "happo_adv"
    ],
    help="Algorithm name. Choose from: happo, hatrpo, haa2c, mappo, and mappo_unshare.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

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

    algo_args = agent_cfg

    algo_args["eval"]["use_eval"] = False
    algo_args["train"]["n_rollout_threads"] = args["num_envs"]
    algo_args["train"]["num_env_steps"] = args["num_env_steps"]
    algo_args["train"]["eval_interval"] = args["save_interval"]
    algo_args["train"]["save_checkpoints"] = args["save_checkpoints"]
    algo_args["train"]["checkpoint_interval"] = args["checkpoint_interval"]
    algo_args["train"]["log_interval"] = args["log_interval"]
    algo_args["train"]["model_dir"] = args["dir"]
    algo_args["seed"]["specify_seed"] = True
    algo_args["seed"]["seed"] = args["seed"]
    algo_args["algo"]["adversarial_training_mode"] = args["adversarial_training_mode"]
    algo_args["algo"]["adversarial_training_iterations"] = args["adversarial_training_iterations"]

    env_args = {}
    env_cfg.scene.num_envs = args["num_envs"]
    env_args["task"] = args["task"]
    env_args["config"] = env_cfg
    hms_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    env_args["video_settings"] = {
        "video": True if args["video"] else False,
        "video_length": args["video_length"],
        "video_interval": args["video_interval"],
        "log_dir": os.path.join(
            algo_args["logger"]["log_dir"],
            "isaaclab",
            args["task"],
            args["algorithm"],
            args["exp_name"],
            "-".join(["seed-{:0>5}".format(agent_cfg["seed"]["seed"]), hms_time]),
            "videos",
        ),
    }

    env_args["headless"] = args["headless"]
    env_args["debug"] = args["debug"]


    # create runner

    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
    runner.run()
    runner.close()


if __name__ == "__main__":
    main() # type: ignore
    simulation_app.close()
