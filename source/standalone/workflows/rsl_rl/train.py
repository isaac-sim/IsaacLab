# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""


import argparse
import os

from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
args_cli = parser.parse_args()

config = {"headless": args_cli.headless}
# load cheaper kit config in headless
if args_cli.headless:
    app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.gym.headless.kit"
else:
    app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.kit"
# launch the simulator
simulation_app = SimulationApp(config, experience=app_experience)

"""Rest everything follows."""


import gym
import os
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from omni.isaac.orbit.utils.dict import print_dict
from omni.isaac.orbit.utils.io import dump_pickle, dump_yaml

import omni.isaac.contrib_envs  # noqa: F401
import omni.isaac.orbit_envs  # noqa: F401
from omni.isaac.orbit_envs.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.orbit_envs.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

from config import parse_rslrl_cfg


def main():
    """Train with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)
    agent_cfg = parse_rslrl_cfg(args_cli.task)
    # override config from command line
    if args_cli.seed is not None:
        agent_cfg["seed"] = args_cli.seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg["runner"]["experiment_name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs
    log_dir = datetime.now().strftime("%b%d_%H-%M-%S")
    if agent_cfg["runner"]["run_name"]:
        log_dir += f"_{agent_cfg['runner']['run_name']}"
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, headless=args_cli.headless, viewport=args_cli.video)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg, log_dir=log_dir, device=agent_cfg["device"])
    # save resume path before creating a new log_dir
    if agent_cfg["runner"]["resume"]:
        # specify directory for logging runs
        run_dir = "*"
        if agent_cfg["runner"]["load_run"] != -1:
            run_dir = agent_cfg["runner"]["load_run"]
        # specify name of checkpoint
        checkpoint_file = None
        if agent_cfg["runner"]["checkpoint"] != -1:
            checkpoint_file = f"model_{agent_cfg['runner']['checkpoint']}.pt"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # set seed of the environment
    env.seed(agent_cfg["seed"])
    # run training
    runner.learn(num_learning_iterations=agent_cfg["runner"]["max_iterations"], init_at_random_ep_len=True)

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
