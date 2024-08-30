# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import optuna
import torch.optim as optim

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")


# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../rsl_rl/'))
sys.path.append(parent_dir)
print(sys.path)
from rsl_rl.runners import OnPolicyRunner

from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

def objective(
    trial: optuna.Trial, env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg, args_cli, log_dir):
    # Randomizations of the config
    agent_cfg.num_steps_per_env = trial.suggest_int("num_steps_per_env", 24, 100)
    agent_cfg.algorithm.clip_param = trial.suggest_float("clip_param", 0.2, 0.8)
    agent_cfg.algorithm.entropy_coef = trial.suggest_float("entropy_coef", 0.0, 0.01)
    agent_cfg.algorithm.learning_rate = trial.suggest_float("learning_rate", 1.0e-4, 0.1)
    agent_cfg.algorithm.value_loss_coef = trial.suggest_float("value_loss_coef", 1.0, 3.0)
    agent_cfg.algorithm.desired_kl = trial.suggest_float("desired_kl", 0.01, 0.1)
    agent_cfg.algorithm.gamma = trial.suggest_float("gamma", 0.92, 0.99)
    agent_cfg.algorithm.num_mini_batches = trial.suggest_int("num_mini_batches", 4, 24)
    agent_cfg.algorithm.num_learning_epochs = trial.suggest_int("num_learning_epochs", 5, 12)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    
    # write git state to logs
    runner.add_git_repo_to_log(__file__)

    # set seed of the environment
    env.seed(agent_cfg.seed)

    # run training
    mean_reward = runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()

    return mean_reward


def param_search():
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        args_cli.task, 
        device=args_cli.device, 
        num_envs=args_cli.num_envs, 
        use_fabric=not args_cli.disable_fabric)

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments (NOTE: needed for rewards to accumulate)
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # Randomize the configs
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, 
        env_cfg=env_cfg,
        agent_cfg=agent_cfg, 
        args_cli=args_cli,
        log_dir=log_dir)
    , n_trials=1)

    print("Done")


if __name__ == "__main__":
    # run the main function
    param_search()
    # close sim app
    simulation_app.close()
