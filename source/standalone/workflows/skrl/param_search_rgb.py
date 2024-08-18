# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer

from policies import Shared
from typing import NamedTuple

# add argparse arguments
parser = argparse.ArgumentParser(description="Param search for an RL agent with skrl.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument("--arch_type", type=str, default="cnn-rgb-state", help="Type of neural network used for policies")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
from datetime import datetime

import skrl
from skrl.utils import set_seed

from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper, process_skrl_cfg

import optuna
import torch.optim as optim

def objective(trial: optuna.Trial, env_cfg: dict, experiment_cfg: dict, agent_cfg: dict, args_cli):
    args_cli_seed = args_cli.seed

    # Randomize
    experiment_cfg["agent"]["rollouts"] = trial.suggest_int("agent_rollouts", 8, 128)

    # NOTE: Enviornment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    set_seed(args_cli_seed if args_cli_seed is not None else experiment_cfg["seed"])

    # NOTE: Policies
    models = {}
    models["policy"] = Shared(env.observation_space, env.action_space, env_cfg.sim.device, type=args_cli.arch_type)
    models["value"] = models["policy"]

    # NOTE: Memory replay
    memory_size = experiment_cfg["agent"]["rollouts"]
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device)

    experiment_cfg["agent"]["rewards_shaper"] = None

    # NOTE: Agent config
    agent_cfg.update(process_skrl_cfg(experiment_cfg["agent"], ml_framework=args_cli.ml_framework))
    agent_cfg["state_preprocessor_kwargs"].update({"size": env.observation_space, "device": env.device})
    agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": env.device})
    agent_cfg["state_preprocessor"] = ""
    agent_cfg["value_preprocessor"] = ""

    # Randomize
    agent_cfg["learning_rate"] = trial.suggest_float("agent_lr", 3.e-4, 3.e-2)
    agent_cfg["entropy_loss_scale"] = trial.suggest_float("agent_entropy_scale", 0.0, 0.09)
    agent_cfg["learning_epochs"] = trial.suggest_int("agent_learning_epochs", 8, 32)
    agent_cfg["mini_batches"] = trial.suggest_int("agent_mini_batches", 8, 32)

    agent = PPO(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )

    # NOTE: Trainer
    trainer_cfg = experiment_cfg["trainer"]
    trainer_cfg["close_environment_at_exit"] = False
    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)

    rewards_metric = trainer.train()

    env.close()

    return rewards_metric

def param_search():
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    env_cfg.sim.device = f"cuda:0"

    experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")
    agent_cfg = PPO_DEFAULT_CONFIG.copy()

    # Randomize the configs
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, 
        env_cfg=env_cfg, 
        experiment_cfg=experiment_cfg, 
        agent_cfg=agent_cfg, 
        args_cli=args_cli)
    , n_trials=1)

    print("Done")
    

if __name__ == "__main__":
    param_search()
    simulation_app.close()
