# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train PPO on any MP-based IsaacLab env with Stable-Baselines3 (optional WandB logging)."""

import argparse
import os
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train PPO on MP environment (SB3).")
parser.add_argument(
    "--base_id", type=str, required=True, help="Base step env id (e.g., Isaac-Box-Pushing-Dense-step-Franka-v0)."
)
parser.add_argument("--mp_id", type=str, default=None, help="MP env id to register/use.")
parser.add_argument("--mp_type", type=str, default="ProDMP", choices=["ProDMP", "ProMP", "DMP"], help="MP backend.")
parser.add_argument(
    "--mp_wrapper",
    type=str,
    default="isaaclab_tasks.manager_based.box_pushing.mp_wrapper:BoxPushingMPWrapper",
    help="MP wrapper class import path.",
)
parser.add_argument("--mp_cfg", type=str, default=None, help="Path to SB3 MP config yaml.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel envs (overrides config if set).")
parser.add_argument("--device", type=str, default="cuda:0", help="Simulation device.")
parser.add_argument("--logger", type=str, default=None, choices=["wandb"], help="Use wandb logging.")
parser.add_argument("--log_project_name", type=str, default="isaaclab-mp", help="wandb project.")
parser.add_argument("--log_run_group", type=str, default=None, help="wandb group.")
parser.add_argument("--headless", action="store_true", help="Run headless.")
args_cli, _ = parser.parse_known_args()

# launch app early
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import importlib
import torch
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, ProgressBarCallback
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab_rl import Sb3MPVecEnvWrapper
from isaaclab_rl.sb3 import process_sb3_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.mp import upgrade

try:
    from wandb.integration.sb3 import WandbCallback

    import wandb
except ImportError:
    wandb = None
    WandbCallback = None


def load_sb3_cfg(cfg_path: str, num_envs_cli: int | None):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    policy_arch = cfg.pop("policy")
    n_timesteps = cfg.pop("n_timesteps")
    seed = cfg.pop("seed")
    cfg_num_envs = cfg.pop("num_envs", None)
    num_envs = num_envs_cli if num_envs_cli is not None else cfg_num_envs
    if num_envs is None:
        raise ValueError("num_envs must be provided via CLI or config (num_envs field).")

    # For MP: allow multiple trajectories per update (fancy_gym style).
    # Require divisibility so each env runs the same number of trajectories.
    n_steps_per_update = cfg.pop("n_steps_per_update")
    if n_steps_per_update % num_envs != 0:
        raise ValueError(f"n_steps_per_update {n_steps_per_update} not divisible by num_envs {num_envs}")
    n_steps = n_steps_per_update // num_envs
    cfg["n_steps_per_update"] = n_steps_per_update
    cfg["n_steps"] = n_steps

    # Use target minibatches as-is; user should supply a divisor of n_steps_per_update
    target_minibatches = cfg.pop("n_minibatches")
    if n_steps_per_update % target_minibatches != 0:
        raise ValueError(f"n_minibatches {target_minibatches} must divide n_steps_per_update {n_steps_per_update}")
    cfg["n_minibatches"] = target_minibatches
    # SB3 expects batch_size; process_sb3_cfg will recompute, but set here for clarity
    cfg["batch_size"] = int(n_steps_per_update / target_minibatches)
    return policy_arch, n_timesteps, seed, cfg, num_envs


def resolve_class(path: str):
    mod_name, cls_name = path.split(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


def main():
    mp_wrapper_cls = resolve_class(args_cli.mp_wrapper)
    mp_id = args_cli.mp_id or f"Isaac_MP/{args_cli.base_id.split('/')[-1]}-{args_cli.mp_type}"

    mp_id = upgrade(
        mp_id=mp_id,
        base_id=args_cli.base_id,
        mp_wrapper_cls=mp_wrapper_cls,
        mp_type=args_cli.mp_type,
        device=args_cli.device,
    )

    cfg_path = args_cli.mp_cfg or os.path.join(
        "source",
        "isaaclab_tasks",
        "isaaclab_tasks",
        "manager_based",
        "box_pushing",
        "config",
        "franka",
        "agents",
        "sb3_ppo_cfg_mp.yaml",
    )
    policy_arch, n_timesteps, seed, agent_cfg, num_envs = load_sb3_cfg(cfg_path, args_cli.num_envs)

    env_cfg = parse_env_cfg(
        args_cli.base_id, device=args_cli.device, num_envs=num_envs, use_fabric=not args_cli.headless
    )

    base_env = gym.make(mp_id, cfg=env_cfg)

    # Sb3 wrapper requires a ManagerBasedRLEnv/DirectRLEnv; unwrap to that type
    vec_env = Sb3MPVecEnvWrapper(base_env)
    vec_env.seed(seed)

    # Process SB3 config (handles batch_size from n_steps/n_minibatches)
    agent_cfg = process_sb3_cfg(agent_cfg, num_envs=num_envs)
    # Parse policy_kwargs if provided as a string in config
    if isinstance(agent_cfg.get("policy_kwargs"), str):
        policy_kwargs_str = agent_cfg["policy_kwargs"]
        # eval with restricted globals; allow access to torch.nn as nn and dict/list literals
        safe_globals = {"nn": torch.nn, "dict": dict, "list": list, "__builtins__": {}}
        agent_cfg["policy_kwargs"] = eval(policy_kwargs_str, safe_globals, {})

    norm_obs = agent_cfg.pop("normalize_input", False)
    norm_rew = agent_cfg.pop("normalize_value", False)
    clip_obs = agent_cfg.pop("clip_obs", None)
    clip_rew = agent_cfg.pop("clip_rew", None)
    if norm_obs or norm_rew:
        vec_env = VecNormalize(
            vec_env,
            training=True,
            norm_obs=norm_obs,
            norm_reward=norm_rew,
            clip_obs=clip_obs,
            gamma=agent_cfg.get("gamma", 0.99),
            clip_reward=clip_rew,
        )

    # Strip keys not accepted by SB3 PPO ctor
    agent_cfg.pop("n_steps_per_update", None)
    agent_cfg.pop("num_envs", None)

    log_dir = os.path.join(
        "logs",
        "sb3",
        f"mp_{mp_id.replace('/', '_')}",
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    os.makedirs(log_dir, exist_ok=True)

    callbacks = []
    if args_cli.logger == "wandb" and wandb is not None:
        wandb.init(
            project=args_cli.log_project_name,
            group=args_cli.log_run_group,
            config={"policy_type": policy_arch, "total_timesteps": n_timesteps, "env_name": mp_id},
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        callbacks.append(
            WandbCallback(
                gradient_save_freq=0,
                model_save_freq=10000,
                model_save_path=os.path.join(log_dir, "models"),
                log="all",
            )
        )
    else:
        callbacks.append(
            CheckpointCallback(save_freq=10000, save_path=os.path.join(log_dir, "models"), name_prefix="ppo")
        )
    # always show training progress bar
    callbacks.append(ProgressBarCallback())

    model = PPO(
        policy_arch,
        vec_env,
        verbose=1,
        tensorboard_log=os.path.join(log_dir, "tb"),
        **agent_cfg,
    )

    model.learn(total_timesteps=n_timesteps, callback=callbacks)

    vec_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
