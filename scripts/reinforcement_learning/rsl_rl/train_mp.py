# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train PPO on an MP-based IsaacLab env with RSL-RL (fancy_gym-aligned defaults)."""

import argparse
import importlib
import os
from datetime import datetime

from isaaclab.app import AppLauncher


def resolve_class(path: str):
    mod_name, cls_name = path.split(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


parser = argparse.ArgumentParser(description="Train PPO on MP environment (RSL-RL).")
parser.add_argument(
    "--base_id",
    type=str,
    required=True,
    help="Base step env id (e.g., Isaac-Box-Pushing-Dense-step-Franka-v0).",
)
parser.add_argument("--mp_id", type=str, default=None, help="MP env id to register/use.")
parser.add_argument("--mp_type", type=str, default="ProDMP", choices=["ProDMP", "ProMP", "DMP"], help="MP backend.")
parser.add_argument(
    "--mp_wrapper",
    type=str,
    default="isaaclab_tasks.manager_based.box_pushing.mp_wrapper:BoxPushingMPWrapper",
    help="MP wrapper class import path.",
)
parser.add_argument(
    "--agent",
    type=str,
    default="isaaclab_tasks.manager_based.box_pushing.config.franka.agents.rsl_rl_cfg:BoxPushingPPORunnerCfg_mp",
    help="RSL-RL runner config class path.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel envs (overrides config if set).")
parser.add_argument("--device", type=str, default="cuda:0", help="Simulation / RL device.")
parser.add_argument("--seed", type=int, default=None, help="Seed override.")
parser.add_argument("--max_iterations", type=int, default=None, help="Max training iterations override.")
parser.add_argument("--logger", type=str, default=None, choices=["wandb", "tensorboard", "neptune"], help="Logger.")
parser.add_argument("--log_project_name", type=str, default=None, help="Project name for wandb/neptune.")
parser.add_argument("--run_name", type=str, default=None, help="Run name suffix.")
parser.add_argument("--resume", action="store_true", help="Resume from checkpoint.")
parser.add_argument("--load_run", type=str, default=None, help="Run name to resume.")
parser.add_argument("--load_checkpoint", type=str, default=None, help="Checkpoint regex to resume.")
parser.add_argument("--headless", action="store_true", help="Run headless.")
args_cli, _ = parser.parse_known_args()

# launch app early
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import dump_yaml

from isaaclab_rl import RslRlMPVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_tasks.utils.mp import upgrade


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

    env_cfg = parse_env_cfg(
        args_cli.base_id, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.headless
    )

    env = gym.make(mp_id, cfg=env_cfg)

    # resolve agent config
    agent_cls = resolve_class(args_cli.agent)
    agent_cfg = agent_cls()
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
        env_cfg.seed = args_cli.seed
    agent_cfg.device = args_cli.device
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    if args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name
    if args_cli.run_name:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.load_checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.load_checkpoint

    # wrap for RSL-RL
    vec_env = RslRlMPVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # logging dir
    log_root_path = os.path.join("logs", "rsl_rl", f"mp_{mp_id.replace('/', '_')}")
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # resume path if needed
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run or ".*", agent_cfg.load_checkpoint or ".*")
    else:
        resume_path = None

    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)
    if resume_path:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    # dump configs
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    vec_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
