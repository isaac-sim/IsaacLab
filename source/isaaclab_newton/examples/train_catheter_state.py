#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Train a catheter navigation policy with PPO (rsl_rl).

Usage
-----
# Quick smoke-test (64 envs, 50 iterations):
./isaaclab.sh -p source/isaaclab_newton/examples/train_catheter_state.py \
    --num_envs 64 --max_iterations 50

# Full training (512 envs, 1500 iterations):
./isaaclab.sh -p source/isaaclab_newton/examples/train_catheter_state.py \
    --num_envs 512 --max_iterations 1500
"""

from __future__ import annotations

import argparse
import os
import sys

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train catheter state-based RL policy")
    parser.add_argument("--num_envs", type=int, default=512, help="Number of parallel environments")
    parser.add_argument("--max_iterations", type=int, default=1500, help="PPO iterations")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--log_dir", type=str, default="logs/catheter_state", help="Tensorboard log dir")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    # ---- build environment ----
    from isaaclab_newton.envs.catheter_state_env import CatheterStateEnv, CatheterStateEnvCfg

    env_cfg = CatheterStateEnvCfg(
        num_envs=args.num_envs,
        device=args.device,
    )
    env = CatheterStateEnv(cfg=env_cfg)

    # ---- wrap for rsl_rl ----
    from isaaclab_newton.envs.rsl_rl_wrapper import CatheterRslRlVecEnvWrapper

    vec_env = CatheterRslRlVecEnvWrapper(env, clip_actions=1.0)

    # ---- agent config (plain dict, no isaaclab_rl dependency) ----
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agents"))
    from rsl_rl_catheter_state_ppo_cfg import get_runner_cfg

    runner_cfg = get_runner_cfg()
    runner_cfg["max_iterations"] = args.max_iterations
    runner_cfg["device"] = args.device
    runner_cfg["seed"] = args.seed

    # ---- build runner ----
    from rsl_rl.runners import OnPolicyRunner

    runner = OnPolicyRunner(vec_env, runner_cfg, log_dir=args.log_dir, device=args.device)

    if args.resume:
        runner.load(os.path.join(args.log_dir, "model.pt"))

    # ---- train ----
    runner.learn(num_learning_iterations=runner_cfg["max_iterations"], init_at_random_ep_len=True)

    print(f"\n[INFO] Training complete.  Logs saved to {args.log_dir}")


if __name__ == "__main__":
    main()
