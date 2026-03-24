# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark script: run 100 steps of Cartpole with 64 envs and report throughput."""

import argparse
import sys
import time

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config

parser = argparse.ArgumentParser(description="Cartpole benchmark for Isaac Lab 3.0")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Cartpole-v0", help="Name of the task.")
parser.add_argument("--num_steps", type=int, default=100, help="Number of steps to run.")
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args


def main():
    torch.manual_seed(42)

    env_cfg, _ = resolve_task_config(args_cli.task, "")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device

        env = gym.make(args_cli.task, cfg=env_cfg)

        print(f"[INFO]: Task: {args_cli.task}")
        print(f"[INFO]: Num envs: {args_cli.num_envs}")
        print(f"[INFO]: Observation space: {env.observation_space}")
        print(f"[INFO]: Action space: {env.action_space}")

        obs, info = env.reset()
        print(f"[INFO]: Initial obs shape: {obs['policy'].shape if isinstance(obs, dict) else obs.shape}")

        # Warmup
        with torch.inference_mode():
            for _ in range(10):
                actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
                env.step(actions)

        # Benchmark
        total_rewards = torch.zeros(args_cli.num_envs, device=env.unwrapped.device)
        obs, info = env.reset()

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()

        with torch.inference_mode():
            for step in range(args_cli.num_steps):
                actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
                obs, rew, terminated, truncated, info = env.step(actions)
                total_rewards += rew
                if step % 20 == 0:
                    print(f"  Step {step:4d}: mean_reward={rew.mean().item():.4f}, obs_sample={obs['policy'][0, :4].tolist() if isinstance(obs, dict) else obs[0, :4].tolist()}")

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - t0

        total_env_steps = args_cli.num_steps * args_cli.num_envs
        throughput = total_env_steps / elapsed

        print(f"\n{'='*60}")
        print(f"BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"Task:              {args_cli.task}")
        print(f"Num envs:          {args_cli.num_envs}")
        print(f"Num steps:         {args_cli.num_steps}")
        print(f"Total env-steps:   {total_env_steps}")
        print(f"Elapsed time:      {elapsed:.4f} s")
        print(f"Throughput:        {throughput:.0f} env-steps/s")
        print(f"Mean total reward: {total_rewards.mean().item():.4f}")
        print(f"{'='*60}")

        env.close()


if __name__ == "__main__":
    main()
