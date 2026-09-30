# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark a whole cart-pole environment step captured as one CUDA graph against the manager-based envs.

Compares, on Newton MJWarp:

- ``Isaac-Cartpole`` with the torch MDP managers (``env.step``),
- the same task on the Warp frontend (per-stage graphs), and
- :class:`~isaaclab_tasks.contrib.newton_step_program.captured_cartpole.CapturedCartpole`, whose Warp MDP and Newton
  step program are recorded into one graph per environment step.

Usage:
    uv run python scripts/benchmarks/benchmark_captured_env_step.py --num_envs 4096
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Benchmark whole-environment-step capture on Newton.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments.")
parser.add_argument("--num_iterations", type=int, default=500, help="Timed environment steps per variant.")
parser.add_argument("--warmup", type=int, default=50, help="Warmup environment steps per variant.")
add_launcher_args(parser)
args_cli = parser.parse_args()

import time

import gymnasium as gym
import torch
import warp as wp

import isaaclab.sim as sim_utils

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.newton_step_program.captured_cartpole import CapturedCartpole
from isaaclab_tasks.utils.hydra import resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

TASK = "Isaac-Cartpole"


def make_cfg():
    """Return the task configuration on Newton MJWarp."""
    cfg = resolve_presets(load_cfg_from_registry(TASK, "env_cfg_entry_point"), selected=("newton_mjwarp",))
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device or "cuda:0"
    return cfg


def time_steps(step) -> float:
    """Return the mean wall time of one environment step [ms] after warmup."""
    for _ in range(args_cli.warmup):
        step()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args_cli.num_iterations):
        step()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3 / args_cli.num_iterations


def run(variant: str) -> float:
    """Build a fresh environment for ``variant`` and time its step."""
    sim_utils.create_new_stage()
    cfg = make_cfg()
    if variant == "warp frontend":
        from isaaclab_experimental.envs.frontend import WarpFrontend

        env = WarpFrontend.build_env(cfg, TASK)
    else:
        env = gym.make(TASK, cfg=cfg)
    try:
        env.unwrapped.sim._app_control_on_stop_handle = None
        env.reset()
        if variant == "whole-step graph":
            mdp = CapturedCartpole(env.unwrapped)
            wp.copy(mdp.actions, wp.from_torch(2 * torch.rand(args_cli.num_envs, 1, device=env.unwrapped.device) - 1))
            mdp.capture()
            return time_steps(mdp.replay)
        actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
        with torch.inference_mode():
            return time_steps(lambda: env.step(actions))
    finally:
        env.close()


def main():
    with launch_simulation(make_cfg(), args_cli):
        results = {variant: run(variant) for variant in ("torch managers", "warp frontend", "whole-step graph")}
    for variant, ms in results.items():
        print(f"{variant:>18}: {ms:7.3f} ms/step  {args_cli.num_envs / ms * 1e3:14,.0f} steps/s")


if __name__ == "__main__":
    main()
