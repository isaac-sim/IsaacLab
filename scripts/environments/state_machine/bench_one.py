# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-point throughput/memory benchmark for Isaac-Lift-Soft-Franka-v0.

Builds the soft-lift environment on a chosen backend with N environments, steps
it with a constant action, and prints one machine-parseable ``BENCH ...`` line
reporting steps/s, env-steps/s, GPU memory, and startup time. Failures
(including CUDA OOM) are caught and reported as ``status=FAIL`` so an external
sweep can detect the environment ceiling.

.. code-block:: bash

    isaaclab.bat -p scripts/environments/state_machine/bench_one.py --backend newton --num_envs 1024
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Single-point soft-lift throughput benchmark.")
parser.add_argument("--backend", type=str, default="newton", choices=["newton", "physx"], help="Physics backend.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--warmup", type=int, default=30, help="Warmup steps before timing.")
parser.add_argument("--steps", type=int, default=120, help="Timed steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaacsim.core.experimental.utils.app import enable_extension

enable_extension("omni.usd.metrics.assembler.ui", enabled=False)

import time

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

TASK = "Isaac-Lift-Soft-Franka-v0"


def gpu_mem_mb():
    """Return (used_MB, total_MB) for the current CUDA device (used = total - free)."""
    try:
        free, total = torch.cuda.mem_get_info()
        return (total - free) / (1024**2), total / (1024**2)
    except Exception:
        return 0.0, 0.0


def main():
    t0 = time.perf_counter()
    status = "OK"
    reason = ""
    startup = 0.0
    steps_per_s = 0.0
    env_steps_per_s = 0.0
    used = 0.0
    total = 0.0
    try:
        cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
        cfg = resolve_presets(cfg, selected=(["physx"] if args_cli.backend == "physx" else []))
        cfg.sim.device = args_cli.device
        cfg.scene.num_envs = args_cli.num_envs

        env = gym.make(TASK, cfg=cfg, render_mode=None)
        env.reset()
        startup = time.perf_counter() - t0

        actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
        actions[:, 3] = 1.0  # valid identity quaternion (w=1) for the IK target

        for _ in range(args_cli.warmup):
            env.step(actions)
        torch.cuda.synchronize()

        t1 = time.perf_counter()
        for _ in range(args_cli.steps):
            env.step(actions)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t1

        steps_per_s = args_cli.steps / dt if dt > 0 else 0.0
        env_steps_per_s = steps_per_s * args_cli.num_envs
        used, total = gpu_mem_mb()
        env.close()
    except BaseException as e:  # catch OOM (may be RuntimeError / cuda error) and anything else
        status = "FAIL"
        reason = (type(e).__name__ + ": " + str(e)).replace("\n", " ").replace('"', "'")[:200]
        used, total = gpu_mem_mb()

    print(
        f"BENCH backend={args_cli.backend} num_envs={args_cli.num_envs} status={status} "
        f"startup_s={startup:.1f} steps_per_s={steps_per_s:.2f} env_steps_per_s={env_steps_per_s:.1f} "
        f'gpu_used_MB={used:.0f} gpu_total_MB={total:.0f} reason="{reason}"',
        flush=True,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
