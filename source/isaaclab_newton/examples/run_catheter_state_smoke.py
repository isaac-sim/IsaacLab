#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke-test the CatheterStateEnv — no RL library required.

Usage
-----
./isaaclab.sh -p source/isaaclab_newton/examples/run_catheter_state_smoke.py
"""

from __future__ import annotations

import time

import torch


def main():
    from isaaclab_newton.envs.catheter_state_env import CatheterStateEnv, CatheterStateEnvCfg

    cfg = CatheterStateEnvCfg(num_envs=4, device="cuda")
    env = CatheterStateEnv(cfg=cfg)

    print(f"[INFO] CatheterStateEnv created")
    print(f"  num_envs         : {env.num_envs}")
    print(f"  observation_space: {env.observation_space}")
    print(f"  action_space     : {env.action_space}")
    print(f"  obs dim          : {env._obs_dim}")

    obs, _ = env.reset()
    print(f"\n[INFO] Reset complete. obs shape: {obs.shape}")
    print(f"  obs sample [env 0]: {obs[0, :6].tolist()} ...")

    num_steps = 200
    total_reward = torch.zeros(env.num_envs, device=env.device)
    t0 = time.perf_counter()

    for step in range(num_steps):
        actions = torch.randn(env.num_envs, 2, device=env.device).clamp(-1, 1)
        obs, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward

        if step % 50 == 0:
            tip_pos = env.solver.data.positions[:, -1, :]
            target = env.target_positions
            dist = torch.norm(tip_pos - target, dim=-1)
            print(
                f"  step {step:4d}  |  reward: {reward.mean().item():+8.3f}"
                f"  |  tip-target dist: {dist.mean().item():.4f} m"
                f"  |  terminated: {terminated.sum().item()}"
            )

    elapsed = time.perf_counter() - t0
    fps = num_steps * env.num_envs / elapsed

    print(f"\n[INFO] {num_steps} steps x {env.num_envs} envs in {elapsed:.2f}s")
    print(f"  throughput: {fps:.0f} env-steps/s")
    print(f"  avg reward: {(total_reward / num_steps).mean().item():+.3f}")
    print("\n[OK] Smoke test passed.")


if __name__ == "__main__":
    main()
