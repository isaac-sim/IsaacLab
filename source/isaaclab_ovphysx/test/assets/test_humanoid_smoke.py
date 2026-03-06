# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test: create humanoid env with ovphysx backend and run 100 RL steps."""

import os
import sys

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

import warp as wp

wp.init()

import sys as _sys
_hidden_pxr = {}
for _k in list(_sys.modules):
    if _k == "pxr" or _k.startswith("pxr."):
        _hidden_pxr[_k] = _sys.modules.pop(_k)
import ovphysx  # noqa: E402,F401
ovphysx.bootstrap()
_sys.modules.update(_hidden_pxr)
del _hidden_pxr

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # registers tasks
from isaaclab_tasks.utils import resolve_task_config, launch_simulation
from isaaclab_tasks.utils.hydra import resolve_preset_defaults


def main():
    # Inject "presets=ovphysx" via sys.argv so Hydra picks it up.
    sys.argv = [sys.argv[0], "presets=ovphysx"]
    env_cfg, agent_cfg = resolve_task_config(
        "Isaac-Humanoid-Direct-v0", "rsl_rl_cfg_entry_point"
    )
    print(f"Physics config: {type(env_cfg.sim.physics).__name__}")

    env_cfg.scene.num_envs = 16

    with launch_simulation(env_cfg, {"headless": True}):
        env = gym.make("Isaac-Humanoid-Direct-v0", cfg=env_cfg)
        obs, info = env.reset()
        print(f"Obs shape: {obs['policy'].shape}")

        rewards = []
        for step in range(100):
            action = torch.zeros(16, 21)
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward.mean().item())
            if step % 20 == 0:
                finite = torch.isfinite(obs["policy"]).all().item()
                print(
                    f"  step {step:3d}: reward={reward.mean():.4f} "
                    f"terminated={terminated.sum().item()}/{16} "
                    f"obs_finite={finite}"
                )

        env.close()
        avg_reward = np.mean(rewards)
        print(f"\nDone. Average reward over 100 steps: {avg_reward:.4f}")
        print("SUCCESS: Humanoid env ran 100 steps with ovphysx backend")


if __name__ == "__main__":
    main()
