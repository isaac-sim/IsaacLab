# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test direct-workflow Cartpole actuation with both PhysX backends."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys

import pytest


def _run_cartpole_diagnostic(physics_backend: str) -> None:
    """Run the Cartpole actuation diagnostic in an isolated simulator process."""
    from isaaclab.app import launch_simulation

    from isaaclab_tasks.core.cartpole.cartpole_direct_env_cfg import CartpoleEnvCfg
    from isaaclab_tasks.utils.hydra import resolve_presets

    cfg = resolve_presets(CartpoleEnvCfg(), (physics_backend,))
    cfg.scene.num_envs = 4
    cfg.sim.device = "cuda:0"
    cfg.seed = 0
    cfg.initial_cart_position_range = (0.0, 0.0)
    cfg.initial_cart_velocity_range = (0.0, 0.0)
    cfg.initial_pole_angle_range = (0.0, 0.0)
    cfg.initial_pole_velocity_range = (0.0, 0.0)

    launcher_args = {
        "device": "cuda:0",
        "headless": True,
        "visualizer": None,
        "visualizer_explicit": True,
    }
    with launch_simulation(cfg, launcher_args):
        import gymnasium as gym
        import torch

        import isaaclab_tasks  # noqa: F401, PLC0415

        env = gym.make("Isaac-Cartpole-Direct", cfg=cfg)
        try:
            has_collision_groups = bool(env.unwrapped.scene.stage.GetPrimAtPath("/World/collisions"))
            assert has_collision_groups == (physics_backend == "isaacsim_physx"), (
                f"Unexpected collision-group state for {physics_backend}: {has_collision_groups}."
            )

            env.reset()
            actions = torch.ones(env.action_space.shape, device=env.unwrapped.device)
            with torch.inference_mode():
                for _ in range(20):
                    env.step(actions)

            cart_idx = env.unwrapped._cart_dof_idx[0]
            cart_vel = env.unwrapped.cartpole.data.joint_vel.torch[:, cart_idx]
            assert torch.all(cart_vel > 1.0), f"Expected every cart to accelerate, got {cart_vel.tolist()}."
            torch.testing.assert_close(cart_vel, cart_vel[0].expand_as(cart_vel))
            print(f"CARTPOLE_DIAGNOSTIC_PASSED {physics_backend}")
        finally:
            env.close()


@pytest.mark.isaacsim_ci
@pytest.mark.integration
@pytest.mark.parametrize("physics_backend", ("isaacsim_physx", "ovphysx"))
def test_cartpole_direct_physx_actuation_is_consistent_across_environments(physics_backend: str):
    """Verify consistent Cartpole actuation and backend-specific collision isolation."""
    if physics_backend == "isaacsim_physx":
        from isaaclab.app import AppLauncher

        if not AppLauncher.is_available():
            pytest.skip("Isaac Sim is not installed")
    if physics_backend == "ovphysx" and importlib.util.find_spec("ovphysx") is None:
        pytest.skip("OVPhysX is not installed")

    result = subprocess.run(
        [sys.executable, __file__, "--physics_backend", physics_backend],
        capture_output=True,
        text=True,
        timeout=180,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"Cartpole diagnostic failed:\n{output}"
    assert "Traceback (most recent call last):" not in output, f"Cartpole diagnostic failed:\n{output}"
    assert f"CARTPOLE_DIAGNOSTIC_PASSED {physics_backend}" in output, f"Cartpole diagnostic failed:\n{output}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--physics_backend", choices=("isaacsim_physx", "ovphysx"), required=True)
    args = parser.parse_args()
    _run_cartpole_diagnostic(args.physics_backend)
