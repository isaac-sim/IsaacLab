# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test direct-workflow Cartpole isolation and CUDA actuation with both PhysX backends."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys

import pytest


def _run_cartpole_diagnostic(physics_backend: str, device: str) -> None:
    """Run the Cartpole diagnostic in an isolated simulator process."""
    from isaaclab.app import launch_simulation

    from isaaclab_tasks.core.cartpole.cartpole_direct_env_cfg import CartpoleEnvCfg
    from isaaclab_tasks.utils.hydra import resolve_presets

    cfg = resolve_presets(CartpoleEnvCfg(), (physics_backend,))
    cfg.scene.num_envs = 4
    cfg.sim.device = device
    cfg.seed = 0
    cfg.initial_cart_position_range = (0.0, 0.0)
    cfg.initial_cart_velocity_range = (0.0, 0.0)
    cfg.initial_pole_angle_range = (0.0, 0.0)
    cfg.initial_pole_velocity_range = (0.0, 0.0)

    launcher_args = {
        "device": device,
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
            expected_collision_groups = physics_backend == "isaacsim_physx" or device == "cpu"
            assert has_collision_groups == expected_collision_groups, (
                f"Unexpected collision-group state for {physics_backend} on {device}: {has_collision_groups}."
            )

            if device == "cuda:0":
                env.reset()
                actions = torch.ones(env.action_space.shape, device=env.unwrapped.device)
                with torch.inference_mode():
                    for _ in range(20):
                        env.step(actions)

                cart_idx = env.unwrapped._cart_dof_idx[0]
                cart_vel = env.unwrapped.cartpole.data.joint_vel.torch[:, cart_idx]
                assert torch.all(cart_vel > 1.0), f"Expected every cart to accelerate, got {cart_vel.tolist()}."
                torch.testing.assert_close(cart_vel, cart_vel[0].expand_as(cart_vel))
            print(f"CARTPOLE_DIAGNOSTIC_PASSED {physics_backend} {device}")
        finally:
            env.close()


@pytest.mark.isaacsim_ci
@pytest.mark.integration
@pytest.mark.parametrize(
    "physics_backend,device",
    (
        pytest.param("isaacsim_physx", "cuda:0", id="isaacsim-physx-cuda"),
        pytest.param("isaacsim_physx", "cpu", id="isaacsim-physx-cpu"),
        pytest.param("ovphysx", "cuda:0", id="ovphysx-cuda"),
        pytest.param("ovphysx", "cpu", id="ovphysx-cpu"),
    ),
)
def test_cartpole_direct_physx_isolation_matches_backend_device(physics_backend: str, device: str):
    """Verify backend-specific collision isolation and consistent CUDA actuation."""
    if physics_backend == "isaacsim_physx":
        from isaaclab.app import AppLauncher

        if not AppLauncher.is_available():
            pytest.skip("Isaac Sim is not installed")
    if physics_backend == "ovphysx" and importlib.util.find_spec("ovphysx") is None:
        pytest.skip("OVPhysX is not installed")

    result = subprocess.run(
        [sys.executable, __file__, "--physics_backend", physics_backend, "--device", device],
        capture_output=True,
        text=True,
        timeout=180,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"Cartpole diagnostic failed:\n{output}"
    assert "Traceback (most recent call last):" not in output, f"Cartpole diagnostic failed:\n{output}"
    assert f"CARTPOLE_DIAGNOSTIC_PASSED {physics_backend} {device}" in output, (
        f"Cartpole diagnostic failed:\n{output}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--physics_backend", choices=("isaacsim_physx", "ovphysx"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    args = parser.parse_args()
    _run_cartpole_diagnostic(args.physics_backend, args.device)
