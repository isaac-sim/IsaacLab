# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import torch
import warnings

import pytest

# Suppress noisy PIL runtime warnings from the bundled Omni Pillow.
warnings.filterwarnings(
    "ignore",
    message="The _imaging extension was built for another version of Pillow or PIL",
    category=RuntimeWarning,
)


def _require_omni_app():
    """Start a headless Omni app or skip if unavailable."""
    try:
        from isaaclab.app import AppLauncher
    except ImportError:
        pytest.skip("AppLauncher unavailable (Omni/Isaac not installed).")
    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    try:
        import omni.kit.app  # noqa: F401
    except ImportError:
        pytest.skip("OmniKit not available.")
    return simulation_app


def test_mp_box_pushing_smoke():
    """Smoke test: one ProDMP rollout on box pushing, verify shapes and aggregation."""
    sim_app = _require_omni_app()

    try:
        from isaaclab_tasks.manager_based.box_pushing.mp_wrapper import BoxPushingMPWrapper
        from isaaclab_tasks.utils import parse_env_cfg
        from isaaclab_tasks.utils.mp import upgrade
    except ImportError as e:
        sim_app.close()
        pytest.skip(f"IsaacLab imports unavailable: {e}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    base_id = "Isaac-Box-Pushing-Dense-step-Franka-v0"
    mp_id = upgrade(
        mp_id="Isaac_MP/Box-Pushing-Dense-ProDMP-Franka-v0",
        base_id=base_id,
        mp_wrapper_cls=BoxPushingMPWrapper,
        mp_type="ProDMP",
        device=device,
    )

    env_cfg = parse_env_cfg(base_id, device=device, num_envs=2, use_fabric=True)
    env = gym.make(mp_id, cfg=env_cfg)

    obs, _ = env.reset()
    assert obs.shape[0] == env_cfg.scene.num_envs

    action = torch.randn(env.action_space.shape, device=device)
    obs, rew, term, trunc, info = env.step(action)

    # reward aggregation should return per-env tensor
    assert torch.is_tensor(rew)
    assert rew.shape[0] == env_cfg.scene.num_envs
    assert rew.dtype == torch.float32

    assert torch.is_tensor(term) and torch.is_tensor(trunc)
    assert term.shape[0] == env_cfg.scene.num_envs
    assert trunc.shape[0] == env_cfg.scene.num_envs

    env.close()
    sim_app.close()
