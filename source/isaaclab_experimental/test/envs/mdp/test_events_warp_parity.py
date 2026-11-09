# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parity tests for warp-first event MDP terms."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import warp as wp

# Skip entire module if no CUDA device available
wp.init()
pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA device required")

import isaaclab_experimental.envs.mdp.events as warp_evt
from parity_helpers import (
    DEVICE,
    NUM_ACTIONS,
    NUM_ENVS,
    NUM_JOINTS,
    MockActionManagerWarp,
    MockArticulation,
    MockArticulationData,
    MockScene,
    MockSceneEntityCfg,
    assert_close,
    copy_np_to_wp,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def _clear_function_caches():
    """Clear first-call caches on warp MDP functions so each test starts fresh.

    Functions that cache warp views via the ``hasattr`` pattern need clearing
    between tests to avoid stale references from prior fixtures.
    """
    yield
    for fn in (
        warp_evt.push_by_setting_velocity,
        warp_evt.apply_external_force_torque,
        warp_evt.reset_root_state_uniform,
        warp_evt.randomize_rigid_body_com,
    ):
        for attr in list(vars(fn)):
            if attr.startswith("_"):
                delattr(fn, attr)


@pytest.fixture()
def art_data():
    return MockArticulationData(NUM_ENVS, NUM_JOINTS, DEVICE)


@pytest.fixture()
def env_origins():
    rng = np.random.RandomState(77)
    origins_np = rng.randn(NUM_ENVS, 3).astype(np.float32)
    return wp.array(origins_np, dtype=wp.vec3f, device=DEVICE)


@pytest.fixture()
def scene(art_data, env_origins):
    return MockScene({"robot": MockArticulation(art_data)}, env_origins)


@pytest.fixture()
def action_wp():
    rng = np.random.RandomState(99)
    a = wp.array(rng.randn(NUM_ENVS, NUM_ACTIONS).astype(np.float32), device=DEVICE)
    b = wp.array(rng.randn(NUM_ENVS, NUM_ACTIONS).astype(np.float32), device=DEVICE)
    return a, b


@pytest.fixture()
def episode_length_buf():
    torch.manual_seed(55)
    return torch.randint(0, 500, (NUM_ENVS,), dtype=torch.int64, device=DEVICE)


@pytest.fixture()
def warp_env(scene, action_wp, episode_length_buf):
    """Env with warp action manager (for experimental functions)."""

    class _Env:
        pass

    env = _Env()
    env.scene = scene
    env.action_manager = MockActionManagerWarp(action_wp[0], action_wp[1])
    env.num_envs = NUM_ENVS
    env.device = DEVICE
    env.episode_length_buf = episode_length_buf
    env.step_dt = 0.02
    env.max_episode_length_s = 10.0
    # RNG state for events (seeded deterministically)
    env.rng_state_wp = wp.array(np.arange(NUM_ENVS, dtype=np.uint32) + 42, device=DEVICE)
    return env


@pytest.fixture()
def all_joints_cfg():
    return MockSceneEntityCfg("robot", list(range(NUM_JOINTS)), NUM_JOINTS, DEVICE)


# ============================================================================
# Event capture-mutate-replay tests
# ============================================================================


def _defaults_beyond_limits(value: float) -> np.ndarray:
    """Defaults at ``value`` with half the joints pushed past the +/-3.14 soft limits."""
    defaults = np.full((NUM_ENVS, NUM_JOINTS), value, dtype=np.float32)
    defaults[:, 0::4] = 5.0
    defaults[:, 1::4] = -5.0
    return defaults


def _clamped(defaults: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.clip(defaults, -3.14, 3.14), device=DEVICE)


class TestEventCapturedDataMutation:
    """Verify event functions are capture-safe and react to mutated input data."""

    # -- reset_joints_by_offset -------------------------------------------------

    def test_reset_joints_by_offset(self, warp_env, art_data, all_joints_cfg):
        """With zero-width offset, result == defaults clamped to limits.  Mutate defaults -> result tracks."""
        cfg = all_joints_cfg
        mask = wp.array([True] * NUM_ENVS, dtype=wp.bool, device=DEVICE)

        # Warm-up
        warp_evt.reset_joints_by_offset(
            warp_env, mask, position_range=(0.0, 0.0), velocity_range=(0.0, 0.0), asset_cfg=cfg
        )

        # Capture
        with wp.ScopedCapture() as cap:
            warp_evt.reset_joints_by_offset(
                warp_env, mask, position_range=(0.0, 0.0), velocity_range=(0.0, 0.0), asset_cfg=cfg
            )

        # Mutate defaults in-place, some beyond the soft limits
        new_defaults = _defaults_beyond_limits(0.5)
        copy_np_to_wp(art_data.default_joint_pos, new_defaults)

        # Replay
        wp.capture_launch(cap.graph)
        wp.synchronize()

        assert_close(art_data.joint_pos.torch, _clamped(new_defaults))
        vel_limits_t = art_data.soft_joint_vel_limits.torch
        expected_vel = art_data.default_joint_vel.torch.clone().clamp(-vel_limits_t, vel_limits_t)
        assert_close(art_data.joint_vel.torch, expected_vel)

    # -- reset_joints_by_scale --------------------------------------------------

    def test_reset_joints_by_scale(self, warp_env, art_data, all_joints_cfg):
        """With scale=1.0, result == defaults clamped to limits.  Mutate defaults -> result tracks."""
        cfg = all_joints_cfg
        mask = wp.array([True] * NUM_ENVS, dtype=wp.bool, device=DEVICE)

        warp_evt.reset_joints_by_scale(
            warp_env, mask, position_range=(1.0, 1.0), velocity_range=(1.0, 1.0), asset_cfg=cfg
        )
        with wp.ScopedCapture() as cap:
            warp_evt.reset_joints_by_scale(
                warp_env, mask, position_range=(1.0, 1.0), velocity_range=(1.0, 1.0), asset_cfg=cfg
            )

        new_defaults = _defaults_beyond_limits(0.25)
        copy_np_to_wp(art_data.default_joint_pos, new_defaults)

        wp.capture_launch(cap.graph)
        wp.synchronize()

        assert_close(art_data.joint_pos.torch, _clamped(new_defaults))
        vel_limits_t = art_data.soft_joint_vel_limits.torch
        expected_vel = art_data.default_joint_vel.torch.clone().clamp(-vel_limits_t, vel_limits_t)
        assert_close(art_data.joint_vel.torch, expected_vel)

    # -- push_by_setting_velocity -----------------------------------------------

    def test_push_by_setting_velocity(self, warp_env, art_data, all_joints_cfg):
        """With zero-width velocity range, the written root velocity == root_vel_w.  Mutate root_vel_w -> tracks."""
        mask = wp.array([True] * NUM_ENVS, dtype=wp.bool, device=DEVICE)
        zero_range = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

        warp_evt.push_by_setting_velocity(warp_env, mask, velocity_range=zero_range)
        with wp.ScopedCapture() as cap:
            warp_evt.push_by_setting_velocity(warp_env, mask, velocity_range=zero_range)

        # Mutate root_vel_w
        new_vel = np.tile([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], (NUM_ENVS, 1)).astype(np.float32)
        copy_np_to_wp(art_data.root_vel_w, new_vel)

        wp.capture_launch(cap.graph)
        wp.synchronize()

        written = wp.to_torch(warp_env.scene["robot"].last_root_velocity)
        expected = torch.tensor([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], device=DEVICE).expand(NUM_ENVS, -1)
        assert_close(written, expected)

    # -- apply_external_force_torque --------------------------------------------

    def test_apply_external_force_torque(self, warp_env, art_data, all_joints_cfg):
        """A degenerate non-zero range reaches masked envs through the replayed kernel; unmasked envs get zero."""
        mask = wp.array([i < NUM_ENVS // 2 for i in range(NUM_ENVS)], dtype=wp.bool, device=DEVICE)

        warp_evt.apply_external_force_torque(warp_env, mask, force_range=(2.0, 2.0), torque_range=(3.0, 3.0))
        with wp.ScopedCapture() as cap:
            warp_evt.apply_external_force_torque(warp_env, mask, force_range=(2.0, 2.0), torque_range=(3.0, 3.0))
        composer = warp_env.scene["robot"].permanent_wrench_composer
        forces = wp.to_torch(composer.last_forces)
        torques = wp.to_torch(composer.last_torques)
        # clear the buffers so only the replayed kernel can produce the expected values
        forces.fill_(-1.0)
        torques.fill_(-1.0)
        wp.capture_launch(cap.graph)
        wp.synchronize()

        half = NUM_ENVS // 2
        assert_close(forces[:half], torch.full_like(forces[:half], 2.0))
        assert_close(torques[:half], torch.full_like(torques[:half], 3.0))
        assert_close(forces[half:], torch.zeros_like(forces[half:]))
        assert_close(torques[half:], torch.zeros_like(torques[half:]))

    # -- env_mask selectivity ---------------------------------------------------

    def test_reset_joints_mask_selectivity(self, warp_env, art_data, all_joints_cfg):
        """Only masked envs are modified; unmasked envs retain their state."""
        cfg = all_joints_cfg
        # Mask: only first half of envs
        mask_np = np.array([i < NUM_ENVS // 2 for i in range(NUM_ENVS)])
        mask = wp.array(mask_np, dtype=wp.bool, device=DEVICE)

        # Set joint_pos to a known value
        sentinel = np.full((NUM_ENVS, NUM_JOINTS), 999.0, dtype=np.float32)
        copy_np_to_wp(art_data.joint_pos, sentinel)

        # Set defaults to 0
        copy_np_to_wp(art_data.default_joint_pos, np.zeros((NUM_ENVS, NUM_JOINTS), dtype=np.float32))

        warp_evt.reset_joints_by_offset(
            warp_env, mask, position_range=(0.0, 0.0), velocity_range=(0.0, 0.0), asset_cfg=cfg
        )
        wp.synchronize()

        result = art_data.joint_pos.torch
        # Masked envs: reset to 0 (defaults + 0 offset)
        assert_close(result[: NUM_ENVS // 2], torch.zeros(NUM_ENVS // 2, NUM_JOINTS, device=DEVICE))
        # Unmasked envs: still 999.0
        assert_close(result[NUM_ENVS // 2 :], torch.full((NUM_ENVS // 2, NUM_JOINTS), 999.0, device=DEVICE))
