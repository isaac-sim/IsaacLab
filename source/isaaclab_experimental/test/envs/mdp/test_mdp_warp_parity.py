# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parity tests: stable (torch) MDP terms vs experimental (Warp-first) implementations.

Verifies that every newly ported Warp-first MDP function produces **identical** results
to the stable torch-based implementation under three execution modes:

  1. **Stable baseline** — the torch implementation in ``isaaclab.envs.mdp``
  2. **Warp uncaptured** — the Warp kernel launched normally
  3. **Warp captured** — the Warp kernel recorded in a CUDA graph and replayed

Usage::

    python -m pytest test_mdp_warp_parity.py -v
"""

from __future__ import annotations

import numpy as np
import torch

import isaaclab_experimental.envs.mdp.events as warp_evt

# ---------------------------------------------------------------------------
# Experimental (Warp-first) implementations
# ---------------------------------------------------------------------------
import isaaclab_experimental.envs.mdp.observations as warp_obs
import isaaclab_experimental.envs.mdp.rewards as warp_rew
import isaaclab_experimental.envs.mdp.terminations as warp_term
import pytest
import warp as wp

# ---------------------------------------------------------------------------
# Shared utilities (from parity_helpers.py)
# ---------------------------------------------------------------------------
from parity_helpers import (
    DEVICE,
    NUM_ACTIONS,
    NUM_ENVS,
    NUM_JOINTS,
    MockActionManagerTorch,
    MockActionManagerWarp,
    MockArticulation,
    MockArticulationData,
    MockScene,
    assert_close,
    assert_equal,
    copy_np_to_wp,
    mutate_root_state,
    run_warp_obs,
    run_warp_obs_captured,
    run_warp_rew,
    run_warp_rew_captured,
    run_warp_term,
    run_warp_term_captured,
)

# ---------------------------------------------------------------------------
# Stable (torch) implementations
# ---------------------------------------------------------------------------
import isaaclab.envs.mdp.observations as stable_obs
import isaaclab.envs.mdp.rewards as stable_rew
import isaaclab.envs.mdp.terminations as stable_term

# ============================================================================
# File-specific mock objects
# ============================================================================


class MockSceneEntityCfg:
    """Unified cfg that works for both stable (joint_ids) and experimental (joint_mask / joint_ids_wp)."""

    def __init__(self, name: str, joint_ids: list[int], num_joints: int, device: str):
        self.name = name
        self.joint_ids = joint_ids

        # Experimental extras
        mask = [False] * num_joints
        for idx in joint_ids:
            mask[idx] = True
        self.joint_mask = wp.array(mask, dtype=wp.bool, device=device)
        self.joint_ids_wp = wp.array(joint_ids, dtype=wp.int32, device=device)


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
    # Newton stores env_origins as a warp vec3f array (stable root_pos_w calls wp.to_torch on it).
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
    return a, b  # (action, prev_action)


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
def stable_env(scene, action_wp, episode_length_buf):
    """Env with torch action manager (for stable functions)."""

    class _Env:
        pass

    env = _Env()
    env.scene = scene
    env.action_manager = MockActionManagerTorch(action_wp[0], action_wp[1])
    env.num_envs = NUM_ENVS
    env.device = DEVICE
    env.episode_length_buf = episode_length_buf
    env.step_dt = 0.02
    env.max_episode_length_s = 10.0
    return env


@pytest.fixture()
def all_joints_cfg():
    return MockSceneEntityCfg("robot", list(range(NUM_JOINTS)), NUM_JOINTS, DEVICE)


@pytest.fixture()
def subset_cfg():
    return MockSceneEntityCfg("robot", [0, 2, 5, 8], NUM_JOINTS, DEVICE)


# ============================================================================
# Observation parity tests
# ============================================================================


class TestObservationParity:
    """Verify experimental observation Warp kernels match stable torch implementations."""

    # -- Root state observations ------------------------------------------------

    def test_base_pos_z(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_obs.base_pos_z(stable_env, asset_cfg=cfg)
        actual = run_warp_obs(warp_obs.base_pos_z, warp_env, (NUM_ENVS, 1), asset_cfg=cfg)
        actual_cap = run_warp_obs_captured(warp_obs.base_pos_z, warp_env, (NUM_ENVS, 1), asset_cfg=cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_base_lin_vel(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_obs.base_lin_vel(stable_env, asset_cfg=cfg)
        actual = run_warp_obs(warp_obs.base_lin_vel, warp_env, (NUM_ENVS, 3), asset_cfg=cfg)
        actual_cap = run_warp_obs_captured(warp_obs.base_lin_vel, warp_env, (NUM_ENVS, 3), asset_cfg=cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_base_ang_vel(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_obs.base_ang_vel(stable_env, asset_cfg=cfg)
        actual = run_warp_obs(warp_obs.base_ang_vel, warp_env, (NUM_ENVS, 3), asset_cfg=cfg)
        actual_cap = run_warp_obs_captured(warp_obs.base_ang_vel, warp_env, (NUM_ENVS, 3), asset_cfg=cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_projected_gravity(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_obs.projected_gravity(stable_env, asset_cfg=cfg)
        actual = run_warp_obs(warp_obs.projected_gravity, warp_env, (NUM_ENVS, 3), asset_cfg=cfg)
        actual_cap = run_warp_obs_captured(warp_obs.projected_gravity, warp_env, (NUM_ENVS, 3), asset_cfg=cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    # -- Joint observations (all joints) ----------------------------------------

    def test_joint_pos_all(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_obs.joint_pos(stable_env, asset_cfg=cfg)
        actual = run_warp_obs(warp_obs.joint_pos, warp_env, (NUM_ENVS, NUM_JOINTS), asset_cfg=cfg)
        actual_cap = run_warp_obs_captured(warp_obs.joint_pos, warp_env, (NUM_ENVS, NUM_JOINTS), asset_cfg=cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_joint_vel_all(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_obs.joint_vel(stable_env, asset_cfg=cfg)
        actual = run_warp_obs(warp_obs.joint_vel, warp_env, (NUM_ENVS, NUM_JOINTS), asset_cfg=cfg)
        actual_cap = run_warp_obs_captured(warp_obs.joint_vel, warp_env, (NUM_ENVS, NUM_JOINTS), asset_cfg=cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    # -- Joint observations (subset) -------------------------------------------

    def test_joint_pos_subset(self, warp_env, stable_env, subset_cfg):
        cfg = subset_cfg
        n_selected = len(cfg.joint_ids)
        expected = stable_obs.joint_pos(stable_env, asset_cfg=cfg)
        actual = run_warp_obs(warp_obs.joint_pos, warp_env, (NUM_ENVS, n_selected), asset_cfg=cfg)
        actual_cap = run_warp_obs_captured(warp_obs.joint_pos, warp_env, (NUM_ENVS, n_selected), asset_cfg=cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_joint_vel_subset(self, warp_env, stable_env, subset_cfg):
        cfg = subset_cfg
        n_selected = len(cfg.joint_ids)
        expected = stable_obs.joint_vel(stable_env, asset_cfg=cfg)
        actual = run_warp_obs(warp_obs.joint_vel, warp_env, (NUM_ENVS, n_selected), asset_cfg=cfg)
        actual_cap = run_warp_obs_captured(warp_obs.joint_vel, warp_env, (NUM_ENVS, n_selected), asset_cfg=cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    # -- Normalized joint position ----------------------------------------------

    def test_joint_pos_limit_normalized(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_obs.joint_pos_limit_normalized(stable_env, asset_cfg=cfg)
        actual = run_warp_obs(warp_obs.joint_pos_limit_normalized, warp_env, (NUM_ENVS, NUM_JOINTS), asset_cfg=cfg)
        actual_cap = run_warp_obs_captured(
            warp_obs.joint_pos_limit_normalized, warp_env, (NUM_ENVS, NUM_JOINTS), asset_cfg=cfg
        )
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    # -- Action observation -----------------------------------------------------

    def test_last_action(self, warp_env, stable_env, action_wp):
        # Stable last_action returns env.action_manager.action (torch tensor)
        expected = stable_obs.last_action(stable_env)
        actual = run_warp_obs(warp_obs.last_action, warp_env, (NUM_ENVS, NUM_ACTIONS))
        actual_cap = run_warp_obs_captured(warp_obs.last_action, warp_env, (NUM_ENVS, NUM_ACTIONS))
        assert_close(actual, expected)
        assert_close(actual_cap, expected)


# ============================================================================
# Reward parity tests
# ============================================================================


class TestRewardParity:
    """Verify experimental reward Warp kernels match stable torch implementations."""

    # -- Root penalties ---------------------------------------------------------

    def test_lin_vel_z_l2(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_rew.lin_vel_z_l2(stable_env, asset_cfg=cfg)
        actual = run_warp_rew(warp_rew.lin_vel_z_l2, warp_env, asset_cfg=cfg)
        actual_cap = run_warp_rew_captured(warp_rew.lin_vel_z_l2, warp_env, asset_cfg=cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_ang_vel_xy_l2(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_rew.ang_vel_xy_l2(stable_env, asset_cfg=cfg)
        actual = run_warp_rew(warp_rew.ang_vel_xy_l2, warp_env, asset_cfg=cfg)
        actual_cap = run_warp_rew_captured(warp_rew.ang_vel_xy_l2, warp_env, asset_cfg=cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_flat_orientation_l2(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_rew.flat_orientation_l2(stable_env, asset_cfg=cfg)
        actual = run_warp_rew(warp_rew.flat_orientation_l2, warp_env, asset_cfg=cfg)
        actual_cap = run_warp_rew_captured(warp_rew.flat_orientation_l2, warp_env, asset_cfg=cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    # -- Joint L2 penalties (masked) --------------------------------------------

    def test_joint_vel_l2(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_rew.joint_vel_l2(stable_env, asset_cfg=cfg)
        actual = run_warp_rew(warp_rew.joint_vel_l2, warp_env, asset_cfg=cfg)
        actual_cap = run_warp_rew_captured(warp_rew.joint_vel_l2, warp_env, asset_cfg=cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_joint_acc_l2(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_rew.joint_acc_l2(stable_env, asset_cfg=cfg)
        actual = run_warp_rew(warp_rew.joint_acc_l2, warp_env, asset_cfg=cfg)
        actual_cap = run_warp_rew_captured(warp_rew.joint_acc_l2, warp_env, asset_cfg=cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_joint_torques_l2(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_rew.joint_torques_l2(stable_env, asset_cfg=cfg)
        actual = run_warp_rew(warp_rew.joint_torques_l2, warp_env, asset_cfg=cfg)
        actual_cap = run_warp_rew_captured(warp_rew.joint_torques_l2, warp_env, asset_cfg=cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    # -- Action penalties -------------------------------------------------------

    def test_action_l2(self, warp_env, stable_env):
        expected = stable_rew.action_l2(stable_env)
        actual = run_warp_rew(warp_rew.action_l2, warp_env)
        actual_cap = run_warp_rew_captured(warp_rew.action_l2, warp_env)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_action_rate_l2(self, warp_env, stable_env):
        expected = stable_rew.action_rate_l2(stable_env)
        actual = run_warp_rew(warp_rew.action_rate_l2, warp_env)
        actual_cap = run_warp_rew_captured(warp_rew.action_rate_l2, warp_env)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    # -- Limit penalties --------------------------------------------------------

    def test_joint_pos_limits(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_rew.joint_pos_limits(stable_env, asset_cfg=cfg)
        actual = run_warp_rew(warp_rew.joint_pos_limits, warp_env, asset_cfg=cfg)
        actual_cap = run_warp_rew_captured(warp_rew.joint_pos_limits, warp_env, asset_cfg=cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    # -- Additional penalties ---------------------------------------------------

    def test_joint_deviation_l1(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_rew.joint_deviation_l1(stable_env, asset_cfg=cfg)
        actual = run_warp_rew(warp_rew.joint_deviation_l1, warp_env, asset_cfg=cfg)
        actual_cap = run_warp_rew_captured(warp_rew.joint_deviation_l1, warp_env, asset_cfg=cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)


# ============================================================================
# Termination parity tests
# ============================================================================


class TestTerminationParity:
    """Verify experimental termination Warp kernels match stable torch implementations."""

    def test_root_height_below_minimum(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        min_h = 0.5
        expected = stable_term.root_height_below_minimum(stable_env, minimum_height=min_h, asset_cfg=cfg)
        actual = run_warp_term(warp_term.root_height_below_minimum, warp_env, minimum_height=min_h, asset_cfg=cfg)
        actual_cap = run_warp_term_captured(
            warp_term.root_height_below_minimum, warp_env, minimum_height=min_h, asset_cfg=cfg
        )
        assert_equal(actual, expected)
        assert_equal(actual_cap, expected)


# ============================================================================
# Capture-then-mutate-then-replay tests
#
# Verify that a captured CUDA graph produces correct results when the
# underlying buffer *data* changes between capture and replay (simulating
# a new simulation step).
# ============================================================================


def _mutate_art_data(art_data: MockArticulationData, warp_env, rng_seed: int = 200):
    """Mutate every data array in-place so captured graphs see fresh values."""
    rng = np.random.RandomState(rng_seed)

    copy_np_to_wp(art_data.joint_pos, rng.randn(NUM_ENVS, NUM_JOINTS).astype(np.float32) * 1.5)
    copy_np_to_wp(art_data.joint_vel, rng.randn(NUM_ENVS, NUM_JOINTS).astype(np.float32) * 3.0)
    copy_np_to_wp(art_data.joint_acc, rng.randn(NUM_ENVS, NUM_JOINTS).astype(np.float32) * 0.8)
    copy_np_to_wp(art_data.default_joint_pos, rng.randn(NUM_ENVS, NUM_JOINTS).astype(np.float32) * 0.02)
    copy_np_to_wp(art_data.applied_torque, rng.randn(NUM_ENVS, NUM_JOINTS).astype(np.float32) * 12.0)
    copy_np_to_wp(art_data.computed_torque, rng.randn(NUM_ENVS, NUM_JOINTS).astype(np.float32) * 12.0)

    # Root state + Tier 1 compounds + derived body-frame (including projected_gravity_b)
    mutate_root_state(rng, art_data)

    # Actions (in-place via warp copy — torch views auto-update)
    copy_np_to_wp(warp_env.action_manager._action, rng.randn(NUM_ENVS, NUM_ACTIONS).astype(np.float32))
    copy_np_to_wp(warp_env.action_manager._prev_action, rng.randn(NUM_ENVS, NUM_ACTIONS).astype(np.float32))

    # Episode length (in-place torch update — warp zero-copy view auto-updates)
    warp_env.episode_length_buf[:] = torch.randint(0, 500, (NUM_ENVS,), dtype=torch.int64, device=DEVICE)

    wp.synchronize()


class TestCapturedDataMutation:
    """Capture a graph, mutate buffer data in-place, replay — results must match stable on the *new* data.

    This verifies every migrated MDP function is truly capture-safe: the CUDA graph
    reads from the same buffer pointers but picks up whatever data is there at replay time.
    """

    # -- helpers ---------------------------------------------------------------

    def _capture_mutate_check_obs(self, warp_fn, stable_fn, warp_env, stable_env, art_data, shape, **kwargs):
        out = wp.zeros(shape, dtype=wp.float32, device=DEVICE)
        warp_fn(warp_env, out, **kwargs)  # warm-up
        with wp.ScopedCapture() as cap:
            warp_fn(warp_env, out, **kwargs)
        _mutate_art_data(art_data, warp_env)
        wp.capture_launch(cap.graph)
        assert_close(wp.to_torch(out).clone(), stable_fn(stable_env, **kwargs))

    def _capture_mutate_check_rew(self, warp_fn, stable_fn, warp_env, stable_env, art_data, **kwargs):
        out = wp.zeros((NUM_ENVS,), dtype=wp.float32, device=DEVICE)
        warp_fn(warp_env, out, **kwargs)
        with wp.ScopedCapture() as cap:
            warp_fn(warp_env, out, **kwargs)
        _mutate_art_data(art_data, warp_env)
        wp.capture_launch(cap.graph)
        assert_close(wp.to_torch(out).clone(), stable_fn(stable_env, **kwargs))

    def _capture_mutate_check_term(self, warp_fn, stable_fn, warp_env, stable_env, art_data, **kwargs):
        out = wp.zeros((NUM_ENVS,), dtype=wp.bool, device=DEVICE)
        warp_fn(warp_env, out, **kwargs)
        with wp.ScopedCapture() as cap:
            warp_fn(warp_env, out, **kwargs)
        _mutate_art_data(art_data, warp_env)
        wp.capture_launch(cap.graph)
        assert_equal(wp.to_torch(out).clone(), stable_fn(stable_env, **kwargs))

    # -- observations -----------------------------------------------------------

    def test_base_pos_z(self, warp_env, stable_env, art_data, all_joints_cfg):
        self._capture_mutate_check_obs(
            warp_obs.base_pos_z,
            stable_obs.base_pos_z,
            warp_env,
            stable_env,
            art_data,
            (NUM_ENVS, 1),
            asset_cfg=all_joints_cfg,
        )

    def test_base_lin_vel(self, warp_env, stable_env, art_data, all_joints_cfg):
        self._capture_mutate_check_obs(
            warp_obs.base_lin_vel,
            stable_obs.base_lin_vel,
            warp_env,
            stable_env,
            art_data,
            (NUM_ENVS, 3),
            asset_cfg=all_joints_cfg,
        )

    def test_base_ang_vel(self, warp_env, stable_env, art_data, all_joints_cfg):
        self._capture_mutate_check_obs(
            warp_obs.base_ang_vel,
            stable_obs.base_ang_vel,
            warp_env,
            stable_env,
            art_data,
            (NUM_ENVS, 3),
            asset_cfg=all_joints_cfg,
        )

    def test_projected_gravity(self, warp_env, stable_env, art_data, all_joints_cfg):
        self._capture_mutate_check_obs(
            warp_obs.projected_gravity,
            stable_obs.projected_gravity,
            warp_env,
            stable_env,
            art_data,
            (NUM_ENVS, 3),
            asset_cfg=all_joints_cfg,
        )

    def test_joint_pos(self, warp_env, stable_env, art_data, all_joints_cfg):
        self._capture_mutate_check_obs(
            warp_obs.joint_pos,
            stable_obs.joint_pos,
            warp_env,
            stable_env,
            art_data,
            (NUM_ENVS, NUM_JOINTS),
            asset_cfg=all_joints_cfg,
        )

    def test_joint_vel(self, warp_env, stable_env, art_data, all_joints_cfg):
        self._capture_mutate_check_obs(
            warp_obs.joint_vel,
            stable_obs.joint_vel,
            warp_env,
            stable_env,
            art_data,
            (NUM_ENVS, NUM_JOINTS),
            asset_cfg=all_joints_cfg,
        )

    def test_joint_pos_limit_normalized(self, warp_env, stable_env, art_data, all_joints_cfg):
        self._capture_mutate_check_obs(
            warp_obs.joint_pos_limit_normalized,
            stable_obs.joint_pos_limit_normalized,
            warp_env,
            stable_env,
            art_data,
            (NUM_ENVS, NUM_JOINTS),
            asset_cfg=all_joints_cfg,
        )

    def test_last_action(self, warp_env, stable_env, art_data):
        self._capture_mutate_check_obs(
            warp_obs.last_action,
            stable_obs.last_action,
            warp_env,
            stable_env,
            art_data,
            (NUM_ENVS, NUM_ACTIONS),
        )

    # -- rewards ----------------------------------------------------------------

    def test_lin_vel_z_l2(self, warp_env, stable_env, art_data, all_joints_cfg):
        self._capture_mutate_check_rew(
            warp_rew.lin_vel_z_l2,
            stable_rew.lin_vel_z_l2,
            warp_env,
            stable_env,
            art_data,
            asset_cfg=all_joints_cfg,
        )

    def test_ang_vel_xy_l2(self, warp_env, stable_env, art_data, all_joints_cfg):
        self._capture_mutate_check_rew(
            warp_rew.ang_vel_xy_l2,
            stable_rew.ang_vel_xy_l2,
            warp_env,
            stable_env,
            art_data,
            asset_cfg=all_joints_cfg,
        )

    def test_flat_orientation_l2(self, warp_env, stable_env, art_data, all_joints_cfg):
        self._capture_mutate_check_rew(
            warp_rew.flat_orientation_l2,
            stable_rew.flat_orientation_l2,
            warp_env,
            stable_env,
            art_data,
            asset_cfg=all_joints_cfg,
        )

    def test_joint_vel_l2(self, warp_env, stable_env, art_data, all_joints_cfg):
        self._capture_mutate_check_rew(
            warp_rew.joint_vel_l2,
            stable_rew.joint_vel_l2,
            warp_env,
            stable_env,
            art_data,
            asset_cfg=all_joints_cfg,
        )

    def test_joint_acc_l2(self, warp_env, stable_env, art_data, all_joints_cfg):
        self._capture_mutate_check_rew(
            warp_rew.joint_acc_l2,
            stable_rew.joint_acc_l2,
            warp_env,
            stable_env,
            art_data,
            asset_cfg=all_joints_cfg,
        )

    def test_joint_torques_l2(self, warp_env, stable_env, art_data, all_joints_cfg):
        self._capture_mutate_check_rew(
            warp_rew.joint_torques_l2,
            stable_rew.joint_torques_l2,
            warp_env,
            stable_env,
            art_data,
            asset_cfg=all_joints_cfg,
        )

    def test_action_l2(self, warp_env, stable_env, art_data):
        self._capture_mutate_check_rew(
            warp_rew.action_l2,
            stable_rew.action_l2,
            warp_env,
            stable_env,
            art_data,
        )

    def test_action_rate_l2(self, warp_env, stable_env, art_data):
        self._capture_mutate_check_rew(
            warp_rew.action_rate_l2,
            stable_rew.action_rate_l2,
            warp_env,
            stable_env,
            art_data,
        )

    def test_joint_pos_limits(self, warp_env, stable_env, art_data, all_joints_cfg):
        self._capture_mutate_check_rew(
            warp_rew.joint_pos_limits,
            stable_rew.joint_pos_limits,
            warp_env,
            stable_env,
            art_data,
            asset_cfg=all_joints_cfg,
        )

    def test_joint_deviation_l1(self, warp_env, stable_env, art_data, all_joints_cfg):
        self._capture_mutate_check_rew(
            warp_rew.joint_deviation_l1,
            stable_rew.joint_deviation_l1,
            warp_env,
            stable_env,
            art_data,
            asset_cfg=all_joints_cfg,
        )

    # -- terminations -----------------------------------------------------------

    def test_root_height_below_minimum(self, warp_env, stable_env, art_data, all_joints_cfg):
        self._capture_mutate_check_term(
            warp_term.root_height_below_minimum,
            stable_term.root_height_below_minimum,
            warp_env,
            stable_env,
            art_data,
            minimum_height=0.5,
            asset_cfg=all_joints_cfg,
        )


# ============================================================================
# Event tests
#
# Events use warp RNG (wp.randf) so exact parity with stable (torch RNG) is
# not possible.  Instead we test:
#   1. Uncaptured run produces structurally correct output
#   2. Captured replay does not crash
#   3. Capture-then-mutate-then-replay: the graph picks up new input data
#      (tested with zero-width ranges to eliminate RNG dependency)
# ============================================================================


class TestEventCapturedDataMutation:
    """Verify event functions are capture-safe and react to mutated input data."""

    # -- reset_joints_by_offset -------------------------------------------------

    def test_reset_joints_by_offset(self, warp_env, art_data, all_joints_cfg):
        """With zero-width offset, result == defaults.  Mutate defaults → result tracks."""
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

        # Mutate defaults in-place
        new_defaults = np.full((NUM_ENVS, NUM_JOINTS), 0.5, dtype=np.float32)
        copy_np_to_wp(art_data.default_joint_pos, new_defaults)

        # Replay
        wp.capture_launch(cap.graph)
        wp.synchronize()

        # With zero offset, joint_pos should equal new defaults (clamped to limits [-3.14, 3.14])
        result = wp.to_torch(art_data.joint_pos)
        expected = torch.full((NUM_ENVS, NUM_JOINTS), 0.5, device=DEVICE)
        assert_close(result, expected)

    # -- reset_joints_by_scale --------------------------------------------------

    def test_reset_joints_by_scale(self, warp_env, art_data, all_joints_cfg):
        """With scale=1.0, result == defaults.  Mutate defaults → result tracks."""
        cfg = all_joints_cfg
        mask = wp.array([True] * NUM_ENVS, dtype=wp.bool, device=DEVICE)

        warp_evt.reset_joints_by_scale(
            warp_env, mask, position_range=(1.0, 1.0), velocity_range=(1.0, 1.0), asset_cfg=cfg
        )
        with wp.ScopedCapture() as cap:
            warp_evt.reset_joints_by_scale(
                warp_env, mask, position_range=(1.0, 1.0), velocity_range=(1.0, 1.0), asset_cfg=cfg
            )

        new_defaults = np.full((NUM_ENVS, NUM_JOINTS), 0.25, dtype=np.float32)
        copy_np_to_wp(art_data.default_joint_pos, new_defaults)

        wp.capture_launch(cap.graph)
        wp.synchronize()

        result = wp.to_torch(art_data.joint_pos)
        expected = torch.full((NUM_ENVS, NUM_JOINTS), 0.25, device=DEVICE)
        assert_close(result, expected)

    # -- push_by_setting_velocity -----------------------------------------------

    def test_push_by_setting_velocity(self, warp_env, art_data, all_joints_cfg):
        """With zero-width velocity range, scratch == root_vel_w.  Mutate root_vel_w → scratch tracks."""
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

        scratch = wp.to_torch(warp_evt.push_by_setting_velocity._scratch_vel)
        expected = torch.tensor([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], device=DEVICE).expand(NUM_ENVS, -1)
        assert_close(scratch, expected)

    # -- apply_external_force_torque --------------------------------------------

    def test_apply_external_force_torque(self, warp_env, art_data, all_joints_cfg):
        """With zero-width ranges, forces/torques are zero.  Non-zero ranges produce non-zero output."""
        mask = wp.array([True] * NUM_ENVS, dtype=wp.bool, device=DEVICE)

        # Zero-range: forces and torques should be zero
        warp_evt.apply_external_force_torque(warp_env, mask, force_range=(0.0, 0.0), torque_range=(0.0, 0.0))
        with wp.ScopedCapture() as cap:
            warp_evt.apply_external_force_torque(warp_env, mask, force_range=(0.0, 0.0), torque_range=(0.0, 0.0))
        wp.capture_launch(cap.graph)
        wp.synchronize()

        forces = wp.to_torch(warp_evt.apply_external_force_torque._scratch_forces)
        torques = wp.to_torch(warp_evt.apply_external_force_torque._scratch_torques)
        assert_close(forces, torch.zeros_like(forces))
        assert_close(torques, torch.zeros_like(torques))

    # -- reset_root_state_uniform -----------------------------------------------

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

        result = wp.to_torch(art_data.joint_pos)
        # Masked envs: reset to 0 (defaults + 0 offset)
        assert_close(result[: NUM_ENVS // 2], torch.zeros(NUM_ENVS // 2, NUM_JOINTS, device=DEVICE))
        # Unmasked envs: still 999.0
        assert_close(result[NUM_ENVS // 2 :], torch.full((NUM_ENVS // 2, NUM_JOINTS), 999.0, device=DEVICE))
