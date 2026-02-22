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
# Stable (torch) implementations
# ---------------------------------------------------------------------------
import isaaclab.envs.mdp.observations as stable_obs
import isaaclab.envs.mdp.rewards as stable_rew
import isaaclab.envs.mdp.terminations as stable_term

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------
NUM_ENVS = 64
NUM_JOINTS = 12
NUM_ACTIONS = 6
DEVICE = "cuda:0"

# Tolerance for float32 comparison (torch vs warp may differ by FMA / instruction order)
ATOL = 1e-5
RTOL = 1e-5


# ============================================================================
# Mock objects
# ============================================================================


class MockArticulationData:
    """Mock articulation data backed by Warp arrays (same storage Newton uses)."""

    def __init__(self, num_envs: int, num_joints: int, device: str, seed: int = 42):
        rng = np.random.RandomState(seed)

        # --- Joint state (float32 2D) ---
        self.joint_pos = wp.array(rng.randn(num_envs, num_joints).astype(np.float32), device=device)
        self.joint_vel = wp.array(rng.randn(num_envs, num_joints).astype(np.float32) * 2.0, device=device)
        self.joint_acc = wp.array(rng.randn(num_envs, num_joints).astype(np.float32) * 0.5, device=device)
        self.default_joint_pos = wp.array(rng.randn(num_envs, num_joints).astype(np.float32) * 0.01, device=device)
        self.default_joint_vel = wp.array(np.zeros((num_envs, num_joints), dtype=np.float32), device=device)
        self.applied_torque = wp.array(rng.randn(num_envs, num_joints).astype(np.float32) * 10.0, device=device)
        self.computed_torque = wp.array(rng.randn(num_envs, num_joints).astype(np.float32) * 10.0, device=device)

        # --- Soft joint position limits (vec2f 2D) ---
        limits_np = np.zeros((num_envs, num_joints, 2), dtype=np.float32)
        limits_np[:, :, 0] = -3.14  # lower
        limits_np[:, :, 1] = 3.14  # upper
        self.soft_joint_pos_limits = wp.array(limits_np, dtype=wp.vec2f, device=device)

        # --- Soft joint velocity limits (float32 2D) ---
        self.soft_joint_vel_limits = wp.array(np.full((num_envs, num_joints), 10.0, dtype=np.float32), device=device)

        # --- Root state ---
        root_pos_np = rng.randn(num_envs, 3).astype(np.float32)
        root_pos_np[:, 2] = np.abs(root_pos_np[:, 2]) + 0.1  # positive heights
        self.root_pos_w = wp.array(root_pos_np, dtype=wp.vec3f, device=device)

        self.root_lin_vel_b = wp.array(rng.randn(num_envs, 3).astype(np.float32), dtype=wp.vec3f, device=device)
        self.root_ang_vel_b = wp.array(rng.randn(num_envs, 3).astype(np.float32), dtype=wp.vec3f, device=device)

        # Gravity projection (unit-ish vectors pointing mostly down)
        gravity_np = np.zeros((num_envs, 3), dtype=np.float32)
        gravity_np[:, 2] = -1.0
        gravity_np += rng.randn(num_envs, 3).astype(np.float32) * 0.1
        gravity_np /= np.linalg.norm(gravity_np, axis=1, keepdims=True)
        self.projected_gravity_b = wp.array(gravity_np, dtype=wp.vec3f, device=device)

        # --- Additional root state for new observations ---
        # Quaternion (random unit quaternions)
        quat_np = rng.randn(num_envs, 4).astype(np.float32)
        quat_np /= np.linalg.norm(quat_np, axis=1, keepdims=True)
        self.root_quat_w = wp.array(quat_np, dtype=wp.quatf, device=device)

        # World-frame velocities
        self.root_lin_vel_w = wp.array(rng.randn(num_envs, 3).astype(np.float32), dtype=wp.vec3f, device=device)
        self.root_ang_vel_w = wp.array(rng.randn(num_envs, 3).astype(np.float32), dtype=wp.vec3f, device=device)

        # --- Event-specific data ---
        # Spatial velocity (6-component: lin + ang)
        self.root_vel_w = wp.array(rng.randn(num_envs, 6).astype(np.float32), dtype=wp.spatial_vectorf, device=device)

        # Default root pose (transformf = position vec3f + quaternion quatf)
        default_pose_np = np.zeros((num_envs, 7), dtype=np.float32)
        default_pose_np[:, 0:3] = rng.randn(num_envs, 3).astype(np.float32) * 0.1  # small position offsets
        default_pose_np[:, 3:7] = [0.0, 0.0, 0.0, 1.0]  # identity quaternion (xyzw)
        self.default_root_pose = wp.array(default_pose_np, dtype=wp.transformf, device=device)

        # Default root velocity (spatial_vectorf)
        self.default_root_vel = wp.array(
            np.zeros((num_envs, 6), dtype=np.float32), dtype=wp.spatial_vectorf, device=device
        )


class MockArticulation:
    def __init__(self, data: MockArticulationData):
        self.data = data
        self.num_bodies = 1
        self.device = DEVICE

    # Stub write APIs for events (no-ops — we verify scratch buffer contents instead)
    def write_root_velocity_to_sim(self, root_velocity, env_ids=None, env_mask=None):
        pass

    def write_root_pose_to_sim(self, root_pose, env_ids=None, env_mask=None):
        pass

    def set_external_force_and_torque(self, forces, torques, body_ids=None, env_ids=None, env_mask=None):
        pass


class MockScene:
    def __init__(self, assets: dict, env_origins: torch.Tensor):
        self._assets = assets
        self.env_origins = env_origins

    def __getitem__(self, name: str):
        return self._assets[name]


class MockActionManagerWarp:
    """Returns warp arrays (for experimental functions)."""

    def __init__(self, action_wp: wp.array, prev_action_wp: wp.array):
        self._action = action_wp
        self._prev_action = prev_action_wp

    @property
    def action(self) -> wp.array:
        return self._action

    @property
    def prev_action(self) -> wp.array:
        return self._prev_action


class MockActionManagerTorch:
    """Returns torch tensors (for stable functions)."""

    def __init__(self, action_wp: wp.array, prev_action_wp: wp.array):
        self._action = wp.to_torch(action_wp)
        self._prev_action = wp.to_torch(prev_action_wp)

    @property
    def action(self) -> torch.Tensor:
        return self._action

    @property
    def prev_action(self) -> torch.Tensor:
        return self._prev_action


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

    Functions like ``current_time_s`` and ``root_pos_w`` cache warp views on
    themselves (``hasattr`` pattern).  Without clearing, a cached view from a
    prior test's fixture would be stale when a new test creates different tensors.
    """
    yield
    for fn in (
        warp_obs.root_pos_w,
        warp_obs.current_time_s,
        warp_obs.remaining_time_s,
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
# Helpers
# ============================================================================


def _run_warp_obs(func, env, shape, device=DEVICE, **kwargs):
    """Run a warp observation function and return the result as a torch tensor."""
    out = wp.zeros(shape, dtype=wp.float32, device=device)
    func(env, out, **kwargs)
    return wp.to_torch(out)


def _run_warp_obs_captured(func, env, shape, device=DEVICE, **kwargs):
    """Run a warp observation function under CUDA graph capture and return the result."""
    out = wp.zeros(shape, dtype=wp.float32, device=device)
    # Warm-up (triggers any first-call lazy init)
    func(env, out, **kwargs)
    # Capture
    with wp.ScopedCapture() as capture:
        func(env, out, **kwargs)
    # Replay
    wp.capture_launch(capture.graph)
    return wp.to_torch(out)


def _run_warp_rew(func, env, device=DEVICE, **kwargs):
    """Run a warp reward function and return the result as a torch tensor."""
    out = wp.zeros((NUM_ENVS,), dtype=wp.float32, device=device)
    func(env, out, **kwargs)
    return wp.to_torch(out)


def _run_warp_rew_captured(func, env, device=DEVICE, **kwargs):
    """Run a warp reward function under CUDA graph capture."""
    out = wp.zeros((NUM_ENVS,), dtype=wp.float32, device=device)
    func(env, out, **kwargs)  # warm-up
    with wp.ScopedCapture() as capture:
        func(env, out, **kwargs)
    wp.capture_launch(capture.graph)
    return wp.to_torch(out)


def _run_warp_term(func, env, device=DEVICE, **kwargs):
    """Run a warp termination function and return the result as a torch tensor."""
    out = wp.zeros((NUM_ENVS,), dtype=wp.bool, device=device)
    func(env, out, **kwargs)
    return wp.to_torch(out)


def _run_warp_term_captured(func, env, device=DEVICE, **kwargs):
    """Run a warp termination function under CUDA graph capture."""
    out = wp.zeros((NUM_ENVS,), dtype=wp.bool, device=device)
    func(env, out, **kwargs)  # warm-up
    with wp.ScopedCapture() as capture:
        func(env, out, **kwargs)
    wp.capture_launch(capture.graph)
    return wp.to_torch(out)


def assert_close(actual: torch.Tensor, expected: torch.Tensor, atol: float = ATOL, rtol: float = RTOL):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def assert_equal(actual: torch.Tensor, expected: torch.Tensor):
    assert torch.equal(actual, expected), f"Mismatch:\n  actual:   {actual}\n  expected: {expected}"


# ============================================================================
# Observation parity tests
# ============================================================================


class TestObservationParity:
    """Verify experimental observation Warp kernels match stable torch implementations."""

    # -- Root state observations ------------------------------------------------

    def test_base_pos_z(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_obs.base_pos_z(stable_env, asset_cfg=cfg)
        actual = _run_warp_obs(warp_obs.base_pos_z, warp_env, (NUM_ENVS, 1), asset_cfg=cfg)
        actual_cap = _run_warp_obs_captured(warp_obs.base_pos_z, warp_env, (NUM_ENVS, 1), asset_cfg=cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_base_lin_vel(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_obs.base_lin_vel(stable_env, asset_cfg=cfg)
        actual = _run_warp_obs(warp_obs.base_lin_vel, warp_env, (NUM_ENVS, 3), asset_cfg=cfg)
        actual_cap = _run_warp_obs_captured(warp_obs.base_lin_vel, warp_env, (NUM_ENVS, 3), asset_cfg=cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_base_ang_vel(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_obs.base_ang_vel(stable_env, asset_cfg=cfg)
        actual = _run_warp_obs(warp_obs.base_ang_vel, warp_env, (NUM_ENVS, 3), asset_cfg=cfg)
        actual_cap = _run_warp_obs_captured(warp_obs.base_ang_vel, warp_env, (NUM_ENVS, 3), asset_cfg=cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_projected_gravity(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_obs.projected_gravity(stable_env, asset_cfg=cfg)
        actual = _run_warp_obs(warp_obs.projected_gravity, warp_env, (NUM_ENVS, 3), asset_cfg=cfg)
        actual_cap = _run_warp_obs_captured(warp_obs.projected_gravity, warp_env, (NUM_ENVS, 3), asset_cfg=cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    # -- Joint observations (all joints) ----------------------------------------

    def test_joint_pos_all(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_obs.joint_pos(stable_env, asset_cfg=cfg)
        actual = _run_warp_obs(warp_obs.joint_pos, warp_env, (NUM_ENVS, NUM_JOINTS), asset_cfg=cfg)
        actual_cap = _run_warp_obs_captured(warp_obs.joint_pos, warp_env, (NUM_ENVS, NUM_JOINTS), asset_cfg=cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_joint_vel_all(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_obs.joint_vel(stable_env, asset_cfg=cfg)
        actual = _run_warp_obs(warp_obs.joint_vel, warp_env, (NUM_ENVS, NUM_JOINTS), asset_cfg=cfg)
        actual_cap = _run_warp_obs_captured(warp_obs.joint_vel, warp_env, (NUM_ENVS, NUM_JOINTS), asset_cfg=cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    # -- Joint observations (subset) -------------------------------------------

    def test_joint_pos_subset(self, warp_env, stable_env, subset_cfg):
        cfg = subset_cfg
        n_selected = len(cfg.joint_ids)
        expected = stable_obs.joint_pos(stable_env, asset_cfg=cfg)
        actual = _run_warp_obs(warp_obs.joint_pos, warp_env, (NUM_ENVS, n_selected), asset_cfg=cfg)
        actual_cap = _run_warp_obs_captured(warp_obs.joint_pos, warp_env, (NUM_ENVS, n_selected), asset_cfg=cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_joint_vel_subset(self, warp_env, stable_env, subset_cfg):
        cfg = subset_cfg
        n_selected = len(cfg.joint_ids)
        expected = stable_obs.joint_vel(stable_env, asset_cfg=cfg)
        actual = _run_warp_obs(warp_obs.joint_vel, warp_env, (NUM_ENVS, n_selected), asset_cfg=cfg)
        actual_cap = _run_warp_obs_captured(warp_obs.joint_vel, warp_env, (NUM_ENVS, n_selected), asset_cfg=cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    # -- Normalized joint position ----------------------------------------------

    def test_joint_pos_limit_normalized(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_obs.joint_pos_limit_normalized(stable_env, asset_cfg=cfg)
        actual = _run_warp_obs(warp_obs.joint_pos_limit_normalized, warp_env, (NUM_ENVS, NUM_JOINTS), asset_cfg=cfg)
        actual_cap = _run_warp_obs_captured(
            warp_obs.joint_pos_limit_normalized, warp_env, (NUM_ENVS, NUM_JOINTS), asset_cfg=cfg
        )
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    # -- Action observation -----------------------------------------------------

    def test_last_action(self, warp_env, stable_env, action_wp):
        # Stable last_action returns env.action_manager.action (torch tensor)
        expected = stable_obs.last_action(stable_env)
        actual = _run_warp_obs(warp_obs.last_action, warp_env, (NUM_ENVS, NUM_ACTIONS))
        actual_cap = _run_warp_obs_captured(warp_obs.last_action, warp_env, (NUM_ENVS, NUM_ACTIONS))
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    # -- Additional root state observations -------------------------------------

    def test_root_pos_w(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_obs.root_pos_w(stable_env, asset_cfg=cfg)
        actual = _run_warp_obs(warp_obs.root_pos_w, warp_env, (NUM_ENVS, 3), asset_cfg=cfg)
        actual_cap = _run_warp_obs_captured(warp_obs.root_pos_w, warp_env, (NUM_ENVS, 3), asset_cfg=cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_root_quat_w(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_obs.root_quat_w(stable_env, asset_cfg=cfg)
        actual = _run_warp_obs(warp_obs.root_quat_w, warp_env, (NUM_ENVS, 4), asset_cfg=cfg)
        actual_cap = _run_warp_obs_captured(warp_obs.root_quat_w, warp_env, (NUM_ENVS, 4), asset_cfg=cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_root_quat_w_unique(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_obs.root_quat_w(stable_env, make_quat_unique=True, asset_cfg=cfg)
        actual = _run_warp_obs(warp_obs.root_quat_w, warp_env, (NUM_ENVS, 4), make_quat_unique=True, asset_cfg=cfg)
        actual_cap = _run_warp_obs_captured(
            warp_obs.root_quat_w, warp_env, (NUM_ENVS, 4), make_quat_unique=True, asset_cfg=cfg
        )
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_root_lin_vel_w(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_obs.root_lin_vel_w(stable_env, asset_cfg=cfg)
        actual = _run_warp_obs(warp_obs.root_lin_vel_w, warp_env, (NUM_ENVS, 3), asset_cfg=cfg)
        actual_cap = _run_warp_obs_captured(warp_obs.root_lin_vel_w, warp_env, (NUM_ENVS, 3), asset_cfg=cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_root_ang_vel_w(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_obs.root_ang_vel_w(stable_env, asset_cfg=cfg)
        actual = _run_warp_obs(warp_obs.root_ang_vel_w, warp_env, (NUM_ENVS, 3), asset_cfg=cfg)
        actual_cap = _run_warp_obs_captured(warp_obs.root_ang_vel_w, warp_env, (NUM_ENVS, 3), asset_cfg=cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_joint_effort(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_obs.joint_effort(stable_env, asset_cfg=cfg)
        actual = _run_warp_obs(warp_obs.joint_effort, warp_env, (NUM_ENVS, NUM_JOINTS), asset_cfg=cfg)
        actual_cap = _run_warp_obs_captured(warp_obs.joint_effort, warp_env, (NUM_ENVS, NUM_JOINTS), asset_cfg=cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    # -- Time observations ------------------------------------------------------

    def test_current_time_s(self, warp_env, stable_env):
        expected = stable_obs.current_time_s(stable_env)
        actual = _run_warp_obs(warp_obs.current_time_s, warp_env, (NUM_ENVS, 1))
        actual_cap = _run_warp_obs_captured(warp_obs.current_time_s, warp_env, (NUM_ENVS, 1))
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_remaining_time_s(self, warp_env, stable_env):
        expected = stable_obs.remaining_time_s(stable_env)
        actual = _run_warp_obs(warp_obs.remaining_time_s, warp_env, (NUM_ENVS, 1))
        actual_cap = _run_warp_obs_captured(warp_obs.remaining_time_s, warp_env, (NUM_ENVS, 1))
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
        actual = _run_warp_rew(warp_rew.lin_vel_z_l2, warp_env, asset_cfg=cfg)
        actual_cap = _run_warp_rew_captured(warp_rew.lin_vel_z_l2, warp_env, asset_cfg=cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_ang_vel_xy_l2(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_rew.ang_vel_xy_l2(stable_env, asset_cfg=cfg)
        actual = _run_warp_rew(warp_rew.ang_vel_xy_l2, warp_env, asset_cfg=cfg)
        actual_cap = _run_warp_rew_captured(warp_rew.ang_vel_xy_l2, warp_env, asset_cfg=cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_flat_orientation_l2(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_rew.flat_orientation_l2(stable_env, asset_cfg=cfg)
        actual = _run_warp_rew(warp_rew.flat_orientation_l2, warp_env, asset_cfg=cfg)
        actual_cap = _run_warp_rew_captured(warp_rew.flat_orientation_l2, warp_env, asset_cfg=cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    # -- Joint L2 penalties (masked) --------------------------------------------

    def test_joint_vel_l2(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_rew.joint_vel_l2(stable_env, asset_cfg=cfg)
        actual = _run_warp_rew(warp_rew.joint_vel_l2, warp_env, asset_cfg=cfg)
        actual_cap = _run_warp_rew_captured(warp_rew.joint_vel_l2, warp_env, asset_cfg=cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_joint_acc_l2(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_rew.joint_acc_l2(stable_env, asset_cfg=cfg)
        actual = _run_warp_rew(warp_rew.joint_acc_l2, warp_env, asset_cfg=cfg)
        actual_cap = _run_warp_rew_captured(warp_rew.joint_acc_l2, warp_env, asset_cfg=cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_joint_torques_l2(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_rew.joint_torques_l2(stable_env, asset_cfg=cfg)
        actual = _run_warp_rew(warp_rew.joint_torques_l2, warp_env, asset_cfg=cfg)
        actual_cap = _run_warp_rew_captured(warp_rew.joint_torques_l2, warp_env, asset_cfg=cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    # -- Action penalties -------------------------------------------------------

    def test_action_l2(self, warp_env, stable_env):
        expected = stable_rew.action_l2(stable_env)
        actual = _run_warp_rew(warp_rew.action_l2, warp_env)
        actual_cap = _run_warp_rew_captured(warp_rew.action_l2, warp_env)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_action_rate_l2(self, warp_env, stable_env):
        expected = stable_rew.action_rate_l2(stable_env)
        actual = _run_warp_rew(warp_rew.action_rate_l2, warp_env)
        actual_cap = _run_warp_rew_captured(warp_rew.action_rate_l2, warp_env)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    # -- Limit penalties --------------------------------------------------------

    def test_joint_pos_limits(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_rew.joint_pos_limits(stable_env, asset_cfg=cfg)
        actual = _run_warp_rew(warp_rew.joint_pos_limits, warp_env, asset_cfg=cfg)
        actual_cap = _run_warp_rew_captured(warp_rew.joint_pos_limits, warp_env, asset_cfg=cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_joint_vel_limits(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_rew.joint_vel_limits(stable_env, soft_ratio=0.9, asset_cfg=cfg)
        actual = _run_warp_rew(warp_rew.joint_vel_limits, warp_env, soft_ratio=0.9, asset_cfg=cfg)
        actual_cap = _run_warp_rew_captured(warp_rew.joint_vel_limits, warp_env, soft_ratio=0.9, asset_cfg=cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_applied_torque_limits(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_rew.applied_torque_limits(stable_env, asset_cfg=cfg)
        actual = _run_warp_rew(warp_rew.applied_torque_limits, warp_env, asset_cfg=cfg)
        actual_cap = _run_warp_rew_captured(warp_rew.applied_torque_limits, warp_env, asset_cfg=cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    # -- Additional penalties ---------------------------------------------------

    def test_joint_deviation_l1(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_rew.joint_deviation_l1(stable_env, asset_cfg=cfg)
        actual = _run_warp_rew(warp_rew.joint_deviation_l1, warp_env, asset_cfg=cfg)
        actual_cap = _run_warp_rew_captured(warp_rew.joint_deviation_l1, warp_env, asset_cfg=cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_base_height_l2(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        target = 0.5
        expected = stable_rew.base_height_l2(stable_env, target_height=target, asset_cfg=cfg)
        actual = _run_warp_rew(warp_rew.base_height_l2, warp_env, target_height=target, asset_cfg=cfg)
        actual_cap = _run_warp_rew_captured(warp_rew.base_height_l2, warp_env, target_height=target, asset_cfg=cfg)
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
        actual = _run_warp_term(warp_term.root_height_below_minimum, warp_env, minimum_height=min_h, asset_cfg=cfg)
        actual_cap = _run_warp_term_captured(
            warp_term.root_height_below_minimum, warp_env, minimum_height=min_h, asset_cfg=cfg
        )
        assert_equal(actual, expected)
        assert_equal(actual_cap, expected)

    def test_bad_orientation(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        limit = 0.5  # ~29 degrees
        expected = stable_term.bad_orientation(stable_env, limit_angle=limit, asset_cfg=cfg)
        actual = _run_warp_term(warp_term.bad_orientation, warp_env, limit_angle=limit, asset_cfg=cfg)
        actual_cap = _run_warp_term_captured(warp_term.bad_orientation, warp_env, limit_angle=limit, asset_cfg=cfg)
        assert_equal(actual, expected)
        assert_equal(actual_cap, expected)

    def test_joint_pos_out_of_limit(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_term.joint_pos_out_of_limit(stable_env, asset_cfg=cfg)
        actual = _run_warp_term(warp_term.joint_pos_out_of_limit, warp_env, asset_cfg=cfg)
        actual_cap = _run_warp_term_captured(warp_term.joint_pos_out_of_limit, warp_env, asset_cfg=cfg)
        assert_equal(actual, expected)
        assert_equal(actual_cap, expected)

    def test_joint_vel_out_of_limit(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_term.joint_vel_out_of_limit(stable_env, asset_cfg=cfg)
        actual = _run_warp_term(warp_term.joint_vel_out_of_limit, warp_env, asset_cfg=cfg)
        actual_cap = _run_warp_term_captured(warp_term.joint_vel_out_of_limit, warp_env, asset_cfg=cfg)
        assert_equal(actual, expected)
        assert_equal(actual_cap, expected)

    # -- Additional joint terminations ------------------------------------------

    def test_joint_vel_out_of_manual_limit(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        max_vel = 5.0
        expected = stable_term.joint_vel_out_of_manual_limit(stable_env, max_velocity=max_vel, asset_cfg=cfg)
        actual = _run_warp_term(warp_term.joint_vel_out_of_manual_limit, warp_env, max_velocity=max_vel, asset_cfg=cfg)
        actual_cap = _run_warp_term_captured(
            warp_term.joint_vel_out_of_manual_limit, warp_env, max_velocity=max_vel, asset_cfg=cfg
        )
        assert_equal(actual, expected)
        assert_equal(actual_cap, expected)

    def test_joint_effort_out_of_limit(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_term.joint_effort_out_of_limit(stable_env, asset_cfg=cfg)
        actual = _run_warp_term(warp_term.joint_effort_out_of_limit, warp_env, asset_cfg=cfg)
        actual_cap = _run_warp_term_captured(warp_term.joint_effort_out_of_limit, warp_env, asset_cfg=cfg)
        assert_equal(actual, expected)
        assert_equal(actual_cap, expected)


# ============================================================================
# Capture-then-mutate-then-replay tests
#
# Verify that a captured CUDA graph produces correct results when the
# underlying buffer *data* changes between capture and replay (simulating
# a new simulation step).
# ============================================================================


def _copy_np_to_wp(dest: wp.array, src_np: np.ndarray):
    """In-place overwrite of a warp array's contents from numpy (preserves pointer)."""
    tmp = wp.array(src_np, dtype=dest.dtype, device=str(dest.device))
    wp.copy(dest, tmp)


def _mutate_art_data(art_data: MockArticulationData, warp_env, rng_seed: int = 200):
    """Mutate every data array in-place so captured graphs see fresh values."""
    rng = np.random.RandomState(rng_seed)

    _copy_np_to_wp(art_data.joint_pos, rng.randn(NUM_ENVS, NUM_JOINTS).astype(np.float32) * 1.5)
    _copy_np_to_wp(art_data.joint_vel, rng.randn(NUM_ENVS, NUM_JOINTS).astype(np.float32) * 3.0)
    _copy_np_to_wp(art_data.joint_acc, rng.randn(NUM_ENVS, NUM_JOINTS).astype(np.float32) * 0.8)
    _copy_np_to_wp(art_data.default_joint_pos, rng.randn(NUM_ENVS, NUM_JOINTS).astype(np.float32) * 0.02)
    _copy_np_to_wp(art_data.applied_torque, rng.randn(NUM_ENVS, NUM_JOINTS).astype(np.float32) * 12.0)
    _copy_np_to_wp(art_data.computed_torque, rng.randn(NUM_ENVS, NUM_JOINTS).astype(np.float32) * 12.0)

    root_pos_np = rng.randn(NUM_ENVS, 3).astype(np.float32)
    root_pos_np[:, 2] = np.abs(root_pos_np[:, 2]) + 0.05
    _copy_np_to_wp(art_data.root_pos_w, root_pos_np)
    _copy_np_to_wp(art_data.root_lin_vel_b, rng.randn(NUM_ENVS, 3).astype(np.float32))
    _copy_np_to_wp(art_data.root_ang_vel_b, rng.randn(NUM_ENVS, 3).astype(np.float32))
    _copy_np_to_wp(art_data.root_lin_vel_w, rng.randn(NUM_ENVS, 3).astype(np.float32))
    _copy_np_to_wp(art_data.root_ang_vel_w, rng.randn(NUM_ENVS, 3).astype(np.float32))

    gravity_np = np.zeros((NUM_ENVS, 3), dtype=np.float32)
    gravity_np[:, 2] = -1.0
    gravity_np += rng.randn(NUM_ENVS, 3).astype(np.float32) * 0.15
    gravity_np /= np.linalg.norm(gravity_np, axis=1, keepdims=True)
    _copy_np_to_wp(art_data.projected_gravity_b, gravity_np)

    quat_np = rng.randn(NUM_ENVS, 4).astype(np.float32)
    quat_np /= np.linalg.norm(quat_np, axis=1, keepdims=True)
    _copy_np_to_wp(art_data.root_quat_w, quat_np)

    # Actions (in-place via warp copy — torch views auto-update)
    _copy_np_to_wp(warp_env.action_manager._action, rng.randn(NUM_ENVS, NUM_ACTIONS).astype(np.float32))
    _copy_np_to_wp(warp_env.action_manager._prev_action, rng.randn(NUM_ENVS, NUM_ACTIONS).astype(np.float32))

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

    def test_root_pos_w(self, warp_env, stable_env, art_data, all_joints_cfg):
        self._capture_mutate_check_obs(
            warp_obs.root_pos_w,
            stable_obs.root_pos_w,
            warp_env,
            stable_env,
            art_data,
            (NUM_ENVS, 3),
            asset_cfg=all_joints_cfg,
        )

    def test_root_quat_w(self, warp_env, stable_env, art_data, all_joints_cfg):
        self._capture_mutate_check_obs(
            warp_obs.root_quat_w,
            stable_obs.root_quat_w,
            warp_env,
            stable_env,
            art_data,
            (NUM_ENVS, 4),
            asset_cfg=all_joints_cfg,
        )

    def test_root_quat_w_unique(self, warp_env, stable_env, art_data, all_joints_cfg):
        self._capture_mutate_check_obs(
            warp_obs.root_quat_w,
            stable_obs.root_quat_w,
            warp_env,
            stable_env,
            art_data,
            (NUM_ENVS, 4),
            make_quat_unique=True,
            asset_cfg=all_joints_cfg,
        )

    def test_root_lin_vel_w(self, warp_env, stable_env, art_data, all_joints_cfg):
        self._capture_mutate_check_obs(
            warp_obs.root_lin_vel_w,
            stable_obs.root_lin_vel_w,
            warp_env,
            stable_env,
            art_data,
            (NUM_ENVS, 3),
            asset_cfg=all_joints_cfg,
        )

    def test_root_ang_vel_w(self, warp_env, stable_env, art_data, all_joints_cfg):
        self._capture_mutate_check_obs(
            warp_obs.root_ang_vel_w,
            stable_obs.root_ang_vel_w,
            warp_env,
            stable_env,
            art_data,
            (NUM_ENVS, 3),
            asset_cfg=all_joints_cfg,
        )

    def test_joint_effort(self, warp_env, stable_env, art_data, all_joints_cfg):
        self._capture_mutate_check_obs(
            warp_obs.joint_effort,
            stable_obs.joint_effort,
            warp_env,
            stable_env,
            art_data,
            (NUM_ENVS, NUM_JOINTS),
            asset_cfg=all_joints_cfg,
        )

    def test_current_time_s(self, warp_env, stable_env, art_data):
        self._capture_mutate_check_obs(
            warp_obs.current_time_s,
            stable_obs.current_time_s,
            warp_env,
            stable_env,
            art_data,
            (NUM_ENVS, 1),
        )

    def test_remaining_time_s(self, warp_env, stable_env, art_data):
        self._capture_mutate_check_obs(
            warp_obs.remaining_time_s,
            stable_obs.remaining_time_s,
            warp_env,
            stable_env,
            art_data,
            (NUM_ENVS, 1),
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

    def test_joint_vel_limits(self, warp_env, stable_env, art_data, all_joints_cfg):
        self._capture_mutate_check_rew(
            warp_rew.joint_vel_limits,
            stable_rew.joint_vel_limits,
            warp_env,
            stable_env,
            art_data,
            soft_ratio=0.9,
            asset_cfg=all_joints_cfg,
        )

    def test_applied_torque_limits(self, warp_env, stable_env, art_data, all_joints_cfg):
        self._capture_mutate_check_rew(
            warp_rew.applied_torque_limits,
            stable_rew.applied_torque_limits,
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

    def test_base_height_l2(self, warp_env, stable_env, art_data, all_joints_cfg):
        self._capture_mutate_check_rew(
            warp_rew.base_height_l2,
            stable_rew.base_height_l2,
            warp_env,
            stable_env,
            art_data,
            target_height=0.5,
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

    def test_bad_orientation(self, warp_env, stable_env, art_data, all_joints_cfg):
        self._capture_mutate_check_term(
            warp_term.bad_orientation,
            stable_term.bad_orientation,
            warp_env,
            stable_env,
            art_data,
            limit_angle=0.5,
            asset_cfg=all_joints_cfg,
        )

    def test_joint_pos_out_of_limit(self, warp_env, stable_env, art_data, all_joints_cfg):
        self._capture_mutate_check_term(
            warp_term.joint_pos_out_of_limit,
            stable_term.joint_pos_out_of_limit,
            warp_env,
            stable_env,
            art_data,
            asset_cfg=all_joints_cfg,
        )

    def test_joint_vel_out_of_limit(self, warp_env, stable_env, art_data, all_joints_cfg):
        self._capture_mutate_check_term(
            warp_term.joint_vel_out_of_limit,
            stable_term.joint_vel_out_of_limit,
            warp_env,
            stable_env,
            art_data,
            asset_cfg=all_joints_cfg,
        )

    def test_joint_vel_out_of_manual_limit(self, warp_env, stable_env, art_data, all_joints_cfg):
        self._capture_mutate_check_term(
            warp_term.joint_vel_out_of_manual_limit,
            stable_term.joint_vel_out_of_manual_limit,
            warp_env,
            stable_env,
            art_data,
            max_velocity=5.0,
            asset_cfg=all_joints_cfg,
        )

    def test_joint_effort_out_of_limit(self, warp_env, stable_env, art_data, all_joints_cfg):
        self._capture_mutate_check_term(
            warp_term.joint_effort_out_of_limit,
            stable_term.joint_effort_out_of_limit,
            warp_env,
            stable_env,
            art_data,
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
        _copy_np_to_wp(art_data.default_joint_pos, new_defaults)

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
        _copy_np_to_wp(art_data.default_joint_pos, new_defaults)

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
        _copy_np_to_wp(art_data.root_vel_w, new_vel)

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

    def test_reset_root_state_uniform(self, warp_env, art_data, all_joints_cfg, env_origins):
        """With zero-width ranges, pose = default + env_origin, vel = default.  Mutate defaults → tracks."""
        mask = wp.array([True] * NUM_ENVS, dtype=wp.bool, device=DEVICE)
        zero_pose = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        zero_vel = dict(zero_pose)

        warp_evt.reset_root_state_uniform(warp_env, mask, pose_range=zero_pose, velocity_range=zero_vel)
        with wp.ScopedCapture() as cap:
            warp_evt.reset_root_state_uniform(warp_env, mask, pose_range=zero_pose, velocity_range=zero_vel)

        # Mutate default_root_pose: set all positions to (1, 2, 3), identity quat
        new_pose = np.zeros((NUM_ENVS, 7), dtype=np.float32)
        new_pose[:, 0:3] = [1.0, 2.0, 3.0]
        new_pose[:, 3:7] = [0.0, 0.0, 0.0, 1.0]  # identity (xyzw)
        _copy_np_to_wp(art_data.default_root_pose, new_pose)

        wp.capture_launch(cap.graph)
        wp.synchronize()

        scratch_pose = wp.to_torch(warp_evt.reset_root_state_uniform._scratch_pose)
        origins_t = wp.to_torch(env_origins)

        # position = default(1,2,3) + env_origin + 0
        expected_pos = torch.tensor([1.0, 2.0, 3.0], device=DEVICE).unsqueeze(0) + origins_t
        assert_close(scratch_pose[:, :3], expected_pos)

        # quaternion = identity * identity_delta = identity = (0,0,0,1) in xyzw
        expected_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=DEVICE).expand(NUM_ENVS, -1)
        assert_close(scratch_pose[:, 3:7], expected_quat)

    # -- env_mask selectivity ---------------------------------------------------

    def test_reset_joints_mask_selectivity(self, warp_env, art_data, all_joints_cfg):
        """Only masked envs are modified; unmasked envs retain their state."""
        cfg = all_joints_cfg
        # Mask: only first half of envs
        mask_np = np.array([i < NUM_ENVS // 2 for i in range(NUM_ENVS)])
        mask = wp.array(mask_np, dtype=wp.bool, device=DEVICE)

        # Set joint_pos to a known value
        sentinel = np.full((NUM_ENVS, NUM_JOINTS), 999.0, dtype=np.float32)
        _copy_np_to_wp(art_data.joint_pos, sentinel)

        # Set defaults to 0
        _copy_np_to_wp(art_data.default_joint_pos, np.zeros((NUM_ENVS, NUM_JOINTS), dtype=np.float32))

        warp_evt.reset_joints_by_offset(
            warp_env, mask, position_range=(0.0, 0.0), velocity_range=(0.0, 0.0), asset_cfg=cfg
        )
        wp.synchronize()

        result = wp.to_torch(art_data.joint_pos)
        # Masked envs: reset to 0 (defaults + 0 offset)
        assert_close(result[: NUM_ENVS // 2], torch.zeros(NUM_ENVS // 2, NUM_JOINTS, device=DEVICE))
        # Unmasked envs: still 999.0
        assert_close(result[NUM_ENVS // 2 :], torch.full((NUM_ENVS // 2, NUM_JOINTS), 999.0, device=DEVICE))
