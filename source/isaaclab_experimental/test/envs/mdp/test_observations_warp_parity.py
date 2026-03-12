# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parity tests for warp-first observation MDP terms."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import warp as wp

# Skip entire module if no CUDA device available
wp.init()
pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA device required")

import isaaclab_experimental.envs.mdp.observations as warp_obs
from parity_helpers import (
    CMD_DIM,
    DEVICE,
    NUM_ACTIONS,
    NUM_BODIES,
    NUM_ENVS,
    NUM_JOINTS,
    MockActionManagerTorch,
    MockActionManagerWarp,
    MockArticulation,
    MockArticulationData,
    MockCommandManager,
    MockCommandTerm,
    MockContactSensor,
    MockContactSensorData,
    MockScene,
    MockSceneEntityCfg,
    assert_close,
    mutate_art_data,
    run_warp_obs,
    run_warp_obs_captured,
)

import isaaclab.envs.mdp.observations as stable_obs

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def _clear_caches():
    yield
    for fn in [warp_obs.generated_commands]:
        for attr in list(vars(fn)):
            if attr.startswith("_"):
                delattr(fn, attr)


@pytest.fixture()
def art_data():
    return MockArticulationData(NUM_ENVS, NUM_JOINTS, DEVICE)


@pytest.fixture()
def art_data_bodies():
    return MockArticulationData(num_bodies=NUM_BODIES)


@pytest.fixture()
def env_origins():
    rng = np.random.RandomState(77)
    origins_np = rng.randn(NUM_ENVS, 3).astype(np.float32)
    return wp.array(origins_np, dtype=wp.vec3f, device=DEVICE)


@pytest.fixture()
def contact_data():
    return MockContactSensorData()


@pytest.fixture()
def cmd_tensor():
    rng = np.random.RandomState(99)
    return torch.tensor(rng.randn(NUM_ENVS, CMD_DIM).astype(np.float32), device=DEVICE)


@pytest.fixture()
def cmd_term():
    return MockCommandTerm()


@pytest.fixture()
def scene(art_data, env_origins):
    return MockScene({"robot": MockArticulation(art_data)}, env_origins)


@pytest.fixture()
def scene_bodies(art_data_bodies, env_origins, contact_data):
    art = MockArticulation(art_data_bodies, num_bodies=NUM_BODIES)
    sensor = MockContactSensor(contact_data)
    return MockScene({"robot": art}, env_origins, sensors={"contact_sensor": sensor})


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
def warp_env_bodies(scene_bodies, action_wp, episode_length_buf, cmd_tensor, cmd_term):
    """Env with body-level data and command manager (for new-terms observation tests)."""

    class _Env:
        pass

    env = _Env()
    env.scene = scene_bodies
    env.action_manager = MockActionManagerWarp(action_wp[0], action_wp[1])
    env.command_manager = MockCommandManager(cmd_tensor, cmd_term)
    env.num_envs = NUM_ENVS
    env.device = DEVICE
    env.episode_length_buf = episode_length_buf
    env.step_dt = 0.02
    env.max_episode_length = 500
    env.max_episode_length_s = 10.0
    env.rng_state_wp = wp.array(np.arange(NUM_ENVS, dtype=np.uint32) + 42, device=DEVICE)
    return env


@pytest.fixture()
def stable_env_bodies(scene_bodies, action_wp, episode_length_buf, cmd_tensor, cmd_term):
    """Env with body-level data and command manager (for stable new-terms observation tests)."""

    class _Env:
        pass

    env = _Env()
    env.scene = scene_bodies
    env.action_manager = MockActionManagerWarp(action_wp[0], action_wp[1])
    env.command_manager = MockCommandManager(cmd_tensor, cmd_term)
    env.num_envs = NUM_ENVS
    env.device = DEVICE
    env.episode_length_buf = episode_length_buf
    env.step_dt = 0.02
    env.max_episode_length = 500
    env.max_episode_length_s = 10.0
    return env


@pytest.fixture()
def all_joints_cfg():
    return MockSceneEntityCfg("robot", list(range(NUM_JOINTS)), NUM_JOINTS, DEVICE)


@pytest.fixture()
def subset_cfg():
    return MockSceneEntityCfg("robot", [0, 2, 5, 8], NUM_JOINTS, DEVICE)


# ============================================================================
# Observation parity tests (from test_mdp_warp_parity.py)
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
# Observation parity tests (from test_mdp_warp_parity_new_terms.py)
# ============================================================================


class TestObservationParityNewTerms:
    """Verify observation Warp kernels for newly migrated terms match stable torch implementations."""

    def test_generated_commands(self, warp_env_bodies, stable_env_bodies):
        expected = stable_obs.generated_commands(stable_env_bodies, command_name="vel")
        actual = run_warp_obs(warp_obs.generated_commands, warp_env_bodies, (NUM_ENVS, CMD_DIM), command_name="vel")
        actual_cap = run_warp_obs_captured(
            warp_obs.generated_commands, warp_env_bodies, (NUM_ENVS, CMD_DIM), command_name="vel"
        )
        assert_close(actual, expected)
        assert_close(actual_cap, expected)


# ============================================================================
# Capture-then-mutate-then-replay observation tests (from test_mdp_warp_parity.py)
# ============================================================================


def _mutate_art_data(art_data: MockArticulationData, warp_env, rng_seed: int = 200):
    """Mutate every data array in-place so captured graphs see fresh values."""
    mutate_art_data(art_data, warp_env, rng_seed=rng_seed)


class TestCapturedDataMutationObservations:
    """Capture a graph, mutate buffer data in-place, replay -- results must match stable on the *new* data.

    This verifies observation MDP functions are truly capture-safe.
    """

    def _capture_mutate_check_obs(self, warp_fn, stable_fn, warp_env, stable_env, art_data, shape, **kwargs):
        out = wp.zeros(shape, dtype=wp.float32, device=DEVICE)
        warp_fn(warp_env, out, **kwargs)  # warm-up
        with wp.ScopedCapture() as cap:
            warp_fn(warp_env, out, **kwargs)
        _mutate_art_data(art_data, warp_env)
        wp.capture_launch(cap.graph)
        assert_close(wp.to_torch(out).clone(), stable_fn(stable_env, **kwargs))

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


# ============================================================================
# Capture-mutate-replay observation tests (from test_mdp_warp_parity_new_terms.py)
# ============================================================================


class TestCapturedDataMutationObservationsNewTerms:
    """Capture graph, mutate buffer data, replay -- verify new-terms observation results match stable."""

    def test_generated_commands(self, warp_env_bodies, stable_env_bodies, art_data_bodies, cmd_tensor):
        """Mutate command tensor, replay captured graph, verify new commands are read."""
        out = wp.zeros((NUM_ENVS, CMD_DIM), dtype=wp.float32, device=DEVICE)
        warp_obs.generated_commands(warp_env_bodies, out, command_name="vel")
        with wp.ScopedCapture() as cap:
            warp_obs.generated_commands(warp_env_bodies, out, command_name="vel")
        # Mutate the command tensor in-place (zero-copy view picks it up)
        cmd_tensor[:] = torch.randn_like(cmd_tensor)
        wp.capture_launch(cap.graph)
        expected = stable_obs.generated_commands(stable_env_bodies, command_name="vel")
        assert_close(wp.to_torch(out).clone(), expected)
