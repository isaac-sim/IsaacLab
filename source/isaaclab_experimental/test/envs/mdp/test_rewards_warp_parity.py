# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parity tests for warp-first reward MDP terms."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import warp as wp

# Skip entire module if no CUDA device available
wp.init()
pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA device required")

import isaaclab_experimental.envs.mdp.rewards as warp_rew
from parity_helpers import (
    BODY_IDS,
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
    MockBodyCfg,
    MockCommandManager,
    MockCommandTerm,
    MockContactSensor,
    MockContactSensorData,
    MockScene,
    MockSceneEntityCfg,
    MockSensorCfg,
    MockTerminationManager,
    assert_close,
    mutate_art_data,
    mutate_body_data,
    run_warp_rew,
    run_warp_rew_captured,
)

import isaaclab.envs.mdp.rewards as stable_rew

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def _clear_caches():
    yield
    for fn in [warp_rew.track_lin_vel_xy_exp, warp_rew.track_ang_vel_z_exp, warp_rew.undesired_contacts]:
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
    return a, b


@pytest.fixture()
def episode_length_buf():
    torch.manual_seed(55)
    return torch.randint(0, 500, (NUM_ENVS,), dtype=torch.int64, device=DEVICE)


@pytest.fixture()
def term_mgr():
    return MockTerminationManager()


@pytest.fixture()
def warp_env(scene, action_wp, episode_length_buf, term_mgr):
    """Env with warp action manager (for experimental functions)."""

    class _Env:
        pass

    env = _Env()
    env.scene = scene
    env.action_manager = MockActionManagerWarp(action_wp[0], action_wp[1])
    env.termination_manager = term_mgr
    env.num_envs = NUM_ENVS
    env.device = DEVICE
    env.episode_length_buf = episode_length_buf
    env.step_dt = 0.02
    env.max_episode_length_s = 10.0
    env.rng_state_wp = wp.array(np.arange(NUM_ENVS, dtype=np.uint32) + 42, device=DEVICE)
    return env


@pytest.fixture()
def stable_env(scene, action_wp, episode_length_buf, term_mgr):
    """Env with torch action manager (for stable functions)."""

    class _Env:
        pass

    env = _Env()
    env.scene = scene
    env.action_manager = MockActionManagerTorch(action_wp[0], action_wp[1])
    env.termination_manager = term_mgr
    env.num_envs = NUM_ENVS
    env.device = DEVICE
    env.episode_length_buf = episode_length_buf
    env.step_dt = 0.02
    env.max_episode_length_s = 10.0
    return env


@pytest.fixture()
def warp_env_bodies(scene_bodies, action_wp, episode_length_buf, cmd_tensor, cmd_term):
    """Env with body-level data and command manager (for new-terms reward tests)."""

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
    """Env with body-level data and command manager (for stable new-terms reward tests)."""

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
def body_cfg():
    return MockBodyCfg("robot", BODY_IDS)


@pytest.fixture()
def sensor_cfg():
    return MockSensorCfg("contact_sensor", BODY_IDS)


# ============================================================================
# Reward parity tests (from test_mdp_warp_parity.py)
# ============================================================================


class TestRewardParity:
    """Verify experimental reward Warp kernels match stable torch implementations."""

    # -- General rewards --------------------------------------------------------

    def test_is_alive(self, warp_env, stable_env, term_mgr):
        # Set some envs as terminated so the reward is non-trivial
        term_mgr.terminated[::2] = True
        expected = stable_rew.is_alive(stable_env)
        actual = run_warp_rew(warp_rew.is_alive, warp_env)
        actual_cap = run_warp_rew_captured(warp_rew.is_alive, warp_env)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_is_terminated(self, warp_env, stable_env, term_mgr):
        term_mgr.terminated[::3] = True
        expected = stable_rew.is_terminated(stable_env)
        actual = run_warp_rew(warp_rew.is_terminated, warp_env)
        actual_cap = run_warp_rew_captured(warp_rew.is_terminated, warp_env)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

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

    # -- Joint L1 penalties (masked) --------------------------------------------

    def test_joint_vel_l1(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        expected = stable_rew.joint_vel_l1(stable_env, asset_cfg=cfg)
        actual = run_warp_rew(warp_rew.joint_vel_l1, warp_env, asset_cfg=cfg)
        actual_cap = run_warp_rew_captured(warp_rew.joint_vel_l1, warp_env, asset_cfg=cfg)
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
# New reward parity tests (from test_mdp_warp_parity_new_terms.py)
# ============================================================================


class TestNewRewardParity:
    """Verify newly migrated reward Warp kernels match stable torch implementations."""

    def test_track_lin_vel_xy_exp(self, warp_env_bodies, stable_env_bodies, body_cfg):
        cfg = MockBodyCfg("robot")
        cfg.joint_ids = list(range(NUM_JOINTS))  # needed for stable
        std = 0.25
        expected = stable_rew.track_lin_vel_xy_exp(stable_env_bodies, std=std, command_name="vel", asset_cfg=cfg)
        actual = run_warp_rew(
            warp_rew.track_lin_vel_xy_exp, warp_env_bodies, std=std, command_name="vel", asset_cfg=cfg
        )
        actual_cap = run_warp_rew_captured(
            warp_rew.track_lin_vel_xy_exp, warp_env_bodies, std=std, command_name="vel", asset_cfg=cfg
        )
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_track_ang_vel_z_exp(self, warp_env_bodies, stable_env_bodies, body_cfg):
        cfg = MockBodyCfg("robot")
        cfg.joint_ids = list(range(NUM_JOINTS))
        std = 0.25
        expected = stable_rew.track_ang_vel_z_exp(stable_env_bodies, std=std, command_name="vel", asset_cfg=cfg)
        actual = run_warp_rew(warp_rew.track_ang_vel_z_exp, warp_env_bodies, std=std, command_name="vel", asset_cfg=cfg)
        actual_cap = run_warp_rew_captured(
            warp_rew.track_ang_vel_z_exp, warp_env_bodies, std=std, command_name="vel", asset_cfg=cfg
        )
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_undesired_contacts(self, warp_env_bodies, stable_env_bodies, sensor_cfg):
        threshold = 1.0
        expected = stable_rew.undesired_contacts(stable_env_bodies, threshold=threshold, sensor_cfg=sensor_cfg)
        actual = run_warp_rew(warp_rew.undesired_contacts, warp_env_bodies, threshold=threshold, sensor_cfg=sensor_cfg)
        actual_cap = run_warp_rew_captured(
            warp_rew.undesired_contacts, warp_env_bodies, threshold=threshold, sensor_cfg=sensor_cfg
        )
        assert_close(actual, expected.float())
        assert_close(actual_cap, expected.float())


# ============================================================================
# Capture-then-mutate-then-replay reward tests (from test_mdp_warp_parity.py)
# ============================================================================


def _mutate_art_data(art_data: MockArticulationData, warp_env, rng_seed: int = 200):
    """Mutate every data array in-place so captured graphs see fresh values."""
    mutate_art_data(art_data, warp_env, rng_seed=rng_seed)


def _mutate_body_data(art_data: MockArticulationData, rng_seed=200):
    """Mutate body-level and root-level data in-place so captured graphs see fresh values."""
    mutate_body_data(art_data, rng_seed=rng_seed)


class TestCapturedDataMutationRewards:
    """Capture a graph, mutate buffer data in-place, replay -- results must match stable on the *new* data.

    This verifies reward MDP functions are truly capture-safe.
    """

    def _capture_mutate_check_rew(self, warp_fn, stable_fn, warp_env, stable_env, art_data, **kwargs):
        out = wp.zeros((NUM_ENVS,), dtype=wp.float32, device=DEVICE)
        warp_fn(warp_env, out, **kwargs)
        with wp.ScopedCapture() as cap:
            warp_fn(warp_env, out, **kwargs)
        _mutate_art_data(art_data, warp_env)
        wp.capture_launch(cap.graph)
        assert_close(wp.to_torch(out).clone(), stable_fn(stable_env, **kwargs))

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


# ============================================================================
# Capture-mutate-replay reward tests for new terms (from test_mdp_warp_parity_new_terms.py)
# ============================================================================


class TestCapturedDataMutationRewardsNewTerms:
    """Capture graph, mutate buffer data, replay -- verify new-terms reward results match stable."""

    def _capture_mutate_check_rew(self, warp_fn, stable_fn, warp_env, stable_env, art_data, **kwargs):
        out = wp.zeros((NUM_ENVS,), dtype=wp.float32, device=DEVICE)
        warp_fn(warp_env, out, **kwargs)
        with wp.ScopedCapture() as cap:
            warp_fn(warp_env, out, **kwargs)
        _mutate_body_data(art_data)
        wp.capture_launch(cap.graph)
        expected = stable_fn(stable_env, **kwargs)
        assert_close(wp.to_torch(out).clone(), expected)

    def test_track_lin_vel_xy_exp(self, warp_env_bodies, stable_env_bodies, art_data_bodies):
        cfg = MockBodyCfg("robot")
        cfg.joint_ids = list(range(NUM_JOINTS))
        self._capture_mutate_check_rew(
            warp_rew.track_lin_vel_xy_exp,
            stable_rew.track_lin_vel_xy_exp,
            warp_env_bodies,
            stable_env_bodies,
            art_data_bodies,
            std=0.25,
            command_name="vel",
            asset_cfg=cfg,
        )

    def test_track_ang_vel_z_exp(self, warp_env_bodies, stable_env_bodies, art_data_bodies):
        cfg = MockBodyCfg("robot")
        cfg.joint_ids = list(range(NUM_JOINTS))
        self._capture_mutate_check_rew(
            warp_rew.track_ang_vel_z_exp,
            stable_rew.track_ang_vel_z_exp,
            warp_env_bodies,
            stable_env_bodies,
            art_data_bodies,
            std=0.25,
            command_name="vel",
            asset_cfg=cfg,
        )
