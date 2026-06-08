# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parity tests for warp-first termination MDP terms."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import warp as wp

# Skip entire module if no CUDA device available
wp.init()
pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA device required")

import isaaclab_experimental.envs.mdp.terminations as warp_term
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
    MockCommandManager,
    MockCommandTerm,
    MockContactSensor,
    MockContactSensorData,
    MockScene,
    MockSceneEntityCfg,
    MockSensorCfg,
    MockTerminationManager,
    assert_equal,
    mutate_art_data,
    mutate_body_data,
    run_warp_term,
    run_warp_term_captured,
)

import isaaclab.envs.mdp.terminations as stable_term

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def _clear_caches():
    yield
    for fn in [warp_term.illegal_contact]:
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
    """Env with body-level data and command manager (for new-terms termination tests)."""

    class _Env:
        pass

    env = _Env()
    env.scene = scene_bodies
    env.action_manager = MockActionManagerWarp(action_wp[0], action_wp[1])
    env.command_manager = MockCommandManager(cmd_tensor, cmd_term)
    env.num_envs = NUM_ENVS
    env.device = DEVICE
    env.episode_length_buf = episode_length_buf
    env._episode_length_buf_wp = wp.from_torch(episode_length_buf)
    env.step_dt = 0.02
    env.max_episode_length = 500
    env.max_episode_length_s = 10.0
    env.rng_state_wp = wp.array(np.arange(NUM_ENVS, dtype=np.uint32) + 42, device=DEVICE)
    return env


@pytest.fixture()
def stable_env_bodies(scene_bodies, action_wp, episode_length_buf, cmd_tensor, cmd_term):
    """Env with body-level data and command manager (for stable new-terms termination tests)."""

    class _Env:
        pass

    env = _Env()
    env.scene = scene_bodies
    env.action_manager = MockActionManagerWarp(action_wp[0], action_wp[1])
    env.command_manager = MockCommandManager(cmd_tensor, cmd_term)
    env.num_envs = NUM_ENVS
    env.device = DEVICE
    env.episode_length_buf = episode_length_buf
    env._episode_length_buf_wp = wp.from_torch(episode_length_buf)
    env.step_dt = 0.02
    env.max_episode_length = 500
    env.max_episode_length_s = 10.0
    # stable termination_manager needed for time_out
    env.termination_manager = MockTerminationManager()
    return env


@pytest.fixture()
def all_joints_cfg():
    return MockSceneEntityCfg("robot", list(range(NUM_JOINTS)), NUM_JOINTS, DEVICE)


@pytest.fixture()
def sensor_cfg():
    return MockSensorCfg("contact_sensor", BODY_IDS)


# ============================================================================
# Termination parity tests (from test_mdp_warp_parity.py)
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

    def test_joint_pos_out_of_manual_limit(self, warp_env, stable_env, all_joints_cfg):
        cfg = all_joints_cfg
        bounds = (-1.0, 1.0)
        expected = stable_term.joint_pos_out_of_manual_limit(stable_env, bounds=bounds, asset_cfg=cfg)
        actual = run_warp_term(warp_term.joint_pos_out_of_manual_limit, warp_env, bounds=bounds, asset_cfg=cfg)
        actual_cap = run_warp_term_captured(
            warp_term.joint_pos_out_of_manual_limit, warp_env, bounds=bounds, asset_cfg=cfg
        )
        assert_equal(actual, expected)
        assert_equal(actual_cap, expected)


# ============================================================================
# Termination parity tests (from test_mdp_warp_parity_new_terms.py)
# ============================================================================


class TestTerminationParityNewTerms:
    """Verify termination Warp kernels for newly migrated terms match stable torch implementations."""

    def test_time_out(self, warp_env_bodies, stable_env_bodies):
        expected = stable_term.time_out(stable_env_bodies)
        actual = run_warp_term(warp_term.time_out, warp_env_bodies)
        actual_cap = run_warp_term_captured(warp_term.time_out, warp_env_bodies)
        assert_equal(actual, expected)
        assert_equal(actual_cap, expected)

    def test_illegal_contact(self, warp_env_bodies, stable_env_bodies, sensor_cfg):
        threshold = 1.0
        expected = stable_term.illegal_contact(stable_env_bodies, threshold=threshold, sensor_cfg=sensor_cfg)
        actual = run_warp_term(warp_term.illegal_contact, warp_env_bodies, threshold=threshold, sensor_cfg=sensor_cfg)
        actual_cap = run_warp_term_captured(
            warp_term.illegal_contact, warp_env_bodies, threshold=threshold, sensor_cfg=sensor_cfg
        )
        assert_equal(actual, expected)
        assert_equal(actual_cap, expected)


# ============================================================================
# Capture-then-mutate-then-replay termination tests (from test_mdp_warp_parity.py)
# ============================================================================


def _mutate_art_data(art_data: MockArticulationData, warp_env, rng_seed: int = 200):
    """Mutate every data array in-place so captured graphs see fresh values."""
    mutate_art_data(art_data, warp_env, rng_seed=rng_seed)


def _mutate_body_data(art_data: MockArticulationData, rng_seed=200):
    """Mutate body-level and root-level data in-place so captured graphs see fresh values."""
    mutate_body_data(art_data, rng_seed=rng_seed)


class TestCapturedDataMutationTerminations:
    """Capture a graph, mutate buffer data in-place, replay -- results must match stable on the *new* data.

    This verifies termination MDP functions are truly capture-safe.
    """

    def _capture_mutate_check_term(self, warp_fn, stable_fn, warp_env, stable_env, art_data, **kwargs):
        out = wp.zeros((NUM_ENVS,), dtype=wp.bool, device=DEVICE)
        warp_fn(warp_env, out, **kwargs)
        with wp.ScopedCapture() as cap:
            warp_fn(warp_env, out, **kwargs)
        _mutate_art_data(art_data, warp_env)
        wp.capture_launch(cap.graph)
        assert_equal(wp.to_torch(out).clone(), stable_fn(stable_env, **kwargs))

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

    def test_joint_pos_out_of_manual_limit(self, warp_env, stable_env, art_data, all_joints_cfg):
        # joint_pos_out_of_manual_limit uses a 2D kernel that only writes True
        # (never clears to False), so the output must be zeroed before each call.
        # We include the zeroing inside the captured graph.
        bounds = (-1.0, 1.0)
        cfg = all_joints_cfg
        out = wp.zeros((NUM_ENVS,), dtype=wp.bool, device=DEVICE)
        # warm-up
        out.zero_()
        warp_term.joint_pos_out_of_manual_limit(warp_env, out, bounds=bounds, asset_cfg=cfg)
        # capture (including the zero)
        with wp.ScopedCapture() as cap:
            out.zero_()
            warp_term.joint_pos_out_of_manual_limit(warp_env, out, bounds=bounds, asset_cfg=cfg)
        _mutate_art_data(art_data, warp_env)
        wp.capture_launch(cap.graph)
        expected = stable_term.joint_pos_out_of_manual_limit(stable_env, bounds=bounds, asset_cfg=cfg)
        assert_equal(wp.to_torch(out).clone(), expected)


# ============================================================================
# Capture-mutate-replay termination tests for new terms (from test_mdp_warp_parity_new_terms.py)
# ============================================================================


class TestCapturedDataMutationTerminationsNewTerms:
    """Capture graph, mutate buffer data, replay -- verify new-terms termination results match stable."""

    def test_time_out(self, warp_env_bodies, stable_env_bodies, art_data_bodies):
        out = wp.zeros((NUM_ENVS,), dtype=wp.bool, device=DEVICE)
        warp_term.time_out(warp_env_bodies, out)
        with wp.ScopedCapture() as cap:
            warp_term.time_out(warp_env_bodies, out)
        # Mutate episode length in-place
        warp_env_bodies.episode_length_buf[:] = torch.randint(0, 600, (NUM_ENVS,), dtype=torch.int64, device=DEVICE)
        wp.capture_launch(cap.graph)
        expected = stable_term.time_out(stable_env_bodies)
        assert_equal(wp.to_torch(out).clone(), expected)
