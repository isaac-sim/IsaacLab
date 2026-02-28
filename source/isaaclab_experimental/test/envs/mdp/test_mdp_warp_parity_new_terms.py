# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parity tests for newly migrated Warp-first MDP terms.

Tests: body observations, command-dependent rewards, contact sensor rewards/terminations,
and new event functions.

Usage::
    python -m pytest test_mdp_warp_parity_new_terms.py -v
"""

from __future__ import annotations

import numpy as np
import torch

import pytest
import warp as wp

# Skip entire module if no CUDA device available
wp.init()
pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA device required")

import isaaclab_experimental.envs.mdp.events as warp_evt
import isaaclab_experimental.envs.mdp.observations as warp_obs
import isaaclab_experimental.envs.mdp.rewards as warp_rew
import isaaclab_experimental.envs.mdp.terminations as warp_term

# ---------------------------------------------------------------------------
# Shared utilities (from parity_helpers.py)
# ---------------------------------------------------------------------------
from parity_helpers import (
    DEVICE,
    NUM_ACTIONS,
    NUM_ENVS,
    NUM_JOINTS,
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

import isaaclab.envs.mdp.observations as stable_obs
import isaaclab.envs.mdp.rewards as stable_rew
import isaaclab.envs.mdp.terminations as stable_term

# File-specific constants
NUM_BODIES = 4
NUM_HISTORY = 3
CMD_DIM = 3
BODY_IDS = [0, 2]  # subset of bodies to test


# ============================================================================
# File-specific mock infrastructure
# ============================================================================


class MockContactSensorData:
    def __init__(self, device=DEVICE, seed=77):
        rng = np.random.RandomState(seed)
        self.net_forces_w_history = torch.tensor(
            rng.randn(NUM_ENVS, NUM_HISTORY, NUM_BODIES, 3).astype(np.float32), device=device
        )


class MockContactSensor:
    def __init__(self, data: MockContactSensorData):
        self.data = data
        self.num_bodies = NUM_BODIES


class MockCommandTerm:
    def __init__(self, device=DEVICE, seed=88):
        rng = np.random.RandomState(seed)
        self.time_left = torch.tensor(rng.rand(NUM_ENVS).astype(np.float32) * 0.05, device=device)
        self.command_counter = torch.tensor(rng.randint(0, 3, (NUM_ENVS,)), dtype=torch.float32, device=device)


class MockCommandManager:
    def __init__(self, command_tensor: torch.Tensor, cmd_term: MockCommandTerm):
        self._cmd = command_tensor
        self._term = cmd_term

    def get_command(self, name: str) -> torch.Tensor:
        return self._cmd

    def get_term(self, name: str):
        return self._term


class MockBodyCfg:
    """SceneEntityCfg-like object for body-level terms."""

    def __init__(self, name="robot", body_ids=None):
        self.name = name
        self.body_ids = body_ids if body_ids is not None else BODY_IDS


class MockSensorCfg:
    """SceneEntityCfg-like object for contact sensor terms."""

    def __init__(self, name="contact_sensor", body_ids=None):
        self.name = name
        self.body_ids = body_ids if body_ids is not None else BODY_IDS


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def _clear_caches():
    yield
    # Clear function-level caches from all new warp functions
    fns_to_clear = [
        warp_obs.generated_commands,
        warp_rew.track_lin_vel_xy_exp,
        warp_rew.track_ang_vel_z_exp,
        warp_rew.undesired_contacts,
        warp_term.illegal_contact,
        warp_evt.randomize_rigid_body_com,
    ]
    for fn in fns_to_clear:
        for attr in list(vars(fn)):
            if attr.startswith("_"):
                delattr(fn, attr)


@pytest.fixture()
def art_data():
    return MockArticulationData(num_bodies=NUM_BODIES)


@pytest.fixture()
def env_origins():
    origins_np = np.random.RandomState(77).randn(NUM_ENVS, 3).astype(np.float32)
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
def scene(art_data, env_origins, contact_data):
    art = MockArticulation(art_data, num_bodies=NUM_BODIES)
    sensor = MockContactSensor(contact_data)
    return MockScene({"robot": art}, env_origins, sensors={"contact_sensor": sensor})


@pytest.fixture()
def action_wp():
    rng = np.random.RandomState(55)
    a = wp.array(rng.randn(NUM_ENVS, NUM_ACTIONS).astype(np.float32), device=DEVICE)
    b = wp.array(rng.randn(NUM_ENVS, NUM_ACTIONS).astype(np.float32), device=DEVICE)
    return a, b


@pytest.fixture()
def episode_length_buf():
    torch.manual_seed(55)
    return torch.randint(0, 500, (NUM_ENVS,), dtype=torch.int64, device=DEVICE)


@pytest.fixture()
def warp_env(scene, action_wp, episode_length_buf, cmd_tensor, cmd_term):
    class _Env:
        pass

    env = _Env()
    env.scene = scene
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
def stable_env(scene, action_wp, episode_length_buf, cmd_tensor, cmd_term):
    class _Env:
        pass

    env = _Env()
    env.scene = scene
    # stable functions access action_manager.action as torch
    env.action_manager = MockActionManagerWarp(action_wp[0], action_wp[1])
    env.command_manager = MockCommandManager(cmd_tensor, cmd_term)
    env.num_envs = NUM_ENVS
    env.device = DEVICE
    env.episode_length_buf = episode_length_buf
    env.step_dt = 0.02
    env.max_episode_length = 500
    env.max_episode_length_s = 10.0
    # stable termination_manager needed for time_out
    env.termination_manager = type("_TM", (), {"terminated": torch.zeros(NUM_ENVS, dtype=torch.bool, device=DEVICE)})()
    return env


@pytest.fixture()
def body_cfg():
    return MockBodyCfg("robot", BODY_IDS)


@pytest.fixture()
def sensor_cfg():
    return MockSensorCfg("contact_sensor", BODY_IDS)


# ============================================================================
# Helpers
# ============================================================================


# ============================================================================
# Body observation parity tests
# ============================================================================


class TestObservationParity:
    """Verify observation Warp kernels match stable torch implementations."""

    def test_generated_commands(self, warp_env, stable_env):
        expected = stable_obs.generated_commands(stable_env, command_name="vel")
        actual = run_warp_obs(warp_obs.generated_commands, warp_env, (NUM_ENVS, CMD_DIM), command_name="vel")
        actual_cap = run_warp_obs_captured(
            warp_obs.generated_commands, warp_env, (NUM_ENVS, CMD_DIM), command_name="vel"
        )
        assert_close(actual, expected)
        assert_close(actual_cap, expected)


# ============================================================================
# New reward parity tests
# ============================================================================


class TestNewRewardParity:
    """Verify newly migrated reward Warp kernels match stable torch implementations."""

    def test_track_lin_vel_xy_exp(self, warp_env, stable_env, body_cfg):
        cfg = MockBodyCfg("robot")
        cfg.joint_ids = list(range(NUM_JOINTS))  # needed for stable
        std = 0.25
        expected = stable_rew.track_lin_vel_xy_exp(stable_env, std=std, command_name="vel", asset_cfg=cfg)
        actual = run_warp_rew(warp_rew.track_lin_vel_xy_exp, warp_env, std=std, command_name="vel", asset_cfg=cfg)
        actual_cap = run_warp_rew_captured(
            warp_rew.track_lin_vel_xy_exp, warp_env, std=std, command_name="vel", asset_cfg=cfg
        )
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_track_ang_vel_z_exp(self, warp_env, stable_env, body_cfg):
        cfg = MockBodyCfg("robot")
        cfg.joint_ids = list(range(NUM_JOINTS))
        std = 0.25
        expected = stable_rew.track_ang_vel_z_exp(stable_env, std=std, command_name="vel", asset_cfg=cfg)
        actual = run_warp_rew(warp_rew.track_ang_vel_z_exp, warp_env, std=std, command_name="vel", asset_cfg=cfg)
        actual_cap = run_warp_rew_captured(
            warp_rew.track_ang_vel_z_exp, warp_env, std=std, command_name="vel", asset_cfg=cfg
        )
        assert_close(actual, expected)
        assert_close(actual_cap, expected)


# ============================================================================
# Termination parity tests
# ============================================================================


class TestTerminationParity:
    """Verify termination Warp kernels match stable torch implementations."""

    def test_time_out(self, warp_env, stable_env):
        expected = stable_term.time_out(stable_env)
        actual = run_warp_term(warp_term.time_out, warp_env)
        actual_cap = run_warp_term_captured(warp_term.time_out, warp_env)
        assert_equal(actual, expected)
        assert_equal(actual_cap, expected)


# ============================================================================
# Capture-mutate-replay tests for new terms
# ============================================================================


def _mutate_body_data(art_data: MockArticulationData, rng_seed=200):
    """Mutate body-level and root-level data in-place so captured graphs see fresh values."""
    rng = np.random.RandomState(rng_seed)

    # Root state + Tier 1 compounds + derived body-frame velocities
    mutate_root_state(rng, art_data)

    # Body data
    grav_np = rng.randn(NUM_ENVS, NUM_BODIES, 3).astype(np.float32)
    grav_np[:, :, 2] = -1.0
    grav_np /= np.linalg.norm(grav_np, axis=2, keepdims=True)
    copy_np_to_wp(art_data.projected_gravity_b, grav_np)

    copy_np_to_wp(art_data.body_lin_acc_w, rng.randn(NUM_ENVS, NUM_BODIES, 3).astype(np.float32))

    pose_np = np.zeros((NUM_ENVS, NUM_BODIES, 7), dtype=np.float32)
    pose_np[:, :, :3] = rng.randn(NUM_ENVS, NUM_BODIES, 3).astype(np.float32)
    pose_np[:, :, 3:7] = [0.0, 0.0, 0.0, 1.0]
    copy_np_to_wp(art_data.body_pose_w, pose_np)

    wp.synchronize()


class TestCapturedDataMutationNewTerms:
    """Capture graph, mutate buffer data, replay — verify results match stable on new data.

    This validates the dynamic dependency update check (test requirement b).
    """

    def _capture_mutate_check_obs(self, warp_fn, stable_fn, warp_env, stable_env, art_data, shape, **kwargs):
        out = wp.zeros(shape, dtype=wp.float32, device=DEVICE)
        warp_fn(warp_env, out, **kwargs)  # warm-up
        with wp.ScopedCapture() as cap:
            warp_fn(warp_env, out, **kwargs)
        _mutate_body_data(art_data)
        wp.capture_launch(cap.graph)
        expected = stable_fn(stable_env, **kwargs)
        assert_close(wp.to_torch(out).clone(), expected)

    def _capture_mutate_check_rew(self, warp_fn, stable_fn, warp_env, stable_env, art_data, **kwargs):
        out = wp.zeros((NUM_ENVS,), dtype=wp.float32, device=DEVICE)
        warp_fn(warp_env, out, **kwargs)
        with wp.ScopedCapture() as cap:
            warp_fn(warp_env, out, **kwargs)
        _mutate_body_data(art_data)
        wp.capture_launch(cap.graph)
        expected = stable_fn(stable_env, **kwargs)
        assert_close(wp.to_torch(out).clone(), expected)

    def _capture_mutate_check_term(self, warp_fn, stable_fn, warp_env, stable_env, art_data, **kwargs):
        out = wp.zeros((NUM_ENVS,), dtype=wp.bool, device=DEVICE)
        warp_fn(warp_env, out, **kwargs)
        with wp.ScopedCapture() as cap:
            warp_fn(warp_env, out, **kwargs)
        _mutate_body_data(art_data)
        wp.capture_launch(cap.graph)
        expected = stable_fn(stable_env, **kwargs)
        assert_equal(wp.to_torch(out).clone(), expected)

    # -- observations ----------------------------------------------------------

    def test_generated_commands(self, warp_env, stable_env, art_data, cmd_tensor):
        """Mutate command tensor, replay captured graph, verify new commands are read."""
        out = wp.zeros((NUM_ENVS, CMD_DIM), dtype=wp.float32, device=DEVICE)
        warp_obs.generated_commands(warp_env, out, command_name="vel")
        with wp.ScopedCapture() as cap:
            warp_obs.generated_commands(warp_env, out, command_name="vel")
        # Mutate the command tensor in-place (zero-copy view picks it up)
        cmd_tensor[:] = torch.randn_like(cmd_tensor)
        wp.capture_launch(cap.graph)
        expected = stable_obs.generated_commands(stable_env, command_name="vel")
        assert_close(wp.to_torch(out).clone(), expected)

    # -- rewards -----------------------------------------------------------

    def test_track_lin_vel_xy_exp(self, warp_env, stable_env, art_data):
        cfg = MockBodyCfg("robot")
        cfg.joint_ids = list(range(NUM_JOINTS))
        self._capture_mutate_check_rew(
            warp_rew.track_lin_vel_xy_exp,
            stable_rew.track_lin_vel_xy_exp,
            warp_env,
            stable_env,
            art_data,
            std=0.25,
            command_name="vel",
            asset_cfg=cfg,
        )

    def test_track_ang_vel_z_exp(self, warp_env, stable_env, art_data):
        cfg = MockBodyCfg("robot")
        cfg.joint_ids = list(range(NUM_JOINTS))
        self._capture_mutate_check_rew(
            warp_rew.track_ang_vel_z_exp,
            stable_rew.track_ang_vel_z_exp,
            warp_env,
            stable_env,
            art_data,
            std=0.25,
            command_name="vel",
            asset_cfg=cfg,
        )

    # -- terminations ------------------------------------------------------

    def test_time_out(self, warp_env, stable_env, art_data):
        out = wp.zeros((NUM_ENVS,), dtype=wp.bool, device=DEVICE)
        warp_term.time_out(warp_env, out)
        with wp.ScopedCapture() as cap:
            warp_term.time_out(warp_env, out)
        # Mutate episode length in-place
        warp_env.episode_length_buf[:] = torch.randint(0, 600, (NUM_ENVS,), dtype=torch.int64, device=DEVICE)
        wp.capture_launch(cap.graph)
        expected = stable_term.time_out(stable_env)
        assert_equal(wp.to_torch(out).clone(), expected)
