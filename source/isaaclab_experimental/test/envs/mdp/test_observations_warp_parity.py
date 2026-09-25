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
    run_warp_captured_mutated,
    run_warp_obs,
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
    """Verify experimental observation Warp kernels match stable torch implementations.

    Each term is checked eagerly, then captured, has its inputs overwritten in place, and is
    replayed: the replay must match stable on the *new* data, which proves it is capture-safe.
    """

    @staticmethod
    def _check(warp_fn, stable_fn, warp_env, stable_env, art_data, shape, **kwargs):
        assert_close(run_warp_obs(warp_fn, warp_env, shape, **kwargs), stable_fn(stable_env, **kwargs))
        actual_cap = run_warp_captured_mutated(
            warp_fn, warp_env, lambda: mutate_art_data(art_data, warp_env), shape=shape, **kwargs
        )
        assert_close(actual_cap, stable_fn(stable_env, **kwargs))

    # -- Root state observations ------------------------------------------------

    def test_base_pos_z(self, warp_env, stable_env, art_data, all_joints_cfg):
        self._check(
            warp_obs.base_pos_z,
            stable_obs.base_pos_z,
            warp_env,
            stable_env,
            art_data,
            (NUM_ENVS, 1),
            asset_cfg=all_joints_cfg,
        )

    def test_base_lin_vel(self, warp_env, stable_env, art_data, all_joints_cfg):
        self._check(
            warp_obs.base_lin_vel,
            stable_obs.base_lin_vel,
            warp_env,
            stable_env,
            art_data,
            (NUM_ENVS, 3),
            asset_cfg=all_joints_cfg,
        )

    def test_base_ang_vel(self, warp_env, stable_env, art_data, all_joints_cfg):
        self._check(
            warp_obs.base_ang_vel,
            stable_obs.base_ang_vel,
            warp_env,
            stable_env,
            art_data,
            (NUM_ENVS, 3),
            asset_cfg=all_joints_cfg,
        )

    def test_projected_gravity(self, warp_env, stable_env, art_data, all_joints_cfg):
        self._check(
            warp_obs.projected_gravity,
            stable_obs.projected_gravity,
            warp_env,
            stable_env,
            art_data,
            (NUM_ENVS, 3),
            asset_cfg=all_joints_cfg,
        )

    # -- Joint observations (non-identity subset, so the id gather is exercised) ----

    def test_joint_pos_subset(self, warp_env, stable_env, art_data, subset_cfg):
        shape = (NUM_ENVS, len(subset_cfg.joint_ids))
        self._check(
            warp_obs.joint_pos, stable_obs.joint_pos, warp_env, stable_env, art_data, shape, asset_cfg=subset_cfg
        )

    def test_joint_vel_subset(self, warp_env, stable_env, art_data, subset_cfg):
        shape = (NUM_ENVS, len(subset_cfg.joint_ids))
        self._check(
            warp_obs.joint_vel, stable_obs.joint_vel, warp_env, stable_env, art_data, shape, asset_cfg=subset_cfg
        )

    # -- Normalized joint position ----------------------------------------------

    def test_joint_pos_limit_normalized(self, warp_env, stable_env, art_data, all_joints_cfg):
        self._check(
            warp_obs.joint_pos_limit_normalized,
            stable_obs.joint_pos_limit_normalized,
            warp_env,
            stable_env,
            art_data,
            (NUM_ENVS, NUM_JOINTS),
            asset_cfg=all_joints_cfg,
        )

    # -- Action observation -----------------------------------------------------

    def test_last_action(self, warp_env, stable_env, art_data):
        # Stable last_action returns env.action_manager.action (torch tensor)
        self._check(
            warp_obs.last_action, stable_obs.last_action, warp_env, stable_env, art_data, (NUM_ENVS, NUM_ACTIONS)
        )


class TestObservationParityNewTerms:
    """Verify observation Warp kernels for newly migrated terms match stable torch implementations."""

    def test_generated_commands(self, warp_env_bodies, stable_env_bodies, cmd_tensor):
        expected = stable_obs.generated_commands(stable_env_bodies, command_name="vel")
        actual = run_warp_obs(warp_obs.generated_commands, warp_env_bodies, (NUM_ENVS, CMD_DIM), command_name="vel")
        assert_close(actual, expected)

        # Mutate the command tensor in-place: the cached zero-copy view must pick it up on replay.
        def mutate():
            cmd_tensor[:] = torch.randn_like(cmd_tensor)

        actual_cap = run_warp_captured_mutated(
            warp_obs.generated_commands, warp_env_bodies, mutate, shape=(NUM_ENVS, CMD_DIM), command_name="vel"
        )
        assert_close(actual_cap, stable_obs.generated_commands(stable_env_bodies, command_name="vel"))
