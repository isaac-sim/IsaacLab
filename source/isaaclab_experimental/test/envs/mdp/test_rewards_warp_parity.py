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
    copy_np_to_wp,
    mutate_art_data,
    mutate_body_data,
    run_warp_captured_mutated,
    run_warp_rew,
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
def subset_cfg():
    """Non-identity joint subset, so the masked kernels' false branch is exercised."""
    return MockSceneEntityCfg("robot", [0, 2, 5, 8], NUM_JOINTS, DEVICE)


@pytest.fixture()
def sensor_cfg():
    return MockSensorCfg("contact_sensor", BODY_IDS)


# ============================================================================
# Reward parity tests (from test_mdp_warp_parity.py)
# ============================================================================


class TestRewardParity:
    """Verify experimental reward Warp kernels match stable torch implementations.

    Each term is checked eagerly, then captured, has its inputs overwritten in place, and is
    replayed: the replay must match stable on the *new* data, which proves it is capture-safe.
    """

    @staticmethod
    def _check(warp_fn, stable_fn, warp_env, stable_env, mutate, **kwargs):
        assert_close(run_warp_rew(warp_fn, warp_env, **kwargs), stable_fn(stable_env, **kwargs))
        actual_cap = run_warp_captured_mutated(warp_fn, warp_env, mutate, **kwargs)
        assert_close(actual_cap, stable_fn(stable_env, **kwargs))

    @staticmethod
    def _mutate_terminated(term_mgr):
        def mutate():
            term_mgr.terminated[:] = torch.rand(NUM_ENVS, device=DEVICE) < 0.5

        return mutate

    # -- General rewards --------------------------------------------------------

    def test_is_alive(self, warp_env, stable_env, term_mgr):
        # Set some envs as terminated so the reward is non-trivial
        term_mgr.terminated[::2] = True
        self._check(warp_rew.is_alive, stable_rew.is_alive, warp_env, stable_env, self._mutate_terminated(term_mgr))

    def test_is_terminated(self, warp_env, stable_env, term_mgr):
        term_mgr.terminated[::3] = True
        self._check(
            warp_rew.is_terminated, stable_rew.is_terminated, warp_env, stable_env, self._mutate_terminated(term_mgr)
        )

    @pytest.mark.parametrize(
        "term_keys",
        [".*", ["success", "fell_over"], ["time_out"]],
        ids=["all", "subset", "timeout_only"],
    )
    def test_is_terminated_term(self, warp_env, stable_env, term_keys):
        """The warp twin resolves the term selection to a column mask at init, so every
        selection shape is compared against the stable term's per-term lookup.

        Capture safety is owned by ``test_capture_safety.py``."""
        from isaaclab.managers.manager_term_cfg import RewardTermCfg

        params = {"term_keys": term_keys}
        stable_fn = stable_rew.is_terminated_term(
            RewardTermCfg(func=stable_rew.is_terminated_term, weight=1.0, params=params), stable_env
        )
        warp_fn = warp_rew.is_terminated_term(
            RewardTermCfg(func=warp_rew.is_terminated_term, weight=1.0, params=params), warp_env
        )
        expected = stable_fn(stable_env, term_keys=term_keys)
        # selecting only the timeout term must score zero everywhere; every other
        # selection must score somewhere, otherwise the comparison proves nothing
        assert expected.any() == (term_keys != ["time_out"])

        assert_close(run_warp_rew(warp_fn, warp_env, term_keys=term_keys), expected)

    # -- Root penalties ---------------------------------------------------------

    def test_lin_vel_z_l2(self, warp_env, stable_env, art_data, all_joints_cfg):
        self._check(
            warp_rew.lin_vel_z_l2,
            stable_rew.lin_vel_z_l2,
            warp_env,
            stable_env,
            lambda: mutate_art_data(art_data, warp_env),
            asset_cfg=all_joints_cfg,
        )

    def test_ang_vel_xy_l2(self, warp_env, stable_env, art_data, all_joints_cfg):
        self._check(
            warp_rew.ang_vel_xy_l2,
            stable_rew.ang_vel_xy_l2,
            warp_env,
            stable_env,
            lambda: mutate_art_data(art_data, warp_env),
            asset_cfg=all_joints_cfg,
        )

    def test_flat_orientation_l2(self, warp_env, stable_env, art_data, all_joints_cfg):
        self._check(
            warp_rew.flat_orientation_l2,
            stable_rew.flat_orientation_l2,
            warp_env,
            stable_env,
            lambda: mutate_art_data(art_data, warp_env),
            asset_cfg=all_joints_cfg,
        )

    # -- Masked joint penalties (subset cfg, so the mask's false branch runs) ---

    @pytest.mark.parametrize(
        "term",
        ["joint_vel_l2", "joint_acc_l2", "joint_torques_l2", "joint_vel_l1", "joint_pos_limits", "joint_deviation_l1"],
    )
    def test_masked_joint_penalty(self, warp_env, stable_env, art_data, subset_cfg, term):
        self._check(
            getattr(warp_rew, term),
            getattr(stable_rew, term),
            warp_env,
            stable_env,
            lambda: mutate_art_data(art_data, warp_env),
            asset_cfg=subset_cfg,
        )

    # -- Action penalties -------------------------------------------------------

    def test_action_l2(self, warp_env, stable_env, art_data):
        self._check(
            warp_rew.action_l2, stable_rew.action_l2, warp_env, stable_env, lambda: mutate_art_data(art_data, warp_env)
        )

    def test_action_rate_l2(self, warp_env, stable_env, art_data):
        self._check(
            warp_rew.action_rate_l2,
            stable_rew.action_rate_l2,
            warp_env,
            stable_env,
            lambda: mutate_art_data(art_data, warp_env),
        )


class TestNewRewardParity:
    """Verify newly migrated reward Warp kernels match stable torch implementations."""

    @pytest.mark.parametrize("term", ["track_lin_vel_xy_exp", "track_ang_vel_z_exp"])
    def test_track_velocity_exp(self, warp_env_bodies, stable_env_bodies, art_data_bodies, term):
        cfg = MockBodyCfg("robot")
        cfg.joint_ids = list(range(NUM_JOINTS))  # needed for stable
        TestRewardParity._check(
            getattr(warp_rew, term),
            getattr(stable_rew, term),
            warp_env_bodies,
            stable_env_bodies,
            lambda: mutate_body_data(art_data_bodies),
            std=0.25,
            command_name="vel",
            asset_cfg=cfg,
        )

    def test_undesired_contacts(self, warp_env_bodies, stable_env_bodies, contact_data, sensor_cfg):
        threshold = 1.0
        expected = stable_rew.undesired_contacts(stable_env_bodies, threshold=threshold, sensor_cfg=sensor_cfg)
        actual = run_warp_rew(warp_rew.undesired_contacts, warp_env_bodies, threshold=threshold, sensor_cfg=sensor_cfg)
        assert_close(actual, expected.float())

        def mutate():
            rng = np.random.RandomState(300)
            history = contact_data.net_normal_forces_w_history
            copy_np_to_wp(history, rng.randn(*history.shape, 3).astype(np.float32) * 2.0)
            wp.synchronize()

        actual_cap = run_warp_captured_mutated(
            warp_rew.undesired_contacts, warp_env_bodies, mutate, threshold=threshold, sensor_cfg=sensor_cfg
        )
        expected = stable_rew.undesired_contacts(stable_env_bodies, threshold=threshold, sensor_cfg=sensor_cfg)
        assert_close(actual_cap, expected.float())
