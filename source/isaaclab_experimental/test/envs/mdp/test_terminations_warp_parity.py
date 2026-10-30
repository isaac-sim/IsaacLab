# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parity tests for warp-first termination MDP terms."""

from __future__ import annotations

from types import SimpleNamespace

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
    MockPoseCommandManager,
    MockScene,
    MockSceneEntityCfg,
    MockSensorCfg,
    MockTerminationManager,
    assert_equal,
    copy_np_to_wp,
    make_pose_command_term,
    mutate_art_data,
    run_warp_captured_mutated,
    run_warp_term,
)

import isaaclab.envs.mdp.terminations as stable_term
from isaaclab.managers.manager_term_cfg import TerminationTermCfg

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
def subset_cfg():
    """Non-identity joint subset, so the masked kernel's false branch is exercised."""
    return MockSceneEntityCfg("robot", [0, 2, 5, 8], NUM_JOINTS, DEVICE)


@pytest.fixture()
def sensor_cfg():
    return MockSensorCfg("contact_sensor", BODY_IDS)


# ============================================================================
# Termination parity tests (from test_mdp_warp_parity.py)
# ============================================================================


class TestTerminationParity:
    """Verify experimental termination Warp kernels match stable torch implementations.

    Each term is checked eagerly, then captured, has its inputs overwritten in place, and is
    replayed: the replay must match stable on the *new* data, which proves it is capture-safe.
    """

    def test_root_height_below_minimum(self, warp_env, stable_env, art_data, all_joints_cfg):
        kwargs = {"minimum_height": 0.5, "asset_cfg": all_joints_cfg}
        expected = stable_term.root_height_below_minimum(stable_env, **kwargs)
        assert_equal(run_warp_term(warp_term.root_height_below_minimum, warp_env, **kwargs), expected)

        actual_cap = run_warp_captured_mutated(
            warp_term.root_height_below_minimum,
            warp_env,
            lambda: mutate_art_data(art_data, warp_env),
            dtype=wp.bool,
            **kwargs,
        )
        assert_equal(actual_cap, stable_term.root_height_below_minimum(stable_env, **kwargs))

    def test_joint_pos_out_of_manual_limit(self, warp_env, stable_env, art_data, subset_cfg):
        bounds = (-1.0, 1.0)
        cfg = subset_cfg
        expected = stable_term.joint_pos_out_of_manual_limit(stable_env, bounds=bounds, asset_cfg=cfg)
        actual = run_warp_term(warp_term.joint_pos_out_of_manual_limit, warp_env, bounds=bounds, asset_cfg=cfg)
        assert_equal(actual, expected)

        # joint_pos_out_of_manual_limit uses a 2D kernel that only writes True
        # (never clears to False), so the output must be zeroed before each call.
        # We include the zeroing inside the captured graph.
        out = wp.zeros((NUM_ENVS,), dtype=wp.bool, device=DEVICE)
        # warm-up
        out.zero_()
        warp_term.joint_pos_out_of_manual_limit(warp_env, out, bounds=bounds, asset_cfg=cfg)
        # capture (including the zero)
        with wp.ScopedCapture() as cap:
            out.zero_()
            warp_term.joint_pos_out_of_manual_limit(warp_env, out, bounds=bounds, asset_cfg=cfg)
        mutate_art_data(art_data, warp_env)
        wp.capture_launch(cap.graph)
        expected = stable_term.joint_pos_out_of_manual_limit(stable_env, bounds=bounds, asset_cfg=cfg)
        assert_equal(wp.to_torch(out).clone(), expected)


class TestTerminationParityNewTerms:
    """Verify termination Warp kernels for newly migrated terms match stable torch implementations."""

    def test_time_out(self, warp_env_bodies, stable_env_bodies):
        # Lengths straddle max_episode_length (500), so both outcomes occur.
        warp_env_bodies.episode_length_buf[:] = torch.randint(0, 600, (NUM_ENVS,), dtype=torch.int64, device=DEVICE)
        expected = stable_term.time_out(stable_env_bodies)
        assert expected.any() and not expected.all()
        assert_equal(run_warp_term(warp_term.time_out, warp_env_bodies), expected)

        def mutate():
            warp_env_bodies.episode_length_buf[:] = torch.randint(0, 600, (NUM_ENVS,), dtype=torch.int64, device=DEVICE)

        actual_cap = run_warp_captured_mutated(warp_term.time_out, warp_env_bodies, mutate, dtype=wp.bool)
        assert_equal(actual_cap, stable_term.time_out(stable_env_bodies))

    def test_illegal_contact(self, warp_env_bodies, stable_env_bodies, contact_data, sensor_cfg):
        kwargs = {"threshold": 1.0, "sensor_cfg": sensor_cfg}
        expected = stable_term.illegal_contact(stable_env_bodies, **kwargs)
        assert_equal(run_warp_term(warp_term.illegal_contact, warp_env_bodies, **kwargs), expected)

        def mutate():
            rng = np.random.RandomState(300)
            history = contact_data.net_normal_forces_w_history
            copy_np_to_wp(history, rng.randn(*history.shape, 3).astype(np.float32) * 2.0)
            wp.synchronize()

        actual_cap = run_warp_captured_mutated(
            warp_term.illegal_contact, warp_env_bodies, mutate, dtype=wp.bool, **kwargs
        )
        assert_equal(actual_cap, stable_term.illegal_contact(stable_env_bodies, **kwargs))


class TestPoseCommandSuccessParity:
    """The warp twin recomputes the pose error in a kernel, so it is checked against the
    stable term for every threshold combination — a frame or quaternion-convention slip
    would otherwise resolve cleanly and silently train against a different MDP."""

    @pytest.mark.parametrize(
        ("position_threshold", "orientation_threshold"),
        [(0.5, 1.0), (0.5, None), (None, 1.0), (None, None)],
        ids=["both", "position_only", "orientation_only", "neither"],
    )
    def test_pose_command_success(self, scene_bodies, position_threshold, orientation_threshold):
        command = make_pose_command_term(
            scene_bodies["robot"],
            position_success_threshold=position_threshold,
            orientation_success_threshold=orientation_threshold,
        )
        env = SimpleNamespace(
            scene=scene_bodies,
            command_manager=MockPoseCommandManager(command),
            num_envs=NUM_ENVS,
            device=DEVICE,
        )
        expected = stable_term.pose_command_success(env, command_name="ee_pose")
        # a degenerate all-False expectation would pass against almost any kernel
        if position_threshold is not None or orientation_threshold is not None:
            assert expected.any(), "thresholds produced no successes; the comparison would be vacuous"

        params = {"command_name": "ee_pose"}
        command._succeeded.zero_()
        warp_fn = warp_term.pose_command_success(
            TerminationTermCfg(func=warp_term.pose_command_success, params=params), env
        )
        assert_equal(run_warp_term(warp_fn, env, **params), expected)
        # the stable term ORs into the sticky tracker; the twin must too, or the terminating
        # step is never recorded before ``reset()`` reads and clears it. Capture safety is
        # owned by ``test_capture_safety.py``.
        assert_equal(command._succeeded, expected)

    def _run_both(self, scene_bodies, **thresholds):
        """Return ``(stable, warp)`` results for one threshold configuration."""
        command = make_pose_command_term(scene_bodies["robot"], **thresholds)
        env = SimpleNamespace(
            scene=scene_bodies,
            command_manager=MockPoseCommandManager(command),
            num_envs=NUM_ENVS,
            device=DEVICE,
        )
        params = {"command_name": "ee_pose"}
        expected = stable_term.pose_command_success(env, **params)
        warp_fn = warp_term.pose_command_success(
            TerminationTermCfg(func=warp_term.pose_command_success, params=params), env
        )
        return expected, run_warp_term(warp_fn, env, **params), command

    def test_non_finite_error_denies_success(self, scene_bodies):
        """A diverged env must not be reported as successful (and rewarded for it)."""
        art_data = scene_bodies["robot"].data
        art_data.body_pos_w.torch[:8, 0, :] = float("nan")

        expected, actual, _ = self._run_both(scene_bodies)

        assert not expected[:8].any(), "stable must deny success on NaN"
        assert_equal(actual, expected)

    def test_negative_threshold_is_configured_not_unset(self, scene_bodies):
        """A negative threshold is a real (unsatisfiable) bound, not an absent one."""
        expected, actual, _ = self._run_both(
            scene_bodies, position_success_threshold=-0.5, orientation_success_threshold=None
        )

        assert not expected.any(), "no distance is below a negative threshold"
        assert_equal(actual, expected)
