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
from isaaclab_experimental.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
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
    MockSceneEntityCfg,
    assert_close,
    copy_np_to_wp,
)

# ============================================================================
# Fixtures
# ============================================================================


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
    env.env_origins_wp = scene.env_origins_wp
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


def _make_event_env(seed: int, *, num_bodies: int = 1):
    """Create an independent Warp event environment for ownership tests."""

    class _Env:
        pass

    data = MockArticulationData(NUM_ENVS, NUM_JOINTS, DEVICE, seed=seed, num_bodies=num_bodies)
    asset = MockArticulation(data, num_bodies=num_bodies)
    origins = wp.zeros(NUM_ENVS, dtype=wp.vec3f, device=DEVICE)
    env = _Env()
    env.scene = MockScene({"robot": asset}, origins)
    env.num_envs = NUM_ENVS
    env.device = DEVICE
    env.env_origins_wp = env.scene.env_origins_wp
    env.rng_state_wp = wp.array(np.arange(NUM_ENVS, dtype=np.uint32) + seed, device=DEVICE)
    return env, data, asset


def _make_event_term(
    term_type: type[ManagerTermBase],
    env: object,
    *,
    mode: str = "reset",
    **params: object,
) -> ManagerTermBase:
    """Create a persistent Warp event term for a mock environment."""
    params.setdefault("asset_cfg", SceneEntityCfg("robot"))
    cfg = EventTermCfg(func=term_type, mode=mode, params=params)
    return term_type(cfg, env)


# ============================================================================
# Event parity tests: deterministic (zero-width range) warp vs stable
# ============================================================================


class TestEventParity:
    """Verify warp event functions produce the same result as stable torch equivalents.

    Since warp and stable use different RNG implementations, parity is tested using
    deterministic (zero-width) ranges where randomness has no effect. Both must
    produce ``default + 0`` (offset) or ``default * 1`` (scale), clamped to limits.
    """

    def test_reset_joints_by_offset_parity(self, warp_env, stable_env, art_data, all_joints_cfg):
        """Zero-offset: both warp and stable should produce clamped defaults."""
        cfg = all_joints_cfg
        mask = wp.array([True] * NUM_ENVS, dtype=wp.bool, device=DEVICE)

        # Set known defaults
        new_defaults = np.full((NUM_ENVS, NUM_JOINTS), 0.5, dtype=np.float32)
        copy_np_to_wp(art_data.default_joint_pos, new_defaults)

        # Run warp version
        warp_evt.reset_joints_by_offset(
            warp_env, mask, position_range=(0.0, 0.0), velocity_range=(0.0, 0.0), asset_cfg=cfg
        )
        wp.synchronize()
        warp_pos = art_data.joint_pos.torch.clone()
        warp_vel = art_data.joint_vel.torch.clone()

        # Run stable version (writes via write_joint_position_to_sim_index — which our mock
        # does not implement, so we compute the expected result directly)
        defaults_t = art_data.default_joint_pos.torch.clone()
        limits_t = art_data.soft_joint_pos_limits.torch
        vel_limits_t = art_data.soft_joint_vel_limits.torch
        expected_pos = defaults_t.clamp(limits_t[..., 0], limits_t[..., 1])
        expected_vel = art_data.default_joint_vel.torch.clone().clamp(-vel_limits_t, vel_limits_t)

        assert_close(warp_pos, expected_pos)
        assert_close(warp_vel, expected_vel)

    def test_reset_joints_by_scale_parity(self, warp_env, stable_env, art_data, all_joints_cfg):
        """Scale=1.0: both warp and stable should produce clamped defaults."""
        cfg = all_joints_cfg
        mask = wp.array([True] * NUM_ENVS, dtype=wp.bool, device=DEVICE)

        # Set known defaults
        new_defaults = np.full((NUM_ENVS, NUM_JOINTS), 0.25, dtype=np.float32)
        copy_np_to_wp(art_data.default_joint_pos, new_defaults)

        # Run warp version
        warp_evt.reset_joints_by_scale(
            warp_env, mask, position_range=(1.0, 1.0), velocity_range=(1.0, 1.0), asset_cfg=cfg
        )
        wp.synchronize()
        warp_pos = art_data.joint_pos.torch.clone()
        warp_vel = art_data.joint_vel.torch.clone()

        # Expected: default * 1.0, clamped to limits
        defaults_t = art_data.default_joint_pos.torch.clone()
        limits_t = art_data.soft_joint_pos_limits.torch
        vel_limits_t = art_data.soft_joint_vel_limits.torch
        expected_pos = defaults_t.clamp(limits_t[..., 0], limits_t[..., 1])
        expected_vel = art_data.default_joint_vel.torch.clone().clamp(-vel_limits_t, vel_limits_t)

        assert_close(warp_pos, expected_pos)
        assert_close(warp_vel, expected_vel)


# ============================================================================
# Event capture-mutate-replay tests (from test_mdp_warp_parity.py)
# ============================================================================


class TestEventCapturedDataMutation:
    """Verify event functions are capture-safe and react to mutated input data."""

    # -- reset_joints_by_offset -------------------------------------------------

    def test_reset_joints_by_offset(self, warp_env, art_data, all_joints_cfg):
        """With zero-width offset, result == defaults.  Mutate defaults -> result tracks."""
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
        result = art_data.joint_pos.torch
        expected = torch.full((NUM_ENVS, NUM_JOINTS), 0.5, device=DEVICE)
        assert_close(result, expected)

    # -- reset_joints_by_scale --------------------------------------------------

    def test_reset_joints_by_scale(self, warp_env, art_data, all_joints_cfg):
        """With scale=1.0, result == defaults.  Mutate defaults -> result tracks."""
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

        result = art_data.joint_pos.torch
        expected = torch.full((NUM_ENVS, NUM_JOINTS), 0.25, device=DEVICE)
        assert_close(result, expected)

    # -- push_by_setting_velocity -----------------------------------------------

    def test_push_by_setting_velocity(self, warp_env, art_data, all_joints_cfg):
        """With zero-width velocity range, scratch == root_vel_w.  Mutate root_vel_w -> scratch tracks."""
        mask = wp.array([True] * NUM_ENVS, dtype=wp.bool, device=DEVICE)
        captured = {}
        warp_env.scene["robot"].write_root_velocity_to_sim_mask = lambda **kwargs: captured.update(kwargs)
        zero_range = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        term = _make_event_term(
            warp_evt.push_by_setting_velocity,
            warp_env,
            mode="interval",
            velocity_range=zero_range,
        )

        term(warp_env, mask, **term.cfg.params)
        with wp.ScopedCapture() as cap:
            term(warp_env, mask, **term.cfg.params)

        # Mutate root_vel_w
        new_vel = np.tile([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], (NUM_ENVS, 1)).astype(np.float32)
        copy_np_to_wp(art_data.root_vel_w, new_vel)

        wp.capture_launch(cap.graph)
        wp.synchronize()

        scratch = wp.to_torch(captured["root_velocity"])
        expected = torch.tensor([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], device=DEVICE).expand(NUM_ENVS, -1)
        assert_close(scratch, expected)

    # -- apply_external_force_torque --------------------------------------------

    def test_apply_external_force_torque(self, warp_env, art_data, all_joints_cfg):
        """With zero-width ranges, forces/torques are zero.  Non-zero ranges produce non-zero output."""
        mask = wp.array([True] * NUM_ENVS, dtype=wp.bool, device=DEVICE)
        captured = {}
        warp_env.scene["robot"].permanent_wrench_composer.set_forces_and_torques_mask = (
            lambda **kwargs: captured.update(kwargs)
        )
        zero_range = (0.0, 0.0)
        term = _make_event_term(
            warp_evt.apply_external_force_torque,
            warp_env,
            force_range=zero_range,
            torque_range=zero_range,
        )

        # Zero-range: forces and torques should be zero
        term(warp_env, mask, **term.cfg.params)
        with wp.ScopedCapture() as cap:
            term(warp_env, mask, **term.cfg.params)
        wp.capture_launch(cap.graph)
        wp.synchronize()

        forces = wp.to_torch(captured["forces"])
        torques = wp.to_torch(captured["torques"])
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

        result = art_data.joint_pos.torch
        # Masked envs: reset to 0 (defaults + 0 offset)
        assert_close(result[: NUM_ENVS // 2], torch.zeros(NUM_ENVS // 2, NUM_JOINTS, device=DEVICE))
        # Unmasked envs: still 999.0
        assert_close(result[NUM_ENVS // 2 :], torch.full((NUM_ENVS // 2, NUM_JOINTS), 999.0, device=DEVICE))


class TestEventStateOwnership:
    """Verify event scratch and parsed ranges are owned by one environment configuration."""

    def test_com_term_is_marked_non_capturable(self):
        assert warp_evt.randomize_rigid_body_com._warp_capturable is False

    def test_push_state_is_not_shared_between_environments(self):
        env_a, data_a, asset_a = _make_event_env(101)
        env_b, data_b, asset_b = _make_event_env(202)
        copy_np_to_wp(data_a.root_vel_w, np.zeros((NUM_ENVS, 6), dtype=np.float32))
        copy_np_to_wp(data_b.root_vel_w, np.zeros((NUM_ENVS, 6), dtype=np.float32))
        captured_a = {}
        captured_b = {}
        asset_a.write_root_velocity_to_sim_mask = lambda **kwargs: captured_a.update(kwargs)
        asset_b.write_root_velocity_to_sim_mask = lambda **kwargs: captured_b.update(kwargs)
        env_mask = wp.array([True] * NUM_ENVS, dtype=wp.bool, device=DEVICE)
        range_a = {"x": (1.0, 1.0)}
        range_b = {"x": (2.0, 2.0)}
        term_a = _make_event_term(warp_evt.push_by_setting_velocity, env_a, mode="interval", velocity_range=range_a)
        term_b = _make_event_term(warp_evt.push_by_setting_velocity, env_b, mode="interval", velocity_range=range_b)

        term_a(env_a, env_mask, **term_a.cfg.params)
        term_b(env_b, env_mask, **term_b.cfg.params)
        wp.synchronize()

        velocity_a = captured_a["root_velocity"]
        velocity_b = captured_b["root_velocity"]
        assert velocity_a.ptr != velocity_b.ptr
        assert_close(wp.to_torch(velocity_a)[:, 0], torch.ones(NUM_ENVS, device=DEVICE))
        assert_close(wp.to_torch(velocity_b)[:, 0], torch.full((NUM_ENVS,), 2.0, device=DEVICE))

    def test_push_state_is_not_shared_between_configurations(self):
        env, data, asset = _make_event_env(252)
        copy_np_to_wp(data.root_vel_w, np.zeros((NUM_ENVS, 6), dtype=np.float32))
        captured = []
        asset.write_root_velocity_to_sim_mask = lambda **kwargs: captured.append(kwargs)
        env_mask = wp.array([True] * NUM_ENVS, dtype=wp.bool, device=DEVICE)
        range_a = {"x": (1.0, 1.0)}
        range_b = {"x": (2.0, 2.0)}
        term_a = _make_event_term(warp_evt.push_by_setting_velocity, env, mode="interval", velocity_range=range_a)
        term_b = _make_event_term(warp_evt.push_by_setting_velocity, env, mode="interval", velocity_range=range_b)

        term_a(env, env_mask, **term_a.cfg.params)
        term_b(env, env_mask, **term_b.cfg.params)
        wp.synchronize()

        velocity_a = captured[0]["root_velocity"]
        velocity_b = captured[1]["root_velocity"]
        assert velocity_a.ptr != velocity_b.ptr
        assert_close(wp.to_torch(velocity_a)[:, 0], torch.ones(NUM_ENVS, device=DEVICE))
        assert_close(wp.to_torch(velocity_b)[:, 0], torch.full((NUM_ENVS,), 2.0, device=DEVICE))

    def test_push_term_snapshots_range_during_initialization(self):
        env, data, asset = _make_event_env(277)
        copy_np_to_wp(data.root_vel_w, np.zeros((NUM_ENVS, 6), dtype=np.float32))
        captured = {}
        asset.write_root_velocity_to_sim_mask = lambda **kwargs: captured.update(kwargs)
        env_mask = wp.array([True] * NUM_ENVS, dtype=wp.bool, device=DEVICE)
        velocity_range = {"x": (1.0, 1.0)}
        term = _make_event_term(
            warp_evt.push_by_setting_velocity,
            env,
            mode="interval",
            velocity_range=velocity_range,
        )

        term(env, env_mask, **term.cfg.params)
        wp.synchronize()
        assert_close(wp.to_torch(captured["root_velocity"])[:, 0], torch.ones(NUM_ENVS, device=DEVICE))

        velocity_range["x"] = (4.0, 4.0)
        term(env, env_mask, **term.cfg.params)
        wp.synchronize()

        assert_close(wp.to_torch(captured["root_velocity"])[:, 0], torch.ones(NUM_ENVS, device=DEVICE))

    def test_push_state_respects_sparse_mask(self):
        env, data, asset = _make_event_env(303)
        copy_np_to_wp(data.root_vel_w, np.zeros((NUM_ENVS, 6), dtype=np.float32))
        captured = {}
        asset.write_root_velocity_to_sim_mask = lambda **kwargs: captured.update(kwargs)
        mask_np = np.arange(NUM_ENVS) % 3 == 0
        env_mask = wp.array(mask_np, dtype=wp.bool, device=DEVICE)
        term = _make_event_term(
            warp_evt.push_by_setting_velocity,
            env,
            mode="interval",
            velocity_range={"x": (3.0, 3.0)},
        )

        term(env, env_mask, **term.cfg.params)
        wp.synchronize()

        velocity = wp.to_torch(captured["root_velocity"])
        mask = torch.from_numpy(mask_np).to(device=DEVICE)
        assert_close(velocity[mask, 0], torch.full((int(mask_np.sum()),), 3.0, device=DEVICE))
        assert_close(velocity[~mask], torch.zeros((int((~mask_np).sum()), 6), device=DEVICE))

    def test_external_wrench_state_is_not_shared_between_environments(self):
        env_a, _, asset_a = _make_event_env(404, num_bodies=2)
        env_b, _, asset_b = _make_event_env(505, num_bodies=2)
        captured_a = {}
        captured_b = {}
        asset_a.permanent_wrench_composer.set_forces_and_torques_mask = lambda **kwargs: captured_a.update(kwargs)
        asset_b.permanent_wrench_composer.set_forces_and_torques_mask = lambda **kwargs: captured_b.update(kwargs)
        env_mask = wp.array([True] * NUM_ENVS, dtype=wp.bool, device=DEVICE)
        force_range_a = (1.0, 1.0)
        force_range_b = (2.0, 2.0)
        torque_range = (0.0, 0.0)
        term_a = _make_event_term(
            warp_evt.apply_external_force_torque,
            env_a,
            force_range=force_range_a,
            torque_range=torque_range,
        )
        term_b = _make_event_term(
            warp_evt.apply_external_force_torque,
            env_b,
            force_range=force_range_b,
            torque_range=torque_range,
        )

        term_a(env_a, env_mask, **term_a.cfg.params)
        term_b(env_b, env_mask, **term_b.cfg.params)
        wp.synchronize()

        forces_a = captured_a["forces"]
        forces_b = captured_b["forces"]
        assert forces_a.ptr != forces_b.ptr
        assert_close(wp.to_torch(forces_a), torch.ones((NUM_ENVS, 2, 3), device=DEVICE))
        assert_close(wp.to_torch(forces_b), torch.full((NUM_ENVS, 2, 3), 2.0, device=DEVICE))

    def test_external_wrench_respects_body_selection(self):
        env, _, asset = _make_event_env(550, num_bodies=3)
        captured = {}
        asset.permanent_wrench_composer.set_forces_and_torques_mask = lambda **kwargs: captured.update(kwargs)
        asset_cfg = SceneEntityCfg("robot", body_ids=[1])
        env_mask = wp.array([True] * NUM_ENVS, dtype=wp.bool, device=DEVICE)
        term = _make_event_term(
            warp_evt.apply_external_force_torque,
            env,
            force_range=(2.0, 2.0),
            torque_range=(0.0, 0.0),
            asset_cfg=asset_cfg,
        )

        term(env, env_mask, **term.cfg.params)
        wp.synchronize()

        expected_body_mask = torch.tensor([False, True, False], dtype=torch.bool, device=DEVICE)
        assert torch.equal(wp.to_torch(captured["body_mask"]), expected_body_mask)
        forces = wp.to_torch(captured["forces"])
        assert_close(forces[:, 0], torch.zeros((NUM_ENVS, 3), device=DEVICE))
        assert_close(forces[:, 1], torch.full((NUM_ENVS, 3), 2.0, device=DEVICE))
        assert_close(forces[:, 2], torch.zeros((NUM_ENVS, 3), device=DEVICE))

    def test_root_reset_state_is_not_shared_between_environments(self):
        env_a, data_a, asset_a = _make_event_env(606)
        env_b, data_b, asset_b = _make_event_env(707)
        default_pose = np.zeros((NUM_ENVS, 7), dtype=np.float32)
        default_pose[:, 6] = 1.0
        copy_np_to_wp(data_a.default_root_pose, default_pose)
        copy_np_to_wp(data_b.default_root_pose, default_pose)
        copy_np_to_wp(data_a.default_root_vel, np.zeros((NUM_ENVS, 6), dtype=np.float32))
        copy_np_to_wp(data_b.default_root_vel, np.zeros((NUM_ENVS, 6), dtype=np.float32))
        captured_a = {}
        captured_b = {}
        asset_a.write_root_pose_to_sim_mask = lambda **kwargs: captured_a.update(kwargs)
        asset_b.write_root_pose_to_sim_mask = lambda **kwargs: captured_b.update(kwargs)
        env_mask = wp.array([True] * NUM_ENVS, dtype=wp.bool, device=DEVICE)
        pose_range_a = {"x": (1.0, 1.0)}
        pose_range_b = {"x": (2.0, 2.0)}
        velocity_range = {}
        term_a = _make_event_term(
            warp_evt.reset_root_state_uniform,
            env_a,
            pose_range=pose_range_a,
            velocity_range=velocity_range,
        )
        term_b = _make_event_term(
            warp_evt.reset_root_state_uniform,
            env_b,
            pose_range=pose_range_b,
            velocity_range=velocity_range,
        )

        term_a(env_a, env_mask, **term_a.cfg.params)
        term_b(env_b, env_mask, **term_b.cfg.params)
        wp.synchronize()

        pose_a = captured_a["root_pose"]
        pose_b = captured_b["root_pose"]
        assert pose_a.ptr != pose_b.ptr
        assert_close(wp.to_torch(pose_a)[:, 0], torch.ones(NUM_ENVS, device=DEVICE))
        assert_close(wp.to_torch(pose_b)[:, 0], torch.full((NUM_ENVS,), 2.0, device=DEVICE))

    def test_com_ranges_are_not_shared_between_environments(self):
        env_a, data_a, asset_a = _make_event_env(808, num_bodies=2)
        env_b, data_b, asset_b = _make_event_env(909, num_bodies=2)
        copy_np_to_wp(data_a.body_com_pos_b, np.zeros((NUM_ENVS, 2, 3), dtype=np.float32))
        copy_np_to_wp(data_b.body_com_pos_b, np.zeros((NUM_ENVS, 2, 3), dtype=np.float32))
        asset_a.set_coms_mask = lambda **kwargs: None
        asset_b.set_coms_mask = lambda **kwargs: None
        asset_cfg_a = SceneEntityCfg("robot")
        asset_cfg_b = SceneEntityCfg("robot")
        env_mask = wp.array([True] * NUM_ENVS, dtype=wp.bool, device=DEVICE)
        com_range_a = {"x": (1.0, 1.0)}
        com_range_b = {"x": (2.0, 2.0)}
        term_a = _make_event_term(
            warp_evt.randomize_rigid_body_com,
            env_a,
            com_range=com_range_a,
            asset_cfg=asset_cfg_a,
        )
        term_b = _make_event_term(
            warp_evt.randomize_rigid_body_com,
            env_b,
            com_range=com_range_b,
            asset_cfg=asset_cfg_b,
        )

        term_a(env_a, env_mask, **term_a.cfg.params)
        term_b(env_b, env_mask, **term_b.cfg.params)
        wp.synchronize()

        assert term_a._default_com.ptr != term_b._default_com.ptr
        assert_close(data_a.body_com_pos_b.torch[..., 0], torch.ones((NUM_ENVS, 2), device=DEVICE))
        assert_close(data_b.body_com_pos_b.torch[..., 0], torch.full((NUM_ENVS, 2), 2.0, device=DEVICE))

    def test_com_randomization_uses_persistent_baseline(self):
        env, data, asset = _make_event_env(1001, num_bodies=2)
        baseline = np.zeros((NUM_ENVS, 2, 3), dtype=np.float32)
        baseline[..., 0] = 0.25
        copy_np_to_wp(data.body_com_pos_b, baseline)
        asset.set_coms_mask = lambda **kwargs: None
        asset_cfg = SceneEntityCfg("robot", body_ids=[0, 1])
        env_mask = wp.array([True] * NUM_ENVS, dtype=wp.bool, device=DEVICE)
        com_range = {"x": (1.0, 1.0)}
        term = _make_event_term(
            warp_evt.randomize_rigid_body_com,
            env,
            com_range=com_range,
            asset_cfg=asset_cfg,
        )

        term(env, env_mask, **term.cfg.params)
        wp.synchronize()
        assert_close(data.body_com_pos_b.torch[..., 0], torch.full((NUM_ENVS, 2), 1.25, device=DEVICE))

        term(env, env_mask, **term.cfg.params)
        wp.synchronize()
        assert_close(data.body_com_pos_b.torch[..., 0], torch.full((NUM_ENVS, 2), 1.25, device=DEVICE))

    def test_com_randomization_broadcasts_one_offset_across_selected_bodies(self):
        env, data, asset = _make_event_env(1111, num_bodies=3)
        baseline = np.zeros((NUM_ENVS, 3, 3), dtype=np.float32)
        copy_np_to_wp(data.body_com_pos_b, baseline)
        asset.set_coms_mask = lambda **kwargs: None
        asset_cfg = SceneEntityCfg("robot", body_ids=[0, 2])
        env_mask = wp.array([True] * NUM_ENVS, dtype=wp.bool, device=DEVICE)
        term = _make_event_term(
            warp_evt.randomize_rigid_body_com,
            env,
            com_range={"x": (-1.0, 1.0), "y": (-2.0, 2.0), "z": (-3.0, 3.0)},
            asset_cfg=asset_cfg,
        )

        term(env, env_mask, **term.cfg.params)
        wp.synchronize()

        assert_close(data.body_com_pos_b.torch[:, 0], data.body_com_pos_b.torch[:, 2])
        assert_close(data.body_com_pos_b.torch[:, 1], torch.zeros((NUM_ENVS, 3), device=DEVICE))
