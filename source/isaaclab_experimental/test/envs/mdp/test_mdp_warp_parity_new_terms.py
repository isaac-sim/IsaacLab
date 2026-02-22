# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
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

import isaaclab.envs.mdp.observations as stable_obs
import isaaclab.envs.mdp.rewards as stable_rew
import isaaclab.envs.mdp.terminations as stable_term

# ---------------------------------------------------------------------------
NUM_ENVS = 64
NUM_JOINTS = 12
NUM_BODIES = 4
NUM_ACTIONS = 6
NUM_HISTORY = 3
CMD_DIM = 3
DEVICE = "cuda:0"
ATOL = 1e-5
RTOL = 1e-5
BODY_IDS = [0, 2]  # subset of bodies to test


# ============================================================================
# Mock infrastructure
# ============================================================================


def _make_rng(seed=42):
    return np.random.RandomState(seed)


class MockMultiBodyArticulationData:
    """Mock articulation data with multi-body arrays for body-level observations."""

    def __init__(self, device=DEVICE, seed=42):
        rng = _make_rng(seed)

        # --- Joint state ---
        self.joint_pos = wp.array(rng.randn(NUM_ENVS, NUM_JOINTS).astype(np.float32), device=device)
        self.joint_vel = wp.array(rng.randn(NUM_ENVS, NUM_JOINTS).astype(np.float32) * 2.0, device=device)
        self.default_joint_pos = wp.array(rng.randn(NUM_ENVS, NUM_JOINTS).astype(np.float32) * 0.01, device=device)
        self.default_joint_vel = wp.array(np.zeros((NUM_ENVS, NUM_JOINTS), dtype=np.float32), device=device)
        self.joint_acc = wp.array(rng.randn(NUM_ENVS, NUM_JOINTS).astype(np.float32) * 0.5, device=device)
        self.applied_torque = wp.array(rng.randn(NUM_ENVS, NUM_JOINTS).astype(np.float32) * 10.0, device=device)
        self.computed_torque = wp.array(rng.randn(NUM_ENVS, NUM_JOINTS).astype(np.float32) * 10.0, device=device)

        # --- Root state ---
        root_pos_np = rng.randn(NUM_ENVS, 3).astype(np.float32)
        root_pos_np[:, 2] = np.abs(root_pos_np[:, 2]) + 0.1
        self.root_pos_w = wp.array(root_pos_np, dtype=wp.vec3f, device=device)
        self.root_lin_vel_b = wp.array(rng.randn(NUM_ENVS, 3).astype(np.float32), dtype=wp.vec3f, device=device)
        self.root_ang_vel_b = wp.array(rng.randn(NUM_ENVS, 3).astype(np.float32), dtype=wp.vec3f, device=device)

        # --- Soft limits ---
        limits_np = np.zeros((NUM_ENVS, NUM_JOINTS, 2), dtype=np.float32)
        limits_np[:, :, 0] = -3.14
        limits_np[:, :, 1] = 3.14
        self.soft_joint_pos_limits = wp.array(limits_np, dtype=wp.vec2f, device=device)
        self.soft_joint_vel_limits = wp.array(np.full((NUM_ENVS, NUM_JOINTS), 10.0, dtype=np.float32), device=device)

        # --- Body-level data (2D vec3f / transformf) ---
        # projected_gravity_b: (num_envs, num_bodies) vec3f
        grav_np = rng.randn(NUM_ENVS, NUM_BODIES, 3).astype(np.float32)
        grav_np[:, :, 2] = -1.0
        norms = np.linalg.norm(grav_np, axis=2, keepdims=True)
        grav_np /= norms
        self.projected_gravity_b = wp.array(grav_np, dtype=wp.vec3f, device=device)

        # body_pose_w: (num_envs, num_bodies) transformf — pos + identity quat
        pose_np = np.zeros((NUM_ENVS, NUM_BODIES, 7), dtype=np.float32)
        pose_np[:, :, :3] = rng.randn(NUM_ENVS, NUM_BODIES, 3).astype(np.float32)
        pose_np[:, :, 3:7] = [0.0, 0.0, 0.0, 1.0]
        self.body_pose_w = wp.array(pose_np, dtype=wp.transformf, device=device)

        # body_lin_acc_w: (num_envs, num_bodies) vec3f
        self.body_lin_acc_w = wp.array(
            rng.randn(NUM_ENVS, NUM_BODIES, 3).astype(np.float32), dtype=wp.vec3f, device=device
        )

        # body_com_pos_b: (num_envs, num_bodies) vec3f
        self.body_com_pos_b = wp.array(
            rng.randn(NUM_ENVS, NUM_BODIES, 3).astype(np.float32) * 0.01, dtype=wp.vec3f, device=device
        )

        # Event-specific
        self.root_vel_w = wp.array(rng.randn(NUM_ENVS, 6).astype(np.float32), dtype=wp.spatial_vectorf, device=device)
        default_pose_np = np.zeros((NUM_ENVS, 7), dtype=np.float32)
        default_pose_np[:, 0:3] = rng.randn(NUM_ENVS, 3).astype(np.float32) * 0.1
        default_pose_np[:, 3:7] = [0.0, 0.0, 0.0, 1.0]
        self.default_root_pose = wp.array(default_pose_np, dtype=wp.transformf, device=device)
        self.default_root_vel = wp.array(
            np.zeros((NUM_ENVS, 6), dtype=np.float32), dtype=wp.spatial_vectorf, device=device
        )

        quat_np = rng.randn(NUM_ENVS, 4).astype(np.float32)
        quat_np /= np.linalg.norm(quat_np, axis=1, keepdims=True)
        self.root_quat_w = wp.array(quat_np, dtype=wp.quatf, device=device)
        self.root_lin_vel_w = wp.array(rng.randn(NUM_ENVS, 3).astype(np.float32), dtype=wp.vec3f, device=device)
        self.root_ang_vel_w = wp.array(rng.randn(NUM_ENVS, 3).astype(np.float32), dtype=wp.vec3f, device=device)

    def resolve_joint_mask(self, joint_ids=None):
        mask = [False] * NUM_JOINTS
        if joint_ids is None or isinstance(joint_ids, slice):
            mask = [True] * NUM_JOINTS
        else:
            for j in joint_ids:
                mask[j] = True
        return wp.array(mask, dtype=wp.bool, device=DEVICE)


class MockMultiBodyArticulation:
    def __init__(self, data: MockMultiBodyArticulationData):
        self.data = data
        self.num_bodies = NUM_BODIES
        self.num_joints = NUM_JOINTS
        self.device = DEVICE

    def write_root_velocity_to_sim(self, *a, **kw):
        pass

    def write_root_pose_to_sim(self, *a, **kw):
        pass

    def write_joint_state_to_sim(self, *a, **kw):
        pass

    def set_external_force_and_torque(self, *a, **kw):
        pass

    def find_joints(self, names, preserve_order=False):
        return None, [f"j{i}" for i in range(NUM_JOINTS)], list(range(NUM_JOINTS))


class MockContactSensorData:
    def __init__(self, device=DEVICE, seed=77):
        rng = _make_rng(seed)
        self.net_forces_w_history = torch.tensor(
            rng.randn(NUM_ENVS, NUM_HISTORY, NUM_BODIES, 3).astype(np.float32), device=device
        )


class MockContactSensor:
    def __init__(self, data: MockContactSensorData):
        self.data = data
        self.num_bodies = NUM_BODIES


class MockCommandTerm:
    def __init__(self, device=DEVICE, seed=88):
        rng = _make_rng(seed)
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


class MockScene:
    def __init__(self, assets: dict, env_origins, sensors=None):
        self._assets = assets
        self.env_origins = env_origins
        self.sensors = sensors or {}
        self.articulations = {k: v for k, v in assets.items()}
        self.rigid_objects = {}
        self.num_envs = NUM_ENVS

    def __getitem__(self, name: str):
        return self._assets[name]


class MockActionManagerWarp:
    def __init__(self, action_wp, prev_action_wp):
        self._action = action_wp
        self._prev_action = prev_action_wp

    @property
    def action(self):
        return self._action

    @property
    def prev_action(self):
        return self._prev_action


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def _clear_caches():
    yield
    # Clear function-level caches from all new warp functions
    fns_to_clear = [
        warp_obs.body_projected_gravity_b,
        warp_obs.body_pose_w,
        warp_obs.generated_commands,
        warp_rew.body_lin_acc_l2,
        warp_rew.track_lin_vel_xy_exp,
        warp_rew.track_ang_vel_z_exp,
        warp_rew.undesired_contacts,
        warp_rew.desired_contacts,
        warp_rew.contact_forces,
        warp_term.command_resample,
        warp_term.illegal_contact,
        warp_evt.reset_root_state_with_random_orientation,
        warp_evt.reset_scene_to_default,
        warp_evt.randomize_rigid_body_com,
    ]
    for fn in fns_to_clear:
        for attr in list(vars(fn)):
            if attr.startswith("_"):
                delattr(fn, attr)


@pytest.fixture()
def art_data():
    return MockMultiBodyArticulationData()


@pytest.fixture()
def env_origins():
    origins_np = _make_rng(77).randn(NUM_ENVS, 3).astype(np.float32)
    return wp.array(origins_np, dtype=wp.vec3f, device=DEVICE)


@pytest.fixture()
def contact_data():
    return MockContactSensorData()


@pytest.fixture()
def cmd_tensor():
    rng = _make_rng(99)
    return torch.tensor(rng.randn(NUM_ENVS, CMD_DIM).astype(np.float32), device=DEVICE)


@pytest.fixture()
def cmd_term():
    return MockCommandTerm()


@pytest.fixture()
def scene(art_data, env_origins, contact_data):
    art = MockMultiBodyArticulation(art_data)
    sensor = MockContactSensor(contact_data)
    return MockScene({"robot": art}, env_origins, sensors={"contact_sensor": sensor})


@pytest.fixture()
def action_wp():
    rng = _make_rng(55)
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


def _run_warp_obs(func, env, shape, **kwargs):
    out = wp.zeros(shape, dtype=wp.float32, device=DEVICE)
    func(env, out, **kwargs)
    return wp.to_torch(out).clone()


def _run_warp_obs_captured(func, env, shape, **kwargs):
    out = wp.zeros(shape, dtype=wp.float32, device=DEVICE)
    func(env, out, **kwargs)
    with wp.ScopedCapture() as cap:
        func(env, out, **kwargs)
    wp.capture_launch(cap.graph)
    return wp.to_torch(out).clone()


def _run_warp_rew(func, env, **kwargs):
    out = wp.zeros((NUM_ENVS,), dtype=wp.float32, device=DEVICE)
    func(env, out, **kwargs)
    return wp.to_torch(out).clone()


def _run_warp_rew_captured(func, env, **kwargs):
    out = wp.zeros((NUM_ENVS,), dtype=wp.float32, device=DEVICE)
    func(env, out, **kwargs)
    with wp.ScopedCapture() as cap:
        func(env, out, **kwargs)
    wp.capture_launch(cap.graph)
    return wp.to_torch(out).clone()


def _run_warp_term(func, env, **kwargs):
    out = wp.zeros((NUM_ENVS,), dtype=wp.bool, device=DEVICE)
    func(env, out, **kwargs)
    return wp.to_torch(out).clone()


def _run_warp_term_captured(func, env, **kwargs):
    out = wp.zeros((NUM_ENVS,), dtype=wp.bool, device=DEVICE)
    func(env, out, **kwargs)
    with wp.ScopedCapture() as cap:
        func(env, out, **kwargs)
    wp.capture_launch(cap.graph)
    return wp.to_torch(out).clone()


def assert_close(actual, expected, atol=ATOL, rtol=RTOL):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def assert_equal(actual, expected):
    assert torch.equal(actual, expected), f"Mismatch:\n  actual:   {actual}\n  expected: {expected}"


# ============================================================================
# Body observation parity tests
# ============================================================================


class TestBodyObservationParity:
    """Verify body-level observation Warp kernels match stable torch implementations."""

    def test_body_projected_gravity_b(self, warp_env, stable_env, body_cfg):
        n_sel = len(body_cfg.body_ids)
        expected = stable_obs.body_projected_gravity_b(stable_env, asset_cfg=body_cfg)
        actual = _run_warp_obs(warp_obs.body_projected_gravity_b, warp_env, (NUM_ENVS, n_sel * 3), asset_cfg=body_cfg)
        actual_cap = _run_warp_obs_captured(
            warp_obs.body_projected_gravity_b, warp_env, (NUM_ENVS, n_sel * 3), asset_cfg=body_cfg
        )
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_body_pose_w(self, warp_env, stable_env, body_cfg):
        n_sel = len(body_cfg.body_ids)
        # Stable body_pose_w calls env.scene.env_origins.unsqueeze(1) — needs torch tensor.
        # Temporarily swap env_origins to torch for the stable call.
        orig_origins = stable_env.scene.env_origins
        stable_env.scene.env_origins = wp.to_torch(orig_origins)
        expected = stable_obs.body_pose_w(stable_env, asset_cfg=body_cfg)
        stable_env.scene.env_origins = orig_origins  # restore
        actual = _run_warp_obs(warp_obs.body_pose_w, warp_env, (NUM_ENVS, n_sel * 7), asset_cfg=body_cfg)
        actual_cap = _run_warp_obs_captured(warp_obs.body_pose_w, warp_env, (NUM_ENVS, n_sel * 7), asset_cfg=body_cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_generated_commands(self, warp_env, stable_env):
        expected = stable_obs.generated_commands(stable_env, command_name="vel")
        actual = _run_warp_obs(warp_obs.generated_commands, warp_env, (NUM_ENVS, CMD_DIM), command_name="vel")
        actual_cap = _run_warp_obs_captured(
            warp_obs.generated_commands, warp_env, (NUM_ENVS, CMD_DIM), command_name="vel"
        )
        assert_close(actual, expected)
        assert_close(actual_cap, expected)


# ============================================================================
# New reward parity tests
# ============================================================================


class TestNewRewardParity:
    """Verify newly migrated reward Warp kernels match stable torch implementations."""

    def test_body_lin_acc_l2(self, warp_env, stable_env, body_cfg):
        expected = stable_rew.body_lin_acc_l2(stable_env, asset_cfg=body_cfg)
        actual = _run_warp_rew(warp_rew.body_lin_acc_l2, warp_env, asset_cfg=body_cfg)
        actual_cap = _run_warp_rew_captured(warp_rew.body_lin_acc_l2, warp_env, asset_cfg=body_cfg)
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_track_lin_vel_xy_exp(self, warp_env, stable_env, body_cfg):
        cfg = MockBodyCfg("robot")
        cfg.joint_ids = list(range(NUM_JOINTS))  # needed for stable
        std = 0.25
        expected = stable_rew.track_lin_vel_xy_exp(stable_env, std=std, command_name="vel", asset_cfg=cfg)
        actual = _run_warp_rew(warp_rew.track_lin_vel_xy_exp, warp_env, std=std, command_name="vel", asset_cfg=cfg)
        actual_cap = _run_warp_rew_captured(
            warp_rew.track_lin_vel_xy_exp, warp_env, std=std, command_name="vel", asset_cfg=cfg
        )
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_track_ang_vel_z_exp(self, warp_env, stable_env, body_cfg):
        cfg = MockBodyCfg("robot")
        cfg.joint_ids = list(range(NUM_JOINTS))
        std = 0.25
        expected = stable_rew.track_ang_vel_z_exp(stable_env, std=std, command_name="vel", asset_cfg=cfg)
        actual = _run_warp_rew(warp_rew.track_ang_vel_z_exp, warp_env, std=std, command_name="vel", asset_cfg=cfg)
        actual_cap = _run_warp_rew_captured(
            warp_rew.track_ang_vel_z_exp, warp_env, std=std, command_name="vel", asset_cfg=cfg
        )
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_undesired_contacts(self, warp_env, stable_env, sensor_cfg):
        threshold = 0.5
        # Stable returns int64 (torch.sum of bools); warp returns float32 — cast for comparison.
        expected = stable_rew.undesired_contacts(stable_env, threshold=threshold, sensor_cfg=sensor_cfg).float()
        actual = _run_warp_rew(warp_rew.undesired_contacts, warp_env, threshold=threshold, sensor_cfg=sensor_cfg)
        actual_cap = _run_warp_rew_captured(
            warp_rew.undesired_contacts, warp_env, threshold=threshold, sensor_cfg=sensor_cfg
        )
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_desired_contacts(self, warp_env, stable_env, sensor_cfg):
        threshold = 0.5
        expected = stable_rew.desired_contacts(stable_env, sensor_cfg=sensor_cfg, threshold=threshold)
        actual = _run_warp_rew(warp_rew.desired_contacts, warp_env, sensor_cfg=sensor_cfg, threshold=threshold)
        actual_cap = _run_warp_rew_captured(
            warp_rew.desired_contacts, warp_env, sensor_cfg=sensor_cfg, threshold=threshold
        )
        assert_close(actual, expected)
        assert_close(actual_cap, expected)

    def test_contact_forces(self, warp_env, stable_env, sensor_cfg):
        threshold = 0.5
        expected = stable_rew.contact_forces(stable_env, threshold=threshold, sensor_cfg=sensor_cfg)
        actual = _run_warp_rew(warp_rew.contact_forces, warp_env, threshold=threshold, sensor_cfg=sensor_cfg)
        actual_cap = _run_warp_rew_captured(
            warp_rew.contact_forces, warp_env, threshold=threshold, sensor_cfg=sensor_cfg
        )
        assert_close(actual, expected)
        assert_close(actual_cap, expected)


# ============================================================================
# New termination parity tests
# ============================================================================


class TestNewTerminationParity:
    """Verify newly migrated termination Warp kernels match stable torch implementations."""

    def test_time_out(self, warp_env, stable_env):
        expected = stable_term.time_out(stable_env)
        actual = _run_warp_term(warp_term.time_out, warp_env)
        actual_cap = _run_warp_term_captured(warp_term.time_out, warp_env)
        assert_equal(actual, expected)
        assert_equal(actual_cap, expected)

    def test_illegal_contact(self, warp_env, stable_env, sensor_cfg):
        threshold = 0.5
        expected = stable_term.illegal_contact(stable_env, threshold=threshold, sensor_cfg=sensor_cfg)
        actual = _run_warp_term(warp_term.illegal_contact, warp_env, threshold=threshold, sensor_cfg=sensor_cfg)
        actual_cap = _run_warp_term_captured(
            warp_term.illegal_contact, warp_env, threshold=threshold, sensor_cfg=sensor_cfg
        )
        assert_equal(actual, expected)
        assert_equal(actual_cap, expected)


# ============================================================================
# New event capture-safety tests
# ============================================================================


def _copy_np_to_wp(dest: wp.array, src_np: np.ndarray):
    tmp = wp.array(src_np, dtype=dest.dtype, device=str(dest.device))
    wp.copy(dest, tmp)


class TestNewEventCaptureSafety:
    """Verify new event functions are capture-safe."""

    def test_reset_root_state_with_random_orientation(self, warp_env, art_data, env_origins):
        """With zero-width position ranges, positions = default + env_origin."""
        mask = wp.array([True] * NUM_ENVS, dtype=wp.bool, device=DEVICE)
        zero_pose = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
        zero_vel = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

        warp_evt.reset_root_state_with_random_orientation(warp_env, mask, pose_range=zero_pose, velocity_range=zero_vel)
        with wp.ScopedCapture() as cap:
            warp_evt.reset_root_state_with_random_orientation(
                warp_env, mask, pose_range=zero_pose, velocity_range=zero_vel
            )

        # Mutate defaults
        new_pose = np.zeros((NUM_ENVS, 7), dtype=np.float32)
        new_pose[:, 0:3] = [1.0, 2.0, 3.0]
        new_pose[:, 3:7] = [0.0, 0.0, 0.0, 1.0]
        _copy_np_to_wp(art_data.default_root_pose, new_pose)

        wp.capture_launch(cap.graph)
        wp.synchronize()

        fn = warp_evt.reset_root_state_with_random_orientation
        scratch_pose = wp.to_torch(fn._scratch_pose)
        origins_t = wp.to_torch(env_origins)

        # Positions: default(1,2,3) + env_origin + 0
        expected_pos = torch.tensor([1.0, 2.0, 3.0], device=DEVICE).unsqueeze(0) + origins_t
        assert_close(scratch_pose[:, :3], expected_pos)

        # Quaternions: should be unit quaternions (random SO(3))
        qnorm = scratch_pose[:, 3:7].norm(dim=1)
        assert_close(qnorm, torch.ones(NUM_ENVS, device=DEVICE), atol=1e-4, rtol=1e-4)

    def test_reset_scene_to_default(self, warp_env, art_data, env_origins):
        """With all envs masked, joints should be reset to defaults."""
        mask = wp.array([True] * NUM_ENVS, dtype=wp.bool, device=DEVICE)

        # Set defaults to known values
        _copy_np_to_wp(art_data.default_joint_pos, np.full((NUM_ENVS, NUM_JOINTS), 0.42, dtype=np.float32))
        _copy_np_to_wp(art_data.default_joint_vel, np.zeros((NUM_ENVS, NUM_JOINTS), dtype=np.float32))

        warp_evt.reset_scene_to_default(warp_env, mask)
        wp.synchronize()

        result_pos = wp.to_torch(art_data.joint_pos)
        expected = torch.full((NUM_ENVS, NUM_JOINTS), 0.42, device=DEVICE)
        assert_close(result_pos, expected)

    def test_reset_scene_to_default_mask_selectivity(self, warp_env, art_data, env_origins):
        """Only masked envs are reset."""
        mask_np = np.array([i < NUM_ENVS // 2 for i in range(NUM_ENVS)])
        mask = wp.array(mask_np, dtype=wp.bool, device=DEVICE)

        # Set joint_pos to sentinel
        _copy_np_to_wp(art_data.joint_pos, np.full((NUM_ENVS, NUM_JOINTS), 999.0, dtype=np.float32))
        # Set defaults to 0
        _copy_np_to_wp(art_data.default_joint_pos, np.zeros((NUM_ENVS, NUM_JOINTS), dtype=np.float32))

        warp_evt.reset_scene_to_default(warp_env, mask)
        wp.synchronize()

        result = wp.to_torch(art_data.joint_pos)
        # Masked: reset to 0
        assert_close(result[: NUM_ENVS // 2], torch.zeros(NUM_ENVS // 2, NUM_JOINTS, device=DEVICE))
        # Unmasked: still 999
        assert_close(result[NUM_ENVS // 2 :], torch.full((NUM_ENVS // 2, NUM_JOINTS), 999.0, device=DEVICE))

    def test_randomize_rigid_body_com(self, warp_env, art_data):
        """With zero-width range, CoM should not change. With nonzero range, CoM should differ."""
        mask = wp.array([True] * NUM_ENVS, dtype=wp.bool, device=DEVICE)
        body_cfg = MockBodyCfg("robot", list(range(NUM_BODIES)))

        # Snapshot original CoM
        original_com = wp.to_torch(art_data.body_com_pos_b).clone()

        # Zero range: no change
        warp_evt.randomize_rigid_body_com(
            warp_env, mask, com_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}, asset_cfg=body_cfg
        )
        wp.synchronize()
        assert_close(wp.to_torch(art_data.body_com_pos_b), original_com)

    def test_reset_root_state_from_terrain(self, warp_env, art_data, env_origins):
        """With zero-width orientation and velocity ranges, verify positions come from terrain patches."""
        # Create mock terrain
        rng = _make_rng(123)
        num_levels, num_types, num_patches = 2, 2, 5
        flat_patches_np = rng.randn(num_levels, num_types, num_patches, 3).astype(np.float32)
        flat_patches_torch = torch.tensor(flat_patches_np, device=DEVICE)

        terrain_levels = torch.zeros(NUM_ENVS, dtype=torch.int32, device=DEVICE)
        terrain_types = torch.zeros(NUM_ENVS, dtype=torch.int32, device=DEVICE)

        # Attach terrain mock to scene
        warp_env.scene.terrain = type(
            "_T",
            (),
            {
                "flat_patches": {"init_pos": flat_patches_torch},
                "terrain_levels": terrain_levels,
                "terrain_types": terrain_types,
            },
        )()

        mask = wp.array([True] * NUM_ENVS, dtype=wp.bool, device=DEVICE)
        zero_pose = {"roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)}
        zero_vel = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

        warp_evt.reset_root_state_from_terrain(warp_env, mask, pose_range=zero_pose, velocity_range=zero_vel)
        wp.synchronize()

        fn = warp_evt.reset_root_state_from_terrain
        scratch_pose = wp.to_torch(fn._scratch_pose)

        # All envs use level=0, type=0 so positions must come from flat_patches[0, 0, *, :]
        valid_positions = flat_patches_torch[0, 0]  # (num_patches, 3)
        default_pos = wp.to_torch(art_data.default_root_pose)[:, :3]

        # Each env's position should be one of the valid patches + default offset
        for i in range(min(8, NUM_ENVS)):  # spot check first 8
            pos = scratch_pose[i, :3]
            diffs = (valid_positions + default_pos[i]) - pos
            min_dist = diffs.norm(dim=1).min()
            assert min_dist < 1e-4, f"env {i}: position {pos} not near any valid patch"

    def test_command_resample(self, warp_env, cmd_term):
        """Parity check for command_resample termination."""
        # Set up deterministic data: half the envs should trigger
        cmd_term.time_left[:] = 0.01  # all below step_dt=0.02
        cmd_term.command_counter[: NUM_ENVS // 2] = 1.0  # match num_resamples=1
        cmd_term.command_counter[NUM_ENVS // 2 :] = 0.0  # no match

        expected = torch.logical_and(
            cmd_term.time_left <= warp_env.step_dt,
            cmd_term.command_counter == 1.0,
        )

        actual = _run_warp_term(warp_term.command_resample, warp_env, command_name="vel", num_resamples=1)
        assert_equal(actual, expected)


# ============================================================================
# Capture-mutate-replay tests for new terms
# ============================================================================


def _mutate_body_data(art_data: MockMultiBodyArticulationData, rng_seed=200):
    """Mutate body-level and root-level data in-place so captured graphs see fresh values."""
    rng = _make_rng(rng_seed)

    # Root state
    root_pos_np = rng.randn(NUM_ENVS, 3).astype(np.float32)
    root_pos_np[:, 2] = np.abs(root_pos_np[:, 2]) + 0.05
    _copy_np_to_wp(art_data.root_pos_w, root_pos_np)
    _copy_np_to_wp(art_data.root_lin_vel_b, rng.randn(NUM_ENVS, 3).astype(np.float32))
    _copy_np_to_wp(art_data.root_ang_vel_b, rng.randn(NUM_ENVS, 3).astype(np.float32))

    # Body data
    grav_np = rng.randn(NUM_ENVS, NUM_BODIES, 3).astype(np.float32)
    grav_np[:, :, 2] = -1.0
    grav_np /= np.linalg.norm(grav_np, axis=2, keepdims=True)
    _copy_np_to_wp(art_data.projected_gravity_b, grav_np)

    _copy_np_to_wp(art_data.body_lin_acc_w, rng.randn(NUM_ENVS, NUM_BODIES, 3).astype(np.float32))

    pose_np = np.zeros((NUM_ENVS, NUM_BODIES, 7), dtype=np.float32)
    pose_np[:, :, :3] = rng.randn(NUM_ENVS, NUM_BODIES, 3).astype(np.float32)
    pose_np[:, :, 3:7] = [0.0, 0.0, 0.0, 1.0]
    _copy_np_to_wp(art_data.body_pose_w, pose_np)

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

    # -- body observations -------------------------------------------------

    def test_body_projected_gravity_b(self, warp_env, stable_env, art_data, body_cfg):
        n_sel = len(body_cfg.body_ids)
        self._capture_mutate_check_obs(
            warp_obs.body_projected_gravity_b,
            stable_obs.body_projected_gravity_b,
            warp_env,
            stable_env,
            art_data,
            (NUM_ENVS, n_sel * 3),
            asset_cfg=body_cfg,
        )

    def test_body_pose_w(self, warp_env, stable_env, art_data, body_cfg):
        n_sel = len(body_cfg.body_ids)

        # Stable needs torch env_origins for unsqueeze
        def stable_body_pose_w_fixed(env, **kw):
            orig = env.scene.env_origins
            env.scene.env_origins = wp.to_torch(orig)
            result = stable_obs.body_pose_w(env, **kw)
            env.scene.env_origins = orig
            return result

        out = wp.zeros((NUM_ENVS, n_sel * 7), dtype=wp.float32, device=DEVICE)
        warp_obs.body_pose_w(warp_env, out, asset_cfg=body_cfg)
        with wp.ScopedCapture() as cap:
            warp_obs.body_pose_w(warp_env, out, asset_cfg=body_cfg)
        _mutate_body_data(art_data)
        wp.capture_launch(cap.graph)
        expected = stable_body_pose_w_fixed(stable_env, asset_cfg=body_cfg)
        assert_close(wp.to_torch(out).clone(), expected)

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

    def test_body_lin_acc_l2(self, warp_env, stable_env, art_data, body_cfg):
        self._capture_mutate_check_rew(
            warp_rew.body_lin_acc_l2,
            stable_rew.body_lin_acc_l2,
            warp_env,
            stable_env,
            art_data,
            asset_cfg=body_cfg,
        )

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

    def test_contact_forces(self, warp_env, stable_env, art_data, contact_data, sensor_cfg):
        """Mutate contact force history, verify captured graph picks up changes."""
        out = wp.zeros((NUM_ENVS,), dtype=wp.float32, device=DEVICE)
        warp_rew.contact_forces(warp_env, out, threshold=0.5, sensor_cfg=sensor_cfg)
        with wp.ScopedCapture() as cap:
            warp_rew.contact_forces(warp_env, out, threshold=0.5, sensor_cfg=sensor_cfg)
        # Mutate contact sensor data in-place
        contact_data.net_forces_w_history[:] = torch.randn_like(contact_data.net_forces_w_history) * 3.0
        wp.capture_launch(cap.graph)
        expected = stable_rew.contact_forces(stable_env, threshold=0.5, sensor_cfg=sensor_cfg)
        assert_close(wp.to_torch(out).clone(), expected)

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

    def test_illegal_contact(self, warp_env, stable_env, art_data, contact_data, sensor_cfg):
        out = wp.zeros((NUM_ENVS,), dtype=wp.bool, device=DEVICE)
        warp_term.illegal_contact(warp_env, out, threshold=0.5, sensor_cfg=sensor_cfg)
        with wp.ScopedCapture() as cap:
            warp_term.illegal_contact(warp_env, out, threshold=0.5, sensor_cfg=sensor_cfg)
        contact_data.net_forces_w_history[:] = torch.randn_like(contact_data.net_forces_w_history) * 5.0
        wp.capture_launch(cap.graph)
        expected = stable_term.illegal_contact(stable_env, threshold=0.5, sensor_cfg=sensor_cfg)
        assert_equal(wp.to_torch(out).clone(), expected)
