# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parity tests for warp-first event MDP terms."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.utils.seed import WarpRng

# Skip entire module if no CUDA device available
wp.init()
pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA device required")

import isaaclab_experimental.envs.mdp.events as warp_evt
from isaaclab_experimental.managers import EventManager, EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab_experimental.utils import CapturedStage
from parity_helpers import (
    DEVICE,
    NUM_ACTIONS,
    NUM_ENVS,
    NUM_JOINTS,
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


@pytest.fixture(autouse=True)
def _reset_warp_rng():
    """Drop the Warp RNG state each test sets."""
    yield
    WarpRng.state = None


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
    env.episode_length_buf = episode_length_buf
    env.step_dt = 0.02
    env.max_episode_length_s = 10.0
    env.env_origins_wp = scene.env_origins
    # RNG state for events (seeded deterministically)
    WarpRng.state = wp.array(np.arange(NUM_ENVS, dtype=np.uint32) + 42, device=DEVICE)
    return env


@pytest.fixture()
def all_joints_cfg():
    return MockSceneEntityCfg("robot", list(range(NUM_JOINTS)), NUM_JOINTS, DEVICE)


def _make_event_term(term_type: type[ManagerTermBase], env, mode: str = "reset", **params) -> ManagerTermBase:
    """Build a class event term the way the event manager does, after resolving its asset's Warp body fields."""
    params.setdefault("asset_cfg", SceneEntityCfg("robot"))
    cfg = EventTermCfg(func=term_type, mode=mode, params=params)
    # the mock scene cannot run SceneEntityCfg.resolve, so fill the fields it would set on the stored cfg
    asset_cfg = cfg.params["asset_cfg"]
    num_bodies = env.scene[asset_cfg.name].num_bodies
    body_ids = list(range(num_bodies)) if asset_cfg.body_ids in (None, slice(None)) else list(asset_cfg.body_ids)
    asset_cfg.body_ids_wp = wp.array(body_ids, dtype=wp.int32, device=DEVICE)
    asset_cfg.body_mask_wp = wp.array([i in body_ids for i in range(num_bodies)], dtype=wp.bool, device=DEVICE)
    return term_type(cfg, env)


def _all_envs() -> wp.array:
    return wp.array([True] * NUM_ENVS, dtype=wp.bool, device=DEVICE)


# ============================================================================
# Event capture-mutate-replay tests
# ============================================================================


def _defaults_beyond_limits(value: float) -> np.ndarray:
    """Defaults at ``value`` with half the joints pushed past the +/-3.14 soft limits."""
    defaults = np.full((NUM_ENVS, NUM_JOINTS), value, dtype=np.float32)
    defaults[:, 0::4] = 5.0
    defaults[:, 1::4] = -5.0
    return defaults


def _clamped(defaults: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.clip(defaults, -3.14, 3.14), device=DEVICE)


class TestEventCapturedDataMutation:
    """Verify event functions are capture-safe and react to mutated input data."""

    # -- reset_joints_by_offset -------------------------------------------------

    def test_reset_joints_by_offset(self, warp_env, art_data, all_joints_cfg):
        """With zero-width offset, result == defaults clamped to limits.  Mutate defaults -> result tracks."""
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

        # Mutate defaults in-place, some beyond the soft limits
        new_defaults = _defaults_beyond_limits(0.5)
        copy_np_to_wp(art_data.default_joint_pos, new_defaults)

        # Replay
        wp.capture_launch(cap.graph)
        wp.synchronize()

        assert_close(art_data.joint_pos.torch, _clamped(new_defaults))
        vel_limits_t = art_data.soft_joint_vel_limits.torch
        expected_vel = art_data.default_joint_vel.torch.clone().clamp(-vel_limits_t, vel_limits_t)
        assert_close(art_data.joint_vel.torch, expected_vel)

    # -- reset_joints_by_scale --------------------------------------------------

    def test_reset_joints_by_scale(self, warp_env, art_data, all_joints_cfg):
        """With scale=1.0, result == defaults clamped to limits.  Mutate defaults -> result tracks."""
        cfg = all_joints_cfg
        mask = wp.array([True] * NUM_ENVS, dtype=wp.bool, device=DEVICE)

        warp_evt.reset_joints_by_scale(
            warp_env, mask, position_range=(1.0, 1.0), velocity_range=(1.0, 1.0), asset_cfg=cfg
        )
        with wp.ScopedCapture() as cap:
            warp_evt.reset_joints_by_scale(
                warp_env, mask, position_range=(1.0, 1.0), velocity_range=(1.0, 1.0), asset_cfg=cfg
            )

        new_defaults = _defaults_beyond_limits(0.25)
        copy_np_to_wp(art_data.default_joint_pos, new_defaults)

        wp.capture_launch(cap.graph)
        wp.synchronize()

        assert_close(art_data.joint_pos.torch, _clamped(new_defaults))
        vel_limits_t = art_data.soft_joint_vel_limits.torch
        expected_vel = art_data.default_joint_vel.torch.clone().clamp(-vel_limits_t, vel_limits_t)
        assert_close(art_data.joint_vel.torch, expected_vel)

    # -- push_by_setting_velocity -----------------------------------------------

    def test_push_by_setting_velocity(self, warp_env, art_data, all_joints_cfg):
        """With zero-width velocity range, the written root velocity == root_vel_w.  Mutate root_vel_w -> tracks."""
        mask = _all_envs()
        zero_range = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        term = _make_event_term(warp_evt.push_by_setting_velocity, warp_env, mode="interval", velocity_range=zero_range)

        with wp.ScopedCapture() as cap:
            term(warp_env, mask, **term.cfg.params)

        # Mutate root_vel_w
        new_vel = np.tile([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], (NUM_ENVS, 1)).astype(np.float32)
        copy_np_to_wp(art_data.root_vel_w, new_vel)

        wp.capture_launch(cap.graph)
        wp.synchronize()

        written = wp.to_torch(warp_env.scene["robot"].last_root_velocity)
        expected = torch.tensor([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], device=DEVICE).expand(NUM_ENVS, -1)
        assert_close(written, expected)

    # -- apply_external_force_torque --------------------------------------------

    def test_apply_external_force_torque(self, warp_env, art_data, all_joints_cfg):
        """A degenerate non-zero range reaches masked envs through the replayed kernel; the composer gets the mask."""
        mask = wp.array([i < NUM_ENVS // 2 for i in range(NUM_ENVS)], dtype=wp.bool, device=DEVICE)
        term = _make_event_term(
            warp_evt.apply_external_force_torque, warp_env, force_range=(2.0, 2.0), torque_range=(3.0, 3.0)
        )

        with wp.ScopedCapture() as cap:
            term(warp_env, mask, **term.cfg.params)
        composer = warp_env.scene["robot"].permanent_wrench_composer
        forces = wp.to_torch(composer.last_forces)
        torques = wp.to_torch(composer.last_torques)
        wp.capture_launch(cap.graph)
        wp.synchronize()

        half = NUM_ENVS // 2
        assert_close(forces[:half], torch.full_like(forces[:half], 2.0))
        assert_close(torques[:half], torch.full_like(torques[:half], 3.0))
        assert composer.last_env_mask is mask

    def test_apply_external_force_torque_zero_ranges_are_no_op(self, warp_env, art_data, all_joints_cfg):
        """Zero ranges keep the composer's wrenches: a placeholder term must not clear them on every reset."""
        term = _make_event_term(
            warp_evt.apply_external_force_torque, warp_env, force_range=(0.0, 0.0), torque_range=(0.0, 0.0)
        )

        term(warp_env, _all_envs(), **term.cfg.params)

        assert warp_env.scene["robot"].permanent_wrench_composer.last_forces is None

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


# ============================================================================
# Class event terms: persistent state and live configuration
# ============================================================================


def _make_event_env(seed: int, num_bodies: int = 1):
    """An independent Warp event environment with its own asset."""
    data = MockArticulationData(NUM_ENVS, NUM_JOINTS, DEVICE, seed=seed, num_bodies=num_bodies)
    asset = MockArticulation(data, num_bodies=num_bodies)
    origins = wp.zeros(NUM_ENVS, dtype=wp.vec3f, device=DEVICE)
    scene = MockScene({"robot": asset}, origins)
    env = SimpleNamespace(scene=scene, num_envs=NUM_ENVS, device=DEVICE, env_origins_wp=origins)
    WarpRng.state = wp.array(np.arange(NUM_ENVS, dtype=np.uint32) + seed, device=DEVICE)
    return env, data, asset


@pytest.mark.parametrize(
    ("term_type", "params"),
    [
        (warp_evt.push_by_setting_velocity, {"velocity_range": {"x": (1.0, 1.0)}}),
        (warp_evt.apply_external_force_torque, {"force_range": (1.0, 1.0), "torque_range": (2.0, 2.0)}),
        (warp_evt.reset_root_state_uniform, {"pose_range": {}, "velocity_range": {}}),
    ],
    ids=["push", "external_wrench", "root_reset"],
)
def test_class_terms_allocate_only_during_initialization(term_type, params, monkeypatch):
    """A recorded stage runs its terms for the first time inside CUDA graph capture, so a call must not allocate."""
    env, _, _ = _make_event_env(81)
    term = _make_event_term(term_type, env, **params)

    def fail(*args, **kwargs):
        pytest.fail("the term allocated during a call")

    monkeypatch.setattr(wp, "zeros", fail)
    monkeypatch.setattr(wp, "empty", fail)
    monkeypatch.setattr(wp, "clone", fail)
    term(env, _all_envs(), **term.cfg.params)
    term(env, _all_envs(), **term.cfg.params)


def test_push_range_change_applies_on_the_next_call():
    """A curriculum that edits the term's range in place changes the next push, as in the stable term."""
    env, data, asset = _make_event_env(277)
    copy_np_to_wp(data.root_vel_w, np.zeros((NUM_ENVS, 6), dtype=np.float32))
    term = _make_event_term(warp_evt.push_by_setting_velocity, env, mode="interval", velocity_range={"x": (1.0, 1.0)})
    term(env, _all_envs(), **term.cfg.params)

    term.cfg.params["velocity_range"]["x"] = (4.0, 4.0)
    term(env, _all_envs(), **term.cfg.params)
    wp.synchronize()

    assert_close(wp.to_torch(asset.last_root_velocity)[:, 0], torch.full((NUM_ENVS,), 4.0, device=DEVICE))


def test_com_randomization_uses_the_first_call_baseline_without_accumulating():
    """Startup terms may move the CoM after construction; each call offsets that baseline, not the last result."""
    env, data, asset = _make_event_env(84, num_bodies=2)
    baseline = np.zeros((NUM_ENVS, 2, 3), dtype=np.float32)
    copy_np_to_wp(data.body_com_pos_b, baseline)
    asset.set_coms_mask = lambda **kwargs: None
    term = _make_event_term(
        warp_evt.randomize_rigid_body_com,
        env,
        mode="startup",
        com_range={"x": (1.0, 1.0)},
        asset_cfg=SceneEntityCfg("robot", body_ids=[0, 1]),
    )
    # another startup term moves the CoM between construction and the first call
    baseline[..., 0] = 0.75
    copy_np_to_wp(data.body_com_pos_b, baseline)

    term(env, _all_envs(), **term.cfg.params)
    term(env, _all_envs(), **term.cfg.params)
    wp.synchronize()

    assert_close(data.body_com_pos_b.torch[..., 0], torch.full((NUM_ENVS, 2), 1.75, device=DEVICE))


@pytest.mark.parametrize(
    ("term_type", "params"),
    [
        (
            warp_evt.randomize_rigid_body_material,
            {
                "static_friction_range": (0.5, 0.5),
                "dynamic_friction_range": (0.5, 0.5),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 1,
            },
        ),
        (warp_evt.randomize_rigid_body_mass, {"mass_distribution_params": (1.0, 1.0), "operation": "scale"}),
    ],
    ids=["material", "mass"],
)
def test_mask_adapters_run_eagerly_in_reset_mode(term_type, params, monkeypatch):
    """The adapters convert the mask to indices on the host, so reset events run them outside the recorded graph."""
    stable_term_type = term_type.__mro__[1]
    calls = []

    def stable_init(self, cfg, env):
        self.cfg, self._env = cfg, env

    monkeypatch.setattr(stable_term_type, "__init__", stable_init)
    monkeypatch.setattr(stable_term_type, "__call__", lambda self, env, env_ids, *args: calls.append(env_ids.tolist()))
    monkeypatch.setattr(CapturedStage, "enabled", True)
    env, _, _ = _make_event_env(5)
    env.sim = SimpleNamespace(is_playing=lambda: True)
    manager = EventManager(
        {
            "randomize": EventTermCfg(
                func=term_type, mode="reset", params={**params, "asset_cfg": SimpleNamespace(name="robot")}
            )
        },
        env,
    )
    env_mask = wp.array([i % 2 == 0 for i in range(NUM_ENVS)], dtype=wp.bool, device=DEVICE)
    selected = [i for i in range(NUM_ENVS) if i % 2 == 0]

    for _ in range(2):
        manager.apply(
            mode="reset", env_mask_wp=env_mask, global_env_step_count=wp.zeros(1, dtype=wp.int32, device=DEVICE)
        )

    assert calls == [selected, selected], "the adapter must run, eagerly, on every reset"
