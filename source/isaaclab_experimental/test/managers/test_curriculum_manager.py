# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Warp-first curriculum manager and terrain-level update path."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_experimental.managers import CurriculumManager, CurriculumTermCfg
from isaaclab_tasks_experimental.manager_based.locomotion.velocity.mdp import TerrainLevelsVel, terrain_levels_vel
from isaaclab_tasks_experimental.manager_based.manipulation.reach.reach_env_cfg import (
    CurriculumCfg as ReachCurriculumCfg,
)

from isaaclab.managers import ManagerTermBase as StableManagerTermBase
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.warp import ProxyArray


class _GlobalCurriculumTerm(StableManagerTermBase):
    """Legacy global term that intentionally ignores compact environment IDs."""

    def __call__(self, env, env_ids, value: float) -> float:
        assert env_ids == slice(None)
        return value


class _LegacyIdCurriculumTerm(StableManagerTermBase):
    """Legacy term that records the compact IDs received by compute and reset."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.compute_env_ids = None
        self.reset_env_ids = None

    def __call__(self, env, env_ids, scale: float) -> float:
        del env
        self.compute_env_ids = env_ids.clone()
        return len(env_ids) * scale

    def reset(self, env_ids=None) -> None:
        self.reset_env_ids = env_ids.clone()


def _structured_state(env, env_ids) -> dict[str, float]:
    """Legacy curriculum state unsupported by the scalar Warp logging buffer."""
    del env, env_ids
    return {"value": 1.0}


class _ArrayProxy:
    """Minimal Torch/Warp array proxy."""

    def __init__(self, values: np.ndarray, dtype):
        self.warp = wp.array(values, dtype=dtype, device="cpu")
        self.torch = wp.to_torch(self.warp)


class _Robot:
    """Minimal robot exposing root positions."""

    def __init__(self, root_positions: np.ndarray):
        self.cfg = SimpleNamespace()
        self.data = SimpleNamespace(root_pos_w=_ArrayProxy(root_positions, wp.vec3f))


class _CommandManager:
    """Minimal command manager exposing persistent Warp storage."""

    def __init__(self, commands: np.ndarray):
        self.command_wp = wp.array(commands, dtype=wp.float32, device="cpu")

    def get_command_wp(self, name: str) -> wp.array(dtype=wp.float32, ndim=2):
        assert name == "base_velocity"
        return self.command_wp


class _RewardManager:
    """Minimal reward manager used by the registered Reach curricula."""

    def __init__(self):
        self._term_names = ["action_rate", "joint_vel"]
        self._cfgs = {
            "action_rate": SimpleNamespace(weight=-0.01),
            "joint_vel": SimpleNamespace(weight=-0.0001),
        }
        self._weights_wp = wp.array([-0.01, -0.0001], dtype=wp.float32, device="cpu")
        stride = self._weights_wp.strides[0]
        self._weight_views_wp = {
            name: wp.array(
                ptr=self._weights_wp.ptr + term_idx * stride,
                dtype=wp.float32,
                shape=(1,),
                strides=(stride,),
                device="cpu",
            )
            for term_idx, name in enumerate(self._term_names)
        }

    def get_term_cfg(self, name: str):
        return self._cfgs[name]

    def set_term_cfg(self, name: str, cfg) -> None:
        self._cfgs[name] = cfg

    def get_term_weight_wp(self, name: str) -> wp.array(dtype=wp.float32):
        return self._weight_views_wp[name]


class _Scene:
    """Minimal scene mapping with terrain-backed environment origins."""

    def __init__(self, robot: _Robot, terrain: TerrainImporter):
        self._entities = {"robot": robot}
        self.terrain = terrain
        self.device = "cpu"

    @property
    def env_origins(self) -> torch.Tensor:
        return self.terrain.env_origins.torch

    @property
    def env_origins_wp(self) -> wp.array(dtype=wp.vec3f):
        return self.terrain.env_origins.warp

    def __getitem__(self, name: str):
        return self._entities[name]

    def keys(self):
        return self._entities.keys()


class _Env:
    """Minimal environment for curriculum manager execution."""

    def __init__(self, terrain: TerrainImporter, root_positions: np.ndarray, commands: np.ndarray):
        self.num_envs = root_positions.shape[0]
        self.device = "cpu"
        self.scene = _Scene(_Robot(root_positions), terrain)
        self.command_manager = _CommandManager(commands)
        self.rng_state_wp = wp.array(np.arange(self.num_envs, dtype=np.uint32) + 101, device=self.device)
        self.max_episode_length_s = 10.0
        self.sim = SimpleNamespace(is_playing=lambda: True)

    @property
    def env_origins_wp(self) -> wp.array(dtype=wp.vec3f):
        """Return the scene's persistent Warp origin view."""
        return self.scene.env_origins_wp


def _make_terrain(levels: list[int]) -> TerrainImporter:
    num_envs = len(levels)
    terrain = TerrainImporter.__new__(TerrainImporter)
    terrain.device = "cpu"
    terrain.cfg = SimpleNamespace(
        num_envs=num_envs,
        max_init_terrain_level=2,
        terrain_generator=SimpleNamespace(size=(8.0, 8.0)),
    )
    level_values = np.asarray(levels, dtype=np.int64)
    type_values = np.asarray([index % 2 for index in range(num_envs)], dtype=np.int64)
    origin_values = np.zeros((3, 2, 3), dtype=np.float32)
    for level in range(3):
        for terrain_type in range(2):
            origin_values[level, terrain_type] = [
                100.0 * level + 10.0 * terrain_type,
                float(level),
                float(terrain_type),
            ]
    terrain.configure_env_origins(origin_values)
    terrain.terrain_levels.torch.copy_(torch.from_numpy(level_values))
    terrain.terrain_types.torch.copy_(torch.from_numpy(type_values))
    terrain.env_origins.torch.copy_(
        terrain.terrain_origins.torch[terrain.terrain_levels.torch, terrain.terrain_types.torch]
    )
    return terrain


def test_terrain_importer_exposes_persistent_proxy_array_views():
    """Canonical terrain storage should expose persistent zero-copy Torch and Warp views."""
    terrain = _make_terrain([0, 1, 2, 1])

    assert isinstance(terrain.terrain_origins, ProxyArray)
    assert isinstance(terrain.env_origins, ProxyArray)
    assert isinstance(terrain.terrain_levels, ProxyArray)
    assert isinstance(terrain.terrain_types, ProxyArray)
    assert terrain.terrain_origins.dtype == wp.vec3f
    assert terrain.env_origins.dtype == wp.vec3f
    assert terrain.terrain_levels.dtype == wp.int64
    assert terrain.terrain_types.dtype == wp.int64
    assert terrain.terrain_origins.torch.data_ptr() == terrain.terrain_origins.warp.ptr
    assert terrain.env_origins.torch.data_ptr() == terrain.env_origins.warp.ptr
    assert terrain.terrain_levels.torch.data_ptr() == terrain.terrain_levels.warp.ptr
    assert terrain.terrain_types.torch.data_ptr() == terrain.terrain_types.warp.ptr

    terrain.terrain_levels.torch[0] = 2
    assert terrain.terrain_levels.warp.numpy()[0] == 2
    replacement_levels = wp.array([1, 0, 2, 1], dtype=wp.int64, device="cpu")
    wp.copy(terrain.terrain_levels.warp, replacement_levels)
    wp.synchronize()
    torch.testing.assert_close(terrain.terrain_levels.torch, torch.tensor([1, 0, 2, 1]))


def test_terrain_importer_grid_origins_use_proxy_array_without_curriculum_buffers():
    """Grid origins should use the same proxy contract without curriculum buffers."""
    terrain = TerrainImporter.__new__(TerrainImporter)
    terrain.device = "cpu"
    terrain.cfg = SimpleNamespace(num_envs=4, env_spacing=2.0)
    terrain.configure_env_origins()

    assert terrain.terrain_origins is None
    assert isinstance(terrain.env_origins, ProxyArray)
    assert terrain.env_origins.shape == (4,)
    assert terrain.env_origins.torch.shape == (4, 3)
    assert terrain.env_origins.torch.data_ptr() == terrain.env_origins.warp.ptr
    assert terrain.terrain_levels is None
    assert terrain.terrain_types is None


def test_terrain_importer_mask_update_preserves_sparse_and_stable_semantics():
    """Mask updates should handle up/down/clamp/wrap while stable IDs remain supported."""
    terrain = _make_terrain([0, 1, 2, 2, 1, 0])
    env_mask = wp.array([True, True, True, True, False, False], dtype=wp.bool, device="cpu")
    move_up = wp.array([False, True, True, False, True, False], dtype=wp.bool, device="cpu")
    move_down = wp.array([True, False, False, True, False, True], dtype=wp.bool, device="cpu")
    rng_state = wp.array(np.arange(6, dtype=np.uint32) + 71, device="cpu")
    rng_before = rng_state.numpy().copy()
    origins_before = terrain.env_origins.torch.clone()
    levels_wp = terrain.terrain_levels.warp
    origins_wp = terrain.env_origins.warp
    levels_ptr = levels_wp.ptr
    origins_ptr = origins_wp.ptr
    assert terrain.terrain_levels.torch.data_ptr() == levels_ptr
    assert terrain.env_origins.torch.data_ptr() == origins_ptr

    terrain.update_env_origins_mask(env_mask, move_up, move_down, rng_state)
    wp.synchronize()

    levels = terrain.terrain_levels.torch.tolist()
    assert levels[0] == 0
    assert levels[1] == 2
    assert 0 <= levels[2] < terrain.max_terrain_level
    assert levels[3] == 1
    assert levels[4:] == [1, 0]
    torch.testing.assert_close(
        terrain.env_origins.torch[:4],
        terrain.terrain_origins.torch[terrain.terrain_levels.torch[:4], terrain.terrain_types.torch[:4]],
    )
    torch.testing.assert_close(terrain.env_origins.torch[4:], origins_before[4:])
    assert np.all(rng_state.numpy()[:4] != rng_before[:4])
    np.testing.assert_array_equal(rng_state.numpy()[4:], rng_before[4:])
    assert terrain.terrain_levels.warp is levels_wp
    assert terrain.env_origins.warp is origins_wp
    assert terrain.terrain_levels.warp.ptr == levels_ptr
    assert terrain.env_origins.warp.ptr == origins_ptr

    levels_before_stable_update = terrain.terrain_levels.torch.clone()
    env_ids = torch.tensor([1, 3], dtype=torch.int64)
    terrain.update_env_origins(
        env_ids,
        move_up=torch.tensor([False, False]),
        move_down=torch.tensor([True, True]),
    )
    expected_levels = levels_before_stable_update.clone()
    expected_levels[env_ids] -= 1
    torch.testing.assert_close(terrain.terrain_levels.torch, expected_levels)
    assert terrain.terrain_levels.warp is levels_wp
    assert terrain.env_origins.warp is origins_wp
    assert terrain.terrain_levels.warp.ptr == levels_ptr
    assert terrain.env_origins.warp.ptr == origins_ptr


def test_curriculum_manager_updates_masked_levels_and_logs_without_host_compaction(monkeypatch: pytest.MonkeyPatch):
    """Manager compute/reset should remain mask-native and expose persistent scalar logging."""
    assert terrain_levels_vel is not TerrainLevelsVel
    terrain = _make_terrain([0, 1, 2, 2, 1, 0])
    root_positions = terrain.env_origins.torch.numpy().copy()
    root_positions[:, 0] += np.array([5.0, 1.0, 3.0, 5.0, 9.0, 0.0], dtype=np.float32)
    commands = np.array(
        [[0.1, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 0.0, 0.0], [2.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.0, 0.0]],
        dtype=np.float32,
    )
    env = _Env(terrain, root_positions, commands)
    manager = CurriculumManager(
        {
            "terrain_levels": CurriculumTermCfg(func=TerrainLevelsVel),
            "global_state": CurriculumTermCfg(
                func=_GlobalCurriculumTerm,
                params={"value": 7.0},
                requires_host_ids=False,
            ),
        },
        env,
    )
    term = manager._term_cfgs[0].func
    env_mask = wp.array([True, True, True, True, False, False], dtype=wp.bool, device="cpu")
    move_up_ptr = term._move_up_wp.ptr
    move_down_ptr = term._move_down_wp.ptr
    state_ptr = manager._term_states_wp.ptr
    extras_ref = manager.reset_extras
    assert not manager.requires_host_ids
    assert manager.requires_host_boundary

    def _fail_host_compaction(*args, **kwargs):
        raise AssertionError("Torch host compaction/scalar extraction reached the Warp curriculum path")

    with monkeypatch.context() as context:
        context.setattr(torch.Tensor, "nonzero", _fail_host_compaction)
        context.setattr(torch.Tensor, "item", _fail_host_compaction)
        manager.compute(env_mask)
        extras = manager.reset(env_mask)
        wp.synchronize()

    levels = terrain.terrain_levels.torch.tolist()
    assert levels[0] == 1
    assert levels[1] == 0
    assert levels[2] == 2
    assert 0 <= levels[3] < terrain.max_terrain_level
    assert levels[4:] == [1, 0]
    np.testing.assert_array_equal(term._move_up_wp.numpy(), [True, False, False, True, False, False])
    np.testing.assert_array_equal(term._move_down_wp.numpy(), [False, True, False, False, False, False])
    assert extras is extras_ref
    assert extras["Curriculum/terrain_levels"] is manager.reset_extras["Curriculum/terrain_levels"]
    torch.testing.assert_close(extras["Curriculum/terrain_levels"], terrain.terrain_levels.torch.float().mean())
    torch.testing.assert_close(extras["Curriculum/global_state"], torch.tensor(7.0))
    assert term._move_up_wp.ptr == move_up_ptr
    assert term._move_down_wp.ptr == move_down_ptr
    assert manager._term_states_wp.ptr == state_ptr


def test_curriculum_manager_compacts_ids_inside_legacy_term_boundary():
    """Legacy terms should receive compact IDs without polluting the environment reset API."""
    env = SimpleNamespace(
        num_envs=4,
        device="cpu",
        sim=SimpleNamespace(is_playing=lambda: True),
    )
    manager = CurriculumManager(
        {"id_count": CurriculumTermCfg(func=_LegacyIdCurriculumTerm, params={"scale": 2.0})},
        env,
    )
    env_mask = wp.array([True, False, True, False], dtype=wp.bool, device="cpu")
    term = manager._term_cfgs[0].func

    assert manager.requires_host_ids
    assert manager.requires_host_boundary
    manager.compute(env_mask)
    extras = manager.reset(env_mask)

    torch.testing.assert_close(term.compute_env_ids, torch.tensor([0, 2]))
    torch.testing.assert_close(term.reset_env_ids, torch.tensor([0, 2]))
    torch.testing.assert_close(extras["Curriculum/id_count"], torch.tensor(4.0))


def test_registered_reach_curricula_update_weights_without_a_host_boundary():
    """Registered Reach reward schedules should update only on a device-selected reset."""
    terrain = _make_terrain([0, 1, 2, 2])
    env = _Env(
        terrain,
        terrain.env_origins.torch.numpy().copy(),
        np.zeros((4, 3), dtype=np.float32),
    )
    env.reward_manager = _RewardManager()
    env.common_step_counter = 0
    manager = CurriculumManager(ReachCurriculumCfg(), env)
    env_mask = wp.array([True, False, True, False], dtype=wp.bool, device="cpu")

    assert not manager.requires_host_ids
    assert not manager.requires_host_boundary
    manager.compute(env_mask)
    extras = manager.reset(env_mask)
    torch.testing.assert_close(extras["Curriculum/action_rate"], torch.tensor(-0.01))
    torch.testing.assert_close(extras["Curriculum/joint_vel"], torch.tensor(-0.0001))

    env.common_step_counter = 4501
    empty_mask = wp.zeros(4, dtype=wp.bool, device="cpu")
    manager.compute(empty_mask)
    np.testing.assert_allclose(env.reward_manager._weights_wp.numpy(), [-0.01, -0.0001])

    manager.compute(env_mask)
    extras = manager.reset(env_mask)
    torch.testing.assert_close(extras["Curriculum/action_rate"], torch.tensor(-0.005))
    torch.testing.assert_close(extras["Curriculum/joint_vel"], torch.tensor(-0.001))
    np.testing.assert_allclose(env.reward_manager._weights_wp.numpy(), [-0.005, -0.001])


def test_curriculum_manager_rejects_structured_legacy_state() -> None:
    """Unsupported fallback logging should fail explicitly instead of narrowing silently."""
    terrain = _make_terrain([0, 1])
    env = _Env(terrain, terrain.env_origins.torch.numpy().copy(), np.zeros((2, 3), dtype=np.float32))
    manager = CurriculumManager(
        {"structured": CurriculumTermCfg(func=_structured_state, requires_host_ids=False)},
        env,
    )
    env_mask = wp.array([True, False], dtype=wp.bool, device="cpu")

    with pytest.raises(TypeError, match="scalar logging states"):
        manager.compute(env_mask)
