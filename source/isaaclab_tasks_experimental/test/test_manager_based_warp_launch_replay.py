# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for recorded launches in task-specific manager-based Warp MDP terms."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_experimental.managers import CurriculumManager, CurriculumTermCfg
from isaaclab_tasks_experimental.manager_based.classic.cartpole.mdp.rewards import joint_pos_target_l2
from isaaclab_tasks_experimental.manager_based.classic.humanoid.mdp.rewards import progress_reward
from isaaclab_tasks_experimental.manager_based.locomotion.velocity.mdp.curriculums import TerrainLevelsVel
from isaaclab_tasks_experimental.manager_based.locomotion.velocity.mdp.terminations import terrain_out_of_bounds
from isaaclab_tasks_experimental.manager_based.manipulation.reach.mdp.curriculums import ModifyRewardWeight

from isaaclab.terrains import TerrainImporter
from isaaclab.utils.warp import ProxyArray, WarpLaunchCache

from isaaclab_tasks.core.cartpole.mdp.rewards import joint_pos_target_l2 as stable_joint_pos_target_l2
from isaaclab_tasks.core.locomotion.mdp.rewards import progress_reward as stable_progress_reward
from isaaclab_tasks.core.velocity.mdp.terminations import terrain_out_of_bounds as stable_terrain_out_of_bounds

wp.init()
pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA device required")

_DEVICE = "cuda:0"
_NUM_ENVS = 4


class _Scene(dict):
    """Dictionary-backed scene with terrain configuration attributes."""


def _make_launch_cache(mode: str = "replay") -> WarpLaunchCache:
    """Create a launch cache with static-argument validation enabled."""
    return WarpLaunchCache(mode=mode, debug=True, device=_DEVICE)


def _make_terrain_env(
    root_positions: np.ndarray,
    *,
    terrain_type: str,
    size: tuple[float, float] = (4.0, 6.0),
    num_rows: int = 2,
    num_cols: int = 3,
    border_width: float = 1.0,
):
    """Create a lightweight environment for the terrain termination."""
    root_pos_w = wp.array(root_positions, dtype=wp.vec3f, device=_DEVICE)
    asset = SimpleNamespace(data=SimpleNamespace(root_pos_w=ProxyArray(root_pos_w)))
    scene = _Scene(robot=asset)
    scene.cfg = SimpleNamespace(terrain=SimpleNamespace(terrain_type=terrain_type))
    scene.terrain = SimpleNamespace(
        cfg=SimpleNamespace(
            terrain_generator=SimpleNamespace(
                size=size,
                num_rows=num_rows,
                num_cols=num_cols,
                border_width=border_width,
            )
        )
    )
    env = SimpleNamespace(
        scene=scene,
        num_envs=_NUM_ENVS,
        device=_DEVICE,
        _warp_launch=_make_launch_cache(),
    )
    return env, root_pos_w


def _copy_vec3(array: wp.array, values: np.ndarray) -> None:
    """Overwrite a vector array without changing its address."""
    wp.copy(array, wp.array(values, dtype=wp.vec3f, device=_DEVICE))


class _CurriculumRewardManager:
    """Minimal reward manager exposing one persistent Warp weight."""

    def __init__(self):
        self.weight_wp = wp.array([-0.01], dtype=wp.float32, device=_DEVICE)

    def get_term_weight_wp(self, name: str) -> wp.array(dtype=wp.float32):
        assert name == "action_rate"
        return self.weight_wp


def _make_reward_curriculum(mode: str) -> tuple[SimpleNamespace, CurriculumManager]:
    """Create a one-term Reach reward curriculum in the requested launch mode."""
    env = SimpleNamespace(
        num_envs=_NUM_ENVS,
        device=_DEVICE,
        sim=SimpleNamespace(is_playing=lambda: True),
        reward_manager=_CurriculumRewardManager(),
        _global_env_step_count_wp=wp.zeros(1, dtype=wp.int32, device=_DEVICE),
        _warp_launch=_make_launch_cache(mode),
    )
    cfg = CurriculumTermCfg(
        func=ModifyRewardWeight,
        params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500},
    )
    return env, CurriculumManager({"action_rate": cfg}, env)


def _run_reward_curriculum(mode: str) -> tuple[np.ndarray, ...]:
    """Run the threshold and mask sequence for one launch mode."""
    env, manager = _make_reward_curriculum(mode)
    env_mask = wp.array([True, False, True, False], dtype=wp.bool, device=_DEVICE)
    empty_mask = wp.zeros(_NUM_ENVS, dtype=wp.bool, device=_DEVICE)

    env._global_env_step_count_wp.fill_(4500)
    manager.compute(env_mask)
    wp.synchronize()
    at_threshold = env.reward_manager.weight_wp.numpy().copy()
    state_at_threshold = manager._term_states_wp.numpy().copy()

    env._global_env_step_count_wp.fill_(4501)
    manager.compute(empty_mask)
    wp.synchronize()
    after_empty_mask = env.reward_manager.weight_wp.numpy().copy()
    state_after_empty_mask = manager._term_states_wp.numpy().copy()

    manager.compute(env_mask)
    wp.synchronize()
    after_selected_mask = env.reward_manager.weight_wp.numpy().copy()
    state_after_selected_mask = manager._term_states_wp.numpy().copy()
    return (
        at_threshold,
        state_at_threshold,
        after_empty_mask,
        state_after_empty_mask,
        after_selected_mask,
        state_after_selected_mask,
    )


def _run_terrain_curriculum(mode: str) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Run the same persistent terrain term twice in one launch mode."""
    initial_levels = np.array([0, 1, 1, 2], dtype=np.int64)
    terrain = TerrainImporter.__new__(TerrainImporter)
    terrain.device = _DEVICE
    terrain.cfg = SimpleNamespace(terrain_generator=SimpleNamespace(size=(8.0, 8.0)))
    terrain.max_terrain_level = 3
    terrain.terrain_levels = torch.tensor(initial_levels, dtype=torch.int64, device=_DEVICE)
    terrain.terrain_types = torch.tensor([0, 1, 0, 1], dtype=torch.int64, device=_DEVICE)
    terrain.terrain_origins = torch.zeros((3, 2, 3), dtype=torch.float32, device=_DEVICE)
    terrain.env_origins = torch.zeros((_NUM_ENVS, 3), dtype=torch.float32, device=_DEVICE)
    terrain._configure_warp_origin_views()

    root_pos_w = wp.array(
        [[5.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=wp.vec3f,
        device=_DEVICE,
    )
    command_wp = wp.array(
        [[0.1, 0.0, 0.0], [1.0, 0.0, 0.0], [0.1, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=wp.float32,
        device=_DEVICE,
    )
    scene = _Scene(robot=SimpleNamespace(data=SimpleNamespace(root_pos_w=ProxyArray(root_pos_w))))
    scene.device = _DEVICE
    scene.terrain = terrain
    env = SimpleNamespace(
        num_envs=_NUM_ENVS,
        device=_DEVICE,
        scene=scene,
        command_manager=SimpleNamespace(get_command_wp=lambda name: command_wp),
        env_origins_wp=wp.from_torch(terrain.env_origins, dtype=wp.vec3f),
        rng_state_wp=wp.array(np.arange(_NUM_ENVS, dtype=np.uint32) + 101, device=_DEVICE),
        max_episode_length_s=10.0,
        _warp_launch=_make_launch_cache(mode),
    )
    term = TerrainLevelsVel(CurriculumTermCfg(func=TerrainLevelsVel), env)
    env_mask = wp.array([True, True, False, False], dtype=wp.bool, device=_DEVICE)
    out = wp.zeros(1, dtype=wp.float32, device=_DEVICE)

    def _launch_and_read() -> tuple[np.ndarray, ...]:
        out.zero_()
        term(env, env_mask, out)
        wp.synchronize()
        return (
            terrain.terrain_levels_wp.numpy().copy(),
            term._move_up_wp.numpy().copy(),
            term._move_down_wp.numpy().copy(),
            out.numpy().copy(),
        )

    first = _launch_and_read()
    wp.copy(terrain.terrain_levels_wp, wp.array(initial_levels, dtype=wp.int64, device=_DEVICE))
    second = _launch_and_read()
    return first, second


def test_terrain_curriculum_eager_replay_parity_for_masked_selection():
    """Terrain updates should preserve masked semantics after command replay."""
    eager_calls = _run_terrain_curriculum("eager")
    replay_calls = _run_terrain_curriculum("replay")
    expected = (
        np.array([1, 0, 1, 2], dtype=np.int64),
        np.array([True, False, False, False]),
        np.array([False, True, False, False]),
        np.array([1.0], dtype=np.float32),
    )

    for calls in (eager_calls, replay_calls):
        for call in calls:
            for actual, expected_value in zip(call, expected):
                np.testing.assert_allclose(actual, expected_value)
    for eager_call, replay_call in zip(eager_calls, replay_calls):
        for eager_value, replay_value in zip(eager_call, replay_call):
            np.testing.assert_allclose(replay_value, eager_value)


def test_reward_curriculum_eager_replay_parity_for_mask_and_threshold():
    """Reach reward scheduling should retain strict threshold and reset-mask semantics."""
    eager = _run_reward_curriculum("eager")
    replay = _run_reward_curriculum("replay")

    for replay_value, eager_value in zip(replay, eager):
        np.testing.assert_allclose(replay_value, eager_value)
    np.testing.assert_allclose(eager[0], [-0.01])
    np.testing.assert_allclose(eager[1], [-0.01])
    np.testing.assert_allclose(eager[2], [-0.01])
    np.testing.assert_allclose(eager[3], [0.0])
    np.testing.assert_allclose(eager[4], [-0.005])
    np.testing.assert_allclose(eager[5], [-0.005])


def test_reward_curriculum_recorded_launches_compose_with_capture():
    """A warmed Reach curriculum should replay both launches inside CUDA capture."""
    env, manager = _make_reward_curriculum("replay")
    env_mask = wp.array([True, False, True, False], dtype=wp.bool, device=_DEVICE)

    env._global_env_step_count_wp.fill_(4500)
    manager.compute(env_mask)
    wp.synchronize()
    env._global_env_step_count_wp.fill_(4501)

    with wp.ScopedCapture() as capture:
        manager.compute(env_mask)
    wp.capture_launch(capture.graph)
    wp.synchronize()

    np.testing.assert_allclose(env.reward_manager.weight_wp.numpy(), [-0.005])
    np.testing.assert_allclose(manager._term_states_wp.numpy(), [-0.005])


def test_terrain_out_of_bounds_replays_inside_capture():
    """Terrain termination should replay with fresh data inside CUDA capture."""
    positions = np.array([[0.0, 0.0, 0.0], [4.1, 0.0, 0.0], [0.0, 9.1, 0.0], [-3.0, -8.0, 0.0]], dtype=np.float32)
    env, root_pos_w = _make_terrain_env(positions, terrain_type="generator")
    out = wp.zeros(_NUM_ENVS, dtype=wp.bool, device=_DEVICE)
    asset_cfg = SimpleNamespace(name="robot")

    terrain_out_of_bounds(env, out, asset_cfg=asset_cfg, distance_buffer=1.0)
    torch.testing.assert_close(
        wp.to_torch(out), stable_terrain_out_of_bounds(env, asset_cfg=asset_cfg, distance_buffer=1.0)
    )

    replay_positions = np.array([[4.2, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, -9.2, 0.0], [3.5, 8.5, 0.0]], dtype=np.float32)
    _copy_vec3(root_pos_w, replay_positions)
    terrain_out_of_bounds(env, out, asset_cfg=asset_cfg, distance_buffer=1.0)
    torch.testing.assert_close(
        wp.to_torch(out), stable_terrain_out_of_bounds(env, asset_cfg=asset_cfg, distance_buffer=1.0)
    )

    with wp.ScopedCapture() as capture:
        terrain_out_of_bounds(env, out, asset_cfg=asset_cfg, distance_buffer=1.0)
    captured_positions = np.array(
        [[0.0, 9.5, 0.0], [-4.5, 0.0, 0.0], [0.0, 0.0, 0.0], [4.5, 9.5, 0.0]], dtype=np.float32
    )
    _copy_vec3(root_pos_w, captured_positions)
    wp.capture_launch(capture.graph)
    torch.testing.assert_close(
        wp.to_torch(out), stable_terrain_out_of_bounds(env, asset_cfg=asset_cfg, distance_buffer=1.0)
    )


def test_terrain_out_of_bounds_does_not_share_configuration_between_envs():
    """Terrain geometry from one environment should not leak into another."""
    positions = np.full((_NUM_ENVS, 3), 100.0, dtype=np.float32)
    generator_env, _ = _make_terrain_env(positions, terrain_type="generator")
    plane_env, _ = _make_terrain_env(positions, terrain_type="plane")
    asset_cfg = SimpleNamespace(name="robot")
    generator_out = wp.zeros(_NUM_ENVS, dtype=wp.bool, device=_DEVICE)
    plane_out = wp.ones(_NUM_ENVS, dtype=wp.bool, device=_DEVICE)

    terrain_out_of_bounds(generator_env, generator_out, asset_cfg=asset_cfg)
    terrain_out_of_bounds(plane_env, plane_out, asset_cfg=asset_cfg)

    torch.testing.assert_close(wp.to_torch(generator_out), stable_terrain_out_of_bounds(generator_env, asset_cfg))
    torch.testing.assert_close(wp.to_torch(plane_out), stable_terrain_out_of_bounds(plane_env, asset_cfg))


def test_progress_reward_replays_stateful_reset_and_compute():
    """The stateful progress term should replay against persistent state buffers."""
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [3.0, 4.0, 0.0]], dtype=np.float32)
    root_pos_w = wp.array(positions, dtype=wp.vec3f, device=_DEVICE)
    asset = SimpleNamespace(data=SimpleNamespace(root_pos_w=ProxyArray(root_pos_w)))
    env = SimpleNamespace(
        scene={"robot": asset},
        num_envs=_NUM_ENVS,
        device=_DEVICE,
        step_dt=0.1,
        _warp_launch=_make_launch_cache(),
        termination_manager=SimpleNamespace(time_outs=torch.zeros(_NUM_ENVS, dtype=torch.bool, device=_DEVICE)),
        extras={},
    )
    target = (3.0, 4.0, 0.0)
    cfg = SimpleNamespace(params={"target_pos": target})
    term = progress_reward(cfg, env)
    stable_term = stable_progress_reward(env, cfg)
    env_mask = wp.array([True, False, True, False], dtype=wp.bool, device=_DEVICE)
    env_ids = torch.tensor([0, 2], dtype=torch.int64, device=_DEVICE)
    asset_cfg = SimpleNamespace(name="robot")

    term.reset(env_mask=env_mask)
    stable_term.reset(env_ids)
    torch.testing.assert_close(wp.to_torch(term.potentials), stable_term.potentials)

    reset_positions = np.array([[3.0, 4.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 4.0, 0.0]], dtype=np.float32)
    _copy_vec3(root_pos_w, reset_positions)
    term.reset(env_mask=env_mask)
    stable_term.reset(env_ids)
    torch.testing.assert_close(wp.to_torch(term.potentials), stable_term.potentials)

    out = wp.zeros(_NUM_ENVS, dtype=wp.float32, device=_DEVICE)
    term(env, out, target_pos=target, asset_cfg=asset_cfg)
    torch.testing.assert_close(wp.to_torch(out), stable_term(env, target_pos=target, asset_cfg=asset_cfg))

    next_positions = np.array([[2.0, 4.0, 0.0], [2.0, 4.0, 0.0], [3.0, 2.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    _copy_vec3(root_pos_w, next_positions)
    term(env, out, target_pos=target, asset_cfg=asset_cfg)
    torch.testing.assert_close(wp.to_torch(out), stable_term(env, target_pos=target, asset_cfg=asset_cfg))


def test_cartpole_reward_records_static_target_variants():
    """Changing a static Cartpole target should select a separate recorded launch."""
    joint_pos = wp.array([[0.0], [2.0], [-1.0], [3.0]], dtype=wp.float32, device=_DEVICE)
    joint_mask = wp.array([True], dtype=wp.bool, device=_DEVICE)
    out = wp.zeros(_NUM_ENVS, dtype=wp.float32, device=_DEVICE)
    asset = SimpleNamespace(data=SimpleNamespace(joint_pos=ProxyArray(joint_pos)))
    env = SimpleNamespace(
        scene={"robot": asset},
        num_envs=_NUM_ENVS,
        device=_DEVICE,
        _warp_launch=_make_launch_cache(),
    )
    asset_cfg = SimpleNamespace(name="robot", joint_mask=joint_mask, joint_ids=[0])

    joint_pos_target_l2(env, out, target=0.0, asset_cfg=asset_cfg)
    torch.testing.assert_close(wp.to_torch(out), stable_joint_pos_target_l2(env, target=0.0, asset_cfg=asset_cfg))

    joint_pos_target_l2(env, out, target=1.0, asset_cfg=asset_cfg)
    torch.testing.assert_close(wp.to_torch(out), stable_joint_pos_target_l2(env, target=1.0, asset_cfg=asset_cfg))
