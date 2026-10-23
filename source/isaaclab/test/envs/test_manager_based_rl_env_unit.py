# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for manager-based RL environments."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest
import torch

from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import (
    DatasetExportMode,
    ObservationGroupCfg,
    ObservationManager,
    ObservationTermCfg,
    RecorderManager,
    RecorderManagerBaseCfg,
    RecorderTerm,
    RecorderTermCfg,
)
from isaaclab.utils import configclass

pytestmark = pytest.mark.unit


def _make_env_with_policy_obs_terms(
    num_envs: int,
    terms: list[tuple[str, tuple[int, ...], tuple[float, float] | None]],
) -> ManagerBasedRLEnv:
    """Build an uninitialized env whose observation manager stubs drive space setup.

    Args:
        num_envs: Number of vectorized environments.
        terms: Non-concatenated policy observation terms as ``(name, shape, clip)``
            where ``clip`` is ``(low, high)`` or ``None``.

    Returns:
        An uninitialized :class:`~isaaclab.envs.ManagerBasedRLEnv` ready for
        :meth:`~isaaclab.envs.ManagerBasedRLEnv._configure_gym_env_spaces`.
    """
    term_names = [name for name, _, _ in terms]
    term_dims = [shape for _, shape, _ in terms]
    term_cfgs = [SimpleNamespace(clip=clip) for _, _, clip in terms]

    env = object.__new__(ManagerBasedRLEnv)
    env._is_closed = True
    env.scene = SimpleNamespace(num_envs=num_envs)
    env.observation_manager = SimpleNamespace(
        active_terms={"policy": term_names},
        group_obs_concatenate={"policy": False},
        group_obs_dim={"policy": term_dims},
        _group_obs_term_cfgs={"policy": term_cfgs},
    )
    env.action_manager = SimpleNamespace(action_term_dim=[0])
    return env


def test_non_concatenated_obs_groups_contain_all_terms():
    """Non-concatenated observation groups expose every term in the Dict space (issue #3133).

    Before the fix, only the last term in each non-concatenated group would be present
    in the observation space Dict. This test ensures all terms are correctly included.
    """
    terms = [
        ("image", (4, 5, 3), (0.0, 1.0)),
        ("matrix", (2, 3), (-2.0, 2.0)),
        ("vector", (3,), None),
    ]
    env = _make_env_with_policy_obs_terms(num_envs=2, terms=terms)
    ManagerBasedRLEnv._configure_gym_env_spaces(env)

    assert isinstance(env.observation_space, gym.spaces.Dict)
    policy_space = env.observation_space.spaces["policy"]
    assert isinstance(policy_space, gym.spaces.Dict)

    expected_policy_terms = ["image", "matrix", "vector"]
    assert list(policy_space.spaces) == expected_policy_terms
    for term_name in expected_policy_terms:
        assert isinstance(policy_space.spaces[term_name], gym.spaces.Box)


def test_obs_space_follows_clip_constraint():
    """Observation space bounds reflect the clip constraint on each non-concatenated term."""
    terms = [
        ("vector", (3,), None),
        ("matrix", (2, 3), (-2.0, 2.0)),
        ("image", (4, 5, 3), (0.0, 1.0)),
    ]
    expected_shapes = {
        "vector": (2, 3),
        "matrix": (2, 2, 3),
        "image": (2, 4, 5, 3),
    }
    term_clips = {name: clip for name, _, clip in terms}
    env = _make_env_with_policy_obs_terms(num_envs=2, terms=terms)
    ManagerBasedRLEnv._configure_gym_env_spaces(env)

    for group_name, group_space in env.observation_space.spaces.items():
        assert isinstance(group_space, gym.spaces.Dict)
        for term_name, term_space in group_space.spaces.items():
            clip = term_clips[term_name]
            low = -np.inf if clip is None else clip[0]
            high = np.inf if clip is None else clip[1]
            assert isinstance(term_space, gym.spaces.Box), (
                f"Expected Box space for {term_name} in {group_name}, got {type(term_space)}"
            )
            assert term_space.shape == expected_shapes[term_name]
            assert np.all(term_space.low == low)
            assert np.all(term_space.high == high)


def _observe_simulated_states(env):
    return env.simulated_states.unsqueeze(-1)


class EpisodeBoundaryRecorder(RecorderTerm):
    """Expose the episode boundaries seen by a recorder term."""

    def record_pre_reset(self, env_ids):
        if self._env.cfg.autoreset_mode == gym.vector.AutoresetMode.DISABLED:
            assert self._env.active_episode_mask[env_ids].all(), "Record terminal data before retiring episodes"
        for env_id in env_ids:
            self._env.recorded_episode_lengths.append((int(env_id), int(self._env.episode_length_buf[env_id])))
        return "terminal_length", self._env.episode_length_buf[env_ids].unsqueeze(-1)

    def record_post_reset(self, env_ids):
        assert self._env.active_episode_mask[env_ids].all(), "Start episodes before recording their initial state"
        return "initial_state", self._env.simulated_states[env_ids].unsqueeze(-1)


@configclass
class EpisodeObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        simulated_state = ObservationTermCfg(func=_observe_simulated_states)

    policy = PolicyCfg()


@configclass
class EpisodeBoundaryRecordersCfg(RecorderManagerBaseCfg):
    boundary = RecorderTermCfg(class_type=EpisodeBoundaryRecorder)
    dataset_export_mode = DatasetExportMode.EXPORT_NONE


def _make_episode_env(monkeypatch, autoreset_mode=gym.vector.AutoresetMode.DISABLED):
    """Use real episode, observation, and recorder behavior without starting a simulator."""

    def initialize_simulation_double(env, cfg):
        env._is_closed = True
        env.cfg = cfg
        env._sim_step_counter = 0
        env._physics_handles_decimation = False
        env.render_enabled = False
        env.has_rtx_sensors = False
        env.video_recorders = []
        env.extras = {}
        env.obs_buf = {}
        env.simulated_states = torch.zeros(3)
        env.recorded_episode_lengths = []
        env.episode_step_limits = torch.tensor([1, 2, 4])
        env.sim = SimpleNamespace(
            device="cpu",
            is_rendering=False,
            is_playing=lambda: True,
            step=Mock(side_effect=lambda **kwargs: env.simulated_states.add_(1)),
            forward=Mock(),
            get_setting=lambda name: False,
            consume_reset_request=Mock(return_value=False),
        )

        def reset_scene(env_ids):
            env.simulated_states[env_ids] = 0

        def restore_scene(state, env_ids, is_relative):
            env.simulated_states[env_ids] = state["rigid_object"]["object"]["root_pose"][:, 0]

        env.scene = SimpleNamespace(
            num_envs=3,
            write_data_to_sim=Mock(),
            update=Mock(),
            reset=Mock(side_effect=reset_scene),
            reset_to=Mock(side_effect=restore_scene),
        )
        env.action_manager = SimpleNamespace(process_action=Mock(), apply_action=Mock(), reset=Mock(return_value={}))
        env.observation_manager = ObservationManager(EpisodeObservationsCfg(), env)
        env.recorder_manager = RecorderManager(EpisodeBoundaryRecordersCfg(), env)

        def compute_terminations():
            episode_length_reached = env.episode_length_buf >= env.episode_step_limits
            env.termination_manager.terminated = episode_length_reached & torch.tensor([True, False, False])
            env.termination_manager.time_outs = episode_length_reached & torch.tensor([False, True, True])
            return episode_length_reached

        env.termination_manager = SimpleNamespace(
            active_terms=[],
            terminated=torch.zeros(3, dtype=torch.bool),
            time_outs=torch.zeros(3, dtype=torch.bool),
            compute=compute_terminations,
            reset=Mock(return_value={}),
        )
        env.reward_manager = SimpleNamespace(compute=lambda **kwargs: torch.ones(3), reset=Mock(return_value={}))
        env.curriculum_manager = SimpleNamespace(compute=Mock(), reset=Mock(return_value={}))
        env.command_manager = SimpleNamespace(compute=Mock(), reset=Mock(return_value={}))
        env.event_manager = SimpleNamespace(active_terms={}, reset=Mock(return_value={}))

    monkeypatch.setattr(ManagerBasedEnv, "__init__", initialize_simulation_double)
    cfg = SimpleNamespace(
        autoreset_mode=autoreset_mode,
        scene=SimpleNamespace(num_envs=3),
        sim=SimpleNamespace(device="cpu", dt=0.1, render_interval=1),
        decimation=1,
        compute_final_obs=True,
        num_rerenders_on_reset=0,
        wait_for_textures=False,
    )
    return ManagerBasedRLEnv(cfg)


def test_disabled_autoreset_waits_for_explicit_resets(monkeypatch):
    """Uneven episodes finish once; unused environments and completed observations stay inactive."""
    env = _make_episode_env(monkeypatch)
    assert env.active_episode_mask.tolist() == [False, False, False]
    mask_snapshot = env.active_episode_mask
    mask_snapshot[:] = True
    assert not env.active_episode_mask.any()
    selected_envs = slice(0, None, 2)
    env.reset(selected_envs)
    assert env.scene.reset.call_args.args[0] is selected_envs
    assert env.active_episode_mask.tolist() == [True, False, True]
    assert env.recorded_episode_lengths == []

    observations, rewards, terminated, truncated, _ = env.step(torch.zeros(3, 1))
    assert observations["policy"].flatten().tolist() == [1.0, 0.0, 1.0]
    assert rewards.tolist() == [1.0, 0.0, 1.0]
    assert terminated.tolist() == [True, False, False]
    assert not truncated.any()
    assert env.active_episode_mask.tolist() == [False, False, True]

    observations, rewards, terminated, truncated, _ = env.step(torch.zeros(3, 1))
    assert observations["policy"].flatten().tolist() == [1.0, 0.0, 2.0]
    assert rewards.tolist() == [0.0, 0.0, 1.0]
    assert not (terminated | truncated).any()
    assert env.episode_length_buf.tolist() == [1, 0, 2]

    # Reuse one completed environment while the longer episode continues.
    env.reset(torch.tensor([0]))
    assert env.active_episode_mask.tolist() == [True, False, True]
    assert env.recorded_episode_lengths == [(0, 1)]
    assert env.obs_buf["policy"].flatten().tolist() == [0.0, 0.0, 2.0]
    env.step(torch.zeros(3, 1))
    observations, _, terminated, truncated, _ = env.step(torch.zeros(3, 1))
    assert observations["policy"].flatten().tolist() == [1.0, 0.0, 4.0]
    assert not terminated.any()
    assert truncated.tolist() == [False, False, True]
    assert not env.active_episode_mask.any()

    observations, rewards, terminated, truncated, _ = env.step(torch.zeros(3, 1))
    assert observations["policy"].flatten().tolist() == [1.0, 0.0, 4.0]
    assert rewards.tolist() == [0.0, 0.0, 0.0]
    assert not (terminated | truncated).any()
    assert env.episode_length_buf.tolist() == [1, 0, 4]
    assert env.recorded_episode_lengths == [(0, 1), (0, 1), (2, 4)]
    assert env.recorder_manager.exported_failed_episode_count == 3
    assert env.scene.reset.call_count == 2
    assert env.simulated_states.tolist() == [3.0, 5.0, 5.0]


def test_disabled_autoreset_restores_requested_state_order(monkeypatch):
    """State restoration restarts requested episodes without rerecording completed episodes."""
    env = _make_episode_env(monkeypatch)
    env.reset(slice(0, None, 2))
    env.step(torch.zeros(3, 1))
    requested_env_ids = torch.tensor([2, 0])
    root_poses = torch.tensor([[20.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    state = {"rigid_object": {"object": {"root_pose": root_poses, "root_velocity": torch.zeros(2, 6)}}}
    observations, _ = env.reset_to(state, requested_env_ids, is_relative=True)
    assert observations["policy"].flatten().tolist() == [10.0, 0.0, 20.0]
    assert env.active_episode_mask.tolist() == [True, False, True]
    assert env.episode_length_buf.tolist() == [0, 0, 0]
    assert env.recorded_episode_lengths == [(0, 1), (2, 1)]
    assert env.recorder_manager.exported_failed_episode_count == 2


def test_same_step_autoreset_keeps_default_recording_path(monkeypatch):
    """The default resets completed environments immediately and records full batches."""
    env = _make_episode_env(monkeypatch, autoreset_mode=ManagerBasedRLEnvCfg().autoreset_mode)
    assert env.metadata["autoreset_mode"] == gym.vector.AutoresetMode.SAME_STEP
    env.reset()
    env.recorder_manager.record_pre_step = Mock(wraps=env.recorder_manager.record_pre_step)
    observations, _, terminated, truncated, extras = env.step(torch.zeros(3, 1))
    assert observations["policy"].flatten().tolist() == [0.0, 1.0, 1.0]
    assert extras["final_obs"]["policy"].flatten().tolist() == [1.0, 1.0, 1.0]
    assert terminated.tolist() == [True, False, False]
    assert not truncated.any()
    assert env.active_episode_mask.tolist() == [True, True, True]
    assert env.episode_length_buf.tolist() == [0, 1, 1]
    assert env.recorder_manager.record_pre_step.call_args.args in ((), (None,))


def test_disabled_autoreset_rejects_visualizer_reset_before_stepping(monkeypatch):
    """A visualizer reset request cannot interrupt explicit episode ownership."""
    env = _make_episode_env(monkeypatch)
    env.reset()
    env.sim.consume_reset_request.return_value = True
    with pytest.raises(RuntimeError, match=r"Use reset\("):
        env.step(torch.zeros(3, 1))
    assert env.episode_length_buf.tolist() == [0, 0, 0]
    assert env.active_episode_mask.tolist() == [True, True, True]
    assert env.recorded_episode_lengths == []
    assert env.scene.reset.call_count == 1
    env.sim.step.assert_not_called()


@pytest.mark.parametrize("reset_kind", ["reset", "reset_to"])
def test_initial_empty_reset_fails_before_mutating_environment(monkeypatch, reset_kind):
    """An empty initial request cannot initialize observations and must not run reset operations."""
    env = _make_episode_env(monkeypatch)
    empty_env_ids = torch.empty(0, dtype=torch.long)
    with pytest.raises(ValueError, match="empty selection"):
        if reset_kind == "reset":
            env.reset(empty_env_ids)
        else:
            state = {"rigid_object": {"object": {"root_pose": torch.zeros(0, 7), "root_velocity": torch.zeros(0, 6)}}}
            env.reset_to(state, empty_env_ids)
    assert not env.active_episode_mask.any()
    assert env.recorded_episode_lengths == []
    env.scene.reset.assert_not_called()
    env.scene.reset_to.assert_not_called()
    env.sim.forward.assert_not_called()


def test_unsupported_autoreset_mode_fails_before_simulator_initialization(monkeypatch):
    """NEXT_STEP must not silently behave like SAME_STEP or start a simulator."""
    initialize_simulation = Mock(side_effect=AssertionError("Unsupported modes must fail before simulator setup"))
    monkeypatch.setattr(ManagerBasedEnv, "__init__", initialize_simulation)
    cfg = ManagerBasedRLEnvCfg(autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP)
    with pytest.raises(ValueError, match="SAME_STEP and DISABLED"):
        ManagerBasedRLEnv(cfg)
    initialize_simulation.assert_not_called()
