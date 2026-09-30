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

from isaaclab.envs import ManagerBasedRLEnv

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


class FiniteEpisodeTestEnv(ManagerBasedRLEnv):
    """Exercise the public environment lifecycle with physics and managers replaced by test doubles."""

    def __init__(self, episode_limit: int):
        self._is_closed = True
        self.episode_limit = episode_limit
        self.num_episodes_started = 0
        self.completed_episode_env_ids = []
        self.reset_env_ids_by_call = []
        self.assigned_episode_mask = torch.zeros(3, dtype=torch.bool)
        self.episode_length_buf = torch.zeros(3, dtype=torch.long)
        self.episode_step_limits = torch.tensor([1, 2, 4])
        self.common_step_counter = 0
        self._sim_step_counter = 0
        self._physics_handles_decimation = False
        self.render_enabled = False
        self.has_rtx_sensors = False
        self.video_recorders = []
        self.extras = {}
        self.cfg = SimpleNamespace(
            decimation=1,
            sim=SimpleNamespace(dt=0.1, render_interval=1),
            compute_final_obs=True,
            num_rerenders_on_reset=0,
            wait_for_textures=False,
        )
        self.sim = SimpleNamespace(
            device="cpu", is_rendering=False, step=Mock(), forward=Mock(), consume_reset_request=lambda: False
        )
        self.scene = SimpleNamespace(num_envs=3, write_data_to_sim=Mock(), update=Mock())
        self.action_manager = SimpleNamespace(process_action=Mock(), apply_action=Mock())
        self.observation_manager = SimpleNamespace(compute=lambda **kwargs: self.episode_length_buf.clone())
        self.recorder_manager = Mock(active_terms=["trajectory"])
        self.termination_manager = SimpleNamespace(
            terminated=torch.zeros(3, dtype=torch.bool),
            time_outs=torch.zeros(3, dtype=torch.bool),
            compute=self._compute_terminations,
        )
        self.reward_manager = SimpleNamespace(compute=lambda **kwargs: torch.ones(3))
        self.command_manager = SimpleNamespace(compute=Mock())
        self.event_manager = SimpleNamespace(available_modes=[])

    @property
    def active_episode_mask(self):
        return self.assigned_episode_mask

    def _validate_reset_request(self, reset_kind):
        if self.num_episodes_started or reset_kind != "reset":
            raise RuntimeError("Finite evaluation does not accept external resets")

    def _select_episode_start_env_ids(self, candidate_env_ids):
        episode_start_env_ids = candidate_env_ids.sort().values[: self.episode_limit - self.num_episodes_started]
        self.num_episodes_started += len(episode_start_env_ids)
        self.assigned_episode_mask[episode_start_env_ids] = True
        return episode_start_env_ids

    def _finish_episodes(self, env_ids):
        completed_env_ids = env_ids[self.assigned_episode_mask[env_ids]]
        if len(completed_env_ids) == 0:
            return
        super()._finish_episodes(completed_env_ids)
        self.completed_episode_env_ids.extend(completed_env_ids.tolist())
        self.assigned_episode_mask[completed_env_ids] = False

    def _reset_idx(self, env_ids):
        self.reset_env_ids_by_call.append(env_ids.tolist())
        self.episode_length_buf[env_ids] = 0
        # Manager resets must not erase completion flags returned by this step.
        self.termination_manager.terminated[env_ids] = False
        self.termination_manager.time_outs[env_ids] = False

    def _compute_terminations(self):
        episode_length_reached = self.episode_length_buf >= self.episode_step_limits
        self.termination_manager.terminated.copy_(episode_length_reached & torch.tensor([True, False, False]))
        self.termination_manager.time_outs.copy_(episode_length_reached & torch.tensor([False, True, True]))
        return episode_length_reached


@pytest.mark.parametrize("physics_handles_decimation", [False, True])
def test_finite_episode_limit_completes_active_episodes(physics_handles_decimation):
    """Completed episodes emit no duplicate done signals while the slowest episode completes."""
    env = FiniteEpisodeTestEnv(episode_limit=4)
    env._physics_handles_decimation = physics_handles_decimation
    env.reset()
    step_results = [env.step(torch.zeros(3, 1)) for _ in range(5)]

    assert env.num_episodes_started == len(env.completed_episode_env_ids) == 4
    assert env.reset_env_ids_by_call == [[0, 1, 2], [0]]
    assert [terminated.tolist() for _, _, terminated, _, _ in step_results] == [
        [True, False, False],
        [True, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
    ]
    assert [truncated.tolist() for _, _, _, truncated, _ in step_results] == [
        [False, False, False],
        [False, True, False],
        [False, False, False],
        [False, False, True],
        [False, False, False],
    ]
    assert [call.args[0].tolist() for call in env.recorder_manager.record_pre_step.call_args_list] == [
        [0, 1, 2],
        [0, 1, 2],
        [2],
        [2],
        [],
    ]
    assert [call.args[0].tolist() for call in env.recorder_manager.record_post_reset.call_args_list] == [
        [0, 1, 2],
        [0],
    ]
    assert step_results[-1][1].tolist() == [0.0, 0.0, 0.0]
    assert env.episode_length_buf.tolist() == [4, 5, 5]


def test_initial_reset_can_leave_environments_unused():
    """A smaller episode limit does not initialize unused environments or record their later timeouts."""
    env = FiniteEpisodeTestEnv(episode_limit=1)
    env.reset()
    for _ in range(5):
        env.step(torch.zeros(3, 1))
    assert env.reset_env_ids_by_call == [[0]]
    assert env.completed_episode_env_ids == [0]
    assert env.recorder_manager.record_pre_reset.call_count == 1


@pytest.mark.parametrize("env_ids", [[1, 2], [2, 0]])
def test_state_restoration_starts_replacement_episodes(env_ids):
    """An evaluator allowing explicit state restoration assigns episodes to the restored environments."""
    env = FiniteEpisodeTestEnv(episode_limit=5)
    env._validate_reset_request = Mock()
    env.scene.reset_to = Mock()
    env.reset()
    env.reset_to({}, env_ids=env_ids)
    assert env.num_episodes_started == 5
    assert env.completed_episode_env_ids == env_ids
    assert env.active_episode_mask.tolist() == [True, True, True]
    assert env.reset_env_ids_by_call == [[0, 1, 2], env_ids]
    assert env.scene.reset_to.call_args.args[1].tolist() == env_ids


@pytest.mark.parametrize("reset_kind", ["reset", "reset_to", "manual"])
def test_external_reset_guard_prevents_new_episode_starts(reset_kind):
    """Every public or visualizer reset passes validation before resetting an environment."""
    env = FiniteEpisodeTestEnv(episode_limit=4)
    env.reset()
    env.episode_step_limits[:] = 10
    with pytest.raises(RuntimeError, match="external resets"):
        if reset_kind == "reset":
            env.reset()
        elif reset_kind == "reset_to":
            env.reset_to({}, env_ids=None)
        else:
            env.sim.consume_reset_request = lambda: True
            env.step(torch.zeros(3, 1))
    assert env.reset_env_ids_by_call == [[0, 1, 2]]
    assert env.completed_episode_env_ids == []
