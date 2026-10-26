# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for manager-based RL environments."""

from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from isaaclab.envs import ManagerBasedRLEnv
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
from isaaclab.utils.noise import UniformNoiseCfg

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


def _state_observation(env):
    return env.scene.state


class _PhysicalRecorder(RecorderTerm):
    requires_post_step_observations = False

    def record_post_step(self):
        self.latest = self._env.scene.state.clone()
        return None, None


class _ObservationRecorder(RecorderTerm):
    def record_post_step(self):
        self.latest = self._env.obs_buf["policy"].clone()
        return None, None


def _make_stepping_env(recorder_type=None):
    """Exercise the real step/observation/recorder owners with a minimal deterministic physics leaf."""
    env = object.__new__(ManagerBasedRLEnv)
    env._is_closed = True
    env.cfg = SimpleNamespace(decimation=1, sim=SimpleNamespace(dt=0.02, render_interval=1), compute_final_obs=False)
    env.scene = SimpleNamespace(num_envs=3, state=torch.zeros(3, 2))
    env.scene.write_data_to_sim = lambda: None
    env.scene.update = lambda dt: None
    env.sim = SimpleNamespace(
        device="cpu",
        is_playing=lambda: True,
        is_rendering=False,
        step=lambda render: env.scene.state.add_(1),
        consume_reset_request=lambda: False,
    )
    env._physics_handles_decimation = False
    env._sim_step_counter = env.common_step_counter = 0
    env.episode_length_buf = torch.zeros(3, dtype=torch.long)
    env.render_enabled = False
    env.video_recorders = []
    env.extras = {}
    env.action_manager = SimpleNamespace(process_action=lambda action: None, apply_action=lambda: None)
    mask = torch.zeros(3, dtype=torch.bool)
    env.termination_manager = SimpleNamespace(compute=lambda: mask, terminated=mask, time_outs=mask)
    env.reward_manager = SimpleNamespace(compute=lambda dt: torch.zeros(3))
    env.command_manager = SimpleNamespace(compute=lambda dt: None)
    env.event_manager = SimpleNamespace(active_terms=[])
    group = ObservationGroupCfg(history_length=3, flatten_history_dim=False, enable_corruption=True)
    group.state = ObservationTermCfg(func=_state_observation, noise=UniformNoiseCfg(n_min=-0.1, n_max=0.1))
    env.observation_manager = ObservationManager({"policy": group}, env)
    cfg = RecorderManagerBaseCfg(dataset_export_mode=DatasetExportMode.EXPORT_NONE)
    if recorder_type is not None:
        cfg.capture = RecorderTermCfg(class_type=recorder_type)
    env.recorder_manager = RecorderManager(cfg, env)
    env.obs_buf = env.observation_manager.compute(update_history=True)
    return env


def test_raw_capture_preserves_observation_noise_history_and_rng():
    """Raw capture dispatch must not consume extra noise or advance the real policy history."""
    outputs, states = [], []
    for recorder_type in (None, _PhysicalRecorder, _ObservationRecorder):
        torch.manual_seed(17)
        env = _make_stepping_env(recorder_type)
        history = [env.step(torch.empty(3, 0))[0]["policy"].clone() for _ in range(4)]
        outputs.append(torch.stack(history))
        states.append(torch.get_rng_state())
        if recorder_type is not None:
            term = env.recorder_manager.get_term("capture")
            assert not env.recorder_manager._fields
            if recorder_type is _PhysicalRecorder:
                assert torch.equal(term.latest, torch.full((3, 2), 4.0))
            else:
                # The default-compatible path really supplies observations to its callback.
                assert term.latest.shape == (3, 3, 2)
        env.recorder_manager.close()
    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(states[0], states[1])
    assert not torch.equal(outputs[0], outputs[2])
    assert not torch.equal(states[0], states[2])
    # After four steps the three history entries contain steps 2, 3 and 4, each noised once.
    expected = torch.tensor([2.0, 3.0, 4.0])[None, :, None]
    assert torch.all((outputs[1][-1] - expected).abs() <= 0.1)
