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


@pytest.mark.parametrize("env_kind", ["manager", "direct", "marl"])
@pytest.mark.parametrize("boundary", ["auto", "manual", "mixed"])
def test_final_observations_precede_resets_and_events(env_kind, boundary):
    """Every boundary captures once, including manual resets, with owned pre-event successors."""
    import torch

    from isaaclab.envs import DirectMARLEnv, DirectRLEnv
    from isaaclab.managers import ObservationGroupCfg, ObservationManager, ObservationTermCfg

    cls = {"manager": ManagerBasedRLEnv, "direct": DirectRLEnv, "marl": DirectMARLEnv}[env_kind]
    env = object.__new__(cls)
    env._is_closed = True
    env._is_observation_preview = False
    env.cfg = SimpleNamespace(
        decimation=1,
        sim=SimpleNamespace(render_interval=1, dt=0.01),
        compute_final_obs=True,
        action_noise_model=None,
        observation_noise_model=None,
        num_rerenders_on_reset=0,
        events=True,
    )
    state = torch.tensor([[10.0], [20.0]])
    reset_calls = []
    preview_values = []
    manual_reset = boundary != "auto"
    ui_request = [manual_reset]

    def consume_reset():
        return ui_request.pop() if ui_request else False

    def observe():
        if env.is_observation_preview:
            preview_values.append(state.clone())
        return state

    def reset(ids):
        reset_calls.append(ids.clone())
        state[ids] = -1

    def event(**kwargs):
        state.add_(1000)

    env.sim = SimpleNamespace(
        device="cpu",
        is_rendering=False,
        is_playing=lambda: True,
        consume_reset_request=consume_reset,
        step=lambda **kwargs: state.add_(1),
    )
    env.scene = SimpleNamespace(num_envs=2, write_data_to_sim=lambda: None, update=lambda **kwargs: None)
    env.render_enabled = False
    env.has_rtx_sensors = False
    env.video_recorders = []
    env._physics_handles_decimation = False
    env._sim_step_counter = env.common_step_counter = 0
    env.episode_length_buf = torch.zeros(2, dtype=torch.long)
    env.reset_buf = torch.zeros(2, dtype=torch.bool)
    env.reset_terminated = torch.zeros(2, dtype=torch.bool)
    env.reset_time_outs = torch.zeros(2, dtype=torch.bool)
    env._reset_idx = reset
    env.event_manager = SimpleNamespace(active_terms=["interval"], available_modes=["interval"], apply=event)
    done = torch.tensor([boundary != "manual", False])
    timeout = torch.zeros(2, dtype=torch.bool)
    reward = torch.zeros(2)
    if env_kind == "manager":

        def managed_observe(env):
            return observe()

        group = ObservationGroupCfg()
        group.value = ObservationTermCfg(func=managed_observe)
        env.observation_manager = ObservationManager({"policy": group}, env)
        env.recorder_manager = SimpleNamespace(
            active_terms=[],
            record_pre_step=lambda: None,
            record_post_physics_decimation_step=lambda: None,
            record_pre_reset=lambda ids: None,
            record_post_reset=lambda ids: None,
        )
        env.action_manager = SimpleNamespace(process_action=lambda action: None, apply_action=lambda: None)
        env.termination_manager = SimpleNamespace(compute=lambda: done.clone(), terminated=done, time_outs=timeout)
        env.reward_manager = SimpleNamespace(compute=lambda **kwargs: reward)
        env.command_manager = SimpleNamespace(compute=lambda **kwargs: state.add_(100))
        env.extras = {"final_obs": {"policy": torch.full_like(state, -999)}}
        actions = torch.zeros(2, 1)
    else:
        env._pre_physics_step = lambda actions: None
        env._apply_action = lambda: None
        if env_kind == "direct":
            env._get_dones = lambda: (done, timeout)
            env._get_rewards = lambda: reward
            env._get_observations = lambda: {"policy": observe()}
            env.extras = {"final_obs": {"policy": torch.full_like(state, -999)}}
            actions = torch.zeros(2, 1)
        else:
            env.possible_agents = ["a", "b"]
            env._get_dones = lambda: ({"a": done, "b": done.clone()}, {"a": timeout, "b": timeout.clone()})
            env._get_rewards = lambda: {"a": reward, "b": reward.clone()}
            env._get_observations = lambda: {"a": observe(), "b": state}
            env.extras = {agent: {"final_obs": torch.full_like(state, -999)} for agent in ("a", "b")}
            actions = {"a": torch.zeros(2, 1), "b": torch.zeros(2, 1)}
    # Initialization may probe observations; only actual step-time previews count.
    preview_values.clear()
    _, _, terminated, _, extras = env.step(actions)
    terminal = extras["a"]["final_obs"] if env_kind == "marl" else extras["final_obs"]["policy"]
    torch.testing.assert_close(terminal, torch.tensor([[11.0], [21.0]]))
    assert len(preview_values) == len(reset_calls) == 1
    assert reset_calls[0].tolist() == ([0, 1] if manual_reset else [0])
    terminated = terminated["a"] if env_kind == "marl" else terminated
    assert terminated.tolist() == [True, manual_reset]
    assert not env.is_observation_preview
    done.zero_()
    env.step(actions)
    if env_kind == "marl":
        assert all("final_obs" not in value for value in env.extras.values())
    else:
        assert "final_obs" not in env.extras
    torch.testing.assert_close(terminal, torch.tensor([[11.0], [21.0]]))
