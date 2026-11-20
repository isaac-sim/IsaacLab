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
def test_final_observations_are_owned_and_cleared_each_step_and_reset(env_kind):
    """An autoreset step exposes owned terminal observations; the next step and an explicit reset drop them."""
    import torch

    from isaaclab.envs import DirectMARLEnv, DirectRLEnv
    from isaaclab.managers import ObservationGroupCfg, ObservationManager, ObservationTermCfg

    cls = {"manager": ManagerBasedRLEnv, "direct": DirectRLEnv, "marl": DirectMARLEnv}[env_kind]
    env = object.__new__(cls)
    env._is_closed = True
    env.cfg = SimpleNamespace(
        decimation=1,
        sim=SimpleNamespace(render_interval=1, dt=0.01),
        compute_final_obs=True,
        action_noise_model=None,
        observation_noise_model=None,
        num_rerenders_on_reset=0,
        wait_for_textures=False,
        events=True,
    )
    agents = ["a", "b"] if env_kind == "marl" else [None]
    state = torch.tensor([[10.0], [20.0]])

    def extras_of(agent):
        return env.extras[agent] if agent else env.extras

    def reset(ids):
        state[ids] = -1
        for agent in agents:
            extras_of(agent)["log"] = {"episode": 1}

    env.sim = SimpleNamespace(
        device="cpu",
        is_rendering=False,
        is_playing=lambda: True,
        consume_reset_request=lambda: False,
        step=lambda **kwargs: state.add_(1),
        forward=lambda: None,
    )
    env.scene = SimpleNamespace(num_envs=2, write_data_to_sim=lambda: None, update=lambda **kwargs: None)
    env.render_enabled = False
    env.has_rtx_sensors = False
    env.video_recorders = []
    env._physics_handles_decimation = False
    env._physics_handles_actions = False
    env._sim_step_counter = env.common_step_counter = 0
    env.episode_length_buf = torch.zeros(2, dtype=torch.long)
    env.reset_buf = torch.zeros(2, dtype=torch.bool)
    env.reset_terminated = torch.zeros(2, dtype=torch.bool)
    env.reset_time_outs = torch.zeros(2, dtype=torch.bool)
    env._reset_idx = reset
    env.event_manager = SimpleNamespace(
        active_terms=["interval"], available_modes=["interval"], apply=lambda **kw: None
    )
    env.extras = {agent: {} for agent in agents} if env_kind == "marl" else {}
    done = torch.zeros(2, dtype=torch.bool)
    timeout = torch.zeros(2, dtype=torch.bool)
    reward = torch.zeros(2)
    actions = torch.zeros(2, 1)
    if env_kind == "manager":
        group = ObservationGroupCfg()
        group.value = ObservationTermCfg(func=lambda env: state)
        env.observation_manager = ObservationManager({"policy": group}, env)
        env.recorder_manager = SimpleNamespace(
            record_pre_step=lambda: None,
            record_post_step=lambda: None,
            record_post_physics_decimation_step=lambda: None,
            record_pre_reset=lambda ids: None,
            record_post_reset=lambda ids: None,
        )
        env.action_manager = SimpleNamespace(process_action=lambda action: None, apply_action=lambda: None)
        env.termination_manager = SimpleNamespace(compute=lambda: done.clone(), terminated=done, time_outs=timeout)
        env.reward_manager = SimpleNamespace(compute=lambda **kwargs: reward)
        env.command_manager = SimpleNamespace(compute=lambda **kwargs: None)
    else:
        env._pre_physics_step = lambda actions: None
        env._apply_action = lambda: None
        if env_kind == "direct":
            env._get_dones = lambda: (done, timeout)
            env._get_rewards = lambda: reward
            # the task returns its live state buffer, which the reset overwrites
            env._get_observations = lambda: {"policy": state}
        else:
            env.cfg.possible_agents = env.possible_agents = agents
            env._get_dones = lambda: ({a: done for a in agents}, {a: timeout for a in agents})
            env._get_rewards = lambda: {a: reward for a in agents}
            env._get_observations = lambda: {a: state for a in agents}
            actions = {a: torch.zeros(2, 1) for a in agents}

    def terminal(agent):
        obs = extras_of(agent).get("final_obs")
        return None if obs is None else (obs if env_kind == "marl" else obs["policy"])

    # step 1 ends env 0's episode: its terminal observation is the post-physics state, unaffected by the reset
    done[0] = True
    env.step(actions)
    for agent in agents:
        torch.testing.assert_close(terminal(agent), torch.tensor([[11.0], [21.0]]))
    # step 2 has no reset, so no terminal observation remains from step 1
    done[0] = False
    env.step(actions)
    assert all(terminal(agent) is None for agent in agents)
    # an explicit reset after an autoreset step drops the terminal observation but keeps the reset's log
    done[0] = True
    env.step(actions)
    assert all(terminal(agent) is not None for agent in agents)
    _, extras = env.reset()
    for agent in agents:
        assert terminal(agent) is None
        assert extras_of(agent)["log"] == {"episode": 1}
