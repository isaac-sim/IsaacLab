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
from isaaclab.managers import TerminationManager, TerminationTermCfg

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


@pytest.mark.kitless
def test_close_retains_borrowed_resources_until_action_terms_release_them():
    """Failed input teardown must be retryable before managers and simulation disappear."""
    events = []
    state = SimpleNamespace(fail=True, closed=False)

    def close_actions():
        if state.closed:
            return
        events.append("actions")
        if state.fail:
            raise RuntimeError("Input still owns a renderer lease")
        state.closed = True

    env = object.__new__(ManagerBasedRLEnv)
    env._is_closed = False
    env._window = None
    env.action_manager = SimpleNamespace(close=close_actions)
    env.sim = SimpleNamespace(stop=lambda: events.append("stop"), clear_instance=lambda: events.append("clear"))
    for name in (
        "command_manager",
        "reward_manager",
        "termination_manager",
        "curriculum_manager",
        "observation_manager",
        "event_manager",
        "recorder_manager",
        "scene",
    ):
        setattr(env, name, object())
    with pytest.raises(RuntimeError, match="renderer lease"):
        env.close()
    assert events == ["actions"]
    assert env.command_manager is not None
    assert env.scene is not None

    state.fail = False
    env.close()
    env.close()
    assert events == ["actions", "actions", "stop", "clear"]
    assert env.action_space is None


def _make_stepping_env() -> tuple[ManagerBasedRLEnv, list]:
    """Exercise the real environment step/reset with deterministic scalar dynamics."""
    env = object.__new__(ManagerBasedRLEnv)
    env._is_closed = True
    env.cfg = SimpleNamespace(
        decimation=2, sim=SimpleNamespace(dt=0.1, render_interval=2), compute_final_obs=True, wait_for_textures=False
    )
    env._physics_handles_decimation = False
    env._sim_step_counter = 0
    env.common_step_counter = 0
    env.episode_length_buf = torch.zeros(4, dtype=torch.long)
    env.reset_buf = torch.zeros(4, dtype=torch.bool)
    env._requested_reset_buf = None
    env.extras = {}
    env.render_enabled = True
    env.has_rtx_sensors = False
    env.video_recorders = []
    events = []

    def scene_reset(env_ids):
        events.append(("scene_reset", torch.arange(4)[env_ids].tolist()))
        env.scene.state[env_ids] = 0.0

    env.scene = SimpleNamespace(
        num_envs=4,
        state=torch.zeros(4),
        reset=scene_reset,
        write_data_to_sim=lambda: None,
        update=lambda dt: None,
    )

    def physics_step(render):
        env.scene.state.add_(env.action_manager.action)
        events.append(("physics", env.scene.state.tolist()))

    def consume_reset_request():
        requested = env.sim.reset_requested
        env.sim.reset_requested = False
        return requested

    env.sim = SimpleNamespace(
        device="cpu",
        is_playing=lambda: True,
        is_rendering=False,
        step=physics_step,
        forward=lambda: None,
        reset_requested=False,
        consume_reset_request=consume_reset_request,
    )

    def process_action(action):
        env.action_manager.action = action[:, 0].clone()

    env.action_manager = SimpleNamespace(process_action=process_action, apply_action=lambda: None, reset=lambda ids: {})
    env.observation_manager = SimpleNamespace(
        compute=lambda update_history=False: {"policy": env.scene.state.clone()}, reset=lambda ids: {}
    )
    env.reward_manager = SimpleNamespace(compute=lambda dt: env.scene.state.clone(), reset=lambda ids: {})
    env.curriculum_manager = SimpleNamespace(compute=lambda env_ids: None, reset=lambda ids: {})
    env.command_manager = SimpleNamespace(compute=lambda dt: None, reset=lambda ids: {})
    env.event_manager = SimpleNamespace(active_terms=[], reset=lambda ids: {})
    env.termination_manager = TerminationManager(
        {
            "success": TerminationTermCfg(func=lambda env: env.scene.state == 6.0),
            "time_out": TerminationTermCfg(func=lambda env: env.scene.state >= 8.0, time_out=True),
        },
        env,
    )
    env.recorder_manager = SimpleNamespace(
        record_pre_step=lambda: None,
        record_post_physics_decimation_step=lambda: None,
        record_post_step=lambda: events.append(("post_step", env.scene.state.tolist())),
        record_pre_reset=lambda ids: events.append(("pre_reset", env.scene.state.tolist())),
        record_post_reset=lambda ids: events.append(("post_reset", env.scene.state.tolist())),
        reset=lambda ids: {},
    )
    return env, events


@pytest.mark.kitless
@pytest.mark.parametrize(
    "viewer_reset, reset_ids, expected_observation",
    [(False, [0, 2, 3], [0.0, 4.0, 0.0, 0.0]), (True, [0, 1, 2, 3], [0.0, 0.0, 0.0, 0.0])],
)
def test_requested_resets_join_real_step_and_autoreset(viewer_reset, reset_ids, expected_observation):
    """Queued operator and viewer requests share one reset boundary without fabricating transitions."""
    env, events = _make_stepping_env()
    env.request_reset(torch.tensor([0, 2, 3]))
    env.request_reset(torch.tensor([0]))
    env.sim.reset_requested = viewer_reset

    observations, rewards, terminated, truncated, extras = env.step(torch.tensor([[1.0], [2.0], [3.0], [4.0]]))

    assert env._sim_step_counter == 2
    assert env.common_step_counter == 1
    assert rewards.tolist() == [2.0, 4.0, 6.0, 8.0]
    assert terminated.tolist() == ([True, True, True, False] if viewer_reset else [True, False, True, False])
    assert truncated.tolist() == [False, False, False, True]
    assert observations["policy"].tolist() == expected_observation
    assert extras["final_obs"]["policy"].tolist() == [2.0, 4.0, 6.0, 8.0]
    assert env.termination_manager.get_term("success").tolist() == [False, False, True, False]
    assert events == [
        ("physics", [1.0, 2.0, 3.0, 4.0]),
        ("physics", [2.0, 4.0, 6.0, 8.0]),
        ("post_step", [2.0, 4.0, 6.0, 8.0]),
        ("pre_reset", [2.0, 4.0, 6.0, 8.0]),
        ("scene_reset", reset_ids),
        ("post_reset", expected_observation),
    ]

    events.clear()
    _, _, terminated, truncated, _ = env.step(torch.zeros((4, 1)))
    assert not terminated.any() and not truncated.any()
    assert all(name != "scene_reset" for name, _ in events)


@pytest.mark.kitless
def test_explicit_reset_consumes_only_its_queued_rows():
    """An explicit reset satisfies pending requests without discarding requests for other rows."""
    env, events = _make_stepping_env()
    env.request_reset(slice(0, 2))
    env.reset(env_ids=torch.tensor([0]))
    events.clear()

    _, _, terminated, _, _ = env.step(torch.zeros((4, 1)))
    assert terminated.tolist() == [False, True, False, False]
    assert [value for name, value in events if name == "scene_reset"] == [[1]]
