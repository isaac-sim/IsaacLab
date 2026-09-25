# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stepping ownership and call ordering without a simulator runtime."""

from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from isaaclab.envs import DirectMARLEnv, DirectRLEnv, ManagerBasedEnv, ManagerBasedRLEnv

pytestmark = pytest.mark.unit
ENV_CLASSES = (ManagerBasedEnv, ManagerBasedRLEnv, DirectRLEnv, DirectMARLEnv)


@pytest.mark.parametrize("env_cls", ENV_CLASSES)
def test_step_has_one_physics_loop(env_cls):
    """Decimation ownership selects a count, never a duplicate stepping program."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(env_cls.step)))
    calls = [
        node for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.unparse(node.func) == "self.sim.step"
    ]
    assert len(calls) == 1
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.For) and calls[0] in ast.walk(node)]
    assert len(loops) == 1
    assert not any(
        isinstance(node, ast.If) and "_physics_handles_decimation" in ast.unparse(node.test) for node in ast.walk(tree)
    )


@pytest.mark.parametrize("env_cls", ENV_CLASSES)
@pytest.mark.parametrize(
    "handles_decimation, decimation, is_rendering, render_enabled",
    [
        (False, 4, True, True),
        (True, 4, True, True),
        (False, 4, True, False),
        (True, 4, False, True),
        (True, 1, True, False),
    ],
)
def test_step_preserves_physics_cadence(env_cls, handles_decimation, decimation, is_rendering, render_enabled):
    """Preserve action/write/step/record/render/update order and timing for either owner."""
    trace = []
    env = object.__new__(env_cls)
    env._is_closed = True
    env.cfg = SimpleNamespace(
        decimation=decimation,
        sim=SimpleNamespace(dt=0.01, render_interval=3),
        action_noise_model=None,
        observation_noise_model=None,
        events=None,
    )
    env._physics_handles_decimation = handles_decimation
    env._sim_step_counter = 0
    env.render_enabled = render_enabled
    env.sim = SimpleNamespace(
        device="cpu",
        is_rendering=is_rendering,
        step=lambda **kwargs: trace.append(("step", env._sim_step_counter, kwargs)),
        render=lambda **kwargs: trace.append(("render", env._sim_step_counter, kwargs)),
        consume_reset_request=lambda: False,
    )
    env.scene = SimpleNamespace(
        num_envs=2,
        write_data_to_sim=lambda: trace.append(("write", env._sim_step_counter)),
        update=lambda dt: trace.append(("update", dt)),
    )
    env._apply_action = lambda: trace.append(("action", env._sim_step_counter))
    env._pre_physics_step = Mock()
    env.action_manager = SimpleNamespace(process_action=Mock(), apply_action=env._apply_action)
    env.recorder_manager = Mock(active_terms=[])
    env.recorder_manager.record_post_physics_decimation_step.side_effect = lambda: trace.append(
        ("record", env._sim_step_counter)
    )
    env.event_manager = SimpleNamespace(available_modes=[])
    env.command_manager = Mock()
    env.video_recorders = []
    env.episode_length_buf = torch.zeros(2, dtype=torch.long)
    env.common_step_counter = 0
    env.reset_terminated = torch.zeros(2, dtype=torch.bool)
    env.reset_time_outs = torch.zeros(2, dtype=torch.bool)
    env.reset_buf = torch.zeros(2, dtype=torch.bool)
    env.termination_manager = SimpleNamespace(
        compute=lambda: env.reset_buf, terminated=env.reset_terminated, time_outs=env.reset_time_outs
    )
    env.reward_manager = SimpleNamespace(compute=lambda dt: torch.zeros(2))
    env.possible_agents = ["agent"]
    env._get_dones = (
        (lambda: ({"agent": env.reset_terminated}, {"agent": env.reset_time_outs}))
        if env_cls is DirectMARLEnv
        else lambda: (env.reset_terminated, env.reset_time_outs)
    )
    env._get_rewards = lambda: torch.zeros(2)
    env._get_observations = lambda: {"agent": torch.zeros(2, 1)}
    env.observation_manager = SimpleNamespace(compute=lambda **kwargs: env._get_observations())
    env.extras = {}
    actions = torch.zeros(2, 1)
    if env_cls is DirectMARLEnv:
        actions = {"agent": actions}

    # A non-dividing render interval distinguishes end-of-call checks from crossed-boundary rendering.
    for _ in range(3):
        env.step(actions)

    end_ticks = (
        range(decimation, 3 * decimation + 1, decimation) if handles_decimation else range(1, 3 * decimation + 1)
    )
    expected = []
    for tick in end_ticks:
        expected.extend([("action", tick), ("write", tick), ("step", tick, {"render": False})])
        if env_cls is ManagerBasedRLEnv:
            expected.append(("record", tick))
        if is_rendering and tick % 3 == 0:
            expected.append(("render", tick, {"skip_app_pumping": not render_enabled}))
        expected.append(("update", 0.01 * decimation if handles_decimation else 0.01))
    assert trace == expected
    assert env._sim_step_counter == 3 * decimation
    if env_cls is not ManagerBasedEnv:
        torch.testing.assert_close(env.episode_length_buf, torch.full((2,), 3, dtype=torch.long))
        assert env.common_step_counter == 3
