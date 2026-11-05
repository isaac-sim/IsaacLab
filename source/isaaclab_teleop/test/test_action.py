# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-owned input arbitration with a real action manager and no device runtime."""

from types import SimpleNamespace

import pytest
import torch
from isaaclab_teleop.action_cfg import TeleopActionCfg
from isaaclab_teleop.source import TeleopSample, TeleopSource
from isaaclab_teleop.source_cfg import TeleopSourceCfg

from isaaclab.managers import ActionManager, ActionTerm, ActionTermCfg
from isaaclab.utils import index_fill_

pytestmark = pytest.mark.kitless


class _Controller(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._raw = torch.zeros(env.num_envs, 2)
        self._processed = torch.zeros_like(self._raw)

    @property
    def action_dim(self):
        return 2

    @property
    def raw_actions(self):
        return self._raw

    @property
    def processed_actions(self):
        return self._processed

    def process_actions(self, actions):
        self._raw.copy_(actions)
        self._processed.copy_((actions * 2).clamp(-10, 10))

    def apply_actions(self):
        self._env.applied = self._processed.clone()

    def reset(self, env_ids=None):
        index_fill_(self._raw, env_ids, 0)

    def close(self):
        self._env.events.append("controller_closed")
        super().close()


class _Source(TeleopSource):
    def __init__(self, env):
        self.env = env
        self.sample = TeleopSample()
        self.polls = 0
        self.resets = []
        self.fail_close = False

    def advance(self):
        self.polls += 1
        return self.sample

    def reset(self, env_ids=None):
        self.resets.append(env_ids.tolist())

    def close(self):
        self.env.events.append("source_closed")
        if self.fail_close:
            raise RuntimeError("Source close failed")


def _build(**kwargs):
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        scene={"robot": object()},
        sim=SimpleNamespace(
            is_playing=lambda: True,
            vis_marker_registry=SimpleNamespace(clear_debug_vis_callback=lambda term: None),
        ),
        events=[],
        applied=None,
    )
    cfg = TeleopActionCfg(
        controller=ActionTermCfg(class_type=_Controller, asset_name="robot"),
        # The source constructor intentionally cannot accept runtime arguments:
        # ordinary task construction must never instantiate it.
        source=TeleopSourceCfg(class_type=object),
        **kwargs,
    )
    env.action_manager = ActionManager(SimpleNamespace(robot=cfg), env)
    return env, env.action_manager.get_term("robot")


def test_task_term_preserves_controller_and_policy_until_capture_takes_control():
    env, term = _build(env_ids=[2, 0])
    manager = env.action_manager
    policy = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    manager.process_action(policy)
    manager.apply_action()
    torch.testing.assert_close(env.applied, torch.tensor([[2.0, 4.0], [6.0, 8.0], [10.0, 10.0]]))

    source = _Source(env)
    term.activate(source)
    source.sample = TeleopSample(torch.tensor([[4.0, 3.0], [-2.0, -1.0]]))
    assert not term.poll_source()
    manager.process_action(policy)
    manager.apply_action()
    torch.testing.assert_close(env.applied, torch.tensor([[-4.0, -2.0], [6.0, 8.0], [8.0, 6.0]]))
    torch.testing.assert_close(manager.action, torch.tensor([[-2.0, -1.0], [3.0, 4.0], [4.0, 3.0]]))
    torch.testing.assert_close(policy, torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    assert source.polls == 1

    source.sample.active = False
    term.poll_source()
    manager.process_action(policy)
    torch.testing.assert_close(manager.action, policy)
    manager.close()
    manager.close()
    assert env.events == ["source_closed", "controller_closed"]


def test_adapter_reset_selection_and_stale_sample_handling():
    def adapter(values, env, env_ids):
        return values + env.num_envs + env_ids[:, None]

    env, term = _build(env_ids=[1], action_adapter=adapter)
    source = _Source(env)
    term.activate(source)
    source.sample = TeleopSample(torch.tensor([1.0, 2.0]))
    term.poll_source()
    env.action_manager.process_action(torch.zeros(3, 2))
    torch.testing.assert_close(env.action_manager.action[1], torch.tensor([5.0, 6.0]))

    env.action_manager.reset(torch.tensor([0]))
    assert source.resets == []
    env.action_manager.reset(slice(1, 3))
    assert source.resets == [[1]]
    env.action_manager.process_action(torch.ones(3, 2))
    torch.testing.assert_close(env.action_manager.action, torch.ones(3, 2))
    env.action_manager.reset()
    assert source.resets == [[1], [1]]

    source.sample = TeleopSample(torch.full((2,), float("nan")), reset=True)
    assert term.poll_source()
    env.action_manager.process_action(torch.ones(3, 2))
    torch.testing.assert_close(env.action_manager.action, torch.ones(3, 2))
    env.action_manager.close()


@pytest.mark.parametrize(
    ("values", "error", "message"),
    [
        (torch.ones(2, 2), ValueError, "shape"),
        (torch.full((2,), float("nan")), ValueError, "finite"),
        ([1, 2], TypeError, "torch.Tensor"),
    ],
)
def test_invalid_capture_actions_do_not_reach_controller(values, error, message):
    env, term = _build()
    source = _Source(env)
    term.activate(source)
    source.sample = TeleopSample(values)
    term.poll_source()
    with pytest.raises(error, match=message):
        env.action_manager.process_action(torch.ones(3, 2))
    torch.testing.assert_close(term.processed_actions, torch.zeros(3, 2))
    env.action_manager.close()


@pytest.mark.parametrize(("env_ids", "message"), [([3], "batch size"), ([0, 0], "duplicates"), ([-1], "non-negative")])
def test_invalid_environment_selection_rejected_before_device_activation(env_ids, message):
    with pytest.raises(ValueError, match=message):
        _build(env_ids=env_ids)


def test_source_failure_preserves_ownership_for_retry_and_deactivation_yields_to_policy():
    env, term = _build()
    source = _Source(env)
    term.activate(source)
    source.sample = TeleopSample(torch.tensor([8.0, 9.0]))
    term.poll_source()
    term.deactivate()
    env.action_manager.process_action(torch.ones(3, 2))
    torch.testing.assert_close(env.action_manager.action, torch.ones(3, 2))

    source = _Source(env)
    source.fail_close = True
    term.activate(source)
    with pytest.raises(RuntimeError, match="Failed to close 1 action term"):
        env.action_manager.close()
    assert env.events == ["source_closed", "source_closed"]
    source.fail_close = False
    env.action_manager.close()
    assert env.events == ["source_closed", "source_closed", "source_closed", "controller_closed"]
