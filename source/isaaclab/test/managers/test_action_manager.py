# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action selection and resource ownership without a simulation runtime."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import ActionInput, ActionInputCfg, ActionManager, ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

pytestmark = pytest.mark.kitless


class _Action(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._raw = torch.zeros(env.num_envs, 3)
        self.encoder_bias = torch.zeros(env.num_envs, 1)
        self.seen_action = None
        self.closed = False

    @property
    def action_dim(self):
        return 1

    @property
    def raw_actions(self):
        # Hierarchical controllers may expose low-level outputs here. They
        # must not replace the manager's raw, task-level action history.
        return self._raw

    @property
    def processed_actions(self):
        return self._raw

    def process_actions(self, actions):
        self.seen_action = self._env.action_manager.action.clone()
        self._raw[:] = actions * self.cfg.scale

    def set_reset_position(self, value):
        self._raw[:] = value

    def reset(self, env_ids=None):
        self._env.events.append((self.cfg.asset_name, "controller_reset"))

    def apply_actions(self):
        pass

    def close(self):
        if self.closed:
            return
        self._env.events.append(self.cfg.asset_name)
        super().close()
        if self.cfg.fail_close:
            raise RuntimeError("Controller close failed")
        self.closed = True


@configclass
class _ActionCfg(ActionTermCfg):
    class_type = _Action
    scale: float = 3.0
    joint_names: list[str] = ["joint"]
    fail_close: bool = False


class _Input(ActionInput):
    def __init__(self, cfg, env, *, action_dim):
        super().__init__(cfg, env, action_dim=action_dim)
        env.events.append((cfg.label, "input_created"))
        self.result = None
        self.closed = False

    def resolve_actions(self, actions):
        if self.result is not None:
            return self.result
        return actions + self.cfg.offset if self.cfg.offset else actions

    def reset(self, env_ids=None):
        self._env.events.append((self.cfg.label, "input_reset"))

    def close(self):
        if self.closed:
            return
        self._env.events.append((self.cfg.label, "input_close"))
        if self.cfg.fail_close:
            raise RuntimeError("Input close failed")
        self.closed = True


@configclass
class _InputCfg(ActionInputCfg):
    class_type = _Input
    label: str = "input"
    offset: float = 10.0
    fail_close: bool = False


def _env():
    return SimpleNamespace(
        num_envs=2,
        device="cpu",
        scene={"first": object(), "second": object()},
        sim=SimpleNamespace(
            is_playing=lambda: True,
            vis_marker_registry=SimpleNamespace(clear_debug_vis_callback=lambda term: None),
        ),
        events=[],
    )


def test_activated_input_preserves_controller_api_and_records_effective_actions():
    """Optional input leaves task/config APIs intact while recorders see the selected raw actions."""
    env = _env()
    cfg = SimpleNamespace(
        first=_ActionCfg(asset_name="first"), second=_ActionCfg(asset_name="second", input=_InputCfg())
    )
    # Inherited task initialization and config overrides keep their original paths.
    cfg.second.scale = 4
    cfg.second.joint_names = ["selected_joint"]
    env.action_manager = manager = ActionManager(cfg, env)
    controller = manager.get_term("second")
    assert type(controller) is _Action and type(controller.cfg) is _ActionCfg
    assert controller.cfg.scale == 4 and controller.cfg.joint_names == ["selected_joint"]
    assert manager.get_input("second") is None and not env.events
    controller.encoder_bias.fill_(0.25)
    controller.set_reset_position(0.5)
    torch.testing.assert_close(controller.raw_actions, torch.full((2, 3), 0.5))
    assert manager.serialize()["second"]["cfg"]["scale"] == 4
    policy = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    manager.process_action(policy)
    torch.testing.assert_close(manager.action, policy)
    torch.testing.assert_close(controller.processed_actions, torch.tensor([[8.0] * 3, [16.0] * 3]))

    action_input = manager.activate_input("second")
    assert manager.get_term("second") is controller
    assert manager.get_input("second") is action_input and action_input.action_dim == 1
    with pytest.raises(RuntimeError, match="already active"):
        manager.activate_input("second")
    assert env.events == [("input", "input_created")]
    manager.process_action(policy)
    expected = torch.tensor([[1.0, 12.0], [3.0, 14.0]])
    torch.testing.assert_close(manager.action, expected)
    torch.testing.assert_close(manager.get_term("first").seen_action, expected)
    torch.testing.assert_close(policy, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    manager.process_action(torch.zeros(2, 2))
    torch.testing.assert_close(manager.prev_action, expected)
    torch.testing.assert_close(manager.action, torch.tensor([[0.0, 10.0], [0.0, 10.0]]))
    manager.reset(torch.tensor([0]))
    assert env.events[-3:] == [("first", "controller_reset"), ("second", "controller_reset"), ("input", "input_reset")]
    torch.testing.assert_close(manager.action[0], torch.zeros(2))
    manager.close_inputs()
    assert manager.get_input("second") is None and not controller.closed
    manager.process_action(policy)
    torch.testing.assert_close(manager.action, policy)
    torch.testing.assert_close(controller.encoder_bias, torch.full((2, 1), 0.25))
    manager.close()


def test_close_releases_other_terms_when_one_fails():
    env = _env()
    manager = ActionManager(
        SimpleNamespace(first=_ActionCfg(asset_name="first", fail_close=True), second=_ActionCfg(asset_name="second")),
        env,
    )
    with pytest.raises(RuntimeError, match="Failed to close 1 action term"):
        manager.close()
    assert env.events == ["first", "second"]
    manager.get_term("first").cfg.fail_close = False
    manager.close()
    manager.close()
    assert env.events == ["first", "second", "first"]


def test_failed_manager_construction_closes_previously_created_terms():
    env = _env()
    with pytest.raises(TypeError, match="not of type ActionTermCfg"):
        ActionManager(SimpleNamespace(first=_ActionCfg(asset_name="first"), second=object()), env)
    assert env.events == ["first"]


def test_failed_input_drain_retains_controllers_and_allows_retry():
    """External input must release the scene before any controller teardown begins."""
    env = _env()
    manager = ActionManager(
        SimpleNamespace(
            first=_ActionCfg(asset_name="first", input=_InputCfg(label="first", fail_close=True)),
            second=_ActionCfg(asset_name="second", input=_InputCfg(label="second")),
        ),
        env,
    )
    first = manager.activate_input("first")
    second = manager.activate_input("second")
    env.events.clear()
    with pytest.raises(RuntimeError, match="Failed to close 1 action input"):
        manager.close()
    assert env.events == [("first", "input_close"), ("second", "input_close")]
    assert not manager.get_term("first").closed and not manager.get_term("second").closed
    assert manager.get_input("first") is first and manager.get_input("second") is second

    first.cfg.fail_close = False
    manager.close()
    manager.close()
    assert env.events == [
        ("first", "input_close"),
        ("second", "input_close"),
        ("first", "input_close"),
        "first",
        "second",
    ]
    assert manager.get_input("first") is None and manager.get_input("second") is None


@pytest.mark.parametrize(
    ("result", "error", "message"),
    [
        (torch.zeros(2, 2), ValueError, "shape"),
        (torch.zeros(2, 1, dtype=torch.float64), ValueError, "dtype"),
        (torch.empty(2, 1, device="meta"), ValueError, "device"),
        ([0.0, 0.0], TypeError, "torch.Tensor"),
    ],
)
def test_invalid_input_result_is_rejected_before_any_controller_processing(result, error, message):
    env = _env()
    env.action_manager = manager = ActionManager(
        SimpleNamespace(
            first=_ActionCfg(asset_name="first"), second=_ActionCfg(asset_name="second", input=_InputCfg())
        ),
        env,
    )
    manager.activate_input("second").result = result
    with pytest.raises(error, match=message):
        manager.process_action(torch.zeros(2, 2))
    assert manager.get_term("first").seen_action is None and manager.get_term("second").seen_action is None
    manager.close()
