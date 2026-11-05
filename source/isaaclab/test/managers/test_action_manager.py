# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action selection and resource ownership without a simulation runtime."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import ActionManager, ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

pytestmark = pytest.mark.kitless


class _Action(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._raw = torch.zeros(env.num_envs, 3)
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

    def resolve_actions(self, actions):
        return actions + self.cfg.offset if self.cfg.offset else actions

    def process_actions(self, actions):
        self.seen_action = self._env.action_manager.action.clone()
        self._raw[:] = actions * 3

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
    offset: float = 0.0
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


def test_effective_actions_precede_all_processing_and_preserve_policy_input():
    """Recorders and cross-term observations see the actions actually selected."""
    env = _env()
    cfg = SimpleNamespace(first=_ActionCfg(asset_name="first"), second=_ActionCfg(asset_name="second", offset=10))
    env.action_manager = manager = ActionManager(cfg, env)
    policy = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    manager.process_action(policy)
    expected = torch.tensor([[1.0, 12.0], [3.0, 14.0]])
    torch.testing.assert_close(manager.action, expected)
    torch.testing.assert_close(manager.get_term("first").seen_action, expected)
    torch.testing.assert_close(policy, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    manager.process_action(torch.zeros(2, 2))
    torch.testing.assert_close(manager.prev_action, expected)
    torch.testing.assert_close(manager.action, torch.tensor([[0.0, 10.0], [0.0, 10.0]]))
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
