# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-owned input arbitration with a real action manager and no device runtime."""

from types import SimpleNamespace

import pytest
import torch
from isaaclab_teleop.input_cfg import TeleopInputCfg
from isaaclab_teleop.source import TeleopSample, TeleopSource
from isaaclab_teleop.source_cfg import TeleopSourceCfg

from isaaclab.managers import ActionManager, ActionTerm, ActionTermCfg
from isaaclab.utils import index_fill_

pytestmark = pytest.mark.kitless


class _Controller(ActionTerm):
    dimension = 2

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._raw = torch.zeros(env.num_envs, self.action_dim)
        self._processed = torch.zeros_like(self._raw)

    @property
    def action_dim(self):
        return self.dimension

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


class _GripperController(_Controller):
    dimension = 1


def _build(*, terms=None, activate=True, **kwargs):
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
    if terms is None:
        terms = {
            "robot": ActionTermCfg(
                class_type=_Controller,
                asset_name="robot",
                input=TeleopInputCfg(source=TeleopSourceCfg(class_type=object), **kwargs),
            )
        }
    env.action_manager = ActionManager(SimpleNamespace(**terms), env)
    return env, env.action_manager.activate_input(next(iter(terms))) if activate else None


def test_optional_input_preserves_controller_and_policy_until_capture_takes_control():
    env, _ = _build(env_ids=[2, 0], activate=False)
    manager = env.action_manager
    policy = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    manager.process_action(policy)
    manager.apply_action()
    torch.testing.assert_close(env.applied, torch.tensor([[2.0, 4.0], [6.0, 8.0], [10.0, 10.0]]))

    controller = manager.get_term("robot")
    assert isinstance(controller, _Controller) and isinstance(controller.cfg, ActionTermCfg)
    assert manager.get_input("robot") is None
    term = manager.activate_input("robot")
    assert manager.get_term("robot") is controller

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


def test_one_sample_drives_arm_and_gripper_without_duplicate_source_lifecycle():
    """Split controllers share input while their adapters and reset state remain isolated."""

    def arm_actions(values, env, env_ids):
        # An in-place conversion must not corrupt the shared Capture sample.
        values.mul_(-1)
        return values[:2]

    env, arm = _build(
        terms={
            "arm": ActionTermCfg(
                class_type=_Controller,
                asset_name="robot",
                input=TeleopInputCfg(
                    source=TeleopSourceCfg(class_type=object), env_ids=[1], action_adapter=arm_actions
                ),
            ),
            "gripper": ActionTermCfg(
                class_type=_GripperController,
                asset_name="robot",
                input=TeleopInputCfg(env_ids=[1], action_adapter=lambda values, env, env_ids: values[2:]),
            ),
        }
    )
    manager = env.action_manager
    gripper = manager.activate_input("gripper")
    source = _Source(env)
    source.sample = TeleopSample(torch.tensor([1.0, 2.0, 0.25]))
    arm.activate(source)
    assert not arm.poll_source()
    gripper.set_sample(arm.sample)
    policy = torch.ones(3, 3)
    manager.process_action(policy)

    torch.testing.assert_close(manager.action, torch.tensor([[1.0, 1.0, 1.0], [-1.0, -2.0, 0.25], [1.0, 1.0, 1.0]]))
    torch.testing.assert_close(manager.get_term("arm").processed_actions[1], torch.tensor([-2.0, -4.0]))
    torch.testing.assert_close(manager.get_term("gripper").processed_actions[1], torch.tensor([0.5]))
    torch.testing.assert_close(source.sample.actions, torch.tensor([1.0, 2.0, 0.25]))
    assert source.polls == 1

    manager.reset(torch.tensor([0]))
    assert source.resets == []
    manager.process_action(policy)
    torch.testing.assert_close(manager.action[1], torch.tensor([-1.0, -2.0, 0.25]))
    manager.reset(torch.tensor([1]))
    assert source.resets == [[1]]
    manager.process_action(policy)
    torch.testing.assert_close(manager.action, policy)

    source.sample.active = False
    arm.poll_source()
    gripper.set_sample(arm.sample)
    manager.process_action(policy)
    torch.testing.assert_close(manager.action, policy)
    manager.close()
    assert env.events == ["source_closed", "controller_closed", "controller_closed"]


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
    torch.testing.assert_close(env.action_manager.get_term("robot").processed_actions, torch.zeros(3, 2))
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
    with pytest.raises(RuntimeError, match="Failed to close 1 action input"):
        env.action_manager.close()
    assert env.events == ["source_closed", "source_closed"]
    source.fail_close = False
    env.action_manager.close()
    assert env.events == ["source_closed", "source_closed", "source_closed", "controller_closed"]
