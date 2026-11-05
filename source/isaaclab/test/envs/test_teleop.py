# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action ownership and input lifecycle without a simulator or physical device."""

from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

from isaaclab.envs import (
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    TeleopCfg,
    TeleopEnvWrapper,
    TeleopSample,
    TeleopSource,
    TeleopSourceCfg,
)

pytestmark = pytest.mark.kitless


class _Input(TeleopSource):
    def __init__(self, cfg, env, *, env_ids):
        self.env = env
        env.input = self
        self.sample = TeleopSample()
        self.resets = []
        self.fail_close = False

    def advance(self):
        return self.sample

    def reset(self, env_ids=None):
        self.resets.append(list(env_ids))

    def close(self):
        self.env.events.append("source_closed")
        if self.fail_close:
            raise RuntimeError("Input teardown failed")


class _Env(gym.Env):
    num_envs = 3
    device = "cpu"
    render_mode = None

    def __init__(self, cfg):
        self.cfg = cfg
        self.single_action_space = gym.spaces.Box(-10, 10, shape=(4,))
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        self.observation_space = gym.spaces.Box(-10, 10, shape=(self.num_envs, 1))
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool)
        self.next_resets = torch.zeros_like(self.reset_buf)
        self.manual_reset = False
        self.sim = SimpleNamespace(request_reset=self.request_reset)
        self.scene = SimpleNamespace(num_envs=self.num_envs)
        self.events = []
        self.received_actions = None

    def request_reset(self):
        self.manual_reset = True

    def reset(self, *, seed=None, options=None):
        self.events.append("env_reset")
        return {}, {}

    def step(self, actions):
        self.received_actions = actions
        self.reset_buf.copy_(self.next_resets)
        self.next_resets.zero_()
        terminated = self.reset_buf.clone()
        if self.manual_reset:
            terminated[:] = True
            self.manual_reset = False
        return {}, torch.zeros(self.num_envs), terminated, torch.zeros_like(self.reset_buf), {}

    def close(self):
        self.events.append("env_closed")


def _cfg(**kwargs):
    return TeleopCfg(source=TeleopSourceCfg(class_type=_Input), **kwargs)


def test_named_terms_override_only_selected_policy_actions():
    """Term order and batch selection must survive policy composition without mutating its buffer."""
    env = _Env(ManagerBasedRLEnvCfg(teleop=_cfg(env_ids=[2, 0], action_terms=["gripper", "arm"])))
    env.action_manager = SimpleNamespace(active_terms=["arm", "base", "gripper"], action_term_dim=[2, 1, 1])
    wrapped = TeleopEnvWrapper(env)
    policy = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    original = policy.clone()
    env.input.sample = TeleopSample(torch.tensor([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]]))

    wrapped.step(policy)

    torch.testing.assert_close(
        env.received_actions, torch.tensor([[5.0, 4.0, 2.0, 6.0], [4.0, 5.0, 6.0, 7.0], [8.0, 7.0, 10.0, 9.0]])
    )
    torch.testing.assert_close(policy, original)
    env.input.sample.active = False
    wrapped.step(policy)
    assert env.received_actions is policy
    wrapped.close()


def test_adapter_and_reset_lifecycle_preserve_environment_ownership():
    """Adapters see scene context; real reset masks notify only controlled source rows."""

    def adapter(values, env, env_ids):
        return values + env.num_envs + env_ids[:, None]

    env = _Env(DirectRLEnvCfg(teleop=_cfg(env_ids=[1], action_indices=[2], action_adapter=adapter)))
    wrapped = TeleopEnvWrapper(env)
    source = env.input
    wrapped.reset()
    assert source.resets == [[1]]
    source.sample = TeleopSample(torch.tensor([2.0]))
    env.next_resets[0] = True
    wrapped.step(torch.zeros(3, 4))
    assert env.received_actions[1, 2] == 6
    assert len(source.resets) == 1
    env.next_resets[1] = True
    wrapped.step(torch.zeros(3, 4))
    assert source.resets == [[1], [1]]

    # A reset pulse does not send the accompanying stale action. The task resets
    # once through its existing UI reset path and reports ordinary done signals.
    source.sample = TeleopSample(torch.full((1,), float("nan")), reset=True)
    result = wrapped.step(torch.ones(3, 4))
    assert result[2].tolist() == [True, True, True]
    assert not env.reset_buf.any()
    torch.testing.assert_close(env.received_actions, torch.ones(3, 4))
    assert source.resets == [[1], [1], [1]]
    wrapped.close()
    wrapped.close()
    assert env.events[-2:] == ["source_closed", "env_closed"]
    assert env.events.count("env_closed") == 1


@pytest.mark.parametrize(
    ("sample", "error", "match"),
    [
        (torch.ones(2, 4), ValueError, "shape"),
        (torch.full((4,), float("nan")), ValueError, "finite"),
        ([0, 0, 0, 0], TypeError, "torch.Tensor"),
    ],
)
def test_invalid_input_fails_before_applying_actions(sample, error, match):
    env = _Env(DirectRLEnvCfg(teleop=_cfg()))
    wrapped = TeleopEnvWrapper(env)
    env.input.sample = TeleopSample(sample)
    with pytest.raises(error, match=match):
        wrapped.step(torch.zeros(3, 4))
    assert env.received_actions is None
    wrapped.close()


@pytest.mark.parametrize(
    "selection,match",
    [
        ({"env_ids": [3]}, "batch size"),
        ({"env_ids": [0, 0]}, "duplicates"),
        ({"action_indices": [4]}, "action dimension"),
        ({"action_terms": ["arm"], "action_indices": [0]}, "not both"),
        ({"action_terms": ["arm"]}, "manager-based"),
    ],
)
def test_invalid_ownership_rejected_before_starting_input(selection, match):
    def unexpected_start(*args, **kwargs):
        pytest.fail("Input started before configuration validation")

    cfg = _cfg(**selection)
    cfg.source.class_type = unexpected_start
    with pytest.raises(ValueError, match=match):
        TeleopEnvWrapper(_Env(DirectRLEnvCfg(teleop=cfg)))


def test_source_teardown_failure_still_closes_environment():
    env = _Env(DirectRLEnvCfg(teleop=_cfg()))
    wrapped = TeleopEnvWrapper(env)
    env.input.fail_close = True
    with pytest.raises(RuntimeError, match="Input teardown failed"):
        wrapped.close()
    assert env.events == ["source_closed", "env_closed"]


def test_discrete_actions_require_an_explicit_integer_mapping():
    env = _Env(DirectRLEnvCfg(teleop=_cfg()))
    env.single_action_space = gym.spaces.Discrete(3)
    wrapped = TeleopEnvWrapper(env)
    env.input.sample = TeleopSample(torch.tensor([1.5]))
    with pytest.raises(TypeError, match="integer dtype"):
        wrapped.step(torch.zeros(3, 1, dtype=torch.long))
    env.input.sample = TeleopSample(torch.tensor([2]))
    wrapped.step(torch.zeros(3, 1, dtype=torch.long))
    torch.testing.assert_close(env.received_actions, torch.tensor([[2], [0], [0]]))
    wrapped.close()


@pytest.mark.parametrize("conversion", ["single_agent", "one_agent"])
def test_multi_agent_conversion_preserves_teleop_actions_and_lifecycle(conversion):
    """The converter must not unwrap and silently bypass the input source."""
    from isaaclab.envs.utils.marl import multi_agent_to_single_agent, multi_agent_with_one_agent

    class MultiAgentEnv(_Env):
        possible_agents = ["left", "right"]
        action_spaces = {agent: gym.spaces.Box(-10, 10, shape=(2,)) for agent in possible_agents}
        observation_spaces = {agent: gym.spaces.Box(-10, 10, shape=(1,)) for agent in possible_agents}

        def reset(self, **kwargs):
            super().reset(**kwargs)
            return {agent: torch.zeros(3, 1) for agent in self.possible_agents}, {}

        def step(self, actions):
            super().step(actions)
            obs = {agent: torch.zeros(3, 1) for agent in self.possible_agents}
            rewards = {agent: torch.zeros(3) for agent in self.possible_agents}
            dones = {agent: self.reset_buf.clone() for agent in self.possible_agents}
            return obs, rewards, dones, dones, {}

    env = MultiAgentEnv(DirectMARLEnvCfg(teleop=_cfg(agent="right", env_ids=[1]), state_space=0))
    teleop = TeleopEnvWrapper(env)
    source = env.input
    converted = (
        multi_agent_to_single_agent(teleop) if conversion == "single_agent" else multi_agent_with_one_agent(teleop)
    )
    converted.reset()
    assert source.resets == [[1]]
    source.sample = TeleopSample(torch.tensor([8.0, 9.0]))
    actions = torch.ones(3, 4)
    converted.step(actions if conversion == "single_agent" else {"single-agent": actions})
    torch.testing.assert_close(env.received_actions["left"], torch.ones(3, 2))
    torch.testing.assert_close(env.received_actions["right"], torch.tensor([[1.0, 1.0], [8.0, 9.0], [1.0, 1.0]]))
    converted.close()
    assert env.events[-2:] == ["source_closed", "env_closed"]


def test_multidimensional_noncontiguous_actions_keep_shape_and_unselected_values():
    env = _Env(DirectRLEnvCfg(teleop=_cfg(action_indices=[1])))
    env.single_action_space = gym.spaces.Box(-10, 10, shape=(2, 2))
    wrapped = TeleopEnvWrapper(env)
    env.input.sample = TeleopSample(torch.tensor([9.0]))
    policy = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2).transpose(1, 2)
    wrapped.step(policy)
    assert env.received_actions.shape == policy.shape
    torch.testing.assert_close(env.received_actions[0], torch.tensor([[0.0, 9.0], [1.0, 3.0]]))
    torch.testing.assert_close(env.received_actions[1:], policy[1:])
    assert policy[0, 0, 1] == 2
    wrapped.close()
