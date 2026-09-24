# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Unit tests for direct multi-agent reinforcement-learning environments."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.markers.vis_marker_registry import VisMarkerRegistry
from isaaclab.test.env_cfgs import make_empty_direct_marl_env_cfg
from isaaclab.utils.noise import ConstantNoiseCfg, NoiseModelWithAdditiveBias, NoiseModelWithAdditiveBiasCfg

pytestmark = pytest.mark.unit


class _StubMARLEnv(DirectMARLEnv):
    """Direct MARL environment stub that skips simulator initialization."""

    def __init__(self, cfg: DirectMARLEnvCfg) -> None:
        self._is_closed = True
        self.cfg = cfg
        self.scene = SimpleNamespace(num_envs=cfg.scene.num_envs)
        self.sim = SimpleNamespace(device=cfg.sim.device)


def test_agent_and_space_configuration():
    """Agent counts and spaces are configured without initializing the simulator."""
    env = _StubMARLEnv(make_empty_direct_marl_env_cfg(device="cpu"))

    env._configure_env_spaces()

    assert env.agents == ["agent_0", "agent_1"]
    assert env.possible_agents == ["agent_0", "agent_1"]
    assert env.num_agents == 2
    assert env.max_num_agents == 2
    assert len(env.observation_spaces) == 2
    assert len(env.action_spaces) == 2
    assert all(isinstance(space, gym.spaces.Box) for space in env.observation_spaces.values())
    assert all(isinstance(space, gym.spaces.Box) for space in env.action_spaces.values())
    assert env.observation_spaces["agent_0"].shape == (3,)
    assert env.observation_spaces["agent_1"].shape == (4,)
    assert env.action_spaces["agent_0"].shape == (1,)
    assert env.action_spaces["agent_1"].shape == (2,)
    assert isinstance(env.state_space, gym.spaces.Box)
    assert env.state_space.shape == (7,)


def test_zero_state_space_disables_centralized_state():
    """A zero state-space follows the documented no-centralized-state contract."""
    cfg = make_empty_direct_marl_env_cfg(device="cpu")
    cfg.state_space = 0
    env = _StubMARLEnv(cfg)

    env._configure_env_spaces()

    assert env.state_space is None


@pytest.mark.parametrize(
    "noisy_agents,per_component",
    [
        ((), False),
        (("agent_0",), False),
        (("agent_0", "agent_1"), False),
        (("agent_0",), True),
        (("agent_0", "agent_1"), True),
    ],
)
def test_reset_applies_observation_noise_per_agent(noisy_agents, per_component):
    """Reset applies each configured noise model after its episode bias has been reset."""
    cfg = make_empty_direct_marl_env_cfg(device="cpu", num_envs=2)
    cfg.observation_noise_model = {
        agent: NoiseModelWithAdditiveBiasCfg(
            noise_cfg=ConstantNoiseCfg(bias=0.5),
            bias_noise_cfg=ConstantNoiseCfg(bias=float(index + 1), operation="abs"),
        )
        for index, agent in enumerate(noisy_agents)
    } or None
    if not per_component:
        for noise_cfg in (cfg.observation_noise_model or {}).values():
            noise_cfg.sample_bias_per_component = False
    env = _StubMARLEnv(cfg)
    env._configure_env_spaces()
    env.scene.reset = lambda ids: None
    env.sim.render_context = SimpleNamespace(reset_scene_state_cadence=lambda: None)
    env.episode_length_buf = torch.ones(2, dtype=torch.long)
    env.extras = {agent: {} for agent in cfg.possible_agents}
    # Observations need not arrive in the order declared by possible_agents.
    env._get_observations = lambda: {"agent_1": torch.zeros(2, 4), "agent_0": torch.zeros(2, 3)}
    env._observation_noise_model = {
        agent: NoiseModelWithAdditiveBias(noise_cfg, num_envs=2, device="cpu")
        for agent, noise_cfg in (cfg.observation_noise_model or {}).items()
    }

    for episode in (1, 2):
        for index, noise_cfg in enumerate((cfg.observation_noise_model or {}).values()):
            noise_cfg.bias_noise_cfg.bias = float(episode * (index + 1))
        observations, extras = env.reset()

        for index, agent in enumerate(cfg.possible_agents):
            expected = episode * (index + 1) + 0.5 if agent in noisy_agents else 0.0
            torch.testing.assert_close(observations[agent], torch.full((2, index + 3), expected))
        assert env.agents == cfg.possible_agents
        assert observations is env.obs_dict
        assert extras is env.extras


class _DebugVisStubMARLEnv(_StubMARLEnv):
    """Stub whose debug visualization is implemented, so ``set_debug_vis`` runs its handle logic."""

    def __init__(self, cfg: DirectMARLEnvCfg) -> None:
        super().__init__(cfg)
        # mirrors what DirectMARLEnv.__init__ derives, which the stub skips
        self.has_debug_vis_implementation = "NotImplementedError" not in inspect.getsource(self._set_debug_vis_impl)
        self._debug_vis_handle = None
        self.sim = SimpleNamespace(device=cfg.sim.device, vis_marker_registry=VisMarkerRegistry())
        self.callback_count = 0

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        pass

    def _debug_vis_callback(self, event) -> None:
        self.callback_count += 1


def test_set_debug_vis_registers_without_kit():
    """Debug visualization registers through the marker registry, so it needs no Kit application.

    Guards against reintroducing the deprecated ``IApp.get_post_update_event_stream`` subscription,
    which raised ``NameError`` in kitless mode because ``omni.kit.app`` is only imported when Kit is
    present.
    """
    env = _DebugVisStubMARLEnv(make_empty_direct_marl_env_cfg(device="cpu"))
    registry = env.sim.vis_marker_registry

    assert env.set_debug_vis(True) is True
    assert isinstance(env._debug_vis_handle, str)

    registry.dispatch_callbacks()
    assert env.callback_count == 1

    env.set_debug_vis(False)
    assert env._debug_vis_handle is None

    registry.dispatch_callbacks()
    assert env.callback_count == 1
