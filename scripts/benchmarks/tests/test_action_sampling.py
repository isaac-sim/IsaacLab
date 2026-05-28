# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :func:`scripts.benchmarks._action_sampling.sample_random_actions`."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from scripts.benchmarks._action_sampling import sample_random_actions


@dataclass
class _BoxSpace:
    """Minimal stand-in for ``gym.spaces.Box``. Only needs ``.sample()`` for
    these tests; we duck-type to avoid pulling gymnasium into the unit-test
    path (gymnasium is installed inside Isaac Sim's python, not the system
    one used by ``python3 -m pytest``)."""

    low: float
    high: float
    shape: tuple

    def sample(self) -> np.ndarray:
        return np.random.uniform(low=self.low, high=self.high, size=self.shape).astype(np.float32)


def _box(low: float = -1.0, high: float = 1.0, shape: tuple = (3,)) -> _BoxSpace:
    return _BoxSpace(low=low, high=high, shape=shape)


@dataclass
class _FakeSingleAgentEnv:
    """Mimic the unwrapped surface of a DirectRLEnv / ManagerBasedRLEnv."""

    num_envs: int = 4
    device: str = "cpu"
    single_action_space: _BoxSpace = field(default_factory=_box)

    @property
    def unwrapped(self):
        return self


@dataclass
class _FakeMARLEnv:
    """Mimic the unwrapped surface of a DirectMARLEnv."""

    num_envs: int = 4
    device: str = "cpu"
    action_spaces: dict = field(
        default_factory=lambda: {
            "cart": _box(shape=(1,)),
            "pendulum": _box(shape=(1,)),
        }
    )

    @property
    def unwrapped(self):
        return self


def test_sample_random_actions_single_agent_returns_stacked_tensor():
    """Single-agent envs must get one tensor of shape ``(num_envs, action_dim)`` —
    ``env.step`` of a DirectRLEnv expects a single tensor, not a dict."""
    env = _FakeSingleAgentEnv(num_envs=8)
    actions = sample_random_actions(env)
    assert isinstance(actions, torch.Tensor)
    assert actions.shape == (8, 3)
    assert actions.dtype == torch.float32


def test_sample_random_actions_multi_agent_returns_dict():
    """Multi-agent envs must get a dict ``{agent_id: tensor}`` — that's the
    shape ``DirectMARLEnv.step`` accepts. The previous code path called
    ``unwrapped.single_action_space.sample()`` and crashed with
    ``AttributeError: 'CartDoublePendulumEnv' object has no attribute
    'single_action_space'`` on every multi-agent benchmark run."""
    env = _FakeMARLEnv(num_envs=4)
    actions = sample_random_actions(env)
    assert isinstance(actions, dict)
    assert set(actions) == {"cart", "pendulum"}
    for agent, tensor in actions.items():
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (4, 1)
        assert tensor.dtype == torch.float32


def test_sample_random_actions_multi_agent_handles_heterogeneous_action_dims():
    """Per-agent action spaces can have different shapes — the helper must
    sample each space at its own dimensionality, not assume a uniform
    shape across agents."""
    env = _FakeMARLEnv(
        num_envs=2,
        action_spaces={
            "small": _box(shape=(1,)),
            "large": _box(shape=(7,)),
        },
    )
    actions = sample_random_actions(env)
    assert actions["small"].shape == (2, 1)
    assert actions["large"].shape == (2, 7)


def test_sample_random_actions_multi_agent_samples_within_space_bounds():
    """Sanity-check that the sampled values come from the declared Box —
    catches a regression where someone replaces ``space.sample()`` with
    e.g. zeros."""
    env = _FakeMARLEnv(
        num_envs=16,
        action_spaces={
            "agent": _box(low=-2.0, high=2.0, shape=(1,)),
        },
    )
    actions = sample_random_actions(env)
    a = actions["agent"]
    assert (a >= -2.0).all() and (a <= 2.0).all()


def test_sample_random_actions_uses_env_device_for_returned_tensors():
    """Per the original code, the returned tensors live on
    ``env.device``; otherwise ``env.step(actions)`` will copy from CPU
    to GPU on every benchmark run and skew the timing."""
    env = _FakeSingleAgentEnv(device="cpu")  # MPS/CUDA not assumed in tests
    actions = sample_random_actions(env)
    assert str(actions.device) == "cpu"


def test_sample_random_actions_passes_through_gym_wrappers():
    """The benchmark runs against a ``gym.make()``-wrapped env; the
    action-space discriminator must read off ``env.unwrapped`` rather
    than the wrapper, otherwise a single-agent gym.Wrapper exposing a
    legacy ``action_spaces`` attribute (Wrapper has none, but
    defensive) wouldn't trick us into the MARL branch."""

    @dataclass
    class _Wrapper:
        inner: object

        @property
        def unwrapped(self):
            return self.inner

    env = _Wrapper(inner=_FakeSingleAgentEnv())
    actions = sample_random_actions(env)
    assert isinstance(actions, torch.Tensor)


def test_sample_random_actions_marl_per_env_independence():
    """Each row in the per-agent action tensor must be an independent
    sample — i.e., sampling N times produces N (likely) distinct rows.
    A regression where the loop replaced ``range(num_envs)`` with a
    single sample broadcasted across rows would slip past the shape
    check but produce trivially correlated actions across envs."""
    env = _FakeMARLEnv(
        num_envs=64,
        action_spaces={"a": _box(shape=(2,))},
    )
    actions = sample_random_actions(env)
    a = actions["a"].numpy()
    # With 64 i.i.d. samples from a continuous Box, np.unique row count
    # is overwhelmingly likely to be 64. Allow some slack just in case
    # of pathological RNG state.
    assert len({tuple(r) for r in a}) >= 60
