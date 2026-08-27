# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-free checks for the observation contract of :class:`RslRlVecEnvWrapper`.

The wrapper reads the environment-owned ``obs_buf`` instead of calling the environment's
private ``_get_observations``. These tests pin that contract without a simulator, so a
regression is caught before the Kit-dependent wrapper tests run.
"""

import inspect

import torch
from tensordict import TensorDict

from isaaclab.envs import DirectRLEnv

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class _FakeEnv:
    """Minimal stand-in exposing only what :meth:`get_observations` reads."""

    def __init__(self):
        self.unwrapped = self
        self.obs_buf = {"policy": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}


def _make_wrapper(env: _FakeEnv, num_envs: int = 2) -> RslRlVecEnvWrapper:
    """Build a wrapper without ``__init__``, which requires a real environment and a live sim."""
    wrapper = object.__new__(RslRlVecEnvWrapper)
    wrapper.env = env
    wrapper.num_envs = num_envs
    return wrapper


def test_get_observations_returns_the_environment_buffer():
    """The wrapper should hand back the environment's own observation buffer."""
    env = _FakeEnv()
    wrapper = _make_wrapper(env)

    observations = wrapper.get_observations()

    assert isinstance(observations, TensorDict)
    torch.testing.assert_close(observations["policy"], env.obs_buf["policy"])


def test_get_observations_tracks_buffer_updates():
    """Successive reads should reflect the latest reset/step observations."""
    env = _FakeEnv()
    wrapper = _make_wrapper(env)

    env.obs_buf = {"policy": torch.tensor([[5.0, 6.0], [7.0, 8.0]])}

    torch.testing.assert_close(wrapper.get_observations()["policy"], env.obs_buf["policy"])


def test_get_observations_does_not_use_private_environment_methods():
    """The wrapper must not reach into the environment's private observation API."""
    env = _FakeEnv()

    def _private_call():
        raise AssertionError("get_observations() must not call the private _get_observations()")

    env._get_observations = _private_call
    wrapper = _make_wrapper(env)

    wrapper.get_observations()


def test_direct_rl_env_stores_the_observation_buffer():
    """The environment side of the contract: reset and step both publish ``obs_buf``."""
    assert "self.obs_buf" in inspect.getsource(DirectRLEnv.reset)
    assert "self.obs_buf" in inspect.getsource(DirectRLEnv.step)
