# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checks for the observation contract of :class:`RslRlVecEnvWrapper`."""

import pytest
import torch
from tensordict import TensorDict

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class _FakeEnv:
    """Minimal stand-in exposing only what :meth:`get_observations` reads."""

    def __init__(self):
        self.unwrapped = self
        self.obs_buf = {"policy": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}


class _UnsupportedEnv:
    pass


class _OuterEnv:
    def __init__(self):
        self.unwrapped = _UnsupportedEnv()


def _make_wrapper(env: _FakeEnv, num_envs: int = 2) -> RslRlVecEnvWrapper:
    """Build a wrapper without ``__init__``, which requires a real environment and a live sim."""
    wrapper = object.__new__(RslRlVecEnvWrapper)
    wrapper.env = env
    wrapper.num_envs = num_envs
    return wrapper


def test_get_observations_tracks_the_environment_buffer():
    """The wrapper returns the environment's current observation buffer."""
    env = _FakeEnv()
    wrapper = _make_wrapper(env)

    observations = wrapper.get_observations()
    assert isinstance(observations, TensorDict)
    torch.testing.assert_close(observations["policy"], env.obs_buf["policy"])

    env.obs_buf = {"policy": torch.tensor([[5.0, 6.0], [7.0, 8.0]])}
    torch.testing.assert_close(wrapper.get_observations()["policy"], env.obs_buf["policy"])


def test_invalid_environment_error_reports_unwrapped_type():
    """The validation error should identify the object that failed the type check."""
    with pytest.raises(ValueError) as exc_info:
        RslRlVecEnvWrapper(_OuterEnv())

    assert "_UnsupportedEnv" in str(exc_info.value)
    assert "_OuterEnv" not in str(exc_info.value)
