# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch

from isaaclab_rl.sb3 import Sb3VecEnvWrapper


def _make_wrapper(fast_variant: bool) -> Sb3VecEnvWrapper:
    wrapper = Sb3VecEnvWrapper.__new__(Sb3VecEnvWrapper)
    wrapper.num_envs = 2
    wrapper.fast_variant = fast_variant
    wrapper._ep_rew_buf = np.array([3.0, 4.0])
    wrapper._ep_len_buf = np.array([5, 6])
    return wrapper


def test_process_extras_uses_terminal_obs_fast_variant():
    wrapper = _make_wrapper(fast_variant=True)
    obs = np.array([[10.0], [20.0]])
    final_obs = np.array([[100.0], [200.0]])

    infos = wrapper._process_extras(
        obs=obs,
        terminated=np.array([True, False]),
        truncated=np.array([False, False]),
        extras={},
        reset_ids=np.array([0]),
        final_obs=final_obs,
    )

    assert np.array_equal(infos[0]["terminal_observation"], np.array([100.0]))


def test_process_extras_uses_terminal_obs_slow_variant():
    wrapper = _make_wrapper(fast_variant=False)
    obs = {"position": np.array([[10.0], [20.0]])}
    final_obs = {"position": np.array([[100.0], [200.0]])}
    extras = {"custom": np.array([7.0, 8.0])}

    infos = wrapper._process_extras(
        obs=obs,
        terminated=np.array([False, False]),
        truncated=np.array([True, False]),
        extras=extras,
        reset_ids=np.array([0]),
        final_obs=final_obs,
    )

    assert np.array_equal(infos[0]["terminal_observation"]["position"], np.array([100.0]))
    assert infos[0]["custom"] == 7.0


class _FakeSameStepEnv:
    def __init__(self):
        self.final_obs = {"policy": {"state": torch.tensor([[100.0], [200.0]])}}

    def step(self, action):
        obs = {"policy": {"state": torch.tensor([[10.0], [20.0]])}}
        rew = torch.tensor([1.0, 2.0])
        terminated = torch.tensor([True, False])
        truncated = torch.tensor([False, False])
        return obs, rew, terminated, truncated, {"final_obs": self.final_obs}


def test_step_wait_uses_final_obs_without_mutating_it():
    wrapper = _make_wrapper(fast_variant=True)
    wrapper.env = _FakeSameStepEnv()
    wrapper._async_actions = torch.zeros(2, 1)
    wrapper.observation_processors = {}

    _, _, _, infos = wrapper.step_wait()

    assert np.array_equal(infos[0]["terminal_observation"]["state"], np.array([100.0]))
    assert isinstance(wrapper.env.final_obs["policy"]["state"], torch.Tensor)
