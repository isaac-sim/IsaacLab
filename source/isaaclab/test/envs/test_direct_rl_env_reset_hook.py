# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from types import SimpleNamespace

import torch

from isaaclab.envs import DirectRLEnv


def test_default_reset_hook_preserves_index_reset() -> None:
    env = object.__new__(DirectRLEnv)
    env._is_closed = True
    env.reset_buf = torch.tensor([False, True, False, True])
    env.cfg = SimpleNamespace(compute_final_obs=False)
    reset_calls = []
    env._reset_idx = lambda env_ids: reset_calls.append(env_ids.clone())

    reset_env_ids = env._reset_envs_from_buffer()

    torch.testing.assert_close(reset_env_ids, torch.tensor([1, 3], dtype=torch.int32))
    assert len(reset_calls) == 1
    torch.testing.assert_close(reset_calls[0], reset_env_ids)


def test_default_reset_hook_preserves_final_observation() -> None:
    env = object.__new__(DirectRLEnv)
    env._is_closed = True
    env.reset_buf = torch.tensor([False, True])
    env.cfg = SimpleNamespace(compute_final_obs=True, observation_noise_model=object())
    env.extras = {}
    env._get_observations = lambda: {"policy": torch.tensor([[1.0], [2.0]])}
    env._observation_noise_model = lambda observation: observation + 0.5
    reset_calls = []
    env._reset_idx = lambda env_ids: reset_calls.append(env_ids.clone())

    reset_env_ids = env._reset_envs_from_buffer()

    torch.testing.assert_close(env.extras["final_obs"]["policy"], torch.tensor([[1.5], [2.5]]))
    assert len(reset_calls) == 1
    torch.testing.assert_close(reset_calls[0], reset_env_ids)
