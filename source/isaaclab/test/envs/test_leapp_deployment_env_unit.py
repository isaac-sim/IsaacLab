# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import types

import torch

from isaaclab.envs.leapp_deployment_env import LeappDeploymentEnv


class _Scene:
    def __init__(self):
        self.env_ids = None

    def reset(self, env_ids):
        self.env_ids = env_ids

    def write_data_to_sim(self):
        pass

    def update(self, dt):
        pass


class _EventManager:
    available_modes = ["reset"]

    def __init__(self):
        self.env_ids = None

    def apply(self, *, mode, env_ids, global_env_step_count):
        self.env_ids = env_ids
        assert tuple(env_ids[:, None].shape) == (1, 1)


class _CommandManager:
    def __init__(self):
        self.env_ids = None

    def reset(self, env_ids):
        self.env_ids = env_ids


class _Inference:
    def reset(self):
        pass


def test_reset_uses_tensor_env_ids_for_managers():
    env = LeappDeploymentEnv.__new__(LeappDeploymentEnv)
    env.has_rtx_sensors = False
    env.cfg = types.SimpleNamespace(
        decimation=2, sim=types.SimpleNamespace(dt=0.01), num_rerenders_on_reset=0, wait_for_textures=False
    )
    env._step_count = 0
    env.scene = _Scene()
    env.event_manager = _EventManager()
    env.command_manager = _CommandManager()
    env.sim = types.SimpleNamespace(device="cpu", forward=lambda: None)
    env._read_inputs = lambda: {}
    env.inference = _Inference()

    env.reset()

    assert torch.equal(env.scene.env_ids, torch.tensor([0], dtype=torch.int32))
    assert torch.equal(env.event_manager.env_ids, torch.tensor([0], dtype=torch.int32))
    assert torch.equal(env.command_manager.env_ids, torch.tensor([0], dtype=torch.int32))
