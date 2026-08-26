# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import gymnasium as gym
import torch
from isaaclab_policy_debug.scenario import (
    ManagerBasedSeededScenarioAdapter,
    PolicyDebugScenarioAdapter,
    resolve_scenario_adapter,
)


class _RegisteredAdapter(PolicyDebugScenarioAdapter):
    def reset_synchronized(self, env, env_ids):
        return None


def test_adapter_resolution_prefers_explicit_instance():
    explicit = _RegisteredAdapter()
    assert resolve_scenario_adapter("unused", explicit) is explicit


def test_adapter_resolution_uses_registered_entry_point(monkeypatch):
    monkeypatch.setattr(
        gym,
        "spec",
        lambda _task_id: SimpleNamespace(kwargs={"policy_debug_adapter_entry_point": _RegisteredAdapter}),
    )
    assert isinstance(resolve_scenario_adapter("registered"), _RegisteredAdapter)


def test_adapter_resolution_falls_back_to_verified_manager_adapter(monkeypatch):
    monkeypatch.setattr(gym, "spec", lambda _task_id: SimpleNamespace(kwargs={}))
    assert isinstance(resolve_scenario_adapter("generic"), ManagerBasedSeededScenarioAdapter)


def test_manager_adapter_passes_tensor_environment_ids_to_reset_to():
    class _Scene:
        def get_state(self, is_relative):
            assert is_relative
            return {
                "articulation": {
                    "robot": {
                        "joint_pos": torch.ones(3, 2),
                    }
                }
            }

    class _Base:
        device = "cpu"
        scene = _Scene()
        command_manager = None

        def __init__(self):
            self.reset_to_ids = None

        def reset(self, seed):
            return None

        def reset_to(self, state, env_ids, seed, is_relative):
            self.reset_to_ids = env_ids

    base = _Base()
    env = SimpleNamespace(unwrapped=base)

    ManagerBasedSeededScenarioAdapter().reset_synchronized(env, [0, 2])

    assert isinstance(base.reset_to_ids, torch.Tensor)
    assert base.reset_to_ids.dtype == torch.long
    assert base.reset_to_ids.tolist() == [0, 2]
