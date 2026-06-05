# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for RSL-RL recurrent state handling in LEAPP export."""

import importlib.util
import sys
import types
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]
_EXPORT_SCRIPT = _REPO_ROOT / "scripts" / "reinforcement_learning" / "leapp" / "rsl_rl" / "export.py"
_EXPORT_MODULE_NAME = "_isaaclab_rsl_rl_leapp_export_recurrent_state"


def _load_export_module():
    """Load the LEAPP RSL-RL export script as an importable module."""
    module = sys.modules.get(_EXPORT_MODULE_NAME)
    if module is not None and hasattr(module, "ensure_actor_hidden_state_initialized"):
        return module

    sys.modules.pop(_EXPORT_MODULE_NAME, None)
    spec = importlib.util.spec_from_file_location(_EXPORT_MODULE_NAME, _EXPORT_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {_EXPORT_SCRIPT}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[_EXPORT_MODULE_NAME] = module
    original_modules = {
        name: sys.modules.get(name) for name in ("isaaclab", "isaaclab.app", "isaaclab_tasks", "isaaclab_tasks.utils")
    }
    isaaclab_module = types.ModuleType("isaaclab")
    isaaclab_app_module = types.ModuleType("isaaclab.app")
    isaaclab_tasks_module = types.ModuleType("isaaclab_tasks")
    isaaclab_tasks_utils_module = types.ModuleType("isaaclab_tasks.utils")

    class _AppLauncher:
        @staticmethod
        def add_app_launcher_args(parser):
            return None

    setattr(isaaclab_app_module, "AppLauncher", _AppLauncher)
    setattr(isaaclab_tasks_utils_module, "fold_preset_tokens", lambda args: args)
    setattr(isaaclab_tasks_utils_module, "setup_preset_cli", lambda parser, argv=None: parser.parse_known_args(argv))
    sys.modules["isaaclab"] = isaaclab_module
    sys.modules["isaaclab.app"] = isaaclab_app_module
    sys.modules["isaaclab_tasks"] = isaaclab_tasks_module
    sys.modules["isaaclab_tasks.utils"] = isaaclab_tasks_utils_module
    try:
        spec.loader.exec_module(module)
    finally:
        for name, original_module in original_modules.items():
            if original_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_module
    setattr(module, "torch", torch)
    return module


class _LegacyMemory(torch.nn.Module):
    """Minimal RSL-RL 3.x recurrent memory shape."""

    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.LSTM(input_size=2, hidden_size=4, num_layers=2)
        self.hidden_state = None


class _LegacyRecurrentPolicy(torch.nn.Module):
    """Minimal RSL-RL 3.x ActorCriticRecurrent shape."""

    is_recurrent = True

    def __init__(self):
        super().__init__()
        self.memory_a = _LegacyMemory()

    def get_hidden_states(self):
        return self.memory_a.hidden_state, None


class _ModularRNN(torch.nn.Module):
    """Minimal RSL-RL 5.x RNN wrapper shape."""

    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.LSTM(input_size=2, hidden_size=4, num_layers=2)
        self.hidden_state = None


class _ModularRecurrentPolicy(torch.nn.Module):
    """Minimal RSL-RL 5.x RNNModel shape."""

    is_recurrent = True

    def __init__(self):
        super().__init__()
        self.rnn = _ModularRNN()

    def get_hidden_state(self):
        return self.rnn.hidden_state


def test_recurrent_state_helpers_support_legacy_actor_critic_lstm():
    """Verify LSTM state registration helpers support RSL-RL 3.x ActorCriticRecurrent."""
    export_module = _load_export_module()
    policy = _LegacyRecurrentPolicy()

    actor_state = export_module.ensure_actor_hidden_state_initialized(
        policy, batch_size=1, device=torch.device("cpu"), dtype=torch.float32
    )

    assert export_module.is_actor_recurrent_policy(policy)
    assert actor_state is policy.memory_a.hidden_state
    assert [tensor.shape for tensor in actor_state] == [(2, 1, 4), (2, 1, 4)]
    assert list(export_module.state_dict_from_actor_hidden(actor_state)) == ["actor_state_0", "actor_state_1"]


def test_recurrent_state_helpers_support_modular_rnn_model_lstm():
    """Verify LSTM state registration helpers support RSL-RL 5.x RNNModel."""
    export_module = _load_export_module()
    policy = _ModularRecurrentPolicy()

    actor_state = export_module.ensure_actor_hidden_state_initialized(
        policy, batch_size=1, device=torch.device("cpu"), dtype=torch.float32
    )
    registered_state = tuple(tensor + 1.0 for tensor in actor_state)
    export_module.set_actor_hidden_state(
        policy,
        export_module.actor_hidden_from_registered(registered_state, actor_state),
    )

    assert export_module.is_actor_recurrent_policy(policy)
    assert export_module.get_actor_memory_module(policy) is policy.rnn
    assert export_module.get_actor_hidden_state(policy) is registered_state
    assert policy.rnn.hidden_state is registered_state


def test_policy_module_helper_supports_bound_method_and_module_policy():
    """Verify inference policies from RSL-RL 3.x and 5.x resolve to their owning module."""
    export_module = _load_export_module()
    legacy_policy = _LegacyRecurrentPolicy()
    modular_policy = _ModularRecurrentPolicy()

    assert export_module.get_policy_module(legacy_policy.forward) is legacy_policy
    assert export_module.get_policy_module(modular_policy) is modular_policy
