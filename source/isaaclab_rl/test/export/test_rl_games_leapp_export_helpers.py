# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from isaaclab_rl.leapp import is_two_tensor_lstm_state

_REPO_ROOT = Path(__file__).resolve().parents[4]
_EXPORT_SCRIPT = _REPO_ROOT / "scripts" / "reinforcement_learning" / "leapp" / "rl_games" / "export.py"
_EXPORT_MODULE_NAME = "_isaaclab_rl_games_leapp_export_helpers"


def _load_export_module():
    """Load RL-Games export.py without importing Isaac Sim runtime modules."""
    sys.modules.pop(_EXPORT_MODULE_NAME, None)
    spec = importlib.util.spec_from_file_location(_EXPORT_MODULE_NAME, _EXPORT_SCRIPT)
    module = importlib.util.module_from_spec(spec)

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
    setattr(isaaclab_tasks_utils_module, "setup_preset_cli", lambda parser, argv=None: parser.parse_known_args(argv))
    sys.modules["isaaclab"] = isaaclab_module
    sys.modules["isaaclab.app"] = isaaclab_app_module
    sys.modules["isaaclab_tasks"] = isaaclab_tasks_module
    sys.modules["isaaclab_tasks.utils"] = isaaclab_tasks_utils_module
    try:
        sys.modules[_EXPORT_MODULE_NAME] = module
        spec.loader.exec_module(module)
    finally:
        for name, original_module in original_modules.items():
            if original_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_module

    module.is_two_tensor_lstm_state = is_two_tensor_lstm_state
    return module


class _TinyRlGamesModel:
    """Minimal RL-Games model with LSTM default state shape."""

    def get_default_rnn_state(self):
        return [torch.zeros(1, 1, 7), torch.zeros(1, 1, 7)]


class _TinyRlGamesPlayer:
    """Minimal RL-Games player state container."""

    is_rnn = True
    batch_size = 1
    device = "cpu"
    model = _TinyRlGamesModel()

    def init_rnn(self):
        rnn_states = self.model.get_default_rnn_state()
        self.states = [
            torch.zeros((state.size()[0], self.batch_size, state.size()[2]), dtype=torch.float32).to(self.device)
            for state in rnn_states
        ]


def test_rl_games_lstm_feedback_detection():
    """Verify RL-Games LSTM feedback can be detected from player state."""
    export_module = _load_export_module()
    agent = _TinyRlGamesPlayer()
    agent.init_rnn()

    assert export_module.is_rl_games_lstm_policy(agent)
    assert [tuple(tensor.shape) for tensor in export_module.get_rl_games_policy_states(agent)] == [
        (1, 1, 7),
        (1, 1, 7),
    ]


def test_rl_games_recurrent_non_lstm_is_rejected():
    """Verify recurrent RL-Games policies without two LSTM tensors are rejected."""
    export_module = _load_export_module()
    agent = types.SimpleNamespace(is_rnn=True, states=[torch.zeros(1, 1, 7)])

    assert not export_module.is_rl_games_lstm_policy(agent)
    with pytest.raises(NotImplementedError, match="Only RL-Games LSTM"):
        export_module._validate_rl_games_recurrent_support(agent)
