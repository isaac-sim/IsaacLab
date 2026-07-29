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

_REPO_ROOT = Path(__file__).resolve().parents[4]
_EXPORT_UTILS_SCRIPT = _REPO_ROOT / "scripts" / "reinforcement_learning" / "leapp" / "export_utils.py"
_EXPORT_UTILS_MODULE_NAME = "_isaaclab_leapp_export_utils"


def _load_export_utils_module():
    """Load shared LEAPP export helpers from the scripts tree."""
    sys.modules.pop(_EXPORT_UTILS_MODULE_NAME, None)
    spec = importlib.util.spec_from_file_location(_EXPORT_UTILS_MODULE_NAME, _EXPORT_UTILS_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_EXPORT_UTILS_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def test_lstm_state_detection_requires_two_tensors():
    """Check that only two-tensor recurrent state is treated as LSTM feedback."""
    export_utils = _load_export_utils_module()
    h = torch.zeros(1, 1, 4)
    c = torch.zeros(1, 1, 4)

    assert export_utils.is_two_tensor_lstm_state([h, c])
    assert export_utils.is_two_tensor_lstm_state((h, c))
    assert not export_utils.is_two_tensor_lstm_state([h])
    assert not export_utils.is_two_tensor_lstm_state([h, c, c])
    assert not export_utils.is_two_tensor_lstm_state([h, object()])


def test_state_sequence_round_trip_from_dict():
    """Check named LEAPP state maps back to framework state order."""
    export_utils = _load_export_utils_module()
    states = [torch.zeros(1, 1, 4), torch.ones(1, 1, 4)]
    state_dict = export_utils.state_dict_from_sequence(states)

    restored = export_utils.state_sequence_from_registered(state_dict, list(state_dict.keys()), states)

    assert list(state_dict.keys()) == ["actor_state_0", "actor_state_1"]
    assert restored == states


def test_common_export_args_include_shared_flags():
    """Shared export args must cover the flags common to every backend."""
    import argparse

    export_utils = _load_export_utils_module()
    parser = argparse.ArgumentParser()

    class _AppLauncher:
        @staticmethod
        def add_app_launcher_args(parser):
            return None

    isaaclab_module = types.ModuleType("isaaclab")
    isaaclab_app_module = types.ModuleType("isaaclab.app")
    setattr(isaaclab_app_module, "AppLauncher", _AppLauncher)
    original_modules = {name: sys.modules.get(name) for name in ("isaaclab", "isaaclab.app")}
    sys.modules["isaaclab"] = isaaclab_module
    sys.modules["isaaclab.app"] = isaaclab_app_module
    try:
        export_utils.add_common_export_args(parser, agent_default="skrl_cfg_entry_point")
    finally:
        for name, original_module in original_modules.items():
            if original_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_module

    actions = {action.dest for action in parser._actions}
    assert {
        "task",
        "agent",
        "checkpoint",
        "use_pretrained_checkpoint",
        "export_task_name",
        "export_method",
        "export_save_path",
        "validation_steps",
        "disable_graph_visualization",
    }.issubset(actions)
    assert "seed" not in actions
