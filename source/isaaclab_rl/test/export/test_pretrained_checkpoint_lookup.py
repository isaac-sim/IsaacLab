# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for backend-aware pretrained checkpoint lookup in LEAPP exporters."""

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_LEAPP_ROOT = _REPO_ROOT / "scripts" / "reinforcement_learning" / "leapp"


@pytest.mark.parametrize("rl_library", ["rl_games", "rsl_rl", "sb3", "skrl"])
def test_exporter_resolves_pretrained_checkpoint_for_active_backends(rl_library: str):
    """Each exporter must request the checkpoint of its resolved environment config, which carries the backends."""
    export_path = _LEAPP_ROOT / rl_library / "export.py"
    tree = ast.parse(export_path.read_text(encoding="utf-8"), filename=str(export_path))

    checkpoint_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "get_published_pretrained_checkpoint"
        and len(node.args) == 2
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == rl_library
        and any(keyword.arg == "env_cfg" for keyword in node.keywords)
    ]

    assert checkpoint_calls
