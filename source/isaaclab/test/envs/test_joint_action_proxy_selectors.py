# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import warp as wp

from isaaclab.utils.warp import ProxyArray

_REPO_ROOT = Path(__file__).parents[4]
_ACTION_FILES = {
    "joint": _REPO_ROOT / "source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py",
    "limits": _REPO_ROOT / "source/isaaclab/isaaclab/envs/mdp/actions/joint_actions_to_limits.py",
}
_WRITER_PATHS = (
    ("joint", "JointPositionAction", "set_joint_position_target_index"),
    ("joint", "JointVelocityAction", "set_joint_velocity_target_index"),
    ("joint", "JointEffortAction", "set_joint_effort_target_index"),
    ("limits", "JointPositionToLimitsAction", "set_joint_position_target_index"),
)


def _class_node(file_key: str, class_name: str) -> ast.ClassDef:
    tree = ast.parse(_ACTION_FILES[file_key].read_text())
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)


def _method(file_key: str, class_name: str, method_name: str):
    class_node = _class_node(file_key, class_name)
    method = next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"torch": torch}
    exec(compile(module, str(_ACTION_FILES[file_key]), "exec"), namespace)
    return namespace[method_name]


@pytest.mark.parametrize(
    ("file_key", "class_name"),
    (("joint", "JointAction"), ("limits", "JointPositionToLimitsAction")),
)
def test_repeated_joint_actions_request_proxy_and_keep_full_slice_fast_path(file_key: str, class_name: str) -> None:
    init = next(
        node
        for node in _class_node(file_key, class_name).body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    finder_call = next(
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "find_joints"
    )
    as_proxy = next(keyword.value for keyword in finder_call.keywords if keyword.arg == "as_proxy")

    assert isinstance(as_proxy, ast.Constant) and as_proxy.value is True
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "slice"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value is None
        for node in ast.walk(init)
    )


@pytest.mark.parametrize(("file_key", "class_name", "writer_name"), _WRITER_PATHS)
def test_joint_action_writer_keeps_proxy_without_torch_materialization(
    file_key: str, class_name: str, writer_name: str
) -> None:
    calls = []

    def writer(**kwargs):
        calls.append(kwargs)

    body_ids_array = wp.array([2, 0], dtype=wp.int32, device="cpu")
    body_ids = ProxyArray(body_ids_array)
    action = SimpleNamespace(
        processed_actions=torch.ones((2, 2)),
        _joint_ids=body_ids,
        _asset=SimpleNamespace(**{writer_name: writer}),
    )

    _method(file_key, class_name, "apply_actions")(action)

    assert calls == [{"target": action.processed_actions, "joint_ids": body_ids}]
    assert body_ids._torch_cache is None


@pytest.mark.parametrize(
    ("file_key", "class_name", "method_name"),
    (
        ("joint", "JointPositionAction", "__init__"),
        ("joint", "RelativeJointPositionAction", "apply_actions"),
        ("joint", "JointVelocityAction", "__init__"),
        ("limits", "JointPositionToLimitsAction", "process_actions"),
        ("limits", "EMAJointPositionToLimitsAction", "reset"),
        ("limits", "EMAJointPositionToLimitsAction", "process_actions"),
    ),
)
def test_joint_action_torch_indexing_uses_explicit_proxy_view(file_key: str, class_name: str, method_name: str) -> None:
    class_node = _class_node(file_key, class_name)
    method = next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    source = ast.get_source_segment(_ACTION_FILES[file_key].read_text(), method)

    assert "_joint_ids" in source
    assert "_joint_ids.torch" in source
