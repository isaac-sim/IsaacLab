# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ensure every joint-wrench backend collects the same physical contract tests."""

import ast

import pytest

pytestmark = pytest.mark.unit


def test_joint_wrench_contract_registration(source_checkout_root):
    source = source_checkout_root / "source"
    contract = source / "isaaclab/test/sensors/joint_wrench_contract.py"
    tree = ast.parse(contract.read_text())
    tests = {node.name for node in tree.body if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")}
    assert tests
    assert not (source / "isaaclab/isaaclab/test/utils/joint_wrench.py").exists()

    implementations = list(source.glob("isaaclab_*/isaaclab_*/sensors/joint_wrench/joint_wrench_sensor.py"))
    backend_packages = {path.parents[2].name for path in implementations}
    assert backend_packages >= {"isaaclab_newton", "isaaclab_physx", "isaaclab_ov"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        else:
            continue
        assert not {module.split(".")[0] for module in modules} & (backend_packages | {"newton", "ovphysx", "isaacsim"})

    for implementation in implementations:
        suite = implementation.parents[3] / "test/sensors/test_joint_wrench_sensor.py"
        backend_tree = ast.parse(suite.read_text())
        imported_tests = {
            alias.name
            for node in backend_tree.body
            if isinstance(node, ast.ImportFrom) and node.module == "joint_wrench_contract"
            for alias in node.names
            if alias.asname is None
        }
        local_functions = {node.name for node in backend_tree.body if isinstance(node, ast.FunctionDef)}
        assert tests <= imported_tests, f"{suite}: missing shared contract tests"
        assert not tests & local_functions, f"{suite}: shadows shared contract tests"
        assert "test_non_identity_joint_frame_transform" not in local_functions
