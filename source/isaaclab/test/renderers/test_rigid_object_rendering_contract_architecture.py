# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Architecture gates for the shared rigid-object rendering contract."""

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CONTRACT = Path(__file__).with_name("rigid_object_rendering_contract.py")
_ADAPTERS = (
    _REPO_ROOT / "source/isaaclab_physx/test/renderers/test_isaac_rtx_renderer_rigid_object_rendering.py",
    _REPO_ROOT / "source/isaaclab_newton/test/renderers/test_newton_warp_renderer_rigid_object_rendering.py",
    _REPO_ROOT / "source/isaaclab_ov/test/test_ovrtx_renderer_rigid_object_rendering.py",
)


def _imported_modules(tree: ast.AST) -> set[str]:
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def test_shared_contract_does_not_depend_on_backend_packages() -> None:
    """Optional backends depend on the contract; the core contract never points back."""
    tree = ast.parse(_CONTRACT.read_text(encoding="utf-8"))
    forbidden_roots = {"isaaclab_newton", "isaaclab_ov", "isaaclab_ovphysx", "isaaclab_physx", "newton", "ovrtx"}
    imported_roots = {module.split(".", 1)[0] for module in _imported_modules(tree)}

    assert imported_roots.isdisjoint(forbidden_roots), imported_roots & forbidden_roots


def test_backend_adapters_do_not_duplicate_scene_ownership() -> None:
    """Adapters may select backends, but scene construction and assertions stay shared."""
    for adapter in _ADAPTERS:
        tree = ast.parse(adapter.read_text(encoding="utf-8"))
        modules = _imported_modules(tree)
        forbidden = {
            module
            for module in modules
            if module == "isaaclab.assets"
            or module.startswith("isaaclab.assets.")
            or module == "isaaclab.scene"
            or module.startswith("isaaclab.scene.")
            or module == "isaaclab.sensors"
            or module.startswith("isaaclab.sensors.")
        }
        contract_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_rigid_object_scale_and_pose_rendering_contract"
        ]

        assert not forbidden, f"{adapter.relative_to(_REPO_ROOT)} duplicates contract ownership: {forbidden}"
        assert len(contract_calls) == 1, f"{adapter.relative_to(_REPO_ROOT)} must compose the shared contract once"
        assert not any(isinstance(node, ast.ClassDef) for node in ast.walk(tree)), (
            f"{adapter.relative_to(_REPO_ROOT)} must not define backend-local scene classes"
        )
