# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests that schema configuration modules do not import backends inside functions."""

import ast
from pathlib import Path

_ISAACLAB_ROOT = Path(__file__).parents[2] / "isaaclab"
_CONFIG_EXPORT_MODULES = (
    _ISAACLAB_ROOT / "sim" / "__init__.py",
    _ISAACLAB_ROOT / "sim" / "schemas" / "__init__.py",
    _ISAACLAB_ROOT / "sim" / "schemas" / "schemas_cfg.py",
    _ISAACLAB_ROOT / "sim" / "spawners" / "materials" / "__init__.py",
    _ISAACLAB_ROOT / "sim" / "spawners" / "materials" / "physics_materials_cfg.py",
)
_BACKEND_PREFIXES = ("isaaclab_newton", "isaaclab_physx", "isaaclab_ovphysx")


def test_config_export_modules_have_no_function_local_backend_imports():
    """Keep backend resolution declarative and out of function bodies."""
    violations = []
    for path in _CONFIG_EXPORT_MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for function in (node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))):
            for node in ast.walk(function):
                module = node.module if isinstance(node, ast.ImportFrom) else None
                names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
                imported_modules = ([module] if module is not None else []) + names
                if any(name.startswith(_BACKEND_PREFIXES) for name in imported_modules):
                    violations.append(f"{path.relative_to(_ISAACLAB_ROOT.parent)}:{function.name}")

    assert not violations, f"function-local backend imports found in: {', '.join(violations)}"
