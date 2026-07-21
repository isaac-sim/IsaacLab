# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test that unit tests only import packages declared in the root pyproject.toml."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest
import tomllib

pytestmark = pytest.mark.integration


def _repo_root() -> Path:
    """Find the Isaac Lab repository root from this test file."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "source").is_dir():
            return parent
    raise RuntimeError("Could not find Isaac Lab repository root.")


def _declared_import_names(root_pyproject: Path) -> frozenset[str]:
    """Return top-level Python import names derivable from the root pyproject.toml dependencies."""
    with root_pyproject.open("rb") as f:
        data = tomllib.load(f)
    deps: list[str] = data.get("project", {}).get("dependencies", [])
    names: set[str] = set()
    for dep in deps:
        # Extract the package name (before any version specifiers or environment markers).
        # PEP 508: name is the first token, which may contain letters, digits, -, _, .
        raw = dep.strip()
        for sep in (" ", "[", ">=", "<=", "!=", "==", ">", "<", ";"):
            raw = raw.split(sep)[0]
        # Normalize: pip install names use '-' but import names use '_' (or vary).
        # Store both forms so the check is permissive.
        name = raw.strip()
        names.add(name)
        names.add(name.replace("-", "_"))
    # Add well-known import-name overrides that differ from the package name.
    _IMPORT_OVERRIDES = {
        "pillow": "PIL",
        "pyyaml": "yaml",
        "hidapi": "hid",
        "warp-lang": "warp",
        "warp_lang": "warp",
        "usd-core": "pxr",
        "usd_core": "pxr",
        "usd-exchange": "pxr",
        "usd_exchange": "pxr",
        "opencv-python": "cv2",
        "opencv_python": "cv2",
        "scikit-learn": "sklearn",
        "scikit_learn": "sklearn",
        "torch": "torch",
        "torchvision": "torchvision",
        "torchaudio": "torchaudio",
    }
    for pkg_name, import_name in _IMPORT_OVERRIDES.items():
        if pkg_name in names or pkg_name.replace("-", "_") in names:
            names.add(import_name)
    return frozenset(names)


def _unit_test_files(test_root: Path) -> list[Path]:
    """Return all test_*.py files that carry pytestmark = pytest.mark.unit."""
    return [p for p in test_root.rglob("test_*.py") if "pytest.mark.unit" in p.read_text(encoding="utf-8")]


def _top_level_imports(path: Path) -> set[str]:
    """Return the set of top-level package names imported by *path* (absolute imports only)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_unit_tests_only_import_declared_deps():
    """Every import in every unit test must be stdlib, pytest, isaaclab*, or a declared dep."""
    root = _repo_root()
    stdlib = sys.stdlib_module_names  # available since Python 3.10
    declared = _declared_import_names(root / "pyproject.toml")

    violations: dict[str, list[str]] = {}
    for pkg_dir in sorted((root / "source").iterdir()):
        test_root = pkg_dir / "test"
        if not test_root.exists():
            continue

        for test_file in sorted(_unit_test_files(test_root)):
            bad = sorted(
                name
                for name in _top_level_imports(test_file)
                if name not in stdlib and name != "pytest" and not name.startswith("isaaclab") and name not in declared
            )
            if bad:
                violations[str(test_file.relative_to(root))] = bad

    assert not violations, (
        "Unit tests import packages not declared in the root pyproject.toml"
        " [project.dependencies]:\n"
        + "\n".join(f"  {path}: {pkgs}" for path, pkgs in violations.items())
        + "\nFix: add the package to the root pyproject.toml dependencies,"
        " or reclassify the test as pytest.mark.integration."
    )
