# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parents[4]
_FINDER_FILES = (
    "source/isaaclab/isaaclab/assets/articulation/base_articulation.py",
    "source/isaaclab/isaaclab/assets/rigid_object/base_rigid_object.py",
    "source/isaaclab/isaaclab/assets/rigid_object_collection/base_rigid_object_collection.py",
    "source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py",
    "source/isaaclab_physx/isaaclab_physx/assets/rigid_object/rigid_object.py",
    "source/isaaclab_physx/isaaclab_physx/assets/rigid_object_collection/rigid_object_collection.py",
    "source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation.py",
    "source/isaaclab_newton/isaaclab_newton/assets/rigid_object/rigid_object.py",
    "source/isaaclab_newton/isaaclab_newton/assets/rigid_object_collection/rigid_object_collection.py",
    "source/isaaclab_ovphysx/isaaclab_ovphysx/assets/articulation/articulation.py",
    "source/isaaclab_ovphysx/isaaclab_ovphysx/assets/rigid_object/rigid_object.py",
    "source/isaaclab_ovphysx/isaaclab_ovphysx/assets/rigid_object_collection/rigid_object_collection.py",
)
_BACKEND_FRAGMENTS = (
    "source/isaaclab_physx/changelog.d/articulation-reordering-p11.rst",
    "source/isaaclab_newton/changelog.d/articulation-reordering-p11.rst",
    "source/isaaclab_ovphysx/changelog.d/articulation-reordering-p11.rst",
)


@pytest.mark.parametrize("path", _FINDER_FILES)
def test_public_finder_docs_cover_lifetime_and_concrete_migration(path: str) -> None:
    tree = ast.parse((_REPO_ROOT / path).read_text())
    finder_docs = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("find_"):
            continue
        if not any(argument.arg == "as_proxy" for argument in node.args.kwonlyargs):
            continue
        finder_docs.append((node.name, ast.get_docstring(node) or ""))

    assert finder_docs
    for finder_name, docstring in finder_docs:
        assert "invalidation" in docstring, finder_name
        assert "as_proxy=True" in docstring, finder_name
        assert ".warp" in docstring, finder_name
        assert ".torch" in docstring, finder_name


@pytest.mark.parametrize(
    "path",
    (
        "source/isaaclab/isaaclab/assets/articulation/base_articulation.py",
        "source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py",
        "source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation.py",
        "source/isaaclab_ovphysx/isaaclab_ovphysx/assets/articulation/articulation.py",
    ),
)
def test_articulation_subset_finder_docs_define_proxy_and_legacy_indices(path: str) -> None:
    tree = ast.parse((_REPO_ROOT / path).read_text())
    for finder_name in ("find_joints", "find_fixed_tendons", "find_spatial_tendons"):
        node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == finder_name)
        docstring = ast.get_docstring(node) or ""
        assert "asset-global" in docstring, finder_name
        assert "legacy" in docstring, finder_name


@pytest.mark.parametrize("path", _BACKEND_FRAGMENTS)
def test_backend_fragments_include_deprecation_migration_guidance(path: str) -> None:
    fragment = (_REPO_ROOT / path).read_text()

    assert "Deprecated" in fragment
    assert "as_proxy=True" in fragment
    assert "as_proxy=False" in fragment
