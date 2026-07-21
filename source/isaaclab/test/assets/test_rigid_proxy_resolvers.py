# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import warp as wp

from isaaclab.utils.warp import ProxyArray

_REPO_ROOT = Path(__file__).parents[4]
_RIGID_IMPLEMENTATIONS = (
    "source/isaaclab_physx/isaaclab_physx/assets/rigid_object/rigid_object.py",
    "source/isaaclab_newton/isaaclab_newton/assets/rigid_object/rigid_object.py",
    "source/isaaclab_ovphysx/isaaclab_ovphysx/assets/rigid_object/rigid_object.py",
    "source/isaaclab_physx/isaaclab_physx/assets/rigid_object_collection/rigid_object_collection.py",
    "source/isaaclab_newton/isaaclab_newton/assets/rigid_object_collection/rigid_object_collection.py",
    "source/isaaclab_ovphysx/isaaclab_ovphysx/assets/rigid_object_collection/rigid_object_collection.py",
)
_PUBLIC_RIGID_INTERFACES = (
    "source/isaaclab/isaaclab/assets/rigid_object/base_rigid_object.py",
    "source/isaaclab/isaaclab/assets/rigid_object_collection/base_rigid_object_collection.py",
    *_RIGID_IMPLEMENTATIONS,
)


def _parse(path: str) -> ast.Module:
    return ast.parse((_REPO_ROOT / path).read_text(), filename=path)


def _extract_resolver(path: str):
    tree = _parse(path)
    resolver = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_resolve_body_ids"
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), resolver],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"ProxyArray": ProxyArray, "Sequence": Sequence, "torch": torch, "wp": wp}
    exec(compile(module, path, "exec"), namespace)
    return namespace["_resolve_body_ids"]


@pytest.mark.parametrize("path", _RIGID_IMPLEMENTATIONS)
def test_rigid_body_resolver_unwraps_proxy_before_comparison(path: str) -> None:
    resolver = _extract_resolver(path)
    body_ids_array = wp.array([2, 0], dtype=wp.int32, device="cpu")
    body_ids = ProxyArray(body_ids_array)
    asset = SimpleNamespace(
        device="cpu",
        _device="cpu",
        num_bodies=3,
        _ALL_BODY_INDICES=wp.array([0, 1, 2], dtype=wp.int32, device="cpu"),
    )

    resolved = resolver(asset, body_ids)

    assert resolved is body_ids_array
    assert body_ids._torch_cache is None


@pytest.mark.parametrize("path", _PUBLIC_RIGID_INTERFACES)
def test_public_rigid_writers_annotate_proxy_body_selectors(path: str) -> None:
    missing = []
    for node in ast.walk(_parse(path)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith(("set_", "write_")):
            continue
        body_ids = next(
            (argument for argument in (*node.args.args, *node.args.kwonlyargs) if argument.arg == "body_ids"), None
        )
        if body_ids is None or body_ids.annotation is None:
            continue
        annotation = ast.unparse(body_ids.annotation)
        if (
            any(selector in annotation for selector in ("Sequence[", "torch.Tensor", "wp.array"))
            and "ProxyArray" not in annotation
        ):
            missing.append(f"{node.name}: {annotation}")

    assert not missing, f"{path} has public body selectors without ProxyArray: {missing}"
