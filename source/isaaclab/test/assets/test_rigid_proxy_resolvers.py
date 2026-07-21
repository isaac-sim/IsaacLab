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

from isaaclab.assets.rigid_object_collection.base_rigid_object_collection import BaseRigidObjectCollection
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
_COLLECTION_INTERFACES = (
    "source/isaaclab/isaaclab/assets/rigid_object_collection/base_rigid_object_collection.py",
    "source/isaaclab_physx/isaaclab_physx/assets/rigid_object_collection/rigid_object_collection.py",
    "source/isaaclab_newton/isaaclab_newton/assets/rigid_object_collection/rigid_object_collection.py",
    "source/isaaclab_ovphysx/isaaclab_ovphysx/assets/rigid_object_collection/rigid_object_collection.py",
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
def test_public_rigid_apis_annotate_proxy_body_and_object_selectors(path: str) -> None:
    missing = []
    for node in ast.walk(_parse(path)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_") or not (node.name.startswith(("set_", "write_")) or node.name == "reset"):
            continue
        for argument in (*node.args.args, *node.args.kwonlyargs):
            if argument.arg not in ("body_ids", "object_ids"):
                continue
            annotation = ast.unparse(argument.annotation) if argument.annotation is not None else "<missing>"
            if "ProxyArray" not in annotation:
                missing.append(f"{node.name}.{argument.arg}: {annotation}")

    assert not missing, f"{path} has public body selectors without ProxyArray: {missing}"


@pytest.mark.parametrize("path", _COLLECTION_INTERFACES)
def test_write_data_to_sim_signature_remains_parameterless(path: str) -> None:
    definitions = [
        node
        for node in ast.walk(_parse(path))
        if isinstance(node, ast.FunctionDef) and node.name == "write_data_to_sim"
    ]

    assert definitions
    for definition in definitions:
        assert [argument.arg for argument in definition.args.args] == ["self"]
        assert not definition.args.posonlyargs
        assert not definition.args.kwonlyargs
        assert definition.args.vararg is None
        assert definition.args.kwarg is None
        assert ast.unparse(definition.returns) == "None"


def test_deprecated_object_writer_forwards_proxy_without_materializing_torch() -> None:
    calls = []

    def write_body_pose_to_sim_index(**kwargs) -> None:
        calls.append(kwargs)

    object_ids_array = wp.array([2, 0], dtype=wp.int32, device="cpu")
    object_ids = ProxyArray(object_ids_array)
    collection = SimpleNamespace(write_body_pose_to_sim_index=write_body_pose_to_sim_index)
    object_pose = torch.zeros((1, 2, 7))

    with pytest.warns(DeprecationWarning):
        BaseRigidObjectCollection.write_object_pose_to_sim(collection, object_pose, object_ids=object_ids)

    assert len(calls) == 1
    assert calls[0]["body_ids"] is object_ids
    assert object_ids._torch_cache is None
