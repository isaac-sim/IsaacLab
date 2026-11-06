# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Fabric visual-material writer hot path."""

from __future__ import annotations

import ast
import inspect
import textwrap
from unittest.mock import MagicMock, patch

import pytest
import torch
from isaaclab_physx.renderers import visual_material
from isaaclab_physx.renderers.visual_material import FabricVisualMaterialWriter


@pytest.mark.parametrize(
    ("shape", "warp_type"),
    [
        pytest.param((2,), visual_material.wp.float32, id="scalar"),
        pytest.param((2, 2), visual_material.wp.vec2f, id="float2"),
        pytest.param((2, 3), visual_material.wp.vec3f, id="float3"),
    ],
)
def test_supported_tensor_layouts(shape, warp_type) -> None:
    assert visual_material._WRITES[(torch.float32, shape[1:])][1] is warp_type


def test_writer_dispatches_only_requested_channels() -> None:
    roughness_selection = MagicMock()
    color_selection = MagicMock()
    writer = FabricVisualMaterialWriter.__new__(FabricVisualMaterialWriter)
    roughness_values = MagicMock(device="cuda:0")
    writer._writes = {
        "roughness": ((object(), roughness_values, roughness_selection, "inverse", "inputs:roughness"),),
        "color": ((object(), object(), color_selection, "inverse", "inputs:diffuseColor"),),
    }
    offsets = MagicMock(__len__=lambda _self: 1)
    env_ids = MagicMock(__len__=lambda _self: 1)

    with (
        patch.object(visual_material.wp, "fabricarray", return_value="output"),
        patch.object(visual_material.wp, "launch") as launch,
    ):
        writer({"roughness": offsets}, env_ids)

    roughness_selection.PrepareForReuse.assert_called_once_with()
    color_selection.PrepareForReuse.assert_not_called()
    launch.assert_called_once_with(
        writer._writes["roughness"][0][0],
        dim=(1, 1),
        inputs=[roughness_values, offsets, env_ids, "inverse", "output"],
        device="cuda:0",
    )


def test_runtime_only_reuses_fabric_addresses_and_launches_kernels() -> None:
    source = textwrap.dedent(inspect.getsource(FabricVisualMaterialWriter.__call__))
    tree = ast.parse(source)
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    symbols = {
        symbol.lower()
        for node in ast.walk(tree)
        for symbol in (
            (node.id,) if isinstance(node, ast.Name) else (node.attr,) if isinstance(node, ast.Attribute) else ()
        )
    }

    assert {"PrepareForReuse", "fabricarray", "launch"}.issubset(calls)
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree))
    forbidden = ("cpu", "gf", "list", "numpy", "path", "pxr", "replicator", "sdf", "tolist", "usd")
    assert not {symbol for symbol in symbols if any(name in symbol for name in forbidden)}
