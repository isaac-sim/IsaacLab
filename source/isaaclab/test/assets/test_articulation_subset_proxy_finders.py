# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from isaaclab.assets.asset_base import AssetBase
from isaaclab.utils.string import resolve_matching_names

_REPO_ROOT = Path(__file__).parents[4]
_IMPLEMENTATIONS = (
    "source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py",
    "source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation.py",
    "source/isaaclab_ovphysx/isaaclab_ovphysx/assets/articulation/articulation.py",
)
_FINDERS = (
    ("find_joints", "joint_subset", "joint_names"),
    ("find_fixed_tendons", "tendon_subsets", "fixed_tendon_names"),
    ("find_spatial_tendons", "tendon_subsets", "spatial_tendon_names"),
)


class _Asset:
    device = "cpu"
    joint_names = ["item_0", "item_1", "item_2"]
    fixed_tendon_names = ["item_0", "item_1", "item_2"]
    spatial_tendon_names = ["item_0", "item_1", "item_2"]
    _resolve_finder_indices = AssetBase._resolve_finder_indices


def _extract_finder(path: str, finder_name: str):
    tree = ast.parse((_REPO_ROOT / path).read_text(), filename=path)
    finder = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == finder_name)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), finder],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"resolve_matching_names": resolve_matching_names}
    exec(compile(module, path, "exec"), namespace)
    return namespace[finder_name]


@pytest.mark.parametrize("path", _IMPLEMENTATIONS)
@pytest.mark.parametrize(("finder_name", "subset_arg", "names_attr"), _FINDERS)
def test_production_subset_proxy_indices_are_asset_global_without_changing_legacy(
    path: str, finder_name: str, subset_arg: str, names_attr: str
) -> None:
    finder = _extract_finder(path, finder_name)
    asset = _Asset()
    subset_kwargs = {subset_arg: ["item_2", "item_0"]}

    with pytest.warns(DeprecationWarning):
        implicit_indices, implicit_names = finder(asset, ".*", **subset_kwargs, preserve_order=True)
    explicit_indices, explicit_names = finder(asset, ".*", **subset_kwargs, preserve_order=True, as_proxy=False)
    subset_proxy, subset_names = finder(asset, ".*", **subset_kwargs, preserve_order=True, as_proxy=True)
    direct_proxy, direct_names = finder(asset, ["item_2", "item_0"], preserve_order=True, as_proxy=True)

    assert implicit_indices == explicit_indices == [0, 1]
    assert implicit_names == explicit_names == ["item_2", "item_0"]
    assert subset_proxy is direct_proxy
    assert subset_proxy.torch.tolist() == [2, 0]
    assert subset_names == direct_names == ["item_2", "item_0"]
