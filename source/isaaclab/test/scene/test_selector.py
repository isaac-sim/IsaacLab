# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :class:`Selector` env/view indexing."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import Selector, SelectorCfg, SelectorTermCfg
from isaaclab.utils.configclass import configclass


def _asset_names(asset_cfgs: dict[str, object], names: list[str]) -> tuple[str, ...]:
    return tuple(name for name in names if name in asset_cfgs)


def _index_to_list(index: slice | torch.Tensor) -> list[int] | slice:
    return index.tolist() if isinstance(index, torch.Tensor) else index


@configclass
class _SelectorCfg(SelectorCfg):
    big_robot = SelectorTermCfg(func=_asset_names, params={"names": ["franka", "ur10"]})
    franka_only = SelectorTermCfg(func=_asset_names, params={"names": ["franka"]})
    disabled = SelectorTermCfg(func=_asset_names, params={"names": ["disabled"]})


def test_selector_resolves_many_assets_to_one_env_selection():
    """A selector can union env rows from multiple assets."""
    selector = Selector(_SelectorCfg(), num_envs=6, device="cpu")
    selector.resolve_terms({"franka": object(), "ur10": object(), "disabled": object()})
    selector.apply_asset_env_ids(
        {
            "franka": torch.tensor([0, 2], dtype=torch.long),
            "ur10": torch.tensor([1, 4], dtype=torch.long),
            "disabled": torch.tensor([], dtype=torch.long),
        }
    )

    franka = selector.get("big_robot", asset="franka")
    ur10 = selector.get("big_robot", asset="ur10")

    assert franka.env_ids.tolist() == [0, 2]
    assert _index_to_list(franka.view_ids) == slice(0, 2)
    assert ur10.env_ids.tolist() == [1, 4]
    assert _index_to_list(ur10.view_ids) == slice(0, 2)
    assert selector.selector_assets["big_robot"] == ("franka", "ur10")


def test_selector_direct_select_uses_asset_view_ids():
    """Direct selector views gather rows from asset-local buffers."""
    selector = Selector(_SelectorCfg(), num_envs=6, device="cpu")
    selector.resolve_terms({"franka": object(), "ur10": object(), "disabled": object()})
    selector.apply_asset_env_ids(
        {
            "franka": torch.tensor([0, 2, 4], dtype=torch.long),
            "ur10": torch.tensor([1, 3], dtype=torch.long),
            "disabled": torch.tensor([], dtype=torch.long),
        }
    )
    data = torch.tensor([[10.0], [20.0], [30.0]])

    view = selector.get("franka_only")

    assert view.env_ids.tolist() == [0, 2, 4]
    assert view.select(data).tolist() == [[10.0], [20.0], [30.0]]


def test_selector_empty_asset_selection_returns_empty_indices():
    """Zero-weight combinations can leave a selector with no rows."""
    selector = Selector(_SelectorCfg(), num_envs=4, device="cpu")
    selector.resolve_terms({"franka": object(), "ur10": object(), "disabled": object()})
    selector.apply_asset_env_ids(
        {
            "franka": torch.tensor([0], dtype=torch.long),
            "ur10": torch.tensor([1], dtype=torch.long),
            "disabled": torch.tensor([], dtype=torch.long),
        }
    )

    view = selector.get("disabled", asset="disabled")

    assert view.env_ids.numel() == 0
    assert view.view_ids.numel() == 0


def test_scene_entity_cfg_resolves_selector_view_ids():
    """SceneEntityCfg resolves env/view ids through the scene selector."""
    selector = Selector(_SelectorCfg(), num_envs=6, device="cpu")
    selector.resolve_terms({"franka": object(), "ur10": object(), "disabled": object()})
    selector.apply_asset_env_ids(
        {
            "franka": torch.tensor([0, 2, 4], dtype=torch.long),
            "ur10": torch.tensor([1, 3], dtype=torch.long),
            "disabled": torch.tensor([], dtype=torch.long),
        }
    )
    scene = SimpleNamespace(keys=lambda: ["franka"], device="cpu", selector=selector)
    cfg = SceneEntityCfg("franka", selector="franka_only")

    cfg.resolve(scene)

    assert cfg.env_ids.tolist() == [0, 2, 4]
    assert cfg.view_ids == slice(0, 3)
