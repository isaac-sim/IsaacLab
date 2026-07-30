# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import torch

from isaaclab.assets.asset_base import AssetBase
from isaaclab.utils.warp import ProxyArray


class _TestAsset(AssetBase):
    @property
    def num_instances(self) -> int:
        return 1

    @property
    def data(self) -> Any:
        return None

    def reset(self, env_ids=None) -> None:
        pass

    def write_data_to_sim(self) -> None:
        pass

    def update(self, dt: float) -> None:
        pass

    def _initialize_impl(self) -> None:
        pass


def _make_asset() -> _TestAsset:
    asset = object.__new__(_TestAsset)
    asset._device = "cpu"
    asset._initialize_handle = None
    asset._invalidate_initialize_handle = None
    asset._prim_deletion_handle = None
    asset._debug_vis_handle = None
    return asset


def _find(asset: _TestAsset, indices, *, domain: str = "joint", as_proxy: bool = True):
    return asset._resolve_finder_indices(
        indices,
        domain=domain,
        as_proxy=as_proxy,
        legacy_type="list",
    )


def test_equivalent_indices_reuse_proxy_and_storage():
    asset = _make_asset()

    first = _find(asset, [3, 1, 3])
    second = _find(asset, (3, 1, 3))

    assert isinstance(first, ProxyArray)
    assert first is second
    assert first.warp is second.warp
    assert first.warp.ptr == second.warp.ptr


def test_domains_do_not_alias():
    asset = _make_asset()

    joints = _find(asset, [0, 2], domain="joint")
    bodies = _find(asset, [0, 2], domain="body")

    assert joints is not bodies
    assert joints.warp is not bodies.warp
    assert joints.warp.ptr != bodies.warp.ptr


def test_assets_do_not_share_entries():
    first_asset = _make_asset()
    second_asset = _make_asset()

    first = _find(first_asset, [0, 2])
    second = _find(second_asset, [0, 2])

    assert first is not second
    assert first.warp is not second.warp
    assert first.warp.ptr != second.warp.ptr


def test_lru_evicts_oldest_entry_at_capacity():
    asset = _make_asset()
    oldest = _find(asset, [0])
    second_oldest = _find(asset, [1])
    for index in range(2, 128):
        _find(asset, [index])

    assert _find(asset, [0]) is oldest
    _find(asset, [128])

    assert _find(asset, [0]) is oldest
    assert _find(asset, [1]) is not second_oldest


def test_clear_releases_cached_entries():
    asset = _make_asset()
    first = _find(asset, [1, 4])

    asset._clear_selector_cache()
    second = _find(asset, [1, 4])

    assert first is not second
    assert first.warp is not second.warp
    assert first.warp.ptr != second.warp.ptr


def test_legacy_list_mode_is_default():
    asset = _make_asset()

    implicit = asset._resolve_finder_indices(
        [2, 5],
        domain="joint",
        legacy_type="list",
    )
    explicit = _find(asset, [2, 5], as_proxy=False)

    assert implicit == explicit == [2, 5]
    assert implicit is not explicit


def test_legacy_tensor_mode_is_int32_on_asset_device():
    asset = _make_asset()

    first = asset._resolve_finder_indices(
        [4, 1],
        domain="body",
        as_proxy=False,
        legacy_type="tensor",
    )
    second = asset._resolve_finder_indices(
        [4, 1],
        domain="body",
        as_proxy=False,
        legacy_type="tensor",
    )

    assert isinstance(first, torch.Tensor)
    assert first.dtype == torch.int32
    assert first.device.type == asset.device
    assert first.tolist() == [4, 1]
    assert first is not second
    assert first.data_ptr() != second.data_ptr()


def test_proxy_indices_can_differ_without_changing_legacy_semantics():
    asset = _make_asset()

    proxy = asset._resolve_finder_indices(
        [0, 1],
        proxy_indices=[2, 0],
        domain="joint",
        as_proxy=True,
        legacy_type="list",
    )
    direct = _find(asset, [2, 0])
    explicit_legacy = asset._resolve_finder_indices(
        [0, 1],
        proxy_indices=[2, 0],
        domain="joint",
        as_proxy=False,
        legacy_type="list",
    )

    assert proxy is direct
    assert proxy.torch.tolist() == [2, 0]
    assert explicit_legacy == [0, 1]
