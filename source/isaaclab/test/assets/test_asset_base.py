# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.assets import AssetBase


class _TestAsset(AssetBase):
    @property
    def num_instances(self):
        return 0

    @property
    def data(self):
        return None

    def reset(self, env_ids=None):
        pass

    def write_data_to_sim(self):
        pass

    def update(self, dt):
        pass

    def _initialize_impl(self):
        pass


def test_asset_base_destructor_clears_before_shutdown(monkeypatch):
    """Asset destructors should still clean up during normal runtime."""

    cleared = False

    def clear_callbacks(_self):
        nonlocal cleared
        cleared = True

    asset = object.__new__(_TestAsset)
    monkeypatch.setattr(_TestAsset, "_clear_callbacks", clear_callbacks)

    asset.__del__()

    assert cleared


def test_asset_base_destructor_skips_clear_after_import_shutdown(monkeypatch):
    """Asset destructors should not import modules after import machinery is torn down."""

    def clear_callbacks(_self):
        raise ImportError("sys.meta_path is None, Python is likely shutting down")

    asset = object.__new__(_TestAsset)
    monkeypatch.setattr(_TestAsset, "_clear_callbacks", clear_callbacks)
    monkeypatch.setattr("sys.meta_path", None)

    asset.__del__()
