# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for renderer cache inventory and change detection."""

import importlib.util
from pathlib import Path


def _load_inventory_module():
    module_path = Path(__file__).with_name("renderer_cache_inventory.py")
    spec = importlib.util.spec_from_file_location("renderer_cache_inventory", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inventory_groups_cache_files_by_top_level_directory(tmp_path: Path) -> None:
    """Inventory should report every cache file without reading its contents."""
    inventory_module = _load_inventory_module()
    (tmp_path / "home").mkdir()
    (tmp_path / "isaac-sim").mkdir()
    (tmp_path / "home" / "shader.bin").write_bytes(b"123")
    (tmp_path / "isaac-sim" / "kit.bin").write_bytes(b"12345")

    total_bytes, file_count, groups = inventory_module.inventory(tmp_path)

    assert total_bytes == 8
    assert file_count == 2
    assert groups == {"home": 3, "isaac-sim": 5}


def test_fingerprint_changes_when_cache_file_changes(tmp_path: Path) -> None:
    """A writer must publish a new snapshot when a cache artifact changes."""
    inventory_module = _load_inventory_module()
    artifact = tmp_path / "shader.bin"
    artifact.write_bytes(b"before")
    before = inventory_module.fingerprint(tmp_path)

    artifact.write_bytes(b"after-content")

    assert inventory_module.fingerprint(tmp_path) != before
