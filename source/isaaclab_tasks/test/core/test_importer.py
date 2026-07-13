# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for task package importing utilities."""

from __future__ import annotations

import sys
import uuid

import gymnasium as gym

from isaaclab_tasks.utils.importer import import_packages


def test_import_packages_only_imports_registration_packages(tmp_path, monkeypatch):
    """Registry-free leaf packages should not execute during recursive task registration."""
    package_root = tmp_path / "sample_tasks"
    (package_root / "registry_free_leaf").mkdir(parents=True)
    (package_root / "registered").mkdir()
    (package_root / "__init__.py").write_text("")
    (package_root / "registry_free_leaf" / "__init__.py").write_text("raise RuntimeError('should not import')\n")

    task_id = f"Isaac-Importer-Test-{uuid.uuid4()}-v0"
    (package_root / "registered" / "__init__.py").write_text(
        "import gymnasium as gym\n"
        f"gym.register(id={task_id!r}, entry_point='sample_tasks.registered:DummyEnv')\n"
        "class DummyEnv:\n"
        "    pass\n"
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("sample_tasks", None)
    sys.modules.pop("sample_tasks.registry_free_leaf", None)
    sys.modules.pop("sample_tasks.registered", None)

    import_packages("sample_tasks")

    assert task_id in gym.registry
    assert "sample_tasks.registry_free_leaf" not in sys.modules
