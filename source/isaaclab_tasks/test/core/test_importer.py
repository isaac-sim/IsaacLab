# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the task package importer."""

import sys

from isaaclab_tasks.utils import import_packages


def test_import_packages_blacklist_matches_whole_package_names(tmp_path, monkeypatch):
    """A blacklisted ``.mdp`` skips the ``mdp`` package but not ``mdp2``."""
    for name in ("blacklist_root", "blacklist_root/mdp", "blacklist_root/mdp2"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "__init__.py").touch()
    monkeypatch.syspath_prepend(tmp_path)

    import_packages("blacklist_root", [".mdp"])

    assert "blacklist_root.mdp2" in sys.modules
    assert "blacklist_root.mdp" not in sys.modules
