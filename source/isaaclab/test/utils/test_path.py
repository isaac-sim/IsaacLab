# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Isaac Lab path resolution."""

from pathlib import Path

import pytest

from isaaclab.utils.path import ISAACLAB_PATH_ENV_VAR, resolve_isaaclab_path

pytestmark = pytest.mark.unit


def test_resolve_isaaclab_path_uses_environment_variable(monkeypatch, tmp_path):
    """An explicit install path is authoritative outside source-checkout layouts."""
    install_root = tmp_path / "share" / "isaaclab"
    monkeypatch.setenv(ISAACLAB_PATH_ENV_VAR, str(install_root))

    assert resolve_isaaclab_path() == install_root


def test_resolve_isaaclab_path_keeps_source_checkout_layout(monkeypatch):
    """The editable checkout layout remains the default when ``ISAACLAB_PATH`` is unset."""
    monkeypatch.delenv(ISAACLAB_PATH_ENV_VAR, raising=False)
    test_file = Path(__file__).resolve()
    package_file = test_file.parents[2] / "isaaclab" / "utils" / "nested" / "path.py"

    assert resolve_isaaclab_path(package_file) == test_file.parents[4]


def test_resolve_isaaclab_path_detects_installed_package_layout(monkeypatch, tmp_path):
    """Installed wheels can carry ``apps`` next to the Python package."""
    monkeypatch.delenv(ISAACLAB_PATH_ENV_VAR, raising=False)
    site_package = tmp_path / "site-packages" / "isaaclab"
    (site_package / "apps").mkdir(parents=True)
    package_file = site_package / "utils" / "path.py"
    package_file.parent.mkdir()
    package_file.touch()

    monkeypatch.setattr("isaaclab.__file__", site_package / "__init__.py")

    assert resolve_isaaclab_path(package_file) == site_package
