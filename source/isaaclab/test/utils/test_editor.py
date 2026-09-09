# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pathlib

import pytest

from isaaclab.utils.editor import _find_isaac_sim_icon, _read_kit_window_identity, setup_desktop_entry

pytestmark = pytest.mark.unit

# A minimal excerpt of apps/isaaclab.python.kit's shape: [settings] is re-declared across
# multiple sections, which is invalid under strict TOML but is what Kit's own files look like.
_KIT_FILE_CONTENT = """
[package]
version = "3.0.0"

[settings]
app.name = "IsaacLab"

[settings.app]
name = "IsaacLab"
version = "3.0.0"

[settings.app.window]
iconPath = "${isaacsim.simulation_app}/data/omni.isaac.sim.png"
title = "Isaac Lab"

[settings]
physics.updateToUsd = false
"""


def test_read_kit_window_identity_parses_duplicate_settings_sections(tmp_path: pathlib.Path):
    """Test title/version are read correctly despite the repeated [settings] header."""
    kit_file = tmp_path / "isaaclab.python.kit"
    kit_file.write_text(_KIT_FILE_CONTENT)

    identity = _read_kit_window_identity(kit_file)

    assert identity == ("Isaac Lab", "3.0.0")


def test_read_kit_window_identity_missing_file_returns_none(tmp_path: pathlib.Path):
    """Test a missing kit file is reported as no identity rather than raising."""
    assert _read_kit_window_identity(tmp_path / "does_not_exist.kit") is None


@pytest.mark.parametrize(
    "content",
    [
        # No [settings.app.window] title.
        '[settings.app]\nversion = "3.0.0"\n',
        # No [settings.app] version.
        '[settings.app.window]\ntitle = "Isaac Lab"\n',
    ],
)
def test_read_kit_window_identity_missing_value_returns_none(tmp_path: pathlib.Path, content: str):
    """Test a kit file missing either title or version is reported as no identity."""
    kit_file = tmp_path / "isaaclab.python.kit"
    kit_file.write_text(content)

    assert _read_kit_window_identity(kit_file) is None


def test_find_isaac_sim_icon_locates_asset_under_package(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """Test the icon is found when the isaacsim package ships the expected asset path."""
    package_root = tmp_path / "isaacsim"
    icon_dir = package_root / "exts" / "isaacsim.simulation_app" / "data"
    icon_dir.mkdir(parents=True)
    icon_file = icon_dir / "omni.isaac.sim.png"
    icon_file.write_bytes(b"")

    class _FakeSpec:
        submodule_search_locations = [str(package_root)]

    monkeypatch.setattr("isaaclab.utils.editor.importlib.util.find_spec", lambda name: _FakeSpec())

    assert _find_isaac_sim_icon() == icon_file


def test_find_isaac_sim_icon_missing_package_returns_none(monkeypatch: pytest.MonkeyPatch):
    """Test an uninstalled isaacsim package is reported as no icon rather than raising."""
    monkeypatch.setattr("isaaclab.utils.editor.importlib.util.find_spec", lambda name: None)

    assert _find_isaac_sim_icon() is None


def test_setup_desktop_entry_noop_on_non_linux(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """Test the desktop entry is not generated on non-Linux platforms."""
    monkeypatch.setattr("isaaclab.utils.editor.platform.system", lambda: "Windows")
    home = tmp_path / "home"
    monkeypatch.setattr(pathlib.Path, "home", lambda: home)

    setup_desktop_entry(tmp_path)

    assert not (home / ".local" / "share" / "applications" / "isaaclab.desktop").exists()


def test_setup_desktop_entry_writes_startup_wm_class_matching_kit_identity(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """Test the generated .desktop file's StartupWMClass matches Kit's composed WM_CLASS.

    Kit composes the running window's WM_CLASS from the kit file's window title and app
    version (e.g. "Isaac Lab 3.0.0"). The whole point of generating this file is for
    StartupWMClass to match that exactly, so a desktop environment's .desktop-based taskbar
    icon lookup succeeds instead of falling back to a generic icon.
    """
    monkeypatch.setattr("isaaclab.utils.editor.platform.system", lambda: "Linux")
    home = tmp_path / "home"
    monkeypatch.setattr(pathlib.Path, "home", lambda: home)

    project_dir = tmp_path / "project"
    apps_dir = project_dir / "apps"
    apps_dir.mkdir(parents=True)
    (apps_dir / "isaaclab.python.kit").write_text(_KIT_FILE_CONTENT)

    package_root = tmp_path / "isaacsim"
    icon_dir = package_root / "exts" / "isaacsim.simulation_app" / "data"
    icon_dir.mkdir(parents=True)
    icon_file = icon_dir / "omni.isaac.sim.png"
    icon_file.write_bytes(b"")

    class _FakeSpec:
        submodule_search_locations = [str(package_root)]

    monkeypatch.setattr("isaaclab.utils.editor.importlib.util.find_spec", lambda name: _FakeSpec())

    setup_desktop_entry(project_dir)

    desktop_file = home / ".local" / "share" / "applications" / "isaaclab.desktop"
    content = desktop_file.read_text()
    assert "StartupWMClass=Isaac Lab 3.0.0" in content
    assert f"Icon={icon_file}" in content
    assert "Name=Isaac Lab" in content


def test_setup_desktop_entry_noop_when_kit_file_missing(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """Test a missing kit file skips desktop entry generation instead of raising."""
    monkeypatch.setattr("isaaclab.utils.editor.platform.system", lambda: "Linux")
    home = tmp_path / "home"
    monkeypatch.setattr(pathlib.Path, "home", lambda: home)

    setup_desktop_entry(tmp_path / "project_without_apps_dir")

    assert not (home / ".local" / "share" / "applications" / "isaaclab.desktop").exists()
