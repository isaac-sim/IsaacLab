# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pathlib

import pytest

from isaaclab.utils import desktop_icons
from isaaclab.utils.desktop_icons import (
    _NEWTON_GL_WM_CLASS,
    _NEWTON_RTX_WM_CLASS,
    _desktop_entry,
    _has_graphical_session,
    _install_desktop_entries,
    _install_icon_files,
    install_desktop_icons,
    newton_icon_source_dir,
    xdg_data_home,
)

pytestmark = pytest.mark.unit


def test_has_graphical_session_true_on_linux_with_display(monkeypatch: pytest.MonkeyPatch):
    """Test a Linux session with DISPLAY set is treated as graphical."""
    monkeypatch.setattr("isaaclab.utils.desktop_icons.platform.system", lambda: "Linux")
    monkeypatch.setattr("isaaclab.utils.desktop_icons.os.path.exists", lambda path: False)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    assert _has_graphical_session() is True


def test_has_graphical_session_false_on_non_linux(monkeypatch: pytest.MonkeyPatch):
    """Test non-Linux platforms are never treated as a Linux graphical session."""
    monkeypatch.setattr("isaaclab.utils.desktop_icons.platform.system", lambda: "Windows")
    monkeypatch.setenv("DISPLAY", ":0")

    assert _has_graphical_session() is False


def test_has_graphical_session_false_without_display_or_wayland(monkeypatch: pytest.MonkeyPatch):
    """Test a Linux session with neither DISPLAY nor WAYLAND_DISPLAY is not graphical."""
    monkeypatch.setattr("isaaclab.utils.desktop_icons.platform.system", lambda: "Linux")
    monkeypatch.setattr("isaaclab.utils.desktop_icons.os.path.exists", lambda path: False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    assert _has_graphical_session() is False


def test_has_graphical_session_false_in_docker(monkeypatch: pytest.MonkeyPatch):
    """Test a containerized session is skipped even with DISPLAY set, matching isaaclab.sh -i's Docker guard."""
    monkeypatch.setattr("isaaclab.utils.desktop_icons.platform.system", lambda: "Linux")
    monkeypatch.setattr("isaaclab.utils.desktop_icons.os.path.exists", lambda path: path == "/.dockerenv")
    monkeypatch.setenv("DISPLAY", ":0")

    assert _has_graphical_session() is False


def test_newton_icon_source_dir_returns_none_when_newton_missing(monkeypatch: pytest.MonkeyPatch):
    """Test a missing Newton install is reported as no icon source rather than raising.

    Resolved via importlib.util.find_spec rather than importing newton's GL renderer module:
    that module pulls in warp/pyglet/GL at import time, which could raise something other than
    ImportError on a broken install and violate install_desktop_icons's never-raises contract.
    """
    monkeypatch.setattr("isaaclab.utils.desktop_icons.importlib.util.find_spec", lambda name: None)

    assert newton_icon_source_dir() is None


def test_newton_icon_source_dir_returns_none_when_icon_dir_missing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """Test a resolved package spec whose expected icon subdirectory doesn't exist returns None."""
    package_root = tmp_path / "newton"
    package_root.mkdir()

    class _FakeSpec:
        submodule_search_locations = [str(package_root)]

    monkeypatch.setattr("isaaclab.utils.desktop_icons.importlib.util.find_spec", lambda name: _FakeSpec())

    assert newton_icon_source_dir() is None


def test_newton_icon_source_dir_finds_real_package():
    """Test the real installed Newton package resolves to a directory containing its icon files."""
    icon_dir = newton_icon_source_dir()

    assert icon_dir is not None
    assert (icon_dir / "icon_64.png").is_file()


def test_xdg_data_home_defaults_to_local_share_when_unset(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """Test a missing XDG_DATA_HOME falls back to ~/.local/share."""
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)

    assert xdg_data_home() == tmp_path / ".local" / "share"


def test_xdg_data_home_defaults_to_local_share_when_empty(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """Test an empty-string XDG_DATA_HOME is treated as unset, per the XDG Base Directory spec."""
    monkeypatch.setenv("XDG_DATA_HOME", "")
    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)

    assert xdg_data_home() == tmp_path / ".local" / "share"


def test_xdg_data_home_honors_explicit_value(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """Test a non-empty XDG_DATA_HOME is used as-is."""
    custom = tmp_path / "custom"
    monkeypatch.setenv("XDG_DATA_HOME", str(custom))

    assert xdg_data_home() == custom


def test_install_icon_files_copies_available_sizes(tmp_path: pathlib.Path):
    """Test each bundled Newton icon size is copied into the hicolor theme layout."""
    icon_source_dir = tmp_path / "newton_gl"
    icon_source_dir.mkdir()
    for size in desktop_icons.NEWTON_ICON_SIZES:
        (icon_source_dir / f"icon_{size}.png").write_bytes(b"fake-png")
    data_home = tmp_path / "data_home"

    copied = _install_icon_files(icon_source_dir, data_home)

    assert copied == len(desktop_icons.NEWTON_ICON_SIZES)
    for size in desktop_icons.NEWTON_ICON_SIZES:
        dst = data_home / "icons" / "hicolor" / f"{size}x{size}" / "apps" / f"{desktop_icons._ICON_NAME}.png"
        assert dst.read_bytes() == b"fake-png"


def test_install_icon_files_reports_number_of_missing_sizes(tmp_path: pathlib.Path):
    """Test a missing source size is skipped (not raised) but still reflected in the return count."""
    icon_source_dir = tmp_path / "newton_gl"
    icon_source_dir.mkdir()
    sizes = list(desktop_icons.NEWTON_ICON_SIZES)
    (icon_source_dir / f"icon_{sizes[0]}.png").write_bytes(b"fake-png")
    data_home = tmp_path / "data_home"

    copied = _install_icon_files(icon_source_dir, data_home)

    assert copied == 1
    present = data_home / "icons" / "hicolor" / f"{sizes[0]}x{sizes[0]}" / "apps" / f"{desktop_icons._ICON_NAME}.png"
    assert present.is_file()
    missing_dir = data_home / "icons" / "hicolor" / f"{sizes[1]}x{sizes[1]}" / "apps"
    assert not missing_dir.exists()


def test_install_icon_files_returns_zero_when_none_present(tmp_path: pathlib.Path):
    """Test an icon source directory with none of the expected files copies nothing."""
    icon_source_dir = tmp_path / "newton_gl"
    icon_source_dir.mkdir()
    data_home = tmp_path / "data_home"

    copied = _install_icon_files(icon_source_dir, data_home)

    assert copied == 0
    assert not (data_home / "icons").exists()


def test_desktop_entry_contains_name_icon_and_wm_class():
    """Test the generated .desktop text has the fields GNOME needs to resolve the icon."""
    entry = _desktop_entry("Newton Viewer", _NEWTON_GL_WM_CLASS)

    assert "Type=Application" in entry
    assert "Name=Newton Viewer" in entry
    assert f"Icon={desktop_icons._ICON_NAME}" in entry
    assert f"StartupWMClass={_NEWTON_GL_WM_CLASS}" in entry


def test_install_desktop_entries_writes_distinct_wm_classes_for_gl_and_rtx(tmp_path: pathlib.Path):
    """Test GL and RTX each get their own entry, since pyglet gives them different WM_CLASS values."""
    apps_dir = _install_desktop_entries(tmp_path)

    gl_content = (apps_dir / "isaaclab-newton-gl-viewer.desktop").read_text()
    rtx_content = (apps_dir / "isaaclab-newton-rtx-viewer.desktop").read_text()
    assert f"StartupWMClass={_NEWTON_GL_WM_CLASS}" in gl_content
    assert f"StartupWMClass={_NEWTON_RTX_WM_CLASS}" in rtx_content


def test_install_desktop_icons_noop_without_graphical_session(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """Test headless/CI/Windows sessions are skipped entirely, writing nothing."""
    monkeypatch.setattr("isaaclab.utils.desktop_icons._has_graphical_session", lambda: False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

    install_desktop_icons()

    assert not (tmp_path / "applications").exists()
    assert not (tmp_path / "icons").exists()


def test_install_desktop_icons_full_flow(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """Test a graphical session with Newton installed writes both icons and both .desktop entries."""
    icon_source_dir = tmp_path / "newton_gl"
    icon_source_dir.mkdir()
    for size in desktop_icons.NEWTON_ICON_SIZES:
        (icon_source_dir / f"icon_{size}.png").write_bytes(b"fake-png")

    monkeypatch.setattr("isaaclab.utils.desktop_icons._has_graphical_session", lambda: True)
    monkeypatch.setattr("isaaclab.utils.desktop_icons.newton_icon_source_dir", lambda: icon_source_dir)
    monkeypatch.setattr("isaaclab.utils.desktop_icons.shutil.which", lambda cmd: None)
    data_home = tmp_path / "data_home"
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))

    install_desktop_icons()

    assert (data_home / "applications" / "isaaclab-newton-gl-viewer.desktop").is_file()
    assert (data_home / "applications" / "isaaclab-newton-rtx-viewer.desktop").is_file()
    assert (data_home / "icons" / "hicolor" / "64x64" / "apps" / f"{desktop_icons._ICON_NAME}.png").is_file()


def test_install_desktop_icons_skips_entries_when_no_icons_copied(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """Test no .desktop entries are written when zero icon files actually copied.

    A partial/corrupted Newton install can resolve a real package directory that doesn't
    contain any of the expected icon_{16,32,64}.png files. Writing .desktop entries in that
    case would reference an Icon= that never resolves to any file, silently degrading to no
    icon while looking successful.
    """
    icon_source_dir = tmp_path / "newton_gl_missing_icons"
    icon_source_dir.mkdir()

    monkeypatch.setattr("isaaclab.utils.desktop_icons._has_graphical_session", lambda: True)
    monkeypatch.setattr("isaaclab.utils.desktop_icons.newton_icon_source_dir", lambda: icon_source_dir)
    data_home = tmp_path / "data_home"
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))

    install_desktop_icons()

    assert not (data_home / "applications").exists()


def test_install_desktop_icons_swallows_filesystem_errors(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """Test an unwritable data home doesn't propagate — isaaclab.sh -i must not fail because of this step."""
    monkeypatch.setattr("isaaclab.utils.desktop_icons._has_graphical_session", lambda: True)
    monkeypatch.setattr("isaaclab.utils.desktop_icons.newton_icon_source_dir", lambda: tmp_path)

    def _raise(*args, **kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr("isaaclab.utils.desktop_icons._install_icon_files", _raise)

    install_desktop_icons()
