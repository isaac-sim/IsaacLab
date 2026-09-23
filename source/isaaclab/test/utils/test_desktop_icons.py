# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pathlib

import pytest

from isaaclab.utils.desktop_icons import _NEWTON_GL_WM_CLASS, _NEWTON_RTX_WM_CLASS, install_desktop_icons

pytestmark = pytest.mark.unit


def _make_fake_newton_package(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a fake installed ``newton`` package with a bundled ``icon_64.png``."""
    icon_dir = tmp_path / "newton" / "_src" / "viewer" / "gl"
    icon_dir.mkdir(parents=True)
    (icon_dir / "icon_64.png").write_bytes(b"fake-png")
    return tmp_path / "newton"


@pytest.mark.parametrize(
    ("platform_name", "has_display", "in_container"),
    [
        ("Windows", True, False),
        ("Linux", False, False),
        ("Linux", True, True),
    ],
    ids=["non-linux", "no-graphical-session", "containerized"],
)
def test_install_desktop_icons_noop_on_unsupported_platform(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    platform_name: str,
    has_display: bool,
    in_container: bool,
):
    """Test non-Linux, headless, and containerized sessions all skip installation entirely."""
    monkeypatch.setattr("isaaclab.utils.desktop_icons.platform.system", lambda: platform_name)
    monkeypatch.setattr(
        "isaaclab.utils.desktop_icons.os.path.exists", lambda path: in_container and path == "/.dockerenv"
    )
    if has_display:
        monkeypatch.setenv("DISPLAY", ":0")
    else:
        monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

    install_desktop_icons()

    assert not (tmp_path / "applications").exists()


def test_install_desktop_icons_writes_entries_pointing_at_bundled_icon_on_linux(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """Test a Linux graphical session with Newton installed writes both viewers' ``.desktop`` entries.

    ``Icon=`` should reference the bundled icon's absolute path directly (the Desktop Entry
    specification allows this) rather than installing a separate icon theme.
    """
    newton_dir = _make_fake_newton_package(tmp_path)
    icon_path = newton_dir / "_src" / "viewer" / "gl" / "icon_64.png"

    class _FakeSpec:
        submodule_search_locations = [str(newton_dir)]

    monkeypatch.setattr("isaaclab.utils.desktop_icons._has_graphical_session", lambda: True)
    monkeypatch.setattr("isaaclab.utils.desktop_icons.importlib.util.find_spec", lambda name: _FakeSpec())
    data_home = tmp_path / "data_home"
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))

    install_desktop_icons()

    gl_content = (data_home / "applications" / "isaaclab-newton-gl-viewer.desktop").read_text()
    rtx_content = (data_home / "applications" / "isaaclab-newton-rtx-viewer.desktop").read_text()
    assert f"Icon={icon_path}" in gl_content
    assert f"Icon={icon_path}" in rtx_content
    assert f"StartupWMClass={_NEWTON_GL_WM_CLASS}" in gl_content
    assert f"StartupWMClass={_NEWTON_RTX_WM_CLASS}" in rtx_content
    assert not (data_home / "icons").exists()


def test_install_desktop_icons_swallows_filesystem_errors(monkeypatch: pytest.MonkeyPatch):
    """Test an unwritable data home doesn't propagate — isaaclab.sh -i must not fail because of this step."""
    monkeypatch.setattr("isaaclab.utils.desktop_icons._has_graphical_session", lambda: True)
    monkeypatch.setattr("isaaclab.utils.desktop_icons.newton_icon_path", lambda: pathlib.Path("/fake/icon_64.png"))

    def _raise(*args, **kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr("isaaclab.utils.desktop_icons._install_desktop_entries", _raise)

    install_desktop_icons()  # must not raise
