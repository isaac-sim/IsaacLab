# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pathlib
import sys

import pytest

from isaaclab.utils.editor import _find_isaac_sim_icon, _read_kit_window_identity, setup_desktop_entry, setup_editor

pytestmark = pytest.mark.unit

# A minimal excerpt of apps/isaaclab.python.kit's shape: [settings] is redeclared across
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


def test_setup_desktop_entry_noop_without_graphical_session(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """Test the desktop entry is not generated outside a Linux graphical session."""
    monkeypatch.setattr("isaaclab.utils.editor._has_graphical_session", lambda: False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    home = tmp_path / "home"
    monkeypatch.setattr(pathlib.Path, "home", lambda: home)

    setup_desktop_entry(tmp_path)

    assert not (home / ".local" / "share" / "applications" / "isaaclab.desktop").exists()


def test_setup_desktop_entry_writes_correct_and_non_intrusive_entry(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """Test the generated .desktop file matches Kit's WM_CLASS and isn't a real, visible launcher.

    Kit composes the running window's WM_CLASS from the kit file's window title and app version
    (e.g. "Isaac Lab 3.0.0"); StartupWMClass must match that exactly for a desktop environment's
    .desktop-based taskbar icon lookup to succeed instead of falling back to a generic icon. The
    entry itself must stay hidden from menus and use a harmless Exec, since it exists purely for
    that WM_CLASS match, not as a real launcher (and must not re-invoke ``sys.executable``
    unquoted, which would split into multiple Exec tokens for interpreter paths containing
    spaces).
    """
    monkeypatch.setattr("isaaclab.utils.editor._has_graphical_session", lambda: True)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
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

    content = (home / ".local" / "share" / "applications" / "isaaclab.desktop").read_text()
    assert "StartupWMClass=Isaac Lab 3.0.0" in content
    assert f"Icon={icon_file}" in content
    assert "Name=Isaac Lab" in content
    assert "NoDisplay=true" in content
    assert "Exec=true" in content
    assert sys.executable not in content


def test_setup_desktop_entry_honors_xdg_data_home(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """Test the applications directory is resolved from XDG_DATA_HOME, not hardcoded ~/.local/share.

    A desktop environment only scans .desktop files under XDG_DATA_HOME's applications/
    subdirectory; writing to ~/.local/share/applications when XDG_DATA_HOME points elsewhere
    means the generated entry is never found.
    """
    monkeypatch.setattr("isaaclab.utils.editor._has_graphical_session", lambda: True)
    data_home = tmp_path / "custom_data_home"
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    # Path.home() must not be consulted when XDG_DATA_HOME is set; point it somewhere that
    # would fail the test if it were.
    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path / "unused_home")

    project_dir = tmp_path / "project"
    apps_dir = project_dir / "apps"
    apps_dir.mkdir(parents=True)
    (apps_dir / "isaaclab.python.kit").write_text(_KIT_FILE_CONTENT)

    package_root = tmp_path / "isaacsim"
    icon_dir = package_root / "exts" / "isaacsim.simulation_app" / "data"
    icon_dir.mkdir(parents=True)
    (icon_dir / "omni.isaac.sim.png").write_bytes(b"")

    class _FakeSpec:
        submodule_search_locations = [str(package_root)]

    monkeypatch.setattr("isaaclab.utils.editor.importlib.util.find_spec", lambda name: _FakeSpec())

    setup_desktop_entry(project_dir)

    assert (data_home / "applications" / "isaaclab.desktop").is_file()
    assert not (tmp_path / "unused_home").exists()


def _project_dir_missing_kit_file(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path / "project_without_apps_dir"


def _project_dir_incomplete_kit_identity(tmp_path: pathlib.Path) -> pathlib.Path:
    project_dir = tmp_path / "project"
    apps_dir = project_dir / "apps"
    apps_dir.mkdir(parents=True)
    (apps_dir / "isaaclab.python.kit").write_text('[settings.app]\nversion = "3.0.0"\n')
    return project_dir


def _project_dir_valid_kit_file(tmp_path: pathlib.Path) -> pathlib.Path:
    project_dir = tmp_path / "project"
    apps_dir = project_dir / "apps"
    apps_dir.mkdir(parents=True)
    (apps_dir / "isaaclab.python.kit").write_text(_KIT_FILE_CONTENT)
    return project_dir


@pytest.mark.parametrize(
    ("make_project_dir", "expected_substring", "icon_missing"),
    [
        (_project_dir_missing_kit_file, "kit file not found", False),
        (_project_dir_incomplete_kit_identity, "could not determine Kit's WM_CLASS identity", False),
        (_project_dir_valid_kit_file, "Isaac Sim icon asset not found", True),
    ],
    ids=["kit-file-missing", "kit-identity-incomplete", "icon-missing"],
)
def test_setup_desktop_entry_warns_and_skips_on_resolution_failures(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    make_project_dir,
    expected_substring: str,
    icon_missing: bool,
):
    """Test each resolution failure (kit file, WM_CLASS identity, icon asset) warns instead of
    silently skipping desktop entry generation."""
    monkeypatch.setattr("isaaclab.utils.editor._has_graphical_session", lambda: True)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    home = tmp_path / "home"
    monkeypatch.setattr(pathlib.Path, "home", lambda: home)
    if icon_missing:
        monkeypatch.setattr("isaaclab.utils.editor.importlib.util.find_spec", lambda name: None)

    setup_desktop_entry(make_project_dir(tmp_path))

    assert not (home / ".local" / "share" / "applications" / "isaaclab.desktop").exists()
    output = capsys.readouterr().out
    assert "[WARN]" in output
    assert expected_substring in output


@pytest.mark.parametrize(
    ("has_graphical_session", "expect_refresh"),
    [(True, True), (False, False)],
    ids=["graphical-session", "no-graphical-session"],
)
def test_setup_editor_installs_newton_desktop_icons_and_refreshes_once(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, has_graphical_session: bool, expect_refresh: bool
):
    """Test setup_editor() installs the Newton viewer desktop icons (not just Kit's) and refreshes
    the desktop database exactly once, only when in a graphical session.

    install_desktop_icons() is called here rather than as a separate step elsewhere (e.g. at the
    tail of ``isaaclab.sh -i``), so ``isaaclab --editor`` has a single place that sets up all
    Linux desktop icons. The desktop database is refreshed exactly once here, covering both
    writers, rather than once per writer (setup_desktop_entry() and install_desktop_icons() each
    write into the same applications directory and no longer refresh internally).
    """
    monkeypatch.setattr("isaaclab.utils.editor.resolve_isaacsim_dir", lambda *args, **kwargs: None)
    monkeypatch.setattr("isaaclab.utils.editor.setup_desktop_entry", lambda project_dir: None)
    monkeypatch.setattr("isaaclab.utils.editor._has_graphical_session", lambda: has_graphical_session)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    home = tmp_path / "home"
    monkeypatch.setattr(pathlib.Path, "home", lambda: home)
    install_calls = []
    monkeypatch.setattr("isaaclab.utils.editor.install_desktop_icons", lambda: install_calls.append(True))
    refreshed_dirs = []
    monkeypatch.setattr("isaaclab.utils.editor.refresh_desktop_database", refreshed_dirs.append)

    setup_editor(tmp_path)

    assert install_calls == [True]
    assert refreshed_dirs == ([home / ".local" / "share" / "applications"] if expect_refresh else [])


def test_setup_editor_swallows_desktop_icon_install_failure(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    """Test a failure from install_desktop_icons() is swallowed the same way setup_desktop_entry()'s are.

    install_desktop_icons() already documents a never-raises contract on its own, but this
    confirms setup_editor()'s guard covers it too, in case that contract is ever violated.
    """
    monkeypatch.setattr("isaaclab.utils.editor.resolve_isaacsim_dir", lambda *args, **kwargs: None)
    monkeypatch.setattr("isaaclab.utils.editor.setup_desktop_entry", lambda project_dir: None)

    def _raise():
        raise OSError("disk full")

    monkeypatch.setattr("isaaclab.utils.editor.install_desktop_icons", _raise)

    setup_editor(tmp_path)  # must not raise

    output = capsys.readouterr().out
    assert "[WARN]" in output
