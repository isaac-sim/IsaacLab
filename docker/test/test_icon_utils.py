# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import re
from pathlib import Path
from unittest.mock import patch

import pytest

from docker.utils import icon_utils

pytestmark = pytest.mark.unit

_FAKE_CID = "fakecid123"


def _docker_run_stub(kit_file_text: str | None, newton_icon_bytes: bytes | None, isaac_sim_icon_bytes: bytes | None):
    """Build a ``subprocess.run`` stub simulating ``docker create``/``docker cp``/``docker rm``.

    Any ``None`` payload makes the corresponding ``docker cp`` fail (returncode 1), so tests can
    exercise the "file missing in image" fallback paths.
    """

    def _run(cmd, **kwargs):
        class _Result:
            pass

        result = _Result()
        if cmd[:2] == ["docker", "create"]:
            result.returncode = 0
            result.stdout = f"{_FAKE_CID}\n"
        elif cmd[:2] == ["docker", "cp"]:
            src, dest = cmd[2], cmd[3]
            payload = None
            if "isaaclab.python.kit" in src:
                payload = kit_file_text.encode() if kit_file_text is not None else None
            elif "newton" in src and "icon" in src:
                payload = newton_icon_bytes
            elif "omni.isaac.sim.png" in src:
                payload = isaac_sim_icon_bytes
            if payload is None:
                result.returncode = 1
            else:
                Path(dest).write_bytes(payload)
                result.returncode = 0
            result.stdout = ""
        else:
            result.returncode = 0
            result.stdout = ""
        return result

    return _run


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


def _desktop_entries(fake_home: Path) -> dict[str, str]:
    apps_dir = fake_home / ".local/share/applications"
    return {p.name: p.read_text() for p in apps_dir.glob("*.desktop")} if apps_dir.exists() else {}


class TestInstallDesktopIcon:
    """Regression coverage for ``install_desktop_icon``'s per-visualizer WM_CLASS matching.

    Each running visualizer (Kit/Isaac Sim, Newton GL, Newton RTX) is a distinct X11
    application with its own WM_CLASS; GNOME only shows the correct taskbar icon when a
    ``.desktop`` entry's ``StartupWMClass`` matches it exactly. These tests pin that mapping
    so a future edit can't silently break one visualizer's icon while "fixing" another's.
    """

    def test_all_three_visualizers_get_matching_desktop_entries(self, fake_home):
        run_stub = _docker_run_stub(
            kit_file_text='[package]\ntitle = "Isaac Lab"\nversion = "3.0.0"\n[dependencies]\n',
            newton_icon_bytes=b"fake-newton-icon",
            isaac_sim_icon_bytes=b"fake-isaac-sim-icon",
        )
        with patch("subprocess.run", side_effect=run_stub):
            icon_utils.install_desktop_icon(image_name="fake-image:latest")

        entries = _desktop_entries(fake_home)
        assert set(entries) == {"IsaacSim.desktop", "Newton.desktop", "NewtonRTX.desktop"}

        assert "StartupWMClass=Isaac Lab 3.0.0" in entries["IsaacSim.desktop"]
        assert entries["Newton.desktop"].endswith("StartupWMClass=Newton")
        assert entries["NewtonRTX.desktop"].endswith("StartupWMClass=Newton RTX Viewer")

        # Newton GL and Newton RTX are different windows (different WM_CLASS) but share one
        # icon asset -- both entries must point at the same extracted file.
        newton_icon_path = re.search(r"Icon=(.+)", entries["Newton.desktop"]).group(1)
        newton_rtx_icon_path = re.search(r"Icon=(.+)", entries["NewtonRTX.desktop"]).group(1)
        assert newton_icon_path == newton_rtx_icon_path
        assert Path(newton_icon_path).read_bytes() == b"fake-newton-icon"

    def test_falls_back_to_default_wm_class_when_experience_file_unreadable(self, fake_home):
        run_stub = _docker_run_stub(
            kit_file_text=None,
            newton_icon_bytes=b"fake-newton-icon",
            isaac_sim_icon_bytes=b"fake-isaac-sim-icon",
        )
        with patch("subprocess.run", side_effect=run_stub):
            icon_utils.install_desktop_icon(image_name="fake-image:latest")

        entries = _desktop_entries(fake_home)
        assert f"StartupWMClass={icon_utils._DEFAULT_WM_CLASS}" in entries["IsaacSim.desktop"]

    def test_no_newton_desktop_entries_when_newton_icon_missing_from_image(self, fake_home):
        run_stub = _docker_run_stub(
            kit_file_text='[package]\ntitle = "Isaac Lab"\nversion = "3.0.0"\n',
            newton_icon_bytes=None,
            isaac_sim_icon_bytes=b"fake-isaac-sim-icon",
        )
        with patch("subprocess.run", side_effect=run_stub):
            icon_utils.install_desktop_icon(image_name="fake-image:latest")

        entries = _desktop_entries(fake_home)
        assert set(entries) == {"IsaacSim.desktop"}

    def test_no_desktop_entries_written_when_isaac_sim_icon_missing_from_image(self, fake_home):
        run_stub = _docker_run_stub(
            kit_file_text='[package]\ntitle = "Isaac Lab"\nversion = "3.0.0"\n',
            newton_icon_bytes=b"fake-newton-icon",
            isaac_sim_icon_bytes=None,
        )
        with patch("subprocess.run", side_effect=run_stub):
            icon_utils.install_desktop_icon(image_name="fake-image:latest")

        # Newton entries are still written (independent of the Isaac Sim icon copy); only the
        # Isaac Sim entry is skipped since its icon wasn't found.
        entries = _desktop_entries(fake_home)
        assert "IsaacSim.desktop" not in entries


class TestReadWmClassFromExperienceFile:
    def test_parses_title_and_version(self, tmp_path):
        run_stub = _docker_run_stub(
            kit_file_text='[package]\ntitle = "Isaac Lab"\nversion = "3.0.0"\ndescription = "x"\n[dependencies]\n',
            newton_icon_bytes=None,
            isaac_sim_icon_bytes=None,
        )
        with patch("subprocess.run", side_effect=run_stub):
            wm_class = icon_utils._read_wm_class_from_experience_file(_FAKE_CID, "/workspace/isaaclab")
        assert wm_class == "Isaac Lab 3.0.0"

    def test_returns_none_when_file_missing(self):
        run_stub = _docker_run_stub(kit_file_text=None, newton_icon_bytes=None, isaac_sim_icon_bytes=None)
        with patch("subprocess.run", side_effect=run_stub):
            assert icon_utils._read_wm_class_from_experience_file(_FAKE_CID, "/workspace/isaaclab") is None

    def test_returns_none_when_title_or_version_missing(self):
        run_stub = _docker_run_stub(
            kit_file_text='[package]\ntitle = "Isaac Lab"\n[dependencies]\n',
            newton_icon_bytes=None,
            isaac_sim_icon_bytes=None,
        )
        with patch("subprocess.run", side_effect=run_stub):
            assert icon_utils._read_wm_class_from_experience_file(_FAKE_CID, "/workspace/isaaclab") is None


def test_newton_wm_class_constants_match_installed_newton_package():
    """Guard against the ``newton`` pip package silently renaming its viewer window captions.

    ``_NEWTON_GL_WM_CLASS``/``_NEWTON_RTX_WM_CLASS`` are hand-derived from the installed
    ``newton`` package's viewer source (pyglet sets WM_CLASS's res_class from the window's
    ``caption``/``title`` verbatim). If a future ``newton`` release changes either string, this
    test fails loudly instead of the taskbar icon silently going generic again.
    """
    pytest.importorskip("newton")
    import inspect

    from newton._src.viewer.gl import opengl as newton_opengl
    from newton._src.viewer import viewer_rtx as newton_viewer_rtx

    gl_source = inspect.getsource(newton_opengl.RendererGL.__init__)
    gl_title_match = re.search(r'title\s*=\s*"([^"]+)"', gl_source)
    assert gl_title_match, "Could not find RendererGL.__init__'s default title= parameter."
    assert gl_title_match.group(1) == icon_utils._NEWTON_GL_WM_CLASS

    rtx_source = inspect.getsource(newton_viewer_rtx.ViewerRTX._init_window)
    rtx_caption_match = re.search(r'caption\s*=\s*"([^"]+)"', rtx_source)
    assert rtx_caption_match, "Could not find ViewerRTX._init_window's pyglet Window caption=."
    assert rtx_caption_match.group(1) == icon_utils._NEWTON_RTX_WM_CLASS
