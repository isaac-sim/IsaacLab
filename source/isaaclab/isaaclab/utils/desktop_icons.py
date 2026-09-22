# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installs XDG ``.desktop`` entries so Linux desktop docks/window-switchers show a real
icon for the Newton viewer windows instead of a generic fallback.

Desktop environments such as GNOME resolve dock and Alt-Tab icons by matching a window's
``WM_CLASS`` against an installed ``.desktop`` file's ``StartupWMClass`` — they do not use
the window's own ``_NET_WM_ICON`` hint for that purpose. Newton's GL and RTX viewer windows
already set (or, for RTX, are made to set — see ``isaaclab_visualizers``) that hint
correctly, but without a matching ``.desktop`` file on the system, GNOME's dock/Alt-Tab
still falls back to a generic icon. This is not specific to any one platform or GPU — it
is generic Linux desktop-environment behavior wherever a ``.desktop`` file is absent.
"""

import contextlib
import importlib.util
import os
import platform
import shutil
import subprocess
from pathlib import Path

# WM_CLASS values pyglet derives from each viewer's window caption at creation time.
# See ``newton/_src/viewer/gl/opengl.py`` (``title="Newton"``) and
# ``newton/_src/viewer/viewer_rtx.py`` (``caption="Newton RTX Viewer"``).
_NEWTON_GL_WM_CLASS = "Newton"
_NEWTON_RTX_WM_CLASS = "Newton RTX Viewer"

_ICON_FILENAME = "icon_64.png"


def xdg_data_home() -> Path:
    """Return ``$XDG_DATA_HOME``, falling back to ``~/.local/share``.

    Per the XDG Base Directory spec, an unset *or empty* ``XDG_DATA_HOME`` both mean "use the
    default" — ``os.environ.get("XDG_DATA_HOME", default)`` alone does not handle the empty-string
    case, since the key is still present in the environment.
    """
    return Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share"))


def _has_graphical_session() -> bool:
    """Return whether a Linux graphical session is present to install desktop icons for."""
    if platform.system().lower() != "linux":
        return False
    if os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"):
        return False
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def newton_icon_path() -> Path | None:
    """Return the path to Newton's bundled ``icon_64.png``, if installed.

    Resolves the path via :func:`importlib.util.find_spec` rather than importing
    ``newton._src.viewer.gl.opengl`` — that module pulls in ``warp``/``pyglet``/GL at import
    time, and importing it here (at the tail of ``isaaclab.sh -i``, best-effort) risks raising
    something other than ``ImportError`` on a broken or partial install, which would violate
    :func:`install_desktop_icons`'s documented never-raises contract.
    """
    spec = importlib.util.find_spec("newton")
    if spec is None or not spec.submodule_search_locations:
        return None
    icon_path = Path(spec.submodule_search_locations[0]) / "_src" / "viewer" / "gl" / _ICON_FILENAME
    return icon_path if icon_path.is_file() else None


def _install_desktop_entries(icon_path: Path, data_home: Path) -> Path:
    """Write one ``.desktop`` entry per Newton viewer window, pointing ``Icon=`` at *icon_path*.

    The Desktop Entry specification allows ``Icon=`` to be an absolute path, so this references
    Newton's bundled icon directly rather than installing it into an icon theme.
    """
    apps_dir = data_home / "applications"
    apps_dir.mkdir(parents=True, exist_ok=True)
    for filename, name, wm_class in (
        ("isaaclab-newton-gl-viewer.desktop", "Newton Viewer", _NEWTON_GL_WM_CLASS),
        ("isaaclab-newton-rtx-viewer.desktop", "Newton RTX Viewer", _NEWTON_RTX_WM_CLASS),
    ):
        (apps_dir / filename).write_text(
            "[Desktop Entry]\n"
            "Type=Application\n"
            f"Name={name}\n"
            "Comment=Isaac Lab Newton physics visualizer\n"
            f"Icon={icon_path}\n"
            "Exec=true\n"
            "Terminal=false\n"
            "NoDisplay=true\n"
            f"StartupWMClass={wm_class}\n"
            "Categories=Development;\n"
        )
    return apps_dir


def refresh_desktop_database(apps_dir: Path) -> None:
    # Best-effort: this tool may be absent (e.g. minimal Linux installs); a missing cache
    # refresh just means the icon appears after the next login rather than immediately.
    if shutil.which("update-desktop-database") is None:
        return
    with contextlib.suppress(subprocess.SubprocessError, OSError):
        subprocess.run(["update-desktop-database", str(apps_dir)], capture_output=True, check=False, timeout=30)


def install_desktop_icons() -> None:
    """Best-effort install of ``.desktop`` entries for Newton's viewer windows.

    No-ops outside a Linux graphical session (headless servers, CI, Docker, Windows), and
    never raises — a failure here should never break ``isaaclab.sh -i``.
    """
    if not _has_graphical_session():
        return

    icon_path = newton_icon_path()
    if icon_path is None:
        # Mirrors isaaclab.cli.utils.print_debug's gating without importing isaaclab.cli
        # (isaaclab.utils should not depend on isaaclab.cli).
        if os.environ.get("DEBUG") == "1":
            print("[DEBUG] Skipping desktop icon install: Newton is not installed in this environment.")
        return

    with contextlib.suppress(OSError, RuntimeError):
        apps_dir = _install_desktop_entries(icon_path, xdg_data_home())
        refresh_desktop_database(apps_dir)
