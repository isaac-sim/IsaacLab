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

from isaaclab.cli.utils import print_debug, print_warning

# WM_CLASS values pyglet derives from each viewer's window caption at creation time.
# See ``newton/_src/viewer/gl/opengl.py`` (``title="Newton"``) and
# ``newton/_src/viewer/viewer_rtx.py`` (``caption="Newton RTX Viewer"``).
_NEWTON_GL_WM_CLASS = "Newton"
_NEWTON_RTX_WM_CLASS = "Newton RTX Viewer"

_ICON_NAME = "isaaclab-newton-viewer"

#: Pixel sizes of Newton's bundled icon (``newton/_src/viewer/gl/icon_*.png``). Shared with
#: ``isaaclab_visualizers.newton.newton_visualizer``, which sets the same icon on the live
#: Newton RTX window; keep both in sync if Newton's bundled sizes ever change.
NEWTON_ICON_SIZES = (16, 32, 64)


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


def newton_icon_source_dir() -> Path | None:
    """Return the directory containing Newton's bundled ``icon_{16,32,64}.png``, if installed.

    Single source of truth for locating Newton's bundled icon directory — also used by
    ``isaaclab_visualizers.newton.newton_visualizer`` to set the live Newton RTX window's icon,
    so both stay consistent if Newton ever moves these files.

    Resolves the path via :func:`importlib.util.find_spec` rather than importing
    ``newton._src.viewer.gl.opengl`` — that module pulls in ``warp``/``pyglet``/GL at import
    time, and importing it here (at the tail of ``isaaclab.sh -i``, best-effort) risks raising
    something other than ``ImportError`` on a broken or partial install, which would violate
    :func:`install_desktop_icons`'s documented never-raises contract.
    """
    spec = importlib.util.find_spec("newton")
    if spec is None or not spec.submodule_search_locations:
        return None
    icon_dir = Path(spec.submodule_search_locations[0]) / "_src" / "viewer" / "gl"
    return icon_dir if icon_dir.is_dir() else None


def _install_icon_files(icon_source_dir: Path, data_home: Path) -> int:
    """Copy Newton's icon into the user's hicolor icon theme. Returns how many sizes were copied."""
    copied = 0
    for size in NEWTON_ICON_SIZES:
        src = icon_source_dir / f"icon_{size}.png"
        if not src.is_file():
            continue
        dst_dir = data_home / "icons" / "hicolor" / f"{size}x{size}" / "apps"
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst_dir / f"{_ICON_NAME}.png")
        copied += 1
    return copied


def _desktop_entry(name: str, wm_class: str) -> str:
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Name={name}\n"
        "Comment=Isaac Lab Newton physics visualizer\n"
        f"Icon={_ICON_NAME}\n"
        "Exec=true\n"
        "Terminal=false\n"
        "NoDisplay=true\n"
        f"StartupWMClass={wm_class}\n"
        "Categories=Development;\n"
    )


def _install_desktop_entries(data_home: Path) -> Path:
    apps_dir = data_home / "applications"
    apps_dir.mkdir(parents=True, exist_ok=True)
    (apps_dir / "isaaclab-newton-gl-viewer.desktop").write_text(_desktop_entry("Newton Viewer", _NEWTON_GL_WM_CLASS))
    (apps_dir / "isaaclab-newton-rtx-viewer.desktop").write_text(
        _desktop_entry("Newton RTX Viewer", _NEWTON_RTX_WM_CLASS)
    )
    return apps_dir


def _refresh_desktop_caches(apps_dir: Path, data_home: Path) -> None:
    # Best-effort: these tools may be absent (e.g. minimal Linux installs); a missing cache
    # refresh just means the icon appears after the next login rather than immediately.
    for cmd in (
        ["update-desktop-database", str(apps_dir)],
        ["gtk-update-icon-cache", "-f", "-t", str(data_home / "icons" / "hicolor")],
    ):
        if shutil.which(cmd[0]) is None:
            continue
        with contextlib.suppress(subprocess.SubprocessError, OSError):
            subprocess.run(cmd, capture_output=True, check=False, timeout=30)


def install_desktop_icons() -> None:
    """Best-effort install of ``.desktop`` entries for Newton's viewer windows.

    No-ops outside a Linux graphical session (headless servers, CI, Docker, Windows), and
    never raises — a failure here should never break ``isaaclab.sh -i``.
    """
    if not _has_graphical_session():
        return

    icon_source_dir = newton_icon_source_dir()
    if icon_source_dir is None:
        print_debug("Skipping desktop icon install: Newton is not installed in this environment.")
        return

    data_home = xdg_data_home()
    with contextlib.suppress(OSError):
        copied = _install_icon_files(icon_source_dir, data_home)
        if copied == 0:
            print_warning("No Newton icon files found; skipping desktop icon install.")
            return
        if copied < len(NEWTON_ICON_SIZES):
            print_warning("Some Newton icon files were missing; desktop icons may be incomplete.")
        apps_dir = _install_desktop_entries(data_home)
        _refresh_desktop_caches(apps_dir, data_home)
