# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for installing desktop icons for Isaac Sim / Newton on the host.

When Isaac Sim runs inside a Docker container with X11 forwarding, its internal
:func:`create_desktop_icon` call writes ``~/.local/share/applications/IsaacSim.desktop``
*inside the container*, where the host GNOME shell cannot see it.  As a result GNOME
cannot match the running window to the correct ``.desktop`` entry and falls back to the
generic Omniverse icon.

This module fixes that by extracting the relevant icons from the image and writing
``.desktop`` files on the *host* before the container starts. It covers both the Kit
(Isaac Sim) window and the Newton GL/RTX viewer windows, which are separate
applications with distinct WM_CLASS values and therefore need separate entries.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Template matches the one used by isaacsim.app.setup.app_utils.create_desktop_icon,
# except for StartupWMClass: Isaac Lab launches Kit with its own branded experience file
# (``apps/isaaclab.python.kit``), whose ``[package] title``/``version`` (not "IsaacSim")
# become the window's WM_CLASS at runtime. {wm_class} is filled in from that file so
# StartupWMClass actually links GNOME to this entry when the container window opens.
_DESKTOP_ENTRY_TEMPLATE = """\
[Desktop Entry]
Version=1.0
Name={name}
Exec=
Icon={icon_path}
Terminal=false
Type=Application
StartupWMClass={wm_class}"""

# Fallback WM_CLASS used when the experience file can't be read from the image
# (matches the upstream isaacsim.app.setup default, in case a non-Isaac-Lab experience
# file is used to launch Kit).
_DEFAULT_WM_CLASS = "IsaacSim"

# Candidate icon paths inside the Isaac Sim image, tried in order.
# Isaac Sim 4.x / 6.x ships the icon under the isaacsim.app.setup extension;
# Isaac Sim 5.x RC moved it to data/icon/ under the root.
_ICON_PATHS_IN_IMAGE = [
    "{root}/exts/isaacsim.app.setup/data/omni.isaac.sim.png",
    "{root}/exts/isaacsim.simulation_app/data/omni.isaac.sim.png",
    "{root}/data/icon/omni.isaac.sim.png",
]

# The experience file used for the default (non-headless, non-camera) launch mode,
# i.e. the one whose window would actually show a taskbar icon.
_EXPERIENCE_FILE_IN_IMAGE = "{isaaclab_path}/apps/isaaclab.python.kit"

# The Newton GL/RTX viewers (newton.viewer.ViewerGL / ViewerRTX, both pyglet windows) are
# separate applications from Kit, with their own WM_CLASS. Only ViewerGL embeds an icon
# itself (via newton/_src/viewer/gl/opengl.py::_set_icon(), which loads icon_{16,32,64}.png
# through pyglet's window.set_icon()); ViewerRTX never calls that, so neither viewer gets a
# taskbar icon without a matching .desktop entry, same root cause as the Isaac Sim icon bug.
_NEWTON_ICON_PATHS_IN_IMAGE = [
    "{root}/kit/python/lib/python3.12/site-packages/newton/_src/viewer/gl/icon_64.png",
    "{root}/kit/python/lib/python3.11/site-packages/newton/_src/viewer/gl/icon_64.png",
    "{root}/kit/python/lib/python3.10/site-packages/newton/_src/viewer/gl/icon_64.png",
    "{root}/python/lib/python3.12/site-packages/newton/_src/viewer/gl/icon_64.png",
]

# WM_CLASS values pyglet assigns each viewer window, derived from the ``caption``/``title``
# passed to ``pyglet.window.Window(...)`` at construction time (pyglet sets WM_CLASS's
# res_class to that string verbatim -- see pyglet.window.xlib.XlibWindow.set_wm_class).
# ViewerGL is constructed with the OpenGLRenderer default ``title="Newton"`` (Isaac Lab's
# integration only changes the *displayed* caption afterward via ``set_title()``, which does
# not touch WM_CLASS); ViewerRTX passes ``caption="Newton RTX Viewer"`` directly.
_NEWTON_GL_WM_CLASS = "Newton"
_NEWTON_RTX_WM_CLASS = "Newton RTX Viewer"


def _copy_first_match(cid: str, src_templates: list[str], dest: Path, **format_kwargs) -> bool:
    """Try ``docker cp`` for each templated source path in order; return True on first success."""
    for src_template in src_templates:
        src_path = src_template.format(**format_kwargs)
        cp_result = subprocess.run(["docker", "cp", f"{cid}:{src_path}", str(dest)], capture_output=True)
        if cp_result.returncode == 0:
            return True
    return False


def _read_wm_class_from_experience_file(cid: str, isaaclab_path: str) -> str | None:
    """Read the ``[package] title``/``version`` from the Kit experience file in a container.

    Kit derives the running window's WM_CLASS from the experience file's ``[package]``
    section as ``"{title} {version}"``. Returns ``None`` if the file can't be read or
    parsed.
    """
    src_path = _EXPERIENCE_FILE_IN_IMAGE.format(isaaclab_path=isaaclab_path)
    with tempfile.TemporaryDirectory() as tmp_dir:
        dest_path = Path(tmp_dir) / "isaaclab.python.kit"
        result = subprocess.run(["docker", "cp", f"{cid}:{src_path}", str(dest_path)], capture_output=True)
        if result.returncode != 0 or not dest_path.exists():
            return None
        text = dest_path.read_text(encoding="utf-8", errors="ignore")
    package_match = re.search(r"^\[package\]\s*$(.*?)^\[", text, re.MULTILINE | re.DOTALL)
    package_section = package_match.group(1) if package_match else text
    title_match = re.search(r'^title\s*=\s*"([^"]+)"', package_section, re.MULTILINE)
    version_match = re.search(r'^version\s*=\s*"([^"]+)"', package_section, re.MULTILINE)
    if not title_match or not version_match:
        return None
    return f"{title_match.group(1)} {version_match.group(1)}"


def _install_newton_desktop_icons(cid: str, image_name: str, isaacsim_root: str) -> None:
    """Extract the Newton viewer icon from a container and write .desktop entries for it.

    Writes ``~/.local/share/icons/newton.png`` and two desktop entries -- one per WM_CLASS
    the GL and RTX viewers use -- so GNOME shows the Newton icon for either viewer's window
    instead of falling back to a generic one.
    """
    icon_dest = Path.home() / ".local/share/icons/newton.png"
    icon_dest.parent.mkdir(parents=True, exist_ok=True)

    if not _copy_first_match(cid, _NEWTON_ICON_PATHS_IN_IMAGE, icon_dest, root=isaacsim_root):
        print(
            f"[WARNING] Newton viewer icon not found in '{image_name}'. Tried: "
            + ", ".join(p.format(root=isaacsim_root) for p in _NEWTON_ICON_PATHS_IN_IMAGE)
        )
        return

    for desktop_filename, name, wm_class in (
        ("Newton.desktop", "Newton", _NEWTON_GL_WM_CLASS),
        ("NewtonRTX.desktop", "Newton RTX", _NEWTON_RTX_WM_CLASS),
    ):
        desktop_dest = Path.home() / ".local/share/applications" / desktop_filename
        desktop_dest.parent.mkdir(parents=True, exist_ok=True)
        desktop_dest.write_text(_DESKTOP_ENTRY_TEMPLATE.format(icon_path=str(icon_dest), wm_class=wm_class, name=name))
        print(f"[INFO] Installed {name} desktop icon: {desktop_dest} (StartupWMClass={wm_class})")


def install_desktop_icon(image_name: str, isaacsim_root: str = "/isaac-sim", isaaclab_path: str = "/workspace/isaaclab") -> None:
    """Extract the Isaac Sim icon from a Docker image and install it on the host.

    Creates ``~/.local/share/icons/omni.isaac.sim.png`` and
    ``~/.local/share/applications/IsaacSim.desktop`` on the host so that GNOME
    correctly shows the Isaac Sim icon for the running container window.

    This function is a no-op on non-Linux platforms and when the host icon is
    already up-to-date.

    Args:
        image_name: Full Docker image name to extract the icon from (e.g.
            ``"isaac-lab-base:latest"``).
        isaacsim_root: Path to the Isaac Sim installation root inside the container.
            Defaults to ``"/isaac-sim"``.
        isaaclab_path: Path to the Isaac Lab checkout inside the container, used to read
            the experience file that determines the window's WM_CLASS. Defaults to
            ``"/workspace/isaaclab"`` (the standard container path).
    """
    if sys.platform != "linux":
        return

    icon_dest = Path.home() / ".local/share/icons/omni.isaac.sim.png"
    desktop_dest = Path.home() / ".local/share/applications/IsaacSim.desktop"

    icon_dest.parent.mkdir(parents=True, exist_ok=True)
    desktop_dest.parent.mkdir(parents=True, exist_ok=True)

    # Create a stopped (never-started) container so we can copy from it.
    result = subprocess.run(
        ["docker", "create", image_name, "bash"],
        capture_output=True,
        text=True,
    )
    cid = result.stdout.strip()
    if not cid:
        print(
            f"[WARNING] Could not create a temporary container from '{image_name}' to extract the"
            " Isaac Sim icon. The desktop icon will not be installed."
        )
        return

    wm_class = None
    try:
        icon_copied = _copy_first_match(cid, _ICON_PATHS_IN_IMAGE, icon_dest, root=isaacsim_root)
        wm_class = _read_wm_class_from_experience_file(cid, isaaclab_path)
        _install_newton_desktop_icons(cid, image_name, isaacsim_root)
    finally:
        subprocess.run(["docker", "rm", cid], capture_output=True)

    if wm_class is None:
        print(
            f"[WARNING] Could not read the Isaac Lab experience file from '{image_name}' to determine"
            f" the window's WM_CLASS. Falling back to '{_DEFAULT_WM_CLASS}', which will not match the"
            " Isaac Lab window and the taskbar icon may still be incorrect."
        )
        wm_class = _DEFAULT_WM_CLASS

    if not icon_copied:
        print(
            f"[WARNING] Isaac Sim icon not found in '{image_name}'. Tried: "
            + ", ".join(p.format(root=isaacsim_root) for p in _ICON_PATHS_IN_IMAGE)
        )
        return

    desktop_dest.write_text(_DESKTOP_ENTRY_TEMPLATE.format(icon_path=str(icon_dest), wm_class=wm_class, name="Isaac Sim"))

    # Refresh the host desktop database so GNOME picks up the new entry immediately.
    if shutil.which("update-desktop-database"):
        subprocess.run(["update-desktop-database", str(desktop_dest.parent)], capture_output=True)

    print(f"[INFO] Installed Isaac Sim desktop icon: {desktop_dest} (StartupWMClass={wm_class})")
