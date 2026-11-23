# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Linux desktop entries that let docks show visualizer window icons."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def write_desktop_entry(filename: str, name: str, wm_class: str, icon: str | Path) -> None:
    """Write a hidden desktop entry so Linux docks show *icon* for windows with *wm_class*.

    Docks such as GNOME's pick a window's icon by matching its ``WM_CLASS`` to a desktop entry's
    ``StartupWMClass`` rather than using the window's own icon. The entry is hidden from menus
    and exists only for that match. Does nothing outside a Linux graphical session or when an
    identical entry exists; failures are logged, never raised.

    Args:
        filename: Desktop entry file name without the ``.desktop`` suffix.
        name: Display name of the application.
        wm_class: ``WM_CLASS`` of the window.
        icon: Absolute path to the icon image.
    """
    if not sys.platform.startswith("linux") or not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return
    content = (
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Name={name}\n"
        f"Icon={icon}\n"
        "Exec=true\n"
        "NoDisplay=true\n"
        f"StartupWMClass={wm_class}\n"
    )
    try:
        data_home = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
        path = data_home / "applications" / f"{filename}.desktop"
        if path.is_file() and path.read_text(encoding="utf-8") == content:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except (OSError, RuntimeError, ValueError) as error:
        logger.debug("Could not write desktop entry for %s: %s", name, error)
