# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-viewer camera controls for the Isaac Sim 7 viewer client (experimental).

The Isaac Sim client binds its camera as Kit does: Alt+left-drag orbits, middle-drag pans and right-drag looks, while a
plain left-drag selects. Isaac Lab's Newton viewer looks with the left button and orbits with the middle one. The
client has no setting for its bindings, so this serves a copy of its shared renderer with the bindings in
``panes/viewport-camera.js`` changed:

* Left-drag looks; a left click without dragging still selects, and Shift, Ctrl or Cmd with the left button still
  select by region. Alt+left-drag still orbits.
* Middle-drag orbits, Shift+middle-drag pans and Ctrl+middle-drag dollies. The wheel dollies as before.
* Right-drag looks as before (Newton drags bodies with it, which this viewer does not support).
* WASD flies while either look drag is held. Always-on WASD would collide with the client's W/E/R transform tools.

Open point for the Isaac Sim team: configurable camera bindings in the client would replace this copy.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

_CAMERA_MODULE = Path("panes") / "viewport-camera.js"

_PATCHES = (
    # Newton-style buttons: left looks, middle orbits, Shift+middle pans, Ctrl+middle dollies.
    (
        """export function gestureFor(event) {
    if (event.button === 1)
        return 'pan';
    if (event.button === 0 && event.altKey)
        return 'orbit';
    if (event.button === 2)
        return event.altKey ? 'dolly' : 'look';
    return null;
}""",
        """export function gestureFor(event) {
    if (event.button === 1)
        return event.shiftKey ? 'pan' : event.ctrlKey ? 'dolly' : 'orbit';
    if (event.button === 0 && event.altKey)
        return 'orbit';
    if (event.button === 0 && !event.shiftKey && !event.ctrlKey && !event.metaKey)
        return 'look';
    if (event.button === 2)
        return event.altKey ? 'dolly' : 'look';
    return null;
}""",
    ),
    # A left click that did not drag still selects.
    (
        """    if (button !== 0 || gesture !== null)
        return 'navigate';""",
        """    if (button !== 0 || (gesture !== null && gesture !== 'look'))
        return 'navigate';""",
    ),
    (
        """        const mode = gesture ? null : marqueeMode(event, pickingEnabled);""",
        """        const leftLook = gesture === 'look' && event.button === 0;
        const mode = gesture && !leftLook ? null : marqueeMode(event, pickingEnabled);""",
    ),
    # Left-drag looks instead of drawing a selection rectangle.
    (
        """        if (mode && regionEnabled) {""",
        """        if (mode && regionEnabled && !leftLook) {""",
    ),
    # Flight ends with whichever look drag opened it.
    (
        """        if (finished.button === 2) {""",
        """        if (finished.gesture === 'look') {""",
    ),
)


def newton_controls_client(shared_renderer: Path, directory: Path) -> Path | None:
    """Copy the client's shared renderer into ``directory`` with Newton-viewer camera bindings.

    Args:
        shared_renderer: The installed client's shared renderer directory.
        directory: Empty directory to write the copy into.

    Returns:
        The patched renderer directory, or ``None`` if this client version does not match the expected bindings.
    """
    module = shared_renderer / _CAMERA_MODULE
    patched = module.read_text() if module.is_file() else ""
    for old, new in _PATCHES:
        if patched.count(old) != 1:
            logger.warning(
                "[IsaacSimVisualizer] This Isaac Sim client's camera bindings differ from the expected ones; keeping"
                " its own controls."
            )
            return None
        patched = patched.replace(old, new)
    target = directory / "renderer"
    shutil.copytree(shared_renderer, target)
    (target / _CAMERA_MODULE).write_text(patched)
    return target
