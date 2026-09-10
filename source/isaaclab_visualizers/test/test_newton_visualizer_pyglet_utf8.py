# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :func:`_disable_pyglet_xlib_utf8_captions`.

Some environments (observed with conda-installed pyglet on DGX Spark) have Xlib's
``Xutf8TextListToTextProperty`` fail with ``Could not create UTF8 text property`` --
typically because the X locale database isn't reachable to that environment's bundled X11
client libraries. Newton's ``RendererGL`` hits this the first time it (transitively) imports
``pyglet.window``, which creates an internal "shadow window" and sets its caption as a side
effect of the import itself -- before any of our own code runs. ``_disable_pyglet_xlib_utf8_captions``
forces pyglet's ASCII-only text-property codepath instead, since Newton's window captions
("Newton", "Newton Viewer", "Newton RTX Viewer") never need UTF-8 encoding.
"""

from __future__ import annotations

import sys
import types

import pytest
from isaaclab_visualizers.newton.newton_visualizer import _disable_pyglet_xlib_utf8_captions

pytestmark = [pytest.mark.unit]


def test_disable_pyglet_xlib_utf8_captions_noop_on_non_linux(monkeypatch: pytest.MonkeyPatch):
    """Test the guard does nothing on platforms that don't use pyglet's Xlib backend."""
    monkeypatch.setattr(sys, "platform", "win32")

    # Would raise if it tried to import pyglet.window.xlib (not exercised here), proving the
    # early return happened.
    _disable_pyglet_xlib_utf8_captions()


def test_disable_pyglet_xlib_utf8_captions_patches_already_imported_module(monkeypatch: pytest.MonkeyPatch):
    """Test the flag is forced off on the real module when importing it raises no error.

    ``import pyglet.window.xlib as x`` resolves ``x`` via attribute lookup on the already
    cached parent ``pyglet.window`` module, not a direct ``sys.modules`` dict lookup -- so this
    exercises the real module rather than a faked ``sys.modules`` entry, which the import
    statement would bypass.
    """
    import pyglet.window.xlib as real_pyglet_xlib

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(real_pyglet_xlib, "_have_utf8", True)

    _disable_pyglet_xlib_utf8_captions()

    assert real_pyglet_xlib._have_utf8 is False


def test_disable_pyglet_xlib_utf8_captions_recovers_from_import_time_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test the guard still patches the flag when importing pyglet.window.xlib itself raises.

    This is the actual failure mode reported on DGX Spark: pyglet's own internal "shadow
    window" creation, triggered as an import-time side effect, raises XlibException before
    our code gets a chance to run -- but the xlib submodule finishes loading and is left in
    sys.modules before that point, so it must still be recoverable and patchable there.
    """
    monkeypatch.setattr(sys, "platform", "linux")
    fake_module = types.SimpleNamespace(_have_utf8=True)
    monkeypatch.setitem(sys.modules, "pyglet.window.xlib", fake_module)

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def _raising_import(name, *args, **kwargs):
        if name == "pyglet.window.xlib":
            raise RuntimeError("Could not create UTF8 text property")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _raising_import)

    _disable_pyglet_xlib_utf8_captions()

    assert fake_module._have_utf8 is False


def test_disable_pyglet_xlib_utf8_captions_noop_when_module_never_loaded(monkeypatch: pytest.MonkeyPatch):
    """Test a total failure to even reach sys.modules is swallowed rather than raised."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delitem(sys.modules, "pyglet.window.xlib", raising=False)

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def _raising_import(name, *args, **kwargs):
        if name == "pyglet.window.xlib":
            raise RuntimeError("Could not create UTF8 text property")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _raising_import)

    # Must not raise, even though pyglet.window.xlib never ends up in sys.modules.
    _disable_pyglet_xlib_utf8_captions()
