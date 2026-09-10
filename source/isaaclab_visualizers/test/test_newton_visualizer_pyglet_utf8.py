# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :func:`_make_pyglet_xlib_caption_setting_resilient`.

Some environments (observed with conda-installed pyglet on DGX Spark) have Xlib's
``Xutf8TextListToTextProperty`` fail with ``Could not create UTF8 text property`` --
typically because the X locale database isn't reachable to that environment's bundled X11
client libraries. Newton's ``RendererGL`` hits this the first time it (transitively) imports
``pyglet.window``, which creates an internal "shadow window" and sets its caption as a side
effect of the import itself -- before any of our own code runs.
``_make_pyglet_xlib_caption_setting_resilient`` wraps ``XlibWindow._set_text_property`` to
retry ASCII-only on that specific failure, deliberately narrower than disabling pyglet's
``_have_utf8`` flag outright (which would also silently disable XIC/IME keyboard input).

Tests that need to observe the patched method drive it through the "import itself raised"
recovery path (patching ``builtins.__import__`` to fail for ``pyglet.window.xlib``) rather
than a plain successful import: ``import pyglet.window.xlib as x`` resolves ``x`` via
attribute lookup on the already-imported real ``pyglet.window`` package in this process, which
would silently bypass a faked ``sys.modules`` entry. The recovery path falls back to a direct
``sys.modules.get(...)`` lookup instead, which does respect it.
"""

from __future__ import annotations

import builtins
import os
import subprocess
import sys
import types

import pytest
from isaaclab_visualizers.newton.newton_visualizer import _make_pyglet_xlib_caption_setting_resilient

pytestmark = [pytest.mark.unit]

_CHARACTERIZATION_SCRIPT = """
import sys
from pyglet.libs.x11 import xlib
xlib.Xutf8TextListToTextProperty = lambda *a, **k: -1
try:
    import pyglet.window
    print("NO_RAISE")
except Exception as e:
    left_loaded = "pyglet.window.xlib" in sys.modules
    has_class = hasattr(sys.modules.get("pyglet.window.xlib"), "XlibWindow")
    print(f"RAISED:{type(e).__name__}:{left_loaded}:{has_class}")
"""


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="requires a real X11 display")
def test_real_pyglet_leaves_patchable_xlib_module_after_failed_shadow_window_import():
    """Characterization test: verifies the real pyglet behavior the recovery path depends on.

    The rest of this file's tests exercise ``_make_pyglet_xlib_caption_setting_resilient``
    against a fake ``pyglet.window.xlib`` module that's set up to always be present in
    ``sys.modules`` by construction -- they can't detect a future pyglet release changing that
    real behavior (e.g. no longer leaving the submodule loaded, or leaving it without a usable
    ``XlibWindow`` class). This test runs in a subprocess for a genuinely fresh interpreter
    (pyglet.window may already be cached from other tests in this process) and drives the real
    import machinery instead, to catch that class of regression directly.
    """
    result = subprocess.run(
        [sys.executable, "-c", _CHARACTERIZATION_SCRIPT],
        capture_output=True,
        text=True,
        timeout=30,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stderr
    output = result.stdout.strip()
    assert output.startswith("RAISED:"), f"expected the real pyglet.window import to raise, got: {output!r}"
    _, exception_name, left_loaded, has_class = output.split(":")
    assert exception_name == "XlibException"
    assert left_loaded == "True", "pyglet.window.xlib was not left in sys.modules after the failed import"
    assert has_class == "True", "pyglet.window.xlib was left without a usable XlibWindow class"


class _FakeXlibException(Exception):
    pass


class _FakeXlibWindow:
    def _set_text_property(self, name: str, value: str, allow_utf8: bool = True) -> None:
        self.calls.append((name, value, allow_utf8))
        if allow_utf8 and self.fail_utf8:
            raise _FakeXlibException("Could not create UTF8 text property")
        if not allow_utf8 and self.fail_ascii:
            raise _FakeXlibException("Could not create text property")


@pytest.fixture
def fake_pyglet_xlib(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    """Install a fake ``pyglet.window.xlib`` module and force the import-recovery code path.

    Simulates the real DGX Spark failure mode: the first ``import pyglet.window.xlib`` raises
    (pyglet's own shadow-window creation), but the submodule is left loaded in ``sys.modules``.
    """
    monkeypatch.setattr(sys, "platform", "linux")
    fake_module = types.SimpleNamespace(XlibWindow=_FakeXlibWindow, XlibException=_FakeXlibException)
    monkeypatch.setitem(sys.modules, "pyglet.window.xlib", fake_module)

    real_import = builtins.__import__

    def _raising_import(name, *args, **kwargs):
        if name == "pyglet.window.xlib":
            raise RuntimeError("Could not create UTF8 text property")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _raising_import)
    return fake_module


def _make_window(fail_utf8: bool = False, fail_ascii: bool = False) -> _FakeXlibWindow:
    window = _FakeXlibWindow()
    window.calls = []
    window.fail_utf8 = fail_utf8
    window.fail_ascii = fail_ascii
    return window


def test_noop_on_non_linux(monkeypatch: pytest.MonkeyPatch):
    """Test the guard does nothing on platforms that don't use pyglet's Xlib backend."""
    monkeypatch.setattr(sys, "platform", "win32")

    # Would raise if it tried to import pyglet.window.xlib (not exercised here), proving the
    # early return happened.
    _make_pyglet_xlib_caption_setting_resilient()


def test_patches_real_module_without_touching_have_utf8(monkeypatch: pytest.MonkeyPatch):
    """Test the real XlibWindow._set_text_property is wrapped, and _have_utf8 is left alone.

    This is the whole point of the narrower fix over disabling _have_utf8 outright: XIC/IME
    keyboard input (gated by _have_utf8) must keep working.
    """
    import pyglet.window.xlib as real_pyglet_xlib

    monkeypatch.setattr(sys, "platform", "linux")
    original_method = real_pyglet_xlib.XlibWindow._set_text_property
    have_utf8_before = real_pyglet_xlib._have_utf8

    try:
        _make_pyglet_xlib_caption_setting_resilient()

        assert real_pyglet_xlib.XlibWindow._set_text_property is not original_method
        assert real_pyglet_xlib._have_utf8 == have_utf8_before
    finally:
        real_pyglet_xlib.XlibWindow._set_text_property = original_method


def test_resilient_set_text_property_retries_ascii_on_xlib_exception(fake_pyglet_xlib: types.SimpleNamespace):
    """Test a UTF8 failure is retried ASCII-only rather than propagating."""
    _make_pyglet_xlib_caption_setting_resilient()

    window = _make_window(fail_utf8=True)
    window._set_text_property("_NET_WM_NAME", "Newton Viewer")

    assert window.calls == [
        ("_NET_WM_NAME", "Newton Viewer", True),
        ("_NET_WM_NAME", "Newton Viewer", False),
    ]


def test_resilient_set_text_property_reraises_when_ascii_also_fails(fake_pyglet_xlib: types.SimpleNamespace):
    """Test a genuine total Xlib failure (both encodings) still propagates rather than being swallowed."""
    _make_pyglet_xlib_caption_setting_resilient()

    window = _make_window(fail_utf8=True, fail_ascii=True)

    with pytest.raises(_FakeXlibException):
        window._set_text_property("_NET_WM_NAME", "Newton Viewer")


def test_resilient_set_text_property_does_not_retry_when_allow_utf8_false(fake_pyglet_xlib: types.SimpleNamespace):
    """Test callers that already requested ASCII-only (allow_utf8=False) get a single attempt."""
    _make_pyglet_xlib_caption_setting_resilient()

    window = _make_window(fail_utf8=True, fail_ascii=True)

    with pytest.raises(_FakeXlibException):
        window._set_text_property("WM_NAME", "Newton Viewer", allow_utf8=False)

    assert window.calls == [("WM_NAME", "Newton Viewer", False)]


def test_resilient_set_text_property_succeeds_without_retry_when_utf8_works(
    fake_pyglet_xlib: types.SimpleNamespace,
):
    """Test the common case (UTF8 just works) makes exactly one call, no unnecessary retry."""
    _make_pyglet_xlib_caption_setting_resilient()

    window = _make_window()
    window._set_text_property("_NET_WM_NAME", "Newton Viewer")

    assert window.calls == [("_NET_WM_NAME", "Newton Viewer", True)]


def test_noop_when_module_never_loaded(monkeypatch: pytest.MonkeyPatch):
    """Test a total failure to even reach sys.modules is swallowed rather than raised."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delitem(sys.modules, "pyglet.window.xlib", raising=False)

    real_import = builtins.__import__

    def _raising_import(name, *args, **kwargs):
        if name == "pyglet.window.xlib":
            raise RuntimeError("Could not create UTF8 text property")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _raising_import)

    # Must not raise, even though pyglet.window.xlib never ends up in sys.modules.
    _make_pyglet_xlib_caption_setting_resilient()
