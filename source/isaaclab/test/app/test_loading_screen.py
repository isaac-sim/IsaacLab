# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the responsive console loading screen."""

import builtins
import errno
import os
import select
import subprocess
import sys
import textwrap
import time
from io import StringIO
from unittest.mock import Mock

import pytest
from rich.cells import cell_len
from rich.console import Console, RenderableType

from isaaclab.app import loading_screen

_ALT_SCREEN_ON = "\x1b[?1049h"
_ALT_SCREEN_OFF = "\x1b[?1049l"


def _render(display: RenderableType, width: int) -> str:
    stream = StringIO()
    Console(file=stream, width=width, color_system=None).print(display)
    return stream.getvalue()


@pytest.mark.parametrize(
    ("terminal_width", "display_width"),
    [(4, 4), (5, 5), (79, 79), (80, 80), (129, 80), (130, 130), (150, 130)],
)
def test_display_reflows_at_responsive_boundaries(terminal_width: int, display_width: int):
    screen = loading_screen.LoadingScreen(2, enabled=True, logo=False)
    screen.summary("Run", {"Task": "Cartpole", "Description": "A long description that wraps cleanly."})
    screen.stage("Loading task")

    output = _render(screen._display, terminal_width)
    lines = output.splitlines()

    assert max(map(cell_len, lines)) <= display_width
    assert cell_len(lines[-1]) == display_width
    if terminal_width >= 79:
        assert "Loading task" in lines[-1]
        assert "cleanly." in output


def test_live_updates_refresh_after_releasing_state_lock():
    screen = loading_screen.LoadingScreen(2, enabled=True)
    lock_states: list[bool] = []

    class _RefreshProbe:
        def refresh(self) -> None:
            lock_states.append(screen._render_lock._is_owned())

    probe = _RefreshProbe()
    screen._live = probe

    screen.summary("Run", {"Task": "Cartpole"})
    screen.stage("Loading task")
    screen.set_activity("Loading assets")

    assert lock_states == [False, False, False]


class _OpenStringIO(StringIO):
    def close(self) -> None:
        pass


def _prepare_redirect(screen: loading_screen.LoadingScreen, monkeypatch: pytest.MonkeyPatch) -> _OpenStringIO:
    output = _OpenStringIO()
    monkeypatch.setenv("TERM", "xterm-256color")

    def redirect() -> None:
        screen._console = output
        screen._saved_fds = (-1, -1)

    def restore() -> str:
        screen._saved_fds = None
        return "hidden diagnostic\n"

    monkeypatch.setattr(screen, "_redirect", redirect)
    monkeypatch.setattr(screen, "_restore", restore)
    return output


def test_live_lifecycle_restores_normal_screen(monkeypatch: pytest.MonkeyPatch):
    screen = loading_screen.LoadingScreen(2, enabled=True)
    output = _prepare_redirect(screen, monkeypatch)
    original_input = builtins.input

    with screen:
        screen.summary("Run", {"Task": "Cartpole"})
        screen.stage("Loading task")
        assert screen._rich_console is not None
        assert screen._rich_console.color_system == "truecolor"
        screen.close()

    rendered = output.getvalue()
    normal_screen = rendered.index(_ALT_SCREEN_OFF)

    assert _ALT_SCREEN_ON in rendered
    assert "Loading task" in rendered[:normal_screen]
    assert "%" in rendered[:normal_screen]
    assert rendered.rindex("Run") > normal_screen
    assert "hidden diagnostic" not in rendered
    assert "Ready in" in rendered
    assert builtins.input is original_input


@pytest.mark.skipif(sys.platform == "win32", reason="Pseudo-terminals require POSIX.")
def test_interactive_input_shows_captured_context_before_waiting_for_answer():
    import pty

    script = textwrap.dedent("""\
        import builtins
        from isaaclab.app.loading_screen import LoadingScreen

        original_input = builtins.input
        with LoadingScreen(1, logo=False) as screen:
            screen.summary("Input QA", {"Task": "Terminal"})
            screen.stage("Launching")
            print("License context before the question.", flush=True)
            with LoadingScreen(1, enabled=False):
                answer = input("Accept terms? ")
            assert builtins.input is original_input
            print(f"Answer: {answer}", flush=True)
            screen.stage("Loading policy after answer")
            screen.summary("Post-prompt QA", {"Task": "Terminal"})
            screen.close()
        assert builtins.input is original_input
        print("Normal output restored.", flush=True)
        """)
    master, slave = pty.openpty()
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", script],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env={**os.environ, "TERM": "xterm-256color"},
    )
    os.close(slave)
    output = bytearray()
    try:
        deadline = time.monotonic() + 10.0
        while b"Accept terms? " not in output and process.poll() is None and time.monotonic() < deadline:
            if select.select([master], [], [], 0.1)[0]:
                try:
                    chunk = os.read(master, 65536)
                except OSError as error:
                    if error.errno == errno.EIO:
                        break
                    raise
                output.extend(chunk)
        assert b"Accept terms? " in output, output[-1000:].decode(errors="replace")
        assert output.index(b"License context before the question.") < output.index(b"Accept terms? ")

        os.write(master, b"NO\n")
        assert process.wait(timeout=10.0) == 0
        while select.select([master], [], [], 0.1)[0]:
            try:
                chunk = os.read(master, 65536)
            except OSError as error:
                if error.errno == errno.EIO:
                    break
                raise
            if not chunk:
                break
            output.extend(chunk)
        assert b"Answer: NO" in output
        assert output.index(b"Loading policy after answer") > output.index(b"Answer: NO")
        assert output.index(b"Post-prompt QA") > output.index(b"Answer: NO")
        assert b"Normal output restored." in output
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10.0)
        os.close(master)


def test_shutdown_restores_output_when_live_rendering_fails(monkeypatch: pytest.MonkeyPatch):
    screen = loading_screen.LoadingScreen(1, enabled=True)
    _prepare_redirect(screen, monkeypatch)
    original_input = builtins.input
    restore = Mock(wraps=screen._restore)
    monkeypatch.setattr(screen, "_restore", restore)

    with pytest.raises(RuntimeError, match="render failed"):
        with screen:
            stop = screen._live.stop

            def fail_after_stop() -> None:
                stop()
                raise RuntimeError("render failed")

            monkeypatch.setattr(screen._live, "stop", fail_after_stop)
            screen.close()

    restore.assert_called_once_with()
    assert builtins.input is original_input


def test_redirect_uses_stream_swapping_on_platforms_without_descriptor_redirection(monkeypatch: pytest.MonkeyPatch):
    original_stdout = _OpenStringIO()
    original_stderr = _OpenStringIO()
    monkeypatch.setattr(loading_screen.sys, "platform", "win32")
    monkeypatch.setattr(loading_screen.sys, "stdout", original_stdout)
    monkeypatch.setattr(loading_screen.sys, "stderr", original_stderr)
    screen = loading_screen.LoadingScreen(1, enabled=True)

    screen._redirect()
    print("hidden output")
    assert screen._saved_fds is None
    assert screen._saved_streams == (original_stdout, original_stderr)

    assert screen._restore() == "hidden output\n"
    assert screen._console is original_stdout
    assert loading_screen.sys.stdout is original_stdout
    assert loading_screen.sys.stderr is original_stderr
