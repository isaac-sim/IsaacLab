# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the responsive console loading screen."""

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
    ("terminal_width", "display_width", "logo"),
    [
        (4, 4, None),
        (5, 5, None),
        (79, 79, None),
        (80, 80, ".-------."),
        (119, 80, ".-------."),
        (120, 120, "██╗"),
        (140, 120, "██╗"),
    ],
)
def test_display_reflows_at_responsive_boundaries(terminal_width: int, display_width: int, logo: str | None):
    screen = loading_screen.LoadingScreen(2, enabled=True)
    screen.summary("Run", {"Task": "Cartpole", "Description": "A long description that wraps cleanly."})
    screen.stage("Loading task")

    output = _render(screen._display, terminal_width)
    lines = output.splitlines()

    assert max(map(cell_len, lines)) <= display_width
    assert cell_len(lines[-1]) == display_width
    assert ("Welcome to Isaac Lab!" in output) is (logo is not None)
    assert (".-------." in output) is (logo == ".-------.")
    assert ("██╗" in output) is (logo == "██╗")
    if terminal_width >= 79:
        assert "Loading task" in lines[-1]
        assert "cleanly." in output
    if logo is not None:
        header = next(line for line in lines if line.startswith("╭"))
        assert header.index("╭") < header.index("Welcome")


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


def test_shutdown_restores_output_when_live_rendering_fails(monkeypatch: pytest.MonkeyPatch):
    screen = loading_screen.LoadingScreen(1, enabled=True)
    screen._live = Mock()
    screen._live.stop.side_effect = RuntimeError("render failed")
    screen._saved_fds = (-1, -1)
    screen._console = Mock()
    restore = Mock(return_value="")
    monkeypatch.setattr(screen, "_restore", restore)

    with pytest.raises(RuntimeError, match="render failed"):
        screen.close()

    restore.assert_called_once_with()


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_redirect_uses_stream_swapping_on_platforms_without_descriptor_redirection(
    monkeypatch: pytest.MonkeyPatch, platform: str
):
    original_stdout = _OpenStringIO()
    original_stderr = _OpenStringIO()
    monkeypatch.setattr(loading_screen.sys, "platform", platform)
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
