# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Console loading screen shown while an Isaac Lab run starts up."""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from typing import IO, Any

from tqdm import tqdm

_LABEL_WIDTH = 14
_STAGE_WIDTH = 22
_ACTIVITY_WIDTH = 24
_TICK_INTERVAL = 0.1
# Reported steps per stage that fill the stage's slice of the bar, and the share
# of that slice they may fill. Sub-steps are not known in advance, so a stage
# that reports more than this keeps its progress just short of the next stage;
# only finishing the stage advances the bar the rest of the way.
_STEPS_PER_STAGE = 8
_STEP_CEILING = 0.9
_BOX = ("╭", "╮", "╰", "╯", "─", "│")
_ASCII_BOX = ("+", "+", "+", "+", "-", "|")

LOGO = r"""Welcome to Isaac Lab!

       \   /
     .-------.
     | o   o |
     |   _   |
     '-------'"""
"""Greeting drawn beside the run summary. Kept short enough to fit alongside it."""

_active_screen: LoadingScreen | None = None


def report_activity(activity: str | None) -> None:
    """Report what the startup path is doing right now.

    Call this *before* the work it names, so the loading screen shows the step
    while it runs rather than after it finishes, and pair it with a
    ``report_activity(None)`` once that work is done. Reports nest: finishing an
    inner step restores the enclosing one, so a long step keeps its label while
    its sub-steps come and go.

    Reporting is a no-op when no screen is open, so call sites do not need to
    know whether one is active.

    Args:
        activity: Short description of the work about to start, e.g.
            ``"Initializing solver"``. Pass None to report that the most
            recently reported work has finished.
    """
    if _active_screen is not None:
        _active_screen.set_activity(activity)


def format_run_summary(title: str, fields: dict[str, str], *, ascii_only: bool = False) -> str:
    """Render a boxed summary of a run's configuration.

    Args:
        title: Text shown in the top border of the box.
        fields: Label to value mapping rendered as one row per entry, in order.
        ascii_only: Whether to draw the box with ASCII characters instead of
            box-drawing characters.

    Returns:
        The rendered box, without a trailing newline.
    """
    top_left, top_right, bottom_left, bottom_right, horizontal, vertical = _ASCII_BOX if ascii_only else _BOX
    rows = [f"{label.ljust(_LABEL_WIDTH)}{value}" for label, value in fields.items()]
    width = max(len(title) + 4, max((len(row) for row in rows), default=0) + 2)
    lines = [f"{top_left}{horizontal} {title} {horizontal * (width - len(title) - 3)}{top_right}"]
    lines += [f"{vertical} {row.ljust(width - 2)} {vertical}" for row in rows]
    lines.append(f"{bottom_left}{horizontal * width}{bottom_right}")
    return "\n".join(lines)


def _join_columns(left: str, right: str, gap: int = 6) -> str:
    """Lay two blocks of text out side by side, top aligned.

    Args:
        left: Block placed in the first column; its lines are padded to a
            common width so the second column stays straight.
        right: Block placed in the second column.
        gap: Blank columns between the two blocks.

    Returns:
        The combined block, without a trailing newline.
    """
    left_lines, right_lines = left.splitlines(), right.splitlines()
    width = max(len(line) for line in left_lines)
    rows = range(max(len(left_lines), len(right_lines)))
    return "\n".join(
        (
            (left_lines[row] if row < len(left_lines) else "").ljust(width)
            + " " * gap
            + (right_lines[row] if row < len(right_lines) else "")
        ).rstrip()
        for row in rows
    )


class LoadingScreen:
    """Staged progress display that owns the console while a run starts up.

    While the screen is open, everything the startup path writes to standard
    output and error -- including native libraries writing straight to the
    underlying file descriptors -- is spooled to a temporary file instead of the
    console, so it cannot break the progress bar. Alongside the bar the screen
    shows the step currently running, as reported by :func:`report_activity`,
    and a clock that keeps ticking through long silent steps. The spool is
    replayed when startup fails and dropped when it succeeds. Closing the screen
    hands the console back, so whatever runs next (typically an RL library's
    training log) prints undisturbed.

    The screen degrades to plain stage lines, with no redirection, when it is
    disabled -- for a non-interactive console or a verbose run -- so nothing is
    hidden from a captured log.

    Example:

    .. code-block:: python

        with LoadingScreen(2) as screen:
            screen.summary("Isaac Lab", {"Task": task})
            screen.stage("Launching simulation")
            ...
            screen.stage("Creating environment")
            ...
            screen.close()
    """

    def __init__(self, num_stages: int, *, enabled: bool | None = None, logo: bool = True) -> None:
        """Initialize the screen.

        Args:
            num_stages: Number of stages the progress bar counts up to.
            enabled: Whether to draw a live progress bar and spool startup
                output. Defaults to None, which enables both when standard
                output is a terminal.
            logo: Whether :meth:`summary` greets the user with :data:`LOGO`.
                Defaults to True.
        """
        self._num_stages = num_stages
        self._logo = logo
        self._enabled = _console_is_interactive() if enabled is None else enabled
        self._console: IO[str] = sys.stdout
        self._ascii_only = not _supports_box_drawing(self._console)
        self._bar: tqdm | None = None
        self._bar_lock = threading.Lock()
        self._spool: IO[str] | None = None
        self._ticker: threading.Thread | None = None
        self._stop_ticking = threading.Event()
        self._saved_fds: tuple[int, int] | None = None
        self._started = 0.0
        self._index = 0
        self._stage = ""
        self._steps = 0
        self._activities: list[str] = []

    def __enter__(self) -> LoadingScreen:
        """Take over the console and start spooling startup output."""
        global _active_screen
        self._started = time.monotonic()
        if self._enabled:
            self._redirect()
            self._ticker = threading.Thread(target=self._tick, daemon=True)
            self._ticker.start()
        _active_screen = self
        return self

    def __exit__(self, *_: Any) -> None:
        """Replay the spooled output unless :meth:`close` already handed the console back.

        Leaving the block with the screen still open means startup never
        finished -- an exception, or an early return -- so the output that was
        hidden is exactly what the user needs to see.
        """
        self._shut_down(replay=True)

    def summary(self, title: str, fields: dict[str, str]) -> None:
        """Print a boxed run summary, greeting alongside it, above the progress bar.

        Args:
            title: Text shown in the top border of the box.
            fields: Label to value mapping rendered as one row per entry.
        """
        summary = format_run_summary(title, fields, ascii_only=self._ascii_only)
        # the greeting shares the summary's rows, so it is skipped rather than
        # allowed to run past the bottom of a short summary
        if self._logo and LOGO.count("\n") <= summary.count("\n"):
            summary = _join_columns(summary, LOGO)
        self._write(f"\n{summary}\n\n")

    def stage(self, name: str) -> None:
        """Complete the running stage, if any, and start the stage named *name*.

        Args:
            name: Human-readable name of the stage that is about to run.
        """
        self._index += 1
        if not self._enabled:
            self._write(f"[{self._index}/{self._num_stages}] {name}\n")
            return
        with self._bar_lock:
            if self._bar is None:
                self._bar = tqdm(
                    total=100,
                    file=self._console,
                    bar_format="  {desc} {bar} {percentage:3.0f}% [{elapsed}]",
                    ascii=self._ascii_only,
                    leave=False,
                    dynamic_ncols=True,
                )
            self._stage = name
            self._activities.clear()
            self._steps = 0
            self._set_progress(self._stage_progress())
            self._redraw_description()

    def set_activity(self, activity: str | None) -> None:
        """Push *activity* as the step currently running, or pop it when None.

        Prefer :func:`report_activity`, which reaches the open screen from
        anywhere in the startup path without threading this object through.
        """
        if not self._enabled:
            return
        with self._bar_lock:
            if activity is None:
                if self._activities:
                    self._activities.pop()
            else:
                self._activities.append(activity)
                self._steps += 1
                self._set_progress(self._stage_progress())
            self._redraw_description()

    def close(self) -> None:
        """Hand the console back to the rest of the run, dropping the spooled output."""
        self._shut_down(replay=False)

    def _shut_down(self, *, replay: bool) -> None:
        """Stop the clock, close the progress bar, and restore the console, once."""
        global _active_screen
        if _active_screen is self:
            _active_screen = None
        self._stop_ticking.set()
        if self._ticker is not None:
            self._ticker.join(timeout=1.0)
            self._ticker = None
        with self._bar_lock:
            if self._bar is not None:
                if not replay:
                    # only a successful hand-over completes the bar
                    self._set_progress(100)
                self._bar.close()
                self._bar = None
        if self._saved_fds is None:
            return
        hidden = self._restore()
        if replay:
            self._write(hidden)
        else:
            elapsed = time.monotonic() - self._started
            lines = hidden.count("\n")
            self._write(f"  Ready in {elapsed:.1f}s ({lines} lines of startup output hidden; use --info to show)\n\n")
        self._console.close()
        self._console = sys.stdout

    def _stage_progress(self) -> float:
        """Bar position for the current stage and the steps reported within it, in percent."""
        span = 100 / self._num_stages
        filled = min(self._steps / _STEPS_PER_STAGE, _STEP_CEILING)
        return (self._index - 1) * span + span * filled

    def _set_progress(self, percent: float) -> None:
        """Move the bar to *percent*. The caller must hold the bar lock."""
        if self._bar is not None:
            self._bar.update(percent - self._bar.n)

    def _redraw_description(self) -> None:
        """Refresh the stage and activity columns. The caller must hold the bar lock."""
        if self._bar is not None:
            activity = self._activities[-1] if self._activities else ""
            # fixed-width columns so the bar does not resize as activities come and go
            self._bar.set_description_str(f"{self._stage.ljust(_STAGE_WIDTH)}{activity.ljust(_ACTIVITY_WIDTH)}")

    def _write(self, text: str) -> None:
        """Write *text* straight to the console, bypassing the spool."""
        self._console.write(text)
        self._console.flush()

    def _redirect(self) -> None:
        """Point the standard output and error file descriptors at a spool file."""
        try:
            console_fd = os.dup(sys.stdout.fileno())
        except (AttributeError, OSError, ValueError):
            # no real file descriptor to duplicate (e.g. a captured stream); stay plain
            self._enabled = False
            return
        # the screen itself is the context manager; the spool closes in _restore
        self._spool = tempfile.TemporaryFile(mode="w+", errors="replace")  # noqa: SIM115
        self._console = os.fdopen(console_fd, "w", errors="replace")
        sys.stdout.flush()
        sys.stderr.flush()
        self._saved_fds = (os.dup(1), os.dup(2))
        os.dup2(self._spool.fileno(), 1)
        os.dup2(self._spool.fileno(), 2)

    def _restore(self) -> str:
        """Restore the standard file descriptors and return the spooled output."""
        sys.stdout.flush()
        sys.stderr.flush()
        saved_out, saved_err = self._saved_fds
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        self._saved_fds = None
        self._spool.seek(0)
        spooled = self._spool.read()
        self._spool.close()
        self._spool = None
        return spooled

    def _tick(self) -> None:
        """Redraw the bar periodically so its clock advances during silent steps."""
        while not self._stop_ticking.is_set():
            with self._bar_lock:
                if self._bar is not None:
                    self._bar.refresh()
            self._stop_ticking.wait(_TICK_INTERVAL)


def _console_is_interactive() -> bool:
    """Return whether standard output is a terminal that can host a progress bar."""
    try:
        return sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def _supports_box_drawing(stream: IO[str]) -> bool:
    """Return whether *stream* can encode box-drawing characters."""
    try:
        "".join(_BOX).encode(getattr(stream, "encoding", None) or "ascii")
    except (LookupError, UnicodeEncodeError):
        return False
    return True
