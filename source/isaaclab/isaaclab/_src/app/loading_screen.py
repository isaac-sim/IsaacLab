# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Console loading screen shown while an Isaac Lab run starts up."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import threading
import time
from typing import IO, Any

from rich.cells import cell_len, chop_cells, set_cell_size
from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.constrain import Constrain
from rich.live import Live
from rich.text import Text

_LABEL_WIDTH = 14
_STAGE_WIDTH = 22
_ACTIVITY_WIDTH = 24
_REFRESH_PER_SECOND = 10
_SUMMARY_WIDTH = 50
_STANDARD_WIDTH = 80
_WIDE_WIDTH = 120
_COLUMN_GAP = 6
# Reported steps per stage that fill the stage's slice of the bar, and the share
# of that slice they may fill. Sub-steps are not known in advance, so a stage
# that reports more than this keeps its progress just short of the next stage;
# only finishing the stage advances the bar the rest of the way.
_STEPS_PER_STAGE = 8
_STEP_CEILING = 0.9
_BOX = ("╭", "╮", "╰", "╯", "─", "│")
_ASCII_BOX = ("+", "+", "+", "+", "-", "|")
_WRAP_CONSOLE = Console(color_system=None, force_terminal=False, width=120)

LOGO = r"""Welcome to Isaac Lab!

       \   /
     .-------.
     | o   o |
     |   _   |
     '-------'"""
"""Greeting drawn beside the run summary. Kept short enough to fit alongside it."""

LOGO_WIDE = r"""Welcome to Isaac Lab!

██╗███████╗ █████╗  █████╗  ██████╗   ██╗      █████╗ ██████╗
██║██╔════╝██╔══██╗██╔══██╗██╔════╝   ██║     ██╔══██╗██╔══██╗
██║███████╗███████║███████║██║        ██║     ███████║██████╔╝
██║╚════██║██╔══██║██╔══██║██║        ██║     ██╔══██║██╔══██╗
██║███████║██║  ██║██║  ██║╚██████╗   ███████╗██║  ██║██████╔╝
╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝   ╚══════╝╚═╝  ╚═╝╚═════╝"""
"""Wide greeting used when a 120-column display is available."""

_active_screen: LoadingScreen | None = None


def _display_width(terminal_width: int) -> int:
    """Return the responsive width of the live display."""
    if terminal_width < _STANDARD_WIDTH:
        return terminal_width
    if terminal_width < _WIDE_WIDTH:
        return _STANDARD_WIDTH
    return _WIDE_WIDTH


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


def _format_run_summary(title: str, fields: dict[str, str], *, width: int, ascii_only: bool = False) -> str:
    """Render a boxed run summary at an exact outer width."""
    if width < 4:
        raise ValueError(f"Summary width must be at least 4 columns, got {width}")

    top_left, top_right, bottom_left, bottom_right, horizontal, vertical = _ASCII_BOX if ascii_only else _BOX
    inner_width = width - 2
    content_width = width - 4
    title_width = max(0, inner_width - 3)
    clipped_title = chop_cells(title, title_width)[0] if title and title_width else ""
    top_label = f"{horizontal} {clipped_title} " if clipped_title else horizontal
    top = f"{top_left}{top_label}{horizontal * (inner_width - cell_len(top_label))}{top_right}"

    label_width = min(_LABEL_WIDTH, max(0, content_width - 1))
    value_width = content_width - label_width
    rows: list[str] = []
    for label, value in fields.items():
        wrapped = Text(value).wrap(_WRAP_CONSOLE, value_width, overflow="fold") if value_width else []
        value_lines = [line.plain for line in wrapped] or [""]
        for index, value_line in enumerate(value_lines):
            row_label = set_cell_size(label if index == 0 else "", label_width)
            row = set_cell_size(f"{row_label}{value_line}", content_width)
            rows.append(f"{vertical} {row} {vertical}")

    lines = [top, *rows, f"{bottom_left}{horizontal * inner_width}{bottom_right}"]
    return "\n".join(lines)


def _format_header(
    title: str,
    fields: dict[str, str],
    logos: tuple[str, ...],
    terminal_width: int,
    *,
    ascii_only: bool = False,
) -> tuple[str, int]:
    """Render the summary and largest fitting logo at the responsive width."""
    display_width = _display_width(terminal_width)
    if display_width < 4:
        return "", display_width
    summary = _format_run_summary(title, fields, width=min(_SUMMARY_WIDTH, display_width), ascii_only=ascii_only)
    if terminal_width < _STANDARD_WIDTH:
        return summary, display_width

    available = display_width - _block_width(summary) - _COLUMN_GAP
    fitting = [logo for logo in logos if _block_width(logo) <= available]
    if fitting:
        summary = _join_columns(summary, max(fitting, key=_block_width))
    return summary, display_width


def _block_width(block: str) -> int:
    """Return the width of the widest line in a multiline text block."""
    return max((cell_len(line) for line in block.splitlines()), default=0)


def _join_columns(left: str, right: str, gap: int = _COLUMN_GAP) -> str:
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
    width = _block_width(left)
    rows = range(max(len(left_lines), len(right_lines)))
    return "\n".join(
        (
            set_cell_size(left_lines[row] if row < len(left_lines) else "", width)
            + " " * gap
            + (right_lines[row] if row < len(right_lines) else "")
        ).rstrip()
        for row in rows
    )


def _format_progress(
    stage: str,
    activity: str,
    percent: float,
    elapsed: float,
    width: int,
    *,
    ascii_only: bool = False,
) -> str:
    """Render the progress row at an exact terminal width."""
    if width <= 0:
        return ""

    percent = min(100.0, max(0.0, percent))
    elapsed_text = _format_elapsed(elapsed)
    suffix = f" {percent:3.0f}% [{elapsed_text}]"
    prefix = "  "
    fixed_description_width = _STAGE_WIDTH + _ACTIVITY_WIDTH
    description_width = min(fixed_description_width, max(0, width - cell_len(prefix + suffix) - 2))
    description = set_cell_size(
        f"{set_cell_size(stage, _STAGE_WIDTH)}{set_cell_size(activity, _ACTIVITY_WIDTH)}", description_width
    )
    bar_width = max(0, width - cell_len(prefix + description + suffix) - 1)
    bar = _format_bar(percent, bar_width, ascii_only=ascii_only)
    separator = " " if bar_width else ""
    return set_cell_size(f"{prefix}{description}{separator}{bar}{suffix}", width)


def _format_bar(percent: float, width: int, *, ascii_only: bool) -> str:
    """Render a tqdm-style block bar."""
    if width <= 0:
        return ""
    subdivisions = 10 if ascii_only else 8
    units = int(width * subdivisions * percent / 100)
    complete, partial = divmod(units, subdivisions)
    fractions = " 123456789" if ascii_only else " ▏▎▍▌▋▊▉"
    filled = "#" if ascii_only else "█"
    partial_cell = fractions[partial] if partial and complete < width else ""
    return f"{filled * complete}{partial_cell}".ljust(width)


def _format_elapsed(elapsed: float) -> str:
    """Format elapsed seconds like tqdm's compact timer."""
    minutes, seconds = divmod(max(0, int(elapsed)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02}:{seconds:02}" if hours else f"{minutes:02}:{seconds:02}"


class LoadingScreen:
    """Staged progress display that owns the console while a run starts up.

    While the screen is open, the startup path writes to standard output and
    error are spooled to a temporary file instead of the console, so they cannot
    break the progress bar. POSIX also captures native writes to the underlying
    file descriptors; Windows and macOS preserve their console descriptors and
    capture Python-level output. Alongside the bar the screen
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

    class _Display:
        """Rich renderable that reads the loading screen's current state."""

        def __init__(self, screen: LoadingScreen) -> None:
            self._screen = screen

        def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
            with self._screen._render_lock:
                renderable = self._screen._render(options.max_width)
            yield renderable

    def __init__(self, num_stages: int, *, enabled: bool | None = None, logo: bool = True) -> None:
        """Initialize the screen.

        Args:
            num_stages: Number of stages the progress bar counts up to.
            enabled: Whether to draw a live progress bar and spool startup
                output. Defaults to None, which enables both when standard
                output is a terminal.
            logo: Whether :meth:`summary` shows a responsive greeting beside
                the run summary. Defaults to True.
        """
        self._num_stages = num_stages
        self._logos = (LOGO, LOGO_WIDE) if logo else ()
        self._enabled = _console_is_interactive() if enabled is None else enabled
        self._console: IO[str] = sys.stdout
        self._ascii_only = not _supports_box_drawing(self._console)
        self._rich_console: Console | None = None
        self._live: Live | None = None
        self._display = self._Display(self)
        self._render_lock = threading.RLock()
        self._spool: IO[str] | None = None
        self._saved_fds: tuple[int, int] | None = None
        self._saved_streams: tuple[IO[str], IO[str]] | None = None
        self._started = 0.0
        self._index = 0
        self._stage = ""
        self._steps = 0
        self._activities: list[str] = []
        self._progress_percent = 0.0
        self._show_progress = False
        self._summary_title: str | None = None
        self._summary_fields: dict[str, str] = {}

    def __enter__(self) -> LoadingScreen:
        """Take over the console and start spooling startup output."""
        global _active_screen
        self._started = time.monotonic()
        if self._enabled:
            self._redirect()
            if self._enabled:
                self._rich_console = Console(
                    file=self._console,
                    color_system="truecolor",
                    force_terminal=True,
                    highlight=False,
                )
                self._live = Live(
                    self._display,
                    console=self._rich_console,
                    screen=True,
                    refresh_per_second=_REFRESH_PER_SECOND,
                    redirect_stdout=False,
                    redirect_stderr=False,
                )
                self._live.start(refresh=True)
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
        with self._render_lock:
            self._summary_title = title
            self._summary_fields = fields.copy()
        if self._enabled:
            self._refresh()
            return
        terminal_width = shutil.get_terminal_size(fallback=(_STANDARD_WIDTH, 24)).columns
        summary, _ = _format_header(title, fields, self._logos, terminal_width, ascii_only=self._ascii_only)
        self._write(f"\n{summary}\n\n")

    def stage(self, name: str) -> None:
        """Complete the running stage, if any, and start the stage named *name*.

        Args:
            name: Human-readable name of the stage that is about to run.
        """
        with self._render_lock:
            self._index += 1
            self._stage = name
            self._activities.clear()
            self._steps = 0
            self._set_progress(self._stage_progress())
            self._show_progress = True
        if self._enabled:
            self._refresh()
        else:
            self._write(f"[{self._index}/{self._num_stages}] {name}\n")

    def set_activity(self, activity: str | None) -> None:
        """Push *activity* as the step currently running, or pop it when None.

        Prefer :func:`report_activity`, which reaches the open screen from
        anywhere in the startup path without threading this object through.
        """
        if not self._enabled:
            return
        with self._render_lock:
            if activity is None:
                if self._activities:
                    self._activities.pop()
            else:
                self._activities.append(activity)
                self._steps += 1
                self._set_progress(self._stage_progress())
        self._refresh()

    def close(self) -> None:
        """Hand the console back to the rest of the run, dropping the spooled output."""
        self._shut_down(replay=False)

    def _shut_down(self, *, replay: bool) -> None:
        """Stop the clock, close the progress bar, and restore the console, once."""
        global _active_screen
        if _active_screen is self:
            _active_screen = None
        live: Live | None
        final_header = ""
        with self._render_lock:
            if not replay:
                self._set_progress(100)
            self._show_progress = False
            live = self._live
            if live is not None and self._rich_console is not None and self._summary_title is not None:
                final_header, _ = _format_header(
                    self._summary_title,
                    self._summary_fields,
                    self._logos,
                    self._rich_console.width,
                    ascii_only=self._ascii_only,
                )
            self._live = None
            self._rich_console = None
        try:
            if live is not None:
                live.stop()
            if final_header:
                self._write(f"\n{final_header}\n\n")
        finally:
            close_console = self._saved_fds is not None
            if self._saved_fds is not None or self._saved_streams is not None:
                hidden = self._restore()
                try:
                    if replay:
                        self._write(hidden)
                    else:
                        elapsed = time.monotonic() - self._started
                        lines = hidden.count("\n")
                        self._write(
                            f"  Ready in {elapsed:.1f}s "
                            f"({lines} lines of startup output hidden; use --info to show)\n\n"
                        )
                finally:
                    if close_console:
                        self._console.close()
                    self._console = sys.stdout

    def _stage_progress(self) -> float:
        """Bar position for the current stage and the steps reported within it, in percent."""
        span = 100 / self._num_stages
        filled = min(self._steps / _STEPS_PER_STAGE, _STEP_CEILING)
        return (self._index - 1) * span + span * filled

    def _set_progress(self, percent: float) -> None:
        """Move the bar to *percent*. The caller must hold the render lock."""
        self._progress_percent = percent

    def _refresh(self) -> None:
        """Refresh the live display when it owns the console."""
        # Rich holds its own lock while rendering the display, which then reads
        # state under ``_render_lock``. Call this only after releasing that lock.
        live = self._live
        if live is not None:
            live.refresh()

    def _render(self, terminal_width: int) -> RenderableType:
        """Render the current header and progress state for a terminal width."""
        display_width = _display_width(terminal_width)
        renderables: list[RenderableType] = []
        if self._summary_title is not None:
            header, _ = _format_header(
                self._summary_title,
                self._summary_fields,
                self._logos,
                terminal_width,
                ascii_only=self._ascii_only,
            )
            if header:
                renderables.extend((Text(""), Text(header, no_wrap=True, overflow="crop")))
        if self._show_progress:
            if renderables:
                renderables.append(Text(""))
            elapsed = time.monotonic() - self._started if self._started else 0.0
            activity = self._activities[-1] if self._activities else ""
            progress = _format_progress(
                self._stage,
                activity,
                self._progress_percent,
                elapsed,
                display_width,
                ascii_only=self._ascii_only,
            )
            renderables.append(Text(progress, no_wrap=True, overflow="crop"))
        return Constrain(Group(*renderables) if renderables else Text(""), display_width)

    def _write(self, text: str) -> None:
        """Write *text* straight to the console, bypassing the spool."""
        self._console.write(text)
        self._console.flush()

    def _redirect(self) -> None:
        """Redirect output to a spool without disrupting the live console."""
        if sys.platform in {"darwin", "win32"}:
            self._redirect_streams()
            return
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

    def _redirect_streams(self) -> None:
        """Spool Python output without replacing platform-specific console descriptors."""
        self._spool = tempfile.TemporaryFile(mode="w+", errors="replace")  # noqa: SIM115
        self._saved_streams = (sys.stdout, sys.stderr)
        sys.stdout = self._spool
        sys.stderr = self._spool

    def _restore(self) -> str:
        """Restore redirected output and return the spooled text."""
        if self._saved_fds is not None:
            sys.stdout.flush()
            sys.stderr.flush()
            saved_out, saved_err = self._saved_fds
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.close(saved_out)
            os.close(saved_err)
            self._saved_fds = None
        elif self._saved_streams is not None:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout, sys.stderr = self._saved_streams
            self._saved_streams = None
        else:
            raise RuntimeError("Loading screen output was not redirected.")
        self._spool.seek(0)
        spooled = self._spool.read()
        self._spool.close()
        self._spool = None
        return spooled


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
