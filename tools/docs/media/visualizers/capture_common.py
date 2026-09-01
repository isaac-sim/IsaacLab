# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for the visualizer doc-media capture scripts.

Used by :mod:`capture_hero`, :mod:`capture_showcase`, and :mod:`capture_streaming`. Only logic
shared across more than one of those three lives here -- pipeline-specific details stay local.

Two kinds of helpers, matching the two process tiers each script runs across:

* Config-time (:func:`headless`, :func:`argument_value`, :func:`make_step_pacer`) run *inside*
  ``play.py``'s process, imported via ``--external_callback``, in the full project ``uv``
  environment.
* Driver-time (:func:`record_windowed`, :func:`launch_browser`, :func:`detect_border_crop`,
  :func:`find_window`) run in the lightweight driver process that launches ``play.py`` as a
  subprocess and does the actual screen/browser recording -- a separate
  ``uv run --no-project --with ...`` environment for ephemeral deps not in this repo's own tree.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

_HEADLESS_FALLBACK_ARGS = [
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-webgl",
    "--ignore-gpu-blocklist",
    "--enable-unsafe-swiftshader",
]

_CROPDETECT_RE = re.compile(r"crop=(\d+):(\d+):(\d+):(\d+)")


# ---------------------------------------------------------------------------
# Config-time helpers (run inside play.py's process)
# ---------------------------------------------------------------------------


def headless(windowed_env_var: str) -> bool:
    """Whether to run offscreen (default) or in a real window for HUD window-capture.

    ``windowed_env_var`` (e.g. ``HERO_WINDOWED``) is truthy when the caller wants a real,
    visible window instead (see :func:`record_windowed`).
    """
    return os.environ.get(windowed_env_var) != "1"


def argument_value(name: str) -> str | None:
    """Return a command-line option value from either supported argparse form."""
    for index, argument in enumerate(sys.argv[1:]):
        if argument == name and index + 2 <= len(sys.argv) - 1:
            return sys.argv[index + 2]
        if argument.startswith(f"{name}="):
            return argument.split("=", 1)[1]
    return None


def require_commands(*commands: str) -> None:
    """Exit with an error if any of ``commands`` isn't on ``PATH`` (e.g. ``"uv"``, ``"ffmpeg"``)."""
    import shutil

    for command in commands:
        if shutil.which(command) is None:
            raise SystemExit(f"Error: {command} is required to generate visualizer media.")


def make_step_pacer(target_step_time_env_var: str, log_step_time: bool = False):
    """Return an ``EventTermCfg``-compatible step-pacer callback.

    Registered with ``mode="interval"``/``interval_range_s=(0, 0)`` so it fires every
    simulation step. If ``target_step_time_env_var`` is set, sleeps out each step to that
    duration (never speeds one up if it already took longer) -- useful when the renderer can't
    keep up with the sim's natural step rate and a wall-clock-bound browser/window capture
    would otherwise pad in duplicate frames (confirmed by testing, reads as jitter once sped
    up in post). Callers that pace a capture must also scale up its recording window to cover
    the same amount of simulated motion.

    When ``log_step_time`` is set, also prints ``HERO_STEP_TIME <elapsed>`` every step,
    consumed by :mod:`capture_hero`'s calibration pass to size its recording window.

    Each call returns an independently-stateful callback, so multiple pacers don't share state.
    """
    last_step_wall_time: list[float] = []

    def _step_pacer(env, env_ids) -> None:
        del env, env_ids
        now = time.monotonic()
        if last_step_wall_time:
            elapsed = now - last_step_wall_time[0]
            if log_step_time:
                print(f"HERO_STEP_TIME {elapsed:.4f}", flush=True)
            target = os.environ.get(target_step_time_env_var)
            if target:
                remaining = float(target) - elapsed
                if remaining > 0:
                    time.sleep(remaining)
        last_step_wall_time[:] = [time.monotonic()]

    return _step_pacer


# ---------------------------------------------------------------------------
# Driver-time helpers (run in the lightweight driver process)
# ---------------------------------------------------------------------------


def launch_browser(playwright, force_headless: bool = False):
    """Launch a real, GPU-accelerated Chrome against ``$DISPLAY`` when possible.

    Headless Chrome falls back to a software (SwiftShader) WebGL rasterizer, markedly slower
    for Rerun/Viser's real-time rendering (confirmed by testing). Falls back to
    headless+SwiftShader only when ``force_headless`` is set or no display can be opened.
    """
    if not force_headless:
        try:
            browser = playwright.chromium.launch(channel="chrome", headless=False)
            print(f"Connected to real X display {os.environ.get('DISPLAY')!r} (hardware WebGL).")
            return browser
        except Exception as exc:
            print(
                f"Could not launch against DISPLAY={os.environ.get('DISPLAY')!r} ({exc});"
                " falling back to headless+SwiftShader."
            )
    return playwright.chromium.launch(channel="chrome", headless=True, args=_HEADLESS_FALLBACK_ARGS)


def detect_border_crop(video_path: Path) -> str | None:
    """Detect and return an exact ``crop=W:H:X:Y`` filter string for ``video_path``, or ``None``.

    Runs ffmpeg's ``cropdetect`` over the whole clip instead of guessing a fixed pixel margin:
    x11grab's window-manager decoration shadow isn't reproducible across captures (0px to
    ~20px depending on edge), so a fixed guess either leaves a sliver or crops real content.
    Returns ``None`` if no crop was detected, so the caller can skip the filter entirely.
    """
    result = subprocess.run(
        ["ffmpeg", "-i", str(video_path), "-vf", "cropdetect=24:2:0", "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    matches = _CROPDETECT_RE.findall(result.stderr)
    if not matches:
        return None
    width, height, x, y = matches[-1]
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    src_width, src_height = (int(v) for v in probe.stdout.strip().split(","))
    if int(width) >= src_width and int(height) >= src_height:
        return None
    return f"crop={width}:{height}:{x}:{y}"


def _search_window(disp, title_substring: str) -> int | None:
    """Search the current window tree for a window whose name contains ``title_substring``."""
    from Xlib.error import XError

    def _search(win) -> int | None:
        try:
            name = win.get_wm_name()
        except XError:
            return None
        if name and title_substring in str(name):
            return win.id
        try:
            children = win.query_tree().children
        except XError:
            return None
        for child in children:
            found = _search(child)
            if found is not None:
                return found
        return None

    return _search(disp.screen().root)


def find_window(disp, title_substring: str, timeout_s: float) -> int:
    """Poll the X server for a top-level window whose name contains ``title_substring``."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        found = _search_window(disp, title_substring)
        if found is not None:
            return found
        time.sleep(1.0)
    raise TimeoutError(f"No window titled like {title_substring!r} found within {timeout_s}s")


def record_windowed(
    repo_root: str,
    viz: str,
    task: str,
    num_envs: int,
    external_callback: str,
    out_mp4: str,
    window_title: str,
    windowed_env_var: str,
    record_s: float = 10.0,
    settle_s: float = 5.0,
    speed_factor: float = 1.0,
    crf: int = 18,
    no_auto_crop: bool = False,
    find_timeout_s: float = 60.0,
    uv_extras: str = "isaacsim,rerun,viser,rsl-rl,ov,video",
    hydra_overrides: list[str] | None = None,
) -> None:
    """Launch ``play.py`` in a real window and record it with the on-screen HUD.

    Kit / Newton GL / Newton RTX's standard capture path (``VideoRecorderCfg`` reading
    ``render_rgb_array()``) never creates the ImGui/Kit-toolbar HUD in headless mode, so there's
    nothing for it to include. Rerun and Viser show their HUD naturally since their capture is
    already a screen recording of a browser page.

    This gets the same effect for Kit / Newton GL / Newton RTX: sets ``windowed_env_var=1`` so
    the calling module's cfg builder passes ``headless=False``, finds the resulting window via
    ``python-xlib`` (an ephemeral dep, not in this repo's own tree), and records it with
    ``ffmpeg -f x11grab -window_id ...``, which tracks the window regardless of how the window
    manager moves it.

    Requires the calling process to have been launched with ``python-xlib`` available.
    """
    from Xlib import display

    hydra_overrides = hydra_overrides or []

    if "DISPLAY" not in os.environ:
        os.environ["DISPLAY"] = ":0"

    out_mp4_path = Path(out_mp4)
    out_mp4_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path("/tmp/capture_window") / viz
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = work_dir / "play.log"

    cmd = [
        "uv",
        "run",
        "--frozen",
        "--extra",
        uv_extras,
        "python",
        "scripts/reinforcement_learning/play.py",
        "--rl_library",
        "rsl_rl",
        "--task",
        task,
        "--checkpoint",
        "pretrained",
        "--num_envs",
        str(num_envs),
        "--viz",
        viz,
        "--external_callback",
        external_callback,
        *hydra_overrides,
    ]
    env_path = f"{repo_root}/tools/docs/media/visualizers"
    print("Launching:", " ".join(cmd))
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=repo_root,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env={
                "PYTHONPATH": env_path,
                windowed_env_var: "1",
                # Suppresses Kit's blocking telemetry-consent modal on first non-headless
                # launch (omni.ui.internal_session_notification.OVERRIDE_ENVVAR).
                "OMNI_DEVONLY_ASSUME_INTERNAL_SESSION_CONSENT": "1",
                **os.environ,
            },
        )
    try:
        disp = display.Display()
        window_id = find_window(disp, window_title, find_timeout_s)
        print(f"Found window {window_id:#x} titled like {window_title!r}")

        # Keep re-polling through the settle period instead of a blind sleep: Kit can close and
        # reopen its window (e.g. a splash/consent dialog replaced by the real one) partway
        # through, leaving the ID above stale (confirmed by testing -- x11grab fails outright).
        # Track the latest window seen so a momentary disappearance doesn't fail the capture.
        settle_deadline = time.monotonic() + settle_s
        while time.monotonic() < settle_deadline:
            found = _search_window(disp, window_title)
            if found is not None:
                window_id = found
            time.sleep(1.0)

        # One more blocking search right before capture, in case of a mid-disappearance above.
        window_id = find_window(disp, window_title, find_timeout_s)
        print(f"Window {window_id:#x} confirmed after settle, starting capture")

        webm = work_dir / "capture.webm"
        ffmpeg_record_cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            # vsync vfr keeps each frame's real capture timestamp so render lag plays back as
            # smooth variable pacing rather than judder.
            "-fflags",
            "+genpts",
            "-vsync",
            "vfr",
            "-f",
            "x11grab",
            "-window_id",
            str(window_id),
            "-framerate",
            "60",
            "-t",
            str(record_s),
            "-i",
            os.environ["DISPLAY"],
            "-c:v",
            "libvpx",
            "-qmin",
            "0",
            "-qmax",
            "20",
            "-crf",
            "8",
            "-b:v",
            "12M",
            str(webm),
        ]
        subprocess.run(ffmpeg_record_cmd, check=True)

        final_cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(webm)]
        video_filters = []
        if not no_auto_crop:
            border_crop = detect_border_crop(webm)
            if border_crop:
                print(f"Detected WM decoration border, applying {border_crop}")
                video_filters.append(border_crop)
        if speed_factor != 1.0:
            video_filters.append(f"setpts=PTS/{speed_factor}")
        if video_filters:
            final_cmd += ["-vf", ",".join(video_filters)]
        final_cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p", str(out_mp4_path)]
        subprocess.run(final_cmd, check=True)
        print("Wrote", out_mp4_path)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
