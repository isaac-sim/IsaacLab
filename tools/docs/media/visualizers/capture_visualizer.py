# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture and generate every clip used by the visualizer docs pages.

Three independent pipelines, each producing a different set of clips:

* **hero**: five ``hero_*.mp4`` clips of the *same* AnymalD flat-terrain trajectory, one per
  visualizer backend (Kit, Newton GL, Newton RTX, Rerun, Viser), for the visualization overview
  page's hero grid.
* **showcase**: five ``showcase_*.mp4`` clips, one per visualizer backend, each running a
  *different* task's published pretrained checkpoint at that visualizer's normal wide
  interactive camera view.
* **streaming**: the streaming-camera-view demo clips (Kit + AnymalD, Newton GL + Galbot) used
  by the visualization overview and tiled-camera-view pages.

Run directly to regenerate clips; pass a pipeline name to run just one, or nothing to run all
three::

    uv run --no-project --with playwright --with pillow --with python-xlib \\
        python capture_visualizer.py [hero|showcase|streaming|all]

Each pipeline plays two roles, matching two different process tiers:

* **Config-time**: swaps in a capture-only environment config before Hydra resolves the task,
  via ``gym.spec(task).kwargs["env_cfg_entry_point"]``. This is the role when this module is
  imported via ``--external_callback capture_visualizer.configure_playback_<pipeline>`` inside
  a ``play.py`` subprocess, running in the full project ``uv`` environment.
* **Driver**: run this file directly to launch ``play.py`` per clip (or, for the streaming
  pipeline's Newton GL clip, launch the environment directly) and post-process each result with
  ffmpeg. Needs ``python-xlib`` (windowed HUD captures) and ``playwright``/``pillow``
  (Rerun/Viser browser captures) -- ephemeral dependencies, not part of this repo's own
  dependency tree.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import gymnasium as gym
import torch

from pxr import Gf, Sdf, UsdShade

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import VideoRecorderCfg
from isaaclab.managers import EventTermCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.meshes import MeshCuboidCfg
from isaaclab.sim.spawners.meshes.meshes import spawn_mesh_cuboid
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.core.lift.config.kuka_allegro.kuka_allegro_env_cfg import KukaAllegroLiftEnvCfg
from isaaclab_tasks.core.reach.config.franka.franka_reach_env_cfg import FrankaReachEnvCfg
from isaaclab_tasks.core.reorient.config.allegro_hand.allegro_hand_manager_env_cfg import AllegroHandManagerEnvCfg
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_direct_env_cfg import ShadowHandEnvCfg
from isaaclab_tasks.core.velocity.config.anymal_d.flat_env_cfg import AnymalDFlatEnvCfg
from isaaclab_tasks.core.velocity.config.anymal_d.rough_env_cfg import AnymalDRoughEnvCfg
from isaaclab_tasks.core.velocity.config.h1.rough_env_cfg import H1RoughEnvCfg
from isaaclab_tasks.utils import resolve_task_config

# ===========================================================================
# Shared helpers
# ===========================================================================
#
# Config-time (headless, argument_value, make_step_pacer) run *inside* play.py's process,
# imported via --external_callback, in the full project uv environment.
#
# Driver-time (record_windowed, launch_browser, detect_border_crop, find_window) run in the
# lightweight driver process that launches play.py as a subprocess and does the actual
# screen/browser recording.

_HEADLESS_FALLBACK_ARGS = [
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-webgl",
    "--ignore-gpu-blocklist",
    "--enable-unsafe-swiftshader",
]

_CROPDETECT_RE = re.compile(r"crop=(\d+):(\d+):(\d+):(\d+)")


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
    consumed by the hero pipeline's calibration pass to size its recording window.

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


# ===========================================================================
# Hero pipeline: five hero_*.mp4 clips of the same AnymalD trajectory, one per visualizer.
# ===========================================================================
#
# Kit/Newton GL/Newton RTX are captured via a step-count-bound VideoRecorderCfg (see
# _HERO_SOURCE_BY_VISUALIZER), so clip content is always exactly video_length simulated steps
# regardless of wall-clock time. Rerun/Viser are instead a wall-clock-bound screen recording of
# a fixed real-time duration, so a slow pipeline covers less simulated motion. _main_hero
# calibrates each browser visualizer's own natural real-time factor, sizes its recording window
# to cover the same simulated duration as the other 3, then speeds the result back up to a
# uniform output length via ffmpeg.
#
# Kit follows the robot via its streaming/tiled camera (streaming_cam_target_prim_path).
# Newton GL, Newton RTX, Rerun, and Viser instead drive their *interactive* camera every step
# via an EventTermCfg calling visualizer.set_camera_view(eye, target):
#
# * Newton GL's streaming camera never draws visualization markers into that sensor, so this
#   uses its interactive view instead, like Kit's.
# * Newton RTX's streaming camera targets Kit-hosted sensors, which throws in kitless
#   NewtonRTXVisualizer sessions, so its clip uses render_rgb_array() instead, with
#   focal_length increased from the 12mm default to match the streaming-camera clips' zoom.
# * Rerun and Viser both have real set_camera_view implementations, so their clips are a
#   headless-browser recording of their own native view, not the shared streaming camera sensor.
#
# _run_combined_capture launches ONE play.py process with HERO_VISUALIZER=combined, attaching
# Newton GL, Rerun, and Viser to the same simulation so all 3 render the exact same simulated
# step at any wall-clock moment, screen-recording both browser URLs concurrently with a fixed
# settle time (_main_hero converts this to an equivalent Newton GL step_offset). Needs
# playwright, so it's invoked as a subprocess of itself
# (``python capture_visualizer.py --combined-capture ...``).

_HERO_TASK = "Isaac-Velocity-Flat-AnymalD"

# Streaming/tiled follow-camera (Kit): same offset/target as the tiled-camera tutorial
# (scripts/tutorials/07_visualizers/run_tiled_camera_visualizer.py), zoomed in further.
_HERO_STREAMING_TARGET_PRIM = "/World/envs/*/Robot/base"
_HERO_STREAMING_EYE = (1.98, 1.98, 1.8)
# Per-step follow-camera (Kit windowed, Newton GL): same offset as the streaming clip.
_HERO_FOLLOW_EYE_OFFSET = _HERO_STREAMING_EYE
# Newton RTX/Rerun/Viser framed closer than Kit/Newton GL: their panels leave more empty
# space around the robot at the shared offset above. Newton RTX framed a bit further back
# than Rerun/Viser, whose tighter framing looked too tight for RTX at the same offset.
_HERO_RTX_FOLLOW_EYE_OFFSET = (1.4, 1.4, 1.28)
_HERO_BROWSER_FOLLOW_EYE_OFFSET = (1.0, 1.0, 0.91)
_HERO_VISER_FOLLOW_EYE_OFFSET = (0.8, 0.8, 0.73)
# Narrows Newton GL/RTX's FOV to match Kit's streaming-camera framing at this eye distance;
# no effect on Rerun (no FOV field) or Viser (already matches at the default 12mm).
_HERO_NARROW_FOCAL_LENGTH = 20.0

_HERO_WINDOW_WIDTH = 960
_HERO_WINDOW_HEIGHT = 600

# Deliberate stylistic override for the Newton backends: a saturated blue-green sky gradient,
# distinct from the neutral default Newton viewer palette.
_HERO_SKY_UPPER_COLOR = (0.05, 0.55, 0.55)
_HERO_SKY_LOWER_COLOR = (0.15, 0.80, 0.65)

# Kit/Newton GL/Newton RTX sources for the standard VideoRecorderCfg path (see
# _hero_configure_capture); Rerun/Viser are captured separately via headless-browser recording.
_HERO_SOURCE_BY_VISUALIZER = {
    "kit": "visualizer:kit:streaming_view",
    "newton_gl": "visualizer:newton_gl",
    "newton_rtx": "visualizer:newton_rtx",
}

# Driven by the per-step _follow_camera event rather than streaming_cam_target_prim_path.
_HERO_FOLLOW_CAM_VISUALIZERS = {"newton_gl", "newton_rtx", "rerun", "viser"}

# Double-exponential (Holt's linear trend) smoothing applied to the robot's raw per-step root
# position before it drives the follow camera (see _follow_camera): raw position has visible
# high-frequency foot-contact noise that otherwise transfers into camera jitter. A trend term
# is tracked alongside the smoothed position so the filter predicts ahead rather than lagging
# behind a steadily-moving target. BETA > ALPHA so the trend estimate doesn't lag behind
# ALPHA's own smoothing. Only applies to Newton GL/RTX/Rerun/Viser -- Kit follows via the
# native, zero-lag streaming_cam_target_prim_path.
_HERO_CAMERA_SMOOTHING_ALPHA = 0.025
_HERO_CAMERA_SMOOTHING_BETA = 0.15
_hero_smoothed_follow_target: list[tuple[float, float, float]] = []
_hero_smoothed_follow_trend: list[tuple[float, float, float]] = []

# Rerun re-sends its whole camera blueprint on every set_camera_view() call; at the task's
# full step rate this backs up its broadcast channel badly enough that no scene data reaches
# the client at all ("Sender has been blocked" in its log). Newton GL/RTX and Viser have no
# such cost, so _follow_camera runs every step and only throttles Rerun's own calls down to
# this stride.
_HERO_RERUN_UPDATE_STRIDE = 3
_hero_follow_camera_call_count: list[int] = [0]

# Forward speed stays fixed; only the turn rate differs from a straight walk (see
# _pin_velocity_command). Exactly one full turn over the clip's 10s duration (2*pi/10 rad/s)
# keeps every clip's start/end orientation aligned despite Newton GL/Rerun/Viser starting a
# few seconds into the shared trajectory that Kit/Newton RTX start at step 0.
_HERO_FIXED_VELOCITY_COMMAND = (1.0, 0.0, 0.0)
_HERO_CLIP_DURATION_S = 10.0
_HERO_ROTATE_ANG_VEL_Z = 2.0 * math.pi / _HERO_CLIP_DURATION_S


def _pin_velocity_command(env, env_ids) -> None:
    """Force a fixed forward speed + constant turn rate every step, instead of the task's
    default random resampling, so the clips show a continuously changing viewing angle.

    ``UniformVelocityCommand._resample_command`` samples via PyTorch's *global* RNG, not a
    seeded local generator, so Kit and the kitless Newton-family processes can draw a
    different random command and diverge from step 0. Overwriting the command tensor directly
    every step sidesteps the random draw regardless of any RNG-state difference.
    """
    del env_ids
    command_term = env.command_manager.get_term("base_velocity")
    # Both would otherwise overwrite this command again inside _update_command(): a heading
    # env recomputes ang_vel_z from a random heading target; a standing env zeroes it outright.
    command_term.is_heading_env[:] = False
    command_term.is_standing_env[:] = False

    command_term.vel_command_b[:, 0] = _HERO_FIXED_VELOCITY_COMMAND[0]
    command_term.vel_command_b[:, 1] = _HERO_FIXED_VELOCITY_COMMAND[1]
    command_term.vel_command_b[:, 2] = _HERO_ROTATE_ANG_VEL_Z


# Logs every step's wall-clock duration as "HERO_STEP_TIME <elapsed>" (consumed by
# _main_hero's calibration pass); see make_step_pacer's docstring for the full rationale.
_hero_step_pacer = make_step_pacer("HERO_TARGET_STEP_TIME", log_step_time=True)


def _follow_camera(env, env_ids) -> None:
    """Point every active visualizer's interactive camera at the first env's robot base.

    Fires every simulation step; only attached for visualizers in
    :data:`_HERO_FOLLOW_CAM_VISUALIZERS` -- Kit already follows via
    ``streaming_cam_target_prim_path``. Selects the eye offset per visualizer *instance* type
    since combined mode has more than one type active at once. The raw per-step root position
    is smoothed first (see :data:`_HERO_CAMERA_SMOOTHING_ALPHA`/:data:`_HERO_CAMERA_SMOOTHING_BETA`).
    """
    del env_ids
    robot = env.scene["robot"]
    pos = robot.data.root_pos_w[0]
    raw_target = (float(pos[0]), float(pos[1]), float(pos[2]))
    if _hero_smoothed_follow_target:
        prev_level = _hero_smoothed_follow_target[0]
        prev_trend = _hero_smoothed_follow_trend[0]
        predicted = tuple(prev_level[i] + prev_trend[i] for i in range(3))
        level = tuple(
            _HERO_CAMERA_SMOOTHING_ALPHA * raw_target[i] + (1.0 - _HERO_CAMERA_SMOOTHING_ALPHA) * predicted[i]
            for i in range(3)
        )
        trend = tuple(
            _HERO_CAMERA_SMOOTHING_BETA * (level[i] - prev_level[i])
            + (1.0 - _HERO_CAMERA_SMOOTHING_BETA) * prev_trend[i]
            for i in range(3)
        )
    else:
        level = raw_target
        trend = (0.0, 0.0, 0.0)
    _hero_smoothed_follow_target[:] = [level]
    _hero_smoothed_follow_trend[:] = [trend]
    target = level

    _hero_follow_camera_call_count[0] += 1
    is_rerun_update_step = _hero_follow_camera_call_count[0] % _HERO_RERUN_UPDATE_STRIDE == 0
    for visualizer in env.sim.visualizers:
        if not hasattr(visualizer, "set_camera_view"):
            continue
        is_rerun = type(visualizer).__name__ == "RerunVisualizer"
        if is_rerun and not is_rerun_update_step:
            continue
        visualizer_name = type(visualizer).__name__
        if is_rerun:
            offset = _HERO_BROWSER_FOLLOW_EYE_OFFSET
        elif visualizer_name == "ViserVisualizer":
            offset = _HERO_VISER_FOLLOW_EYE_OFFSET
        elif visualizer_name == "NewtonRTXVisualizer":
            offset = _HERO_RTX_FOLLOW_EYE_OFFSET
        else:
            offset = _HERO_FOLLOW_EYE_OFFSET
        eye = (target[0] + offset[0], target[1] + offset[1], target[2] + offset[2])
        visualizer.set_camera_view(eye, target)


def _hero_headless() -> bool:
    """Whether to run offscreen (default) or in a real window for HUD window-capture."""
    return headless("HERO_WINDOWED")


def _hero_make_kit_visualizer_cfg() -> VisualizerCfg:
    from isaaclab_visualizers.kit import KitVisualizerCfg

    if _hero_headless():
        # streaming_cam_target_prim_path follows the robot via a separate "Streaming View"
        # panel, captured through the standard VideoRecorderCfg path.
        return KitVisualizerCfg(
            headless=True,
            window_width=_HERO_WINDOW_WIDTH,
            window_height=_HERO_WINDOW_HEIGHT,
            streaming_view=True,
            streaming_envs=1,
            streaming_cam_target_prim_path=_HERO_STREAMING_TARGET_PRIM,
            streaming_cam_eye=_HERO_STREAMING_EYE,
            enable_markers=True,
        )
    # Windowed capture records the "Viewport" tab, not "Streaming View", so it follows via
    # _follow_camera instead, like the other _HERO_FOLLOW_CAM_VISUALIZERS.
    return KitVisualizerCfg(
        headless=False,
        window_width=_HERO_WINDOW_WIDTH,
        window_height=_HERO_WINDOW_HEIGHT,
        streaming_view=False,
        eye=_HERO_FOLLOW_EYE_OFFSET,
        lookat=(0.0, 0.0, 0.0),
        enable_markers=True,
    )


def _hero_make_newton_gl_visualizer_cfg() -> VisualizerCfg:
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

    # No streaming_view: markers aren't drawn into that camera sensor (see module docstring).
    # eye/lookat seed the initial pose before _follow_camera takes over.
    return NewtonGLVisualizerCfg(
        headless=_hero_headless(),
        window_width=_HERO_WINDOW_WIDTH,
        window_height=_HERO_WINDOW_HEIGHT,
        eye=_HERO_FOLLOW_EYE_OFFSET,
        lookat=(0.0, 0.0, 0.0),
        focal_length=_HERO_NARROW_FOCAL_LENGTH,
        sky_upper_color=_HERO_SKY_UPPER_COLOR,
        sky_lower_color=_HERO_SKY_LOWER_COLOR,
        enable_markers=True,
    )


def _hero_make_newton_rtx_visualizer_cfg() -> VisualizerCfg:
    from isaaclab_visualizers.newton import NewtonRTXVisualizerCfg

    # No streaming_view: unusable for kitless RTX (see module docstring). Markers are
    # unsupported on this backend regardless of enable_markers (hard-blocked for the kitless
    # viewer).
    return NewtonRTXVisualizerCfg(
        headless=_hero_headless(),
        window_width=_HERO_WINDOW_WIDTH,
        window_height=_HERO_WINDOW_HEIGHT,
        eye=_HERO_RTX_FOLLOW_EYE_OFFSET,
        lookat=(0.0, 0.0, 0.0),
        focal_length=_HERO_NARROW_FOCAL_LENGTH,
        rtx_environment="studio",
        sky_upper_color=_HERO_SKY_UPPER_COLOR,
        sky_lower_color=_HERO_SKY_LOWER_COLOR,
    )


def _hero_make_rerun_visualizer_cfg() -> VisualizerCfg:
    from isaaclab_visualizers.rerun import RerunVisualizerCfg

    # No streaming_view: captures its own native 3D view (see module docstring).
    return RerunVisualizerCfg(eye=_HERO_BROWSER_FOLLOW_EYE_OFFSET, lookat=(0.0, 0.0, 0.0), enable_markers=True)


def _hero_make_viser_visualizer_cfg() -> VisualizerCfg:
    from isaaclab_visualizers.viser import ViserVisualizerCfg

    # Same switch to native-view capture as Rerun; see _hero_make_rerun_visualizer_cfg().
    return ViserVisualizerCfg(eye=_HERO_VISER_FOLLOW_EYE_OFFSET, lookat=(0.0, 0.0, 0.0), enable_markers=True)


_HERO_VISUALIZER_BUILDERS = {
    "kit": _hero_make_kit_visualizer_cfg,
    "newton_gl": _hero_make_newton_gl_visualizer_cfg,
    "newton_rtx": _hero_make_newton_rtx_visualizer_cfg,
    "rerun": _hero_make_rerun_visualizer_cfg,
    "viser": _hero_make_viser_visualizer_cfg,
}

# Attached simultaneously (one physics process, one shared trajectory) when
# HERO_VISUALIZER=combined, avoiding the per-process browser-settle timing misalignment
# described in the module docstring. Kit and Newton RTX stay separate: they're already
# well-aligned via their own step-count-bound VideoRecorderCfg captures, and Kit needs a full
# Kit app process that can't coexist with Newton's kitless viewers.
_HERO_COMBINED_VISUALIZERS = ("newton_gl", "rerun", "viser")


@configclass
class AnymalDTileCaptureCfg(AnymalDFlatEnvCfg):
    """AnymalD flat-terrain configuration with a tile-capture visualizer and recorder."""

    def __post_init__(self):
        super().__post_init__()
        _hero_configure_capture(self)


def configure_playback_hero() -> list[str]:
    """Register the tile capture config and preserve Hydra arguments."""
    task = argument_value("--task")
    if task != _HERO_TASK:
        raise ValueError(
            f"capture_visualizer's hero pipeline is only configured for task {_HERO_TASK!r}, got {task!r}."
        )
    gym.spec(task).kwargs["env_cfg_entry_point"] = "capture_visualizer:AnymalDTileCaptureCfg"
    return sys.argv[1:]


def _hero_configure_capture(env_cfg: AnymalDTileCaptureCfg) -> None:
    """Attach the requested visualizer and video recorder to the AnymalD flat config."""
    # Read the visualizer selection from HERO_VISUALIZER, not --viz off sys.argv: by the time
    # __post_init__ runs, rsl_rl has already rewritten sys.argv and AppLauncher's parser has
    # consumed --viz.
    visualizer = os.environ.get("HERO_VISUALIZER")
    if visualizer != "combined" and visualizer not in _HERO_VISUALIZER_BUILDERS:
        raise ValueError(
            f"Set HERO_VISUALIZER=<{'|'.join(_HERO_VISUALIZER_BUILDERS)}|combined> to select the hero capture"
            f" visualizer, got {visualizer!r}."
        )
    if visualizer == "combined":
        env_cfg.sim.visualizer_cfgs = [_HERO_VISUALIZER_BUILDERS[name]() for name in _HERO_COMBINED_VISUALIZERS]
    else:
        env_cfg.sim.visualizer_cfgs = [_HERO_VISUALIZER_BUILDERS[visualizer]()]

    # reset_base's default mdp.reset_root_state_uniform samples through PyTorch's *global*
    # RNG, not a seeded local generator -- same anti-pattern as the velocity command (see
    # _pin_velocity_command's docstring). Zeroing the range pins every process to the exact
    # same default root pose regardless of --seed.
    env_cfg.events.reset_base.params["pose_range"] = {}
    env_cfg.events.reset_base.params["velocity_range"] = {}

    # Same anti-pattern, two more places: reset_robot_joints randomizes starting limb
    # posture/gait phase, and add_base_mass randomizes base mass. Both pinned to identity.
    env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    env_cfg.events.add_base_mass.params["mass_distribution_params"] = (1.0, 1.0)

    env_cfg.events.step_pacer = EventTermCfg(
        func=_hero_step_pacer, mode="interval", interval_range_s=(0.0, 0.0), is_global_time=True
    )
    # See _pin_velocity_command's docstring for why this is needed for cross-process
    # alignment, not just a nicer-looking constant-turn walk.
    env_cfg.events.pin_velocity_command = EventTermCfg(
        func=_pin_velocity_command, mode="interval", interval_range_s=(0.0, 0.0), is_global_time=True
    )

    if (
        visualizer in _HERO_FOLLOW_CAM_VISUALIZERS
        or visualizer == "combined"
        or (visualizer == "kit" and not _hero_headless())
    ):
        env_cfg.events.follow_camera = EventTermCfg(
            func=_follow_camera, mode="interval", interval_range_s=(0.0, 0.0), is_global_time=True
        )

    output_dir = os.environ.get("HERO_VIDEO_DIR")
    if output_dir:
        if visualizer == "combined":
            # Only Newton GL is step-count-recorded here; Rerun/Viser are screen-recorded by
            # _run_combined_capture from the same process.
            env_cfg.video_recorders = [
                VideoRecorderCfg(
                    source=_HERO_SOURCE_BY_VISUALIZER["newton_gl"],
                    output_dir=output_dir,
                    output_filename_prefix=os.environ.get("HERO_VIDEO_PREFIX", "clip"),
                    fps=50,
                    step_offset=int(os.environ.get("HERO_COMBINED_STEP_OFFSET", "0")),
                )
            ]
        else:
            if visualizer not in _HERO_SOURCE_BY_VISUALIZER:
                raise ValueError(
                    f"'{visualizer}' is captured via a headless-browser recording, not VideoRecorderCfg (see"
                    " _main_hero); unset HERO_VIDEO_DIR to launch it for that instead."
                )
            env_cfg.video_recorders = [
                VideoRecorderCfg(
                    source=_HERO_SOURCE_BY_VISUALIZER[visualizer],
                    output_dir=output_dir,
                    output_filename_prefix=os.environ.get("HERO_VIDEO_PREFIX", "clip"),
                    # Matches the task's control rate (decimation=4, sim.dt=0.005s -> 50 Hz).
                    fps=50,
                    # Skips ahead to the same trajectory point a browser-captured visualizer
                    # starts recording at, so both clip types stay aligned.
                    step_offset=int(os.environ.get("HERO_STEP_OFFSET", "0")),
                )
            ]


# ---------------------------------------------------------------------------
# Hero driver: combined Newton GL/Rerun/Viser capture (needs playwright)
# ---------------------------------------------------------------------------


def _hero_find_urls(log_path: Path, timeout_s: float) -> tuple[str, str]:
    """Poll ``log_path`` for both the Rerun (port 9090) and Viser (port 8080) viewer URLs."""
    deadline = time.time() + timeout_s
    rerun_pattern = re.compile(r"https?://(?:localhost|127\.0\.0\.1):9090\S*")
    viser_pattern = re.compile(r"https?://(?:localhost|127\.0\.0\.1):8080\S*")
    rerun_url = viser_url = None
    seen = 0
    while time.time() < deadline:
        if log_path.exists():
            text = log_path.read_text(errors="ignore")
            if len(text) > seen:
                for line in text[seen:].splitlines():
                    if rerun_url is None and (match := rerun_pattern.search(line)):
                        rerun_url = match.group(0)
                    if viser_url is None and (match := viser_pattern.search(line)):
                        viser_url = match.group(0)
                seen = len(text)
        if rerun_url and viser_url:
            return rerun_url, viser_url
        time.sleep(1.0)
    raise TimeoutError(
        f"Rerun/Viser URLs not both found in {log_path} within {timeout_s}s (rerun={rerun_url!r}, viser={viser_url!r})"
    )


def _run_combined_capture(args: argparse.Namespace) -> None:
    """Launch play.py with HERO_VISUALIZER=combined and screen-record Rerun/Viser concurrently.

    See the hero pipeline's section docstring for the full rationale.
    """
    from playwright.sync_api import sync_playwright

    if "DISPLAY" not in os.environ and not args.force_headless:
        os.environ["DISPLAY"] = ":0"

    work_dir = Path("/tmp/capture_tile_combined")
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = work_dir / "play.log"
    video_dir = work_dir / "video"
    video_dir.mkdir(exist_ok=True)
    for stale in video_dir.glob("*.webm"):
        stale.unlink()

    newton_gl_video_dir = work_dir / "newton_gl_video"
    cmd = [
        "uv",
        "run",
        "--frozen",
        "--extra",
        args.uv_extras,
        "python",
        "scripts/reinforcement_learning/play.py",
        "--rl_library",
        "rsl_rl",
        "--task",
        args.task,
        "--checkpoint",
        "pretrained",
        # Must list all 3 types env_cfg.sim.visualizer_cfgs attaches: SimulationContext
        # ._resolve_visualizer_cfgs() filters to only the CLI-requested types otherwise.
        "--num_envs",
        "1",
        "--viz",
        "newton_gl,rerun,viser",
        "--external_callback",
        "capture_visualizer.configure_playback_hero",
        "physics=newton_mjwarp",
    ]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    proc_env = {"PYTHONPATH": f"{args.repo_root}/tools/docs/media/visualizers", "HERO_VISUALIZER": "combined"}
    if args.out_newton_gl_mp4:
        # --video_length is play.py's *total* rollout length, not just the recorder's clip
        # length, so it must include the step offset too.
        cmd += ["--video", "--video_length", str(args.newton_gl_step_offset + args.newton_gl_video_length)]
        newton_gl_video_dir.mkdir(exist_ok=True)
        proc_env |= {
            "HERO_VIDEO_DIR": str(newton_gl_video_dir),
            "HERO_VIDEO_PREFIX": "newton_gl",
            "HERO_COMBINED_STEP_OFFSET": str(args.newton_gl_step_offset),
        }
    print("Launching:", " ".join(cmd))
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=args.repo_root,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env={**proc_env, **os.environ},
        )
    try:
        rerun_url, viser_url = _hero_find_urls(log_path, args.find_timeout_s)
        print("Found Rerun URL:", rerun_url)
        print("Found Viser URL:", viser_url)

        with sync_playwright() as p:
            browser = launch_browser(p, force_headless=args.force_headless)

            # One continuous connection per viewer: reconnecting forces a full scene re-sync.
            contexts = []
            pages = {}
            for name, url in (("rerun", rerun_url), ("viser", viser_url)):
                context = browser.new_context(
                    viewport={"width": 960, "height": 600},
                    record_video_dir=str(video_dir),
                    record_video_size={"width": 960, "height": 600},
                )
                page = context.new_page()
                page.goto(url, timeout=30000)
                contexts.append(context)
                pages[name] = page

            # Fixed settle time (not motion-detection): both scenes must start recording at
            # the same moment; _main_hero converts this to an equivalent Newton GL step_offset.
            time.sleep(args.settle_s)
            time.sleep(args.record_s)

            if args.out_newton_gl_mp4:
                # Keep both browser clients connected while Newton GL's VideoRecorderCfg
                # finishes writing (it needs a few more steps than the Rerun/Viser window
                # alone): closing the contexts first stalls the *shared* step loop, since
                # Rerun's server-side rr.log() calls block on backpressure once their only
                # client disconnects.
                deadline = time.time() + 90.0
                last_size = -1
                stable_checks = 0
                newton_gl_path = None
                while time.time() < deadline and stable_checks < 3:
                    candidates = sorted(newton_gl_video_dir.glob("newton_gl_*.mp4"))
                    if candidates:
                        newton_gl_path = candidates[-1]
                        size = newton_gl_path.stat().st_size
                        stable_checks = stable_checks + 1 if size == last_size else 0
                        last_size = size
                    time.sleep(2.0)
                if newton_gl_path is None:
                    raise RuntimeError(f"No Newton GL video written to {newton_gl_video_dir}")
                out_path = Path(args.out_newton_gl_mp4)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                newton_gl_path.replace(out_path)
                print("Wrote", out_path)

            # Capture each page's exact video path before closing (record_video_dir's on-disk
            # filenames aren't otherwise orderable back to rerun vs. viser).
            webm_paths = {name: page.video.path() for name, page in pages.items()}
            for context in contexts:
                context.close()
            browser.close()

        # mpdecimate drops the duplicate frames Playwright pads in whenever the browser's
        # render rate falls behind its capture rate, shrinking the post-decimation duration by
        # an amount not known in advance. So decimate first, measure the actual resulting
        # duration, then compute the exact speed-up to hit target_duration_s.
        target_duration_s = args.record_s / args.speed_factor
        for name, out_mp4 in (("rerun", args.out_rerun_mp4), ("viser", args.out_viser_mp4)):
            webm = webm_paths[name]
            out_path = Path(out_mp4)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Pass 1: trim [settle_s, settle_s + record_s) from the start (not the last
            # record_s seconds -- the page stays open through the Newton GL catch-up wait
            # above, so the raw recording's *end* no longer lines up with the intended
            # window) and decimate, with no speed change yet.
            intermediate = out_path.with_name(out_path.stem + "_predecimate" + out_path.suffix)
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-ss",
                    str(args.settle_s),
                    "-t",
                    str(args.record_s),
                    "-i",
                    str(webm),
                    "-vf",
                    "mpdecimate,setpts=N/FRAME_RATE/TB",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(intermediate),
                ],
                check=True,
            )
            actual_duration = float(
                subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(intermediate)],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            )

            # Pass 2: apply the exact speed-up needed to reach target_duration_s now that the
            # real post-decimation duration is known.
            out_speed = actual_duration / target_duration_s
            video_filters = [f"setpts=PTS/{out_speed}", "crop=iw:ih-20:0:10"]
            if name == "viser":
                # Viser's viewer has no directional/fill light and renders the floor dark;
                # raising ambient light only brightens the sky, not the floor, so this
                # brightens the video directly instead. A gamma curve (exponent <1) lifts dark
                # values without pushing already-bright pixels toward white the way a flat
                # offset did. Exponents computed by sampling mean floor-region RGB in Viser vs.
                # the equivalent Newton GL frame and solving for the exponent that moves 37.5%
                # of that gap. The curves= pass afterward deepens shadows the gamma lift alone
                # left looking gray on the robot's legs.
                video_filters.append(
                    "lutrgb=r='255*pow(val/255,0.545)':g='255*pow(val/255,0.544)':b='255*pow(val/255,0.549)',"
                    "curves=all='0/0 0.2/0.12 1/1'"
                )
            final_cmd = [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(intermediate),
                "-vf",
                ",".join(video_filters),
                "-c:v",
                "libx264",
            ]
            if name == "rerun":
                # crf 27 keeps file size down; left at the default for viser so the color
                # grading above isn't compounded with extra compression artifacts.
                final_cmd += ["-preset", "slow", "-crf", "27"]
            final_cmd += ["-pix_fmt", "yuv420p", str(out_path)]
            subprocess.run(final_cmd, check=True)
            intermediate.unlink(missing_ok=True)
            print(f"Wrote {out_path} (actual_duration={actual_duration:.3f}s out_speed={out_speed:.4f})")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _combined_capture_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_root")
    parser.add_argument("task")
    parser.add_argument("out_rerun_mp4")
    parser.add_argument("out_viser_mp4")
    parser.add_argument("--settle-s", type=float, default=15.0, help="Fixed wait for both scenes to sync.")
    parser.add_argument("--record-s", type=float, default=10.0, help="Recorded (real wall-clock) length.")
    parser.add_argument("--speed-factor", type=float, default=1.0, help="Speed up both clips by this factor.")
    parser.add_argument("--find-timeout-s", type=float, default=180.0)
    parser.add_argument("--uv-extras", default="isaacsim,rerun,viser,rsl-rl,ov,video")
    parser.add_argument("--force-headless", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="Environment seed forwarded to play.py.")
    parser.add_argument(
        "--out-newton-gl-mp4",
        help="Also record Newton GL from this same process (step-count-bound VideoRecorderCfg), so all 3 hero"
        " clips share one deterministic trajectory.",
    )
    parser.add_argument(
        "--newton-gl-video-length", type=int, default=500, help="Newton GL clip length in simulated steps."
    )
    parser.add_argument(
        "--newton-gl-step-offset",
        type=int,
        default=0,
        help="Steps to skip so Newton GL's clip starts at the same trajectory point as the Rerun/Viser"
        " recordings (which start after --settle-s of real time, not step 0).",
    )
    return parser


# ---------------------------------------------------------------------------
# Hero driver: top-level orchestration (generates all 5 hero_*.mp4 clips)
# ---------------------------------------------------------------------------

# 10s clip at this task's control rate: decimation=4, sim.dt=0.005s -> step_dt=0.02s (50 Hz).
# Kit and Newton RTX capture exactly this many steps.
_HERO_CAPTURE_STEPS = 500
# Newton GL/Rerun/Viser capture 10% more steps while _HERO_TARGET_SIM_SECONDS stays at 10 --
# the combined capture's speed_factor compresses that extra motion into the same ~10s output.
_HERO_COMBINED_CAPTURE_STEPS = round(_HERO_CAPTURE_STEPS * 1.1)
# Calibrated real-time throughput target for the combined (Newton GL + Rerun + Viser) process.
_HERO_TARGET_SIM_SECONDS = 10
# Guards against a runaway record_s if the combined process is catastrophically slow to connect.
_HERO_MAX_RECORD_S = 90
# Fixed real-time wait before Rerun/Viser start recording, converted to an equivalent Newton
# GL step_offset below.
_HERO_COMBINED_SETTLE_S = 15
_HERO_UV_EXTRAS = "isaacsim,rerun,viser,rsl-rl,ov,video"


def _hero_record_visualizer(
    script_dir: Path,
    repo_root: Path,
    output_dir: Path,
    work_dir: Path,
    seed: int,
    visualizer: str,
    steps: int,
    step_offset: int,
    out_filename: Path,
    output_speed_factor: float = 1.0,
) -> None:
    """Capture Kit or Newton RTX's hero clip via the standard step-count-bound VideoRecorderCfg path.

    Kit, Newton GL, and Newton RTX are all captured through this clean-render, no-HUD path
    (the showcase clips use windowed HUD capture instead, since a UI panel would distract from
    the hero tile's side-by-side comparison). step_offset (the same value
    _hero_calibrate_combined() derives for Newton GL/Rerun/Viser) makes Kit/Newton RTX skip
    ahead to the same trajectory point instead of starting at step 0 -- otherwise the two
    groups' first frames would show a large, obvious rotational phase jump given the constant
    turn rate this pipeline uses.
    """
    output = work_dir / visualizer
    cmd = [
        "uv",
        "run",
        "--frozen",
        "--extra",
        _HERO_UV_EXTRAS,
        "python",
        "scripts/reinforcement_learning/play.py",
        "--rl_library",
        "rsl_rl",
        "--task",
        _HERO_TASK,
        "--checkpoint",
        "pretrained",
        "--num_envs",
        "1",
        "--seed",
        str(seed),
        "--video",
        "--video_length",
        str(step_offset + steps),
        "--viz",
        visualizer,
        "--external_callback",
        "capture_visualizer.configure_playback_hero",
        "physics=newton_mjwarp",
    ]
    env = {
        "HERO_VIDEO_DIR": str(output),
        "HERO_VIDEO_PREFIX": visualizer,
        "HERO_VISUALIZER": visualizer,
        "HERO_STEP_OFFSET": str(step_offset),
        "PYTHONPATH": f"{script_dir}{(':' + os.environ['PYTHONPATH']) if 'PYTHONPATH' in os.environ else ''}",
        **os.environ,
    }
    print("Launching:", " ".join(cmd))
    subprocess.run(cmd, cwd=repo_root, env=env, check=True)

    raw = output / f"{visualizer}_0000.mp4"
    # Trims 10px off the top and bottom (Newton GL is handled separately below).
    video_filter = "crop=iw:ih-20:0:10"
    if output_speed_factor != 1.0:
        video_filter += f",setpts=PTS/{output_speed_factor}"
    # crf 24 keeps file size down without visible artifacts.
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(raw),
            "-filter:v",
            video_filter,
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "24",
            "-pix_fmt",
            "yuv420p",
            str(out_filename),
        ],
        check=True,
    )


def _hero_calibrate_combined(script_dir: Path, repo_root: Path, work_dir: Path, seed: int | None) -> float:
    """Run the combined capture briefly, unrecorded, to measure its natural real-time throughput.

    A short calibration window undercounts: Rerun's per-step camera-blueprint log can build up
    backpressure over tens of seconds that a brief run never encounters. Calibrating over a
    window close to the real production duration (settle + record) captures the same
    sustained-load behavior.
    """
    log = Path("/tmp/capture_tile_combined/play.log")
    log.unlink(missing_ok=True)
    combined_cli_args = [
        str(repo_root),
        _HERO_TASK,
        str(work_dir / "calibrate_rerun.mp4"),
        str(work_dir / "calibrate_viser.mp4"),
        "--settle-s",
        str(_HERO_COMBINED_SETTLE_S),
        "--record-s",
        "20",
        *(["--seed", str(seed)] if seed is not None else []),
    ]
    cmd = [
        "uv",
        "run",
        "--no-project",
        "--with",
        "playwright",
        "--with",
        "pillow",
        "python",
        str(script_dir / "capture_visualizer.py"),
        "--combined-capture",
        *combined_cli_args,
    ]
    with open(work_dir / "calibrate_combined.log", "w") as log_file:
        with contextlib.suppress(subprocess.TimeoutExpired):
            subprocess.run(cmd, timeout=90, stdout=log_file, stderr=subprocess.STDOUT)

    # Average step time, skipping the first 20 steps as warm-up.
    if not log.exists():
        return 0.0
    step_times = []
    for line in log.read_text(errors="ignore").splitlines():
        if line.startswith("HERO_STEP_TIME"):
            step_times.append(float(line.split()[1]))
    step_times = step_times[19:]  # tail -n +20 is 1-indexed and inclusive
    return sum(step_times) / len(step_times) if step_times else 0.0


def _main_hero(seed: int = 42) -> None:
    """Generate all 5 hero_*.mp4 clips."""
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[3]
    output_dir = repo_root / "docs/source/_static/visualizers"
    output_dir.mkdir(parents=True, exist_ok=True)

    work_dir = Path(tempfile.mkdtemp())
    try:
        os.chdir(repo_root)

        print("Calibrating combined Newton GL/Rerun/Viser real-time throughput...")
        step_time = _hero_calibrate_combined(script_dir, repo_root, work_dir, seed)
        # 20% safety margin: sustained throughput over the full production run can still run
        # a bit slower than the calibration.
        record_s = _HERO_COMBINED_CAPTURE_STEPS * step_time * 1.2 if step_time > 0 else _HERO_TARGET_SIM_SECONDS
        record_s = max(record_s, _HERO_TARGET_SIM_SECONDS)
        record_s = min(record_s, _HERO_MAX_RECORD_S)
        speed_factor = record_s / _HERO_TARGET_SIM_SECONDS
        step_offset = int(_HERO_COMBINED_SETTLE_S / step_time) if step_time > 0 else 0
        print(
            f"  combined: {step_time:.4f}s/step -> record-s={record_s:.1f}, "
            f"speed-factor={speed_factor:.3f}, step-offset={step_offset}"
        )

        # Same step_offset as the combined group so all 5 clips start at the same point.
        # Played back 10% slower than the other 3 -- see _hero_record_visualizer's docstring.
        for visualizer in ("kit", "newton_rtx"):
            _hero_record_visualizer(
                script_dir,
                repo_root,
                output_dir,
                work_dir,
                seed,
                visualizer,
                _HERO_CAPTURE_STEPS,
                step_offset,
                output_dir / f"hero_{visualizer}.mp4",
                output_speed_factor=0.9,
            )

        # Newton GL, Rerun, and Viser are captured together from one shared physics process.
        combined_cli_args = [
            str(repo_root),
            _HERO_TASK,
            str(output_dir / "hero_rerun.mp4"),
            str(output_dir / "hero_viser.mp4"),
            "--settle-s",
            str(_HERO_COMBINED_SETTLE_S),
            "--record-s",
            str(record_s),
            "--speed-factor",
            str(speed_factor),
            "--out-newton-gl-mp4",
            str(output_dir / "hero_newton_gl.mp4"),
            "--newton-gl-video-length",
            str(_HERO_COMBINED_CAPTURE_STEPS),
            "--newton-gl-step-offset",
            str(step_offset),
            *(["--seed", str(seed)] if seed is not None else []),
        ]
        cmd = [
            "uv",
            "run",
            "--no-project",
            "--with",
            "playwright",
            "--with",
            "pillow",
            "python",
            str(script_dir / "capture_visualizer.py"),
            "--combined-capture",
            *combined_cli_args,
        ]
        subprocess.run(cmd, check=True)

        # Unlike Rerun/Viser (already compressed to ~10s by _run_combined_capture's own
        # speed_factor), Newton GL's clip comes out of VideoRecorderCfg at native fps
        # (~11s), so it needs the same step-count ratio applied here to match.
        newton_gl_speedup = _HERO_COMBINED_CAPTURE_STEPS / _HERO_CAPTURE_STEPS
        newton_gl_raw = work_dir / "hero_newton_gl_raw.mp4"
        (output_dir / "hero_newton_gl.mp4").rename(newton_gl_raw)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(newton_gl_raw),
                "-filter:v",
                f"setpts=PTS/{newton_gl_speedup}",
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(output_dir / "hero_newton_gl.mp4"),
            ],
            check=True,
        )

        print()
        print(f"All 5 hero clips written to {output_dir}.")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ===========================================================================
# Showcase pipeline: five showcase_*.mp4 clips, one per visualizer, each a different task.
# ===========================================================================
#
# Newton GL (Isaac-Reorient-Cube-Allegro), Kit (Isaac-Reach-Franka), Newton RTX
# (Isaac-Velocity-Rough-H1, stairs-only terrain, see _SHOWCASE_STAIRS_TERRAINS_CFG), Rerun
# (Isaac-Reorient-Cube-Shadow-Direct, 64 envs), and Viser (Isaac-Lift-KukaAllegro, 128 envs).
#
# Rerun and Viser both render through Newton's streaming/camera-composited path, which has a
# known ground-material bug (see newton_adapter.py); _add_floor_overlay works around it.
# Neither backend implements render_rgb_array(), so their clips are captured via a real Chrome
# browser instead of VideoRecorderCfg -- see record_showcase_browser.
#
# The per-task capture overrides are injected via a capture-only *ShowcaseCfg subclass that
# swaps gym.spec(task).kwargs["env_cfg_entry_point"] before Hydra resolves the config, so the
# real task configs are never edited (see configure_playback_showcase).

# Restricts the H1 showcase terrain to the ascending pyramid-stairs sub-terrain only, so every
# spawned robot lands on a staircase (see _showcase_frame_wide_camera_from_env_origins).
_SHOWCASE_STAIRS_TERRAINS_CFG = dataclasses.replace(
    ROUGH_TERRAINS_CFG,
    sub_terrains={
        "pyramid_stairs": dataclasses.replace(ROUGH_TERRAINS_CFG.sub_terrains["pyramid_stairs"], proportion=1.0),
    },
)

_SHOWCASE_WIDE_LOOKAT = (0.0, 0.0, 0.0)
_SHOWCASE_REACH_EYE = (3.6, -3.6, 3.8)
_SHOWCASE_REACH_LOOKAT = (0.3, 0.0, -0.3)
# Kuka Allegro lift: same pitch/distance as Franka reach, viewed from the opposite horizontal
# side (x/y negated) so the two clips don't look like mirrored duplicates.
_SHOWCASE_KUKA_ALLEGRO_LIFT_EYE = (-_SHOWCASE_REACH_EYE[0], -_SHOWCASE_REACH_EYE[1], _SHOWCASE_REACH_EYE[2])
_SHOWCASE_KUKA_ALLEGRO_LIFT_LOOKAT = _SHOWCASE_REACH_LOOKAT
_SHOWCASE_KUKA_ALLEGRO_LIFT_ENV_SPACING = 2.4
# Pulled back so the top row of envs isn't cropped out of frame.
_SHOWCASE_NEWTON_GL_EYE = (2.1, -2.1, 2.3)
_SHOWCASE_NEWTON_GL_LOOKAT = (0.0, 0.0, -0.1)
# Zoomed in close: Shadow Hand's env_spacing is tiny, so a wide camera leaves the hands as an
# indistinct speck.
_SHOWCASE_SHADOW_HAND_EYE = (1.1, -1.1, 2.15)
_SHOWCASE_SHADOW_HAND_LOOKAT = (0.0, 0.0, -0.15)
_SHOWCASE_SHADOW_HAND_ENV_SPACING = 0.75
# Offset from the framed terrain-origin center (see _showcase_frame_wide_camera_from_env_origins).
_SHOWCASE_H1_STAIRS_EYE_OFFSET = (1.57, -2.97, 2.50)

_SHOWCASE_WINDOW_WIDTH = 960
_SHOWCASE_WINDOW_HEIGHT = 600
# Rounded to an even pixel count: libx264's yuv420p output requires even width/height.
_SHOWCASE_NEWTON_GL_WINDOW_WIDTH = round(_SHOWCASE_WINDOW_WIDTH * 1.3 * 1.15 / 2) * 2
_SHOWCASE_NEWTON_GL_WINDOW_HEIGHT = round(_SHOWCASE_WINDOW_HEIGHT * 1.3 * 1.15 / 2) * 2
# Newton RTX's path tracer benefits from extra resolution the rasterized backends don't need.
_SHOWCASE_H1_STAIRS_WINDOW_WIDTH = _SHOWCASE_WINDOW_WIDTH * 2
_SHOWCASE_H1_STAIRS_WINDOW_HEIGHT = _SHOWCASE_WINDOW_HEIGHT * 2


def _showcase_headless() -> bool:
    """Whether to run offscreen (default) or in a real window for HUD window-capture."""
    return headless("SHOWCASE_WINDOWED")


def _showcase_make_kit_visualizer_cfg() -> VisualizerCfg:
    from isaaclab_visualizers.kit import KitVisualizerCfg

    return KitVisualizerCfg(
        headless=_showcase_headless(),
        window_width=_SHOWCASE_WINDOW_WIDTH,
        window_height=_SHOWCASE_WINDOW_HEIGHT,
        eye=_SHOWCASE_REACH_EYE,
        lookat=_SHOWCASE_REACH_LOOKAT,
        enable_markers=True,
    )


def _showcase_make_newton_gl_visualizer_cfg() -> VisualizerCfg:
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

    # streaming_view defaults to True; this showcase only wants the interactive camera.
    return NewtonGLVisualizerCfg(
        headless=_showcase_headless(),
        window_width=_SHOWCASE_NEWTON_GL_WINDOW_WIDTH,
        window_height=_SHOWCASE_NEWTON_GL_WINDOW_HEIGHT,
        streaming_view=False,
        eye=_SHOWCASE_NEWTON_GL_EYE,
        lookat=_SHOWCASE_NEWTON_GL_LOOKAT,
        enable_markers=True,
    )


def _showcase_make_newton_rtx_visualizer_cfg() -> VisualizerCfg:
    from isaaclab_visualizers.newton import NewtonRTXVisualizerCfg

    # Markers are hard-blocked in NewtonRTXVisualizer for the kitless viewer.
    return NewtonRTXVisualizerCfg(
        headless=_showcase_headless(),
        window_width=_SHOWCASE_H1_STAIRS_WINDOW_WIDTH,
        window_height=_SHOWCASE_H1_STAIRS_WINDOW_HEIGHT,
        rtx_environment="default",
        eye=_SHOWCASE_H1_STAIRS_EYE_OFFSET,
        lookat=_SHOWCASE_WIDE_LOOKAT,
    )


def _showcase_make_rerun_visualizer_cfg() -> VisualizerCfg:
    from isaaclab_visualizers.rerun import RerunVisualizerCfg

    # Live plots disabled: not useful in a short clip, and adds overhead to an already
    # browser-bottlenecked capture.
    return RerunVisualizerCfg(
        eye=_SHOWCASE_SHADOW_HAND_EYE, lookat=_SHOWCASE_SHADOW_HAND_LOOKAT, enable_markers=True, enable_live_plots=False
    )


def _showcase_make_viser_visualizer_cfg() -> VisualizerCfg:
    from isaaclab_visualizers.viser import ViserVisualizerCfg

    return ViserVisualizerCfg(
        eye=_SHOWCASE_KUKA_ALLEGRO_LIFT_EYE, lookat=_SHOWCASE_KUKA_ALLEGRO_LIFT_LOOKAT, enable_markers=True
    )


_SHOWCASE_VISUALIZER_BUILDERS = {
    "kit": _showcase_make_kit_visualizer_cfg,
    "newton_gl": _showcase_make_newton_gl_visualizer_cfg,
    "newton_rtx": _showcase_make_newton_rtx_visualizer_cfg,
    "rerun": _showcase_make_rerun_visualizer_cfg,
    "viser": _showcase_make_viser_visualizer_cfg,
}

_SHOWCASE_SOURCE_BY_VISUALIZER = {
    "kit": "visualizer:kit",
    "newton_gl": "visualizer:newton_gl",
    "newton_rtx": "visualizer:newton_rtx",
}


# Light-blue overlay color for _add_floor_overlay; deliberately more saturated than Newton
# GL's own muted built-in ground color so it reads as clearly blue in the browser views.
_SHOWCASE_FLOOR_OVERLAY_COLOR = (0.45, 0.7, 1.0)

_SHOWCASE_FLOOR_OVERLAY_PRIM_PATH = "/World/FloorOverlay"


def _spawn_floor_overlay_cuboid(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn the floor-overlay cuboid as a mesh with a hand-authored bound material.

    ``visual_material``/``PreviewSurfaceCfg`` doesn't work here: its shader creation goes
    through a Kit command that silently no-ops outside a full Kit app (the kitless sessions
    these showcase captures run under). Newton's USD importer does fall back to a geometry
    prim's ``displayColor`` when no material is bound, but only if *some* material is bound
    and fails to resolve a color -- never for a prim with nothing bound at all (confirmed by
    testing: a ``displayColor``-only overlay still rendered flat gray). So this authors a real
    ``UsdShade.Material``/``UsdPreviewSurface`` by hand via plain ``pxr`` calls (no Kit
    command) and binds it directly. Also requires real ``UsdGeom.Mesh`` geometry
    (:class:`MeshCuboidCfg`) rather than a ``UsdGeom.Cube`` implicit shape -- Newton's material
    resolution didn't apply color to implicit shapes in testing.
    """
    prim = spawn_mesh_cuboid(prim_path, cfg, translation, orientation, **kwargs)
    stage = prim.GetStage()
    material_path = f"{prim_path}/FloorOverlayMaterial"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*_SHOWCASE_FLOOR_OVERLAY_COLOR))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    return prim


def _add_floor_overlay(env_cfg: object) -> None:
    """Add a thin light-blue cuboid just above the task's real ground plane.

    Works around the ground-material bug on Newton's streaming/RTX-camera-composited path
    (see ``newton_adapter.py``) by covering the existing floor with a plain surface Newton
    renders correctly regardless of the underlying material. Only meaningful for flat-ground
    tasks; the overlay is a plain scene asset so it works for both Manager- and Direct-style
    configs regardless of how the task's own ground plane was spawned.
    """
    env_cfg.scene.floor_overlay = AssetBaseCfg(
        prim_path=_SHOWCASE_FLOOR_OVERLAY_PRIM_PATH,
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.01)),
        spawn=MeshCuboidCfg(
            func=_spawn_floor_overlay_cuboid,
            # Matches GroundPlaneCfg's default 100x100m size to cover every cloned env.
            size=(100.0, 100.0, 0.01),
        ),
    )


def _showcase_resolved_sim_cfgs(env_cfg: object) -> list[SimulationCfg]:
    """Return every concrete :class:`SimulationCfg` on ``env_cfg.sim``.

    Some task configs (e.g. the cabinet tasks) hold a per-backend preset container on
    ``sim`` instead of a single resolved :class:`SimulationCfg` at this point in
    construction, since backend selection happens later via Hydra presets.
    """
    sim = env_cfg.sim
    if isinstance(sim, SimulationCfg):
        return [sim]
    return [value for field in dataclasses.fields(sim) if isinstance(value := getattr(sim, field.name), SimulationCfg)]


def _showcase_frame_wide_camera_from_env_origins(env, env_ids) -> None:
    """Point the interactive camera at the mean spawn position of the visible envs.

    H1's rough-terrain generator spreads sub-terrains across a large grid, so a fixed
    world-origin eye/lookat can easily miss every robot. Registered as an ``EventTermCfg``
    with ``mode="reset"`` so it only recomputes on env reset, not every step.
    """
    del env_ids
    center = env.scene.env_origins.mean(dim=0)
    eye = (
        float(center[0]) + _SHOWCASE_H1_STAIRS_EYE_OFFSET[0],
        float(center[1]) + _SHOWCASE_H1_STAIRS_EYE_OFFSET[1],
        float(center[2]) + _SHOWCASE_H1_STAIRS_EYE_OFFSET[2],
    )
    target = (float(center[0]), float(center[1]), float(center[2]))
    for visualizer in env.sim.visualizers:
        if hasattr(visualizer, "set_camera_view"):
            visualizer.set_camera_view(eye, target)


# Paces windowed HUD captures (see record_showcase_window) to a slower wall-clock rate so the
# renderer has more headroom per frame, reducing stutter; _main_showcase speeds the clip back
# up in post. See make_step_pacer's docstring for the full rationale.
_showcase_step_pacer = make_step_pacer("SHOWCASE_TARGET_STEP_TIME")


def _showcase_configure_capture(env_cfg: object, visualizer: str) -> None:
    """Attach the requested showcase visualizer and video recorder to ``env_cfg``."""
    for sim_cfg in _showcase_resolved_sim_cfgs(env_cfg):
        sim_cfg.visualizer_cfgs = [_SHOWCASE_VISUALIZER_BUILDERS[visualizer]()]

    if os.environ.get("SHOWCASE_WINDOWED") == "1":
        env_cfg.events.step_pacer = EventTermCfg(
            func=_showcase_step_pacer, mode="interval", interval_range_s=(0.0, 0.0), is_global_time=True
        )

    output_dir = os.environ.get("SHOWCASE_VIDEO_DIR")
    if output_dir:
        if visualizer not in _SHOWCASE_SOURCE_BY_VISUALIZER:
            raise ValueError(
                f"VideoRecorderCfg capture is not supported for the '{visualizer}' showcase visualizer (no"
                " render_rgb_array() implementation); see record_showcase_browser instead. Unset"
                " SHOWCASE_VIDEO_DIR to launch it for that."
            )
        env_cfg.video_recorders = [
            VideoRecorderCfg(
                source=_SHOWCASE_SOURCE_BY_VISUALIZER[visualizer],
                output_dir=output_dir,
                output_filename_prefix=os.environ.get("SHOWCASE_VIDEO_PREFIX", "clip"),
                video_length=int(os.environ.get("SHOWCASE_VIDEO_LENGTH", "150")),
                step_offset=int(os.environ.get("SHOWCASE_VIDEO_STEP_OFFSET", "0")),
            )
        ]


@configclass
class AllegroReorientShowcaseCfg(AllegroHandManagerEnvCfg):
    """Allegro cube reorientation configuration with the Newton GL showcase visualizer."""

    def __post_init__(self):
        super().__post_init__()
        _showcase_configure_capture(self, "newton_gl")
        # ReorientManagerEnvBaseCfg never overrides GroundPlaneCfg.color, which defaults to
        # black; fix it just for this capture rather than editing the real task.
        self.scene.ground.spawn.color = (1.0, 1.0, 1.0)
        _add_floor_overlay(self)


@configclass
class FrankaReachShowcaseCfg(FrankaReachEnvCfg):
    """Franka reach configuration with the Kit showcase visualizer."""

    def __post_init__(self):
        super().__post_init__()
        _showcase_configure_capture(self, "kit")
        # No _add_floor_overlay: Kit isn't affected by the floor bug, and the task's mounting
        # table sits at z=0, so an overlay here would hide it (confirmed by testing).
        self.scene.env_spacing = 2.0


@configclass
class H1RoughShowcaseCfg(H1RoughEnvCfg):
    """H1 stairs-only rough-terrain locomotion configuration with the Newton RTX showcase visualizer."""

    def __post_init__(self):
        super().__post_init__()
        _showcase_configure_capture(self, "newton_rtx")
        self.scene.terrain.terrain_generator = _SHOWCASE_STAIRS_TERRAINS_CFG
        self.events.frame_wide_camera = EventTermCfg(func=_showcase_frame_wide_camera_from_env_origins, mode="reset")


@configclass
class ShadowHandReorientShowcaseCfg(ShadowHandEnvCfg):
    """Shadow Hand cube reorientation configuration with the Rerun showcase visualizer."""

    def __post_init__(self):
        super().__post_init__()
        _showcase_configure_capture(self, "rerun")
        self.scene.env_spacing = _SHOWCASE_SHADOW_HAND_ENV_SPACING
        _add_floor_overlay(self)


@configclass
class KukaAllegroLiftShowcaseCfg(KukaAllegroLiftEnvCfg):
    """Kuka Allegro cube lift configuration with the Viser showcase visualizer."""

    def __post_init__(self):
        super().__post_init__()
        _showcase_configure_capture(self, "viser")
        self.scene.env_spacing = _SHOWCASE_KUKA_ALLEGRO_LIFT_ENV_SPACING
        _add_floor_overlay(self)


_SHOWCASE_TASK_CONFIGS = {
    "Isaac-Reorient-Cube-Allegro": "capture_visualizer:AllegroReorientShowcaseCfg",
    "Isaac-Reach-Franka": "capture_visualizer:FrankaReachShowcaseCfg",
    "Isaac-Velocity-Rough-H1": "capture_visualizer:H1RoughShowcaseCfg",
    "Isaac-Reorient-Cube-Shadow-Direct": "capture_visualizer:ShadowHandReorientShowcaseCfg",
    "Isaac-Lift-KukaAllegro": "capture_visualizer:KukaAllegroLiftShowcaseCfg",
}


def configure_playback_showcase() -> list[str]:
    """Register the task-specific showcase capture config and preserve Hydra arguments."""
    task = argument_value("--task")
    if task not in _SHOWCASE_TASK_CONFIGS:
        raise ValueError(f"No showcase capture configuration is registered for task {task!r}.")
    gym.spec(task).kwargs["env_cfg_entry_point"] = _SHOWCASE_TASK_CONFIGS[task]
    return sys.argv[1:]


# ---------------------------------------------------------------------------
# Showcase driver: launches play.py as a subprocess and records the result.
# ---------------------------------------------------------------------------

# Browser viewport / recorded video size for both Rerun and Viser.
_SHOWCASE_VIEWPORT_WIDTH = 1200
_SHOWCASE_VIEWPORT_HEIGHT = 750

# Pixel location of the "Software WebGL rendering detected" banner's close button
# (headless-Chrome-only artifact), scaled from its calibrated position at 960x600.
_SHOWCASE_WEBGL_BANNER_CLOSE_XY = (
    round(288 * _SHOWCASE_VIEWPORT_WIDTH / 960),
    round(81 * _SHOWCASE_VIEWPORT_HEIGHT / 600),
)


def _showcase_find_viewer_url(log_path: Path, timeout_s: float) -> str:
    """Poll ``log_path`` for the visualizer's own printed URL (always localhost/127.0.0.1).

    Excludes unrelated URLs earlier in the log (e.g. the checkpoint-download URL).
    """
    deadline = time.time() + timeout_s
    pattern = re.compile(r"https?://(?:localhost|127\.0\.0\.1)\S*")
    seen = 0
    while time.time() < deadline:
        if log_path.exists():
            text = log_path.read_text(errors="ignore")
            if len(text) > seen:
                for line in text[seen:].splitlines():
                    match = pattern.search(line)
                    if match:
                        return match.group(0)
                seen = len(text)
        time.sleep(1.0)
    raise TimeoutError(f"No viewer URL found in {log_path} within {timeout_s}s")


def _showcase_mean_abs_pixel_diff(a, b) -> float:
    """Mean absolute grayscale pixel difference between two images, downsampled for speed."""
    a = a.convert("L").resize((160, 100))
    b = b.convert("L").resize((160, 100))
    pixels_a, pixels_b = a.tobytes(), b.tobytes()
    return sum(abs(x - y) for x, y in zip(pixels_a, pixels_b)) / len(pixels_a)


def _showcase_dismiss_webgl_banner(page) -> None:
    with contextlib.suppress(Exception):
        page.mouse.click(*_SHOWCASE_WEBGL_BANNER_CLOSE_XY)


# Window titles for record_windowed's X11 WM_NAME match.
_SHOWCASE_WINDOW_TITLES = {
    "kit": "Isaac Lab",
    "newton_gl": "Newton Viewer",
    "newton_rtx": "Newton RTX Viewer",
}

# TODO: Kit's windowed capture (--viz kit) segfaults consistently ~5s into startup on
# Isaac-Reach-Franka in this environment (native crash in Kit's extension-loading code).
# Needs Kit-internal debugging -- until resolved, run with --skip kit.
# showcase_kit_franka_reach.mp4 is left as whatever was last successfully captured.


def record_showcase_window(
    repo_root: str,
    output_dir: Path,
    visualizer: str,
    task: str,
    num_envs: int,
    filename: str,
    record_s: float,
    output_speed_factor: float,
    settle_s: float | None,
    target_step_time: str,
    crf: int = 26,
) -> None:
    """Run Kit/Newton GL/Newton RTX non-headless and screen-record the real window, so the HUD
    is visible in the clip -- unlike a clean, HUD-less render_rgb_array() capture.
    """
    os.environ["SHOWCASE_TARGET_STEP_TIME"] = target_step_time
    kwargs = {}
    if settle_s is not None:
        kwargs["settle_s"] = settle_s
    record_windowed(
        repo_root=repo_root,
        viz=visualizer,
        task=task,
        num_envs=num_envs,
        external_callback="capture_visualizer.configure_playback_showcase",
        out_mp4=str(output_dir / filename),
        window_title=_SHOWCASE_WINDOW_TITLES[visualizer],
        windowed_env_var="SHOWCASE_WINDOWED",
        record_s=record_s,
        speed_factor=output_speed_factor,
        crf=crf,
        **kwargs,
    )


def record_showcase(
    repo_root: str,
    script_dir: str,
    output_dir: Path,
    work_dir: Path,
    uv_extras: str,
    visualizer: str,
    task: str,
    num_envs: int,
    video_length: int,
    step_offset: int,
    filename: str,
    hydra_overrides: list[str],
) -> None:
    """Newton RTX's window stays black under record_showcase_window (its path-traced output
    isn't blitted to the window in that flow -- confirmed by testing), so it uses the
    step-count-bound VideoRecorderCfg path instead.
    """
    output = work_dir / visualizer
    prefix = f"{visualizer}_showcase"

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
        "--video",
        "--viz",
        visualizer,
        "--external_callback",
        "capture_visualizer.configure_playback_showcase",
        *hydra_overrides,
    ]
    env = {
        "SHOWCASE_VIDEO_DIR": str(output),
        "SHOWCASE_VIDEO_PREFIX": prefix,
        "SHOWCASE_VIDEO_LENGTH": str(video_length),
        "SHOWCASE_VIDEO_STEP_OFFSET": str(step_offset),
        "PYTHONPATH": script_dir,
        **os.environ,
    }
    subprocess.run(cmd, cwd=repo_root, env=env, check=True)

    # VideoRecorderCfg's own writer defaults to a fairly compressed encode; re-encode at
    # crf 28 (confirmed by testing to show no visible artifacts here) for better quality.
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(output / f"{prefix}_0000.mp4"),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "28",
            "-pix_fmt",
            "yuv420p",
            str(output_dir / filename),
        ],
        check=True,
    )


def record_showcase_browser(
    repo_root: str,
    script_dir: str,
    output_dir: Path,
    uv_extras: str,
    visualizer: str,
    task: str,
    num_envs: int,
    checkpoint: str,
    filename: str,
    reset: bool,
    speed_factor: float,
    record_s: float = 10.0,
    max_wait_s: float = 60.0,
    motion_threshold: float = 0.5,
    force_headless: bool = False,
    extra_args: list[str] | None = None,
) -> None:
    """Browser-based capture for Rerun/Viser clips.

    Neither backend implements ``render_rgb_array()``, so ``VideoRecorderCfg`` can't capture
    them like Kit/Newton GL/Newton RTX. This launches ``play.py`` as a subprocess, connects a
    Chrome browser (via Playwright) to the printed viewer URL, waits for two consecutive
    frames to differ enough to confirm the policy is actively moving, then records ``record_s``
    seconds of clean footage.

    Prefers connecting to a real, GPU-bound X display over launching headless: headless Chrome
    falls back to a software (SwiftShader) WebGL rasterizer with much lower framerates
    (confirmed via ``WEBGL_debug_renderer_info``). Set ``DISPLAY`` to target a different X
    display; falls back to headless+SwiftShader automatically if none is reachable.

    Requires Google Chrome (or Chromium) and ffmpeg on PATH; run
    ``uv run --no-project --with playwright python -m playwright install ffmpeg`` once
    beforehand to fetch Playwright's own bundled ffmpeg.
    """
    from playwright.sync_api import sync_playwright

    extra_args = extra_args or []

    if "DISPLAY" not in os.environ and not force_headless:
        os.environ["DISPLAY"] = ":0"

    out_mp4 = output_dir / filename
    work_dir = Path("/tmp/capture_browser") / visualizer
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = work_dir / "play.log"
    video_dir = work_dir / "video"
    video_dir.mkdir(exist_ok=True)

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
        checkpoint,
        "--num_envs",
        str(num_envs),
        "--viz",
        visualizer,
        "--external_callback",
        "capture_visualizer.configure_playback_showcase",
        *extra_args,
    ]
    env_path = script_dir
    print("Launching:", " ".join(cmd))
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=repo_root,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env={"PYTHONPATH": env_path, **os.environ},
        )
    try:
        url = _showcase_find_viewer_url(log_path, timeout_s=180.0)
        print("Found viewer URL:", url)

        with sync_playwright() as p:
            browser = launch_browser(p, force_headless=force_headless)

            # Single continuous connection: reconnecting forces a full re-sync of everything
            # already logged, which never finishes for large scene geometry.
            context = browser.new_context(
                viewport={"width": _SHOWCASE_VIEWPORT_WIDTH, "height": _SHOWCASE_VIEWPORT_HEIGHT},
                record_video_dir=str(video_dir),
                record_video_size={"width": _SHOWCASE_VIEWPORT_WIDTH, "height": _SHOWCASE_VIEWPORT_HEIGHT},
            )
            page = context.new_page()
            page.goto(url, timeout=30000)

            page.wait_for_timeout(3000)
            _showcase_dismiss_webgl_banner(page)
            page.wait_for_timeout(8000)  # let the websocket scene sync start populating

            if reset:
                try:
                    page.get_by_text("Reset Episode", exact=True).first.click(timeout=5000, force=True)
                    print("Clicked 'Reset Episode'")
                    page.wait_for_timeout(3000)
                except Exception as exc:
                    print(f"Could not click 'Reset Episode' (expected on Rerun): {exc}")

            deadline = time.time() + max_wait_s
            prev_frame = _showcase_open_image(page.screenshot())
            motion_detected = False
            while time.time() < deadline:
                page.wait_for_timeout(1500)
                cur_frame = _showcase_open_image(page.screenshot())
                diff = _showcase_mean_abs_pixel_diff(prev_frame, cur_frame)
                print(f"frame diff: {diff:.2f}")
                if diff >= motion_threshold:
                    motion_detected = True
                    break
                prev_frame = cur_frame
            if not motion_detected:
                print(f"WARNING: no motion >= {motion_threshold} detected within {max_wait_s}s; recording anyway.")

            page.wait_for_timeout(int(record_s * 1000))
            context.close()
            browser.close()

        webm_files = sorted(video_dir.glob("*.webm"), key=lambda p: p.stat().st_mtime)
        if not webm_files:
            raise RuntimeError("No .webm recorded")
        webm = webm_files[-1]
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        # Keep the last record_s seconds (skips the page-reconnect settle at the start).
        # mpdecimate drops near-duplicate frames before speed-up, since Playwright's fixed-rate
        # recording repeats frames under render lag, which would otherwise freeze-then-jump
        # once sped up; setpts closes the resulting gaps.
        ffmpeg_cmd = ["ffmpeg", "-y", "-v", "error", "-sseof", f"-{record_s}", "-i", str(webm)]
        vf_filters = ["mpdecimate", "setpts=N/FRAME_RATE/TB"]
        if speed_factor != 1.0:
            vf_filters.append(f"setpts=PTS/{speed_factor}")
        if visualizer == "viser":
            # Newton's Viser viewer has no fill light; raising ambient light only brightens
            # the sky, not the floor/robot, so brighten the video instead.
            vf_filters.append("eq=brightness=0.08:saturation=1.05")
        ffmpeg_cmd += ["-vf", ",".join(vf_filters)]
        ffmpeg_cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_mp4)]
        subprocess.run(ffmpeg_cmd, check=True)
        print("Wrote", out_mp4)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _showcase_open_image(png_bytes: bytes):
    from PIL import Image

    return Image.open(io.BytesIO(png_bytes))


def _main_showcase(num_envs: int = 512, skip: list[str] | None = None) -> None:
    """Generate all 5 showcase_*.mp4 clips."""
    skip = skip or []

    script_dir = str(Path(__file__).resolve().parent)
    repo_root = str(Path(script_dir).parents[3])
    output_dir = Path(repo_root) / "docs/source/_static/visualizers"
    work_dir = Path("/tmp/capture_visualizer_showcase_work")
    work_dir.mkdir(parents=True, exist_ok=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    uv_extras = "isaacsim,rerun,viser,rsl-rl,ov,video"
    # H1's stairs-only terrain only needs enough envs to populate the visible frame.
    h1_num_envs = 160
    # Extra settle time before H1's headless VideoRecorderCfg capture starts, on top of the
    # fixed step count (see VideoRecorderCfg.step_offset).
    h1_extra_wait_s = 10

    # Both windowed rows share control rate decimation=4, sim.dt=1/120s -> 0.0333s/step.
    # pace_speed_factor slows simulation down (more render headroom, less GPU-contention
    # stutter) and speeds the recorded clip back up by the same factor in post (see
    # record_windowed's speed_factor / _step_pacer above); record_s is stretched by the same
    # factor so 10s of *simulated* motion is still covered.
    control_dt = 0.0333
    pace_speed_factor = 4.0
    target_step_time = f"{control_dt * pace_speed_factor:.4f}"
    window_record_s = round(10 * pace_speed_factor, 1)
    # Franka's clip played back faster than real motion at the shared compensation factor;
    # this output-only speed-up (applied only in the final re-encode) plays it back 20% slower.
    franka_output_speed_factor = round(pace_speed_factor * 0.8, 2)

    # Windowed clips (Rerun/Viser have no render_rgb_array() -- see browser rows below).
    # Target clip length: 10s at each task's fixed control rate (decimation * sim.dt).
    window_showcase_rows = [
        # (visualizer, task, num_envs, filename, settle_s, output_speed_factor, crf)
        (
            "newton_gl",
            "Isaac-Reorient-Cube-Allegro",
            num_envs,
            "showcase_newton_gl_allegro.mp4",
            None,
            pace_speed_factor,
            30,
        ),
        (
            "kit",
            "Isaac-Reach-Franka",
            num_envs,
            "showcase_kit_franka_reach.mp4",
            45.0,
            franka_output_speed_factor,
            26,
        ),
    ]

    # Newton RTX's window stays black under record_showcase_window (see its docstring), so it
    # uses the step-count-bound VideoRecorderCfg path. H1's control rate: decimation=4,
    # sim.dt=0.005s -> 0.02s/step (50 Hz) -> 500 steps for a 10s clip.
    h1_step_offset = round(h1_extra_wait_s / 0.02)
    headless_showcase_rows = [
        # (visualizer, task, num_envs, video_length, step_offset, filename, hydra_overrides)
        (
            "newton_rtx",
            "Isaac-Velocity-Rough-H1",
            h1_num_envs,
            500,
            h1_step_offset,
            "showcase_newton_rtx_h1_stairs.mp4",
            ["physics=newton_mjwarp"],
        ),
    ]

    # Captured via record_showcase_browser (row: visualizer, task, num_envs, checkpoint,
    # filename, reset, speed_factor). reset=True forces a fresh episode so the capture lands
    # on active motion. num_envs is kept well below the default 512 so motion stays visible.
    browser_rows = [
        # Shadow Hand has no sim-pacing mechanism like the windowed rows' step_pacer, and runs
        # faster than real time at this env count; speed_factor=0.8 slows the clip back down.
        (
            "rerun",
            "Isaac-Reorient-Cube-Shadow-Direct",
            64,
            "pretrained",
            "showcase_rerun_shadow_reorient.mp4",
            True,
            0.8,
        ),
        ("viser", "Isaac-Lift-KukaAllegro", 128, "pretrained", "showcase_viser_lift_kuka_allegro.mp4", True, 1.0),
    ]

    def should_skip(visualizer: str) -> bool:
        return visualizer in skip

    for visualizer, task, envs, filename, settle_s, output_speed_factor, crf in window_showcase_rows:
        if should_skip(visualizer):
            print(f"--- skipping {visualizer} (--skip) ---")
            continue
        record_showcase_window(
            repo_root,
            output_dir,
            visualizer,
            task,
            envs,
            filename,
            window_record_s,
            output_speed_factor,
            settle_s,
            target_step_time,
            crf,
        )

    for visualizer, task, envs, video_length, step_offset, filename, hydra_overrides in headless_showcase_rows:
        if should_skip(visualizer):
            print(f"--- skipping {visualizer} (--skip) ---")
            continue
        record_showcase(
            repo_root,
            script_dir,
            output_dir,
            work_dir,
            uv_extras,
            visualizer,
            task,
            envs,
            video_length,
            step_offset,
            filename,
            hydra_overrides,
        )

    for visualizer, task, envs, checkpoint, filename, reset, speed_factor in browser_rows:
        if should_skip(visualizer):
            print(f"--- skipping {visualizer} (--skip) ---")
            continue
        record_showcase_browser(
            repo_root,
            script_dir,
            output_dir,
            uv_extras,
            visualizer,
            task,
            envs,
            checkpoint,
            filename,
            reset,
            speed_factor,
        )

    print()
    print(f"Showcase clips written to {output_dir} (skipped: {skip or 'none'}).")


# ===========================================================================
# Streaming pipeline: the streaming-camera-view demo clips.
# ===========================================================================
#
# * Kit + Isaac-Velocity-Rough-AnymalD uses the pretrained checkpoint via play.py's
#   --external_callback (see configure_playback_streaming). Its streaming panel docks as a
#   second tab in the *same* dock group as the interactive viewport, not side by side, so this
#   captures two separate clips (toggling STREAMING_KIT_SHOW_PANEL).
# * Newton GL + the Galbot cube-stacking task has no pretrained checkpoint, so it keeps the
#   tutorial script's random-action setup and is launched directly (see
#   _streaming_newton_gl_worker_main), never through play.py. Its streaming panel is an ImGui
#   sidebar combo that starts hidden; this sets image_logger._selected directly between two
#   back-to-back recordings in the same process/window.

# ---------------------------------------------------------------------------
# Streaming: Kit + AnymalD (config-time code, imported via play.py's --external_callback)
# ---------------------------------------------------------------------------

_STREAMING_KIT_TASK = "Isaac-Velocity-Rough-AnymalD"

# Auto-follow camera path (AnymalD has no ego_cam scene camera), matching
# run_tiled_camera_visualizer.py's _make_kit_visualizer_cfg.
_STREAMING_ENVS = 36
_STREAMING_CAM_EYE = (3.0, 3.0, 3.0)
_STREAMING_CAM_TARGET_PRIM_PATH = "/World/envs/*/Robot/base"

# Pulled back to fit the 128-env clone grid (~28x25m at 2.5m env_spacing) in frame.
_STREAMING_INTERACTIVE_EYE = (18.0, -18.0, 14.0)
_STREAMING_INTERACTIVE_LOOKAT = (0.0, 0.0, 0.0)

# Streaming panel size (see KitVisualizer._setup_camera_image_window).
_STREAMING_WINDOW_WIDTH = 960
_STREAMING_WINDOW_HEIGHT = 600

# Orchestration constants shared by both Kit calls in _main_streaming().
_STREAMING_KIT_NUM_ENVS = 128
_STREAMING_KIT_WINDOW_TITLE = "Isaac Lab"


def _streaming_headless() -> bool:
    """Whether to run offscreen (default) or in a real window for HUD window-capture."""
    return headless("STREAMING_WINDOWED")


def _streaming_make_kit_visualizer_cfg() -> VisualizerCfg:
    from isaaclab_visualizers.kit import KitVisualizerCfg

    return KitVisualizerCfg(
        headless=_streaming_headless(),
        window_width=_STREAMING_WINDOW_WIDTH,
        window_height=_STREAMING_WINDOW_HEIGHT,
        eye=_STREAMING_INTERACTIVE_EYE,
        lookat=_STREAMING_INTERACTIVE_LOOKAT,
        # Toggled per capture run (see _main_streaming()) -- panel docks as a second tab, not
        # side by side.
        streaming_view=os.environ.get("STREAMING_KIT_SHOW_PANEL") == "1",
        streaming_envs=_STREAMING_ENVS,
        streaming_cam_eye=_STREAMING_CAM_EYE,
        streaming_cam_target_prim_path=_STREAMING_CAM_TARGET_PRIM_PATH,
        enable_markers=False,
    )


_streaming_step_pacer = make_step_pacer("STREAMING_TARGET_STEP_TIME")


def _focus_streaming_tab(env, env_ids) -> None:
    """Select the "Streaming View" tab instead of leaving "Viewport" active.

    Docking a second tab does not make it active, and the dock itself happens asynchronously
    over several app updates (confirmed by testing), so this runs every step until it wins.
    """
    del env_ids
    for visualizer in env.sim.visualizers:
        window = getattr(visualizer, "_camera_image_window", None)
        if window is not None:
            window.focus()


@configclass
class AnymalDStreamingShowcaseCfg(AnymalDRoughEnvCfg):
    """AnymalD rough-terrain locomotion configuration with the Kit streaming-view demo visualizer."""

    def __post_init__(self):
        super().__post_init__()
        self.sim.visualizer_cfgs = [_streaming_make_kit_visualizer_cfg()]
        if os.environ.get("STREAMING_WINDOWED") == "1":
            self.events.step_pacer = EventTermCfg(
                func=_streaming_step_pacer, mode="interval", interval_range_s=(0.0, 0.0), is_global_time=True
            )
        if os.environ.get("STREAMING_KIT_SHOW_PANEL") == "1":
            self.events.focus_streaming_tab = EventTermCfg(
                func=_focus_streaming_tab, mode="interval", interval_range_s=(0.0, 0.0), is_global_time=True
            )


_STREAMING_TASK_CONFIGS = {_STREAMING_KIT_TASK: "capture_visualizer:AnymalDStreamingShowcaseCfg"}


def configure_playback_streaming() -> list[str]:
    """Register the streaming-view capture config and preserve Hydra arguments."""
    task = argument_value("--task")
    if task not in _STREAMING_TASK_CONFIGS:
        raise ValueError(f"No streaming-view capture configuration is registered for task {task!r}.")
    gym.spec(task).kwargs["env_cfg_entry_point"] = _STREAMING_TASK_CONFIGS[task]
    return sys.argv[1:]


# ---------------------------------------------------------------------------
# Streaming: Newton GL + Galbot (standalone worker -- never goes through play.py)
# ---------------------------------------------------------------------------

_STREAMING_NEWTON_GL_TASK = "IsaacContrib-Stack-Cube-Galbot-Left-Arm-Gripper-Visuomotor"
_STREAMING_NEWTON_GL_NUM_ENVS = 25
_STREAMING_NEWTON_GL_STREAMING_ENVS = 12
_STREAMING_NEWTON_GL_PANEL_KEY = "Streaming View"
_STREAMING_NEWTON_GL_WINDOW_TITLE = "Newton Viewer"
# The Galbot task's config module imports isaaclab_teleop (stack_joint_pos_env_cfg.py); "mimic"
# is the lightest extra providing it without the heavier full "teleop" extra's deps.
_STREAMING_NEWTON_GL_UV_EXTRAS = "isaacsim,rerun,viser,rsl-rl,ov,video,mimic"


def _streaming_resolve_env_regex_path(prim_path: str) -> str:
    """Resolve scene config env namespace macros to the cloned-env regex."""
    return prim_path.format(ENV_REGEX_NS="/World/envs/env_.*")


def _streaming_make_newton_gl_visualizer_cfg(env_cfg):
    """Matches run_tiled_camera_visualizer.py's _make_newton_visualizer_cfg for this task."""
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

    cfg = NewtonGLVisualizerCfg()
    cfg.streaming_view = True
    cfg.streaming_envs = _STREAMING_NEWTON_GL_STREAMING_ENVS

    ego_cam_cfg = getattr(env_cfg.scene, "ego_cam", None)
    if ego_cam_cfg is not None:
        cfg.streaming_sensor_prim_path = _streaming_resolve_env_regex_path(ego_cam_cfg.prim_path)
    else:
        cfg.streaming_cam_eye = (3.0, 3.0, 3.0)
        cfg.streaming_cam_target_prim_path = "/World/envs/*/Robot/base"
    return cfg


def _streaming_step_for(env, seconds: float) -> None:
    """Step the env with random actions until ``seconds`` of wall-clock time has passed."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        with torch.inference_mode():
            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            env.step(actions)


def _streaming_force_large_streaming_panel(visualizer, width_px: float, height_px: float) -> None:
    """Force the streaming panel to a large fixed on-screen size every frame.

    Newton's own auto-fit sizing comes out small when the panel is opened by setting
    ``image_logger._selected`` directly instead of a real UI combo click (confirmed by
    testing). Overrides ``image_logger.draw`` to set a fixed size *and* a recentered position
    (``Cond_.always``) every frame, delegating to the previously-installed draw function.
    Position must be forced too, or the panel stays anchored to the small auto-fit corner.
    """
    image_logger = visualizer._viewer._image_logger
    orig_draw = image_logger.draw
    imgui = visualizer._viewer.ui.imgui
    sidebar_w = float(getattr(image_logger, "_sidebar_width_px", 0.0))

    def _draw_forced():
        vp = imgui.get_main_viewport()
        avail_w = max(width_px, vp.work_size.x - sidebar_w)
        avail_h = max(height_px, vp.work_size.y)
        x = sidebar_w + max(0.0, (avail_w - width_px) / 2.0)
        y = max(0.0, (avail_h - height_px) / 2.0)
        imgui.set_next_window_pos(imgui.ImVec2(float(x), float(y)), imgui.Cond_.always)
        imgui.set_next_window_size(imgui.ImVec2(float(width_px), float(height_px)), imgui.Cond_.always)
        return orig_draw()

    image_logger.draw = _draw_forced


def _streaming_finalize_newton_gl_clip(webm: Path, out_mp4: Path, speed_factor: float = 1.0) -> None:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(webm)]
    video_filters = []
    border_crop = detect_border_crop(webm)
    if border_crop:
        video_filters.append(border_crop)
    if speed_factor != 1.0:
        video_filters.append(f"setpts=PTS/{speed_factor}")
    if video_filters:
        cmd += ["-vf", ",".join(video_filters)]
    cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", "24", "-pix_fmt", "yuv420p", str(out_mp4)]
    subprocess.run(cmd, check=True)


def _streaming_record_newton_gl_segment(env, window_id: int, out_webm: Path, seconds: float) -> None:
    """Start the x11grab recording and step the env for the same real-time duration."""
    display_env = os.environ["DISPLAY"]
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
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
            str(seconds),
            "-i",
            display_env,
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
            str(out_webm),
        ]
    )
    # Overshoot slightly so ffmpeg's own -t bound determines the cutoff.
    _streaming_step_for(env, seconds + 1.0)
    returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(f"ffmpeg x11grab recording to {out_webm} exited with code {returncode}.")


def _streaming_newton_gl_worker_main(argv: list[str]) -> None:
    """Standalone launcher + window capture for the Newton GL + Galbot streaming-view demo.

    Runs in its own process (spawned by :func:`_main_streaming`), never through play.py.
    """
    parser = argparse.ArgumentParser(description=_streaming_newton_gl_worker_main.__doc__)
    parser.add_argument("out_interactive_mp4")
    parser.add_argument("out_tiled_mp4")
    parser.add_argument("--record-s", type=float, default=10.0, help="Recorded (real wall-clock) length per clip.")
    parser.add_argument("--settle-s", type=float, default=15.0, help="Wait after the window appears before recording.")
    parser.add_argument(
        "--window-title", default=_STREAMING_NEWTON_GL_WINDOW_TITLE, help="Substring to match against WM_NAME."
    )
    parser.add_argument("--find-timeout-s", type=float, default=60.0)
    parser.add_argument(
        "--streaming-panel-width", type=float, default=1400.0, help="Forced on-screen streaming panel width [px]."
    )
    parser.add_argument(
        "--streaming-panel-height", type=float, default=900.0, help="Forced on-screen streaming panel height [px]."
    )
    parser.add_argument(
        "--speed-factor",
        type=float,
        default=1.0,
        help="Speed up the output clips by this factor (baked into the video, not CSS/JS playback rate).",
    )
    add_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args(argv)
    args_cli.task = _STREAMING_NEWTON_GL_TASK
    sys.argv = [sys.argv[0]] + hydra_args

    if "DISPLAY" not in os.environ:
        os.environ["DISPLAY"] = ":0"

    env_cfg, _ = resolve_task_config(_STREAMING_NEWTON_GL_TASK, "")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = _STREAMING_NEWTON_GL_NUM_ENVS
        env_cfg.sim.visualizer_cfgs = [_streaming_make_newton_gl_visualizer_cfg(env_cfg)]

        env = gym.make(_STREAMING_NEWTON_GL_TASK, cfg=env_cfg)
        print("env created, resetting...", flush=True)
        env.reset()
        print("env reset complete", flush=True)

        from Xlib import display

        disp = display.Display()
        window_id = find_window(disp, args_cli.window_title, args_cli.find_timeout_s)
        print(f"Found window {window_id:#x} titled like {args_cli.window_title!r}, settling...", flush=True)
        _streaming_step_for(env, args_cli.settle_s)
        print("settle complete", flush=True)
        window_id = find_window(disp, args_cli.window_title, args_cli.find_timeout_s)

        work_dir = Path("/tmp/capture_visualizer_streaming_newton_gl")
        work_dir.mkdir(parents=True, exist_ok=True)

        # Streaming panel left hidden for the interactive segment.
        interactive_webm = work_dir / "interactive.webm"
        print("recording interactive segment...", flush=True)
        _streaming_record_newton_gl_segment(env, window_id, interactive_webm, args_cli.record_s)
        print("interactive segment recorded", flush=True)

        # Force the streaming panel open for the tiled segment.
        sim = env.unwrapped.sim
        visualizer = sim.visualizers[0]
        image_logger = getattr(visualizer._viewer, "_image_logger", None)
        if image_logger is None or _STREAMING_NEWTON_GL_PANEL_KEY not in getattr(image_logger, "_images", {}):
            raise RuntimeError(
                "Streaming panel was not registered after the settle/interactive steps; "
                "increase --settle-s or --record-s."
            )
        image_logger._selected = _STREAMING_NEWTON_GL_PANEL_KEY
        _streaming_force_large_streaming_panel(
            visualizer, args_cli.streaming_panel_width, args_cli.streaming_panel_height
        )
        print("streaming panel forced open, recording tiled segment...", flush=True)

        tiled_webm = work_dir / "tiled.webm"
        _streaming_record_newton_gl_segment(env, window_id, tiled_webm, args_cli.record_s)
        print("tiled segment recorded", flush=True)

        # Finalize the mp4s *before* env.close(): Isaac Sim's app shutdown can terminate the
        # process before control returns here (confirmed by testing), so anything that must
        # land on disk has to happen while the app is still alive.
        _streaming_finalize_newton_gl_clip(interactive_webm, Path(args_cli.out_interactive_mp4), args_cli.speed_factor)
        _streaming_finalize_newton_gl_clip(tiled_webm, Path(args_cli.out_tiled_mp4), args_cli.speed_factor)
        print("Wrote", args_cli.out_interactive_mp4, "and", args_cli.out_tiled_mp4, flush=True)

        env.close()


# ---------------------------------------------------------------------------
# Streaming: top-level orchestration
# ---------------------------------------------------------------------------


def _main_streaming(skip: list[str] | None = None) -> None:
    """Regenerate every streaming-view clip.

    ``--with python-xlib`` alone covers the Kit portion (runs directly in this process); the
    Newton GL portion is spawned as its own subprocess with the extras it needs. Pass
    ``skip=["kit"]`` / ``skip=["newton_gl"]`` to skip a mode.
    """
    skip = set(skip or [])

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[3]
    output_dir = repo_root / "docs" / "source" / "_static" / "visualizers"
    output_dir.mkdir(parents=True, exist_ok=True)

    record_s = 10.0
    # Baked into the output clips, so a 10s capture plays back as a 5s clip.
    speed_factor = 2.0

    if "kit" not in skip:
        print("--- kit + Isaac-Velocity-Rough-AnymalD (streaming view) ---")

        # Two separate captures since the streaming panel docks as a second tab in the same
        # group as the viewport (see AnymalDStreamingShowcaseCfg), not side by side.
        os.environ["STREAMING_KIT_SHOW_PANEL"] = "0"
        record_windowed(
            repo_root=str(repo_root),
            viz="kit",
            task=_STREAMING_KIT_TASK,
            num_envs=_STREAMING_KIT_NUM_ENVS,
            external_callback="capture_visualizer.configure_playback_streaming",
            out_mp4=str(output_dir / "streaming_kit_anymal_interactive.mp4"),
            window_title=_STREAMING_KIT_WINDOW_TITLE,
            windowed_env_var="STREAMING_WINDOWED",
            record_s=record_s,
            settle_s=90.0,
            speed_factor=speed_factor,
            crf=24,
            hydra_overrides=["physics=newton_mjwarp"],
        )

        os.environ["STREAMING_KIT_SHOW_PANEL"] = "1"
        record_windowed(
            repo_root=str(repo_root),
            viz="kit",
            task=_STREAMING_KIT_TASK,
            num_envs=_STREAMING_KIT_NUM_ENVS,
            external_callback="capture_visualizer.configure_playback_streaming",
            out_mp4=str(output_dir / "streaming_kit_anymal_tiled.mp4"),
            window_title=_STREAMING_KIT_WINDOW_TITLE,
            windowed_env_var="STREAMING_WINDOWED",
            record_s=record_s,
            settle_s=90.0,
            speed_factor=speed_factor,
            crf=24,
            hydra_overrides=["physics=newton_mjwarp"],
        )
    else:
        print("--- skipping kit (--skip) ---")

    if "newton_gl" not in skip:
        print("--- newton_gl + Galbot cube stacking (streaming view) ---")
        interactive_mp4 = output_dir / "streaming_newton_galbot_interactive.mp4"
        tiled_mp4 = output_dir / "streaming_newton_galbot_tiled.mp4"
        subprocess.run(
            [
                "uv",
                "run",
                "--frozen",
                "--extra",
                _STREAMING_NEWTON_GL_UV_EXTRAS,
                "--with",
                "python-xlib",
                "python",
                str(script_dir / "capture_visualizer.py"),
                "--newton-gl-worker",
                str(interactive_mp4),
                str(tiled_mp4),
                "--record-s",
                str(record_s),
                "--settle-s",
                "60",
                "--speed-factor",
                str(speed_factor),
            ],
            cwd=repo_root,
            check=True,
        )
    else:
        print("--- skipping newton_gl (--skip) ---")

    print()
    print(f"Streaming-view clips written to {output_dir} (skipped: {', '.join(sorted(skip)) or 'none'}).")


# ===========================================================================
# CLI entry point
# ===========================================================================


def main() -> None:
    """Regenerate visualizer doc-media clips.

    Run directly with::

        uv run --no-project --with playwright --with pillow --with python-xlib \\
            python capture_visualizer.py [hero|showcase|streaming|all]

    With no pipeline given (or ``all``), runs hero, then showcase, then streaming.
    """
    parser = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="pipeline")

    hero_parser = subparsers.add_parser("hero", help="Regenerate the 5 hero_*.mp4 clips.")
    hero_parser.add_argument("--seed", type=int, default=42)

    showcase_parser = subparsers.add_parser("showcase", help="Regenerate the 5 showcase_*.mp4 clips.")
    showcase_parser.add_argument("--num-envs", type=int, default=512)
    showcase_parser.add_argument(
        "--skip", action="append", default=[], help="Visualizer to skip entirely (repeatable)."
    )

    streaming_parser = subparsers.add_parser("streaming", help="Regenerate the streaming-view clips.")
    streaming_parser.add_argument("--skip", action="append", default=[], choices=["kit", "newton_gl"])

    subparsers.add_parser("all", help="Regenerate every clip (default).")

    args = parser.parse_args()

    require_commands("uv", "ffmpeg")

    if args.pipeline == "hero":
        _main_hero(seed=args.seed)
    elif args.pipeline == "showcase":
        _main_showcase(num_envs=args.num_envs, skip=args.skip)
    elif args.pipeline == "streaming":
        _main_streaming(skip=args.skip)
    else:
        _main_hero()
        _main_showcase()
        _main_streaming()


if __name__ == "__main__":
    # Internal dispatch flags used only when this file re-invokes itself as a subprocess with a
    # different uv environment; not part of the public CLI.
    if len(sys.argv) > 1 and sys.argv[1] == "--combined-capture":
        _run_combined_capture(_combined_capture_argparser().parse_args(sys.argv[2:]))
    elif len(sys.argv) > 1 and sys.argv[1] == "--newton-gl-worker":
        try:
            _streaming_newton_gl_worker_main(sys.argv[2:])
        except BaseException:
            import traceback

            traceback.print_exc()
            sys.exit(1)
    else:
        main()
