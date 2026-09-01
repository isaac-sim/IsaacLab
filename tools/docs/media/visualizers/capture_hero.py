# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture + generate the visualizer overview hero clips.

Produces five short clips of the *same* AnymalD flat-terrain trajectory, one per visualizer
backend (Kit, Newton GL, Newton RTX, Rerun, Viser), all framed by the same follow-camera so
the clips can be compared side by side on the visualizer docs page. Uses
``Isaac-Velocity-Flat-AnymalD`` with a constant nonzero turn rate (see
:func:`_pin_velocity_command`) so the clips show the robot from a continuously changing range
of viewing angles.

This module plays two roles, matching two different process tiers:

* **Config-time**: swaps in the capture-only ``AnymalDTileCaptureCfg`` before Hydra resolves
  the task, via ``gym.spec(task).kwargs["env_cfg_entry_point"]``. This is the role when the
  module is imported via ``--external_callback capture_hero.configure_playback`` inside a
  ``play.py`` subprocess, running in the full project ``uv`` environment.
* **Driver**: run this file directly (``python capture_hero.py``) to regenerate all five
  ``hero_*.mp4`` clips end to end -- calibrates timing, launches ``play.py`` per visualizer,
  and post-processes each clip with ffmpeg. Requires ``uv run --no-project --with playwright
  --with pillow python capture_hero.py``. Kit and Newton RTX use a real window too
  (``HERO_WINDOWED``), but still capture through the standard ``VideoRecorderCfg`` path, not
  an X11 screen-grab.

Kit/Newton GL/Newton RTX are captured via a step-count-bound ``VideoRecorderCfg`` (see
:data:`_SOURCE_BY_VISUALIZER`), so clip content is always exactly ``video_length`` simulated
steps regardless of wall-clock time. Rerun/Viser are instead a wall-clock-bound screen
recording of a fixed real-time duration, so a slow pipeline covers less simulated motion.
:func:`main` calibrates each browser visualizer's own natural real-time factor, sizes its
recording window to cover the same simulated duration as the other 3, then speeds the result
back up to a uniform output length via ffmpeg.

Kit follows the robot via its streaming/tiled camera (``streaming_cam_target_prim_path``).
Newton GL, Newton RTX, Rerun, and Viser instead drive their *interactive* camera every step
via an ``EventTermCfg`` calling ``visualizer.set_camera_view(eye, target)``:

* Newton GL's streaming camera never draws visualization markers into that sensor, so this
  uses its interactive view instead, like Kit's.
* Newton RTX's streaming camera targets Kit-hosted sensors, which throws in kitless
  ``NewtonRTXVisualizer`` sessions, so its clip uses ``render_rgb_array()`` instead, with
  ``focal_length`` increased from the 12mm default to match the streaming-camera clips' zoom.
* Rerun and Viser both have real ``set_camera_view`` implementations, so their clips are a
  headless-browser recording of their own native view, not the shared streaming camera sensor.

:func:`_run_combined_capture` launches ONE ``play.py`` process with
``HERO_VISUALIZER=combined``, attaching Newton GL, Rerun, and Viser to the same simulation so
all 3 render the exact same simulated step at any wall-clock moment, screen-recording both
browser URLs concurrently with a fixed settle time (:func:`main` converts this to an
equivalent Newton GL ``step_offset``). Needs playwright, so it's invoked as a subprocess of
itself (``python capture_hero.py --combined-capture ...``).
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import capture_common
import gymnasium as gym

from isaaclab.envs import VideoRecorderCfg
from isaaclab.managers import EventTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.core.velocity.config.anymal_d.flat_env_cfg import AnymalDFlatEnvCfg

_TASK = "Isaac-Velocity-Flat-AnymalD"

# Streaming/tiled follow-camera (Kit): same offset/target as the tiled-camera tutorial
# (scripts/tutorials/07_visualizers/run_tiled_camera_visualizer.py), zoomed in further.
_STREAMING_TARGET_PRIM = "/World/envs/*/Robot/base"
_STREAMING_EYE = (1.98, 1.98, 1.8)
# Per-step follow-camera (Kit windowed, Newton GL): same offset as the streaming clip.
_FOLLOW_EYE_OFFSET = _STREAMING_EYE
# Newton RTX/Rerun/Viser framed closer than Kit/Newton GL: their panels leave more empty
# space around the robot at the shared offset above. Newton RTX framed a bit further back
# than Rerun/Viser, whose tighter framing looked too tight for RTX at the same offset.
_RTX_FOLLOW_EYE_OFFSET = (1.4, 1.4, 1.28)
_BROWSER_FOLLOW_EYE_OFFSET = (1.0, 1.0, 0.91)
_VISER_FOLLOW_EYE_OFFSET = (0.8, 0.8, 0.73)
# Narrows Newton GL/RTX's FOV to match Kit's streaming-camera framing at this eye distance;
# no effect on Rerun (no FOV field) or Viser (already matches at the default 12mm).
_NARROW_FOCAL_LENGTH = 20.0

_WINDOW_WIDTH = 960
_WINDOW_HEIGHT = 600

# Deliberate stylistic override for the Newton backends: a saturated blue-green sky gradient,
# distinct from the neutral default Newton viewer palette.
_SKY_UPPER_COLOR = (0.05, 0.55, 0.55)
_SKY_LOWER_COLOR = (0.15, 0.80, 0.65)

# Kit/Newton GL/Newton RTX sources for the standard VideoRecorderCfg path (see
# _configure_capture); Rerun/Viser are captured separately via headless-browser recording.
_SOURCE_BY_VISUALIZER = {
    "kit": "visualizer:kit:streaming_view",
    "newton_gl": "visualizer:newton_gl",
    "newton_rtx": "visualizer:newton_rtx",
}

# Driven by the per-step _follow_camera event rather than streaming_cam_target_prim_path.
_FOLLOW_CAM_VISUALIZERS = {"newton_gl", "newton_rtx", "rerun", "viser"}

# Double-exponential (Holt's linear trend) smoothing applied to the robot's raw per-step root
# position before it drives the follow camera (see _follow_camera): raw position has visible
# high-frequency foot-contact noise that otherwise transfers into camera jitter. A trend term
# is tracked alongside the smoothed position so the filter predicts ahead rather than lagging
# behind a steadily-moving target. BETA > ALPHA so the trend estimate doesn't lag behind
# ALPHA's own smoothing. Only applies to Newton GL/RTX/Rerun/Viser -- Kit follows via the
# native, zero-lag streaming_cam_target_prim_path.
_CAMERA_SMOOTHING_ALPHA = 0.025
_CAMERA_SMOOTHING_BETA = 0.15
_smoothed_follow_target: list[tuple[float, float, float]] = []
_smoothed_follow_trend: list[tuple[float, float, float]] = []

# Rerun re-sends its whole camera blueprint on every set_camera_view() call; at the task's
# full step rate this backs up its broadcast channel badly enough that no scene data reaches
# the client at all ("Sender has been blocked" in its log). Newton GL/RTX and Viser have no
# such cost, so _follow_camera runs every step and only throttles Rerun's own calls down to
# this stride.
_RERUN_UPDATE_STRIDE = 3
_follow_camera_call_count: list[int] = [0]

# Forward speed stays fixed; only the turn rate differs from a straight walk (see
# _pin_velocity_command). Exactly one full turn over the clip's 10s duration (2*pi/10 rad/s)
# keeps every clip's start/end orientation aligned despite Newton GL/Rerun/Viser starting a
# few seconds into the shared trajectory that Kit/Newton RTX start at step 0.
_FIXED_VELOCITY_COMMAND = (1.0, 0.0, 0.0)
_HERO_CLIP_DURATION_S = 10.0
_ROTATE_ANG_VEL_Z = 2.0 * math.pi / _HERO_CLIP_DURATION_S


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

    command_term.vel_command_b[:, 0] = _FIXED_VELOCITY_COMMAND[0]
    command_term.vel_command_b[:, 1] = _FIXED_VELOCITY_COMMAND[1]
    command_term.vel_command_b[:, 2] = _ROTATE_ANG_VEL_Z


# Logs every step's wall-clock duration as "HERO_STEP_TIME <elapsed>" (consumed by main()'s
# calibration pass); see capture_common.make_step_pacer's docstring for the full rationale.
_step_pacer = capture_common.make_step_pacer("HERO_TARGET_STEP_TIME", log_step_time=True)


def _follow_camera(env, env_ids) -> None:
    """Point every active visualizer's interactive camera at the first env's robot base.

    Fires every simulation step; only attached for visualizers in
    :data:`_FOLLOW_CAM_VISUALIZERS` -- Kit already follows via
    ``streaming_cam_target_prim_path``. Selects the eye offset per visualizer *instance* type
    since combined mode has more than one type active at once. The raw per-step root position
    is smoothed first (see :data:`_CAMERA_SMOOTHING_ALPHA`/:data:`_CAMERA_SMOOTHING_BETA`).
    """
    del env_ids
    robot = env.scene["robot"]
    pos = robot.data.root_pos_w[0]
    raw_target = (float(pos[0]), float(pos[1]), float(pos[2]))
    if _smoothed_follow_target:
        prev_level = _smoothed_follow_target[0]
        prev_trend = _smoothed_follow_trend[0]
        predicted = tuple(prev_level[i] + prev_trend[i] for i in range(3))
        level = tuple(
            _CAMERA_SMOOTHING_ALPHA * raw_target[i] + (1.0 - _CAMERA_SMOOTHING_ALPHA) * predicted[i] for i in range(3)
        )
        trend = tuple(
            _CAMERA_SMOOTHING_BETA * (level[i] - prev_level[i]) + (1.0 - _CAMERA_SMOOTHING_BETA) * prev_trend[i]
            for i in range(3)
        )
    else:
        level = raw_target
        trend = (0.0, 0.0, 0.0)
    _smoothed_follow_target[:] = [level]
    _smoothed_follow_trend[:] = [trend]
    target = level

    _follow_camera_call_count[0] += 1
    is_rerun_update_step = _follow_camera_call_count[0] % _RERUN_UPDATE_STRIDE == 0
    for visualizer in env.sim.visualizers:
        if not hasattr(visualizer, "set_camera_view"):
            continue
        is_rerun = type(visualizer).__name__ == "RerunVisualizer"
        if is_rerun and not is_rerun_update_step:
            continue
        visualizer_name = type(visualizer).__name__
        if is_rerun:
            offset = _BROWSER_FOLLOW_EYE_OFFSET
        elif visualizer_name == "ViserVisualizer":
            offset = _VISER_FOLLOW_EYE_OFFSET
        elif visualizer_name == "NewtonRTXVisualizer":
            offset = _RTX_FOLLOW_EYE_OFFSET
        else:
            offset = _FOLLOW_EYE_OFFSET
        eye = (target[0] + offset[0], target[1] + offset[1], target[2] + offset[2])
        visualizer.set_camera_view(eye, target)


def _headless() -> bool:
    """Whether to run offscreen (default) or in a real window for HUD window-capture (see capture_common)."""
    return capture_common.headless("HERO_WINDOWED")


def _make_kit_visualizer_cfg() -> VisualizerCfg:
    from isaaclab_visualizers.kit import KitVisualizerCfg

    if _headless():
        # streaming_cam_target_prim_path follows the robot via a separate "Streaming View"
        # panel, captured through the standard VideoRecorderCfg path.
        return KitVisualizerCfg(
            headless=True,
            window_width=_WINDOW_WIDTH,
            window_height=_WINDOW_HEIGHT,
            streaming_view=True,
            streaming_envs=1,
            streaming_cam_target_prim_path=_STREAMING_TARGET_PRIM,
            streaming_cam_eye=_STREAMING_EYE,
            enable_markers=True,
        )
    # Windowed capture records the "Viewport" tab, not "Streaming View", so it follows via
    # _follow_camera instead, like the other _FOLLOW_CAM_VISUALIZERS.
    return KitVisualizerCfg(
        headless=False,
        window_width=_WINDOW_WIDTH,
        window_height=_WINDOW_HEIGHT,
        streaming_view=False,
        eye=_FOLLOW_EYE_OFFSET,
        lookat=(0.0, 0.0, 0.0),
        enable_markers=True,
    )


def _make_newton_gl_visualizer_cfg() -> VisualizerCfg:
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

    # No streaming_view: markers aren't drawn into that camera sensor (see module docstring).
    # eye/lookat seed the initial pose before _follow_camera takes over.
    return NewtonGLVisualizerCfg(
        headless=_headless(),
        window_width=_WINDOW_WIDTH,
        window_height=_WINDOW_HEIGHT,
        eye=_FOLLOW_EYE_OFFSET,
        lookat=(0.0, 0.0, 0.0),
        focal_length=_NARROW_FOCAL_LENGTH,
        sky_upper_color=_SKY_UPPER_COLOR,
        sky_lower_color=_SKY_LOWER_COLOR,
        enable_markers=True,
    )


def _make_newton_rtx_visualizer_cfg() -> VisualizerCfg:
    from isaaclab_visualizers.newton import NewtonRTXVisualizerCfg

    # No streaming_view: unusable for kitless RTX (see module docstring). Markers are
    # unsupported on this backend regardless of enable_markers (hard-blocked for the kitless
    # viewer).
    return NewtonRTXVisualizerCfg(
        headless=_headless(),
        window_width=_WINDOW_WIDTH,
        window_height=_WINDOW_HEIGHT,
        eye=_RTX_FOLLOW_EYE_OFFSET,
        lookat=(0.0, 0.0, 0.0),
        focal_length=_NARROW_FOCAL_LENGTH,
        rtx_environment="studio",
        sky_upper_color=_SKY_UPPER_COLOR,
        sky_lower_color=_SKY_LOWER_COLOR,
    )


def _make_rerun_visualizer_cfg() -> VisualizerCfg:
    from isaaclab_visualizers.rerun import RerunVisualizerCfg

    # No streaming_view: captures its own native 3D view (see module docstring).
    return RerunVisualizerCfg(eye=_BROWSER_FOLLOW_EYE_OFFSET, lookat=(0.0, 0.0, 0.0), enable_markers=True)


def _make_viser_visualizer_cfg() -> VisualizerCfg:
    from isaaclab_visualizers.viser import ViserVisualizerCfg

    # Same switch to native-view capture as Rerun; see _make_rerun_visualizer_cfg().
    return ViserVisualizerCfg(eye=_VISER_FOLLOW_EYE_OFFSET, lookat=(0.0, 0.0, 0.0), enable_markers=True)


_VISUALIZER_BUILDERS = {
    "kit": _make_kit_visualizer_cfg,
    "newton_gl": _make_newton_gl_visualizer_cfg,
    "newton_rtx": _make_newton_rtx_visualizer_cfg,
    "rerun": _make_rerun_visualizer_cfg,
    "viser": _make_viser_visualizer_cfg,
}

# Attached simultaneously (one physics process, one shared trajectory) when
# HERO_VISUALIZER=combined, avoiding the per-process browser-settle timing misalignment
# described in the module docstring. Kit and Newton RTX stay separate: they're already
# well-aligned via their own step-count-bound VideoRecorderCfg captures, and Kit needs a full
# Kit app process that can't coexist with Newton's kitless viewers.
_COMBINED_VISUALIZERS = ("newton_gl", "rerun", "viser")


@configclass
class AnymalDTileCaptureCfg(AnymalDFlatEnvCfg):
    """AnymalD flat-terrain configuration with a tile-capture visualizer and recorder."""

    def __post_init__(self):
        super().__post_init__()
        _configure_capture(self)


def configure_playback() -> list[str]:
    """Register the tile capture config and preserve Hydra arguments."""
    task = capture_common.argument_value("--task")
    if task != _TASK:
        raise ValueError(f"capture_hero is only configured for task {_TASK!r}, got {task!r}.")
    gym.spec(task).kwargs["env_cfg_entry_point"] = "capture_hero:AnymalDTileCaptureCfg"
    return sys.argv[1:]


def _configure_capture(env_cfg: AnymalDTileCaptureCfg) -> None:
    """Attach the requested visualizer and video recorder to the AnymalD flat config."""
    # Read the visualizer selection from HERO_VISUALIZER, not --viz off sys.argv: by the time
    # __post_init__ runs, rsl_rl has already rewritten sys.argv and AppLauncher's parser has
    # consumed --viz.
    visualizer = os.environ.get("HERO_VISUALIZER")
    if visualizer != "combined" and visualizer not in _VISUALIZER_BUILDERS:
        raise ValueError(
            f"Set HERO_VISUALIZER=<{'|'.join(_VISUALIZER_BUILDERS)}|combined> to select the hero capture"
            f" visualizer, got {visualizer!r}."
        )
    if visualizer == "combined":
        env_cfg.sim.visualizer_cfgs = [_VISUALIZER_BUILDERS[name]() for name in _COMBINED_VISUALIZERS]
    else:
        env_cfg.sim.visualizer_cfgs = [_VISUALIZER_BUILDERS[visualizer]()]

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
        func=_step_pacer, mode="interval", interval_range_s=(0.0, 0.0), is_global_time=True
    )
    # See _pin_velocity_command's docstring for why this is needed for cross-process
    # alignment, not just a nicer-looking constant-turn walk.
    env_cfg.events.pin_velocity_command = EventTermCfg(
        func=_pin_velocity_command, mode="interval", interval_range_s=(0.0, 0.0), is_global_time=True
    )

    if visualizer in _FOLLOW_CAM_VISUALIZERS or visualizer == "combined" or (visualizer == "kit" and not _headless()):
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
                    source=_SOURCE_BY_VISUALIZER["newton_gl"],
                    output_dir=output_dir,
                    output_filename_prefix=os.environ.get("HERO_VIDEO_PREFIX", "clip"),
                    fps=50,
                    step_offset=int(os.environ.get("HERO_COMBINED_STEP_OFFSET", "0")),
                )
            ]
        else:
            if visualizer not in _SOURCE_BY_VISUALIZER:
                raise ValueError(
                    f"'{visualizer}' is captured via a headless-browser recording, not VideoRecorderCfg (see"
                    " main()); unset HERO_VIDEO_DIR to launch it for that instead."
                )
            env_cfg.video_recorders = [
                VideoRecorderCfg(
                    source=_SOURCE_BY_VISUALIZER[visualizer],
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
# Driver: combined Newton GL/Rerun/Viser capture (needs playwright; see main())
# ---------------------------------------------------------------------------


def _find_urls(log_path: Path, timeout_s: float) -> tuple[str, str]:
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

    See the module docstring's "Rerun and Viser's hero clips previously each ran..."
    paragraph for the full rationale.
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
        "capture_hero.configure_playback",
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
        rerun_url, viser_url = _find_urls(log_path, args.find_timeout_s)
        print("Found Rerun URL:", rerun_url)
        print("Found Viser URL:", viser_url)

        with sync_playwright() as p:
            browser = capture_common.launch_browser(p, force_headless=args.force_headless)

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
            # the same moment; main() converts this to an equivalent Newton GL step_offset.
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
# Driver: top-level orchestration (generates all 5 hero_*.mp4 clips)
# ---------------------------------------------------------------------------

# 10s clip at this task's control rate: decimation=4, sim.dt=0.005s -> step_dt=0.02s (50 Hz).
# Kit and Newton RTX capture exactly this many steps.
_CAPTURE_STEPS = 500
# Newton GL/Rerun/Viser capture 10% more steps while _TARGET_SIM_SECONDS stays at 10 -- the
# combined capture's speed_factor compresses that extra motion into the same ~10s output.
_COMBINED_CAPTURE_STEPS = round(_CAPTURE_STEPS * 1.1)
# Calibrated real-time throughput target for the combined (Newton GL + Rerun + Viser) process.
_TARGET_SIM_SECONDS = 10
# Guards against a runaway record_s if the combined process is catastrophically slow to connect.
_MAX_RECORD_S = 90
# Fixed real-time wait before Rerun/Viser start recording, converted to an equivalent Newton
# GL step_offset below.
_COMBINED_SETTLE_S = 15
_UV_EXTRAS = "isaacsim,rerun,viser,rsl-rl,ov,video"


def _record_visualizer(
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
    _calibrate_combined() derives for Newton GL/Rerun/Viser) makes Kit/Newton RTX skip ahead
    to the same trajectory point instead of starting at step 0 -- otherwise the two groups'
    first frames would show a large, obvious rotational phase jump given the constant turn
    rate this pipeline uses.
    """
    output = work_dir / visualizer
    cmd = [
        "uv",
        "run",
        "--frozen",
        "--extra",
        _UV_EXTRAS,
        "python",
        "scripts/reinforcement_learning/play.py",
        "--rl_library",
        "rsl_rl",
        "--task",
        _TASK,
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
        "capture_hero.configure_playback",
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


def _calibrate_combined(script_dir: Path, repo_root: Path, work_dir: Path, seed: int | None) -> float:
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
        _TASK,
        str(work_dir / "calibrate_rerun.mp4"),
        str(work_dir / "calibrate_viser.mp4"),
        "--settle-s",
        str(_COMBINED_SETTLE_S),
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
        str(script_dir / "capture_hero.py"),
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


def main() -> None:
    """Generate all 5 hero_*.mp4 clips.

    Must be launched as ``uv run --no-project --with playwright --with pillow
    python capture_hero.py [--seed N]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[3]
    output_dir = repo_root / "docs/source/_static/visualizers"
    output_dir.mkdir(parents=True, exist_ok=True)

    capture_common.require_commands("uv", "ffmpeg")

    work_dir = Path(tempfile.mkdtemp())
    try:
        os.chdir(repo_root)

        print("Calibrating combined Newton GL/Rerun/Viser real-time throughput...")
        step_time = _calibrate_combined(script_dir, repo_root, work_dir, args.seed)
        # 20% safety margin: sustained throughput over the full production run can still run
        # a bit slower than the calibration.
        record_s = _COMBINED_CAPTURE_STEPS * step_time * 1.2 if step_time > 0 else _TARGET_SIM_SECONDS
        record_s = max(record_s, _TARGET_SIM_SECONDS)
        record_s = min(record_s, _MAX_RECORD_S)
        speed_factor = record_s / _TARGET_SIM_SECONDS
        step_offset = int(_COMBINED_SETTLE_S / step_time) if step_time > 0 else 0
        print(
            f"  combined: {step_time:.4f}s/step -> record-s={record_s:.1f}, "
            f"speed-factor={speed_factor:.3f}, step-offset={step_offset}"
        )

        # Same step_offset as the combined group so all 5 clips start at the same point.
        # Played back 10% slower than the other 3 -- see _record_visualizer's docstring.
        for visualizer in ("kit", "newton_rtx"):
            _record_visualizer(
                script_dir,
                repo_root,
                output_dir,
                work_dir,
                args.seed,
                visualizer,
                _CAPTURE_STEPS,
                step_offset,
                output_dir / f"hero_{visualizer}.mp4",
                output_speed_factor=0.9,
            )

        # Newton GL, Rerun, and Viser are captured together from one shared physics process.
        combined_cli_args = [
            str(repo_root),
            _TASK,
            str(output_dir / "hero_rerun.mp4"),
            str(output_dir / "hero_viser.mp4"),
            "--settle-s",
            str(_COMBINED_SETTLE_S),
            "--record-s",
            str(record_s),
            "--speed-factor",
            str(speed_factor),
            "--out-newton-gl-mp4",
            str(output_dir / "hero_newton_gl.mp4"),
            "--newton-gl-video-length",
            str(_COMBINED_CAPTURE_STEPS),
            "--newton-gl-step-offset",
            str(step_offset),
            *(["--seed", str(args.seed)] if args.seed is not None else []),
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
            str(script_dir / "capture_hero.py"),
            "--combined-capture",
            *combined_cli_args,
        ]
        subprocess.run(cmd, check=True)

        # Unlike Rerun/Viser (already compressed to ~10s by _run_combined_capture's own
        # speed_factor), Newton GL's clip comes out of VideoRecorderCfg at native fps
        # (~11s), so it needs the same step-count ratio applied here to match.
        newton_gl_speedup = _COMBINED_CAPTURE_STEPS / _CAPTURE_STEPS
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


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--combined-capture":
        _run_combined_capture(_combined_capture_argparser().parse_args(sys.argv[2:]))
    else:
        main()
