# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generates the streaming-camera-view demo clips used by docs/source/concepts/visualization.rst
and docs/source/features/visualizer_tiled_camera.rst, captured as 10s video instead of the
tutorial script's static screenshots.

* **Kit + Isaac-Velocity-Rough-AnymalD** uses the pretrained checkpoint via ``play.py``'s
  ``--external_callback`` (see :func:`configure_playback`). Its streaming panel docks as a
  second tab in the *same* dock group as the interactive viewport, not side by side, so this
  captures two separate clips (toggling ``STREAMING_KIT_SHOW_PANEL``).
* **Newton GL + the Galbot cube-stacking task** has no pretrained checkpoint, so it keeps the
  tutorial script's random-action setup and is launched directly (see
  :func:`_newton_gl_worker_main`), never through ``play.py``. Its streaming panel is an ImGui
  sidebar combo that starts hidden; this sets ``image_logger._selected`` directly between two
  back-to-back recordings in the same process/window.

Run directly to regenerate every clip (each mode spawned by :func:`main` as its own subprocess
with the extras it needs -- see :func:`main`'s docstring)::

    uv run --no-project --with python-xlib python capture_streaming.py [--skip kit] [--skip newton_gl]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import capture_common
import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.managers import EventTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.core.velocity.config.anymal_d.rough_env_cfg import AnymalDRoughEnvCfg
from isaaclab_tasks.utils import resolve_task_config

# ---------------------------------------------------------------------------
# Kit + AnymalD (config-time code, imported via play.py's --external_callback)
# ---------------------------------------------------------------------------

_KIT_TASK = "Isaac-Velocity-Rough-AnymalD"

# Auto-follow camera path (AnymalD has no ego_cam scene camera), matching
# run_tiled_camera_visualizer.py's _make_kit_visualizer_cfg.
_STREAMING_ENVS = 36
_STREAMING_CAM_EYE = (3.0, 3.0, 3.0)
_STREAMING_CAM_TARGET_PRIM_PATH = "/World/envs/*/Robot/base"

# Pulled back to fit the 128-env clone grid (~28x25m at 2.5m env_spacing) in frame.
_INTERACTIVE_EYE = (18.0, -18.0, 14.0)
_INTERACTIVE_LOOKAT = (0.0, 0.0, 0.0)

# Streaming panel size (see KitVisualizer._setup_camera_image_window).
_WINDOW_WIDTH = 960
_WINDOW_HEIGHT = 600

# Orchestration constants shared by both Kit calls in main().
_KIT_NUM_ENVS = 128
_KIT_WINDOW_TITLE = "Isaac Lab"


def _headless() -> bool:
    """Whether to run offscreen (default) or in a real window for HUD window-capture (see capture_common)."""
    return capture_common.headless("STREAMING_WINDOWED")


def _make_kit_streaming_visualizer_cfg() -> VisualizerCfg:
    from isaaclab_visualizers.kit import KitVisualizerCfg

    return KitVisualizerCfg(
        headless=_headless(),
        window_width=_WINDOW_WIDTH,
        window_height=_WINDOW_HEIGHT,
        eye=_INTERACTIVE_EYE,
        lookat=_INTERACTIVE_LOOKAT,
        # Toggled per capture run (see main()) -- panel docks as a second tab, not side by side.
        streaming_view=os.environ.get("STREAMING_KIT_SHOW_PANEL") == "1",
        streaming_envs=_STREAMING_ENVS,
        streaming_cam_eye=_STREAMING_CAM_EYE,
        streaming_cam_target_prim_path=_STREAMING_CAM_TARGET_PRIM_PATH,
        enable_markers=False,
    )


_step_pacer = capture_common.make_step_pacer("STREAMING_TARGET_STEP_TIME")


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
        self.sim.visualizer_cfgs = [_make_kit_streaming_visualizer_cfg()]
        if os.environ.get("STREAMING_WINDOWED") == "1":
            self.events.step_pacer = EventTermCfg(
                func=_step_pacer, mode="interval", interval_range_s=(0.0, 0.0), is_global_time=True
            )
        if os.environ.get("STREAMING_KIT_SHOW_PANEL") == "1":
            self.events.focus_streaming_tab = EventTermCfg(
                func=_focus_streaming_tab, mode="interval", interval_range_s=(0.0, 0.0), is_global_time=True
            )


_TASK_CONFIGS = {_KIT_TASK: "capture_streaming:AnymalDStreamingShowcaseCfg"}


def configure_playback() -> list[str]:
    """Register the streaming-view capture config and preserve Hydra arguments."""
    task = capture_common.argument_value("--task")
    if task not in _TASK_CONFIGS:
        raise ValueError(f"No streaming-view capture configuration is registered for task {task!r}.")
    gym.spec(task).kwargs["env_cfg_entry_point"] = _TASK_CONFIGS[task]
    return sys.argv[1:]


# ---------------------------------------------------------------------------
# Newton GL + Galbot (standalone worker -- never goes through play.py)
# ---------------------------------------------------------------------------

_NEWTON_GL_TASK = "IsaacContrib-Stack-Cube-Galbot-Left-Arm-Gripper-Visuomotor"
_NEWTON_GL_NUM_ENVS = 25
_NEWTON_GL_STREAMING_ENVS = 12
_NEWTON_GL_PANEL_KEY = "Streaming View"
_NEWTON_GL_WINDOW_TITLE = "Newton Viewer"
# The Galbot task's config module imports isaaclab_teleop (stack_joint_pos_env_cfg.py); "mimic"
# is the lightest extra providing it without the heavier full "teleop" extra's deps.
_NEWTON_GL_UV_EXTRAS = "isaacsim,rerun,viser,rsl-rl,ov,video,mimic"


def _resolve_env_regex_path(prim_path: str) -> str:
    """Resolve scene config env namespace macros to the cloned-env regex."""
    return prim_path.format(ENV_REGEX_NS="/World/envs/env_.*")


def _make_newton_gl_visualizer_cfg(env_cfg):
    """Matches run_tiled_camera_visualizer.py's _make_newton_visualizer_cfg for this task."""
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

    cfg = NewtonGLVisualizerCfg()
    cfg.streaming_view = True
    cfg.streaming_envs = _NEWTON_GL_STREAMING_ENVS

    ego_cam_cfg = getattr(env_cfg.scene, "ego_cam", None)
    if ego_cam_cfg is not None:
        cfg.streaming_sensor_prim_path = _resolve_env_regex_path(ego_cam_cfg.prim_path)
    else:
        cfg.streaming_cam_eye = (3.0, 3.0, 3.0)
        cfg.streaming_cam_target_prim_path = "/World/envs/*/Robot/base"
    return cfg


def _step_for(env, seconds: float) -> None:
    """Step the env with random actions until ``seconds`` of wall-clock time has passed."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        with torch.inference_mode():
            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            env.step(actions)


def _force_large_streaming_panel(visualizer, width_px: float, height_px: float) -> None:
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


def _finalize_newton_gl_clip(webm: Path, out_mp4: Path, speed_factor: float = 1.0) -> None:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(webm)]
    video_filters = []
    border_crop = capture_common.detect_border_crop(webm)
    if border_crop:
        video_filters.append(border_crop)
    if speed_factor != 1.0:
        video_filters.append(f"setpts=PTS/{speed_factor}")
    if video_filters:
        cmd += ["-vf", ",".join(video_filters)]
    cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", "24", "-pix_fmt", "yuv420p", str(out_mp4)]
    subprocess.run(cmd, check=True)


def _record_newton_gl_segment(env, window_id: int, out_webm: Path, seconds: float) -> None:
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
    _step_for(env, seconds + 1.0)
    returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(f"ffmpeg x11grab recording to {out_webm} exited with code {returncode}.")


def _newton_gl_worker_main(argv: list[str]) -> None:
    """Standalone launcher + window capture for the Newton GL + Galbot streaming-view demo.

    Runs in its own process (spawned by :func:`main`), never through play.py.
    """
    parser = argparse.ArgumentParser(description=_newton_gl_worker_main.__doc__)
    parser.add_argument("out_interactive_mp4")
    parser.add_argument("out_tiled_mp4")
    parser.add_argument("--record-s", type=float, default=10.0, help="Recorded (real wall-clock) length per clip.")
    parser.add_argument("--settle-s", type=float, default=15.0, help="Wait after the window appears before recording.")
    parser.add_argument("--window-title", default=_NEWTON_GL_WINDOW_TITLE, help="Substring to match against WM_NAME.")
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
    args_cli.task = _NEWTON_GL_TASK
    sys.argv = [sys.argv[0]] + hydra_args

    if "DISPLAY" not in os.environ:
        os.environ["DISPLAY"] = ":0"

    env_cfg, _ = resolve_task_config(_NEWTON_GL_TASK, "")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = _NEWTON_GL_NUM_ENVS
        env_cfg.sim.visualizer_cfgs = [_make_newton_gl_visualizer_cfg(env_cfg)]

        env = gym.make(_NEWTON_GL_TASK, cfg=env_cfg)
        print("env created, resetting...", flush=True)
        env.reset()
        print("env reset complete", flush=True)

        from Xlib import display

        disp = display.Display()
        window_id = capture_common.find_window(disp, args_cli.window_title, args_cli.find_timeout_s)
        print(f"Found window {window_id:#x} titled like {args_cli.window_title!r}, settling...", flush=True)
        _step_for(env, args_cli.settle_s)
        print("settle complete", flush=True)
        window_id = capture_common.find_window(disp, args_cli.window_title, args_cli.find_timeout_s)

        work_dir = Path("/tmp/capture_streaming_newton_gl")
        work_dir.mkdir(parents=True, exist_ok=True)

        # Streaming panel left hidden for the interactive segment.
        interactive_webm = work_dir / "interactive.webm"
        print("recording interactive segment...", flush=True)
        _record_newton_gl_segment(env, window_id, interactive_webm, args_cli.record_s)
        print("interactive segment recorded", flush=True)

        # Force the streaming panel open for the tiled segment.
        sim = env.unwrapped.sim
        visualizer = sim.visualizers[0]
        image_logger = getattr(visualizer._viewer, "_image_logger", None)
        if image_logger is None or _NEWTON_GL_PANEL_KEY not in getattr(image_logger, "_images", {}):
            raise RuntimeError(
                "Streaming panel was not registered after the settle/interactive steps; "
                "increase --settle-s or --record-s."
            )
        image_logger._selected = _NEWTON_GL_PANEL_KEY
        _force_large_streaming_panel(visualizer, args_cli.streaming_panel_width, args_cli.streaming_panel_height)
        print("streaming panel forced open, recording tiled segment...", flush=True)

        tiled_webm = work_dir / "tiled.webm"
        _record_newton_gl_segment(env, window_id, tiled_webm, args_cli.record_s)
        print("tiled segment recorded", flush=True)

        # Finalize the mp4s *before* env.close(): Isaac Sim's app shutdown can terminate the
        # process before control returns here (confirmed by testing), so anything that must
        # land on disk has to happen while the app is still alive.
        _finalize_newton_gl_clip(interactive_webm, Path(args_cli.out_interactive_mp4), args_cli.speed_factor)
        _finalize_newton_gl_clip(tiled_webm, Path(args_cli.out_tiled_mp4), args_cli.speed_factor)
        print("Wrote", args_cli.out_interactive_mp4, "and", args_cli.out_tiled_mp4, flush=True)

        env.close()


# ---------------------------------------------------------------------------
# Orchestration (replaces generate_streaming.sh)
# ---------------------------------------------------------------------------


def main() -> None:
    """Regenerate every streaming-view clip.

    Run directly with::

        uv run --no-project --with python-xlib python capture_streaming.py

    ``--with python-xlib`` alone covers the Kit portion (runs directly in this process); the
    Newton GL portion is spawned as its own subprocess with the extras it needs. Use
    ``--skip kit`` / ``--skip newton_gl`` (repeatable) to skip a mode.
    """
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[3]
    output_dir = repo_root / "docs" / "source" / "_static" / "visualizers"

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--skip", action="append", default=[], choices=["kit", "newton_gl"])
    args = parser.parse_args()
    skip = set(args.skip)

    capture_common.require_commands("uv", "ffmpeg")

    output_dir.mkdir(parents=True, exist_ok=True)

    record_s = 10.0
    # Baked into the output clips, so a 10s capture plays back as a 5s clip.
    speed_factor = 2.0

    if "kit" not in skip:
        print("--- kit + Isaac-Velocity-Rough-AnymalD (streaming view) ---")

        # Two separate captures since the streaming panel docks as a second tab in the same
        # group as the viewport (see AnymalDStreamingShowcaseCfg), not side by side.
        os.environ["STREAMING_KIT_SHOW_PANEL"] = "0"
        capture_common.record_windowed(
            repo_root=str(repo_root),
            viz="kit",
            task=_KIT_TASK,
            num_envs=_KIT_NUM_ENVS,
            external_callback="capture_streaming.configure_playback",
            out_mp4=str(output_dir / "streaming_kit_anymal_interactive.mp4"),
            window_title=_KIT_WINDOW_TITLE,
            windowed_env_var="STREAMING_WINDOWED",
            record_s=record_s,
            settle_s=90.0,
            speed_factor=speed_factor,
            crf=24,
            hydra_overrides=["physics=newton_mjwarp"],
        )

        os.environ["STREAMING_KIT_SHOW_PANEL"] = "1"
        capture_common.record_windowed(
            repo_root=str(repo_root),
            viz="kit",
            task=_KIT_TASK,
            num_envs=_KIT_NUM_ENVS,
            external_callback="capture_streaming.configure_playback",
            out_mp4=str(output_dir / "streaming_kit_anymal_tiled.mp4"),
            window_title=_KIT_WINDOW_TITLE,
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
                _NEWTON_GL_UV_EXTRAS,
                "--with",
                "python-xlib",
                "python",
                str(script_dir / "capture_streaming.py"),
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


if __name__ == "__main__":
    # "--newton-gl-worker" is an internal dispatch flag used only when main() re-invokes this
    # file as a subprocess; not part of the public CLI.
    if len(sys.argv) > 1 and sys.argv[1] == "--newton-gl-worker":
        try:
            _newton_gl_worker_main(sys.argv[2:])
        except BaseException:
            import traceback

            traceback.print_exc()
            sys.exit(1)
    else:
        main()
