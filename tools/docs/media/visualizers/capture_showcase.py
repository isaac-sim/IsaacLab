# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture and generate the visualizer showcase clips.

Produces five short clips, one per visualizer backend, each running a *different* task's
published pretrained checkpoint at that visualizer's normal wide interactive camera view:
Newton GL (Isaac-Reorient-Cube-Allegro), Kit (Isaac-Reach-Franka), Newton RTX
(Isaac-Velocity-Rough-H1, stairs-only terrain, see :data:`_STAIRS_TERRAINS_CFG`), Rerun
(Isaac-Reorient-Cube-Shadow-Direct, 64 envs), and Viser (Isaac-Lift-KukaAllegro, 128 envs).

Rerun and Viser both render through Newton's streaming/camera-composited path, which has a
known ground-material bug (see ``newton_adapter.py``); :func:`_add_floor_overlay` works
around it. Neither backend implements ``render_rgb_array()``, so their clips are captured via
a real Chrome browser instead of ``VideoRecorderCfg`` -- see :func:`record_showcase_browser`.

The per-task capture overrides are injected via a capture-only ``*ShowcaseCfg`` subclass that
swaps ``gym.spec(task).kwargs["env_cfg_entry_point"]`` before Hydra resolves the config, so
the real task configs are never edited (see :func:`configure_playback`).

This module runs across two process tiers:

* Config-time code (module-level constants, ``_make_*_visualizer_cfg``, ``*ShowcaseCfg``,
  :func:`configure_playback`) is imported via ``--external_callback`` *inside* ``play.py``'s
  own process, in the full project ``uv`` environment (``--frozen --extra isaacsim,...``).
* Driver-time code (:func:`record_showcase_window`, :func:`record_showcase`,
  :func:`record_showcase_browser`, :func:`main`) runs in a separate driver process that
  launches ``play.py`` as a subprocess per clip and does the window/browser recording. Needs
  ``python-xlib`` (windowed HUD captures) and ``playwright``/``pillow`` (Rerun/Viser browser
  captures) -- ephemeral dependencies, not part of this repo's own dependency tree.

Usage (regenerate all five clips):
    uv run --no-project --with python-xlib --with playwright --with pillow \\
        python capture_showcase.py [--num-envs N] [--skip <visualizer>]...
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import capture_common
import gymnasium as gym

from pxr import Gf, Sdf, UsdShade

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
from isaaclab_tasks.core.velocity.config.h1.rough_env_cfg import H1RoughEnvCfg

# ===========================================================================
# Config-time code: imported via --external_callback into play.py's process.
# ===========================================================================

# Restricts the H1 showcase terrain to the ascending pyramid-stairs sub-terrain only, so every
# spawned robot lands on a staircase (see _frame_wide_camera_from_env_origins).
_STAIRS_TERRAINS_CFG = dataclasses.replace(
    ROUGH_TERRAINS_CFG,
    sub_terrains={
        "pyramid_stairs": dataclasses.replace(ROUGH_TERRAINS_CFG.sub_terrains["pyramid_stairs"], proportion=1.0),
    },
)

_WIDE_LOOKAT = (0.0, 0.0, 0.0)
_REACH_EYE = (3.6, -3.6, 3.8)
_REACH_LOOKAT = (0.3, 0.0, -0.3)
# Kuka Allegro lift: same pitch/distance as Franka reach, viewed from the opposite horizontal
# side (x/y negated) so the two clips don't look like mirrored duplicates.
_KUKA_ALLEGRO_LIFT_EYE = (-_REACH_EYE[0], -_REACH_EYE[1], _REACH_EYE[2])
_KUKA_ALLEGRO_LIFT_LOOKAT = _REACH_LOOKAT
_KUKA_ALLEGRO_LIFT_ENV_SPACING = 2.4
# Pulled back so the top row of envs isn't cropped out of frame.
_NEWTON_GL_EYE = (2.1, -2.1, 2.3)
_NEWTON_GL_LOOKAT = (0.0, 0.0, -0.1)
# Zoomed in close: Shadow Hand's env_spacing is tiny, so a wide camera leaves the hands as an
# indistinct speck.
_SHADOW_HAND_EYE = (1.1, -1.1, 2.15)
_SHADOW_HAND_LOOKAT = (0.0, 0.0, -0.15)
_SHADOW_HAND_ENV_SPACING = 0.75
# Offset from the framed terrain-origin center (see _frame_wide_camera_from_env_origins).
_H1_STAIRS_EYE_OFFSET = (1.57, -2.97, 2.50)

_WINDOW_WIDTH = 960
_WINDOW_HEIGHT = 600
# Rounded to an even pixel count: libx264's yuv420p output requires even width/height.
_NEWTON_GL_WINDOW_WIDTH = round(_WINDOW_WIDTH * 1.3 * 1.15 / 2) * 2
_NEWTON_GL_WINDOW_HEIGHT = round(_WINDOW_HEIGHT * 1.3 * 1.15 / 2) * 2
# Newton RTX's path tracer benefits from extra resolution the rasterized backends don't need.
_H1_STAIRS_WINDOW_WIDTH = _WINDOW_WIDTH * 2
_H1_STAIRS_WINDOW_HEIGHT = _WINDOW_HEIGHT * 2


def _headless() -> bool:
    """Whether to run offscreen (default) or in a real window for HUD window-capture (see capture_common)."""
    return capture_common.headless("SHOWCASE_WINDOWED")


def _make_kit_visualizer_cfg() -> VisualizerCfg:
    from isaaclab_visualizers.kit import KitVisualizerCfg

    return KitVisualizerCfg(
        headless=_headless(),
        window_width=_WINDOW_WIDTH,
        window_height=_WINDOW_HEIGHT,
        eye=_REACH_EYE,
        lookat=_REACH_LOOKAT,
        enable_markers=True,
    )


def _make_newton_gl_visualizer_cfg() -> VisualizerCfg:
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

    # streaming_view defaults to True; this showcase only wants the interactive camera.
    return NewtonGLVisualizerCfg(
        headless=_headless(),
        window_width=_NEWTON_GL_WINDOW_WIDTH,
        window_height=_NEWTON_GL_WINDOW_HEIGHT,
        streaming_view=False,
        eye=_NEWTON_GL_EYE,
        lookat=_NEWTON_GL_LOOKAT,
        enable_markers=True,
    )


def _make_newton_rtx_visualizer_cfg() -> VisualizerCfg:
    from isaaclab_visualizers.newton import NewtonRTXVisualizerCfg

    # Markers are hard-blocked in NewtonRTXVisualizer for the kitless viewer.
    return NewtonRTXVisualizerCfg(
        headless=_headless(),
        window_width=_H1_STAIRS_WINDOW_WIDTH,
        window_height=_H1_STAIRS_WINDOW_HEIGHT,
        rtx_environment="default",
        eye=_H1_STAIRS_EYE_OFFSET,
        lookat=_WIDE_LOOKAT,
    )


def _make_rerun_visualizer_cfg() -> VisualizerCfg:
    from isaaclab_visualizers.rerun import RerunVisualizerCfg

    # Live plots disabled: not useful in a short clip, and adds overhead to an already
    # browser-bottlenecked capture.
    return RerunVisualizerCfg(
        eye=_SHADOW_HAND_EYE, lookat=_SHADOW_HAND_LOOKAT, enable_markers=True, enable_live_plots=False
    )


def _make_viser_visualizer_cfg() -> VisualizerCfg:
    from isaaclab_visualizers.viser import ViserVisualizerCfg

    return ViserVisualizerCfg(eye=_KUKA_ALLEGRO_LIFT_EYE, lookat=_KUKA_ALLEGRO_LIFT_LOOKAT, enable_markers=True)


_VISUALIZER_BUILDERS = {
    "kit": _make_kit_visualizer_cfg,
    "newton_gl": _make_newton_gl_visualizer_cfg,
    "newton_rtx": _make_newton_rtx_visualizer_cfg,
    "rerun": _make_rerun_visualizer_cfg,
    "viser": _make_viser_visualizer_cfg,
}

_SOURCE_BY_VISUALIZER = {
    "kit": "visualizer:kit",
    "newton_gl": "visualizer:newton_gl",
    "newton_rtx": "visualizer:newton_rtx",
}


# Light-blue overlay color for _add_floor_overlay; deliberately more saturated than Newton
# GL's own muted built-in ground color so it reads as clearly blue in the browser views.
_FLOOR_OVERLAY_COLOR = (0.45, 0.7, 1.0)

_FLOOR_OVERLAY_PRIM_PATH = "/World/FloorOverlay"


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
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*_FLOOR_OVERLAY_COLOR))
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
        prim_path=_FLOOR_OVERLAY_PRIM_PATH,
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.01)),
        spawn=MeshCuboidCfg(
            func=_spawn_floor_overlay_cuboid,
            # Matches GroundPlaneCfg's default 100x100m size to cover every cloned env.
            size=(100.0, 100.0, 0.01),
        ),
    )


def _resolved_sim_cfgs(env_cfg: object) -> list[SimulationCfg]:
    """Return every concrete :class:`SimulationCfg` on ``env_cfg.sim``.

    Some task configs (e.g. the cabinet tasks) hold a per-backend preset container on
    ``sim`` instead of a single resolved :class:`SimulationCfg` at this point in
    construction, since backend selection happens later via Hydra presets.
    """
    sim = env_cfg.sim
    if isinstance(sim, SimulationCfg):
        return [sim]
    return [value for field in dataclasses.fields(sim) if isinstance(value := getattr(sim, field.name), SimulationCfg)]


def _frame_wide_camera_from_env_origins(env, env_ids) -> None:
    """Point the interactive camera at the mean spawn position of the visible envs.

    H1's rough-terrain generator spreads sub-terrains across a large grid, so a fixed
    world-origin eye/lookat can easily miss every robot. Registered as an ``EventTermCfg``
    with ``mode="reset"`` so it only recomputes on env reset, not every step.
    """
    del env_ids
    center = env.scene.env_origins.mean(dim=0)
    eye = (
        float(center[0]) + _H1_STAIRS_EYE_OFFSET[0],
        float(center[1]) + _H1_STAIRS_EYE_OFFSET[1],
        float(center[2]) + _H1_STAIRS_EYE_OFFSET[2],
    )
    target = (float(center[0]), float(center[1]), float(center[2]))
    for visualizer in env.sim.visualizers:
        if hasattr(visualizer, "set_camera_view"):
            visualizer.set_camera_view(eye, target)


# Paces windowed HUD captures (see record_showcase_window) to a slower wall-clock rate so the
# renderer has more headroom per frame, reducing stutter; main() speeds the clip back up in
# post. See capture_common.make_step_pacer's docstring for the full rationale.
_step_pacer = capture_common.make_step_pacer("SHOWCASE_TARGET_STEP_TIME")


def _configure_capture(env_cfg: object, visualizer: str) -> None:
    """Attach the requested showcase visualizer and video recorder to ``env_cfg``."""
    for sim_cfg in _resolved_sim_cfgs(env_cfg):
        sim_cfg.visualizer_cfgs = [_VISUALIZER_BUILDERS[visualizer]()]

    if os.environ.get("SHOWCASE_WINDOWED") == "1":
        env_cfg.events.step_pacer = EventTermCfg(
            func=_step_pacer, mode="interval", interval_range_s=(0.0, 0.0), is_global_time=True
        )

    output_dir = os.environ.get("SHOWCASE_VIDEO_DIR")
    if output_dir:
        if visualizer not in _SOURCE_BY_VISUALIZER:
            raise ValueError(
                f"VideoRecorderCfg capture is not supported for the '{visualizer}' showcase visualizer (no"
                " render_rgb_array() implementation); see record_showcase_browser instead. Unset"
                " SHOWCASE_VIDEO_DIR to launch it for that."
            )
        env_cfg.video_recorders = [
            VideoRecorderCfg(
                source=_SOURCE_BY_VISUALIZER[visualizer],
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
        _configure_capture(self, "newton_gl")
        # ReorientManagerEnvBaseCfg never overrides GroundPlaneCfg.color, which defaults to
        # black; fix it just for this capture rather than editing the real task.
        self.scene.ground.spawn.color = (1.0, 1.0, 1.0)
        _add_floor_overlay(self)


@configclass
class FrankaReachShowcaseCfg(FrankaReachEnvCfg):
    """Franka reach configuration with the Kit showcase visualizer."""

    def __post_init__(self):
        super().__post_init__()
        _configure_capture(self, "kit")
        # No _add_floor_overlay: Kit isn't affected by the floor bug, and the task's mounting
        # table sits at z=0, so an overlay here would hide it (confirmed by testing).
        self.scene.env_spacing = 2.0


@configclass
class H1RoughShowcaseCfg(H1RoughEnvCfg):
    """H1 stairs-only rough-terrain locomotion configuration with the Newton RTX showcase visualizer."""

    def __post_init__(self):
        super().__post_init__()
        _configure_capture(self, "newton_rtx")
        self.scene.terrain.terrain_generator = _STAIRS_TERRAINS_CFG
        self.events.frame_wide_camera = EventTermCfg(func=_frame_wide_camera_from_env_origins, mode="reset")


@configclass
class ShadowHandReorientShowcaseCfg(ShadowHandEnvCfg):
    """Shadow Hand cube reorientation configuration with the Rerun showcase visualizer."""

    def __post_init__(self):
        super().__post_init__()
        _configure_capture(self, "rerun")
        self.scene.env_spacing = _SHADOW_HAND_ENV_SPACING
        _add_floor_overlay(self)


@configclass
class KukaAllegroLiftShowcaseCfg(KukaAllegroLiftEnvCfg):
    """Kuka Allegro cube lift configuration with the Viser showcase visualizer."""

    def __post_init__(self):
        super().__post_init__()
        _configure_capture(self, "viser")
        self.scene.env_spacing = _KUKA_ALLEGRO_LIFT_ENV_SPACING
        _add_floor_overlay(self)


_TASK_CONFIGS = {
    "Isaac-Reorient-Cube-Allegro": "capture_showcase:AllegroReorientShowcaseCfg",
    "Isaac-Reach-Franka": "capture_showcase:FrankaReachShowcaseCfg",
    "Isaac-Velocity-Rough-H1": "capture_showcase:H1RoughShowcaseCfg",
    "Isaac-Reorient-Cube-Shadow-Direct": "capture_showcase:ShadowHandReorientShowcaseCfg",
    "Isaac-Lift-KukaAllegro": "capture_showcase:KukaAllegroLiftShowcaseCfg",
}


def configure_playback() -> list[str]:
    """Register the task-specific showcase capture config and preserve Hydra arguments."""
    task = capture_common.argument_value("--task")
    if task not in _TASK_CONFIGS:
        raise ValueError(f"No showcase capture configuration is registered for task {task!r}.")
    gym.spec(task).kwargs["env_cfg_entry_point"] = _TASK_CONFIGS[task]
    return sys.argv[1:]


# ===========================================================================
# Driver-time code: launches play.py as a subprocess and records the result.
# ===========================================================================

# Browser viewport / recorded video size for both Rerun and Viser.
_VIEWPORT_WIDTH = 1200
_VIEWPORT_HEIGHT = 750

# Pixel location of the "Software WebGL rendering detected" banner's close button
# (headless-Chrome-only artifact), scaled from its calibrated position at 960x600.
_WEBGL_BANNER_CLOSE_XY = (round(288 * _VIEWPORT_WIDTH / 960), round(81 * _VIEWPORT_HEIGHT / 600))


def _find_viewer_url(log_path: Path, timeout_s: float) -> str:
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


def _mean_abs_pixel_diff(a, b) -> float:
    """Mean absolute grayscale pixel difference between two images, downsampled for speed."""
    a = a.convert("L").resize((160, 100))
    b = b.convert("L").resize((160, 100))
    pixels_a, pixels_b = a.tobytes(), b.tobytes()
    return sum(abs(x - y) for x, y in zip(pixels_a, pixels_b)) / len(pixels_a)


def _dismiss_webgl_banner(page) -> None:
    with contextlib.suppress(Exception):
        page.mouse.click(*_WEBGL_BANNER_CLOSE_XY)


# Window titles for capture_common.record_windowed's X11 WM_NAME match.
WINDOW_TITLES = {
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
    capture_common.record_windowed(
        repo_root=repo_root,
        viz=visualizer,
        task=task,
        num_envs=num_envs,
        external_callback="capture_showcase.configure_playback",
        out_mp4=str(output_dir / filename),
        window_title=WINDOW_TITLES[visualizer],
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
        "capture_showcase.configure_playback",
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
        "capture_showcase.configure_playback",
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
        url = _find_viewer_url(log_path, timeout_s=180.0)
        print("Found viewer URL:", url)

        with sync_playwright() as p:
            browser = capture_common.launch_browser(p, force_headless=force_headless)

            # Single continuous connection: reconnecting forces a full re-sync of everything
            # already logged, which never finishes for large scene geometry.
            context = browser.new_context(
                viewport={"width": _VIEWPORT_WIDTH, "height": _VIEWPORT_HEIGHT},
                record_video_dir=str(video_dir),
                record_video_size={"width": _VIEWPORT_WIDTH, "height": _VIEWPORT_HEIGHT},
            )
            page = context.new_page()
            page.goto(url, timeout=30000)

            page.wait_for_timeout(3000)
            _dismiss_webgl_banner(page)
            page.wait_for_timeout(8000)  # let the websocket scene sync start populating

            if reset:
                try:
                    page.get_by_text("Reset Episode", exact=True).first.click(timeout=5000, force=True)
                    print("Clicked 'Reset Episode'")
                    page.wait_for_timeout(3000)
                except Exception as exc:
                    print(f"Could not click 'Reset Episode' (expected on Rerun): {exc}")

            deadline = time.time() + max_wait_s
            prev_frame = _open_image(page.screenshot())
            motion_detected = False
            while time.time() < deadline:
                page.wait_for_timeout(1500)
                cur_frame = _open_image(page.screenshot())
                diff = _mean_abs_pixel_diff(prev_frame, cur_frame)
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


def _open_image(png_bytes: bytes):
    from PIL import Image

    return Image.open(io.BytesIO(png_bytes))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--skip", action="append", default=[], help="Visualizer to skip entirely (repeatable).")
    args = parser.parse_args()

    script_dir = str(Path(__file__).resolve().parent)
    repo_root = str(Path(script_dir).parents[3])
    output_dir = Path(repo_root) / "docs/source/_static/visualizers"
    work_dir = Path("/tmp/capture_showcase_work")
    work_dir.mkdir(parents=True, exist_ok=True)

    capture_common.require_commands("uv", "ffmpeg")

    output_dir.mkdir(parents=True, exist_ok=True)

    uv_extras = "isaacsim,rerun,viser,rsl-rl,ov,video"
    num_envs = args.num_envs
    # H1's stairs-only terrain only needs enough envs to populate the visible frame.
    h1_num_envs = 160
    # Extra settle time before H1's headless VideoRecorderCfg capture starts, on top of the
    # fixed step count (see VideoRecorderCfg.step_offset).
    h1_extra_wait_s = 10

    # Both windowed rows share control rate decimation=4, sim.dt=1/120s -> 0.0333s/step.
    # pace_speed_factor slows simulation down (more render headroom, less GPU-contention
    # stutter) and speeds the recorded clip back up by the same factor in post (see
    # capture_common.record_windowed's speed_factor / _step_pacer above); record_s is
    # stretched by the same factor so 10s of *simulated* motion is still covered.
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
        return visualizer in args.skip

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
    print(f"Showcase clips written to {output_dir} (skipped: {args.skip or 'none'}).")


if __name__ == "__main__":
    main()
