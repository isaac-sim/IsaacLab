# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests: cartpole env + per-backend visualizers (Kit Replicator, tiled camera, GL, Rerun, Viser).

Visualizer packages use ``logging.getLogger(__name__)``, so loggers are named like
``isaaclab_visualizers.kit.kit_visualizer`` and ``isaaclab.visualizers.base_visualizer``.
:class:`~isaaclab.sim.simulation_context.SimulationContext` uses
``logging.getLogger(__name__)`` → ``isaaclab.sim.simulation_context``.

We filter :class:`~pytest.LogCaptureFixture` records with :data:`_VIS_LOGGER_PREFIXES`
so only those namespaces count (not Omniverse, PhysX, or unrelated warnings).

Set :data:`ASSERT_VISUALIZER_WARNINGS` to ``True`` locally or in CI if you want tests to
fail on WARNING-level records from those loggers; by default only ERROR+ fails.
"""

from __future__ import annotations

import pyglet
import warp as wp

# Pyglet must use HeadlessWindow (EGL) before ``pyglet.window`` is imported so Newton
# ViewerGL can construct without an X11 display (matches ``headless=True`` on NewtonVisualizerCfg).
pyglet.options["headless"] = True

from isaaclab.app import AppLauncher

# launch Kit app
simulation_app = AppLauncher(headless=True, enable_cameras=True).app

import contextlib
import copy
import logging
import socket
from pathlib import Path

import numpy as np
import pytest
import torch
from isaaclab_visualizers.kit import KitVisualizer, KitVisualizerCfg
from isaaclab_visualizers.newton import NewtonVisualizer, NewtonVisualizerCfg

import omni.timeline

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

from isaaclab_tasks.direct.cartpole.cartpole_camera_env import CartpoleCameraEnv
from isaaclab_tasks.direct.cartpole.cartpole_camera_presets_env_cfg import CartpoleCameraPresetsEnvCfg
from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpolePhysicsCfg

# When True, tests also fail on WARNING-level records from visualizer-related loggers.
ASSERT_VISUALIZER_WARNINGS = False

_MAX_FRAME_CHECK_STEPS = 8
"""Steps for tiled camera / Rerun / Viser smoke tests (early exit ok when frame is non-flat)."""

_CARTPOLE_INTEGRATION_NUM_ENVS = 1
"""Vectorized env count for cartpole + visualizer integration tests."""

# TODO: Compare these direct-env visualizer checks with RSL-RL train runs;
# manager-based training can show different frozen-visualizer behavior.

_CARTPOLE_INTEGRATION_VISUALIZER_EYE: tuple[float, float, float] = (3.0, 3.0, 3.0)
"""Passed to :class:`~isaaclab.visualizers.visualizer_cfg.VisualizerCfg` subclasses (``eye``)."""

_CARTPOLE_INTEGRATION_VISUALIZER_LOOKAT: tuple[float, float, float] = (-4.0, -4.0, 0.0)
"""Passed to visualizer cfgs (``lookat``); also applied to :class:`~isaaclab.envs.common.ViewerCfg` for the env."""

# Resolution overrides for this test module (cartpole preset defaults: tiled camera 100×100; Kit helper was 320×240).
_CARTPOLE_KIT_INTEGRATION_RENDER_RESOLUTION: tuple[int, int] = (500, 500)
"""Kit: Replicator ``render_product`` (width, height) for viewport RGB in the motion check."""

_CARTPOLE_NEWTON_INTEGRATION_WINDOW_SIZE: tuple[int, int] = (500, 500)
"""Newton: ``NewtonVisualizerCfg`` framebuffer (window_width × window_height) for ``get_frame()``."""

_CARTPOLE_TILED_CAMERA_INTEGRATION_WH: tuple[int, int] = (500, 500)
"""Tiled camera per-env tile width/height (preset default is 100×100); keeps ``observation_space`` consistent."""

_START_BUFFER_STEPS = 5
"""Warmup steps before capturing the first motion-test frame."""

_CYCLE_FRAME_STEPS = 10
"""Steps to run for each motion, pause, and play segment."""

# Early vs late frame motion: void background stays similar; only count *strongly* differing pixels.
_FRAME_MOTION_CHANNEL_DIFF_THRESHOLD = 50
"""A pixel counts as differing if max(|ΔR|, |ΔG|, |ΔB|) >= this (0–255 space)."""

_FRAME_MOTION_MIN_DIFFERING_PIXELS = 100
"""Minimum number of such pixels between early and late frames (stale/frozen viz should be near zero)."""

_FRAME_MIN_CHANNEL_RANGE = 10
"""Minimum per-frame channel range to reject all-one-color images."""

_VIS_LOGGER_PREFIXES = (
    "isaaclab.visualizers",
    "isaaclab_visualizers",
    "isaaclab.sim.simulation_context",
)

_WRITE_VIS_DEBUG_FRAMES = False
"""Whether to emit visualizer debug PNGs during integration tests."""

_VIS_DEBUG_IMAGE_DIR = Path("logs/visualizer_debug")
"""Directory for opt-in visualizer debug images emitted by integration tests."""


def _logger_name_matches_visualizer_scope(logger_name: str) -> bool:
    """Return True if *logger_name* is a visualizer / SimulationContext visualizer path."""
    return any(logger_name.startswith(prefix) for prefix in _VIS_LOGGER_PREFIXES)


def _assert_no_visualizer_log_issues(caplog: pytest.LogCaptureFixture, *, fail_on_warnings: bool | None = None) -> None:
    """Fail if captured records include ERROR/CRITICAL (always) or WARNING (if *fail_on_warnings*).

    *fail_on_warnings* defaults to :data:`ASSERT_VISUALIZER_WARNINGS`.
    """
    if fail_on_warnings is None:
        fail_on_warnings = ASSERT_VISUALIZER_WARNINGS

    error_logs = [
        r for r in caplog.records if r.levelno >= logging.ERROR and _logger_name_matches_visualizer_scope(r.name)
    ]
    assert not error_logs, "Visualizer-related error logs: " + "; ".join(
        f"{r.name}: {r.getMessage()}" for r in error_logs
    )

    if fail_on_warnings:
        warning_logs = [
            r for r in caplog.records if r.levelno == logging.WARNING and _logger_name_matches_visualizer_scope(r.name)
        ]
        assert not warning_logs, "Visualizer-related warning logs: " + "; ".join(
            f"{r.name}: {r.getMessage()}" for r in warning_logs
        )


def _configure_sim_for_visualizer_test(env: CartpoleCameraEnv) -> None:
    """Settings used by the previous smoke tests; keep RTX sensors enabled for camera paths."""
    env.sim.set_setting("/isaaclab/render/rtx_sensors", True)
    env.sim._app_control_on_stop_handle = None  # type: ignore[attr-defined]


def _find_free_tcp_port(host: str = "127.0.0.1") -> int:
    """Ask OS for a currently free local TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _allocate_rerun_test_ports(host: str = "127.0.0.1") -> tuple[int, int]:
    """Allocate distinct free ports for rerun web and gRPC endpoints."""
    grpc_port = _find_free_tcp_port(host)
    web_port = _find_free_tcp_port(host)
    while web_port == grpc_port:
        web_port = _find_free_tcp_port(host)
    return web_port, grpc_port


def _cartpole_integration_visualizer_camera_kwargs() -> dict[str, tuple[float, float, float]]:
    """Eye/lookat for all :class:`~isaaclab.visualizers.visualizer_cfg.VisualizerCfg` subclasses in these tests."""
    return {
        "eye": _CARTPOLE_INTEGRATION_VISUALIZER_EYE,
        "lookat": _CARTPOLE_INTEGRATION_VISUALIZER_LOOKAT,
    }


def _get_visualizer_cfg(visualizer_kind: str):
    """Return (visualizer_cfg, expected_visualizer_cls) for the given visualizer kind."""
    cam = _cartpole_integration_visualizer_camera_kwargs()
    if visualizer_kind == "newton":
        __import__("newton")
        nw, nh = _CARTPOLE_NEWTON_INTEGRATION_WINDOW_SIZE
        return (
            NewtonVisualizerCfg(
                headless=True,
                window_width=nw,
                window_height=nh,
                randomly_sample_visible_envs=False,
                **cam,
            ),
            NewtonVisualizer,
        )
    if visualizer_kind == "viser":
        __import__("newton")
        __import__("viser")
        from isaaclab_visualizers.viser import ViserVisualizer, ViserVisualizerCfg

        port = _find_free_tcp_port(host="127.0.0.1")
        return (
            ViserVisualizerCfg(open_browser=False, port=port, randomly_sample_visible_envs=False, **cam),
            ViserVisualizer,
        )
    if visualizer_kind == "rerun":
        __import__("newton")
        __import__("rerun")
        from isaaclab_visualizers.rerun import RerunVisualizer, RerunVisualizerCfg

        web_port, grpc_port = _allocate_rerun_test_ports(host="127.0.0.1")
        return (
            RerunVisualizerCfg(
                bind_address="127.0.0.1",
                open_browser=False,
                web_port=web_port,
                grpc_port=grpc_port,
                randomly_sample_visible_envs=False,
                **cam,
            ),
            RerunVisualizer,
        )
    return KitVisualizerCfg(randomly_sample_visible_envs=False, **cam), KitVisualizer


def _get_physics_cfg(backend_kind: str):
    """Return physics config and expected backend substring for the given backend kind."""
    if backend_kind == "physx":
        __import__("isaaclab_physx")
        preset = CartpolePhysicsCfg()
        physics_cfg = getattr(preset, "physx", None)
        if physics_cfg is None:
            from isaaclab_physx.physics import PhysxCfg

            physics_cfg = PhysxCfg()
        return physics_cfg, "physx"
    if backend_kind == "newton":
        __import__("newton")
        __import__("isaaclab_newton")
        preset = CartpolePhysicsCfg()
        physics_cfg = getattr(preset, "newton_mjwarp", None)
        if physics_cfg is None:
            from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

            physics_cfg = NewtonCfg(
                solver_cfg=MJWarpSolverCfg(
                    njmax=5,
                    nconmax=3,
                    cone="pyramidal",
                    impratio=1,
                    integrator="implicitfast",
                ),
                num_substeps=1,
                debug_mode=False,
                use_cuda_graph=True,
            )
        return physics_cfg, "newton"
    raise ValueError(f"Unknown backend: {backend_kind!r}")


def _assert_non_flat_tensor(image_tensor: torch.Tensor) -> None:
    """Assert every camera-like tensor image has non-flat content."""
    assert isinstance(image_tensor, torch.Tensor), f"Expected torch.Tensor, got {type(image_tensor)!r}"
    assert image_tensor.numel() > 0, "Image tensor is empty."
    finite_tensor = torch.where(torch.isfinite(image_tensor), image_tensor, torch.zeros_like(image_tensor))
    finite_tensor = finite_tensor.float()
    if finite_tensor.numel() > 0 and float(finite_tensor.max().item()) <= 1.0 + 1e-6:
        finite_tensor = finite_tensor * 255.0
    flat = finite_tensor.reshape(finite_tensor.shape[0], -1)
    channel_ranges = flat.max(dim=1).values - flat.min(dim=1).values
    min_range = float(channel_ranges.min().item())
    assert min_range >= _FRAME_MIN_CHANNEL_RANGE, (
        f"Rendered frame appears flat / single-color (min channel range {min_range:.3f} < {_FRAME_MIN_CHANNEL_RANGE})."
    )


def _frame_to_numpy(frame) -> np.ndarray:
    """Convert viewer ``get_frame()`` output (numpy, torch, or Warp array) to host ``numpy.ndarray``.

    ``np.asarray(wp.array)`` is unsafe: NumPy can trigger Warp indexing that raises at dimension edges.
    """
    if isinstance(frame, np.ndarray):
        return frame
    if isinstance(frame, torch.Tensor):
        return frame.detach().cpu().numpy()
    if isinstance(frame, wp.array):
        return wp.to_torch(frame).detach().cpu().numpy()
    return np.asarray(frame)


def _assert_non_flat_frame_array(frame) -> None:
    """Assert viewer-captured frame has non-flat content."""
    frame_arr = _frame_to_numpy(frame)
    assert frame_arr.size > 0, "Viewer returned an empty frame."
    if frame_arr.ndim != 2:
        assert frame_arr.shape[-1] >= 3, f"Expected at least 3 channels, got shape {frame_arr.shape}."
    rgb = _frame_rgb_255_space(frame)
    channel_range = float(np.max(rgb) - np.min(rgb))
    assert channel_range >= _FRAME_MIN_CHANNEL_RANGE, (
        f"Viewer frame appears flat / single-color (channel range {channel_range:.3f} < {_FRAME_MIN_CHANNEL_RANGE})."
    )


def _frame_rgb_255_space(frame) -> np.ndarray:
    """Return HxWx3 float in ~0–255 space for per-channel differencing."""
    arr = _frame_to_numpy(frame)
    if arr.ndim == 2:
        rgb = np.stack([arr, arr, arr], axis=-1)
    else:
        rgb = arr[..., :3]
    rgb = np.asarray(rgb, dtype=np.float64)
    # Normalized HDR buffers: scale so threshold matches (0,255) semantics.
    if rgb.size > 0 and float(np.nanmax(rgb)) <= 1.0 + 1e-6:
        rgb = rgb * 255.0
    return rgb


def _save_visualizer_debug_frame(frame, *, viz_kind: str, physics_kind: str, phase: str, frame_idx: int) -> None:
    """Save a visualizer frame to a clearly named PNG for pause/motion debugging."""
    if not _WRITE_VIS_DEBUG_FRAMES:
        return
    from PIL import Image

    rgb = np.clip(_frame_rgb_255_space(frame), 0, 255).astype(np.uint8)
    _VIS_DEBUG_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    file_name = f"{viz_kind}_viz_{physics_kind}_physics_{phase}_frame_{frame_idx:03d}.png"
    Image.fromarray(rgb).save(_VIS_DEBUG_IMAGE_DIR / file_name)


def _save_visualizer_debug_delta(
    frame_a, frame_b, *, viz_kind: str, physics_kind: str, phase: str, frame_idx: int
) -> None:
    """Save an amplified absolute-difference image for a start/end frame pair."""
    if not _WRITE_VIS_DEBUG_FRAMES:
        return
    from PIL import Image

    a = _frame_rgb_255_space(frame_a)
    b = _frame_rgb_255_space(frame_b)
    assert a.shape == b.shape, f"Frame shape mismatch for delta image: {a.shape} vs {b.shape}."
    delta = np.clip(np.abs(a - b) * 4.0, 0, 255).astype(np.uint8)
    _VIS_DEBUG_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    file_name = f"{viz_kind}_viz_{physics_kind}_physics_{phase}_delta_frame_{frame_idx:03d}.png"
    Image.fromarray(delta).save(_VIS_DEBUG_IMAGE_DIR / file_name)


def _clear_visualizer_debug_frames(viz_kind: str, physics_kind: str) -> None:
    if not _WRITE_VIS_DEBUG_FRAMES:
        return
    _VIS_DEBUG_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for path in _VIS_DEBUG_IMAGE_DIR.glob(f"{viz_kind}_viz_{physics_kind}_physics_*.png"):
        path.unlink()


def _count_significantly_differing_pixels(
    frame_a,
    frame_b,
    *,
    channel_diff_threshold: float = _FRAME_MOTION_CHANNEL_DIFF_THRESHOLD,
) -> int:
    """Count pixels where max(|ΔR|, |ΔG|, |ΔB|) >= *channel_diff_threshold* (0–255 space)."""
    a = _frame_rgb_255_space(frame_a)
    b = _frame_rgb_255_space(frame_b)
    assert a.shape == b.shape, f"Frame shape mismatch for motion check: {a.shape} vs {b.shape}."
    per_pixel_max = np.max(np.abs(a - b), axis=-1)
    return int(np.count_nonzero(per_pixel_max >= channel_diff_threshold))


def _assert_frames_remain_stable(frame_a, frame_b, *, max_differing_pixels: int = 10) -> None:
    """Assert two viewport frames are effectively unchanged while simulation is paused."""
    n_diff = _count_significantly_differing_pixels(frame_a, frame_b)
    assert n_diff <= max_differing_pixels, (
        f"Paused viewport frames changed unexpectedly ({n_diff} > {max_differing_pixels} differing pixels)."
    )


def _assert_frames_differ(
    frame_a,
    frame_b,
    *,
    channel_diff_threshold: float = _FRAME_MOTION_CHANNEL_DIFF_THRESHOLD,
    min_differing_pixels: int = _FRAME_MOTION_MIN_DIFFERING_PIXELS,
) -> None:
    """Fail if two frames lack enough strongly differing pixels (stale/frozen bodies)."""
    n_diff = _count_significantly_differing_pixels(frame_a, frame_b, channel_diff_threshold=channel_diff_threshold)
    assert n_diff >= min_differing_pixels, (
        "Viewport frame pair has too few strongly differing pixels "
        f"({n_diff} < {min_differing_pixels}; threshold per channel={channel_diff_threshold} in 0–255 space). "
        "Possible frozen or stale robot visualization."
    )


def _assert_frame_sequence_has_motion(frames: list) -> None:
    """Assert a captured frame sequence is visible and changes over time."""
    _assert_non_flat_frame_array(frames[-1])
    _assert_frames_differ(frames[0], frames[-1])


def _step_until_non_flat_camera(env, actions: torch.Tensor, *, max_steps: int = _MAX_FRAME_CHECK_STEPS) -> None:
    """Step env until the env's tiled camera RGB tensor is non-flat, bounded by *max_steps*."""
    last_rgb = None
    for _ in range(max_steps):
        env.step(action=actions)
        rgb = env._tiled_camera.data.output.get("rgb")
        if rgb is None:
            rgb = env._tiled_camera.data.output[env.cfg.tiled_camera.data_types[0]]
        last_rgb = rgb
        try:
            _assert_non_flat_tensor(rgb)
            return
        except AssertionError:
            continue
    _assert_non_flat_tensor(last_rgb)


def _select_newton_training_control_button(viewer, target_label: str) -> None:
    """Trigger one Newton visualizer training-control button by label."""

    class _FakeImgui:
        def separator(self):
            pass

        def text(self, _text):
            pass

        def button(self, label):
            return label == target_label

        def slider_int(self, _label, value, _min_value, _max_value, _format):
            return False, value

        def is_item_hovered(self):
            return False

        def set_tooltip(self, _text):
            pass

    viewer._render_training_controls(_FakeImgui())


def _select_newton_pause_simulation_button(viewer) -> None:
    """Trigger the Newton visualizer's Pause/Resume Simulation UI button."""
    label = "Resume Simulation" if viewer.is_training_paused() else "Pause Simulation"
    _select_newton_training_control_button(viewer, label)


def _select_newton_pause_rendering_button(viewer) -> None:
    """Trigger the Newton visualizer's Pause/Resume Rendering UI button."""
    label = "Resume Rendering" if viewer.is_rendering_paused() else "Pause Rendering"
    _select_newton_training_control_button(viewer, label)


def _run_newton_viewer_frame_motion_test(
    viewer,
    *,
    visualizer: NewtonVisualizer,
    step_hook,
    get_physics_step_count,
    physics_kind: str,
    viz_kind: str = "newton",
) -> None:
    """Check Newton viewer simulation pause, rendering pause, then resumed motion."""
    _clear_visualizer_debug_frames(viz_kind, physics_kind)
    for _ in range(_START_BUFFER_STEPS):
        step_hook()
    motion_start_frame = viewer.get_frame()
    _save_visualizer_debug_frame(
        motion_start_frame, viz_kind=viz_kind, physics_kind=physics_kind, phase="motion_start", frame_idx=0
    )
    for _ in range(_CYCLE_FRAME_STEPS):
        step_hook()
    motion_end_frame = viewer.get_frame()
    _save_visualizer_debug_frame(
        motion_end_frame, viz_kind=viz_kind, physics_kind=physics_kind, phase="motion_end", frame_idx=_CYCLE_FRAME_STEPS
    )
    _save_visualizer_debug_delta(
        motion_start_frame,
        motion_end_frame,
        viz_kind=viz_kind,
        physics_kind=physics_kind,
        phase="motion",
        frame_idx=_CYCLE_FRAME_STEPS,
    )
    _assert_non_flat_frame_array(motion_end_frame)
    _assert_frames_differ(motion_start_frame, motion_end_frame)

    _select_newton_pause_simulation_button(viewer)
    paused_start_frame = viewer.get_frame()
    _save_visualizer_debug_frame(
        paused_start_frame, viz_kind=viz_kind, physics_kind=physics_kind, phase="simulation_pause_start", frame_idx=0
    )
    for _ in range(_CYCLE_FRAME_STEPS):
        visualizer.step(0.0)
    paused_end_frame = viewer.get_frame()
    _save_visualizer_debug_frame(
        paused_end_frame,
        viz_kind=viz_kind,
        physics_kind=physics_kind,
        phase="simulation_pause_end",
        frame_idx=_CYCLE_FRAME_STEPS,
    )
    _save_visualizer_debug_delta(
        paused_start_frame,
        paused_end_frame,
        viz_kind=viz_kind,
        physics_kind=physics_kind,
        phase="simulation_pause",
        frame_idx=_CYCLE_FRAME_STEPS,
    )
    _assert_frames_remain_stable(paused_start_frame, paused_end_frame)

    _select_newton_pause_simulation_button(viewer)
    simulation_play_start_frame = viewer.get_frame()
    _save_visualizer_debug_frame(
        simulation_play_start_frame,
        viz_kind=viz_kind,
        physics_kind=physics_kind,
        phase="simulation_play_start",
        frame_idx=0,
    )
    for _ in range(_CYCLE_FRAME_STEPS):
        step_hook()
    simulation_play_end_frame = viewer.get_frame()
    _save_visualizer_debug_frame(
        simulation_play_end_frame,
        viz_kind=viz_kind,
        physics_kind=physics_kind,
        phase="simulation_play_end",
        frame_idx=_CYCLE_FRAME_STEPS,
    )
    _save_visualizer_debug_delta(
        simulation_play_start_frame,
        simulation_play_end_frame,
        viz_kind=viz_kind,
        physics_kind=physics_kind,
        phase="simulation_play",
        frame_idx=_CYCLE_FRAME_STEPS,
    )
    _assert_non_flat_frame_array(simulation_play_end_frame)
    _assert_frames_differ(simulation_play_start_frame, simulation_play_end_frame)

    _select_newton_pause_rendering_button(viewer)
    rendering_paused_start_frame = viewer.get_frame()
    _save_visualizer_debug_frame(
        rendering_paused_start_frame,
        viz_kind=viz_kind,
        physics_kind=physics_kind,
        phase="rendering_pause_start",
        frame_idx=0,
    )
    physics_step_before_render_pause = get_physics_step_count()
    for _ in range(_CYCLE_FRAME_STEPS):
        step_hook()
    rendering_paused_end_frame = viewer.get_frame()
    _save_visualizer_debug_frame(
        rendering_paused_end_frame,
        viz_kind=viz_kind,
        physics_kind=physics_kind,
        phase="rendering_pause_end",
        frame_idx=_CYCLE_FRAME_STEPS,
    )
    _save_visualizer_debug_delta(
        rendering_paused_start_frame,
        rendering_paused_end_frame,
        viz_kind=viz_kind,
        physics_kind=physics_kind,
        phase="rendering_pause",
        frame_idx=_CYCLE_FRAME_STEPS,
    )
    _assert_frames_remain_stable(rendering_paused_start_frame, rendering_paused_end_frame)
    assert get_physics_step_count() > physics_step_before_render_pause, (
        "Physics did not advance while Newton visualizer rendering was paused."
    )

    _select_newton_pause_rendering_button(viewer)
    rendering_play_start_frame = viewer.get_frame()
    _save_visualizer_debug_frame(
        rendering_play_start_frame,
        viz_kind=viz_kind,
        physics_kind=physics_kind,
        phase="rendering_play_start",
        frame_idx=0,
    )
    for _ in range(_CYCLE_FRAME_STEPS):
        step_hook()
    rendering_play_end_frame = viewer.get_frame()
    _save_visualizer_debug_frame(
        rendering_play_end_frame,
        viz_kind=viz_kind,
        physics_kind=physics_kind,
        phase="rendering_play_end",
        frame_idx=_CYCLE_FRAME_STEPS,
    )
    _save_visualizer_debug_delta(
        rendering_play_start_frame,
        rendering_play_end_frame,
        viz_kind=viz_kind,
        physics_kind=physics_kind,
        phase="rendering_play",
        frame_idx=_CYCLE_FRAME_STEPS,
    )
    _assert_non_flat_frame_array(rendering_play_end_frame)
    _assert_frames_differ(rendering_play_start_frame, rendering_play_end_frame)


def _step_env_without_frame_check(env, actions: torch.Tensor, *, max_steps: int = _MAX_FRAME_CHECK_STEPS) -> None:
    """Step the env to exercise visualizers that do not implement ``get_frame`` (e.g. Rerun, Viser)."""
    for _ in range(max_steps):
        env.step(action=actions)


def _build_rgb_annotator_for_camera(
    camera_path: str,
    *,
    resolution: tuple[int, int] | None = None,
):
    """Create CPU RGB annotator attached to a camera render product."""
    import omni.replicator.core as rep

    if resolution is None:
        resolution = _CARTPOLE_KIT_INTEGRATION_RENDER_RESOLUTION
    render_product = rep.create.render_product(camera_path, resolution=resolution)
    annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    annotator.attach([render_product])
    return annotator, render_product


def _annotator_rgb_to_numpy(rgb_data) -> np.ndarray:
    """Convert replicator annotator output to HxWx3 uint8 numpy array."""
    rgb_array = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
    if rgb_array.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return rgb_array[:, :, :3]


def _run_kit_viewport_frame_motion_test(
    env,
    kit_visualizer: KitVisualizer,
    *,
    physics_kind: str,
    viz_kind: str = "kit",
) -> None:
    """Check Kit viewport motion, timeline pause freeze, then resumed motion."""
    _clear_visualizer_debug_frames(viz_kind, physics_kind)
    camera_path = getattr(kit_visualizer, "_controlled_camera_path", None)
    assert camera_path, "Kit visualizer does not expose a controlled viewport camera path."

    annotator = None
    render_product = None
    try:
        annotator, render_product = _build_rgb_annotator_for_camera(camera_path)
        actions = torch.zeros((env.num_envs, env.action_space.shape[-1]), device=env.device)
        for _ in range(_START_BUFFER_STEPS):
            env.step(action=actions)
        motion_start_frame = _capture_kit_viewport_rgb(annotator)
        _save_visualizer_debug_frame(
            motion_start_frame, viz_kind=viz_kind, physics_kind=physics_kind, phase="motion_start", frame_idx=0
        )
        for _ in range(_CYCLE_FRAME_STEPS):
            env.step(action=actions)
        motion_end_frame = _capture_kit_viewport_rgb(annotator)
        _save_visualizer_debug_frame(
            motion_end_frame,
            viz_kind=viz_kind,
            physics_kind=physics_kind,
            phase="motion_end",
            frame_idx=_CYCLE_FRAME_STEPS,
        )
        _save_visualizer_debug_delta(
            motion_start_frame,
            motion_end_frame,
            viz_kind=viz_kind,
            physics_kind=physics_kind,
            phase="motion",
            frame_idx=_CYCLE_FRAME_STEPS,
        )
        _assert_non_flat_frame_array(motion_end_frame)
        _assert_frames_differ(motion_start_frame, motion_end_frame)

        timeline = omni.timeline.get_timeline_interface()
        timeline.pause()
        simulation_app.update()
        paused_start_frame = _capture_kit_viewport_rgb(annotator)
        _save_visualizer_debug_frame(
            paused_start_frame, viz_kind=viz_kind, physics_kind=physics_kind, phase="timeline_pause_start", frame_idx=0
        )
        try:
            for _ in range(_CYCLE_FRAME_STEPS):
                env.sim.render()
            paused_end_frame = _capture_kit_viewport_rgb(annotator)
            _save_visualizer_debug_frame(
                paused_end_frame,
                viz_kind=viz_kind,
                physics_kind=physics_kind,
                phase="timeline_pause_end",
                frame_idx=_CYCLE_FRAME_STEPS,
            )
            _save_visualizer_debug_delta(
                paused_start_frame,
                paused_end_frame,
                viz_kind=viz_kind,
                physics_kind=physics_kind,
                phase="timeline_pause",
                frame_idx=_CYCLE_FRAME_STEPS,
            )
            _assert_frames_remain_stable(paused_start_frame, paused_end_frame)
        finally:
            timeline.play()
            simulation_app.update()

        play_start_frame = _capture_kit_viewport_rgb(annotator)
        _save_visualizer_debug_frame(
            play_start_frame, viz_kind=viz_kind, physics_kind=physics_kind, phase="timeline_play_start", frame_idx=0
        )
        for _ in range(_CYCLE_FRAME_STEPS):
            env.step(action=actions)
        play_end_frame = _capture_kit_viewport_rgb(annotator)
        _save_visualizer_debug_frame(
            play_end_frame,
            viz_kind=viz_kind,
            physics_kind=physics_kind,
            phase="timeline_play_end",
            frame_idx=_CYCLE_FRAME_STEPS,
        )
        _save_visualizer_debug_delta(
            play_start_frame,
            play_end_frame,
            viz_kind=viz_kind,
            physics_kind=physics_kind,
            phase="timeline_play",
            frame_idx=_CYCLE_FRAME_STEPS,
        )
        _assert_non_flat_frame_array(play_end_frame)
        _assert_frames_differ(play_start_frame, play_end_frame)
    finally:
        if annotator is not None and render_product is not None:
            with contextlib.suppress(Exception):
                annotator.detach([render_product])


def _capture_kit_viewport_rgb(annotator) -> np.ndarray:
    frame = _annotator_rgb_to_numpy(annotator.get_data())
    for _ in range(5):
        if frame.shape[:2] != (1, 1) or np.count_nonzero(frame) > 0:
            return frame
        simulation_app.update()
        frame = _annotator_rgb_to_numpy(annotator.get_data())
    return frame


def _make_cartpole_camera_env(visualizer_kind: str, backend_kind: str) -> CartpoleCameraEnv:
    """Create cartpole camera env configured with selected visualizer and physics backend."""
    env_cfg_root = CartpoleCameraPresetsEnvCfg()
    env_cfg = getattr(env_cfg_root, "default", None)
    if env_cfg is None:
        env_cfg = getattr(type(env_cfg_root), "default", None)
    if env_cfg is None:
        raise RuntimeError(
            "CartpoleCameraPresetsEnvCfg does not expose a 'default' preset config. "
            f"Available attributes: {sorted(vars(env_cfg_root).keys())}"
        )
    env_cfg = copy.deepcopy(env_cfg)
    env_cfg.scene.num_envs = _CARTPOLE_INTEGRATION_NUM_ENVS
    env_cfg.viewer.eye = _CARTPOLE_INTEGRATION_VISUALIZER_EYE
    env_cfg.viewer.lookat = _CARTPOLE_INTEGRATION_VISUALIZER_LOOKAT
    tw, th = _CARTPOLE_TILED_CAMERA_INTEGRATION_WH
    env_cfg.tiled_camera.width = tw
    env_cfg.tiled_camera.height = th
    if isinstance(env_cfg.observation_space, list) and len(env_cfg.observation_space) >= 3:
        env_cfg.observation_space = [th, tw, env_cfg.observation_space[2]]
    env_cfg.seed = None
    env_cfg.sim.physics, _ = _get_physics_cfg(backend_kind)
    visualizer_cfg, _ = _get_visualizer_cfg(visualizer_kind)
    env_cfg.sim.visualizer_cfgs = visualizer_cfg
    return CartpoleCameraEnv(env_cfg)


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize(
    "backend_kind",
    ["physx", "newton"],
)
def test_cartpole_kit_visualizer_replicator_viewport_rgb_motion(
    backend_kind: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Kit + cartpole: Replicator RGB on viewport camera; frames are non-flat and motion changes pixels; logs."""
    env = None
    try:
        sim_utils.create_new_stage()
        env = _make_cartpole_camera_env(visualizer_kind="kit", backend_kind=backend_kind)
        _configure_sim_for_visualizer_test(env)
        with caplog.at_level(logging.WARNING):
            env.reset()
            kit_visualizers = [viz for viz in env.sim.visualizers if isinstance(viz, KitVisualizer)]
            assert kit_visualizers, "Expected an initialized Kit visualizer."
            _run_kit_viewport_frame_motion_test(env, kit_visualizers[0], physics_kind=backend_kind)
        _assert_no_visualizer_log_issues(caplog)
    finally:
        if env is not None:
            env.close()
        else:
            SimulationContext.clear_instance()


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("backend_kind", ["physx", "newton"])
def test_cartpole_newton_visualizer_tiled_camera_rgb_non_flat(
    backend_kind: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Newton visualizer + cartpole: env tiled-camera RGB becomes non-flat within a few steps; clean logs."""
    env = None
    try:
        sim_utils.create_new_stage()
        env = _make_cartpole_camera_env(visualizer_kind="newton", backend_kind=backend_kind)
        _configure_sim_for_visualizer_test(env)
        with caplog.at_level(logging.WARNING):
            env.reset()
            actions = torch.zeros((env.num_envs, env.action_space.shape[-1]), device=env.device)
            _step_until_non_flat_camera(env, actions, max_steps=_MAX_FRAME_CHECK_STEPS)
        _assert_no_visualizer_log_issues(caplog)
    finally:
        if env is not None:
            env.close()
        else:
            SimulationContext.clear_instance()


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("backend_kind", ["physx", "newton"])
def test_cartpole_newton_visualizer_viewergl_rgb_motion(backend_kind: str, caplog: pytest.LogCaptureFixture) -> None:
    """Newton GL (``ViewerGL.get_frame``): frames are non-flat and motion changes pixels; logs."""
    env = None
    try:
        sim_utils.create_new_stage()
        env = _make_cartpole_camera_env(visualizer_kind="newton", backend_kind=backend_kind)
        _configure_sim_for_visualizer_test(env)
        with caplog.at_level(logging.WARNING):
            env.reset()
            actions = torch.zeros((env.num_envs, env.action_space.shape[-1]), device=env.device)
            newton_visualizers = [viz for viz in env.sim.visualizers if isinstance(viz, NewtonVisualizer)]
            assert newton_visualizers, "Expected an initialized Newton visualizer."
            viewer = getattr(newton_visualizers[0], "_viewer", None)
            assert viewer is not None, "Newton viewer was not created."

            def _step_env() -> None:
                env.step(action=actions)

            _run_newton_viewer_frame_motion_test(
                viewer,
                visualizer=newton_visualizers[0],
                step_hook=_step_env,
                get_physics_step_count=lambda: env.sim._physics_step_count,
                physics_kind=backend_kind,
            )
        _assert_no_visualizer_log_issues(caplog)
    finally:
        if env is not None:
            env.close()
        else:
            SimulationContext.clear_instance()


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("backend_kind", ["physx", "newton"])
def test_cartpole_rerun_visualizer_smoke_steps_and_logs(backend_kind: str, caplog: pytest.LogCaptureFixture) -> None:
    """Rerun + cartpole: visualizer and viewer initialize; env steps exercise the pipeline; clean logs.

    Rerun does not expose a per-frame RGB API like ``get_frame``, so we do not assert pixel content.
    """
    env = None
    try:
        sim_utils.create_new_stage()
        env = _make_cartpole_camera_env(visualizer_kind="rerun", backend_kind=backend_kind)
        _configure_sim_for_visualizer_test(env)
        with caplog.at_level(logging.WARNING):
            env.reset()
            actions = torch.zeros((env.num_envs, env.action_space.shape[-1]), device=env.device)
            from isaaclab_visualizers.rerun import RerunVisualizer

            rerun_visualizers = [viz for viz in env.sim.visualizers if isinstance(viz, RerunVisualizer)]
            assert rerun_visualizers, "Expected an initialized Rerun visualizer."
            assert getattr(rerun_visualizers[0], "_viewer", None) is not None, "Rerun viewer was not created."
            _step_env_without_frame_check(env, actions, max_steps=_MAX_FRAME_CHECK_STEPS)
        _assert_no_visualizer_log_issues(caplog)
    finally:
        if env is not None:
            env.close()
        else:
            SimulationContext.clear_instance()


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("backend_kind", ["physx", "newton"])
def test_cartpole_viser_visualizer_smoke_steps_and_logs(backend_kind: str, caplog: pytest.LogCaptureFixture) -> None:
    """Viser + cartpole: visualizer and viewer initialize; env steps exercise the pipeline; clean logs.

    No per-frame RGB assertion (Viser does not mirror the Newton ``get_frame`` path used elsewhere).
    """
    env = None
    try:
        sim_utils.create_new_stage()
        env = _make_cartpole_camera_env(visualizer_kind="viser", backend_kind=backend_kind)
        _configure_sim_for_visualizer_test(env)
        with caplog.at_level(logging.WARNING):
            env.reset()
            actions = torch.zeros((env.num_envs, env.action_space.shape[-1]), device=env.device)
            from isaaclab_visualizers.viser import ViserVisualizer

            viser_visualizers = [viz for viz in env.sim.visualizers if isinstance(viz, ViserVisualizer)]
            assert viser_visualizers, "Expected an initialized Viser visualizer."
            assert getattr(viser_visualizers[0], "_viewer", None) is not None, "Viser viewer was not created."
            _step_env_without_frame_check(env, actions, max_steps=_MAX_FRAME_CHECK_STEPS)
        _assert_no_visualizer_log_issues(caplog)
    finally:
        if env is not None:
            env.close()
        else:
            SimulationContext.clear_instance()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--maxfail=1"])
