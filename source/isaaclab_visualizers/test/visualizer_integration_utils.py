# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for Cartpole visualizer integration tests.

The suite covers four visualizers: Kit, Newton, Rerun, and Viser. All visualizers
must initialize and step without visualizer-scoped log errors on both physics backends.

Kit and Newton also expose image-producing paths, so they get stronger checks:
- frames are non-flat
- frames change while simulation is playing
- frames remain stable while rendering or simulation is paused
- frames change again after play resumes

Newton has separate rendering-pause and simulation-pause controls, so those tests
also verify that physics continues during rendering pause and stays frozen during
simulation pause.
"""

from __future__ import annotations

import contextlib
import copy
import gc
import logging
import os
import re
import socket
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_visualizers.kit import KitVisualizer, KitVisualizerCfg
from isaaclab_visualizers.newton import NewtonVisualizer, NewtonVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.app import AppLauncher
from isaaclab.envs.utils.camera_view import camera_rgb_batch, compose_rgb_grid_tensor
from isaaclab.sim import SimulationContext

from isaaclab_tasks.core.cartpole.cartpole_direct_camera_env import CartpoleCameraEnv
from isaaclab_tasks.core.cartpole.cartpole_direct_camera_env_cfg import CartpoleCameraEnvCfg
from isaaclab_tasks.core.cartpole.cartpole_manager_env_cfg import CartpolePhysicsCfg

# TODO: Several test cases currently show flakiness with frozen bodies. Remove the test-level retry once fixed.

# Debugging mode configs.

_WRITE_VIS_DEBUG_FRAMES = False
"""Whether to emit visualizer debug PNGs during integration tests."""

_VIS_DEBUG_IMAGE_DIR = Path("logs/viz_integration_captures")
"""Directory for opt-in visualizer debug images emitted by integration tests."""


# When True, tests also fail on WARNING-level records from visualizer-related loggers.
ASSERT_VISUALIZER_WARNINGS = False

_NEWTON_IMGUI_BUNDLE_PRINT_WARNING = "Warning: imgui_bundle not found"
_MAX_FRAME_CHECK_STEPS = 5
"""Steps for Rerun / Viser smoke tests."""

_CARTPOLE_INTEGRATION_NUM_ENVS = 1
"""Vectorized env count for cartpole + visualizer integration tests."""

_CARTPOLE_TILED_CAMERA_INTEGRATION_NUM_ENVS = 4
"""Vectorized env count for generated visualizer tiled-camera integration tests."""

_CARTPOLE_INTEGRATION_VISUALIZER_EYE: tuple[float, float, float] = (2.25, 0.0, 3.5)
"""Passed to :class:`~isaaclab.visualizers.visualizer_cfg.VisualizerCfg` subclasses (``eye``)."""

_CARTPOLE_INTEGRATION_VISUALIZER_LOOKAT: tuple[float, float, float] = (0.0, 0.0, 2.25)
"""Passed to visualizer cfgs (``lookat``); also applied to :class:`~isaaclab.envs.common.ViewerCfg` for the env."""

_CARTPOLE_INTEGRATION_TILED_CAMERA_EYE_OFFSET: tuple[float, float, float] = tuple(
    eye - lookat for eye, lookat in zip(_CARTPOLE_INTEGRATION_VISUALIZER_EYE, _CARTPOLE_INTEGRATION_VISUALIZER_LOOKAT)
)
"""Generated tiled-camera target-relative eye offset matching the shared visualizer viewing direction."""

# Resolution overrides for this test module (cartpole preset defaults: tiled camera 100×100; Kit helper was 320×240).
_CARTPOLE_KIT_INTEGRATION_RENDER_RESOLUTION: tuple[int, int] = (400, 400)
"""Kit: Replicator ``render_product`` (width, height) for viewport RGB in the motion check."""

_CARTPOLE_NEWTON_INTEGRATION_WINDOW_SIZE: tuple[int, int] = (400, 400)
"""Newton: ``NewtonVisualizerCfg`` framebuffer (window_width × window_height) for ``get_frame()``."""

_CARTPOLE_TILED_CAMERA_INTEGRATION_WH: tuple[int, int] = (400, 400)
"""Tiled camera per-env tile width/height (preset default is 100×100); keeps ``observation_space`` consistent."""

_CARTPOLE_VISUALIZER_TILED_CAMERA_NUM_TILES = 4
"""Number of generated visualizer camera tiles exercised by tiled-camera integration tests."""

_CARTPOLE_VISUALIZER_TILED_CAMERA_TARGET_PRIM_PATH = "/World/envs/*/Robot"
"""Cartpole articulation root prim followed by generated visualizer tiled cameras."""

_START_BUFFER_STEPS = 20
"""Warmup physics steps before capturing the first debug frame."""

_KIT_RTX_RENDER_PRODUCT_WARMUP_STEPS = 20
"""Render/app updates after creating a Kit RTX render product before sampling RGB."""

_NEWTON_VIEWER_WARMUP_FRAMES = 20
"""Viewer-only updates after physics warmup before sampling Newton RGB."""

_VISUALIZER_STARTUP_DRAIN_UPDATES = 5
"""Kit app updates before each flaky retry creates a fresh stage/env."""

_VISUALIZER_SHUTDOWN_DRAIN_UPDATES = 10
"""Kit app updates after each flaky retry closes visualizer resources."""

_KIT_APP_DRAIN_SLEEP_SECONDS = 0.01
"""Short sleep between app updates while draining startup/shutdown work."""

PLAY_VIZ_N_STEP = 20
"""Steps to run for each motion or resumed-play segment."""

PAUSE_VIZ_N_STEP = 5
"""Steps to run for each paused visualization segment."""

# Early vs late frame motion: void background stays similar; only count *strongly* differing pixels.
_FRAME_MOTION_CHANNEL_DIFF_THRESHOLD = 50
"""A pixel counts as differing if max(|ΔR|, |ΔG|, |ΔB|) >= this (0–255 space)."""

_FRAME_MOTION_MIN_DIFFERING_PIXELS = 100
"""Minimum number of such pixels between early and late frames (stale/frozen viz should be near zero)."""

_TILED_CAMERA_MOTION_CHANNEL_DIFF_THRESHOLD = 5
"""Lower per-channel threshold for Cartpole's fixed tiled camera view, where motion is more subtle."""

_TILED_CAMERA_MOTION_MIN_DIFFERING_PIXELS = 25
"""Minimum differing pixels for tiled camera motion checks."""

_FRAME_MIN_CHANNEL_RANGE = 10
"""Minimum per-frame channel range to reject all-one-color images."""

_BODY_STATE_STABLE_MAX_DELTA = 1.0e-6
"""Maximum body-state delta allowed while simulation is paused."""

_BODY_STATE_MOTION_MIN_DELTA = 1.0e-5
"""Minimum body-state delta expected while physics continues to advance."""

_VIS_LOGGER_PREFIXES = (
    "isaaclab.visualizers",
    "isaaclab_visualizers",
    "isaaclab.sim.simulation_context",
)

_PYTEST_CURRENT_TEST_SUFFIX_PATTERN = re.compile(r"\s+\((setup|call|teardown)\)$")
_VIS_DEBUG_TEST_ID_OVERRIDE_ENV = "ISAACLAB_VISUALIZER_DEBUG_TEST_ID"

_DEBUG_TEST_DIR_PREFIXES = {
    "test_cartpole_env_visualizers_motion_with_play_pause_physx": "visualizers_physx",
    "test_cartpole_env_visualizers_motion_with_play_pause_newton": "visualizers_newton",
    "test_visualizer_tiled_integration_physx": "visualizers_physx",
    "test_visualizer_tiled_integration_newton": "visualizers_newton",
}

_DEBUG_TEST_TILED_SUFFIXES = {
    "test_visualizer_tiled_integration_physx",
    "test_visualizer_tiled_integration_newton",
}


_BACKEND_DISPLAY_NAMES = {
    "physx": "PhysX",
    "newton": "Newton MJWarp",
}

_VISUALIZER_DISPLAY_NAMES = {
    "kit": "Kit Visualizer",
    "newton": "Newton Visualizer",
    "rerun": "Rerun Visualizer",
    "viser": "Viser Visualizer",
}

_SIMULATION_APP = None


def set_visualizer_integration_simulation_app(simulation_app) -> None:
    """Register the Kit app launched by a backend-specific test module."""
    global _SIMULATION_APP
    _SIMULATION_APP = simulation_app


def _visualizer_case_label(viz_kind: str, physics_kind: str) -> str:
    visualizer = _VISUALIZER_DISPLAY_NAMES.get(viz_kind, f"{viz_kind.title()} Visualizer")
    backend = _BACKEND_DISPLAY_NAMES.get(physics_kind, physics_kind)
    return f"{visualizer} on {backend}"


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


def assert_no_newton_imgui_bundle_warning(capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture) -> None:
    """Fail when Newton reports that its imgui HUD dependency is missing."""
    captured = capsys.readouterr()
    captured_output = captured.out + captured.err
    printed_warning = _NEWTON_IMGUI_BUNDLE_PRINT_WARNING in captured_output
    logged_warnings = [record for record in caplog.records if _NEWTON_IMGUI_BUNDLE_PRINT_WARNING in record.getMessage()]
    assert not printed_warning and not logged_warnings, (
        "Newton viewer reported that imgui_bundle could not be imported, which disables HUD controls. "
        f"Captured output: {captured_output!r}. "
        "Captured logs: " + "; ".join(f"{record.name}: {record.getMessage()}" for record in logged_warnings)
    )


def _configure_sim_for_visualizer_test(env: CartpoleCameraEnv) -> None:
    """Settings used by the previous smoke tests; keep RTX sensors enabled for camera paths."""
    AppLauncher.apply_rtx_determinism_settings()
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


def _get_visualizer_cfg(visualizer_kind: str, *, tiled_camera: bool = False):
    """Return (visualizer_cfg, expected_visualizer_cls) for the given visualizer kind."""
    cam = _cartpole_integration_visualizer_camera_kwargs()
    tiled_cam = (
        {
            "tiled_cam_view": True,
            "tiled_cam_num": _CARTPOLE_VISUALIZER_TILED_CAMERA_NUM_TILES,
            "tiled_cam_prim_path": None,
            "tiled_cam_eye": _CARTPOLE_INTEGRATION_TILED_CAMERA_EYE_OFFSET,
            "tiled_cam_target_prim_path": _CARTPOLE_VISUALIZER_TILED_CAMERA_TARGET_PRIM_PATH,
        }
        if tiled_camera
        else {}
    )
    if visualizer_kind == "newton":
        __import__("newton")
        nw, nh = _CARTPOLE_NEWTON_INTEGRATION_WINDOW_SIZE
        return (
            NewtonVisualizerCfg(
                headless=True,
                window_width=nw,
                window_height=nh,
                randomly_sample_visible_envs=False,
                **tiled_cam,
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
    return (
        KitVisualizerCfg(
            window_width=_CARTPOLE_KIT_INTEGRATION_RENDER_RESOLUTION[0],
            window_height=_CARTPOLE_KIT_INTEGRATION_RENDER_RESOLUTION[1],
            randomly_sample_visible_envs=False,
            **tiled_cam,
            **cam,
        ),
        KitVisualizer,
    )


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


def _frame_to_numpy(frame) -> np.ndarray:
    """Convert viewer ``get_frame()`` output (numpy, torch, or Warp array) to host ``numpy.ndarray``.

    ``np.asarray(wp.array)`` is unsafe: NumPy can trigger Warp indexing that raises at dimension edges.
    """
    if isinstance(frame, np.ndarray):
        return frame
    if torch.is_tensor(frame):
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


def _current_visualizer_debug_dir() -> Path:
    override_test_id = os.environ.get(_VIS_DEBUG_TEST_ID_OVERRIDE_ENV)
    if override_test_id:
        safe_override_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", override_test_id).strip("_").lower()
        return _VIS_DEBUG_IMAGE_DIR / (safe_override_id or "manual_run")
    current_test = os.environ.get("PYTEST_CURRENT_TEST", "manual_run")
    test_id = _PYTEST_CURRENT_TEST_SUFFIX_PATTERN.sub("", current_test).split("::")[-1]
    is_tiled_test = False
    match = re.fullmatch(r"(?P<test_name>[^\[]+)(?:\[(?P<backend>[^\]]+)\])?", test_id)
    if match:
        test_name = match.group("test_name")
        prefix = _DEBUG_TEST_DIR_PREFIXES.get(test_name, test_name)
        backend = match.group("backend")
        if backend:
            test_id = f"{prefix}_{backend}"
        else:
            test_id = prefix
        is_tiled_test = test_name in _DEBUG_TEST_TILED_SUFFIXES
    if is_tiled_test:
        test_id = f"{test_id}_tiled"
    safe_test_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", test_id).strip("_").lower() or "manual_run"
    return _VIS_DEBUG_IMAGE_DIR / safe_test_id


@contextlib.contextmanager
def _visualizer_debug_case(viz_kind: str, physics_kind: str, *, tiled: bool = False):
    """Route debug PNGs to the same per-visualizer folders even in combined tests."""
    previous = os.environ.get(_VIS_DEBUG_TEST_ID_OVERRIDE_ENV)
    test_id = f"{viz_kind}_viz_{physics_kind}"
    if tiled:
        test_id = f"{test_id}_tiled"
    os.environ[_VIS_DEBUG_TEST_ID_OVERRIDE_ENV] = test_id
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_VIS_DEBUG_TEST_ID_OVERRIDE_ENV, None)
        else:
            os.environ[_VIS_DEBUG_TEST_ID_OVERRIDE_ENV] = previous


def _save_visualizer_debug_image(frame, file_name: str, *, tiled: bool = False) -> None:
    """Save a visualizer frame to a clearly named PNG for pause/motion debugging."""
    if not _WRITE_VIS_DEBUG_FRAMES:
        return
    from PIL import Image

    rgb = np.clip(_frame_rgb_255_space(frame), 0, 255).astype(np.uint8)
    debug_dir = _current_visualizer_debug_dir()
    debug_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(debug_dir / file_name)


def _save_visualizer_debug_delta(frame_a, frame_b, file_name: str, *, tiled: bool = False) -> None:
    """Save an amplified absolute-difference image for a start/end frame pair."""
    if not _WRITE_VIS_DEBUG_FRAMES:
        return
    from PIL import Image

    a = _frame_rgb_255_space(frame_a)
    b = _frame_rgb_255_space(frame_b)
    assert a.shape == b.shape, f"Frame shape mismatch for delta image: {a.shape} vs {b.shape}."
    delta = np.clip(np.abs(a - b) * 4.0, 0, 255).astype(np.uint8)
    debug_dir = _current_visualizer_debug_dir()
    debug_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(delta).save(debug_dir / file_name)


def _save_visualizer_debug_phase_images(
    frame_a,
    frame_b,
    *,
    prefix: str,
    phase: str,
    frame_start_idx: int,
    frame_end_idx: int,
    tiled: bool = False,
) -> None:
    """Save start/end/delta PNGs for one visualizer test phase."""
    _save_visualizer_debug_image(frame_a, f"{prefix}a_{phase}_frame_{frame_start_idx:02d}.png", tiled=tiled)
    _save_visualizer_debug_image(frame_b, f"{prefix}b_{phase}_frame_{frame_end_idx:02d}.png", tiled=tiled)
    _save_visualizer_debug_delta(
        frame_a,
        frame_b,
        f"{prefix}c_{phase}_frame_{frame_start_idx:02d}_{frame_end_idx:02d}_delta.png",
        tiled=tiled,
    )


def _clear_visualizer_debug_frames() -> None:
    if not _WRITE_VIS_DEBUG_FRAMES:
        return
    debug_dir = _current_visualizer_debug_dir()
    debug_dir.mkdir(parents=True, exist_ok=True)
    for path in debug_dir.glob("*.png"):
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


def _frame_shape_for_message(frame) -> tuple[int, ...]:
    return tuple(_frame_rgb_255_space(frame).shape)


def _assert_frames_remain_stable(
    frame_a,
    frame_b,
    *,
    case_label: str,
    phase: str,
    debug_phase: str,
    max_differing_pixels: int = 100,
) -> None:
    """Assert two viewport frames are effectively unchanged while simulation is paused."""
    n_diff = _count_significantly_differing_pixels(frame_a, frame_b)
    assert n_diff <= max_differing_pixels, (
        f"{case_label} failed to pause during {phase}: {n_diff} pixels differed, expected at most "
        f"{max_differing_pixels}. Frame shape={_frame_shape_for_message(frame_a)}. "
        f"Debug frames: {_current_visualizer_debug_dir()}/*{debug_phase}*.png."
    )


def _assert_frames_differ(
    frame_a,
    frame_b,
    *,
    case_label: str,
    phase: str,
    debug_phase: str,
    channel_diff_threshold: float = _FRAME_MOTION_CHANNEL_DIFF_THRESHOLD,
    min_differing_pixels: int = _FRAME_MOTION_MIN_DIFFERING_PIXELS,
) -> None:
    """Fail if two frames lack enough strongly differing pixels (stale/frozen bodies)."""
    n_diff = _count_significantly_differing_pixels(frame_a, frame_b, channel_diff_threshold=channel_diff_threshold)
    assert n_diff >= min_differing_pixels, (
        f"{case_label} is frozen during {phase}: {n_diff} pixels differed, expected at least "
        f"{min_differing_pixels} with per-channel threshold {channel_diff_threshold} in 0-255 space. "
    )


def _assert_tiled_camera_frames_differ(frame_a, frame_b, *, case_label: str, phase: str, debug_phase: str) -> None:
    """Fail if tiled camera frames lack enough motion for the fixed Cartpole camera view."""
    _assert_frames_differ(
        frame_a,
        frame_b,
        case_label=case_label,
        phase=phase,
        debug_phase=debug_phase,
        channel_diff_threshold=_TILED_CAMERA_MOTION_CHANNEL_DIFF_THRESHOLD,
        min_differing_pixels=_TILED_CAMERA_MOTION_MIN_DIFFERING_PIXELS,
    )


def _assert_tiled_camera_frame_non_flat(frame) -> None:
    """Assert the tiled camera frame has visible content."""
    _assert_non_flat_frame_array(frame)


def _assert_tiled_camera_frames_remain_stable(
    frame_a, frame_b, *, case_label: str, phase: str, debug_phase: str
) -> None:
    """Assert tiled camera frames are stable."""
    _assert_frames_remain_stable(frame_a, frame_b, case_label=case_label, phase=phase, debug_phase=debug_phase)


def _cartpole_body_state(env) -> torch.Tensor:
    """Return a compact body transform state for cartpole motion/stability checks."""
    cartpole = env.scene.articulations["cartpole"]
    pos = cartpole.data.body_pos_w.torch
    quat = cartpole.data.body_quat_w.torch
    return torch.cat((pos.reshape(-1), quat.reshape(-1))).detach().clone()


def _body_state_delta(state_a: torch.Tensor, state_b: torch.Tensor) -> float:
    """Return max absolute body-state delta."""
    assert state_a.shape == state_b.shape, f"Body state shape mismatch: {state_a.shape} vs {state_b.shape}."
    return float(torch.max(torch.abs(state_a - state_b)).item())


def _assert_body_state_changed(
    state_a: torch.Tensor,
    state_b: torch.Tensor,
    *,
    case_label: str,
    phase: str,
    min_delta: float = _BODY_STATE_MOTION_MIN_DELTA,
) -> None:
    delta = _body_state_delta(state_a, state_b)
    assert delta >= min_delta, (
        f"{case_label} physics/body state did not advance during {phase}: max body-state delta {delta:.6g}, "
        f"expected at least {min_delta:.6g}."
    )


def _assert_body_state_stable(
    state_a: torch.Tensor,
    state_b: torch.Tensor,
    *,
    case_label: str,
    phase: str,
    max_delta: float = _BODY_STATE_STABLE_MAX_DELTA,
) -> None:
    delta = _body_state_delta(state_a, state_b)
    assert delta <= max_delta, (
        f"{case_label} physics/body state changed during {phase}: max body-state delta {delta:.6g}, "
        f"expected at most {max_delta:.6g}."
    )


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


def _set_newton_simulation_paused(viewer, paused: bool) -> None:
    """Put Newton visualizer simulation pause control into a desired state."""
    if viewer.is_training_paused() != paused:
        _select_newton_pause_simulation_button(viewer)


def _select_newton_pause_rendering_button(viewer) -> None:
    """Trigger the Newton visualizer's Pause/Resume Rendering UI button."""
    label = "Resume Rendering" if viewer.is_rendering_paused() else "Pause Rendering"
    _select_newton_training_control_button(viewer, label)


def _set_newton_rendering_paused(viewer, paused: bool) -> None:
    """Put Newton visualizer rendering pause control into a desired state."""
    if viewer.is_rendering_paused() != paused:
        _select_newton_pause_rendering_button(viewer)


def _warm_newton_viewer(visualizer: NewtonVisualizer, viewer) -> None:
    """Pump Newton viewer frames before sampling ``get_frame()`` after cold starts."""
    for _ in range(_NEWTON_VIEWER_WARMUP_FRAMES):
        visualizer.step(0.0)
        with contextlib.suppress(Exception):
            viewer.get_frame()


def _run_newton_viewer_frame_motion_test(
    env,
    viewer,
    *,
    visualizer: NewtonVisualizer,
    step_hook,
    get_physics_step_count,
    physics_kind: str,
    viz_kind: str = "newton",
) -> None:
    """Check Newton viewer motion, rendering pause, simulation pause, and resumed motion."""
    _clear_visualizer_debug_frames()
    case_label = _visualizer_case_label(viz_kind, physics_kind)
    for _ in range(_START_BUFFER_STEPS):
        step_hook()
    _warm_newton_viewer(visualizer, viewer)

    motion_start_frame = viewer.get_frame()
    for _ in range(PLAY_VIZ_N_STEP):
        step_hook()
    play_end_idx = PLAY_VIZ_N_STEP
    motion_end_frame = viewer.get_frame()
    _save_visualizer_debug_phase_images(
        motion_start_frame,
        motion_end_frame,
        prefix="1",
        phase="playing",
        frame_start_idx=0,
        frame_end_idx=play_end_idx,
    )
    _assert_non_flat_frame_array(motion_end_frame)
    _assert_frames_differ(
        motion_start_frame,
        motion_end_frame,
        case_label=case_label,
        phase="playing",
        debug_phase="playing",
    )

    rendering_pause_start_idx = play_end_idx
    rendering_pause_end_idx = rendering_pause_start_idx + PAUSE_VIZ_N_STEP

    def _attempt_rendering_pause():
        _set_newton_rendering_paused(viewer, True)
        rendering_paused_start_frame = viewer.get_frame()
        rendering_pause_start_state = _cartpole_body_state(env)
        physics_step_before_render_pause = get_physics_step_count()
        for _ in range(PAUSE_VIZ_N_STEP):
            step_hook()
        rendering_pause_end_state = _cartpole_body_state(env)
        rendering_paused_end_frame = viewer.get_frame()
        _save_visualizer_debug_phase_images(
            rendering_paused_start_frame,
            rendering_paused_end_frame,
            prefix="2",
            phase="pausing_rendering",
            frame_start_idx=rendering_pause_start_idx,
            frame_end_idx=rendering_pause_end_idx,
        )
        _assert_frames_remain_stable(
            rendering_paused_start_frame,
            rendering_paused_end_frame,
            case_label=case_label,
            phase="pausing_rendering",
            debug_phase="pausing_rendering",
        )
        return physics_step_before_render_pause, rendering_pause_start_state, rendering_pause_end_state

    physics_step_before_render_pause, rendering_pause_start_state, rendering_pause_end_state = (
        _attempt_rendering_pause()
    )
    assert get_physics_step_count() > physics_step_before_render_pause, (
        f"{case_label} physics step count did not advance during pausing_rendering."
    )
    _assert_body_state_changed(
        rendering_pause_start_state,
        rendering_pause_end_state,
        case_label=case_label,
        phase="pausing_rendering",
    )

    rendering_play_start_idx = rendering_pause_end_idx
    rendering_play_end_idx = rendering_play_start_idx + PLAY_VIZ_N_STEP

    def _attempt_rendering_play():
        _set_newton_rendering_paused(viewer, False)
        rendering_play_start_frame = viewer.get_frame()
        for _ in range(PLAY_VIZ_N_STEP):
            step_hook()
        rendering_play_end_frame = viewer.get_frame()
        _save_visualizer_debug_phase_images(
            rendering_play_start_frame,
            rendering_play_end_frame,
            prefix="3",
            phase="playing",
            frame_start_idx=rendering_play_start_idx,
            frame_end_idx=rendering_play_end_idx,
        )
        _assert_non_flat_frame_array(rendering_play_end_frame)
        _assert_frames_differ(
            rendering_play_start_frame,
            rendering_play_end_frame,
            case_label=case_label,
            phase="playing after rendering pause",
            debug_phase="playing",
        )

    _attempt_rendering_play()

    simulation_pause_start_idx = rendering_play_end_idx
    simulation_pause_end_idx = simulation_pause_start_idx + PAUSE_VIZ_N_STEP

    def _attempt_simulation_pause():
        _set_newton_simulation_paused(viewer, True)
        simulation_paused_start_frame = viewer.get_frame()
        simulation_pause_start_state = _cartpole_body_state(env)
        physics_step_before_simulation_pause = get_physics_step_count()
        for _ in range(PAUSE_VIZ_N_STEP):
            visualizer.step(0.0)
        simulation_pause_end_state = _cartpole_body_state(env)
        simulation_paused_end_frame = viewer.get_frame()
        _save_visualizer_debug_phase_images(
            simulation_paused_start_frame,
            simulation_paused_end_frame,
            prefix="4",
            phase="pausing_simulation",
            frame_start_idx=simulation_pause_start_idx,
            frame_end_idx=simulation_pause_end_idx,
        )
        _assert_frames_remain_stable(
            simulation_paused_start_frame,
            simulation_paused_end_frame,
            case_label=case_label,
            phase="pausing_simulation",
            debug_phase="pausing_simulation",
        )
        return physics_step_before_simulation_pause, simulation_pause_start_state, simulation_pause_end_state

    physics_step_before_simulation_pause, simulation_pause_start_state, simulation_pause_end_state = (
        _attempt_simulation_pause()
    )
    assert get_physics_step_count() == physics_step_before_simulation_pause, (
        f"{case_label} physics step count advanced during pausing_simulation."
    )
    _assert_body_state_stable(
        simulation_pause_start_state,
        simulation_pause_end_state,
        case_label=case_label,
        phase="pausing_simulation",
    )

    simulation_play_start_idx = simulation_pause_end_idx
    simulation_play_end_idx = simulation_play_start_idx + PLAY_VIZ_N_STEP

    def _attempt_simulation_play():
        _set_newton_simulation_paused(viewer, False)
        simulation_play_start_frame = viewer.get_frame()
        for _ in range(PLAY_VIZ_N_STEP):
            step_hook()
        simulation_play_end_frame = viewer.get_frame()
        _save_visualizer_debug_phase_images(
            simulation_play_start_frame,
            simulation_play_end_frame,
            prefix="5",
            phase="playing",
            frame_start_idx=simulation_play_start_idx,
            frame_end_idx=simulation_play_end_idx,
        )
        _assert_non_flat_frame_array(simulation_play_end_frame)
        _assert_frames_differ(
            simulation_play_start_frame,
            simulation_play_end_frame,
            case_label=case_label,
            phase="playing after simulation pause",
            debug_phase="playing",
        )

    _attempt_simulation_play()


def _step_env_without_frame_check(env, actions: torch.Tensor, *, max_steps: int = _MAX_FRAME_CHECK_STEPS) -> None:
    """Step the env to exercise visualizers that do not implement ``get_frame`` (e.g. Rerun, Viser)."""
    for _ in range(max_steps):
        env.step(action=actions)


def _set_kit_simulation_paused(env, paused: bool) -> None:
    """Put Kit simulation play/pause state into a desired state."""
    if paused:
        env.sim.pause()
    else:
        env.sim.play()


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


def _update_active_simulation_app() -> None:
    """Pump the active Kit app launched by the backend test module."""
    if _SIMULATION_APP is not None:
        _SIMULATION_APP.update()
        return

    from isaacsim import SimulationApp

    sim_app = None
    if hasattr(SimulationApp, "_instance") and SimulationApp._instance is not None:
        sim_app = SimulationApp._instance
    elif hasattr(SimulationApp, "instance") and callable(SimulationApp.instance):
        sim_app = SimulationApp.instance()
    assert sim_app is not None, "Isaac Sim app is not running."
    sim_app.update()


def _drain_kit_app_updates(num_updates: int) -> None:
    """Let Kit process pending renderer/extension work between retry attempts."""
    for _ in range(max(0, int(num_updates))):
        with contextlib.suppress(Exception):
            _update_active_simulation_app()
        time.sleep(_KIT_APP_DRAIN_SLEEP_SECONDS)


def _prepare_visualizer_test_process() -> None:
    """Reset Python-side sim state and let Kit settle before a flaky retry starts."""
    with contextlib.suppress(Exception):
        SimulationContext.clear_instance()
    _drain_kit_app_updates(_VISUALIZER_STARTUP_DRAIN_UPDATES)


def _cleanup_visualizer_test_process(env) -> None:
    """Close per-test resources and drain Kit so state does not leak into retries."""
    try:
        if env is not None:
            env.close()
        else:
            SimulationContext.clear_instance()
    finally:
        with contextlib.suppress(Exception):
            SimulationContext.clear_instance()
        gc.collect()
        if torch.cuda.is_available():
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()
        _drain_kit_app_updates(_VISUALIZER_SHUTDOWN_DRAIN_UPDATES)


def _reapply_kit_camera_pose(env, kit_visualizer: KitVisualizer) -> None:
    """Re-apply Kit camera pose after Newton MJWarp stage/render-product setup settles."""
    kit_visualizer.set_camera_view(kit_visualizer.cfg.eye, kit_visualizer.cfg.lookat)
    env.sim.render()
    _update_active_simulation_app()


def _warm_kit_rtx_render_product(env, annotator) -> None:
    """Pump Kit/RTX render-product updates before sampling the annotator after cold starts."""
    for _ in range(_KIT_RTX_RENDER_PRODUCT_WARMUP_STEPS):
        env.sim.render()
        _update_active_simulation_app()
        with contextlib.suppress(Exception):
            annotator.get_data()


def _run_kit_viewport_frame_motion_test(
    env,
    kit_visualizer: KitVisualizer,
    *,
    physics_kind: str,
    viz_kind: str = "kit",
) -> None:
    """Check Kit viewport motion, SimulationContext pause freeze, then resumed motion."""
    _clear_visualizer_debug_frames()
    case_label = _visualizer_case_label(viz_kind, physics_kind)
    camera_path = getattr(kit_visualizer, "_controlled_camera_path", None)
    assert camera_path, "Kit visualizer does not expose a controlled viewport camera path."

    annotator = None
    render_product = None
    try:
        annotator, render_product = _build_rgb_annotator_for_camera(camera_path)
        _warm_kit_rtx_render_product(env, annotator)
        # TODO: Remove this workaround step during the Visualizer class refactor
        if viz_kind == "kit" and physics_kind == "newton":
            _reapply_kit_camera_pose(env, kit_visualizer)
        actions = torch.zeros((env.num_envs, env.action_space.shape[-1]), device=env.device)
        for _ in range(_START_BUFFER_STEPS):
            env.step(action=actions)
        motion_start_frame = _capture_kit_viewport_rgb(annotator)
        for _ in range(PLAY_VIZ_N_STEP):
            env.step(action=actions)
        play_end_idx = PLAY_VIZ_N_STEP
        motion_end_frame = _capture_kit_viewport_rgb(annotator)
        _save_visualizer_debug_phase_images(
            motion_start_frame,
            motion_end_frame,
            prefix="1",
            phase="playing",
            frame_start_idx=0,
            frame_end_idx=play_end_idx,
        )
        _assert_non_flat_frame_array(motion_end_frame)
        _assert_frames_differ(
            motion_start_frame,
            motion_end_frame,
            case_label=case_label,
            phase="playing",
            debug_phase="playing",
        )

        pause_start_idx = play_end_idx
        pause_end_idx = pause_start_idx + PAUSE_VIZ_N_STEP

        def _attempt_kit_pause():
            _set_kit_simulation_paused(env, True)
            paused_start_frame = _capture_kit_viewport_rgb(annotator)
            for _ in range(PAUSE_VIZ_N_STEP):
                env.sim.render()
            paused_end_frame = _capture_kit_viewport_rgb(annotator)
            _save_visualizer_debug_phase_images(
                paused_start_frame,
                paused_end_frame,
                prefix="2",
                phase="pausing",
                frame_start_idx=pause_start_idx,
                frame_end_idx=pause_end_idx,
            )
            _assert_frames_remain_stable(
                paused_start_frame,
                paused_end_frame,
                case_label=case_label,
                phase="pausing",
                debug_phase="pausing",
            )

        try:
            _attempt_kit_pause()
        finally:
            _set_kit_simulation_paused(env, False)

        replay_start_idx = pause_end_idx
        replay_end_idx = replay_start_idx + PLAY_VIZ_N_STEP

        def _attempt_kit_replay():
            _set_kit_simulation_paused(env, False)
            play_start_frame = _capture_kit_viewport_rgb(annotator)
            for _ in range(PLAY_VIZ_N_STEP):
                env.step(action=actions)
            play_end_frame = _capture_kit_viewport_rgb(annotator)
            _save_visualizer_debug_phase_images(
                play_start_frame,
                play_end_frame,
                prefix="3",
                phase="playing",
                frame_start_idx=replay_start_idx,
                frame_end_idx=replay_end_idx,
            )
            _assert_non_flat_frame_array(play_end_frame)
            _assert_frames_differ(
                play_start_frame,
                play_end_frame,
                case_label=case_label,
                phase="playing after pause",
                debug_phase="playing",
            )

        _attempt_kit_replay()
    finally:
        if annotator is not None and render_product is not None:
            with contextlib.suppress(Exception):
                annotator.detach([render_product])


def _capture_kit_viewport_rgb(annotator) -> np.ndarray:
    frame = _annotator_rgb_to_numpy(annotator.get_data())
    for _ in range(5):
        if frame.shape[:2] != (1, 1) or np.count_nonzero(frame) > 0:
            return frame
        _update_active_simulation_app()
        frame = _annotator_rgb_to_numpy(annotator.get_data())
    return frame


def _capture_visualizer_tiled_camera_rgb(visualizer, *, label: str = "capture") -> np.ndarray:
    """Return the visualizer-owned/generated tiled camera RGB frame as an HxWx3 array."""
    camera_sensor = visualizer._camera_sensor
    assert camera_sensor is not None, "Visualizer did not create a tiled camera sensor."
    camera_indices = [int(index) for index in (visualizer._camera_sensor_indices or [0])]
    if getattr(visualizer, "_camera_is_owned", False):
        visualizer._update_owned_camera_poses()
        if isinstance(visualizer, KitVisualizer):
            visualizer._sync_camera_pose_updates_to_kit()
            _update_active_simulation_app()
        camera_sensor.update(dt=0.0, force_recompute=True)
    rgb_batch = camera_rgb_batch(camera_sensor, camera_indices)
    frame = compose_rgb_grid_tensor(rgb_batch).detach().cpu().numpy()
    assert frame.ndim == 3, f"Expected tiled camera RGB frame to be HxWxC, got shape {frame.shape}."
    assert frame.shape[-1] >= 3, f"Expected tiled camera RGB frame to have at least 3 channels, got {frame.shape}."
    return frame[..., :3]


def _run_visualizer_tiled_camera_motion_test(env, visualizer, *, physics_kind: str, viz_kind: str) -> None:
    """Check generated visualizer tiled-camera RGB moves, pauses, and resumes."""
    _clear_visualizer_debug_frames()
    case_label = f"{_visualizer_case_label(viz_kind, physics_kind)} tiled camera"
    actions = torch.zeros((env.num_envs, env.action_space.shape[-1]), device=env.device)
    for _ in range(_START_BUFFER_STEPS):
        env.step(action=actions)

    motion_start_frame = _capture_visualizer_tiled_camera_rgb(visualizer, label="1a_playing_frame_00")
    for _ in range(PLAY_VIZ_N_STEP):
        env.step(action=actions)
    play_end_idx = PLAY_VIZ_N_STEP
    motion_end_frame = _capture_visualizer_tiled_camera_rgb(visualizer, label="1b_playing_frame_20")
    _save_visualizer_debug_phase_images(
        motion_start_frame,
        motion_end_frame,
        prefix="1",
        phase="playing",
        frame_start_idx=0,
        frame_end_idx=play_end_idx,
        tiled=True,
    )
    _assert_tiled_camera_frame_non_flat(motion_end_frame)
    _assert_tiled_camera_frames_differ(
        motion_start_frame,
        motion_end_frame,
        case_label=case_label,
        phase="playing",
        debug_phase="playing_tiled",
    )

    pause_start_idx = play_end_idx
    pause_end_idx = pause_start_idx + PAUSE_VIZ_N_STEP

    def _attempt_pause():
        _set_kit_simulation_paused(env, True)
        paused_start_frame = _capture_visualizer_tiled_camera_rgb(visualizer, label="2a_pausing_frame_20")
        for _ in range(PAUSE_VIZ_N_STEP):
            env.sim.render()
        paused_end_frame = _capture_visualizer_tiled_camera_rgb(visualizer, label="2b_pausing_frame_25")
        _save_visualizer_debug_phase_images(
            paused_start_frame,
            paused_end_frame,
            prefix="2",
            phase="pausing",
            frame_start_idx=pause_start_idx,
            frame_end_idx=pause_end_idx,
            tiled=True,
        )
        _assert_tiled_camera_frame_non_flat(paused_end_frame)
        _assert_tiled_camera_frames_remain_stable(
            paused_start_frame,
            paused_end_frame,
            case_label=case_label,
            phase="pausing",
            debug_phase="pausing_tiled",
        )

    try:
        _attempt_pause()
    finally:
        _set_kit_simulation_paused(env, False)

    replay_start_idx = pause_end_idx
    replay_end_idx = replay_start_idx + PLAY_VIZ_N_STEP

    def _attempt_replay():
        _set_kit_simulation_paused(env, False)
        play_start_frame = _capture_visualizer_tiled_camera_rgb(visualizer, label="3a_playing_frame_25")
        for _ in range(PLAY_VIZ_N_STEP):
            env.step(action=actions)
        play_end_frame = _capture_visualizer_tiled_camera_rgb(visualizer, label="3b_playing_frame_45")
        _save_visualizer_debug_phase_images(
            play_start_frame,
            play_end_frame,
            prefix="3",
            phase="playing",
            frame_start_idx=replay_start_idx,
            frame_end_idx=replay_end_idx,
            tiled=True,
        )
        _assert_tiled_camera_frame_non_flat(play_end_frame)
        _assert_tiled_camera_frames_differ(
            play_start_frame,
            play_end_frame,
            case_label=case_label,
            phase="playing after pause",
            debug_phase="playing_tiled",
        )

    _attempt_replay()


def _make_cartpole_camera_env(
    visualizer_kind: str | tuple[str, ...], backend_kind: str, *, tiled_camera: bool = False
) -> CartpoleCameraEnv:
    """Create cartpole camera env configured with selected visualizer and physics backend."""
    env_cfg_root = CartpoleCameraEnvCfg()
    env_cfg = getattr(env_cfg_root, "default", None)
    if env_cfg is None:
        env_cfg = getattr(type(env_cfg_root), "default", None)
    if env_cfg is None:
        raise RuntimeError(
            "CartpoleCameraEnvCfg does not expose a 'default' preset config. "
            f"Available attributes: {sorted(vars(env_cfg_root).keys())}"
        )
    env_cfg = copy.deepcopy(env_cfg)
    env_cfg.scene.num_envs = (
        _CARTPOLE_TILED_CAMERA_INTEGRATION_NUM_ENVS if tiled_camera else _CARTPOLE_INTEGRATION_NUM_ENVS
    )
    env_cfg.viewer.eye = _CARTPOLE_INTEGRATION_VISUALIZER_EYE
    env_cfg.viewer.lookat = _CARTPOLE_INTEGRATION_VISUALIZER_LOOKAT
    tw, th = _CARTPOLE_TILED_CAMERA_INTEGRATION_WH
    env_cfg.tiled_camera.width = tw
    env_cfg.tiled_camera.height = th
    if isinstance(env_cfg.observation_space, list) and len(env_cfg.observation_space) >= 3:
        env_cfg.observation_space = [th, tw, env_cfg.observation_space[2]]
    env_cfg.seed = None
    env_cfg.sim.physics, _ = _get_physics_cfg(backend_kind)
    visualizer_kinds = (visualizer_kind,) if isinstance(visualizer_kind, str) else tuple(visualizer_kind)
    visualizer_cfgs = [_get_visualizer_cfg(kind, tiled_camera=tiled_camera)[0] for kind in visualizer_kinds]
    env_cfg.sim.visualizer_cfgs = visualizer_cfgs[0] if len(visualizer_cfgs) == 1 else visualizer_cfgs
    return CartpoleCameraEnv(env_cfg)


def run_cartpole_env_visualizers_motion_with_play_pause(backend_kind: str, caplog: pytest.LogCaptureFixture) -> None:
    """Cartpole env + all non-tiled visualizers: frame checks and no visualizer log errors."""
    env = None
    try:
        _prepare_visualizer_test_process()
        sim_utils.create_new_stage()
        env = _make_cartpole_camera_env(
            visualizer_kind=("kit", "newton", "rerun", "viser"),
            backend_kind=backend_kind,
        )
        _configure_sim_for_visualizer_test(env)
        with caplog.at_level(logging.WARNING):
            env.reset()
            kit_visualizers = [viz for viz in env.sim.visualizers if isinstance(viz, KitVisualizer)]
            assert kit_visualizers, "Expected an initialized Kit visualizer."
            with _visualizer_debug_case("kit", backend_kind):
                _run_kit_viewport_frame_motion_test(env, kit_visualizers[0], physics_kind=backend_kind)

            actions = torch.zeros((env.num_envs, env.action_space.shape[-1]), device=env.device)
            newton_visualizers = [viz for viz in env.sim.visualizers if isinstance(viz, NewtonVisualizer)]
            assert newton_visualizers, "Expected an initialized Newton visualizer."
            viewer = getattr(newton_visualizers[0], "_viewer", None)
            assert viewer is not None, "Newton viewer was not created."

            def _step_env() -> None:
                env.step(action=actions)

            with _visualizer_debug_case("newton", backend_kind):
                _run_newton_viewer_frame_motion_test(
                    env,
                    viewer,
                    visualizer=newton_visualizers[0],
                    step_hook=_step_env,
                    get_physics_step_count=lambda: env.sim._physics_step_count,
                    physics_kind=backend_kind,
                )

            from isaaclab_visualizers.rerun import RerunVisualizer
            from isaaclab_visualizers.viser import ViserVisualizer

            rerun_visualizers = [viz for viz in env.sim.visualizers if isinstance(viz, RerunVisualizer)]
            assert rerun_visualizers, "Expected an initialized Rerun visualizer."
            assert getattr(rerun_visualizers[0], "_viewer", None) is not None, "Rerun viewer was not created."
            _step_env_without_frame_check(env, actions, max_steps=_MAX_FRAME_CHECK_STEPS)

            viser_visualizers = [viz for viz in env.sim.visualizers if isinstance(viz, ViserVisualizer)]
            assert viser_visualizers, "Expected an initialized Viser visualizer."
            assert getattr(viser_visualizers[0], "_viewer", None) is not None, "Viser viewer was not created."
            _step_env_without_frame_check(env, actions, max_steps=_MAX_FRAME_CHECK_STEPS)
        _assert_no_visualizer_log_issues(caplog)
    finally:
        _cleanup_visualizer_test_process(env)


def run_cartpole_env_visualizers_tiled_camera_motion(backend_kind: str, caplog: pytest.LogCaptureFixture) -> None:
    """Cartpole env + tiled Kit/Newton visualizers: RGB moves, pauses, and resumes without log errors."""
    env = None
    try:
        _prepare_visualizer_test_process()
        sim_utils.create_new_stage()
        env = _make_cartpole_camera_env(
            visualizer_kind=("kit", "newton"),
            backend_kind=backend_kind,
            tiled_camera=True,
        )
        _configure_sim_for_visualizer_test(env)
        with caplog.at_level(logging.WARNING):
            env.reset()
            kit_visualizers = [viz for viz in env.sim.visualizers if isinstance(viz, KitVisualizer)]
            assert kit_visualizers, "Expected an initialized Kit visualizer."
            with _visualizer_debug_case("kit", backend_kind, tiled=True):
                _run_visualizer_tiled_camera_motion_test(
                    env, kit_visualizers[0], physics_kind=backend_kind, viz_kind="kit"
                )

            newton_visualizers = [viz for viz in env.sim.visualizers if isinstance(viz, NewtonVisualizer)]
            assert newton_visualizers, "Expected an initialized Newton visualizer."
            with _visualizer_debug_case("newton", backend_kind, tiled=True):
                _run_visualizer_tiled_camera_motion_test(
                    env, newton_visualizers[0], physics_kind=backend_kind, viz_kind="newton"
                )
        _assert_no_visualizer_log_issues(caplog)
    finally:
        _cleanup_visualizer_test_process(env)
