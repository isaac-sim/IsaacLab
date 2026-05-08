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

# Pyglet must use HeadlessWindow (EGL) before ``pyglet.window`` is imported so Newton
# ViewerGL can construct without an X11 display (matches ``headless=True`` on NewtonVisualizerCfg).
import pyglet

pyglet.options["headless"] = True

from isaaclab.app import AppLauncher

# launch Kit app
simulation_app = AppLauncher(headless=True, enable_cameras=True).app

import contextlib
import copy
import logging
import socket

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_visualizers.kit import KitVisualizer, KitVisualizerCfg
from isaaclab_visualizers.newton import NewtonVisualizer, NewtonVisualizerCfg
from isaaclab_visualizers.rerun import RerunVisualizer, RerunVisualizerCfg
from isaaclab_visualizers.viser import ViserVisualizer, ViserVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

from isaaclab_tasks.direct.cartpole.cartpole_camera_env import CartpoleCameraEnv
from isaaclab_tasks.direct.cartpole.cartpole_camera_presets_env_cfg import CartpoleCameraPresetsEnvCfg
from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpolePhysicsCfg

# When True, tests also fail on WARNING-level records from visualizer-related loggers.
ASSERT_VISUALIZER_WARNINGS = False

_MAX_NON_BLACK_STEPS = 8
"""Steps for tiled camera / Rerun / Viser smoke tests (early exit ok when non-black)."""

_CARTPOLE_INTEGRATION_NUM_ENVS = 1
"""Vectorized env count for cartpole + visualizer integration tests."""

_CARTPOLE_INTEGRATION_VISUALIZER_EYE: tuple[float, float, float] = (3.0, 3.0, 3.0)
"""Passed to :class:`~isaaclab.visualizers.visualizer_cfg.VisualizerCfg` subclasses (``eye``)."""

_CARTPOLE_INTEGRATION_VISUALIZER_LOOKAT: tuple[float, float, float] = (-4.0, -4.0, 0.0)
"""Passed to visualizer cfgs (``lookat``); also applied to :class:`~isaaclab.envs.common.ViewerCfg` for the env."""

# Resolution overrides for this test module (cartpole preset defaults: tiled camera 100×100; Kit helper was 320×240).
_CARTPOLE_KIT_INTEGRATION_RENDER_RESOLUTION: tuple[int, int] = (600, 600)
"""Kit: Replicator ``render_product`` (width, height) for viewport RGB in the motion check."""

_CARTPOLE_NEWTON_INTEGRATION_WINDOW_SIZE: tuple[int, int] = (600, 600)
"""Newton: ``NewtonVisualizerCfg`` framebuffer (window_width × window_height) for ``get_frame()``."""

_CARTPOLE_TILED_CAMERA_INTEGRATION_WH: tuple[int, int] = (600, 600)
"""Tiled camera per-env tile width/height (preset default is 100×100); keeps ``observation_space`` consistent."""

_VIS_FRAME_TEST_STEPS = 60
"""Steps for Kit / Newton frame capture: no early exit."""

# Motion check compares the 2nd vs last captured frame (e.g. 2nd vs 60th when *_STEPS* is 60).
_MOTION_FRAME_EARLY_IDX = 1
"""0-based index of the *early* frame (2nd capture)."""

_MOTION_FRAME_LATE_IDX = _VIS_FRAME_TEST_STEPS - 1
"""0-based index of the *late* frame (e.g. 60th capture when :data:`_VIS_FRAME_TEST_STEPS` is 60)."""

# Early vs late frame motion: void background stays similar; only count *strongly* differing pixels.
_FRAME_MOTION_CHANNEL_DIFF_THRESHOLD = 50
"""A pixel counts as differing if max(|ΔR|, |ΔG|, |ΔB|) >= this (0–255 space)."""

_FRAME_MOTION_MIN_DIFFERING_PIXELS = 100
"""Minimum number of such pixels between early and late frames (stale/frozen viz should be near zero)."""

_VIS_LOGGER_PREFIXES = (
    "isaaclab.visualizers",
    "isaaclab_visualizers",
    "isaaclab.sim.simulation_context",
)


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
        port = _find_free_tcp_port(host="127.0.0.1")
        return (
            ViserVisualizerCfg(open_browser=False, port=port, randomly_sample_visible_envs=False, **cam),
            ViserVisualizer,
        )
    if visualizer_kind == "rerun":
        __import__("newton")
        __import__("rerun")
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


def _assert_non_black_tensor(image_tensor: torch.Tensor, *, min_nonzero_pixels: int = 1) -> None:
    """Assert camera-like tensor contains non-black pixels."""
    assert isinstance(image_tensor, torch.Tensor), f"Expected torch.Tensor, got {type(image_tensor)!r}"
    assert image_tensor.numel() > 0, "Image tensor is empty."
    finite_tensor = torch.where(torch.isfinite(image_tensor), image_tensor, torch.zeros_like(image_tensor))
    if finite_tensor.dtype.is_floating_point:
        nonzero = torch.count_nonzero(torch.abs(finite_tensor) > 1e-6).item()
    else:
        nonzero = torch.count_nonzero(finite_tensor > 0).item()
    assert nonzero >= min_nonzero_pixels, "Rendered frame appears black (no non-zero pixels)."


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


def _assert_non_black_frame_array(frame) -> None:
    """Assert viewer-captured frame has visible, non-black content."""
    frame_arr = _frame_to_numpy(frame)
    assert frame_arr.size > 0, "Viewer returned an empty frame."
    if frame_arr.ndim == 2:
        color = frame_arr
    else:
        assert frame_arr.shape[-1] >= 3, f"Expected at least 3 channels, got shape {frame_arr.shape}."
        color = frame_arr[..., :3]
    finite = np.where(np.isfinite(color), color, 0)
    assert np.count_nonzero(finite) > 0, "Viewer frame appears fully black."


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


def _assert_early_and_late_motion_frames_differ(
    frames: list,
    *,
    channel_diff_threshold: float = _FRAME_MOTION_CHANNEL_DIFF_THRESHOLD,
    min_differing_pixels: int = _FRAME_MOTION_MIN_DIFFERING_PIXELS,
) -> None:
    """Fail if early vs late frames lack enough strongly differing pixels (stale/frozen bodies).

    Compares :data:`_MOTION_FRAME_EARLY_IDX` vs :data:`_MOTION_FRAME_LATE_IDX` (e.g. 2nd vs 60th capture).

    Voids/background stay near-identical; we only count pixels that change by at least
    *channel_diff_threshold* on some channel (0–255).
    """
    assert len(frames) >= _VIS_FRAME_TEST_STEPS, (
        f"Need at least {_VIS_FRAME_TEST_STEPS} frames for motion check, got {len(frames)}."
    )
    i_early = _MOTION_FRAME_EARLY_IDX
    i_late = _MOTION_FRAME_LATE_IDX
    early_1 = i_early + 1
    late_1 = i_late + 1
    n_diff = _count_significantly_differing_pixels(
        frames[i_early], frames[i_late], channel_diff_threshold=channel_diff_threshold
    )
    assert n_diff >= min_differing_pixels, (
        f"Viewport captures #{early_1} and #{late_1} have too few strongly differing pixels "
        f"({n_diff} < {min_differing_pixels}; threshold per channel={channel_diff_threshold} in 0–255 space). "
        "Possible frozen or stale robot visualization."
    )


def _step_until_non_black_camera(env, actions: torch.Tensor, *, max_steps: int = _MAX_NON_BLACK_STEPS) -> None:
    """Step env until the env's tiled camera RGB tensor is non-black, bounded by *max_steps*."""
    last_rgb = None
    for _ in range(max_steps):
        env.step(action=actions)
        rgb = env._tiled_camera.data.output.get("rgb")
        if rgb is None:
            rgb = env._tiled_camera.data.output[env.cfg.tiled_camera.data_types[0]]
        last_rgb = rgb
        try:
            _assert_non_black_tensor(rgb)
            return
        except AssertionError:
            continue
    _assert_non_black_tensor(last_rgb)


def _run_newton_viewer_frame_motion_test(
    viewer,
    *,
    step_hook,
    physics_kind: str,
    viz_kind: str = "newton",
) -> None:
    """Exactly ``_VIS_FRAME_TEST_STEPS`` sim steps; last frame non-black; early vs late motion check."""
    frames: list = []
    for _ in range(_VIS_FRAME_TEST_STEPS):
        step_hook()
        frames.append(viewer.get_frame())
    _assert_non_black_frame_array(frames[-1])
    _assert_early_and_late_motion_frames_differ(frames)


def _step_env_without_frame_check(env, actions: torch.Tensor, *, max_steps: int = _MAX_NON_BLACK_STEPS) -> None:
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
    """Exactly ``_VIS_FRAME_TEST_STEPS`` env steps; last Replicator frame non-black; early vs late motion check."""
    camera_path = getattr(kit_visualizer, "_controlled_camera_path", None)
    assert camera_path, "Kit visualizer does not expose a controlled viewport camera path."

    annotator = None
    render_product = None
    try:
        annotator, render_product = _build_rgb_annotator_for_camera(camera_path)
        actions = torch.zeros((env.num_envs, env.action_space.shape[-1]), device=env.device)
        frames: list = []
        for _ in range(_VIS_FRAME_TEST_STEPS):
            env.step(action=actions)
            rgb_data = annotator.get_data()
            frames.append(_annotator_rgb_to_numpy(rgb_data))
        _assert_non_black_frame_array(frames[-1])
        _assert_early_and_late_motion_frames_differ(frames)
    finally:
        if annotator is not None and render_product is not None:
            with contextlib.suppress(Exception):
                annotator.detach([render_product])


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
    [
        # xfail: Kit visualizer + PhysX only (Newton backend uses skip below — separate CUDA issue).
        pytest.param(
            "physx",
            marks=pytest.mark.xfail(
                reason=("Kit visualizer + PhysX: TODO remove xfail when stale Fabric transforms bug in Kit is fixed"),
                strict=False,
            ),
        ),
        pytest.param(
            "newton",
            marks=pytest.mark.skip(
                reason=(
                    "TODO: Kit visualizer + Newton physics + Isaac RTX tiled camera can hit CUDA illegal access "
                    "or bad GPU state. Repro: rl_games train Isaac-Cartpole-Camera-Presets-Direct-v0 "
                    "--enable_cameras presets=newton_mjwarp --viz kit. Re-enable when fixed."
                )
            ),
        ),
    ],
)
def test_cartpole_kit_visualizer_replicator_viewport_rgb_motion(
    backend_kind: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Kit + cartpole: Replicator RGB on viewport camera; last frame non-black; early vs late frame differ; logs."""
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
def test_cartpole_newton_visualizer_tiled_camera_rgb_non_black(
    backend_kind: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Newton visualizer + cartpole: env tiled-camera RGB becomes non-black within a few steps; clean logs."""
    env = None
    try:
        sim_utils.create_new_stage()
        env = _make_cartpole_camera_env(visualizer_kind="newton", backend_kind=backend_kind)
        _configure_sim_for_visualizer_test(env)
        with caplog.at_level(logging.WARNING):
            env.reset()
            actions = torch.zeros((env.num_envs, env.action_space.shape[-1]), device=env.device)
            _step_until_non_black_camera(env, actions, max_steps=_MAX_NON_BLACK_STEPS)
        _assert_no_visualizer_log_issues(caplog)
    finally:
        if env is not None:
            env.close()
        else:
            SimulationContext.clear_instance()


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("backend_kind", ["physx", "newton"])
def test_cartpole_newton_visualizer_viewergl_rgb_motion(backend_kind: str, caplog: pytest.LogCaptureFixture) -> None:
    """Newton GL (``ViewerGL.get_frame``): full motion steps, last frame non-black; early vs late differ; logs."""
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

            _run_newton_viewer_frame_motion_test(viewer, step_hook=_step_env, physics_kind=backend_kind)
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
            rerun_visualizers = [viz for viz in env.sim.visualizers if isinstance(viz, RerunVisualizer)]
            assert rerun_visualizers, "Expected an initialized Rerun visualizer."
            assert getattr(rerun_visualizers[0], "_viewer", None) is not None, "Rerun viewer was not created."
            _step_env_without_frame_check(env, actions, max_steps=_MAX_NON_BLACK_STEPS)
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
            viser_visualizers = [viz for viz in env.sim.visualizers if isinstance(viz, ViserVisualizer)]
            assert viser_visualizers, "Expected an initialized Viser visualizer."
            assert getattr(viser_visualizers[0], "_viewer", None) is not None, "Viser viewer was not created."
            _step_env_without_frame_check(env, actions, max_steps=_MAX_NON_BLACK_STEPS)
        _assert_no_visualizer_log_issues(caplog)
    finally:
        if env is not None:
            env.close()
        else:
            SimulationContext.clear_instance()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--maxfail=1"])
