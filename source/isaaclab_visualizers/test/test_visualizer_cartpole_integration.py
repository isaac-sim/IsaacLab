# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests: Cartpole + visualizers, non-black frames, and log hygiene.

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


def _get_visualizer_cfg(visualizer_kind: str):
    """Return (visualizer_cfg, expected_visualizer_cls) for the given visualizer kind."""
    if visualizer_kind == "newton":
        __import__("newton")
        return NewtonVisualizerCfg(headless=True), NewtonVisualizer
    if visualizer_kind == "viser":
        __import__("newton")
        __import__("viser")
        port = _find_free_tcp_port(host="127.0.0.1")
        return ViserVisualizerCfg(open_browser=False, port=port), ViserVisualizer
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
            ),
            RerunVisualizer,
        )
    return KitVisualizerCfg(), KitVisualizer


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
        physics_cfg = getattr(preset, "newton", None)
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


def _assert_non_black_frame_array(frame) -> None:
    """Assert viewer-captured frame has visible, non-black content."""
    frame_arr = np.asarray(frame)
    assert frame_arr.size > 0, "Viewer returned an empty frame."
    if frame_arr.ndim == 2:
        color = frame_arr
    else:
        assert frame_arr.shape[-1] >= 3, f"Expected at least 3 channels, got shape {frame_arr.shape}."
        color = frame_arr[..., :3]
    finite = np.where(np.isfinite(color), color, 0)
    assert np.count_nonzero(finite) > 0, "Viewer frame appears fully black."


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


def _step_until_non_black_viewer_get_frame(viewer, *, max_steps: int, step_hook) -> None:
    """Call *step_hook* each iteration until *viewer*.``get_frame()`` is non-black within *max_steps*."""
    last_frame = None
    for _ in range(max_steps):
        step_hook()
        last_frame = viewer.get_frame()
        try:
            _assert_non_black_frame_array(last_frame)
            return
        except AssertionError:
            continue
    _assert_non_black_frame_array(last_frame)


def _build_rgb_annotator_for_camera(camera_path: str, *, resolution: tuple[int, int] = (320, 240)):
    """Create CPU RGB annotator attached to a camera render product."""
    import omni.replicator.core as rep

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


def _step_until_non_black_kit_viewport(
    env, kit_visualizer: KitVisualizer, *, max_steps: int = _MAX_NON_BLACK_STEPS
) -> None:
    """Step env until Kit viewport camera render product is non-black, bounded by max_steps."""
    camera_path = getattr(kit_visualizer, "_controlled_camera_path", None)
    assert camera_path, "Kit visualizer does not expose a controlled viewport camera path."

    annotator = None
    render_product = None
    try:
        annotator, render_product = _build_rgb_annotator_for_camera(camera_path)
        actions = torch.zeros((env.num_envs, env.action_space.shape[-1]), device=env.device)
        last_frame = None
        for _ in range(max_steps):
            env.step(action=actions)
            rgb_data = annotator.get_data()
            frame = _annotator_rgb_to_numpy(rgb_data)
            last_frame = frame
            try:
                _assert_non_black_frame_array(frame)
                return
            except AssertionError:
                continue
        _assert_non_black_frame_array(last_frame)
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
    env_cfg.scene.num_envs = 1
    env_cfg.seed = None
    env_cfg.sim.physics, _ = _get_physics_cfg(backend_kind)
    visualizer_cfg, _ = _get_visualizer_cfg(visualizer_kind)
    env_cfg.sim.visualizer_cfgs = visualizer_cfg
    return CartpoleCameraEnv(env_cfg)


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("backend_kind", ["physx", "newton"])
def test_kit_visualizer_non_black_viewport_frame(backend_kind: str, caplog: pytest.LogCaptureFixture) -> None:
    """Kit visualizer viewport (Replicator RGB) is not black; no visualizer ERROR (optional WARNING)."""
    env = None
    try:
        sim_utils.create_new_stage()
        env = _make_cartpole_camera_env(visualizer_kind="kit", backend_kind=backend_kind)
        _configure_sim_for_visualizer_test(env)
        with caplog.at_level(logging.WARNING):
            env.reset()
            kit_visualizers = [viz for viz in env.sim.visualizers if isinstance(viz, KitVisualizer)]
            assert kit_visualizers, "Expected an initialized Kit visualizer."
            _step_until_non_black_kit_viewport(env, kit_visualizers[0], max_steps=_MAX_NON_BLACK_STEPS)
        _assert_no_visualizer_log_issues(caplog)
    finally:
        if env is not None:
            env.close()
        else:
            SimulationContext.clear_instance()


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("backend_kind", ["physx", "newton"])
def test_cartpole_tiled_camera_rgb_non_black(backend_kind: str, caplog: pytest.LogCaptureFixture) -> None:
    """Tiled camera RGB is not all-black; no visualizer ERROR (optional WARNING)."""
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
def test_newton_visualizer_non_black_viewer_frame(backend_kind: str, caplog: pytest.LogCaptureFixture) -> None:
    """Newton GL ``get_frame`` is not all-black; no visualizer ERROR (optional WARNING)."""
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

            _step_until_non_black_viewer_get_frame(viewer, max_steps=_MAX_NON_BLACK_STEPS, step_hook=_step_env)
        _assert_no_visualizer_log_issues(caplog)
    finally:
        if env is not None:
            env.close()
        else:
            SimulationContext.clear_instance()


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("backend_kind", ["physx", "newton"])
def test_rerun_visualizer_non_black_viewer_frame(backend_kind: str, caplog: pytest.LogCaptureFixture) -> None:
    """Rerun ``ViewerRerun.get_frame`` is not all-black; no visualizer ERROR (optional WARNING)."""
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
            viewer = getattr(rerun_visualizers[0], "_viewer", None)
            assert viewer is not None, "Rerun viewer was not created."

            def _step_env() -> None:
                env.step(action=actions)

            _step_until_non_black_viewer_get_frame(viewer, max_steps=_MAX_NON_BLACK_STEPS, step_hook=_step_env)
        _assert_no_visualizer_log_issues(caplog)
    finally:
        if env is not None:
            env.close()
        else:
            SimulationContext.clear_instance()


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("backend_kind", ["physx", "newton"])
def test_viser_visualizer_non_black_viewer_frame(backend_kind: str, caplog: pytest.LogCaptureFixture) -> None:
    """Viser ``ViewerViser.get_frame`` is not all-black; no visualizer ERROR (optional WARNING)."""
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
            viewer = getattr(viser_visualizers[0], "_viewer", None)
            assert viewer is not None, "Viser viewer was not created."

            def _step_env() -> None:
                env.step(action=actions)

            _step_until_non_black_viewer_get_frame(viewer, max_steps=_MAX_NON_BLACK_STEPS, step_hook=_step_env)
        _assert_no_visualizer_log_issues(caplog)
    finally:
        if env is not None:
            env.close()
        else:
            SimulationContext.clear_instance()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--maxfail=1"])
