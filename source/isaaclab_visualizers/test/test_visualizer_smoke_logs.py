# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test visualizer stepping and error logging."""

from isaaclab.app import AppLauncher

# launch Kit app
simulation_app = AppLauncher(headless=True, enable_cameras=True).app

import logging
import socket

import pytest
import torch
from isaaclab_visualizers.kit import KitVisualizer, KitVisualizerCfg
from isaaclab_visualizers.newton import NewtonVisualizer, NewtonVisualizerCfg
from isaaclab_visualizers.rerun import RerunVisualizer, RerunVisualizerCfg
from isaaclab_visualizers.viser import ViserVisualizer, ViserVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import (
    CartpolePhysicsCfg,
    CartpoleSceneCfg,
)

# Set to False to only fail on visualizer errors; when True, also fail on warnings.
ASSERT_VISUALIZER_WARNINGS = True

_SMOKE_STEPS = 4
_VIS_LOGGER_PREFIXES = (
    "isaaclab.visualizers",
    "isaaclab_visualizers",
    "isaaclab.sim.simulation_context",
)


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


@configclass
class _SmokeEnvCfg(DirectRLEnvCfg):
    decimation: int = 2
    action_space: int = 0
    observation_space: int = 0
    episode_length_s: float = 5.0
    sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=2, visualizer_cfgs=KitVisualizerCfg())
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=1.0)


class _SmokeEnv(DirectRLEnv):
    def _pre_physics_step(self, actions):
        return

    def _apply_action(self):
        return

    def _get_observations(self):
        return {}

    def _get_rewards(self):
        return {}

    def _get_dones(self):
        return torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)


def _get_visualizer_cfg(visualizer_kind: str):
    """Return (visualizer_cfg, expected_visualizer_cls) for the given visualizer kind."""
    if visualizer_kind == "newton":
        __import__("newton")
        return NewtonVisualizerCfg(headless=True), NewtonVisualizer
    if visualizer_kind == "viser":
        __import__("newton")
        __import__("viser")
        return ViserVisualizerCfg(open_browser=False), ViserVisualizer
    if visualizer_kind == "rerun":
        __import__("newton")
        __import__("rerun")
        web_port, grpc_port = _allocate_rerun_test_ports(host="127.0.0.1")
        # Use dynamically allocated non-default ports in smoke tests to avoid collisions.
        # TODO: Consider supporting cleanup/termination of stale rerun processes when ports are occupied.
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
    """Return physics config and expected backend substring for the given backend kind.

    Uses cartpole preset instance so we work whether presets are class or instance attributes.
    Fallback: build PhysxCfg/NewtonCfg in-test if preset does not expose that backend.
    """
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


def _resolve_case(visualizer_kind: str, backend_kind: str):
    """Resolve (env_cfg, expected_visualizer_cls, expected_backend_substring) for one smoke test.

    Uses cartpole scene for all combinations (works with both PhysX and Newton).
    """
    scene_cfg = CartpoleSceneCfg(num_envs=1, env_spacing=1.0)
    viz_cfg, expected_viz_cls = _get_visualizer_cfg(visualizer_kind)
    physics_cfg, expected_backend = _get_physics_cfg(backend_kind)

    cfg = _SmokeEnvCfg()
    cfg.scene = scene_cfg
    cfg.sim = SimulationCfg(
        dt=0.005,
        render_interval=2,
        visualizer_cfgs=viz_cfg,
        physics=physics_cfg,
    )
    return cfg, expected_viz_cls, expected_backend


def _run_smoke_test(cfg, expected_visualizer_cls, expected_backend: str, caplog) -> None:
    """Run smoke steps and assert no visualizer errors; optionally no warnings (see ASSERT_VISUALIZER_WARNINGS)."""
    env = None
    try:
        sim_utils.create_new_stage()
        env = _SmokeEnv(cfg=cfg)
        backend_name = env.sim.physics_manager.__name__.lower()
        assert expected_backend in backend_name, (
            f"Expected physics backend containing {expected_backend!r}, got {backend_name!r}"
        )
        env.sim.set_setting("/isaaclab/render/rtx_sensors", True)
        env.sim._app_control_on_stop_handle = None  # type: ignore[attr-defined]

        actions = torch.zeros((env.num_envs, 0), device=env.device)
        with caplog.at_level(logging.WARNING):
            env.reset()
            assert env.sim.visualizers
            assert isinstance(env.sim.visualizers[0], expected_visualizer_cls)
            for _ in range(_SMOKE_STEPS):
                env.step(action=actions)

        # Always fail on errors
        error_logs = [
            r for r in caplog.records if r.levelno >= logging.ERROR and r.name.startswith(_VIS_LOGGER_PREFIXES)
        ]
        assert not error_logs, "Visualizer emitted error logs during smoke stepping: " + "; ".join(
            f"{r.name}: {r.message}" for r in error_logs
        )

        # Optionally fail on warnings
        if ASSERT_VISUALIZER_WARNINGS:
            warning_logs = [
                r for r in caplog.records if r.levelno >= logging.WARNING and r.name.startswith(_VIS_LOGGER_PREFIXES)
            ]
            assert not warning_logs, "Visualizer emitted warning logs during smoke stepping: " + "; ".join(
                f"{r.name}: {r.message}" for r in warning_logs
            )
    finally:
        if env is not None:
            env.close()
        else:
            SimulationContext.clear_instance()


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("visualizer_kind", ["kit", "newton", "rerun", "viser"])
@pytest.mark.parametrize("backend_kind", ["physx", "newton"])
def test_visualizer_backend_smoke(visualizer_kind: str, backend_kind: str, caplog):
    """Smoke test each (visualizer, backend) pair; assert no errors (optionally no warnings)."""
    cfg, expected_viz_cls, expected_backend = _resolve_case(visualizer_kind, backend_kind)
    _run_smoke_test(cfg, expected_viz_cls, expected_backend, caplog)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--maxfail=1"])
