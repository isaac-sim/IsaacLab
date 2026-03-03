"""Smoke test visualizer stepping and error logging."""

from isaaclab.app import AppLauncher

# launch Kit app
simulation_app = AppLauncher(headless=True, enable_cameras=True).app

import logging
import shutil

import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass
from isaaclab_newton.visualizers import NewtonVisualizer, NewtonVisualizerCfg, RerunVisualizer, RerunVisualizerCfg
from isaaclab_physx.visualizers import KitVisualizer, KitVisualizerCfg

_SMOKE_STEPS = 4
_VIS_LOGGER_PREFIXES = (
    "isaaclab.visualizers",
    "isaaclab_physx.visualizers",
    "isaaclab_newton.visualizers",
)


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


def _resolve_visualizer_case(visualizer_kind: str):
    if visualizer_kind == "newton":
        pytest.importorskip("newton")
        cfg = _SmokeEnvCfg()
        cfg.sim = SimulationCfg(dt=0.005, render_interval=2, visualizer_cfgs=NewtonVisualizerCfg())
        return cfg, NewtonVisualizer

    if visualizer_kind == "rerun":
        pytest.importorskip("newton")
        pytest.importorskip("rerun")
        if shutil.which("rerun") is None:
            pytest.skip("rerun binary not found in PATH")
        cfg = _SmokeEnvCfg()
        cfg.sim = SimulationCfg(
            dt=0.005,
            render_interval=2,
            visualizer_cfgs=RerunVisualizerCfg(bind_address="127.0.0.1", open_browser=False),
        )
        return cfg, RerunVisualizer

    cfg = _SmokeEnvCfg()
    return cfg, KitVisualizer


def _run_visualizer_smoke_test(
    cfg, expected_visualizer_cls, caplog, *, capture_level: int, assert_min_level: int, assert_label: str
) -> None:
    env = None
    try:
        sim_utils.create_new_stage()
        env = _SmokeEnv(cfg=cfg)
        env.sim.set_setting("/isaaclab/render/rtx_sensors", True)
        env.sim._app_control_on_stop_handle = None  # type: ignore[attr-defined]

        actions = torch.zeros((env.num_envs, 0), device=env.device)
        with caplog.at_level(capture_level):
            env.reset()
            assert env.sim.visualizers
            assert isinstance(env.sim.visualizers[0], expected_visualizer_cls)
            for _ in range(_SMOKE_STEPS):
                env.step(action=actions)

        visualizer_logs = [
            r
            for r in caplog.records
            if r.levelno >= assert_min_level
            and r.name.startswith(_VIS_LOGGER_PREFIXES)
        ]
        assert not visualizer_logs, (
            f"Visualizer emitted {assert_label} logs during smoke stepping: "
            + "; ".join(f"{r.name}: {r.message}" for r in visualizer_logs)
        )
    finally:
        if env is not None:
            env.close()
        else:
            SimulationContext.clear_instance()


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("visualizer_kind", ["kit", "newton", "rerun"])
def test_visualizer_steps_without_visualizer_errors(visualizer_kind: str, caplog):
    # This module launches exactly one SimulationApp (module scope above),
    # and reuses it across all visualizer smoke cases.
    cfg, expected_cls = _resolve_visualizer_case(visualizer_kind)
    _run_visualizer_smoke_test(
        cfg,
        expected_cls,
        caplog,
        capture_level=logging.ERROR,
        assert_min_level=logging.ERROR,
        assert_label="error",
    )


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("visualizer_kind", ["kit", "newton", "rerun"])
def test_visualizer_steps_without_visualizer_warnings(visualizer_kind: str, caplog):
    cfg, expected_cls = _resolve_visualizer_case(visualizer_kind)
    _run_visualizer_smoke_test(
        cfg,
        expected_cls,
        caplog,
        capture_level=logging.WARNING,
        assert_min_level=logging.WARNING,
        assert_label="warning",
    )
