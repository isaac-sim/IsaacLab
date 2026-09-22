# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch Kit app
# need to set "enable_cameras" true to be able to do rendering tests
simulation_app = AppLauncher(headless=True, enable_cameras=True).app

"""Rest everything follows."""

import pytest
import torch
from isaaclab_physx.physics import IsaacEvents
from isaaclab_visualizers.kit import KitVisualizer, KitVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.envs import (
    DirectRLEnv,
    DirectRLEnvCfg,
    ManagerBasedEnv,
    ManagerBasedEnvCfg,
    ManagerBasedRLEnv,
    ManagerBasedRLEnvCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass

pytestmark = [pytest.mark.integration, pytest.mark.rendering, pytest.mark.isaacsim_ci]

DT = 0.005
DECIMATION = 4
ENV_TYPES = ["manager_based_env", "manager_based_rl_env", "direct_rl_env"]


@configclass
class EmptyManagerCfg:
    """Empty specifications for the environment."""

    pass


class _DirectEnv(DirectRLEnv):
    """Direct environment without assets whose episodes time out after ``max_episode_length`` steps."""

    def _pre_physics_step(self, actions):
        pass

    def _apply_action(self):
        pass

    def _get_observations(self):
        return {}

    def _get_rewards(self):
        return {}

    def _get_dones(self):
        time_out = self.episode_length_buf >= self.max_episode_length
        return torch.zeros_like(time_out), time_out


def create_env(env_type: str, render_interval: int, episode_length_steps: int | None = None, visualizer: bool = True):
    """Create an empty environment of the given workflow type.

    Args:
        env_type: One of :data:`ENV_TYPES`.
        render_interval: Render interval in physics steps.
        episode_length_steps: If provided, episodes time out after this many env steps (direct envs only).
        visualizer: Whether to attach the Kit visualizer. Without it, offscreen render is the only render path.
    """
    sim = SimulationCfg(
        dt=DT, render_interval=render_interval, visualizer_cfgs=KitVisualizerCfg() if visualizer else []
    )
    scene = InteractiveSceneCfg(num_envs=1, env_spacing=1.0)
    episode_length_s = episode_length_steps * DT * DECIMATION if episode_length_steps is not None else 100.0
    if env_type == "manager_based_env":
        cfg = ManagerBasedEnvCfg(
            decimation=DECIMATION, sim=sim, scene=scene, actions=EmptyManagerCfg(), observations=EmptyManagerCfg()
        )
        return ManagerBasedEnv(cfg=cfg)
    if env_type == "manager_based_rl_env":
        cfg = ManagerBasedRLEnvCfg(
            decimation=DECIMATION,
            episode_length_s=episode_length_s,
            sim=sim,
            scene=scene,
            actions=EmptyManagerCfg(),
            observations=EmptyManagerCfg(),
            rewards=EmptyManagerCfg(),
            terminations=EmptyManagerCfg(),
        )
        return ManagerBasedRLEnv(cfg=cfg)
    cfg = DirectRLEnvCfg(
        decimation=DECIMATION,
        action_space=0,
        observation_space=0,
        episode_length_s=episode_length_s,
        sim=sim,
        scene=scene,
    )
    return _DirectEnv(cfg=cfg)


class _StepCounters:
    """Counts the physics steps and the Kit visualizer renders of an environment."""

    def __init__(self, env):
        self.env = env
        self.num_physics_steps = 0
        self.num_render_steps = 0
        self._physics_handle = env.sim.physics_manager.register_callback(
            self._on_physics_step, IsaacEvents.POST_PHYSICS_STEP, name="physics_step"
        )
        self._viz = env.sim.visualizers[0]
        assert isinstance(self._viz, KitVisualizer)
        self._original_step = self._viz.step
        self._viz.step = self._on_render_step

    def _on_physics_step(self, dt):
        self.num_physics_steps += 1

    def _on_render_step(self, dt):
        self._original_step(dt)
        self.num_render_steps += 1

    def close(self):
        self._viz.step = self._original_step
        self._physics_handle.deregister()


@pytest.fixture
def make_env():
    """Create RTX-rendering environments with step counters and close them at teardown."""
    created = []

    def _make(env_type: str, render_interval: int, **kwargs) -> tuple:
        sim_utils.create_new_stage()
        env = create_env(env_type, render_interval, **kwargs)
        # fake camera rendering so the render cadence is exercised without sensors
        env.sim.set_setting("/isaaclab/render/rtx_sensors", True)
        # keep the app alive when the environment is closed
        env.sim._app_control_on_stop_handle = None  # type: ignore
        # visualizers are created lazily in reset()
        env.reset()
        counters = _StepCounters(env)
        created.append((env, counters))
        return env, counters

    yield _make

    for env, counters in created:
        counters.close()
        env.close()
    if not created:
        SimulationContext.clear_instance()


def _step(env, num_steps: int):
    actions = torch.zeros((env.num_envs, 0), device=env.device)
    for _ in range(num_steps):
        env.step(action=actions)


@pytest.mark.parametrize("env_type", ENV_TYPES)
@pytest.mark.parametrize("render_interval", [1, 4, 10])
def test_env_rendering_logic(make_env, env_type, render_interval):
    """Physics advances ``decimation`` steps per env step and renders once per ``render_interval`` physics steps.

    A reset in between must force the next camera read to republish the renderer scene state.
    """
    env, counters = make_env(env_type, render_interval)
    for i in range(1, 11):
        _step(env, 1)
        assert counters.num_physics_steps == i * DECIMATION
        assert counters.num_render_steps == i * DECIMATION // render_interval

    env.sim.render_context._last_scene_state_step = 7
    env.reset()
    assert env.sim.render_context._last_scene_state_step is None


@pytest.mark.parametrize("env_type", ENV_TYPES)
def test_env_render_flag_mixed_steps(make_env, env_type):
    """Toggling ``render_enabled`` between steps skips Kit rendering while physics keeps advancing."""
    env, counters = make_env(env_type, render_interval=1)
    expected_render_steps = 0
    for i in range(10):
        env.render_enabled = i < 5
        _step(env, 1)
        if env.render_enabled:
            expected_render_steps += DECIMATION
        assert counters.num_physics_steps == (i + 1) * DECIMATION
        assert counters.num_render_steps == expected_render_steps


def test_env_render_false_with_resets(make_env):
    """``render_enabled=False`` also skips the post-reset re-renders during short episodes."""
    # 3-step episodes: resets occur at steps 3, 6, 9
    env, counters = make_env("direct_rl_env", render_interval=1, episode_length_steps=3)
    env.render_enabled = False
    _step(env, 10)
    assert counters.num_physics_steps == 10 * DECIMATION
    assert counters.num_render_steps == 0


def test_headless_offscreen_render_does_not_pump_kit_every_step():
    """Regression test for issue #6316.

    With headless video recording (offscreen render enabled) but no continuous-rendering consumer (GUI, RTX
    sensors, visualizers, XR), the decimation loop must NOT call :meth:`~isaaclab.sim.SimulationContext.render`
    (which pumps Kit's ``app.update()``). Frames are produced on demand only when :meth:`render` is explicitly
    called. Before the fix, ``is_rendering`` reported offscreen rendering as continuous rendering.
    """
    sim_utils.create_new_stage()
    env = create_env("manager_based_env", render_interval=1, visualizer=False)
    try:
        # simulate ``--video``; leave rtx_sensors False so offscreen is the only render reason
        env.sim.set_setting("/isaaclab/video/enabled", True)
        env.sim._app_control_on_stop_handle = None  # type: ignore
        env.reset()

        assert env.sim.has_offscreen_render
        assert not env.sim.is_rendering
        assert not env.sim.visualizers

        render_calls = 0
        original_render = env.sim.render

        def counting_render(*args, **kwargs):
            nonlocal render_calls
            render_calls += 1
            return original_render(*args, **kwargs)

        env.sim.render = counting_render  # type: ignore[method-assign]
        try:
            _step(env, 10)
            assert render_calls == 0
            # on-demand rendering (what RecordVideo does to grab a frame) still works
            env.sim.render()
            assert render_calls == 1
        finally:
            env.sim.render = original_render  # type: ignore[method-assign]
    finally:
        env.close()
