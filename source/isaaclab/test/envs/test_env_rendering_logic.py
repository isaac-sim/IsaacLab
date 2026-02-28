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
from isaaclab.visualizers.kit_visualizer import KitVisualizer
from isaaclab.visualizers.kit_visualizer_cfg import KitVisualizerCfg


@configclass
class EmptyManagerCfg:
    """Empty specifications for the environment."""

    pass


def create_manager_based_env(render_interval: int):
    """Create a manager based environment."""

    @configclass
    class EnvCfg(ManagerBasedEnvCfg):
        """Configuration for the test environment."""

        decimation: int = 4
        episode_length_s: float = 100.0
        sim: SimulationCfg = SimulationCfg(
            dt=0.005, render_interval=render_interval, visualizer_cfgs=KitVisualizerCfg()
        )
        scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=1.0)
        actions: EmptyManagerCfg = EmptyManagerCfg()
        observations: EmptyManagerCfg = EmptyManagerCfg()

    return ManagerBasedEnv(cfg=EnvCfg())


def create_manager_based_rl_env(render_interval: int):
    """Create a manager based RL environment."""

    @configclass
    class EnvCfg(ManagerBasedRLEnvCfg):
        """Configuration for the test environment."""

        decimation: int = 4
        episode_length_s: float = 100.0
        sim: SimulationCfg = SimulationCfg(
            dt=0.005, render_interval=render_interval, visualizer_cfgs=KitVisualizerCfg()
        )
        scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=1.0)
        actions: EmptyManagerCfg = EmptyManagerCfg()
        observations: EmptyManagerCfg = EmptyManagerCfg()
        rewards: EmptyManagerCfg = EmptyManagerCfg()
        terminations: EmptyManagerCfg = EmptyManagerCfg()

    return ManagerBasedRLEnv(cfg=EnvCfg())


def create_direct_rl_env(render_interval: int):
    """Create a direct RL environment."""

    @configclass
    class EnvCfg(DirectRLEnvCfg):
        """Configuration for the test environment."""

        decimation: int = 4
        action_space: int = 0
        observation_space: int = 0
        episode_length_s: float = 100.0
        sim: SimulationCfg = SimulationCfg(
            dt=0.005, render_interval=render_interval, visualizer_cfgs=KitVisualizerCfg()
        )
        scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=1.0)

    class Env(DirectRLEnv):
        """Test environment."""

        def _pre_physics_step(self, actions):
            pass

        def _apply_action(self):
            pass

        def _get_observations(self):
            return {}

        def _get_rewards(self):
            return {}

        def _get_dones(self):
            return torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)

    return Env(cfg=EnvCfg())


@pytest.fixture
def physics_callback():
    """Create a physics callback for tracking physics steps."""
    physics_time = 0.0
    num_physics_steps = 0

    def callback(dt):
        nonlocal physics_time, num_physics_steps
        physics_time += dt
        num_physics_steps += 1

    return callback, lambda: (physics_time, num_physics_steps)


@pytest.fixture
def render_callback():
    """Create a render callback for tracking render steps."""
    render_time = 0.0
    num_render_steps = 0

    def callback(dt):
        nonlocal render_time, num_render_steps
        render_time += dt
        num_render_steps += 1

    return callback, lambda: (render_time, num_render_steps)


@pytest.mark.parametrize("env_type", ["manager_based_env", "manager_based_rl_env", "direct_rl_env"])
@pytest.mark.parametrize("render_interval", [1, 2, 4, 8, 10])
def test_env_rendering_logic(env_type, render_interval, physics_callback, render_callback):
    """Test the rendering logic of the different environment workflows."""
    physics_cb, get_physics_stats = physics_callback
    render_cb, get_render_stats = render_callback

    env = None
    physics_handle = None
    original_step = None
    viz = None

    try:
        # create a new stage
        sim_utils.create_new_stage()

        # create environment
        if env_type == "manager_based_env":
            env = create_manager_based_env(render_interval)
        elif env_type == "manager_based_rl_env":
            env = create_manager_based_rl_env(render_interval)
        else:
            env = create_direct_rl_env(render_interval)

        # enable the flag to render the environment
        # note: this is only done for the unit testing to "fake" camera rendering.
        #   normally this is set to True when cameras are created.
        env.sim.set_setting("/isaaclab/render/rtx_sensors", True)

        # disable the app from shutting down when the environment is closed
        # FIXME: Why is this needed in this test but not in the other tests?
        #   Without it, the test will exit after the environment is closed
        env.sim._app_control_on_stop_handle = None  # type: ignore

        # Reset to initialize visualizers (they're created lazily in reset())
        env.reset()

        # Ensure the default Kit visualizer is active for rendering callbacks.
        assert isinstance(env.sim.visualizers[0], KitVisualizer)

        # add physics callback via physics manager (IsaacEvents is PhysX-specific)
        physics_handle = env.sim.physics_manager.register_callback(
            physics_cb, IsaacEvents.POST_PHYSICS_STEP, name="physics_step"
        )

        # Wrap visualizer step to track render calls
        viz = env.sim.visualizers[0]
        original_step = viz.step
        render_dt = env.cfg.sim.dt * env.cfg.sim.render_interval

        def wrapped_step(dt):
            original_step(dt)
            render_cb(render_dt)

        viz.step = wrapped_step

        # create a zero action tensor for stepping the environment
        actions = torch.zeros((env.num_envs, 0), device=env.device)

        # run the environment and check the rendering logic
        for i in range(50):
            # apply zero actions
            env.step(action=actions)

            # check that we have completed the correct number of physics steps
            _, num_physics_steps = get_physics_stats()
            assert num_physics_steps == (i + 1) * env.cfg.decimation, "Physics steps mismatch"
            # check that we have simulated physics for the correct amount of time
            physics_time, _ = get_physics_stats()
            assert abs(physics_time - num_physics_steps * env.cfg.sim.dt) < 1e-6, "Physics time mismatch"

            # check that we have completed the correct number of rendering steps
            _, num_render_steps = get_render_stats()
            assert num_render_steps == (i + 1) * env.cfg.decimation // env.cfg.sim.render_interval, (
                "Render steps mismatch"
            )
            # check that we have rendered for the correct amount of time
            render_time, _ = get_render_stats()
            assert abs(render_time - num_render_steps * env.cfg.sim.dt * env.cfg.sim.render_interval) < 1e-6, (
                "Render time mismatch"
            )

    finally:
        # Restore original step method
        if viz is not None and original_step is not None:
            viz.step = original_step
        # Deregister physics callback
        if physics_handle is not None:
            physics_handle.deregister()
        # Close environment (this also clears SimulationContext)
        if env is not None:
            env.close()
        else:
            # If env creation failed, still clear the singleton
            SimulationContext.clear_instance()
