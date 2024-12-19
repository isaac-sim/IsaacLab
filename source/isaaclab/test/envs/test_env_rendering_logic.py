# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# launch omniverse app
# need to set "enable_cameras" true to be able to do rendering tests
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import unittest

import omni.usd

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
        sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=render_interval)
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
        sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=render_interval)
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
        sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=render_interval)
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


class TestEnvRenderingLogic(unittest.TestCase):
    """Test the rendering logic of the different environment workflows."""

    def _physics_callback(self, dt):
        # called at every physics step
        self.physics_time += dt
        self.num_physics_steps += 1

    def _render_callback(self, event):
        # called at every render step
        self.render_time += event.payload["dt"]
        self.num_render_steps += 1

    def test_env_rendering_logic(self):
        for env_type in ["manager_based_env", "manager_based_rl_env", "direct_rl_env"]:
            for render_interval in [1, 2, 4, 8, 10]:
                with self.subTest(env_type=env_type, render_interval=render_interval):
                    # time tracking variables
                    self.physics_time = 0.0
                    self.render_time = 0.0
                    # step tracking variables
                    self.num_physics_steps = 0
                    self.num_render_steps = 0

                    # create a new stage
                    omni.usd.get_context().new_stage()
                    try:
                        # create environment
                        if env_type == "manager_based_env":
                            env = create_manager_based_env(render_interval)
                        elif env_type == "manager_based_rl_env":
                            env = create_manager_based_rl_env(render_interval)
                        else:
                            env = create_direct_rl_env(render_interval)
                    except Exception as e:
                        if "env" in locals() and hasattr(env, "_is_closed"):
                            env.close()
                        else:
                            if hasattr(e, "obj") and hasattr(e.obj, "_is_closed"):
                                e.obj.close()
                        self.fail(f"Failed to set-up the environment {env_type}. Error: {e}")

                    # enable the flag to render the environment
                    # note: this is only done for the unit testing to "fake" camera rendering.
                    #   normally this is set to True when cameras are created.
                    env.sim.set_setting("/isaaclab/render/rtx_sensors", True)

                    # disable the app from shutting down when the environment is closed
                    # FIXME: Why is this needed in this test but not in the other tests?
                    #   Without it, the test will exit after the environment is closed
                    env.sim._app_control_on_stop_handle = None  # type: ignore

                    # check that we are in partial rendering mode for the environment
                    # this is enabled due to app launcher setting "enable_cameras=True"
                    self.assertEqual(env.sim.render_mode, SimulationContext.RenderMode.PARTIAL_RENDERING)

                    # add physics and render callbacks
                    env.sim.add_physics_callback("physics_step", self._physics_callback)
                    env.sim.add_render_callback("render_step", self._render_callback)

                    # create a zero action tensor for stepping the environment
                    actions = torch.zeros((env.num_envs, 0), device=env.device)

                    # run the environment and check the rendering logic
                    for i in range(50):
                        # apply zero actions
                        env.step(action=actions)

                        # check that we have completed the correct number of physics steps
                        self.assertEqual(
                            self.num_physics_steps, (i + 1) * env.cfg.decimation, msg="Physics steps mismatch"
                        )
                        # check that we have simulated physics for the correct amount of time
                        self.assertAlmostEqual(
                            self.physics_time, self.num_physics_steps * env.cfg.sim.dt, msg="Physics time mismatch"
                        )

                        # check that we have completed the correct number of rendering steps
                        self.assertEqual(
                            self.num_render_steps,
                            (i + 1) * env.cfg.decimation // env.cfg.sim.render_interval,
                            msg="Render steps mismatch",
                        )
                        # check that we have rendered for the correct amount of time
                        self.assertAlmostEqual(
                            self.render_time,
                            self.num_render_steps * env.cfg.sim.dt * env.cfg.sim.render_interval,
                            msg="Render time mismatch",
                        )

                    # close the environment
                    env.close()


if __name__ == "__main__":
    run_tests()
