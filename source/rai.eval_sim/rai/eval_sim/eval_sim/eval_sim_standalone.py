# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs import ManagerBasedEnv
from rai.eval_sim.eval_sim import EvalSim, EvalSimCfg
from rai.eval_sim.utils import log_warn, zero_actions


class EvalSimStandalone(EvalSim):
    def __init__(self, cfg: EvalSimCfg) -> None:
        super().__init__(cfg)

        self.load()

        # Take a zero action to initialize the environment and get the first observation
        self.obs, *_ = ManagerBasedEnv.step(self.env, action=zero_actions(self.env))

        if not self.cfg.enable_ros:
            log_warn("EvalSimCfg.enable_ros is set to False. ROS will be enabled for standalone mode.")
        self.enable_ros()

    def step_deployment(self):
        """Step the ROS 2 communication and step the environment."""
        # profiling at decimation rate
        self.log_wallclock_time()
        # publish observations through ROS
        if self.ros_manager and self._ros_enabled:
            self.ros_manager.publish(self.obs)

            # subscribe to action input through ROS
            action = self.ros_manager.subscribe()
        else:
            action = zero_actions(self.env)

        # TODO: need to handle control_substeps (decimation in Isaac Lab)
        # we only want to step the environment if we have a new action
        if action is not None:
            self.obs, *_ = ManagerBasedEnv.step(self.env, action=action)

    def reload(self):
        """Reload EvalSim configs and the currently loaded environment."""
        self.pause()

        self.load_default_configs()

        self.set_env_cfg(self.cfg.env_cfg)

        self.clear()

        self.load()

        self.enable_ros()

        self.play()
