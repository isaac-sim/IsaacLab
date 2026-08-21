# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression test for Newton actuator authoring in manager-based clone paths.

The default G1 config uses ``ImplicitActuatorCfg`` for every group, which
intentionally skips ``NewtonActuator`` USD authoring. To exercise the
authoring path we override the scene's robot actuators with a single
explicit ``DCMotorCfg`` group covering all joints.
"""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import unittest  # noqa: E402

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg  # noqa: E402
from isaaclab_newton.physics import NewtonManager as SimulationManager  # noqa: E402

from isaaclab.actuators import DCMotorCfg  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.sim import SimulationCfg  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.core.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg  # noqa: E402

_NEWTON_CFG = NewtonCfg(
    solver_cfg=MJWarpSolverCfg(
        njmax=95,
        nconmax=10,
        cone="pyramidal",
        impratio=1,
        integrator="implicitfast",
    ),
    num_substeps=1,
    debug_mode=False,
)


class TestManagerBasedSceneNewtonActuatorAuthoring(unittest.TestCase):
    """Regression test for Newton actuator authoring in manager-based clone paths."""

    def test_newton_actuators_present_for_g1_manager_env(self):
        env_cfg = G1FlatEnvCfg()
        env_cfg.scene.num_envs = 1
        env_cfg.decimation = 1
        env_cfg.scene.contact_forces = None
        env_cfg.rewards.feet_air_time = None
        env_cfg.rewards.feet_slide = None
        env_cfg.terminations.base_contact = None
        env_cfg.sim = SimulationCfg(physics=_NEWTON_CFG, use_newton_actuators=True)
        # A single explicit group covering all joints is enough to exercise the
        # NewtonActuator authoring clone path; the group count is not the point.
        env_cfg.scene.robot.actuators = {
            "all": DCMotorCfg(
                joint_names_expr=[".*"],
                saturation_effort=300.0,
                actuator_effort_limit=300.0,
                actuator_velocity_limit=20.0,
                stiffness=150.0,
                damping=5.0,
            ),
        }
        env = ManagerBasedRLEnv(cfg=env_cfg)
        try:
            stage = env.unwrapped.sim.stage
            actuator_prim_count = sum(1 for prim in stage.Traverse() if prim.GetTypeName() == "NewtonActuator")
            self.assertGreater(
                actuator_prim_count,
                0,
                "Expected authored NewtonActuator prims in manager-based scene workflow.",
            )
            self.assertGreater(
                len(SimulationManager.get_model().actuators),
                0,
                "Expected Newton model actuators to be non-empty with use_newton_actuators=True.",
            )
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
