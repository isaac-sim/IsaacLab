# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.app import AppLauncher, run_tests

HEADLESS = True

# if not AppLauncher.instance():
simulation_app = AppLauncher(headless=HEADLESS).app

"""Rest of imports follows"""

import torch
import unittest

from omni.isaac.core.utils.types import ArticulationActions
from omni.isaac.lab.actuators import IdealPDActuator, IdealPDActuatorCfg
from omni.isaac.lab.utils.buffers import DelayBuffer


class TestIdealPDActuator(unittest.TestCase):

    def test_ideal_pd_actuator_init_minimum(self):
        """Test initialization of ideal pd actuator with minimum configuration."""
        for num_envs in (1, 2):
            for num_joints in (1, 2):
                for device in ("cuda:0", "cpu"):
                    for usd_default in (False, True):
                        with self.subTest(
                            num_envs=num_envs, num_joints=num_joints, device=device, usd_default=usd_default
                        ):
                            joint_names = [f"joint_{d}" for d in range(num_joints)]
                            joint_ids = [d for d in range(num_joints)]
                            stiffness = None if usd_default else 200
                            damping = None if usd_default else 10
                            friction = None if usd_default else 0.1
                            armature = None if usd_default else 0.2

                            actuator_cfg = IdealPDActuatorCfg(
                                joint_names_expr=joint_names,
                                stiffness=stiffness,
                                damping=damping,
                                armature=armature,
                                friction=friction,
                            )
                            # assume Articulation class:
                            #   - finds joints (names and ids) associate with the provided joint_names_expr

                            # faux usd defaults
                            stiffness_default = 300
                            damping_default = 20
                            friction_default = 0.0
                            armature_default = 0.0

                            actuator = actuator_cfg.class_type(
                                actuator_cfg,
                                joint_names=joint_names,
                                joint_ids=joint_ids,
                                num_envs=num_envs,
                                device=device,
                                stiffness=stiffness_default,
                                damping=damping_default,
                                friction=friction_default,
                                armature=armature_default,
                            )

                            # check device and shape
                            torch.testing.assert_close(
                                actuator.computed_effort, torch.zeros(num_envs, num_joints, device=device)
                            )
                            torch.testing.assert_close(
                                actuator.applied_effort, torch.zeros(num_envs, num_joints, device=device)
                            )

                            torch.testing.assert_close(
                                actuator.effort_limit,
                                torch.inf * torch.ones(num_envs, num_joints, device=device),
                            )
                            torch.testing.assert_close(
                                actuator.velocity_limit, torch.inf * torch.ones(num_envs, num_joints, device=device)
                            )

                            if not usd_default:
                                torch.testing.assert_close(
                                    actuator.stiffness, stiffness * torch.ones(num_envs, num_joints, device=device)
                                )
                                torch.testing.assert_close(
                                    actuator.damping, damping * torch.ones(num_envs, num_joints, device=device)
                                )
                                torch.testing.assert_close(
                                    actuator.armature, armature * torch.ones(num_envs, num_joints, device=device)
                                )
                                torch.testing.assert_close(
                                    actuator.friction, friction * torch.ones(num_envs, num_joints, device=device)
                                )
                            else:
                                torch.testing.assert_close(
                                    actuator.stiffness,
                                    stiffness_default * torch.ones(num_envs, num_joints, device=device),
                                )
                                torch.testing.assert_close(
                                    actuator.damping,
                                    damping_default * torch.ones(num_envs, num_joints, device=device),
                                )
                                torch.testing.assert_close(
                                    actuator.armature,
                                    armature_default * torch.ones(num_envs, num_joints, device=device),
                                )
                                torch.testing.assert_close(
                                    actuator.friction,
                                    friction_default * torch.ones(num_envs, num_joints, device=device),
                                )

    def test_ideal_pd_compute(self):
        """Test the computation of the ideal pd actuator."""
        for num_envs in (1, 2):
            for num_joints in (1, 2):
                for device in ("cuda:0", "cpu"):
                    for effort_lim in (None, 300):
                        with self.subTest(
                            num_envs=num_envs, num_joints=num_joints, device=device, effort_lim=effort_lim
                        ):
                            joint_names = [f"joint_{d}" for d in range(num_joints)]
                            joint_ids = [d for d in range(num_joints)]
                            stiffness = 200
                            damping = 10
                            actuator_cfg = IdealPDActuatorCfg(
                                joint_names_expr=joint_names,
                                stiffness=stiffness,
                                damping=damping,
                                effort_limit=effort_lim,
                            )

                            actuator = actuator_cfg.class_type(
                                actuator_cfg,
                                joint_names=joint_names,
                                joint_ids=joint_ids,
                                num_envs=num_envs,
                                device=device,
                                stiffness=actuator_cfg.stiffness,
                                damping=actuator_cfg.damping,
                            )
                            desired_pos = 10.0
                            desired_vel = 0.1
                            measured_joint_pos = 1.0
                            measured_joint_vel = -0.1

                            desired_control_action = ArticulationActions()
                            desired_control_action.joint_positions = desired_pos * torch.ones(
                                num_envs, num_joints, device=device
                            )
                            desired_control_action.joint_velocities = desired_vel * torch.ones(
                                num_envs, num_joints, device=device
                            )
                            desired_control_action.joint_efforts = torch.zeros(num_envs, num_joints, device=device)

                            expected_comp_joint_effort = stiffness * (desired_pos - measured_joint_pos) + damping * (
                                desired_vel - measured_joint_vel
                            )

                            computed_control_action = actuator.compute(
                                desired_control_action,
                                measured_joint_pos * torch.ones(num_envs, num_joints, device=device),
                                measured_joint_vel * torch.ones(num_envs, num_joints, device=device),
                            )

                            torch.testing.assert_close(
                                expected_comp_joint_effort * torch.ones(num_envs, num_joints, device=device),
                                actuator.computed_effort,
                            )

                            if effort_lim is None:
                                torch.testing.assert_close(
                                    expected_comp_joint_effort * torch.ones(num_envs, num_joints, device=device),
                                    actuator.applied_effort,
                                )
                            else:
                                torch.testing.assert_close(
                                    effort_lim * torch.ones(num_envs, num_joints, device=device),
                                    actuator.applied_effort,
                                )
                            torch.testing.assert_close(
                                actuator.applied_effort,
                                computed_control_action.joint_efforts,
                            )

    def test_ideal_pd_actuator_init_delay(self):
        """Test initialization of ideal pd actuator with delay."""
        for num_envs in (1, 2):
            for num_joints in (1, 2):
                for device in ("cuda:0", "cpu"):
                    with self.subTest(num_envs=num_envs, num_joints=num_joints, device=device):
                        joint_names = [f"joint_{d}" for d in range(num_joints)]
                        joint_ids = [d for d in range(num_joints)]
                        stiffness = 200
                        damping = 10
                        effort_limit = 60.0
                        velocity_limit = 50
                        motor_strength = (1.0, 1.0)
                        delay = 5

                        actuator_cfg = IdealPDActuatorCfg(
                            joint_names_expr=joint_names,
                            stiffness=stiffness,
                            damping=damping,
                            effort_limit=effort_limit,
                            velocity_limit=velocity_limit,
                            motor_strength=motor_strength,
                            min_delay=delay,
                            max_delay=delay,
                        )

                        # assume Articulation class:
                        #   - finds joints (names and ids) associate with the provided joint_names_expr

                        actuator = actuator_cfg.class_type(
                            actuator_cfg,
                            joint_names=joint_names,
                            joint_ids=joint_ids,
                            num_envs=num_envs,
                            device=device,
                        )
                        assert isinstance(actuator, IdealPDActuator)
                        # check device and shape
                        torch.testing.assert_close(
                            actuator.computed_effort, torch.zeros(num_envs, num_joints, device=device)
                        )
                        torch.testing.assert_close(
                            actuator.applied_effort, torch.zeros(num_envs, num_joints, device=device)
                        )
                        torch.testing.assert_close(
                            actuator.effort_limit,
                            effort_limit * torch.ones(num_envs, num_joints, device=device),
                        )
                        torch.testing.assert_close(
                            actuator.velocity_limit, velocity_limit * torch.ones(num_envs, num_joints, device=device)
                        )

                        # check delay buffers
                        assert isinstance(actuator.positions_delay_buffer, DelayBuffer)
                        assert isinstance(actuator.velocities_delay_buffer, DelayBuffer)
                        assert isinstance(actuator.efforts_delay_buffer, DelayBuffer)

                        assert actuator.positions_delay_buffer.history_length == delay
                        assert actuator.velocities_delay_buffer.history_length == delay
                        assert actuator.efforts_delay_buffer.history_length == delay

                        # check motor strength
                        torch.testing.assert_close(
                            actuator._current_motor_strength, torch.ones((num_envs, 1), device=device)
                        )

    def test_delay_pd_actuator_compute(self):
        """Test the computation of the delay pd actuator."""
        for num_envs in (1, 2):
            for num_joints in (1, 2):
                for device in ("cuda:0", "cpu"):
                    for effort_lim in (None, 80):
                        for motor_strength_scalar in (1.0, 0.0):
                            with self.subTest(
                                num_envs=num_envs,
                                num_joints=num_joints,
                                device=device,
                                effort_lim=effort_lim,
                                motor_strength_scalar=motor_strength_scalar,
                            ):
                                joint_names = [f"joint_{d}" for d in range(num_joints)]
                                joint_ids = [d for d in range(num_joints)]
                                stiffness = 20
                                damping = 1
                                effort_limit = effort_lim
                                velocity_limit = 50
                                motor_strength = (motor_strength_scalar, motor_strength_scalar)
                                delay = 3
                                # configure actuator
                                actuator_cfg = IdealPDActuatorCfg(
                                    joint_names_expr=joint_names,
                                    stiffness=stiffness,
                                    damping=damping,
                                    effort_limit=effort_limit,
                                    velocity_limit=velocity_limit,
                                    motor_strength=motor_strength,
                                    min_delay=delay,
                                    max_delay=delay,
                                )
                                # instantiate actuator
                                actuator = actuator_cfg.class_type(
                                    actuator_cfg,
                                    joint_names=joint_names,
                                    joint_ids=joint_ids,
                                    num_envs=num_envs,
                                    device=device,
                                    stiffness=actuator_cfg.stiffness,
                                    damping=actuator_cfg.damping,
                                )
                                # requires a restart to create the delay
                                actuator.reset(range(num_envs))

                                desired_pos = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
                                desired_vel = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
                                feedforward_effort = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

                                measured_joint_pos = 1.0
                                measured_joint_vel = 0.1

                                # check delay buffer filling properly
                                for i in range(delay + 4):

                                    # check motor strength
                                    torch.testing.assert_close(
                                        motor_strength[0] * torch.ones(num_envs, 1, device=device),
                                        actuator._current_motor_strength,
                                    )

                                    desired_control_action = ArticulationActions()
                                    desired_control_action.joint_positions = desired_pos[i] * torch.ones(
                                        num_envs, num_joints, device=device
                                    )
                                    desired_control_action.joint_velocities = desired_vel[i] * torch.ones(
                                        num_envs, num_joints, device=device
                                    )
                                    desired_control_action.joint_efforts = feedforward_effort[i] * torch.ones(
                                        num_envs, num_joints, device=device
                                    )

                                    computed_control_action = actuator.compute(
                                        desired_control_action,
                                        measured_joint_pos * torch.ones(num_envs, num_joints, device=device),
                                        measured_joint_vel * torch.ones(num_envs, num_joints, device=device),
                                    )

                                    if i <= delay:
                                        expect = motor_strength[0] * (
                                            stiffness * (desired_pos[0] - measured_joint_pos)
                                            + damping * (desired_vel[0] - measured_joint_vel)
                                            + feedforward_effort[0]
                                        )
                                    else:
                                        expect = motor_strength[0] * (
                                            stiffness * (desired_pos[i - delay] - measured_joint_pos)
                                            + damping * (desired_vel[i - delay] - measured_joint_vel)
                                            + feedforward_effort[i - delay]
                                        )

                                    if effort_lim is not None:
                                        expect_apply = min(expect, effort_lim)
                                    else:
                                        expect_apply = expect

                                    torch.testing.assert_close(
                                        expect * torch.ones(num_envs, num_joints, device=device),
                                        actuator.computed_effort,
                                    )
                                    torch.testing.assert_close(
                                        expect_apply * torch.ones(num_envs, num_joints, device=device),
                                        actuator.applied_effort,
                                    )
                                    torch.testing.assert_close(
                                        actuator.applied_effort,
                                        computed_control_action.joint_efforts,
                                    )
                                # check reset single env
                                actuator.reset([0])
                                assert actuator.positions_delay_buffer._circular_buffer._num_pushes[0] == 0
                                assert actuator.velocities_delay_buffer._circular_buffer._num_pushes[0] == 0
                                assert actuator.efforts_delay_buffer._circular_buffer._num_pushes[0] == 0
                                if num_envs > 1:
                                    assert actuator.positions_delay_buffer._circular_buffer._num_pushes[1] == i + 1
                                    assert actuator.velocities_delay_buffer._circular_buffer._num_pushes[1] == i + 1
                                    assert actuator.efforts_delay_buffer._circular_buffer._num_pushes[1] == i + 1
                                # check actuator reset all
                                actuator.reset(range(num_envs))


if __name__ == "__main__":
    run_tests()
