# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""
import random
import time
import unittest

import rclpy
from rai.eval_sim.eval_sim import EvalSimCfg
from rai.eval_sim.eval_sim.eval_sim_standalone import EvalSimStandalone
from rai.eval_sim.ros_manager import QOS_PROFILE
from rai.eval_sim.tasks import PER_ROBOT_EVAL_SIM_CFGS
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Empty


class TestEvalSimStandalone(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test case for all test methods.

        NOTE: We reuse the same EvalSim instance for all test cases, as
        if we call eval_sim.close(), the clean up of the stage is not properly
        done. As long as clear() and load() are called in setUp, we get a clean
        environment for each test case.
        """
        default_robot = "anymal"
        eval_sim_cfg = EvalSimCfg(
            sync_to_real_time=False,
            env_cfg=str(PER_ROBOT_EVAL_SIM_CFGS[default_robot][0]),
            ros_manager_cfg=str(PER_ROBOT_EVAL_SIM_CFGS[default_robot][1]),
            auto_save=False,
        )

        cls.eval_sim = EvalSimStandalone(cfg=eval_sim_cfg)

        # TODO: Minor - get this working with SceneEntityCfg
        joint_names = cls.eval_sim.env.scene["robot"].joint_names

        num_joints = len(joint_names)

        cls.msg = JointState(
            name=joint_names,
            position=[random.uniform(-1, 1) for _ in range(num_joints)],
            velocity=[random.uniform(-1, 1) for _ in range(num_joints)],
            effort=[random.uniform(-1, 1) for _ in range(num_joints)],
        )

    @classmethod
    def tearDownClass(cls):
        """Tear down the test case, cleaning up EvalSim.

        This method is called after all test methods have been executed.
        """
        cls.eval_sim.close()

    def setUp(self):
        self.eval_sim.clear()
        self.eval_sim.load()
        self.eval_sim.enable_ros()

        self._check_steps = 0
        self.node = Node("mock_controller")
        self.mock_controller_publisher = self.node.create_publisher(
            msg_type=JointState, topic="/joint_command", qos_profile=QOS_PROFILE
        )
        self.mock_controller_subscriber = self.node.create_subscription(
            msg_type=JointState,
            topic="/joint_state",
            qos_profile=QOS_PROFILE,
            callback=self._control_step,
        )

    def tearDown(self):
        # Note: Ideally, the clear would happen here, but this causes errors where
        # the SimulationContext is not properly cleaned up. Strangely, this works fine
        # if in setUp, before the load.
        self.node.destroy_node()

    def _control_step(self, msg: JointState):
        self._check_steps += 1

    def test_eval_sim_standalone_anymal(self):
        """Test the EvalSimStandalone with the Anymal robot.

        This test case publishes random joint states to the EvalSim environment.
        """
        start_ns = time.time_ns()
        prev_ns = start_ns
        num_steps = 1000

        for _ in range(num_steps):
            self.mock_controller_publisher.publish(self.msg)
            self.eval_sim.step()
            rclpy.spin_once(node=self.node, timeout_sec=0.1)
            curr_ns = time.time_ns()
            delta_ns = curr_ns - prev_ns
            prev_ns = curr_ns
            self.assertTrue(delta_ns < 1e9)

        # check the messages were published
        self.assertTrue(num_steps == self._check_steps)

        print("[PERFORMANCE]")
        print("---------------------------------------------------")
        print(
            "Desired Physics Rate:",
            (
                f"{self.eval_sim.physics_dt * 1000:.3f} ms",
                f"({int(1.0/self.eval_sim.physics_dt)}) Hz",
            ),
        )
        print("Sim Physics Rate:", self.eval_sim.get_simulation_time_profile())
        print("---------------------------------------------------")

    def test_reload(self):
        """Test the reload functionality of EvalSimStandalone.

        This test case publishes random joint states to the EvalSim environment,
        reloads the environment, and then publishes more random joint states to
        ensure the environment is reloaded correctly.
        """
        # Create the reload service client
        self.reload_service_client = self.node.create_client(Empty, "/reload")
        self.reload_service_client.wait_for_service(timeout_sec=1.0)

        for _ in range(100):
            self.mock_controller_publisher.publish(self.msg)
            self.eval_sim.step()

        # Construct and send the request to reload the environment
        req = Empty.Request()
        future = self.reload_service_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=1.0)
        # TODO this is not triggering the service due to the need for both nodes to spin
        # in parallel. To check, see that self.eval_sim.env.current_time doesn't reset
        # after the service call. Also the future returns false for future.result

        for _ in range(100):
            self.mock_controller_publisher.publish(self.msg)
            self.eval_sim.step()


if __name__ == "__main__":
    run_tests()
