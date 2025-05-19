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

import random
import time
import unittest

import rclpy
from rai.eval_sim.eval_sim import EvalSimCfg
from rai.eval_sim.eval_sim.eval_sim_standalone import EvalSimStandalone
from rai.eval_sim.ros_manager import QOS_PROFILE
from rai.eval_sim.tasks.testing import TESTING_CFGS
from rai.eval_sim.utils import zero_actions
from rclpy.node import Node
from rclpy.publisher import Publisher
from sensor_msgs.msg import JointState


def fullname(o):
    klass = o.__class__
    module = klass.__module__
    if module == "builtins":
        return klass.__qualname__  # avoid outputs like 'builtins.str'
    return module + "." + klass.__qualname__


def _sub_callback(msg: JointState):
    pass


def run_eval_sim(
    eval_sim: EvalSimStandalone,
    ctrl_node: Node | None = None,
    ctrl_pub: Publisher | None = None,
    gains_pub: Publisher | None = None,
) -> (float, float, float, tuple):
    """Function that runs an eval sim instance and times the execution.

    Args:
        eval_sim: The EvalSim instance.
        ctrl_node: The mock control node that publishes and subscribes.
        ctrl_pub: The mock publisher for joint_state.
        gains_pub: The mock publisher for joint gains using the joint_state message.

    Returns:
        A tuple of the sim/realtime percentage, control dt (sec), physics dt (sec).
    """
    # create evalsim
    joint_names = eval_sim.env.scene["robot"].joint_names
    num_joints = len(joint_names)
    msg = JointState(
        name=joint_names,
        position=[random.uniform(-1, 1) for _ in range(num_joints)],
        velocity=[random.uniform(-1, 1) for _ in range(num_joints)],
        effort=[random.uniform(-1, 1) for _ in range(num_joints)],
    )
    pd_gains_msg = JointState(
        name=joint_names,
        position=[200 for _ in range(num_joints)],
        velocity=[20 for _ in range(num_joints)],
        effort=[0 for _ in range(num_joints)],
    )
    start_ns = time.time_ns()
    prev_ns = start_ns
    num_steps = 1000
    warmup_steps = 300
    timeout = 0.1
    for i in range(num_steps):
        if i == warmup_steps:
            start_ns = time.time_ns()
            prev_ns = start_ns
        # add pdgain msg to publish queue
        gains_pub.publish(pd_gains_msg)
        # add ctrl msg to publish queue
        ctrl_pub.publish(msg)
        eval_sim.step()
        rclpy.spin_once(node=ctrl_node, timeout_sec=timeout)
        curr_ns = time.time_ns()
        delta_ns = curr_ns - prev_ns
        if delta_ns > timeout * 1e9:
            print("timeout")
        prev_ns = curr_ns
    performance = eval_sim.env.step_dt / float((curr_ns - start_ns) / 1e9 / (num_steps - warmup_steps)) * 100.0
    profile = eval_sim.get_simulation_time_profile()
    return performance, eval_sim.env.step_dt, eval_sim.env.physics_dt, profile


class TestProfileEvalSimStandalone(unittest.TestCase):

    def test_run_profile(self):
        # create evalsim instance
        eval_sim_cfg = EvalSimCfg(
            sync_to_real_time=False,
            env_cfg=str(fullname(TESTING_CFGS["BaseLine"][0]())),
            ros_manager_cfg=str(fullname(TESTING_CFGS["BaseLine"][1]())),
            auto_save=False,
        )
        eval_sim = EvalSimStandalone(eval_sim_cfg)

        # create mock ros control node
        mock_node = Node("mock_controller")
        mock_controller_publisher = mock_node.create_publisher(
            msg_type=JointState, topic="/joint_command", qos_profile=QOS_PROFILE
        )
        mock_pd_gains_publisher = mock_node.create_publisher(
            msg_type=JointState, topic="/joint_command_pd_gains", qos_profile=QOS_PROFILE
        )
        _ = mock_node.create_subscription(
            msg_type=JointState,
            topic="/joint_state",
            qos_profile=QOS_PROFILE,
            callback=_sub_callback,
        )

        performance = dict()
        for prof_name, cfgs in TESTING_CFGS.items():
            cfgs = TESTING_CFGS[prof_name]
            eval_sim.clear()
            eval_sim.set_env_cfg(str(fullname(cfgs[0]())))
            eval_sim.set_ros_manager_cfg(str(fullname(cfgs[1]())))
            eval_sim.env_cfg.sim.dt = 0.002
            eval_sim.env_cfg.decimation = 1
            eval_sim.env_cfg.sim.render_interval = 30
            eval_sim.load()
            eval_sim.step()
            eval_sim.enable_ros()
            # Take a zero action to initialize the environment and get the first observation
            _ = eval_sim.env.step(zero_actions(eval_sim.env))

            performance[prof_name] = run_eval_sim(
                eval_sim, mock_node, mock_controller_publisher, mock_pd_gains_publisher
            )
            eval_sim.stop_ros_node()

        print("=======================Performance============================")
        print(f"{'Task':<20}{'sim/real %':^12}{'control dt (s)':^15s}{'physics dt (s)':^12s}")
        print("---------------------------------------------------------")
        for task, perf in performance.items():
            print(f"{task:<20}{perf[0]:^12.1f}{perf[1]:^15.3f}{perf[2]:^15.3f}")
        print("==============================================================")

        mock_node.destroy_node()
        eval_sim.close()


if __name__ == "__main__":
    run_tests()
