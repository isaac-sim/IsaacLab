# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
from math import pi

import rclpy
from sensor_msgs.msg import JointState

from .controller import Controller


class AnymalController(Controller):
    def __init__(self, name: str = "anymal_controller", frequency: int = 1000):
        # save inputs
        self.frequency = frequency

        # initialize controller
        super().__init__(name)

        # initialize controller variables
        zeros = [0.0 for i in range(12)]
        self.command = JointState(position=zeros, velocity=zeros, effort=zeros)
        self.command.name = [
            "LF_HAA",
            "LH_HAA",
            "RF_HAA",
            "RH_HAA",
            "LF_HFE",
            "LH_HFE",
            "RF_HFE",
            "RH_HFE",
            "LF_KFE",
            "LH_KFE",
            "RF_KFE",
            "RH_KFE",
        ]
        self.step_count = 0

        # print controller information after initialization
        print(str(self))

    def control_step(self, msg: JointState):
        # update command control
        zeros = [0.0 for i in range(12)]
        zeros[0] = ((self.step_count % self.frequency) / self.frequency - 0.5) * pi
        self.command.position = zeros

        # publish command
        self.publisher.publish(self.command)

        # update step counter
        self.step_count += 1

    def __str__(self):
        message = (
            f"Node: {self.name}"
            + f"\n  INPUT: topic name - {self.subscriber.topic_name}, msg type - {self.subscriber.msg_type}"
            + f"\n  OUTPUT: topic name - {self.publisher.topic_name}, msg type - {self.publisher.msg_type}"
            + f"\n  Zeros input sample: {self.sample_command()}"
        )
        return message

    def sample_command(self):
        zeros = [0.0 for i in range(12)]
        msg_command = "{" + f"'position': {zeros}, 'velocity': {zeros}, 'effort': {zeros}" + "}"
        ros_command = f'ros2 topic pub -r 1000 /joint_state sensor_msgs/msg/JointState "{msg_command}"'
        return ros_command


def main(args=None):
    # set up controller
    rclpy.init(args=args)
    controller = AnymalController()

    # run controller
    with contextlib.suppress(KeyboardInterrupt):
        while rclpy.ok():
            rclpy.spin_once(controller, timeout_sec=1.0)  # Adjust timeout as needed

    # shut down
    controller.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
