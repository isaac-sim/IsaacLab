# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import rclpy
from sensor_msgs.msg import JointState

from .controller import Controller


class UMVController(Controller):
    def __init__(self, name: str = "umv_controller", frequency: int = 1000):
        # save inputs
        self.frequency = frequency

        # initialize controller
        super().__init__(name)

        # initialize controller variables
        self.command = JointState(position=[0.0, 0.0, 0.0], velocity=[0.0, 1.0, 0.0], effort=[0.0, 0.0, 0.0])
        self.step_count = 0

        # print controller information after initialization
        print(str(self))

    def control_step(self, msg: JointState):
        # update command control
        self.command.position = [0.0, self.step_count / 50.0, 0.0]

        # publish command
        self.publisher.publish(self.command)

        # update step counter
        self.step_count += 1


def main(args=None):
    # set up controller
    rclpy.init(args=args)
    controller = UMVController()

    # run controller
    try:  # noqa
        while rclpy.ok():
            rclpy.spin_once(controller, timeout_sec=1.0)  # Adjust timeout as needed
    except KeyboardInterrupt:
        pass

    # shut down
    controller.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
