# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Filename: node_a.py
import contextlib
from time import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header


class NodeA(Node):
    def __init__(self):
        super().__init__("node_a")
        self.publisher_ = self.create_publisher(Header, "/request", 10)
        self.subscription = self.create_subscription(Header, "/response", self.listener_callback, 10)
        self.dt_request = None
        self.get_logger().info("Node A has started, waiting to send requests.")

    def send_request(self):
        msg = Header()
        msg.stamp = self.get_clock().now().to_msg()
        self.dt_request = time()
        self.publisher_.publish(msg)
        self.get_logger().info("Request sent.")

    def listener_callback(self, msg):
        dt_response = time() - self.dt_request
        self.get_logger().info(f"Response received. Round trip time: {dt_response} seconds.")


def main(args=None):
    rclpy.init(args=args)
    node_a = NodeA()

    with contextlib.suppress(KeyboardInterrupt):
        while rclpy.ok():
            node_a.send_request()
            rclpy.spin_once(node_a, timeout_sec=1.0)  # Adjust timeout as needed

            # wait for a while
            import time

            time.sleep(2)

    node_a.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
