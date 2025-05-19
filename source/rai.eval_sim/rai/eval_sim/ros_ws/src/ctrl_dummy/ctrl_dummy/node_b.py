# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Filename: node_b.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header


class NodeB(Node):
    def __init__(self):
        super().__init__("node_b")
        self.subscription = self.create_subscription(Header, "/request", self.listener_callback, 10)
        self.publisher_ = self.create_publisher(Header, "/response", 10)
        self.get_logger().info("Node B has started, waiting for requests.")

    def listener_callback(self, msg):
        self.get_logger().info("Request received, sending response.")
        response_msg = Header()
        response_msg.stamp = self.get_clock().now().to_msg()
        self.publisher_.publish(response_msg)


def main(args=None):
    rclpy.init(args=args)
    node_b = NodeB()
    rclpy.spin(node_b)
    node_b.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
