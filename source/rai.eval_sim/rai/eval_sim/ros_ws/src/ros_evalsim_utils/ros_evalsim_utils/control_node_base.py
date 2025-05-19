# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod

from rclpy.node import Node
from rclpy.qos import (
    Duration,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

QOS_PROFILE = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    lifespan=Duration(seconds=1000),
)


class Controller(Node, ABC):
    def __init__(self, name: str = "controller"):
        # save inputs
        self.name = name

        # initialize node
        super().__init__(self.name)

        # create subscriber
        self.subscriber_terms = []
        self.publisher = None

    @abstractmethod
    def step(self):
        raise NotImplementedError
