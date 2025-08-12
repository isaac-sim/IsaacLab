# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from collections.abc import Sequence

import pinocchio as pin
from pink.exceptions import TargetNotSet
from pink.tasks.frame_task import FrameTask

from isaaclab.controllers.pink_kinematics_configuration import PinkKinematicsConfiguration


class LocalFrameTask(FrameTask):
    """
    A task that computes error in a local (custom) frame.
    Inherits from FrameTask but overrides compute_error.
    """

    def __init__(
        self,
        frame: str,
        base_link_frame_name: str,
        position_cost: float | Sequence[float],
        orientation_cost: float | Sequence[float],
        lm_damping: float = 0.0,
        gain: float = 1.0,
    ):
        """
        Initialize the LocalFrameTask with configuration.

        Args:
            base_link_frame_name: Name of the base link frame.
        """
        super().__init__(frame, position_cost, orientation_cost, lm_damping, gain)
        self.base_link_frame_name = base_link_frame_name
        self.transform_target_to_base = None

    def set_target(self, transform_target_to_base: pin.SE3) -> None:
        """Set task target pose in the world frame.

        Args:
            transform_target_to_world: Transform from the task target frame to
                the world frame.
        """
        self.transform_target_to_base = transform_target_to_base.copy()

    def set_target_from_configuration(self, configuration: PinkKinematicsConfiguration) -> None:
        """Set task target pose from a robot configuration.

        Args:
            configuration: Robot configuration.
        """
        if not isinstance(configuration, PinkKinematicsConfiguration):
            raise ValueError("configuration must be a PinkKinematicsConfiguration")
        self.set_target(configuration.get_transform(self.frame, self.base_link_frame_name))

    def compute_error(self, configuration: PinkKinematicsConfiguration) -> np.ndarray:
        """
        Compute the error between current and target pose in a local frame.
        """
        if not isinstance(configuration, PinkKinematicsConfiguration):
            raise ValueError("configuration must be a PinkKinematicsConfiguration")
        if self.transform_target_to_base is None:
            raise ValueError(f"no target set for frame '{self.frame}'")

        transform_frame_to_base = configuration.get_transform(self.frame, self.base_link_frame_name)
        transform_target_to_frame = transform_frame_to_base.actInv(self.transform_target_to_base)

        error_in_frame: np.ndarray = pin.log(transform_target_to_frame).vector
        return error_in_frame

    def compute_jacobian(self, configuration: PinkKinematicsConfiguration) -> np.ndarray:
        r"""Compute the frame task Jacobian.

        The task Jacobian :math:`J(q) \in \mathbb{R}^{6 \times n_v}` is the
        derivative of the task error :math:`e(q) \in \mathbb{R}^6` with respect
        to the configuration :math:`q`. The formula for the frame task is:

        .. math::

            J(q) = -\text{Jlog}_6(T_{tb}) {}_b J_{0b}(q)

        The derivation of the formula for this Jacobian is detailed in
        [Caron2023]_. See also
        :func:`pink.tasks.task.Task.compute_jacobian` for more context on task
        Jacobians.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Jacobian matrix :math:`J`, expressed locally in the frame.
        """
        if self.transform_target_to_base is None:
            raise TargetNotSet(f"no target set for frame '{self.frame}'")
        transform_frame_to_base = configuration.get_transform(self.frame, self.base_link_frame_name)
        transform_frame_to_target = self.transform_target_to_base.actInv(transform_frame_to_base)
        jacobian_in_frame = configuration.get_frame_jacobian(self.frame)
        J = -pin.Jlog6(transform_frame_to_target) @ jacobian_in_frame
        return J
