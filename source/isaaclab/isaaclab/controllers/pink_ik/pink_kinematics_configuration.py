# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import pinocchio as pin
from pink.configuration import Configuration
from pink.exceptions import FrameNotFound
from pinocchio.robot_wrapper import RobotWrapper


class PinkKinematicsConfiguration(Configuration):
    """
    A configuration class that maintains both a "controlled" (reduced) model and a "full" model.

    This class extends the standard Pink Configuration to allow for selective joint control:

    - The "controlled" model/data/q represent the subset of joints being actively controlled
      (e.g., a kinematic chain or arm).
    - The "full" model/data/q represent the complete robot, including all joints.

    This is useful for scenarios where only a subset of joints are being optimized or controlled, but
    full-model kinematics (e.g., for collision checking, full-body Jacobians, or visualization) are still required.

    The class ensures that both models are kept up to date, and provides methods to update both the controlled and full
    configurations as needed.
    """

    def __init__(
        self,
        controlled_joint_names: list[str],
        urdf_path: str,
        mesh_path: str | None = None,
        copy_data: bool = True,
        forward_kinematics: bool = True,
    ):
        """
        Initialize PinkKinematicsConfiguration.


        This constructor initializes the PinkKinematicsConfiguration, which maintains both a "controlled"
        (reduced) model and a "full" model. The controlled model/data/q represent the subset of joints
        being actively controlled, while the full model/data/q represent the complete robot. This is useful
        for scenarios where only a subset of joints are being optimized or controlled, but full-model
        kinematics are still required.

        Args:
            urdf_path: Path to the robot URDF file.
            mesh_path: Path to the mesh files for the robot.
            controlled_joint_names: List of joint names to be actively controlled.
            copy_data: If True, work on an internal copy of the input data. Defaults to True.
            forward_kinematics: If True, compute forward kinematics from the configuration vector. Defaults to True.
        """
        self._controlled_joint_names = controlled_joint_names

        # Build robot model with all joints
        if mesh_path:
            self.robot_wrapper = RobotWrapper.BuildFromURDF(urdf_path, mesh_path)
        else:
            self.robot_wrapper = RobotWrapper.BuildFromURDF(urdf_path)
        self.full_model = self.robot_wrapper.model
        self.full_data = self.robot_wrapper.data
        self.full_q = self.robot_wrapper.q0

        # import pdb; pdb.set_trace()
        self._all_joint_names = self.full_model.names.tolist()[1:]
        # controlled_joint_indices: indices in all_joint_names for joints that are in controlled_joint_names,
        # preserving all_joint_names order
        self._controlled_joint_indices = [
            idx for idx, joint_name in enumerate(self._all_joint_names) if joint_name in self._controlled_joint_names
        ]

        # Build the reduced model with only the controlled joints
        joints_to_lock = []
        for joint_name in self._all_joint_names:
            if joint_name not in self._controlled_joint_names:
                joints_to_lock.append(self.full_model.getJointId(joint_name))

        if len(joints_to_lock) == 0:
            # No joints to lock, controlled model is the same as full model
            self.controlled_model = self.full_model
            self.controlled_data = self.full_data
            self.controlled_q = self.full_q
        else:
            self.controlled_model = pin.buildReducedModel(self.full_model, joints_to_lock, self.full_q)
            self.controlled_data = self.controlled_model.createData()
            self.controlled_q = self.full_q[self._controlled_joint_indices]

        # Pink will should only have the controlled model
        super().__init__(self.controlled_model, self.controlled_data, self.controlled_q, copy_data, forward_kinematics)

    def update(self, q: np.ndarray | None = None) -> None:
        """Update configuration to a new vector.

        Calling this function runs forward kinematics and computes
        collision-pair distances, if applicable.

        Args:
            q: New configuration vector.
        """
        if q is not None and len(q) != len(self._all_joint_names):
            raise ValueError("q must have the same length as the number of joints in the model")
        if q is not None:
            super().update(q[self._controlled_joint_indices])

            q_readonly = q.copy()
            q_readonly.setflags(write=False)
            self.full_q = q_readonly
            pin.computeJointJacobians(self.full_model, self.full_data, q)
            pin.updateFramePlacements(self.full_model, self.full_data)
        else:
            super().update()
            pin.computeJointJacobians(self.full_model, self.full_data, self.full_q)
            pin.updateFramePlacements(self.full_model, self.full_data)

    def get_frame_jacobian(self, frame: str) -> np.ndarray:
        r"""Compute the Jacobian matrix of a frame velocity.

        Denoting our frame by :math:`B` and the world frame by :math:`W`, the
        Jacobian matrix :math:`{}_B J_{WB}` is related to the body velocity
        :math:`{}_B v_{WB}` by:

        .. math::

            {}_B v_{WB} = {}_B J_{WB} \dot{q}

        Args:
            frame: Name of the frame, typically a link name from the URDF.

        Returns:
            Jacobian :math:`{}_B J_{WB}` of the frame.

        When the robot model includes a floating base
        (pin.JointModelFreeFlyer), the configuration vector :math:`q` consists
        of:

        - ``q[0:3]``: position in [m] of the floating base in the inertial
          frame, formatted as :math:`[p_x, p_y, p_z]`.
        - ``q[3:7]``: unit quaternion for the orientation of the floating base
          in the inertial frame, formatted as :math:`[q_x, q_y, q_z, q_w]`.
        - ``q[7:]``: joint angles in [rad].
        """
        if not self.full_model.existFrame(frame):
            raise FrameNotFound(frame, self.full_model.frames)
        frame_id = self.full_model.getFrameId(frame)
        J: np.ndarray = pin.getFrameJacobian(self.full_model, self.full_data, frame_id, pin.ReferenceFrame.LOCAL)
        return J[:, self._controlled_joint_indices]

    def get_transform_frame_to_world(self, frame: str) -> pin.SE3:
        """Get the pose of a frame in the current configuration.

        We override this method from the super class to solve the issue that in the default
        Pink implementation, the frame placements do not take into account the non-controlled joints
        being not at initial pose (which is a bad assumption when they are controlled by other
        controllers like a lower body controller).

        Args:
            frame: Name of a frame, typically a link name from the URDF.

        Returns:
            Current transform from the given frame to the world frame.

        Raises:
            FrameNotFound: if the frame name is not found in the robot model.
        """
        frame_id = self.full_model.getFrameId(frame)
        try:
            return self.full_data.oMf[frame_id].copy()
        except IndexError as index_error:
            raise FrameNotFound(frame, self.full_model.frames) from index_error

    def check_limits(self, tol: float = 1e-6, safety_break: bool = True) -> None:
        """Check if limits are violated only if safety_break is enabled"""
        if safety_break:
            super().check_limits(tol, safety_break)

    @property
    def controlled_joint_names_pinocchio_order(self) -> list[str]:
        """Get the names of the controlled joints in the order of the pinocchio model."""
        return [self._all_joint_names[i] for i in self._controlled_joint_indices]

    @property
    def all_joint_names_pinocchio_order(self) -> list[str]:
        """Get the names of all joints in the order of the pinocchio model."""
        return self._all_joint_names
