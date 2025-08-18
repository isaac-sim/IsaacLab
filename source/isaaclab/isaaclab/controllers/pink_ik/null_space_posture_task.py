# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

import pinocchio as pin
from pink.configuration import Configuration
from pink.tasks import Task
from pink.utils import get_root_joint_dim


class NullSpacePostureTask(Task):
    r"""Pink-based task that enforces a postural constraint on the robot using null space projection.

    This task implements a null space posture control strategy within the Pink inverse kinematics framework.
    It enforces a desired joint configuration while operating in the null space of higher priority tasks
    (typically end-effector pose tasks). The mathematical formulation follows the Pink optimization framework.

    **Mathematical Formulation:**

    The Pink inverse kinematics framework solves the following constrained optimization problem:

    .. math::

        \begin{align*}
            &\min_{\mathbf{v} \in \mathcal{C}} \sum_{\text{task } e} \left\| J_e(\mathbf{q}) \mathbf{v} + \alpha_e(\mathbf{q}) \right\|_{W_e}^2 \\
            &\text{subject to} \quad \mathbf{v}_{\min}(\mathbf{q}) \leq \mathbf{v} \leq \mathbf{v}_{\max}(\mathbf{q})
        \end{align*}

    where:
        - :math:`\mathbf{v}` is the joint velocity vector (optimization variable)
        - :math:`J_e(\mathbf{q})` is the task Jacobian for task :math:`e`
        - :math:`\alpha_e(\mathbf{q})` is the task error/residual for task :math:`e`
        - :math:`W_e` is the task weighting matrix
        - :math:`\mathcal{C}` is the set of feasible velocities (joint velocity limits)
        - :math:`\mathbf{v}_{\min}(\mathbf{q})`, :math:`\mathbf{v}_{\max}(\mathbf{q})` are the lower and upper joint velocity bounds


    This NullSpacePostureTask implements one of the residual functions

    .. math::

        e(q) = N^+(q) \cdot (q^* - q)

    where:
    - :math:`N^+(q)` is the pseudoinverse of the null space projector :math:`N(q) = I - J^+J`
      :math:`J` is the Jacobian of the primary task
    - :math:`q^*` is the target joint configuration
    - :math:`q` is the current joint configuration

    """

    def __init__(
        self,
        cost: float,
        lm_damping: float = 0.0,
        gain: float = 1.0,
        frame_task_controlled_joints: dict[str, list[str]] | None = None,
    ) -> None:
        r"""Create task.

        Args:
            cost: value used to cast joint angle differences to a homogeneous
                cost, in :math:`[\mathrm{cost}] / [\mathrm{rad}]`.
            lm_damping: Unitless scale of the Levenberg-Marquardt (only when
                the error is large) regularization term, which helps when
                targets are infeasible. Increase this value if the task is too
                jerky under infeasible targets, but beware that too large a
                damping can slow down the task.
            gain: Task gain :math:`\alpha \in [0, 1]` for additional low-pass
                filtering. Defaults to 1.0 (no filtering).
            frame_task_controlled_joints: Dictionary of frame link names to controlled joint names.
        """
        super().__init__(cost=cost, gain=gain, lm_damping=lm_damping)
        self.target_q: np.ndarray | None = None
        self.frame_task_controlled_joints: dict[str, list[str]] = frame_task_controlled_joints or {}
        self._joint_name_to_index: dict[str, int] | None = None
        self._selected_joint_indices: list[np.ndarray] | None = None
        self._jacobian_dim: int = 0

    def __repr__(self) -> str:
        """Human-readable representation of the task."""
        return (
            f"NullSpacePostureTask(cost={self.cost}, gain={self.gain}, lm_damping={self.lm_damping},"
            f" frame_task_controlled_joints={self.frame_task_controlled_joints})"
        )

    def _build_joint_mapping(self, configuration: Configuration) -> None:
        """Build efficient joint name to index mapping.

        Args:
            configuration: Robot configuration.
        """
        if self._joint_name_to_index is not None:
            return

        # Create O(1) lookup dictionary for joint names
        joint_names = configuration.model.names.tolist()[1:]  # Skip root joint
        self._joint_name_to_index = {name: idx for idx, name in enumerate(joint_names)}

        # Build selected joint indices efficiently
        self._selected_joint_indices = []
        for controlled_joints in self.frame_task_controlled_joints.values():
            indices = [self._joint_name_to_index[joint] for joint in controlled_joints]
            self._selected_joint_indices.append(np.array(sorted(indices), dtype=int))

    def set_target(self, target_q: np.ndarray) -> None:
        """Set target posture.

        Args:
            target_q: Target vector in the configuration space. If the model
                has a floating base, then this vector should include
                floating-base coordinates (although they have no effect on the
                posture task).
        """
        self.target_q = target_q.copy()

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        """Set target posture from a robot configuration.

        Args:
            configuration: Robot configuration.
        """
        self.set_target(configuration.q)

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute posture task error.

        This method computes the posture error for the null space posture task.
        The error is the difference between the target and current configuration,
        excluding the floating base coordinates (if present).

        Note:
            The actual null space projection is applied in the Jacobian (see `compute_jacobian`).
            This function only returns the configuration difference in the actuated joint space.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Posture task error :math:`e(q) = q^* - q`, where only actuated joints are considered.

        Raises:
            ValueError: If no posture target has been set.
        """
        if self.target_q is None:
            raise ValueError("No posture target has been set. Call set_target() first.")

        _, root_nv = get_root_joint_dim(configuration.model)

        # Compute configuration difference
        diff = pin.difference(
            configuration.model,
            self.target_q,
            configuration.q,
        )[root_nv:]

        return diff

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the posture task Jacobian (null space projector).

        This method computes the sum of the left and right null space projectors,
        each defined as :math:`N(q) = I - J^+J`, where :math:`J^+` is the pseudoinverse
        of the corresponding end-effector Jacobian. The null space projectors are
        masked to only affect the selected controlled joints for each side.

        The final Jacobian returned is:
            :math:`J(q) = N_{\text{left}}(q) + N_{\text{right}}(q)`

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Posture task Jacobian :math:`J(q)`, which is the sum of the left and right
            null space projectors, each masked to their respective controlled joints.
        """
        # Initialize joint mapping if needed
        if self._joint_name_to_index is None:
            self._build_joint_mapping(configuration)

        # Early return if no frame tasks defined
        if not self.frame_task_controlled_joints:
            print("No frame tasks defined")
            # Return identity matrix if no frame tasks defined
            n_joints = len(configuration.model.names) - 1  # Exclude root joint
            return np.eye(n_joints)

        # Get frame names
        frame_names = list(self.frame_task_controlled_joints.keys())

        # Get Jacobians for all frame tasks
        J_frame_tasks = [configuration.get_frame_jacobian(frame_name) for frame_name in frame_names]

        # Cache Jacobian dimension
        if self._jacobian_dim == 0:
            self._jacobian_dim = J_frame_tasks[0].shape[1]

        # Initialize null space task matrix
        null_space_task = np.zeros((self._jacobian_dim, self._jacobian_dim))

        # Compute null space projectors efficiently
        for i, (J_frame_task, selected_indices) in enumerate(zip(J_frame_tasks, self._selected_joint_indices or [])):
            # Create mask for selected joints
            mask = np.zeros(J_frame_task.shape[1], dtype=bool)
            mask[selected_indices] = True

            # Compute pseudoinverse and null space projector
            J_pinv = np.linalg.pinv(J_frame_task)
            null_space_full = np.eye(J_frame_task.shape[1]) - J_pinv @ J_frame_task

            # Only update rows corresponding to selected joints
            null_space_task[mask, :] += null_space_full[mask, :]

        return null_space_task
