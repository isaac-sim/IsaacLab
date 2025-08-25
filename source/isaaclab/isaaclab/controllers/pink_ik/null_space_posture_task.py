# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

import pinocchio as pin
from pink.configuration import Configuration
from pink.tasks import Task


class NullSpacePostureTask(Task):
    r"""Pink-based task that adds a posture objective that is in the null space projection of other tasks.

    This task implements posture control in the null space of higher priority tasks
    (typically end-effector pose tasks) within the Pink inverse kinematics framework.

    **Mathematical Formulation:**

    For details on Pink Inverse Kinematics optimization formulation visit: https://github.com/stephane-caron/pink

    **Null Space Posture Task Implementation:**

    This task consists of two components:

    1. **Error Function**: The posture error is computed as:

    .. math::

        \mathbf{e}(\mathbf{q}) = \mathbf{M} \cdot (\mathbf{q}^* - \mathbf{q})

    where:
        - :math:`\mathbf{q}^*` is the target joint configuration
        - :math:`\mathbf{q}` is the current joint configuration
        - :math:`\mathbf{M}` is a joint selection mask matrix

    2. **Jacobian Matrix**: The task Jacobian is the null space projector:

    .. math::

        \mathbf{J}_{\text{posture}}(\mathbf{q}) = \mathbf{N}(\mathbf{q}) = \mathbf{I} - \mathbf{J}_{\text{primary}}^+ \mathbf{J}_{\text{primary}}

    where:
        - :math:`\mathbf{J}_{\text{primary}}` is the combined Jacobian of all higher priority tasks
        - :math:`\mathbf{J}_{\text{primary}}^+` is the pseudoinverse of the primary task Jacobian
        - :math:`\mathbf{N}(\mathbf{q})` is the null space projector matrix

    For example, if there are two frame tasks (e.g., controlling the pose of two end-effectors), the combined Jacobian
    :math:`\mathbf{J}_{\text{primary}}` is constructed by stacking the individual Jacobians for each frame vertically:

    .. math::

        \mathbf{J}_{\text{primary}} =
        \begin{bmatrix}
            \mathbf{J}_1(\mathbf{q}) \\
            \mathbf{J}_2(\mathbf{q})
        \end{bmatrix}

    where :math:`\mathbf{J}_1(\mathbf{q})` and :math:`\mathbf{J}_2(\mathbf{q})` are the Jacobians for the first and second frame tasks, respectively.

    The null space projector ensures that joint velocities in the null space produce zero velocity
    for the primary tasks: :math:`\mathbf{J}_{\text{primary}} \cdot \dot{\mathbf{q}}_{\text{null}} = \mathbf{0}`.

    **Task Integration:**

    When integrated into the Pink framework, this task contributes to the optimization as:

    .. math::

        \left\| \mathbf{N}(\mathbf{q}) \mathbf{v} + \mathbf{M} \cdot (\mathbf{q}^* - \mathbf{q}) \right\|_{W_{\text{posture}}}^2

    This formulation allows the robot to maintain a desired posture while respecting the constraints
    imposed by higher priority tasks (e.g., end-effector positioning).

    """

    def __init__(
        self,
        cost: float,
        lm_damping: float = 0.0,
        gain: float = 1.0,
        controlled_frames: list[str] | None = None,
        controlled_joints: list[str] | None = None,
    ) -> None:
        r"""Initialize the null space posture task.

        This task maintains a desired joint posture in the null space of higher-priority
        frame tasks. Joint selection allows excluding specific joints (e.g., wrist joints
        in humanoid manipulation) to prevent large rotational ranges from overwhelming
        errors in critical joints like shoulders and waist.

        Args:
            cost: Task weighting factor in the optimization objective.
                Units: :math:`[\text{cost}] / [\text{rad}]`.
            lm_damping: Levenberg-Marquardt regularization scale (unitless). Defaults to 0.0.
            gain: Task gain :math:`\alpha \in [0, 1]` for low-pass filtering.
                Defaults to 1.0 (no filtering).
            controlled_frames: Frame names whose Jacobians define the primary tasks for
                null space projection. If None or empty, no projection is applied.
            controlled_joints: Joint names to control in the posture task. If None or
                empty, all actuated joints are controlled.
        """
        super().__init__(cost=cost, gain=gain, lm_damping=lm_damping)
        self.target_q: np.ndarray | None = None
        self.controlled_frames: list[str] = controlled_frames or []
        self.controlled_joints: list[str] = controlled_joints or []
        self._joint_mask: np.ndarray | None = None
        self._frame_names: list[str] | None = None

    def __repr__(self) -> str:
        """Human-readable representation of the task."""
        return (
            f"NullSpacePostureTask(cost={self.cost}, gain={self.gain}, lm_damping={self.lm_damping},"
            f" controlled_frames={self.controlled_frames}, controlled_joints={self.controlled_joints})"
        )

    def _build_joint_mapping(self, configuration: Configuration) -> None:
        """Build joint mask and cache frequently used values.

        Creates a binary mask that selects which joints should be controlled
        in the posture task.

        Args:
            configuration: Robot configuration containing the model and joint information.
        """
        # Create joint mask for full configuration size
        self._joint_mask = np.zeros(configuration.model.nq)

        # Create dictionary for joint names to indices (exclude root joint)
        joint_names = configuration.model.names.tolist()[1:]

        # Build joint mask efficiently
        for i, joint_name in enumerate(joint_names):
            if joint_name in self.controlled_joints:
                self._joint_mask[i] = 1.0

        # Cache frame names for performance
        self._frame_names = list(self.controlled_frames)

    def set_target(self, target_q: np.ndarray) -> None:
        """Set target posture configuration.

        Args:
            target_q: Target vector in the configuration space. If the model
                has a floating base, then this vector should include
                floating-base coordinates (although they have no effect on the
                posture task since only actuated joints are controlled).
        """
        self.target_q = target_q.copy()

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        """Set target posture from a robot configuration.

        Args:
            configuration: Robot configuration whose joint angles will be used
                as the target posture.
        """
        self.set_target(configuration.q)

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute posture task error.

        The error computation follows:

        .. math::

            \mathbf{e}(\mathbf{q}) = \mathbf{M} \cdot (\mathbf{q}^* - \mathbf{q})

        where :math:`\mathbf{M}` is the joint selection mask and :math:`\mathbf{q}^* - \mathbf{q}`
        is computed using Pinocchio's difference function to handle joint angle wrapping.

        Args:
            configuration: Robot configuration :math:`\mathbf{q}`.

        Returns:
            Posture task error :math:`\mathbf{e}(\mathbf{q})` with the same dimension
            as the configuration vector, but with zeros for non-controlled joints.

        Raises:
            ValueError: If no posture target has been set.
        """
        if self.target_q is None:
            raise ValueError("No posture target has been set. Call set_target() first.")

        # Initialize joint mapping if needed
        if self._joint_mask is None:
            self._build_joint_mapping(configuration)

        # Compute configuration difference using Pinocchio's difference function
        # This handles joint angle wrapping correctly
        err = pin.difference(
            configuration.model,
            self.target_q,
            configuration.q,
        )

        # Apply pre-computed joint mask to select only controlled joints
        return self._joint_mask * err

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the null space projector Jacobian.

        The null space projector is defined as:

        .. math::

            \mathbf{N}(\mathbf{q}) = \mathbf{I} - \mathbf{J}_{\text{primary}}^+ \mathbf{J}_{\text{primary}}

        where:
            - :math:`\mathbf{J}_{\text{primary}}` is the combined Jacobian of all controlled frames
            - :math:`\mathbf{J}_{\text{primary}}^+` is the pseudoinverse of the primary task Jacobian
            - :math:`\mathbf{I}` is the identity matrix

        The null space projector ensures that joint velocities in the null space produce
        zero velocity for the primary tasks: :math:`\mathbf{J}_{\text{primary}} \cdot \dot{\mathbf{q}}_{\text{null}} = \mathbf{0}`.

        If no controlled frames are specified, returns the identity matrix.

        Args:
            configuration: Robot configuration :math:`\mathbf{q}`.

        Returns:
            Null space projector matrix :math:`\mathbf{N}(\mathbf{q})` with dimensions
            :math:`n_q \times n_q` where :math:`n_q` is the number of configuration variables.
        """
        # Initialize joint mapping if needed
        if self._frame_names is None:
            self._build_joint_mapping(configuration)

        # If no frame tasks are defined, return identity matrix (no null space projection)
        if not self._frame_names:
            return np.eye(configuration.model.nq)

        # Get Jacobians for all frame tasks and combine them
        J_frame_tasks = [configuration.get_frame_jacobian(frame_name) for frame_name in self._frame_names]
        J_combined = np.concatenate(J_frame_tasks, axis=0)

        # Compute null space projector: N = I - J^+ * J
        N_combined = np.eye(J_combined.shape[1]) - np.linalg.pinv(J_combined) @ J_combined

        return N_combined
