# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence

import numpy as np
import pinocchio as pin
from pink.tasks import DampingTask as PinkDampingTask
from pink.tasks.frame_task import FrameTask as PinkFrameTask

from .pink_kinematics_configuration import PinkKinematicsConfiguration


class FrameTask(PinkFrameTask):
    """Thin wrapper around Pink's :class:`~pink.tasks.frame_task.FrameTask`.

    Adds support for the ``class_type(cfg)`` construction pattern used by
    Isaac Lab task configuration dataclasses, while remaining fully compatible
    with the original string-based constructor.
    """

    def __init__(
        self,
        cfg_or_frame,
        position_cost: float | Sequence[float] | None = None,
        orientation_cost: float | Sequence[float] | None = None,
        lm_damping: float = 0.0,
        gain: float = 1.0,
    ):
        """Initialize the FrameTask.

        Args:
            cfg_or_frame: Either a *string* naming the controlled frame, or a
                configuration dataclass whose attributes (``frame``,
                ``position_cost``, ``orientation_cost``, ``lm_damping``,
                ``gain``) supply all parameters.
            position_cost: Cost weight(s) for position error.  A single float
                applies uniform weighting; a sequence of 3 floats gives per-axis
                weights.  Required when *cfg_or_frame* is a string.
            orientation_cost: Cost weight(s) for orientation error (same
                convention as *position_cost*).  Required when *cfg_or_frame* is
                a string.
            lm_damping: Levenberg-Marquardt damping factor for numerical
                stability.  Defaults to 0.0 (no damping).
            gain: Task gain that scales the overall task contribution.
                Defaults to 1.0.
        """
        if isinstance(cfg_or_frame, str):
            frame = cfg_or_frame
        else:
            cfg = cfg_or_frame
            frame = cfg.frame
            position_cost = cfg.position_cost
            orientation_cost = cfg.orientation_cost
            lm_damping = cfg.lm_damping
            gain = cfg.gain

        if position_cost is None or orientation_cost is None:
            raise ValueError("position_cost and orientation_cost must be provided")

        super().__init__(
            frame,
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            lm_damping=lm_damping,
            gain=gain,
        )


class DampingTask(PinkDampingTask):
    """Thin wrapper around Pink's :class:`~pink.tasks.DampingTask`.

    Adds joint-velocity damping to the IK problem for numerical stability.
    Accepts either a configuration dataclass (``class_type(cfg)`` pattern) or a
    direct scalar cost value.
    """

    def __init__(self, cfg_or_cost, cost: float | None = None):
        """Initialize the DampingTask.

        Args:
            cfg_or_cost: Either a numeric cost value, or a configuration
                dataclass with a ``cost`` attribute.
            cost: Explicit cost override.  When *cfg_or_cost* is numeric and
                *cost* is also provided, *cost* takes precedence.
        """
        if isinstance(cfg_or_cost, (int, float)):
            _cost = float(cfg_or_cost if cost is None else cost)
        else:
            _cost = cfg_or_cost.cost
        super().__init__(cost=_cost)


class LocalFrameTask(FrameTask):
    """A task that computes pose error in a local (custom) frame.

    Inherits from :class:`FrameTask` but overrides error and Jacobian computation
    to express them relative to a specified base-link frame rather than the world
    frame.  This allows control strategies where the reference frame can be chosen
    independently (e.g. the robot base).
    """

    def __init__(
        self,
        frame,
        base_link_frame_name: str | None = None,
        position_cost: float | Sequence[float] | None = None,
        orientation_cost: float | Sequence[float] | None = None,
        lm_damping: float = 0.0,
        gain: float = 1.0,
    ):
        """Initialize the LocalFrameTask.

        The first positional argument may be either a *string* (frame name) or a
        configuration dataclass that carries all parameters.

        Args:
            frame: Name of the frame to control (end-effector or target frame),
                **or** a configuration object whose attributes mirror the remaining
                arguments.
            base_link_frame_name: Name of the base-link frame used as the
                reference for computing transforms and errors.  Required when
                *frame* is a string.
            position_cost: Cost weight(s) for position error.  A single float
                applies uniform weighting; a sequence of 3 floats gives per-axis
                weights.
            orientation_cost: Cost weight(s) for orientation error (same
                convention as *position_cost*).
            lm_damping: Levenberg-Marquardt damping factor for numerical
                stability.  Defaults to 0.0 (no damping).
            gain: Task gain that scales the overall task contribution.
                Defaults to 1.0.
        """
        if isinstance(frame, str):
            resolved_frame = frame
            if base_link_frame_name is None:
                raise ValueError("base_link_frame_name must be provided")
        else:
            cfg = frame
            resolved_frame = cfg.frame
            base_link_frame_name = cfg.base_link_frame_name
            position_cost = cfg.position_cost
            orientation_cost = cfg.orientation_cost
            lm_damping = cfg.lm_damping
            gain = cfg.gain

        if position_cost is None or orientation_cost is None:
            raise ValueError("position_cost and orientation_cost must be provided")

        super().__init__(resolved_frame, position_cost, orientation_cost, lm_damping, gain)
        self.base_link_frame_name = base_link_frame_name
        self.transform_target_to_base = None

    def set_target(self, transform_target_to_base: pin.SE3) -> None:
        """Set the task target pose relative to the base-link frame.

        Args:
            transform_target_to_base: Desired transform from the target frame
                to the base-link frame.
        """
        self.transform_target_to_base = transform_target_to_base.copy()

    def set_target_from_configuration(self, configuration: PinkKinematicsConfiguration) -> None:
        """Set the task target pose from the current robot configuration.

        The target is computed as the transform of :attr:`frame` relative to
        :attr:`base_link_frame_name` in the given configuration.

        Args:
            configuration: Robot configuration to read the current pose from.
        """
        if not isinstance(configuration, PinkKinematicsConfiguration):
            raise ValueError("configuration must be a PinkKinematicsConfiguration")
        self.set_target(configuration.get_transform(self.frame, self.base_link_frame_name))

    def compute_error(self, configuration: PinkKinematicsConfiguration) -> np.ndarray:
        """Compute the error between current and target pose in the local frame.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            6D error vector (3 position + 3 orientation) expressed in the
            controlled frame.
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
        r"""Compute the frame task Jacobian expressed in the local frame.

        The task Jacobian :math:`J(q) \in \mathbb{R}^{6 \times n_v}` is the
        derivative of the task error :math:`e(q) \in \mathbb{R}^6` with respect
        to the configuration :math:`q`:

        .. math::

            J(q) = -\text{Jlog}_6(T_{tb}) \; {}_b J_{0b}(q)

        See [Caron2023]_ for a full derivation and
        :func:`pink.tasks.task.Task.compute_jacobian` for more context on task
        Jacobians.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Jacobian matrix :math:`J`, expressed locally in the frame.
        """
        if self.transform_target_to_base is None:
            raise Exception(f"no target set for frame '{self.frame}'")
        transform_frame_to_base = configuration.get_transform(self.frame, self.base_link_frame_name)
        transform_frame_to_target = self.transform_target_to_base.actInv(transform_frame_to_base)
        jacobian_in_frame = configuration.get_frame_jacobian(self.frame)
        J = -pin.Jlog6(transform_frame_to_target) @ jacobian_in_frame
        return J
