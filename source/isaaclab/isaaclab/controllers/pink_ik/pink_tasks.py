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
    """Frame task that also supports ``class_type(cfg)`` construction."""

    def __init__(
        self,
        cfg_or_frame,
        position_cost: float | Sequence[float] | None = None,
        orientation_cost: float | Sequence[float] | None = None,
        lm_damping: float = 0.0,
        gain: float = 1.0,
    ):
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
    """Damping task accepting either cfg or direct cost argument."""

    def __init__(self, cfg_or_cost, cost: float | None = None):
        if isinstance(cfg_or_cost, (int, float)):
            _cost = float(cfg_or_cost if cost is None else cost)
        else:
            _cost = cfg_or_cost.cost
        super().__init__(cost=_cost)


class LocalFrameTask(FrameTask):
    """
    A task that computes error in a local (custom) frame.
    Inherits from FrameTask but overrides compute_error.
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
        self.transform_target_to_base = transform_target_to_base.copy()

    def set_target_from_configuration(self, configuration: PinkKinematicsConfiguration) -> None:
        if not isinstance(configuration, PinkKinematicsConfiguration):
            raise ValueError("configuration must be a PinkKinematicsConfiguration")
        self.set_target(configuration.get_transform(self.frame, self.base_link_frame_name))

    def compute_error(self, configuration: PinkKinematicsConfiguration) -> np.ndarray:
        if not isinstance(configuration, PinkKinematicsConfiguration):
            raise ValueError("configuration must be a PinkKinematicsConfiguration")
        if self.transform_target_to_base is None:
            raise ValueError(f"no target set for frame '{self.frame}'")

        transform_frame_to_base = configuration.get_transform(self.frame, self.base_link_frame_name)
        transform_target_to_frame = transform_frame_to_base.actInv(self.transform_target_to_base)

        error_in_frame: np.ndarray = pin.log(transform_target_to_frame).vector
        return error_in_frame

    def compute_jacobian(self, configuration: PinkKinematicsConfiguration) -> np.ndarray:
        if self.transform_target_to_base is None:
            raise Exception(f"no target set for frame '{self.frame}'")
        transform_frame_to_base = configuration.get_transform(self.frame, self.base_link_frame_name)
        transform_frame_to_target = self.transform_target_to_base.actInv(transform_frame_to_base)
        jacobian_in_frame = configuration.get_frame_jacobian(self.frame)
        J = -pin.Jlog6(transform_frame_to_target) @ jacobian_in_frame
        return J
