# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .ackermann import AckermannController


@configclass
class AckermannControllerCfg:
    """Configuration for an Ackermann steering controller."""

    class_type: type[AckermannController] | str = "{DIR}.ackermann:AckermannController"
    """The associated controller class."""

    wheel_radius: float = MISSING
    """Common wheel radius [m]."""

    wheel_base: float = MISSING
    """Longitudinal distance between the steerable and non-steerable axles [m]."""

    track_width: float = MISSING
    """Lateral distance between the left and right steerable wheels [m]."""

    non_steerable_wheel_offsets: tuple[float, ...] = ()
    """Signed lateral offsets of the non-steerable wheels [m]. Defaults to an empty tuple.

    Values use the Isaac Lab body-frame convention: positive offsets lie to the left of the vehicle centerline.
    Their order defines the order of non-steerable wheel-velocity outputs after the left and right steerable wheels.
    """

    max_linear_speed: float = float("inf")
    """Maximum magnitude of the body-frame forward-speed command [m/s]. Defaults to infinity."""

    max_steering_angle: float = math.radians(80.0)
    r"""Maximum magnitude of the virtual steering-angle command [rad]. Defaults to 80 degrees.

    The value must be strictly less than :math:`\pi / 2` because the body-forward-speed convention becomes
    singular at 90 degrees.
    """

    steerable_wheels_at_rear: bool = False
    """Whether the two steerable wheels are on the rear axle. Defaults to False."""

    def __post_init__(self) -> None:
        """Validate vehicle geometry and command limits."""
        for name, value in (
            ("wheel_radius", self.wheel_radius),
            ("wheel_base", self.wheel_base),
            ("track_width", self.track_width),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and greater than zero, got {value}.")
        if math.isnan(self.max_linear_speed) or self.max_linear_speed < 0.0:
            raise ValueError(f"max_linear_speed must be non-negative, got {self.max_linear_speed}.")
        if not math.isfinite(self.max_steering_angle) or not 0.0 <= self.max_steering_angle < math.pi / 2.0:
            raise ValueError(
                f"max_steering_angle must be finite, non-negative, and less than pi / 2, got {self.max_steering_angle}."
            )
        if not all(math.isfinite(offset) for offset in self.non_steerable_wheel_offsets):
            raise ValueError(
                f"non_steerable_wheel_offsets must contain only finite values, got {self.non_steerable_wheel_offsets}."
            )
