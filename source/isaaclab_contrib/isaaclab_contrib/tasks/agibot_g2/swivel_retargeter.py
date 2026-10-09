# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Elbow-swivel retargeter for the ik_7d 7-DoF arm.

A 6-DoF end-effector pose does not determine a 7-DoF arm's configuration: the
elbow is free to rotate about the shoulder-wrist axis. ik_7d exposes that
redundancy as ``arm_plane_angle``, and :class:`~isaaclab_contrib.mdp.Ik7dAction`
takes it as the eighth element of each arm's action. Nothing in a stock
``Se3AbsRetargeter`` produces it, so this module maps it onto the VR
controller's thumbstick.

The mapping is *rate* control, not position control. Absolute control would put
the elbow wherever the thumbstick happens to be resting when a session starts,
including at a limit; integrating the deflection instead means a centred stick
holds whatever swivel the operator arrived at, which is what a
"nudge the elbow out of the way" control wants to do.
"""

from __future__ import annotations

from dataclasses import dataclass

from isaacteleop.retargeting_engine.interface import BaseRetargeter, RetargeterIOType
from isaacteleop.retargeting_engine.interface.retargeter_core_types import RetargeterIO
from isaacteleop.retargeting_engine.interface.tensor_group_type import OptionalType, TensorGroupType
from isaacteleop.retargeting_engine.tensor_types import ControllerInput, ControllerInputIndex, FloatType


@dataclass
class SwivelRetargeterConfig:
    """Configuration for :class:`SwivelRetargeter`."""

    controller_side: str = "right"
    """Which controller's thumbstick drives the swivel -- ``"left"`` or ``"right"``."""

    rate_rad_per_s: float = 1.0
    """Swivel speed at full thumbstick deflection."""

    limit_rad: float = 1.2
    """Symmetric clamp on the accumulated swivel command.

    ``Ik7dController`` routes a positive command along the arm's *measured* free
    direction, so this is a limit on how far the elbow can be pushed away from
    its home swivel, in the one direction that is actually available. The
    default is a little under the ~1.4 rad of travel measured on ``crs``.
    """

    deadzone: float = 0.15
    """Thumbstick deflection below which the axis reads as centred."""

    invert: bool = False
    """Flip the sign of the thumbstick axis."""


class SwivelRetargeter(BaseRetargeter):
    """Integrates a controller thumbstick axis into an elbow-swivel command.

    Emits a single scalar, ``swivel``, in radians, to be fed to the eighth
    element of an arm's :class:`~isaaclab_contrib.mdp.Ik7dAction` slice.
    """

    def __init__(self, config: SwivelRetargeterConfig, name: str) -> None:
        """Initialize the retargeter.

        Args:
            config: The retargeter configuration.
            name: Node name within the retargeting graph.

        Raises:
            ValueError: If ``controller_side`` is neither ``"left"`` nor ``"right"``.
        """
        if config.controller_side not in ("left", "right"):
            raise ValueError(f"controller_side must be 'left' or 'right', got: {config.controller_side}")
        self._config = config
        # Set before ``super().__init__``: the base class calls ``input_spec``
        # from its constructor.
        self._input_key = f"controller_{config.controller_side}"
        super().__init__(name=name)

        self._angle = 0.0
        self._last_time_ns: int | None = None

    def input_spec(self) -> RetargeterIOType:
        """Requires one controller, which may be absent on any given frame."""
        return {self._input_key: OptionalType(ControllerInput())}

    def output_spec(self) -> RetargeterIOType:
        """Outputs the accumulated swivel angle in radians."""
        return {"swivel": TensorGroupType("swivel", [FloatType("angle")])}

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        """Integrate the thumbstick deflection and emit the clamped angle."""
        if context.execution_events.reset:
            self._angle = 0.0
            self._last_time_ns = None

        swivel_out = outputs["swivel"]

        now_ns = context.graph_time.sim_time_ns
        dt = 0.0 if self._last_time_ns is None else max(0.0, (now_ns - self._last_time_ns) * 1e-9)
        self._last_time_ns = now_ns

        controller = inputs[self._input_key]
        if controller.is_none:
            # Hold, rather than recentre: a dropped frame is not a command to
            # move the elbow back.
            swivel_out[0] = self._angle
            return

        axis = float(controller[ControllerInputIndex.THUMBSTICK_X])
        if self._config.invert:
            axis = -axis
        if abs(axis) < self._config.deadzone:
            axis = 0.0

        limit = self._config.limit_rad
        self._angle = min(limit, max(-limit, self._angle + axis * self._config.rate_rad_per_s * dt))
        swivel_out[0] = self._angle
