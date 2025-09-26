# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import dataclass
from typing import Optional

# Local foot pedal handler
from isaaclab.devices.openxr.retargeters.humanoid.foot_pedal_handler import FootPedalHandler, FootPedalOutput, PedalMode
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg


@dataclass
class G1LowerBodyFootPedalRetargeterCfg(RetargeterCfg):
    """Configuration for the G1 lower body foot pedal retargeter."""

    # Default standing hip height (m). Used when no vertical command is given.
    hip_height: float = 0.72

    # Velocity and position limits
    max_linear_vel_mps: float = 1.0
    max_angular_vel_radps: float = 1.0

    # Hip height range for vertical mode (m)
    min_squat_pos: float = 0.60
    max_squat_pos: float = 0.72

    # Deadzone: values <= threshold are treated as zero
    deadzone_threshold: float = 0.0

    # Foot pedal device configuration
    device_path: str = "/dev/input/js0"
    foot_pedal_update_interval: float = 0.02  # seconds


class G1LowerBodyFootPedalRetargeter(RetargeterBase):
    """Maps foot pedal input to G1 lower body base commands.

    Output tensor format: [vx, vy, wz, hip_height]
      - vx: forward (+) / backward (-) velocity (m/s)
      - vy: left (+) / right (-) lateral velocity (m/s)
      - wz: yaw rate (rad/s)
      - hip_height: commanded hip height (m)

    Mode semantics (click to toggle modes handled by FootPedalHandler):
      - FORWARD_MODE:
          both pedals > 0 -> vx > 0 (geometric mean for balance)
          only left  > 0 -> vy > 0 (step left)
          only right > 0 -> vy < 0 (step right)
      - REVERSE_MODE:
          right pedal > 0 -> vx < 0 (reverse)
      - VERTICAL_MODE:
          left  pedal > 0 -> decrease hip height from max toward min
    Rudder (axis 2) always contributes to wz.
    """

    def __init__(self, cfg: G1LowerBodyFootPedalRetargeterCfg):
        """Initialize the retargeter and start the foot pedal handler if available."""
        self.cfg = cfg
        self._handler: FootPedalHandler | None = None
        self._started: bool = False

        # Ensure sensible hip height defaults
        if self.cfg.max_squat_pos < self.cfg.min_squat_pos:
            self.cfg.max_squat_pos, self.cfg.min_squat_pos = (
                self.cfg.min_squat_pos,
                self.cfg.max_squat_pos,
            )

        # Try to start the foot pedal handler; if unavailable, fall back gracefully
        try:
            self._handler = FootPedalHandler(
                device_path=self.cfg.device_path,
                foot_pedal_update_interval=self.cfg.foot_pedal_update_interval,
            )
            self._handler.start()
            self._started = True
        except Exception:
            # Device may be missing; operate with zeros and default hip height
            self._handler = None
            self._started = False

    def _apply_deadzone(self, value: float) -> float:
        if value <= self.cfg.deadzone_threshold:
            return 0.0
        return value

    def _default_command(self) -> torch.Tensor:
        # No pedal input -> stand at max_squat_pos (or hip_height if provided higher)
        default_hip = max(self.cfg.max_squat_pos, self.cfg.hip_height)
        return torch.tensor([0.0, 0.0, 0.0, default_hip], device=self.cfg.sim_device)

    def _compute_from_output(self, pedal_out: FootPedalOutput) -> torch.Tensor:
        # Extract raw axis values: [L, R, Rz]
        left = float(pedal_out.raw_axis_values[0].item())
        right = float(pedal_out.raw_axis_values[1].item())
        rudder = float(pedal_out.raw_axis_values[2].item())

        # Apply deadzone only to pedals (rudder handled with small threshold)
        left = self._apply_deadzone(left)
        right = self._apply_deadzone(right)

        vx = 0.0
        vy = 0.0
        wz = 0.0
        hip = self.cfg.max_squat_pos

        # Mode-specific processing
        if pedal_out.current_mode == PedalMode.FORWARD_MODE:
            if left > 0.0 and right > 0.0:
                # Geometric mean encourages balanced pressing
                forward_intensity = (left * right) ** 0.5
                vx = forward_intensity * self.cfg.max_linear_vel_mps
            elif right <= 0.0 < left:
                vy = left * self.cfg.max_linear_vel_mps
            elif left <= 0.0 < right:
                vy = -right * self.cfg.max_linear_vel_mps
        elif pedal_out.current_mode == PedalMode.REVERSE_MODE:
            if right > 0.0:
                vx = -right * self.cfg.max_linear_vel_mps
        elif pedal_out.current_mode == PedalMode.VERTICAL_MODE:
            if left > 0.0:
                # Map left pedal depth to hip height within [min, max]
                hip = self.cfg.max_squat_pos - left * (self.cfg.max_squat_pos - self.cfg.min_squat_pos)
            else:
                hip = self.cfg.max_squat_pos
        else:
            # Unknown mode -> default output
            return self._default_command()

        # Rudder contributes to yaw rate with small threshold
        if abs(rudder) > 0.1:
            wz = max(
                -self.cfg.max_angular_vel_radps,
                min(self.cfg.max_angular_vel_radps, rudder * self.cfg.max_angular_vel_radps),
            )

        return torch.tensor([vx, vy, wz, hip], device=self.cfg.sim_device)

    def retarget(self, data: dict) -> torch.Tensor:
        # If handler unavailable, return default standing command
        if self._handler is None:
            return self._default_command()

        # Try to start if not started yet (e.g., device became available later)
        if not self._started:
            try:
                self._handler.start()
                self._started = True
            except Exception:
                return self._default_command()

        # Read current pedal state and compute command
        try:
            pedal_out = self._handler.get_raw_axis_values()
            return self._compute_from_output(pedal_out)
        except Exception:
            # Any runtime error -> safe default
            return self._default_command()
