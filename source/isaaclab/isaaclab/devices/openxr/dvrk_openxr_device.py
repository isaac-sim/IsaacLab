# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OpenXR motion-controller device with lifecycle forwarding for the stateful dVRK retargeter."""

from __future__ import annotations

from dataclasses import dataclass

import carb

from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.openxr.openxr_device import OpenXRDevice, OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters.manipulator.dvrk_psm_retargeter import DVRKPSMRetargeter
from isaaclab.devices.retargeter_base import RetargeterBase


class DVRKOpenXRDevice(OpenXRDevice):
    """OpenXR motion-controller device that owns the dVRK retargeter's session lifecycle.

    Raw controller tracking, anchors, application callbacks, and XRCore resources remain
    owned by :class:`OpenXRDevice`. This subclass only forwards START, STOP,
    and RESET transitions to its single :class:`DVRKPSMRetargeter`.
    """

    def __init__(
        self,
        cfg: DVRKOpenXRDeviceCfg,
        retargeters: list[RetargeterBase] | None = None,
    ):
        if retargeters is None or len(retargeters) != 1 or not isinstance(retargeters[0], DVRKPSMRetargeter):
            raise ValueError("DVRKOpenXRDevice requires exactly one DVRKPSMRetargeter")
        self._dvrk_retargeter = retargeters[0]
        super().__init__(cfg=cfg, retargeters=retargeters)
        if cfg.teleoperation_active_default:
            self._dvrk_retargeter.start()
        else:
            self._dvrk_retargeter.stop()

    def reset(self) -> None:
        """Reset dVRK target state once, then reset the base OpenXR state."""
        self._dvrk_retargeter.reset()
        super().reset()

    def _on_teleop_command(self, event: carb.events.IEvent) -> None:
        """Apply dVRK START/STOP transitions before base application callbacks."""
        message = event.payload["message"]
        if "start" in message:
            self._dvrk_retargeter.start()
        elif "stop" in message:
            self._dvrk_retargeter.stop()

        # The base RESET path invokes the application callback and then calls
        # this class's reset() override, keeping the internal transition singular.
        super()._on_teleop_command(event)


@dataclass
class DVRKOpenXRDeviceCfg(OpenXRDeviceCfg):
    """Configuration for paired OpenXR motion controllers driving a bimanual dVRK retargeter."""

    class_type: type[DeviceBase] = DVRKOpenXRDevice


__all__ = ["DVRKOpenXRDevice", "DVRKOpenXRDeviceCfg"]
