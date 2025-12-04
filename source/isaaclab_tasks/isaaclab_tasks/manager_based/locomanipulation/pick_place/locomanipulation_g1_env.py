# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from isaacsim.core.utils.stage import get_current_stage
from pxr import Sdf, UsdGeom, UsdShade

from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.openxr import OpenXRDeviceMotionController
from isaaclab.devices.openxr.xr_cfg import XrAnchorRotationMode
from isaaclab.envs import ManagerBasedRLEnv

from .locomanipulation_g1_env_cfg import LocomanipulationG1EnvCfg


class LocomanipulationG1ManagerBasedRLEnv(ManagerBasedRLEnv):
    cfg: LocomanipulationG1EnvCfg

    def __init__(self, cfg: LocomanipulationG1EnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        print("[INFO]: Completed setting up the LocomanipulationG1ManagerBasedRLEnv...")

    def setup_for_teleop_device(self, teleop_device: DeviceBase, args_cli: argparse.Namespace) -> None:
        """Configure environment based on the teleop device type.

        Args:
            teleop_device: The teleop device being used.
        """

        if not args_cli.xr:
            return

        stage = get_current_stage()

        # Hide the robot head mesh for XR comfort
        inst_root_path = "/World/envs/env_0/Robot/torso_link/visuals"
        target_path = "/World/envs/env_0/Robot/torso_link/visuals/head_link/mesh"

        inst_root = stage.GetPrimAtPath(inst_root_path)
        if inst_root:
            # Uninstance this robot's visuals so children are editable
            if inst_root.IsInstance():
                with Sdf.ChangeBlock():
                    inst_root.SetInstanceable(False)

        # Now hide the head mesh on this instance only
        target = stage.GetPrimAtPath(target_path)
        if target:
            UsdGeom.Imageable(target).MakeInvisible()

        # Only run ground material change for OpenXR motion controller devices
        if not isinstance(teleop_device, OpenXRDeviceMotionController):
            return

        # Change the material of the ground plane for comfort when we are using FOLLOW_PRIM
        # Rotating while parented to the robot's pelvis could cause nausea when viewing a grid plane ground.
        if (
            teleop_device._xr_cfg.anchor_rotation_mode == XrAnchorRotationMode.FOLLOW_PRIM
            or teleop_device._xr_cfg.anchor_rotation_mode == XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED
        ):
            ground_prim = stage.GetPrimAtPath("/World/GroundPlane/Environment/Geometry")
            if ground_prim is not None and ground_prim.IsValid():
                # Change material to robot's default material, which doesn't have a grid
                material_path = "/World/envs/env_0/Robot/Looks/DefaultMaterial"
                material_prim = stage.GetPrimAtPath(material_path)
                if material_prim and material_prim.IsValid():
                    # Apply the material to the ground prim
                    material = UsdShade.Material(material_prim)
                    UsdShade.MaterialBindingAPI(ground_prim).Bind(material)
