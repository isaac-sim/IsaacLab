# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional RTSP spectator stream for Kit XR teleoperation."""

from __future__ import annotations

import os


class EgocentricSpectatorRtsp:
    """Stream a fixed overview camera, optionally following the XR headset.

    The feature is enabled with ``ISAACLAB_SPECTATOR_RTSP=1``. The top-down
    table view is the default; set ``ISAACLAB_SPECTATOR_VIEW=egocentric`` to
    follow the headset instead. Construction must happen after Kit launches.
    """

    def __init__(self, env=None) -> None:
        self._enabled = os.environ.get("ISAACLAB_SPECTATOR_RTSP", "0") == "1"
        self._view = os.environ.get("ISAACLAB_SPECTATOR_VIEW", "topdown")
        self._follow_head = self._view == "egocentric"
        self._camera_transform_op = None
        self._camera_sensor = None
        self._xr_core = None
        self._tracking_announced = False
        if not self._enabled:
            return

        import omni.graph.core as og
        import omni.kit.app
        import omni.usd
        from pxr import Gf, UsdGeom

        extension_manager = omni.kit.app.get_app().get_extension_manager()
        extension_manager.set_extension_enabled_immediate("isaacsim.core.nodes", True)
        extension_manager.set_extension_enabled_immediate("isaacsim.streaming.rtsp", True)

        if not self._follow_head and env is not None and "spectator_camera" in env.scene.sensors:
            # Use the Isaac Lab camera's managed render product. Accessing data
            # drives the renderer once now and on every subsequent update.
            self._camera_sensor = env.scene.sensors["spectator_camera"]
            _ = self._camera_sensor.data.output["rgb"]
            render_product_path = self._camera_sensor._render_data.render_product.path
            camera_path = None
        else:
            stage = omni.usd.get_context().get_stage()
            camera_path = "/EgocentricSpectatorCamera"
            camera = UsdGeom.Camera.Define(stage, camera_path)
            camera.GetFocalLengthAttr().Set(24.0)
            self._camera_transform_op = UsdGeom.Xformable(camera).AddTransformOp()
            fixed_camera_transform = (
                Gf.Matrix4d()
                .SetLookAt(
                    Gf.Vec3d(0.0, 0.55, 3.2),
                    Gf.Vec3d(0.0, 0.55, 1.0),
                    Gf.Vec3d(0.0, 1.0, 0.0),
                )
                .GetInverse()
            )
            self._camera_transform_op.Set(fixed_camera_transform)
            render_product_path = None
        if self._follow_head:
            from omni.kit.xr.core import XRCore

            self._xr_core = XRCore.get_singleton()

        graph_spec = {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("RTSPHelper", "isaacsim.streaming.rtsp.RTSPCameraHelper"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("RTSPHelper.inputs:port", 8554),
                ("RTSPHelper.inputs:mountPath", "/snap-circuits"),
                ("RTSPHelper.inputs:useRawEncoding", False),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "RTSPHelper.inputs:execIn"),
            ],
        }
        if render_product_path is not None:
            graph_spec[og.Controller.Keys.SET_VALUES].append(
                ("RTSPHelper.inputs:renderProductPath", render_product_path)
            )
        else:
            graph_spec[og.Controller.Keys.CREATE_NODES].insert(
                1,
                ("CreateRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
            )
            graph_spec[og.Controller.Keys.SET_VALUES][0:0] = [
                ("CreateRenderProduct.inputs:cameraPrim", camera_path),
                ("CreateRenderProduct.inputs:width", 1280),
                ("CreateRenderProduct.inputs:height", 720),
            ]
            graph_spec[og.Controller.Keys.CONNECT] = [
                ("OnPlaybackTick.outputs:tick", "CreateRenderProduct.inputs:execIn"),
                ("CreateRenderProduct.outputs:execOut", "RTSPHelper.inputs:execIn"),
                ("CreateRenderProduct.outputs:renderProductPath", "RTSPHelper.inputs:renderProductPath"),
            ]

        og.Controller.edit(
            {"graph_path": "/EgocentricSpectatorRTSPGraph", "evaluator_name": "execution"},
            graph_spec,
        )
        view_name = "egocentric" if self._follow_head else "top-down"
        print(f"[INFO] {view_name.capitalize()} spectator stream: rtsp://<server-ip>:8554/snap-circuits")

    def update(self) -> None:
        """Copy the valid XR head pose to the spectator camera."""
        if not self._enabled:
            return
        if self._camera_sensor is not None:
            _ = self._camera_sensor.data.output["rgb"]
            return
        if not self._follow_head or self._xr_core is None:
            return

        from omni.kit.xr.core import XRPoseValidityFlags

        input_device = self._xr_core.get_input_device("displayDevice")
        if input_device is None:
            input_device = self._xr_core.get_input_device("/user/head")
        if input_device is None:
            return

        pose_desc = input_device.get_virtual_world_pose_desc("")
        if pose_desc is None:
            return
        required = XRPoseValidityFlags.POSITION_VALID | XRPoseValidityFlags.ORIENTATION_VALID
        if pose_desc.validity_flags & required != required:
            return

        self._camera_transform_op.Set(pose_desc.pose_matrix)
        if not self._tracking_announced:
            self._tracking_announced = True
            print("[INFO] Egocentric spectator camera is following the XR headset pose.")
