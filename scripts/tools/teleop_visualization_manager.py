import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.ui.xr_widgets.instruction_widget import hide_instruction
from isaaclab.ui.xr_widgets import VisualizationManager, TriggerType, DataCollector, VisualizationManager
from pxr import Gf, Usd
import numpy as np
from isaaclab.devices.openxr.openxr_device import OpenXRDevice
import torch
from typing import Any, Final
import omni.kit.app
import json
import carb
from omni.kit.viewport.utility.camera_state import ViewportCameraState

def send_message_to_client(message: dict):
    """Send a message to the CloudXR client.
    
    Args:
        message (dict or str): Message to send (will be converted to JSON if it's a dict)
    """
    if isinstance(message, dict):
        message_str = json.dumps(message)
    else:
        message_str = message

    omni.kit.app.queue_event("omni.kit.cloudxr.send_message", payload={"message": message_str})

class TeleopVisualizationManager(VisualizationManager):
    """Specialized visualization manager for teleoperation scenarios.
    For sample and debug use.
    
    Provides teleoperation-specific visualization features including:
    - IK error handling and display
    - Hand position tracking and range indicators
    - Real-time data panels to display data in DataCollector
    """
    
    def __init__(self, data_collector: DataCollector):
        """Initialize the teleop visualization manager and register callbacks.
        
        Args:
            data_collector: DataCollector instance to read data for visualization use.
        """
        super().__init__(data_collector)

        # Register the event alias for sending messages to the CloudXR client
        carb_event = carb.events.type_from_string("omni.kit.cloudxr.send_message")
        omni.kit.app.register_event_alias(carb_event, "omni.kit.cloudxr.send_message")

        # Config whether to visualize the markers. Default to False.
        self._enable_visualization = False

        # Register callback to update the enable_visualization
        self.register_callback(TriggerType.TRIGGER_ON_EVENT, {"event_name": "enable_teleop_visualization"}, self._handle_enable_visualization)

        # Handle error event
        self._error_text_color = 0xFF0000FF
        self.ik_error_widget_id = "/ik_solver_failed"

        #self.display_widget("IK Error Detected", self.ik_error_widget_id, VisualizationManager.message_widget_preset() | {"text_color": self._error_text_color, "display_duration": None})
        
        self.register_callback(TriggerType.TRIGGER_ON_EVENT, {"event_name": "ik_error"}, self._handle_ik_error)

        # Handle torque skeleton
        self._num_open_xr_hand_joints = 52
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/skeleton_joints",
            markers={
                "grey": sim_utils.SphereCfg(
                    radius=0.005,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.5, 0.3)),
                ),
                "green": sim_utils.SphereCfg(
                    radius=0.005,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
                "yellow": sim_utils.SphereCfg(
                    radius=0.005,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
                ),
                "red": sim_utils.SphereCfg(
                    radius=0.005,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            },
        )
        self._markers_joints = VisualizationMarkers(marker_cfg)
        self._markers_joints.set_visibility(False)

        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/skeleton_lines",
            markers={
                "grey": sim_utils.CylinderCfg(
                    radius=0.001,
                    height=1,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.5, 0.3)),
                ),
                "green": sim_utils.CylinderCfg(
                    radius=0.001,
                    height=1,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
                "yellow": sim_utils.CylinderCfg(
                    radius=0.001,
                    height=1,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
                ),
                "red": sim_utils.CylinderCfg(
                    radius=0.001,
                    height=1,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            },
        )
        self._markers_lines = VisualizationMarkers(marker_cfg)
        self._markers_lines.set_visibility(False)
        # Expect to update the skeleton every frame
        self.register_callback(TriggerType.TRIGGER_ON_UPDATE, {}, self._update_torque_skeleton)

        # Todo: enable this after StreamSDK supports sending large data through the channel.
        # For now, client app crashes when receiving large data (~1KB per frame)
        # self.register_callback(TriggerType.TRIGGER_ON_UPDATE, {}, self._send_visualization_data_to_client)

    def _handle_enable_visualization(self, mgr: VisualizationManager, data_collector: DataCollector, enabled: Any) -> None:
        """Update the enable visualization.
        
        Args:
            data_collector: DataCollector instance containing current data
        """
        if enabled is not None and enabled != self._enable_visualization:
            self._enable_visualization = enabled
            self._markers_joints.set_visibility(enabled)
            self._markers_lines.set_visibility(enabled)

    def _handle_ik_error(self, mgr: VisualizationManager, data_collector: DataCollector, params: Any = None) -> None:
        """Handle IK error events by displaying an error message widget.
        
        Args:
            data_collector: DataCollector instance (unused in this handler)
        """
        # Todo: move display_widget to instruction_widget.py
        if not hasattr(mgr, "_ik_error_widget_timer"):
            self.display_widget("IK Error Detected", mgr.ik_error_widget_id, VisualizationManager.message_widget_preset() | {"text_color": self._error_text_color, "display_duration": None})
            mgr._ik_error_widget_timer = mgr.register_callback(TriggerType.TRIGGER_ON_PERIOD, {"period": 3.0, "initial_countdown": 3.0}, self._hide_ik_error_widget)
            if mgr._ik_error_widget_timer is None:
                mgr.cancel_rule(TriggerType.TRIGGER_ON_PERIOD, mgr._ik_error_widget_timer)
                mgr.cancel_rule(TriggerType.TRIGGER_ON_EVENT, "ik_solver_failed")
                raise RuntimeWarning("Failed to register IK error widget timer")
        else:
            mgr._ik_error_widget_timer.countdown = 3.0

    def _hide_ik_error_widget(self, mgr: VisualizationManager, data_collector: DataCollector) -> None:
        """Hide the IK error widget.
        
        Args:
            data_collector: DataCollector instance (unused in this handler)
        """

        hide_instruction(mgr.ik_error_widget_id)
        mgr.cancel_rule(TriggerType.TRIGGER_ON_PERIOD, mgr._ik_error_widget_timer)
        delattr(mgr, "_ik_error_widget_timer")


    def _update_torque_skeleton(self, mgr: VisualizationManager, data_collector: DataCollector) -> None:
        """Update the torque skeleton.
        
        Args:
            data_collector: DataCollector instance containing current data
        """
        if not mgr._enable_visualization:
            return

        data = data_collector.get_data("device_raw_data")
        sim_device = data_collector.get_data("sim_device")
        if data is None or sim_device is None:
            return

        left_hand_poses = data.get(OpenXRDevice.TrackingTarget.HAND_LEFT)
        right_hand_poses = data.get(OpenXRDevice.TrackingTarget.HAND_RIGHT)
        
        joints_position = np.zeros((self._num_open_xr_hand_joints, 3))

        # Extract joint positions from both hands
        left_joints = np.array([pose[:3] for pose in left_hand_poses.values()])
        right_joints = np.array([pose[:3] for pose in right_hand_poses.values()])
        
        # Fill the first part with left hand joints
        num_left = len(left_joints)
        joints_position[:num_left] = left_joints
        
        # Fill the second part with right hand joints
        joints_position[num_left:num_left + len(right_joints)] = right_joints

        viewport_api = omni.kit.viewport.utility.get_active_viewport()
        # camera_state = ViewportCameraState(viewport_api.get_active_camera(), viewport_api)
        # camera_position = np.array(camera_state.position)

        camera_state = ViewportCameraState(viewport_api.get_active_camera(), viewport_api)
        world_transform = camera_state.usd_camera.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        camera_position = np.array(world_transform.ExtractTranslation())

        # Move all joints closer to the camera for better visualization
        direction_to_camera = camera_position - joints_position
        distance_to_camera = np.linalg.norm(direction_to_camera, axis=1, keepdims=True)
        joints_position += direction_to_camera / (distance_to_camera + 1e-8) * 0.03

        # Calculate midpoints between consecutive joints for line visualization
        joints_midpoints = (joints_position[:-1] + joints_position[1:]) / 2
        
        # Calculate direction vectors
        directions = joints_position[1:] - joints_position[:-1]
        
        # Remove unnecessary joints
        indices_to_remove = [0, 1, 5, 10, 15, 20, 25, 26, 27, 31, 36, 41, 46]
        joints_midpoints = np.delete(joints_midpoints, indices_to_remove, axis=0)
        directions = np.delete(directions, indices_to_remove, axis=0)

        # Calculate lengths
        lengths = np.linalg.norm(directions, axis=1)
        
        # Normalize direction vectors
        normalized_directions = directions / (lengths[:, np.newaxis] + 1e-8)  # Add small epsilon to avoid division by zero
        
        # Calculate orientations (quaternions) to align cylinders with direction vectors
        # Cylinder default axis is Z-axis [0, 0, 1]
        default_axis = np.array([0, 0, 1])
        orientations = []
        
        for direction in normalized_directions:
            # Calculate rotation quaternion from default axis to target direction
            dot = np.dot(default_axis, direction)
            if dot > 0.9999:  # Vectors are already aligned
                quat = np.array([1, 0, 0, 0])  # Identity quaternion (w, x, y, z)
            elif dot < -0.9999:  # Vectors are opposite
                # Find perpendicular axis for 180-degree rotation
                perp = np.array([1, 0, 0]) if abs(default_axis[0]) < 0.9 else np.array([0, 1, 0])
                axis = np.cross(default_axis, perp)
                if np.linalg.norm(axis) > 0:
                    axis = axis / np.linalg.norm(axis)
                quat = np.array([0, axis[0], axis[1], axis[2]])  # 180-degree rotation
            else:
                # General case: calculate rotation axis and angle
                axis = np.cross(default_axis, direction)
                if np.linalg.norm(axis) > 0:
                    axis = axis / np.linalg.norm(axis)
                angle = np.arccos(np.clip(dot, -1, 1))
                half_angle = angle / 2
                w = np.cos(half_angle)
                xyz = axis * np.sin(half_angle)
                quat = np.array([w, xyz[0], xyz[1], xyz[2]])  # (w, x, y, z)
            
            orientations.append(quat)
        
        orientations = np.array(orientations)
        
        # Set scales: keep radius at 1, set height to the distance between points
        scales = np.column_stack([np.ones(len(lengths)), np.ones(len(lengths)), lengths])

        colors_lines = np.zeros(len(joints_midpoints))
        colors_joints = np.zeros(len(joints_position))   

        joints_torque : Final = data_collector.get_data("joints_torque")
        joints_torque_limit : Final = data_collector.get_data("joints_torque_limit")
        joints_name : Final = data_collector.get_data("joints_name")
        hand_torque_mapping : Final = data_collector.get_data("hand_torque_mapping")

        # enable_torque_color needs to be manually set by calling XRVisualization.set_attrs({"enable_torque_color": True})
        if getattr(mgr, "enable_torque_color") and joints_torque is not None and joints_torque_limit is not None and joints_name is not None and hand_torque_mapping is not None and len(hand_torque_mapping) == len(colors_lines) - 10:
            # Insert empty strings at positions that are not fingers
            hand_torque_mapping_copy = hand_torque_mapping.copy()
            hand_torque_mapping_copy.append("")
            torque_mapping_lines_index = np.array([-1, 0, 1,
                                    -1, 2, 3, 4,
                                    -1, 5, 6, 7,
                                    -1, 8, 9, 10,
                                    -1, 11, 12, 13,
                                    -1, 14, 15,
                                    -1, 16, 17, 18,
                                    -1, 19, 20, 21,
                                    -1, 22, 23, 24,
                                    -1, 25, 26, 27])
            torque_mapping_joints_index = np.array([-1, -1, -1, 0, 1, 1,
                                    -1, 2, 3, 4, 4,
                                    -1, 5, 6, 7, 7,
                                    -1, 8, 9, 10, 10,
                                    -1, 11, 12, 13, 13,
                                    -1, -1, -1, 14, 15, 15,
                                    -1, 16, 17, 18, 18,
                                    -1, 19, 20, 21, 21,
                                    -1, 22, 23, 24, 24,
                                    -1, 25, 26, 27, 27])
            torque_mapping_lines = np.array(hand_torque_mapping_copy)[torque_mapping_lines_index]
            torque_mapping_joints = np.array(hand_torque_mapping_copy)[torque_mapping_joints_index]
            
            # Set colors: 0: grey, 1: green, 2: yellow, 3: red 
            for i, key in enumerate(torque_mapping_lines):
                if key in joints_name:
                    ratio = joints_torque[joints_name.index(key)] / joints_torque_limit[joints_name.index(key)]
                    colors_lines[i] = 0 if ratio < 0.05 else 1 if ratio < 0.5 else 2 if ratio < 0.8 else 3
            for i, key in enumerate(torque_mapping_joints):
                if key in joints_name:
                    ratio = joints_torque[joints_name.index(key)] / joints_torque_limit[joints_name.index(key)]
                    colors_joints[i] = 0 if ratio < 0.05 else 1 if ratio < 0.5 else 2 if ratio < 0.8 else 3

        self._markers_joints.visualize(
            translations=torch.tensor(joints_position, device=sim_device), marker_indices=torch.tensor(colors_joints, device=sim_device)
        )
        self._markers_lines.visualize(
            translations=torch.tensor(joints_midpoints, device=sim_device), 
            orientations=torch.tensor(orientations, device=sim_device), 
            scales=torch.tensor(scales, device=sim_device),
            marker_indices=torch.tensor(colors_lines, device=sim_device)
        )

    def _send_visualization_data_to_client(self, mgr: VisualizationManager, data_collector: DataCollector) -> None:
        """Send the data to the CloudXR client.
        
        Args:
            data_collector: DataCollector instance containing current data
        """
        dic = {"Type": "visualization_data"}

        ellipsoid = data_collector.get_data("manipulability_ellipsoid")
        if ellipsoid is not None:
            dic["ellipsoid"] = ellipsoid
        distance_to_limit = data_collector.get_data("joints_distance_percentage_to_limit")
        if distance_to_limit is not None:
            dic["distance_to_limit"] = distance_to_limit
        torque_to_limit = data_collector.get_data("joints_torque_percentage_of_limit")
        if torque_to_limit is not None:
            dic["torque_to_limit"] = torque_to_limit
        joints_name = data_collector.get_data("joints_name")
        if joints_name is not None:
            dic["joints_name"] = joints_name

        if len(dic) > 1:
            send_message_to_client(dic)
