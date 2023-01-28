# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera class in Omniverse workflows."""


import builtins
import math
import numpy as np
import scipy.spatial.transform as tf
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple, Union

import carb
import omni.isaac.core.utils.prims as prim_utils
import omni.replicator.core as rep
import omni.usd
from omni.isaac.core.prims import XFormPrimView
from omni.isaac.core.simulation_context import SimulationContext
from pxr import UsdGeom

# omni-isaac-orbit
from omni.isaac.orbit.utils import class_to_dict, to_camel_case
from omni.isaac.orbit.utils.math import convert_quat

from ..sensor_base import SensorBase
from .camera_cfg import FisheyeCameraCfg, PinholeCameraCfg

__all__ = ["Camera", "CameraData"]


@dataclass
class CameraData:
    """Data container for the camera sensor."""

    position: np.ndarray = None
    """Position of the sensor origin in world frame, following ROS convention."""
    orientation: np.ndarray = None
    """Quaternion orientation `(w, x, y, z)` of the sensor origin in world frame, following ROS convention.."""
    intrinsic_matrix: np.ndarray = None
    """The intrinsic matrix for the camera."""
    image_shape: Tuple[int, int] = None
    """A tuple containing (height, width) of the camera sensor."""
    output: Dict[str, Any] = None
    """The retrieved sensor data with sensor types as key.

    The format of the data is available in the `Replicator Documentation`_.

    .. _Replicator Documentation: https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_replicator/annotators_details.html#annotator-output
    """


class Camera(SensorBase):
    r"""The camera sensor for acquiring visual data.

    Summarizing from the `replicator extension`_, the following sensor types are supported:

    - ``"rgb"``: A rendered color image.
    - ``"distance_to_camera"``: An image containing the distance to camera optical center.
    - ``"distance_to_image_plane"``: An image containing distances of 3D points from camera plane along camera's z-axis.
    - ``"normals"``: An image containing the local surface normal vectors at each pixel.
    - ``"motion_vectors"``: An image containing the motion vector data at each pixel.
    - ``"instance_segmentation"``: The instance segmentation data.
    - ``"semantic_segmentation"``: The semantic segmentation data.
    - ``"bounding_box_2d_tight"``: The tight 2D bounding box data (only contains non-occluded regions).
    - ``"bounding_box_2d_loose"``: The loose 2D bounding box data (contains occluded regions).
    - ``"bounding_box_3d"``: The 3D view space bounding box data.
    - ``"occlusion"``: The occlusion information (such as instance id, semantic id and occluded ratio).

    The camera sensor supports the following projection types:

    - ``"pinhole"``: Standard pinhole camera model (disables fisheye parameters).
    - ``"fisheye_orthographic"``: Fisheye camera model using orthographic correction.
    - ``"fisheye_equidistant"``: Fisheye camera model using equidistant correction.
    - ``"fisheye_equisolid"``: Fisheye camera model using equisolid correction.
    - ``"fisheye_polynomial"``: Fisheye camera model with :math:`360^{\circ}` spherical projection.
    - ``"fisheye_spherical"``: Fisheye camera model with :math:`360^{\circ}` full-frame projection.

    Typically, the sensor comprises of two prims:

    1. **Camera rig**: A dummy Xform prim to which the camera is attached to.
    2. **Camera prim**: An instance of the `USDGeom Camera`_.

    However, for the sake of generality, we allow omission of the camera rig prim. This is mostly the case when
    the camera is static. In such cases, any request to set the camera pose is directly set on the camera prim,
    instead of setting the pose of the camera rig Xform prim.

    .. _replicator extension: https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_replicator/annotators_details.html#annotator-output
    .. _USDGeom Camera: https://graphics.pixar.com/usd/docs/api/class_usd_geom_camera.html

    """

    def __init__(self, cfg: Union[PinholeCameraCfg, FisheyeCameraCfg], device: str = "cpu"):
        """Initializes the camera sensor.

        If the ``device`` is ``"cpu"``, the output data is returned as a numpy array. If the ``device`` is
        ``"cuda"``, then a Warp array is returned. Note that only the valid sensor types will be moved to GPU.

        Args:
            cfg (Union[PinholeCameraCfg, FisheyeCameraCfg]): The configuration parameters.
            device (str): The device on which to receive data. Defaults to "cpu".
        """
        # store inputs
        self.cfg = cfg
        self.device = device
        # initialize base class
        super().__init__(self.cfg.sensor_tick)
        # change the default rendering settings
        # TODO: Should this be done here or maybe inside the app config file?
        rep.settings.set_render_rtx_realtime(antialiasing="FXAA")

        # Acquire simulation context
        self._sim_context = SimulationContext.instance()
        # Xform prim for the camera rig
        self._sensor_rig_prim: XFormPrimView = None
        # UsdGeom Camera prim for the sensor
        self._sensor_prim: UsdGeom.Camera = None
        # Create empty variables for storing output data
        self._data = CameraData()
        self._data.output = dict.fromkeys(self.cfg.data_types, None)
        # Flag to check that sensor is spawned.
        self._is_spawned = False

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        # message for class
        return (
            f"Camera @ '{self.prim_path}': \n"
            f"\tdata types   : {list(self._data.output.keys())} \n"
            f"\ttick rate (s): {self.sensor_tick}\n"
            f"\ttimestamp (s): {self.timestamp}\n"
            f"\tframe        : {self.frame}\n"
            f"\tshape        : {self.image_shape}\n"
            f"\tposition     : {self._data.position} \n"
            f"\torientation  : {self._data.orientation} \n"
        )

    """
    Properties
    """

    @property
    def prim_path(self) -> str:
        """The path to the camera prim."""
        return prim_utils.get_prim_path(self._sensor_prim)

    @property
    def render_product_path(self) -> str:
        """The path of the render product for the camera.

        This can be used via replicator interfaces to attach to writes or external annotator registry.
        """
        return self._render_product_path

    @property
    def data(self) -> CameraData:
        """Data related to Camera sensor."""
        return self._data

    @property
    def image_shape(self) -> Tuple[int, int]:
        """A tuple containing (height, width) of the camera sensor."""
        return (self.cfg.height, self.cfg.width)

    """
    Configuration
    """

    def set_visibility(self, visible: bool):
        """Set visibility of the instance in the scene.

        Args:
            visible (bool): Whether to make instance visible or invisible.

        Raises:
            RuntimeError: If the camera prim is not set. Need to call `initialize(...)` first.
        """
        # check camera prim
        if self._sensor_prim is None:
            raise RuntimeError("Camera prim is None. Please call 'initialize(...)' first.")
        # get imageable object
        imageable = UsdGeom.Imageable(self._sensor_prim)
        # set visibility
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()

    def set_intrinsic_matrix(self, matrix: np.ndarray, focal_length: float = 1.0):
        """Set parameters of the USD camera from its intrinsic matrix.

        Due to limitations of Omniverse camera, we need to assume that the camera is a spherical lens,
        i.e. has square pixels, and the optical center is centered at the camera eye. If this assumption
        is not true in the input intrinsic matrix, then the camera will not set up correctly.

        Args:
            intrinsic_matrix (np.ndarray): The intrinsic matrix for the camera.
            focal_length (float, optional): Focal length to use when computing aperture values. Defaults to 1.0.
        """
        # convert to numpy for sanity
        intrinsic_matrix = np.asarray(matrix).astype(np.float)
        # extract parameters from matrix
        f_x = intrinsic_matrix[0, 0]
        c_x = intrinsic_matrix[0, 2]
        f_y = intrinsic_matrix[1, 1]
        c_y = intrinsic_matrix[1, 2]
        # get viewport parameters
        height, width = self.image_shape
        height, width = float(height), float(width)
        # resolve parameters for usd camera
        params = {
            "focal_length": focal_length,
            "horizontal_aperture": width * focal_length / f_x,
            "vertical_aperture": height * focal_length / f_y,
            "horizontal_aperture_offset": (c_x - width / 2) / f_x,
            "vertical_aperture_offset": (c_y - height / 2) / f_y,
        }
        # set parameters for camera
        for param_name, param_value in params.items():
            # convert to camel case (CC)
            param_name = to_camel_case(param_name, to="CC")
            # get attribute from the class
            param_attr = getattr(self._sensor_prim, f"Get{param_name}Attr")
            # set value
            # note: We have to do it this way because the camera might be on a different layer (default cameras are on session layer),
            #   and this is the simplest way to set the property on the right layer.
            omni.usd.utils.set_prop_val(param_attr(), param_value)

    """
    Operations - Set pose.
    """

    def set_world_pose_ros(self, pos: Sequence[float] = None, quat: Sequence[float] = None):
        """Set the pose of the camera w.r.t. world frame using ROS convention.

        This methods uses the ROS convention to resolve the input pose. In this convention,
        we assume that the camera front-axis is +Z-axis and up-axis is -Y-axis.

        Args:
            pos (Sequence[float], optional): The cartesian coordinates (in meters). Defaults to None.
            quat (Sequence[float], optional): The quaternion orientation in (w, x, y, z). Defaults to None.

        Raises:
            RuntimeError: If the camera prim is not set. Need to call :meth:`initialize` method first.
        """
        # add note that this function is not working correctly
        # FIXME: Fix this function. Getting the camera pose and setting back over here doesn't work.
        carb.log_warn("The function `set_world_pose_ros` is currently not implemented correctly.")

        # check camera prim exists
        if self._sensor_prim is None:
            raise RuntimeError("Camera prim is None. Please call 'initialize(...)' first.")
        # convert from meters to stage units
        if pos is not None:
            pos = np.asarray(pos)
        # convert rotation matrix from ROS convention to OpenGL
        if quat is not None:
            rotm = tf.Rotation.from_quat(convert_quat(quat, "xyzw")).as_matrix()
            rotm[:, 2] = -rotm[:, 2]
            rotm[:, 1] = -rotm[:, 1]
            rotm = rotm.transpose()
            quat_gl = tf.Rotation.from_matrix(rotm).as_quat()
            # convert to isaac-sim convention
            quat_gl = convert_quat(quat_gl, "wxyz")
        else:
            quat_gl = None
        # set the pose
        if self._sensor_rig_prim is None:
            # Note: Technically, we should prefer not to do this.
            cam_prim = XFormPrimView(self.prim_path, reset_xform_properties=True)
            cam_prim.set_world_poses(pos, quat_gl)
        else:
            self._sensor_rig_prim.set_world_poses(pos, quat_gl)

    def set_world_pose_from_ypr(
        self, target_position: Sequence[float], distance: float, yaw: float, pitch: float, roll: float, up_axis: str
    ):
        """Computes the view matrix from the inputs and sets the camera prim pose.

        The implementation follows from the `computeViewMatrixFromYawPitchRoll()` function in Bullet3.

        Args:
            target_position (Sequence[float]): Target focus point in cartesian world coordinates.
            distance (float): Distance from eye to focus point.
            yaw (float): Yaw angle in degrees (up, down).
            pitch (float): Pitch angle in degrees around up vector.
            roll (float): Roll angle in degrees around forward vector.
            up_axis (str): The up axis for the camera. Either 'y', 'Y' or 'z', 'Z' axis.

        Raises:
            RuntimeError: If the camera prim is not set. Need to call :meth:`initialize` method first.
            ValueError: When the ``up_axis`` is not "y/Y" or "z/Z".
        """
        # sanity conversion
        camera_target_position = np.asarray(target_position)
        up_axis = up_axis.upper()
        # check camera prim exists
        if self._sensor_prim is None:
            raise RuntimeError("Camera prim is None. Please call 'initialize(...)' first.")
        # check inputs
        if up_axis not in ["Y", "Z"]:
            raise ValueError(f"Invalid specified up axis '{up_axis}'. Valid options: ['Y', 'Z'].")
        # compute camera's eye pose
        if up_axis == "Y":
            eye_position = np.array([0.0, 0.0, -distance])
            eye_rotation = tf.Rotation.from_euler("ZYX", [roll, yaw, -pitch], degrees=True).as_matrix()
            up_vector = np.array([0.0, 1.0, 0.0])
        else:
            eye_position = np.array([0.0, -distance, 0.0])
            eye_rotation = tf.Rotation.from_euler("ZYX", [yaw, roll, pitch], degrees=True).as_matrix()
            up_vector = np.array([0.0, 0.0, 1.0])
        # transform eye to get camera position
        cam_up_vector = np.dot(eye_rotation, up_vector)
        cam_position = np.dot(eye_rotation, eye_position) + camera_target_position
        # axis direction for camera's view matrix
        f = (camera_target_position - cam_position) / np.linalg.norm(camera_target_position - cam_position)
        u = cam_up_vector / np.linalg.norm(cam_up_vector)
        s = np.cross(f, u)
        # create camera's view matrix: camera_T_world
        cam_view_matrix = np.eye(4)
        cam_view_matrix[:3, 0] = s
        cam_view_matrix[:3, 1] = u
        cam_view_matrix[:3, 2] = -f
        cam_view_matrix[3, 0] = -np.dot(s, cam_position)
        cam_view_matrix[3, 1] = -np.dot(u, cam_position)
        cam_view_matrix[3, 2] = np.dot(f, cam_position)
        # compute camera transform: world_T_camera
        cam_tf = np.linalg.inv(cam_view_matrix)
        cam_quat = tf.Rotation.from_matrix(cam_tf[:3, :3].T).as_quat()
        cam_pos = cam_tf[3, :3]
        # set camera pose
        self.set_camera_pose(cam_pos, cam_quat)

    def set_world_pose_from_view(self, eye: Sequence[float], target: Sequence[float] = [0, 0, 0], vel: float = 0.0):
        """Set the pose of the camera from the eye position and look-at target position.

        Warn:
            This method directly sets the camera prim pose and not the pose of the camera rig.
            It is advised not to use it when the camera is part of a sensor rig.

        Args:
            eye (Sequence[float]): The position of the camera's eye.
            target (Sequence[float], optional): The target location to look at. Defaults to [0, 0, 0].
            vel (float, optional): The velocity of the camera.. Defaults to 0.0.
        """
        with self._rep_camera:
            rep.modify.pose(position=eye, look_at=target)
        # FIXME: add note that this function is not working correctly
        carb.log_warn("The function `set_world_pose_from_view` is currently not implemented correctly.")

    """
    Operations
    """

    def spawn(self, parent_prim_path: str, translation: Sequence[float] = None, orientation: Sequence[float] = None):
        """Spawns the sensor into the stage.

        The sensor is spawned under the parent prim at the path ``parent_prim_path`` with the provided input
        rotation and translation. The USD Camera prim is attached to the parent prim.

        Args:
            parent_prim_path (str): The path of the parent prim to attach sensor to.
            translation (Sequence[float], optional): The local position offset w.r.t. parent prim. Defaults to None.
            orientation (Sequence[float], optional): The local rotation offset in `(w, x, y, z)` w.r.t. parent prim. Defaults to None.
        """
        # Convert to camel case
        projection_type = to_camel_case(self.cfg.projection_type, to="cC")
        # Create camera using replicator. This creates under it two prims:
        # 1) the rig: at the path f"{prim_path}/Camera_Xform"
        # 2) the USD camera: at the path f"{prim_path}/Camera_Xform/Camera"
        self._rep_camera = rep.create.camera(
            parent=parent_prim_path,
            projection_type=projection_type,
            **class_to_dict(self.cfg.usd_params),
        )
        # Acquire the sensor prims
        # 1) the rig
        cam_rig_prim_path = rep.utils.get_node_targets(self._rep_camera.node, "inputs:prims")[0]
        self._sensor_rig_prim = XFormPrimView(cam_rig_prim_path, reset_xform_properties=False)
        # 2) the USD camera
        cam_prim = prim_utils.get_prim_children(prim_utils.get_prim_at_path(cam_rig_prim_path))[0]
        self._sensor_prim = UsdGeom.Camera(cam_prim)
        # Set the transformation of the camera
        # Note: As mentioned in Isaac Sim documentation, it is better to never transform to camera directly.
        #   Hence, we only transform the camera rig.
        self._sensor_rig_prim.set_local_poses(translation, orientation)
        # Set spawning to true
        self._is_spawned = True

    def initialize(self, cam_prim_path: str = None, has_rig: bool = False):
        """Initializes the sensor handles and internal buffers.

        This function creates handles and registers the provided data types with the replicator registry to
        be able to access the data from the sensor. It also initializes the internal buffers to store the data.

        The function also allows initializing to a camera not spawned by using the :meth:`spawn` method.
        For instance, connecting to the default viewport camera "/Omniverse_persp". In such cases, it is
        the user's responsibility to ensure that the camera is valid and inform the sensor class whether
        the camera is part of a rig or not.

        Args:
            cam_prim_path (str, optional): The prim path to existing camera. Defaults to None.
            has_rig (bool, optional): Whether the passed camera prim path is attached to a rig. Defaults to False.

        Raises:
            RuntimeError: When input `cam_prim_path` is :obj:`None`, the method defaults to using the last
                camera prim path set when calling the :meth:`spawn()` function. In case, the camera was not spawned
                and no valid `cam_prim_path` is provided, the function throws an error.
        """
        # Check that sensor has been spawned
        if cam_prim_path is None:
            if not self._is_spawned:
                raise RuntimeError("Initialize the camera failed! Please provide a valid argument for `prim_path`.")
        else:
            # Force to set active camera to input prim path/
            cam_prim = prim_utils.get_prim_at_path(cam_prim_path)
            self._sensor_prim = UsdGeom.Camera(cam_prim)
            # Check rig
            if has_rig:
                self._sensor_rig_prim = XFormPrimView(cam_prim_path.rsplit("/", 1)[0], reset_xform_properties=True)
            else:
                self._sensor_rig_prim = None

        # Enable synthetic data sensors
        self._render_product_path = rep.create.render_product(
            self.prim_path, resolution=(self.cfg.width, self.cfg.height)
        )
        # Attach the sensor data types to render node
        self._rep_registry: Dict[str, rep.annotators.Annotator] = dict.fromkeys(self.cfg.data_types, None)
        # -- iterate over each data type
        for name in self.cfg.data_types:
            # init params -- Checked from rep.scripts.writes_default.basic_writer.py
            # note: we are verbose here to make it easier to understand the code.
            #   if colorize is true, the data is mapped to colors and a uint8 4 channel image is returned.
            #   if colorize is false, the data is returned as a uint32 image with ids as values.
            if name in ["bounding_box_2d_tight", "bounding_box_2d_loose", "bounding_box_3d"]:
                init_params = {"semanticTypes": self.cfg.semantic_types}
            elif name in ["semantic_segmentation", "instance_segmentation"]:
                init_params = {"semanticTypes": self.cfg.semantic_types, "colorize": False}
            elif name in ["instance_id_segmentation"]:
                init_params = {"colorize": False}
            else:
                init_params = None
            # create annotator node
            rep_annotator = rep.AnnotatorRegistry.get_annotator(name, init_params, device=self.device)
            rep_annotator.attach([self._render_product_path])
            # add to registry
            self._rep_registry[name] = rep_annotator
        # Reset internal buffers
        self.reset()
        # When running in standalone mode, need to render a few times to fill all the buffers
        # FIXME: Check with simulation team to get rid of this. What if someone has render or other callbacks?
        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
            for _ in range(4):
                self._sim_context.render()

    def reset(self):
        # reset the timestamp
        super().reset()
        # reset the buffer
        self._data.position = None
        self._data.orientation = None
        self._data.intrinsic_matrix = self._compute_intrinsic_matrix()
        self._data.image_shape = self.image_shape
        self._data.output = dict.fromkeys(self._data.output, None)

    def buffer(self):
        """Updates the internal buffer with the latest data from the sensor.

        This function reads the intrinsic matrix and pose of the camera. It also reads the data from
        the annotator registry and updates the internal buffer.

        Note:
            When running in standalone mode, the function renders the scene a few times to fill all the buffers.
            During this time, the physics simulation is paused. This is a known issue with Isaac Sim.
        """
        # When running in standalone mode, need to render a few times to fill all the buffers
        # FIXME: Check with simulation team to get rid of this. What if someone has render or other callbacks?
        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
            for _ in range(4):
                self._sim_context.render()
        # -- intrinsic matrix
        self._data.intrinsic_matrix = self._compute_intrinsic_matrix()
        # -- pose
        self._data.position, self._data.orientation = self._compute_ros_pose()
        # -- read the data from annotator registry
        for name in self._rep_registry:
            self._data.output[name] = self._rep_registry[name].get_data()
        # -- update the trigger call data (needed by replicator BasicWriter method)
        self._data.output["trigger_outputs"] = {"on_time": self.frame}

    """
    Private Helpers
    """

    def _compute_intrinsic_matrix(self) -> np.ndarray:
        """Compute camera's matrix of intrinsic parameters.

        Also called calibration matrix. This matrix works for linear depth images. We assume square pixels.

        Note:
            The calibration matrix projects points in the 3D scene onto an imaginary screen of the camera.
            The coordinates of points on the image plane are in the homogeneous representation.

        Returns:
            np.ndarray: A 3 x 3 numpy array containing the intrinsic parameters for the camera.

        Raises:
            RuntimeError: If the camera prim is not set. Need to call `initialize(...)` first.
        """
        # check camera prim exists
        if self._sensor_prim is None:
            raise RuntimeError("Camera prim is None. Please call 'initialize(...)' first.")
        # get camera parameters
        focal_length = self._sensor_prim.GetFocalLengthAttr().Get()
        horiz_aperture = self._sensor_prim.GetHorizontalApertureAttr().Get()
        # get viewport parameters
        height, width = self.image_shape
        # calculate the field of view
        fov = 2 * math.atan(horiz_aperture / (2 * focal_length))
        # calculate the focal length in pixels
        focal_px = width * 0.5 / math.tan(fov / 2)
        # create intrinsic matrix for depth linear
        a = focal_px
        b = width * 0.5
        c = focal_px
        d = height * 0.5
        # return the matrix
        return np.array([[a, 0, b], [0, c, d], [0, 0, 1]], dtype=float)

    def _compute_ros_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the pose of the camera in the world frame with ROS convention.

        This methods uses the ROS convention to resolve the input pose. In this convention,
        we assume that the camera front-axis is +Z-axis and up-axis is -Y-axis.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple of the position (in meters) and quaternion (w, x, y, z).
        """
        # get camera's location in world space
        prim_tf = UsdGeom.Xformable(self._sensor_prim).ComputeLocalToWorldTransform(0.0)
        # GfVec datatypes are row vectors that post-multiply matrices to effect transformations,
        # which implies, for example, that it is the fourth row of a GfMatrix4d that specifies
        # the translation of the transformation. Thus, we take transpose here to make it post multiply.
        prim_tf = np.transpose(prim_tf)
        # extract the position (convert it to SI units-- assumed that stage units is 1.0)
        pos = prim_tf[0:3, 3]
        # extract rotation
        # Note: OpenGL camera transform is such that camera faces along -z axis and +y is up.
        #   In robotics, we need to rotate it such that the camera is along +z axis and -y is up.
        cam_rotm = prim_tf[0:3, 0:3]
        # make +z forward
        cam_rotm[:, 2] = -cam_rotm[:, 2]
        # make +y down
        cam_rotm[:, 1] = -cam_rotm[:, 1]
        # convert rotation to quaternion
        quat = tf.Rotation.from_matrix(cam_rotm).as_quat()

        return pos, convert_quat(quat, "wxyz")
