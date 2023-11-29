# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera class in Omniverse workflows."""

from __future__ import annotations

import builtins
import math
import numpy as np
import scipy.spatial.transform as tf
from dataclasses import dataclass
from typing import Any, Sequence

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
import omni.replicator.core as rep
import omni.usd
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.rotations import gf_quat_to_np_array
from pxr import Gf, Sdf, Usd, UsdGeom

from omni.isaac.orbit.utils import class_to_dict, to_camel_case
from omni.isaac.orbit.utils.math import convert_quat

from ..sensor_base import SensorBase
from .camera_cfg import FisheyeCameraCfg, PinholeCameraCfg


@dataclass
class CameraData:
    """Data container for the camera sensor."""

    position: np.ndarray = None
    """Position of the sensor origin in world frame, following ROS convention."""
    orientation: np.ndarray = None
    """Quaternion orientation `(w, x, y, z)` of the sensor origin in world frame, following ROS convention."""
    intrinsic_matrix: np.ndarray = None
    """The intrinsic matrix for the camera."""
    image_shape: tuple[int, int] = None
    """A tuple containing (height, width) of the camera sensor."""
    output: dict[str, Any] = None
    """The retrieved sensor data with sensor types as key.

    The format of the data is available in the `Replicator Documentation`_.

    .. _Replicator Documentation: https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_replicator/annotators_details.html#annotator-output
    """


class Camera(SensorBase):
    r"""The camera sensor for acquiring visual data.

    This class wraps over the `UsdGeom Camera`_ for providing a consistent API for acquiring visual data.
    It ensures that the camera follows the ROS convention for the coordinate system.

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

    .. _replicator extension: https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_replicator/annotators_details.html#annotator-output
    .. _USDGeom Camera: https://graphics.pixar.com/usd/docs/api/class_usd_geom_camera.html

    """

    def __init__(self, cfg: PinholeCameraCfg | FisheyeCameraCfg, device: str = "cpu"):
        """Initializes the camera sensor.

        If the ``device`` is ``"cpu"``, the output data is returned as a numpy array. If the ``device`` is
        ``"cuda"``, then a Warp array is returned. Note that only the valid sensor types will be moved to GPU.

        Args:
            cfg: The configuration parameters.
            device: The device on which to receive data. Defaults to "cpu".
        """
        # store inputs
        self.cfg = cfg
        self.device = device
        # initialize base class
        super().__init__(self.cfg.sensor_tick)
        # change the default rendering settings
        # TODO: Should this be done here or maybe inside the app config file?
        rep.settings.set_render_rtx_realtime(antialiasing="FXAA")

        # Xform prim for handling transforms
        self._sensor_xform: XFormPrim = None
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
    def image_shape(self) -> tuple[int, int]:
        """A tuple containing (height, width) of the camera sensor."""
        return (self.cfg.height, self.cfg.width)

    """
    Configuration
    """

    def set_visibility(self, visible: bool):
        """Set visibility of the instance in the scene.

        Args:
            visible: Whether to make instance visible or invisible.

        Raises:
            RuntimeError: If the camera prim is not set. Need to call :meth:`initialize` first.
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

        The intrinsic matrix and focal length are used to set the following parameters to the USD camera:

        - ``focal_length``: The focal length of the camera.
        - ``horizontal_aperture``: The horizontal aperture of the camera.
        - ``vertical_aperture``: The vertical aperture of the camera.
        - ``horizontal_aperture_offset``: The horizontal offset of the camera.
        - ``vertical_aperture_offset``: The vertical offset of the camera.

        .. warning::

            Due to limitations of Omniverse camera, we need to assume that the camera is a spherical lens,
            i.e. has square pixels, and the optical center is centered at the camera eye. If this assumption
            is not true in the input intrinsic matrix, then the camera will not set up correctly.

        Args:
            intrinsic_matrix: The intrinsic matrix for the camera.
            focal_length: Focal length to use when computing aperture values. Defaults to 1.0.
        """
        # convert to numpy for sanity
        intrinsic_matrix = np.asarray(matrix, dtype=float)
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
        r"""Set the pose of the camera w.r.t. world frame using ROS convention.

        In USD, the camera is always in **Y up** convention. This means that the camera is looking down the -Z axis
        with the +Y axis pointing up , and +X axis pointing right. However, in ROS, the camera is looking down
        the +Z axis with the +Y axis pointing down, and +X axis pointing right. Thus, the camera needs to be rotated
        by :math:`180^{\circ}` around the X axis to follow the ROS convention.

        .. math::

            T_{ROS} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & -1 & 0 & 0 \\ 0 & 0 & -1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix} T_{USD}

        Args:
            pos: The cartesian coordinates (in meters). Defaults to None.
            quat: The quaternion orientation in (w, x, y, z). Defaults to None.

        Raises:
            RuntimeError: If the camera prim is not set. Need to call :meth:`initialize` method first.
        """
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
            # convert to isaac-sim convention
            quat_gl = tf.Rotation.from_matrix(rotm).as_quat()
            quat_gl = convert_quat(quat_gl, "wxyz")
        else:
            quat_gl = None
        # set the pose
        self._sensor_xform.set_world_pose(pos, quat_gl)

    def set_world_pose_from_view(self, eye: Sequence[float], target: Sequence[float] = [0, 0, 0]):
        """Set the pose of the camera from the eye position and look-at target position.

        Args:
            eye: The position of the camera's eye.
            target: The target location to look at. Defaults to [0, 0, 0].

        Raises:
            RuntimeError: If the camera prim is not set. Need to call :meth:`initialize` method first.
            NotImplementedError: If the stage up-axis is not "Y" or "Z".
        """
        # check camera prim exists
        if self._sensor_prim is None:
            raise RuntimeError("Camera prim is None. Please call 'initialize(...)' first.")
        # compute camera's eye pose
        eye_position = Gf.Vec3d(np.asarray(eye).tolist())
        target_position = Gf.Vec3d(np.asarray(target).tolist())
        # compute forward direction
        forward_dir = (eye_position - target_position).GetNormalized()
        # get up axis
        up_axis_token = stage_utils.get_stage_up_axis()
        if up_axis_token == UsdGeom.Tokens.y:
            # deal with degenerate case
            if forward_dir == Gf.Vec3d(0, 1, 0):
                up_axis = Gf.Vec3d(0, 0, 1)
            elif forward_dir == Gf.Vec3d(0, -1, 0):
                up_axis = Gf.Vec3d(0, 0, -1)
            else:
                up_axis = Gf.Vec3d(0, 1, 0)
        elif up_axis_token == UsdGeom.Tokens.z:
            # deal with degenerate case
            if forward_dir == Gf.Vec3d(0, 0, 1):
                up_axis = Gf.Vec3d(0, 1, 0)
            elif forward_dir == Gf.Vec3d(0, 0, -1):
                up_axis = Gf.Vec3d(0, -1, 0)
            else:
                up_axis = Gf.Vec3d(0, 0, 1)
        else:
            raise NotImplementedError(f"This method is not supported for up-axis '{up_axis_token}'.")
        # compute matrix transformation
        # view matrix: camera_T_world
        matrix_gf = Gf.Matrix4d(1).SetLookAt(eye_position, target_position, up_axis)
        # camera position and rotation in world frame
        matrix_gf = matrix_gf.GetInverse()
        cam_pos = np.array(matrix_gf.ExtractTranslation())
        cam_quat = gf_quat_to_np_array(matrix_gf.ExtractRotationQuat())
        # set camera pose
        self._sensor_xform.set_world_pose(cam_pos, cam_quat)

    """
    Operations
    """

    def spawn(self, parent_prim_path: str, translation: Sequence[float] = None, orientation: Sequence[float] = None):
        """Spawns the sensor into the stage.

        The USD Camera prim is spawned under the parent prim at the path ``parent_prim_path`` with the provided input
        translation and orientation.

        Args:
            parent_prim_path: The path of the parent prim to attach sensor to.
            translation: The local position offset w.r.t. parent prim. Defaults to None.
            orientation: The local rotation offset in (w, x, y, z) w.r.t.
                parent prim. Defaults to None.
        """
        # Check if sensor is already spawned
        if self._is_spawned:
            raise RuntimeError(f"The camera sensor instance has already been spawned at: {self.prim_path}.")
        # Create sensor prim path
        prim_path = stage_utils.get_next_free_path(f"{parent_prim_path}/Camera")
        # Create sensor prim
        self._sensor_prim = UsdGeom.Camera(prim_utils.define_prim(prim_path, "Camera"))
        # Add replicator camera attributes
        self._define_usd_camera_attributes()
        # Set the transformation of the camera
        self._sensor_xform = XFormPrim(self.prim_path)
        self._sensor_xform.set_local_pose(translation, orientation)
        # Set spawning to true
        self._is_spawned = True

    def initialize(self, cam_prim_path: str = None):
        """Initializes the sensor handles and internal buffers.

        This function creates handles and registers the provided data types with the replicator registry to
        be able to access the data from the sensor. It also initializes the internal buffers to store the data.

        The function also allows initializing to a camera not spawned by using the :meth:`spawn` method.
        For instance, cameras that already exist in the USD stage. In such cases, the USD settings present on
        the camera prim is used instead of the settings passed in the configuration object.

        Args:
            cam_prim_path: The prim path to existing camera. Defaults to None.
            has_rig: Whether the passed camera prim path is attached to a rig. Defaults to False.

        Raises:
            RuntimeError: When input `cam_prim_path` is None, the method defaults to using the last
                camera prim path set when calling the :meth:`spawn` function. In case, the camera was not spawned
                and no valid `cam_prim_path` is provided, the function throws an error.
        """
        # Check that sensor has been spawned
        if cam_prim_path is None:
            if not self._is_spawned:
                raise RuntimeError("Initialize the camera failed! Please provide a valid argument for `prim_path`.")
        else:
            # Get prim at path
            cam_prim = prim_utils.get_prim_at_path(cam_prim_path)
            # Check if prim is valid
            if cam_prim.IsValid() is False:
                raise RuntimeError(f"Initialize the camera failed! Invalid prim path: {cam_prim_path}.")
            # Force to set active camera to input prim path
            self._sensor_prim = UsdGeom.Camera(cam_prim)
            self._sensor_xform = XFormPrim(cam_prim_path)

        # Enable synthetic data sensors
        self._render_product_path = rep.create.render_product(
            self.prim_path, resolution=(self.cfg.width, self.cfg.height)
        )
        # Attach the sensor data types to render node
        self._rep_registry: dict[str, rep.annotators.Annotator] = dict.fromkeys(self.cfg.data_types, None)
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
            # get simulation context
            sim_context = SimulationContext.instance()
            # render a few times
            for _ in range(4):
                sim_context.render()

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

    def _define_usd_camera_attributes(self):
        """Creates and sets USD camera attributes.

        This function creates additional attributes on the camera prim used by Replicator.
        It also sets the default values for these attributes based on the camera configuration.
        """
        # camera attributes
        # reference: omni.replicator.core.scripts.create.py: camera()
        attribute_types = {
            "cameraProjectionType": "token",
            "fthetaWidth": "float",
            "fthetaHeight": "float",
            "fthetaCx": "float",
            "fthetaCy": "float",
            "fthetaMaxFov": "float",
            "fthetaPolyA": "float",
            "fthetaPolyB": "float",
            "fthetaPolyC": "float",
            "fthetaPolyD": "float",
            "fthetaPolyE": "float",
        }
        # get camera prim
        prim = prim_utils.get_prim_at_path(self.prim_path)
        # create attributes
        for attr_name, attr_type in attribute_types.items():
            # check if attribute does not exist
            if prim.GetAttribute(attr_name).Get() is None:
                # create attribute based on type
                if attr_type == "token":
                    prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Token)
                elif attr_type == "float":
                    prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Float)
        # set attribute values
        # -- projection type
        projection_type = to_camel_case(self.cfg.projection_type, to="cC")
        prim.GetAttribute("cameraProjectionType").Set(projection_type)
        # -- other attributes
        for param_name, param_value in class_to_dict(self.cfg.usd_params).items():
            # check if value is valid
            if param_value is None:
                continue
            # convert to camel case (CC)
            param = to_camel_case(param_name, to="cC")
            # get attribute from the class
            prim.GetAttribute(param).Set(param_value)

    def _compute_intrinsic_matrix(self) -> np.ndarray:
        """Compute camera's matrix of intrinsic parameters.

        Also called calibration matrix. This matrix works for linear depth images. We assume square pixels.

        Note:
            The calibration matrix projects points in the 3D scene onto an imaginary screen of the camera.
            The coordinates of points on the image plane are in the homogeneous representation.

        Returns:
            A 3 x 3 numpy array containing the intrinsic parameters for the camera.

        Raises:
            RuntimeError: If the camera prim is not set. Need to call :meth:`initialize` first.
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

    def _compute_ros_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Computes the pose of the camera in the world frame with ROS convention.

        This methods uses the ROS convention to resolve the input pose. In this convention,
        we assume that the camera front-axis is +Z-axis and up-axis is -Y-axis.

        Returns:
            A tuple of the position (in meters) and quaternion (w, x, y, z).
        """
        # get camera's location in world space
        prim_tf = self._sensor_prim.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
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
