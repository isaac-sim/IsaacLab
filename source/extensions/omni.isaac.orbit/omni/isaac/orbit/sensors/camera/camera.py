# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera class in Omniverse workflows."""


import builtins
import math
import numpy as np
import torch
from tensordict import TensorDict
import scipy.spatial.transform as tf
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple, Union, List, Optional, Iterable

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
import omni.kit.commands
import omni.replicator.core as rep
import omni.usd
from omni.isaac.core.prims import XFormPrimView
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.rotations import gf_quat_to_np_array
from pxr import Gf, Sdf, Usd, UsdGeom

# omni-isaac-orbit
from omni.isaac.orbit.utils import class_to_dict, to_camel_case
from omni.isaac.orbit.utils.array import convert_to_torch, TensorData
from omni.isaac.orbit.utils.math import convert_quat

from ..sensor_base import SensorBase
from .camera_cfg import FisheyeCameraCfg, PinholeCameraCfg

__all__ = ["Camera", "CameraData"]


@dataclass
class CameraData:
    """Data container for the camera sensor."""

    position: TensorData = None
    """Position of the sensor origin in world frame, following ROS convention. Shape: (N, 3)."""
    orientation: TensorData = None
    """Quaternion orientation `(w, x, y, z)` of the sensor origin in world frame, following ROS convention. Shape: (N, 4)."""
    intrinsic_matrices: TensorData = None
    """The intrinsic matrices for the camera. Shape: (N, 3, 3)."""
    image_shape: Tuple[int, int] = None
    """A tuple containing (height, width) of the camera sensor."""
    output: Union[Dict[str, np.ndarray], TensorDict] = None
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

    def __init__(self, cfg: Union[PinholeCameraCfg, FisheyeCameraCfg]):
        """Initializes the camera sensor.

        Args:
            cfg (Union[PinholeCameraCfg, FisheyeCameraCfg]): The configuration parameters.
        """
        # store inputs
        self.cfg = cfg
        # initialize base class
        super().__init__(self.cfg.sensor_tick)
        # change the default rendering settings
        # TODO: Should this be done here or maybe inside the app config file?
        rep.settings.set_render_rtx_realtime(antialiasing="FXAA")

        # Xform prim for handling transforms
        self.xforms: XFormPrimView = None
        # UsdGeom Camera prim for the sensor
        self._sensor_prims: List[UsdGeom.Camera] = list()
        # Create empty variables for storing output data
        self._data = CameraData()
        # Flag to check that sensor is spawned.
        self._is_spawned = False

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        # message for class
        return (
            f"Camera @ '{self._view._regex_prim_paths}': \n"
            f"\tdata types   : {list(self._data.output.keys())} \n"
            f"\ttick rate (s): {self.sensor_tick}\n"
            f"\tshape        : {self.image_shape}\n"
        )

    """
    Properties
    """

    @property
    def render_product_paths(self) -> List[str]:
        """The path of the render products for the cameras.

        This can be used via replicator interfaces to attach to writes or external annotator registry.
        """
        return self._render_product_paths

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

    def set_intrinsic_matrix(self, matrices: TensorData, focal_length: float = 1.0, indices: Optional[Sequence[int]] = None):
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
            matrices (TensorData): The intrinsic matrices for the camera. Shape: :math:`(N, 3, 3)`.
            focal_length (float, optional): Focal length to use when computing aperture values. Defaults to 1.0.
            indices (Sequence[int], optional): A list of indices of length :obj:`N` to specify the prims to manipulate.
                Defaults to None, which means all prims will be manipulated.
        """
        # resolve indices
        # check camera prim exists
        if not self._is_initialized:
            raise RuntimeError("Camera is not initialized. Please call 'initialize(...)' first.")
        # resolve indices
        indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
        # iterate over indices
        for i, matrix in zip(indices, matrices):
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
            # change data for corresponding camera index
            sensor_prim = self._sensor_prims[i]
            # set parameters for camera
            for param_name, param_value in params.items():
                # convert to camel case (CC)
                param_name = to_camel_case(param_name, to="CC")
                # get attribute from the class
                param_attr = getattr(sensor_prim, f"Get{param_name}Attr")
                # set value
                # note: We have to do it this way because the camera might be on a different layer (default cameras are on session layer),
                #   and this is the simplest way to set the property on the right layer.
                omni.usd.utils.set_prop_val(param_attr(), param_value)

    """
    Operations - Set pose.
    """

    def set_world_poses_ros(self, positions: TensorData = None, orientations: TensorData = None, indices: Optional[Sequence[int]] = None):
        r"""Set the pose of the camera w.r.t. world frame using ROS convention.

        In USD, the camera is always in **Y up** convention. This means that the camera is looking down the -Z axis
        with the +Y axis pointing up , and +X axis pointing right. However, in ROS, the camera is looking down
        the +Z axis with the +Y axis pointing down, and +X axis pointing right. Thus, the camera needs to be rotated
        by :math:`180^{\circ}` around the X axis to follow the ROS convention.

        .. math::

            T_{ROS} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & -1 & 0 & 0 \\ 0 & 0 & -1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix} T_{USD}

        Args:
            positions (TensorData, optional): The cartesian coordinates (in meters). Shape: :math:`(N, 3)`. Defaults to None.
            orientations (TensorData, optional): The quaternion orientation in (w, x, y, z). Shape: :math:`(N, 4)`. Defaults to None.
            indices (Sequence[int], optional): A list of indices of length :obj:`N` to specify the prims to manipulate.
                Defaults to None, which means all prims will be manipulated.

        Raises:
            RuntimeError: If the camera prim is not set. Need to call :meth:`initialize` method first.
        """
        # check camera prim exists
        if not self._is_initialized:
            raise RuntimeError("Camera is not initialized. Please call 'initialize(...)' first.")
        # resolve indices
        indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
        # convert to backend tensor
        if positions is not None:
            positions = self._backend_utils.convert(positions, self._device)
        # convert rotation matrix from ROS convention to OpenGL
        if orientations is not None:
            # TODO: Make this more efficient
            for index, quat in enumerate(orientations):
                rotm = tf.Rotation.from_quat(convert_quat(quat, "xyzw")).as_matrix()
                rotm[:, 2] = -rotm[:, 2]
                rotm[:, 1] = -rotm[:, 1]
                # convert to isaac-sim convention
                quat_gl = tf.Rotation.from_matrix(rotm).as_quat()
                orientations[index] = convert_quat(quat_gl, "wxyz")
            # convert to backend tensor
            orientations = self._backend_utils.convert(orientations, self._device)
        else:
            orientations = None
        # set the pose
        self._view.set_world_poses(positions, orientations, indices)

    def set_world_poses_from_view(self, eyes: TensorData, targets: TensorData, indices: Optional[Sequence[int]] = None):
        """Set the poses of the camera from the eye position and look-at target position.

        Args:
            eyes (TensorData): The positions of the camera's eye. Shape: :math:`(N, 3)`.
            targets (TensorData, optional): The target locations to look at. Shape: :math:`(N, 3)`.
            indices (Sequence[int], optional): A list of indices of length :obj:`N` to specify the prims to manipulate.
                Defaults to None, which means all prims will be manipulated.

        Raises:
            RuntimeError: If the camera prim is not set. Need to call :meth:`initialize` method first.
            NotImplementedError: If the stage up-axis is not "Y" or "Z".
        """
        # check camera prim exists
        if not self._is_initialized:
            raise RuntimeError("Camera is not initialized. Please call 'initialize(...)' first.")
        # resolve indices
        indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
        # create tensors for storing poses
        positions = self._backend_utils.create_zeros_tensor((len(indices), 3), "float32", self._device)
        orientations = self._backend_utils.create_zeros_tensor((len(indices), 4), "float32", self._device)
        # check if targets are provided
        if targets is None:
            targets = np.zeros_like(eyes)
        # iterate over all indices
        # TODO: Can we do this in parallel?
        for i, eye, target in zip(indices, eyes, targets):
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
            positions[i] = self._backend_utils.convert(np.asarray(matrix_gf.ExtractTranslation()), self._device)
            orientations[i] = self._backend_utils.convert(gf_quat_to_np_array(matrix_gf.ExtractRotationQuat()), self._device)
        # set camera poses using the view
        self._view.set_world_poses(positions, orientations, indices)

    """
    Operations
    """

    def spawn(self, prim_path: str, translation: Sequence[float] = None, orientation: Sequence[float] = None):
        """Spawns the sensor into the stage.

        The USD Camera prim is spawned under the parent prim at the path ``parent_prim_path`` with the provided input
        translation and orientation.

        Args:
            prim_path (str): The path of the prim to attach sensor to.
            translation (Sequence[float], optional): The local position offset w.r.t. parent prim. Defaults to None.
            orientation (Sequence[float], optional): The local rotation offset in ``(w, x, y, z)`` w.r.t.
                parent prim. Defaults to None.

        Raises:
            RuntimeError: If a prim already exists at the path.
        """
        # Create sensor prim
        if not prim_utils.is_prim_path_valid(prim_path):
            prim_utils.create_prim(prim_path, "Camera", translation=translation, orientation=orientation)
        else:
            raise RuntimeError(f"Unable to spawn camera. A prim already exists at path '{prim_path}'.")
        # Add replicator camera attributes
        self._define_usd_camera_attributes(prim_path)
        # Save prim path for later use
        self._spawn_prim_path = prim_path
        # Set spawning to true
        self._is_spawned = True

    def initialize(self, prim_paths_expr: str = None):
        """Initializes the sensor handles and internal buffers.

        This function creates handles and registers the provided data types with the replicator registry to
        be able to access the data from the sensor. It also initializes the internal buffers to store the data.

        The function also allows initializing to a camera not spawned by using the :meth:`spawn` method.
        For instance, cameras that already exist in the USD stage. In such cases, the USD settings present on
        the camera prim is used instead of the settings passed in the configuration object.

        Args:
            prim_paths_expr (str, optional): The prim path expression to cameras. Defaults to None.

        Raises:
            RuntimeError: When input `cam_prim_path` is :obj:`None`, the method defaults to using the last
                camera prim path set when calling the :meth:`spawn` function. In case, the camera was not spawned
                and no valid `cam_prim_path` is provided, the function throws an error.
        """
        # Check that sensor has been spawned
        if prim_paths_expr is None:
            if not self._is_spawned:
                raise RuntimeError("Initialize the camera failed! Please provide a valid argument for `prim_paths_expr`.")
            else:
                prim_paths_expr = self._spawn_prim_path
        # Initialize parent class
        super().initialize(prim_paths_expr)

        # Attach the sensor data types to render node
        self._render_product_paths: List[str] = list()
        self._rep_registry: Dict[str, List[rep.annotators.Annotator]] = {name: list() for name in self.cfg.data_types}
        # Resolve device name
        if self._backend != "numpy":
            device_name = self._device.split(":")[0]
        else:
            device_name = "cpu"
        # Convert all encapsulated prims to Camera
        for cam_prim_path in self._view.prim_paths:
            # Get camera prim
            cam_prim = prim_utils.get_prim_at_path(cam_prim_path)
            # Check if prim is a camera
            if not cam_prim.IsA(UsdGeom.Camera):
                raise RuntimeError(f"Prim at path '{cam_prim_path}' is not a Camera.")
            else:
                sensor_prim = UsdGeom.Camera(cam_prim)
                self._sensor_prims.append(sensor_prim)
            # Get render product
            render_prod_path = rep.create.render_product(
                cam_prim_path, resolution=(self.cfg.width, self.cfg.height)
            )
            self._render_product_paths.append(render_prod_path)
            # Iterate over each data type and create annotator
            # TODO: This will move out of the loop once Replicator supports multiple render products within a single
            #  annotator, i.e.: rep_annotator.attach(self._render_product_paths)
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
                rep_annotator = rep.AnnotatorRegistry.get_annotator(name, init_params, device=device_name)
                rep_annotator.attach(render_prod_path)
                # add to registry
                self._rep_registry[name].append(rep_annotator)
        # Create internal buffers
        self._create_buffers()
        # When running in standalone mode, need to render a few times to fill all the buffers
        # FIXME: Check with simulation team to get rid of this. What if someone has render or other callbacks?
        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
            # acquire simulation context
            sim = SimulationContext.instance()
            # render a few times
            if sim is not None:
                for _ in range(4):
                    sim.render()

    def reset_buffers(self, sensor_ids: Optional[Sequence[int]] = None):
        # reset the timestamps
        super().reset_buffers(sensor_ids)
        # reset the data
        # note: this recomputation is useful if one performs randomization on the camera poses.
        self._update_ros_poses(sensor_ids)
        self._update_intrinsic_matrices(sensor_ids)

    def buffer(self, sensor_ids: Optional[Sequence[int]] = None):
        # check camera prim exists
        if not self._is_initialized:
            raise RuntimeError("Camera is not initialized. Please call 'initialize(...)' first.")
        # Resolve sensor ids
        if sensor_ids is None:
            sensor_ids = self._ALL_INDICES
        # -- intrinsic matrix
        self._update_intrinsic_matrices(sensor_ids)
        # -- pose
        self._update_ros_poses(sensor_ids)
        # -- read the data from annotator registry
        # check if buffer is called for the first time. If so then, allocate the memory
        if self._data.output is None:
            # this is the first time buffer is called
            # it allocates memory for all the sensors
            self._create_annotator_data()
        else:
            # iterate over all the data types
            for name, annotators in self._rep_registry.items():
                # iterate over all the annotators
                for index in sensor_ids:
                    # get the data
                    data = annotators[index].get_data()
                    # convert data to torch tensor
                    if self._backend == "torch":
                        data = convert_to_torch(data, device=self.device)
                    # add data to output
                    self._data.output[name][index] = data

    """
    Private Helpers
    """

    def _create_buffers(self):
        """Create buffers for storing data."""
        # create the data object
        # -- pose of the cameras
        self._data.position = self._backend_utils.create_zeros_tensor((self.count, 3), dtype="float32", device=self.device)
        self._data.orientation = self._backend_utils.create_zeros_tensor((self.count, 4), dtype="float32", device=self.device)
        # -- intrinsic matrix
        self._data.intrinsic_matrices = self._backend_utils.create_zeros_tensor((self.count, 3, 3), dtype="float32", device=self.device)
        self._data.image_shape = self.image_shape
        # -- output data
        # since the size of the output data is not known in advance, we leave it as None
        # the memory will be allocated when the buffer() function is called for the first time.
        self._data.output = None

    def _define_usd_camera_attributes(self, prim_path: str):
        """Creates and sets USD camera attributes.

        This function creates additional attributes on the camera prim used by Replicator.
        It also sets the default values for these attributes based on the camera configuration.

        Args:
            prim_path (str): The prim path to the camera.
        """
        # lock camera from viewport (this disables viewport movement for camera)
        kwargs = {
            "prop_path": Sdf.Path(f"{prim_path}.omni:kit:cameraLock"),
            "value": True,
            "prev": None,
            "type_to_create_if_not_exist": Sdf.ValueTypeNames.Bool,
        }
        omni.kit.commands.execute("ChangePropertyCommand", **kwargs)

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
        prim = prim_utils.get_prim_at_path(prim_path)
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

    def _update_intrinsic_matrices(self, sensor_ids: Iterable[int]):
        """Compute camera's matrix of intrinsic parameters.

        Also called calibration matrix. This matrix works for linear depth images. We assume square pixels.

        Note:
            The calibration matrix projects points in the 3D scene onto an imaginary screen of the camera.
            The coordinates of points on the image plane are in the homogeneous representation.
        """
        # iterate over all cameras
        for i in sensor_ids:
            # Get corresponding sensor prim
            sensor_prim = self._sensor_prims[i]
            # get camera parameters
            focal_length = sensor_prim.GetFocalLengthAttr().Get()
            horiz_aperture = sensor_prim.GetHorizontalApertureAttr().Get()
            # get viewport parameters
            height, width = self.image_shape
            # calculate the field of view
            fov = 2 * math.atan(horiz_aperture / (2 * focal_length))
            # calculate the focal length in pixels
            focal_px = width * 0.5 / math.tan(fov / 2)
            # create intrinsic matrix for depth linear
            self._data.intrinsic_matrices[i, 0, 0] = focal_px
            self._data.intrinsic_matrices[i, 0, 2] = width * 0.5
            self._data.intrinsic_matrices[i, 1, 1] = focal_px
            self._data.intrinsic_matrices[i, 1, 2] = height * 0.5
            self._data.intrinsic_matrices[i, 2, 2] = 1

    def _update_ros_poses(self, sensor_ids: Iterable[int]):
        """Computes the pose of the camera in the world frame with ROS convention.

        This methods uses the ROS convention to resolve the input pose. In this convention,
        we assume that the camera front-axis is +Z-axis and up-axis is -Y-axis.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple of the position (in meters) and quaternion (w, x, y, z).
        """
        # check camera prim exists
        if len(self._sensor_prims) == 0:
            raise RuntimeError("Camera prim is None. Please call 'initialize(...)' first.")
        # iterate over all cameras
        for i in sensor_ids:
            # obtain corresponding sensor prim
            sensor_prim = self._sensor_prims[i]
            # get camera's location in world space
            prim_tf = sensor_prim.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            # GfVec datatypes are row vectors that post-multiply matrices to effect transformations,
            # which implies, for example, that it is the fourth row of a GfMatrix4d that specifies
            # the translation of the transformation. Thus, we take transpose here to make it post multiply.
            prim_tf = np.transpose(prim_tf)
            # extract the position (convert it to SI units-- assumed that stage units is 1.0)
            self._data.position[i] = self._backend_utils.convert(prim_tf[0:3, 3], device=self._device)
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
            self._data.orientation[i] = self._backend_utils.convert(convert_quat(quat, "wxyz"), device=self._device)

    def _create_annotator_data(self):
        """Create the buffers to store the annotator data.

        We create a buffer for each annotator and store the data in a dictionary. Since the data
        shape is not known beforehand, we create a list of buffers and concatenate them later.

        This is an expensive operation and should be called only once.
        """
        # lazy allocation of data dictionary
        if self._backend == "numpy":
            self._data.output = {name: None for name in self.cfg.data_types}
        elif self._backend == "torch":
            self._data.output = TensorDict({}, batch_size=self.count, device=self.device)
        else:
            raise ValueError(f"Unknown backend: {self._backend}. Supported backends: ['numpy', 'torch']")
        # add data from the annotators
        for name, annotators in self._rep_registry.items():
            # create a list to store the data for each annotator
            data_all_cameras = list()
            # iterate over all the annotators
            for index in self._ALL_INDICES:
                # get the data
                data = annotators[index].get_data()
                # convert data to torch tensor
                if self._backend == "torch":
                    data = convert_to_torch(data, device=self.device)
                # append the data
                data_all_cameras.append(data)
            # concatenate the data along the batch dimension
            if self._backend == "numpy":
                self._data.output[name] = np.stack(data_all_cameras, axis=0)
            elif self._backend == "torch":
                self._data.output[name] = torch.stack(data_all_cameras, dim=0)
            else:
                raise ValueError(f"Unsupported backend: {self._backend}")
