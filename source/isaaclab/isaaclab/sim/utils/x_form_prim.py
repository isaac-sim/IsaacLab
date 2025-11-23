# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import logging
import numpy as np
import torch
from scipy.spatial.transform import Rotation

import usdrt
from isaacsim.core.utils.prims import get_prim_parent
from pxr import Gf, Usd, UsdGeom

import isaaclab.sim.utils.prims as prim_utils
from isaaclab.sim import SimulationContext

# import logger
logger = logging.getLogger(__name__)

# NOTE: currently removed all callbacks, especially the reset callback (catches reset event and sets position and
# orientation to defaiult -- do we need that?)


class XFormPrim:
    """Provides high level functions to deal with a Xform prim view (one or many) and its descendants
    as well as its attributes/properties.

    This class wraps all matching Xforms found at the regex provided at the ``prim_paths_expr`` argument

    .. note::

        Each prim will have ``xformOp:orient``, ``xformOp:translate`` and ``xformOp:scale`` only post-init,
        unless it is a non-root articulation link.

    Args:
        prim_paths_expr: prim paths regex to encapsulate all prims that match it.
                                example: "/World/Env[1-5]/Franka" will match /World/Env1/Franka,
                                /World/Env2/Franka..etc.
                                (a non regex prim path can also be used to encapsulate one Xform). Additionally a
                                list of regex can be provided. example ["/World/Env[1-5]/Franka", "/World/Env[10-19]/Franka"].
        positions (Optional[Union[np.ndarray, torch.Tensor]], optional):
                                                        default positions in the world frame of the prim.
                                                        shape is (N, 3).
                                                        Defaults to None, which means left unchanged.
        translations (Optional[Union[np.ndarray, torch.Tensor]], optional):
                                                        default translations in the local frame of the prims
                                                        (with respect to its parent prims). shape is (N, 3).
                                                        Defaults to None, which means left unchanged.
        orientations (Optional[Union[np.ndarray, torch.Tensor]], optional):
                                                        default quaternion orientations in the world/ local frame of the prim
                                                        (depends if translation or position is specified).
                                                        quaternion is scalar-first (w, x, y, z). shape is (N, 4).
                                                        Defaults to None, which means left unchanged.
        scales (Optional[Union[np.ndarray, torch.Tensor]], optional): local scales to be applied to
                                                        the prim's dimensions. shape is (N, 3).
                                                        Defaults to None, which means left unchanged.
        visibilities (Optional[Union[np.ndarray, torch.Tensor]], optional): set to false for an invisible prim in
                                                                            the stage while rendering. shape is (N,).
                                                                            Defaults to None.
        reset_xform_properties (bool, optional): True if the prims don't have the right set of xform properties
                                                (i.e: translate, orient and scale) ONLY and in that order.
                                                Set this parameter to False if the object were cloned using using
                                                the cloner api in isaacsim.core.cloner. Defaults to True.

    Raises:
        Exception: if translations and positions defined at the same time.
        Exception: No prim was matched using the prim_paths_expr provided.

    Example:

    .. code-block:: python

        >>> import isaacsim.core.utils.stage as stage_utils
        >>> from isaacsim.core.cloner import GridCloner
        >>> from isaacsim.core.prims import XFormPrim
        >>> from pxr import UsdGeom
        >>>
        >>> env_zero_path = "/World/envs/env_0"
        >>> num_envs = 5
        >>>
        >>> # load the Franka Panda robot USD file
        >>> stage_utils.add_reference_to_stage(usd_path, prim_path=f"{env_zero_path}/panda")  # /World/envs/env_0/panda
        >>>
        >>> # clone the environment (num_envs)
        >>> cloner = GridCloner(spacing=1.5)
        >>> cloner.define_base_env(env_zero_path)
        >>> UsdGeom.Xform.Define(stage_utils.get_current_stage(), env_zero_path)
        >>> env_pos = cloner.clone(
        ...     source_prim_path=env_zero_path,
        ...     prim_paths=cloner.generate_paths("/World/envs/env", num_envs),
        ...     copy_from_source=True
        ... )
        >>>
        >>> # wrap all Xforms
        >>> prims = XFormPrim(prim_paths_expr="/World/envs/env.*", name="xform_view")
        >>> prims
        <isaacsim.core.prims.xform_prim.XFormPrim object at 0x7f8ffd22ebc0>
    """

    def __init__(
        self,
        prim_paths_expr: str | list[str],
        positions: torch.Tensor | None = None,
        translations: torch.Tensor | None = None,
        orientations: torch.Tensor | None = None,
        scales: torch.Tensor | None = None,
        visibilities: torch.Tensor | None = None,
        reset_xform_properties: bool = True,
    ):
        if not isinstance(prim_paths_expr, list):
            prim_paths_expr = [prim_paths_expr]

        # configure prim paths
        self._prim_paths = []
        self._regex_prim_paths = prim_paths_expr
        for prim_path_expression in prim_paths_expr:
            self._prim_paths = self._prim_paths + prim_utils.find_matching_prim_paths(prim_path_expression)
        if len(self._prim_paths) == 0:
            raise Exception(
                "Prim path expression {} is invalid, a prim matching the expression needs to created before wrapping it"
                " as view".format(prim_paths_expr)
            )
        self._prims = []
        for prim_path in self._prim_paths:
            self._prims.append(prim_utils.get_prim_at_path(prim_path))

        self._is_valid = True
        self._count = len(self._prim_paths)

        # get an instance of the simulation context to set the device
        sim = SimulationContext()
        self._device = sim.device

        # set attributes and properties
        if reset_xform_properties:
            self._set_xform_properties()
        if translations is not None and positions is not None:
            raise Exception("You can not define translation and position at the same time")
        if positions is not None or orientations is not None or translations is not None:
            if translations is not None:
                self.set_local_poses(translations, orientations)
            else:
                self.set_world_poses(positions, orientations)
        if scales is not None:
            self.set_local_scales(scales)
        if visibilities is not None:
            self.set_visibilities(visibilities=visibilities)

        default_positions, default_orientations = self.get_world_poses()
        self._positions = default_positions
        self._orientations = default_orientations

        self._ALL_INDICES = torch.arange(self.count, device=self._device)

    def __del__(self):
        self._prims = []
        self._prim_paths = []
        self._count = 0
        self._is_valid = False

    @property
    def prim_paths(self) -> list[str]:
        """
        Returns list of prim paths in the stage encapsulated in this view.
        """
        return self._prim_paths

    @property
    def count(self) -> int:
        """
        Returns:
            Number of prims encapsulated in this view.
        """
        return self._count

    @property
    def prims(self) -> list[Usd.Prim]:
        """
        Returns:
            List of USD Prim objects encapsulated in this view.

        Example:

        .. code-block:: python

            >>> prims.prims
            [Usd.Prim(</World/envs/env_0>), Usd.Prim(</World/envs/env_1>), Usd.Prim(</World/envs/env_2>),
             Usd.Prim(</World/envs/env_3>), Usd.Prim(</World/envs/env_4>)]
        """
        return self._prims

    def is_valid(self) -> bool:
        """Check that all prims have a valid USD Prim

        Returns:
            True if all prim paths specified in the view correspond to a valid prim in stage. False otherwise.

        Example:

        .. code-block:: python

            >>> prims.is_valid()
            True
        """
        return self._is_valid

    def set_visibilities(
        self,
        visibilities: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ):
        """Set the visibilities of the prims in stage

        Args:
            visibilities: flag to set the visibilities of the usd prims in stage.
                                                            Shape (M,). Where M <= size of the encapsulated prims in the view.
            env_ids (torch.Tensor | None, optional): env_ids to specify which prims
                                                                                 to manipulate. Shape (M,).
                                                                                 Defaults to None (i.e: all prims in the view).

        Example:

        .. code-block:: python

            >>> # make all prims not visible in the stage
            >>> prims.set_visibilities(visibilities=[False] * num_envs)
        """
        if self._is_valid:
            raise Exception(f"prim view {self._regex_prim_paths} is not a valid view")

        if env_ids is None:
            env_ids = self._ALL_INDICES

        for idx, env_id in enumerate(env_ids.tolist()):
            imageable = UsdGeom.Imageable(self._prims[env_id])
            if visibilities[idx]:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()

    def get_visibilities(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Returns the current visibilities of the prims in stage.

        Args:
            env_ids: env_ids to specify which prims to query. Shape (M,). Where M <= size of the encapsulated prims in the view. Defaults to None (i.e: all prims in the view).

        Returns:
            Shape (M,) with type bool, where each item holds True if the prim is visible in stage. False otherwise.

        Example:

        .. code-block:: python

            >>> # get all visibilities. Returned shape is (5,) for the example: 5 envs
            >>> prims.get_visibilities()
            [ True  True  True  True  True]
            >>>
            >>> # get the visibilities for the first, middle and last of the 5 envs. Returned shape is (3,)
            >>> prims.get_visibilities(env_ids=np.array([0, 2, 4]))
            [ True  True  True]
        """
        if not self._is_valid:
            raise Exception(f"prim view {self._regex_prim_paths} is not a valid view")

        if env_ids is None:
            env_ids = self._ALL_INDICES

        # convert to list
        env_ids = env_ids.tolist()

        visibilities = np.zeros(shape=env_ids.shape[0], dtype="bool")
        for idx, curr_env_id in env_ids:
            visibilities[idx] = (
                UsdGeom.Imageable(self._prims[curr_env_id]).ComputeVisibility(Usd.TimeCode.Default())
                != UsdGeom.Tokens.invisible
            )

        return torch.tensor(visibilities, device=self._device)

    def get_world_poses(self, env_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the poses of the prims in the view with respect to the world's frame

        Args:
            env_ids (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): env_ids to specify which prims
                                                                                 to query. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
            usd (bool, optional): True to query from usd. Otherwise False to query from Fabric data. Defaults to True.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor], Tuple[wp.indexedarray, wp.indexedarray]]: first index is positions in the world frame of the prims. shape is (M, 3).
                                                                                     second index is quaternion orientations in the world frame of the prims.
                                                                                     quaternion is scalar-first (w, x, y, z). shape is (M, 4).

        Example:

        .. code-block:: python

            >>> # get all prims poses with respect to the world's frame.
            >>> # Returned shape is position (5, 3) and orientation (5, 4) for the example: 5 envs
            >>> positions, orientations = prims.get_world_poses()
            >>> positions
            [[ 1.5  -0.75  0.  ]
             [ 1.5   0.75  0.  ]
             [ 0.   -0.75  0.  ]
             [ 0.    0.75  0.  ]
             [-1.5  -0.75  0.  ]]
            >>> orientations
            [[1. 0. 0. 0.]
             [1. 0. 0. 0.]
             [1. 0. 0. 0.]
             [1. 0. 0. 0.]
             [1. 0. 0. 0.]]
            >>>
            >>> # get only the prims poses with respect to the world's frame for the first, middle and last of the 5 envs.
            >>> # Returned shape is position (3, 3) and orientation (3, 4) for the example: 3 envs selected
            >>> positions, orientations = prims.get_world_poses(env_ids=np.array([0, 2, 4]))
            >>> positions
            [[ 1.5  -0.75  0.  ]
             [ 0.   -0.75  0.  ]
             [-1.5  -0.75  0.  ]]
            >>> orientations
            [[1. 0. 0. 0.]
             [1. 0. 0. 0.]
             [1. 0. 0. 0.]]
        """
        if self._is_valid:
            raise Exception(f"prim view {self._regex_prim_paths} is not a valid view")

        if env_ids is None:
            env_ids = self._ALL_INDICES

        positions = np.zeros((env_ids.shape[0], 3), dtype=np.float32)
        orientations = np.zeros((env_ids.shape[0], 4), dtype=np.float32)

        for idx, curr_env_id in enumerate(env_ids.tolist()):
            result_transform = _get_world_pose_transform_w_scale(self._prim_paths[curr_env_id], False)
            result_transform.Orthonormalize()
            result_transform = np.transpose(result_transform)
            r = Rotation.from_matrix(result_transform[:3, :3])
            positions[idx] = result_transform[:3, 3]
            orientations[idx] = r.as_quat()[[3, 0, 1, 2]]
        positions = torch.tensor(positions, device=self._device)
        orientations = torch.tensor(orientations, device=self._device)
        return positions, orientations

    def set_world_poses(
        self,
        positions: torch.Tensor | None = None,
        orientations: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
    ):
        """Set prim poses in the view with respect to the world's frame

        .. warning::

            This method will change (teleport) the prim poses immediately to the indicated value

        Args:
            positions (Optional[Union[np.ndarray, torch.Tensor, wp.array]], optional): positions in the world frame of the prims. shape is (M, 3).
                                                                             Defaults to None, which means left unchanged.
            orientations (Optional[Union[np.ndarray, torch.Tensor, wp.array]], optional): quaternion orientations in the world frame of the prims.
                                                                                quaternion is scalar-first (w, x, y, z). shape is (M, 4).
                                                                                Defaults to None, which means left unchanged.
            env_ids (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): env_ids to specify which prims
                                                                                 to query. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
            usd (bool, optional): True to query from usd. Otherwise False to query from Fabric data. Defaults to True.

        .. hint::

            This method belongs to the methods used to set the prim state

        Example:

        .. code-block:: python

            >>> # reposition all prims in row (x-axis)
            >>> positions = np.zeros((num_envs, 3))
            >>> positions[:,0] = np.arange(num_envs)
            >>> orientations = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (num_envs, 1))
            >>> prims.set_world_poses(positions, orientations)
            >>>
            >>> # reposition only the prims for the first, middle and last of the 5 envs in column (y-axis)
            >>> positions = np.zeros((3, 3))
            >>> positions[:,1] = np.arange(3)
            >>> orientations = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (3, 1))
            >>> prims.set_world_poses(positions, orientations, env_ids=np.array([0, 2, 4]))
        """
        if self._is_valid:
            raise Exception(f"prim view {self._regex_prim_paths} is not a valid view")

        if positions is None or orientations is None:
            current_positions, current_orientations = self.get_world_poses(env_ids=env_ids)
            if positions is None:
                positions = current_positions
            if orientations is None:
                orientations = current_orientations

        parent_transforms = np.zeros(shape=(env_ids.shape[0], 4, 4), dtype="float32")

        for idx, curr_env_id in enumerate(env_ids.tolist()):
            parent_transforms[idx] = np.array(
                UsdGeom.Xformable(get_prim_parent(self._prims[curr_env_id])).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()
                ),
                dtype="float32",
            )

        calculated_translations, calculated_orientations = self._backend_utils.get_local_from_world(
            parent_transforms, positions, orientations, self._device
        )
        self.set_local_poses(
            translations=calculated_translations, orientations=calculated_orientations, env_ids=env_ids
        )

    def get_local_poses(self, env_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get prim poses in the view with respect to the local frame (the prim's parent frame)

        Args:
            env_ids (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): env_ids to specify which prims
                                                                                 to query. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor], Tuple[wp.indexedarray, wp.indexedarray]]:
                                          first index is translations in the local frame of the prims. shape is (M, 3).
                                            second index is quaternion orientations in the local frame of the prims.
                                            quaternion is scalar-first (w, x, y, z). shape is (M, 4).

        Example:

        .. code-block:: python

            >>> # get all prims poses with respect to the local frame.
            >>> # Returned shape is position (5, 3) and orientation (5, 4) for the example: 5 envs
            >>> positions, orientations = prims.get_local_poses()
            >>> positions
            [[ 1.5  -0.75  0.  ]
             [ 1.5   0.75  0.  ]
             [ 0.   -0.75  0.  ]
             [ 0.    0.75  0.  ]
             [-1.5  -0.75  0.  ]]
            >>> orientations
            [[1. 0. 0. 0.]
             [1. 0. 0. 0.]
             [1. 0. 0. 0.]
             [1. 0. 0. 0.]
             [1. 0. 0. 0.]]
            >>>
            >>> # get only the prims poses with respect to the local frame for the first, middle and last of the 5 envs.
            >>> # Returned shape is position (3, 3) and orientation (3, 4) for the example: 3 envs selected
            >>> positions, orientations = prims.get_local_poses(env_ids=np.array([0, 2, 4]))
            >>> positions
            [[ 1.5  -0.75  0.  ]
             [ 0.   -0.75  0.  ]
             [-1.5  -0.75  0.  ]]
            >>> orientations
            [[1. 0. 0. 0.]
             [1. 0. 0. 0.]
             [1. 0. 0. 0.]]
        """
        if self._is_valid:
            raise Exception(f"prim view {self._regex_prim_paths} is not a valid view")

        translations = np.zeros(shape=(env_ids.shape[0], 3), dtype="float32")
        orientations = np.zeros(shape=(env_ids.shape[0], 4), dtype="float32")

        for idx, curr_env_id in enumerate(env_ids.tolist()):
            usd_prim = self._prims[curr_env_id]
            local_transform = usdrt.Gf.Matrix4d(
                UsdGeom.Xformable(usd_prim).GetLocalTransformation(Usd.TimeCode.Default())
            )
            local_transform.Orthonormalize()
            translations[idx] = np.array(local_transform.ExtractTranslation())
            orientations[idx] = np.array(local_transform.ExtractRotationQuat())[[3, 0, 1, 2]]

        translations = torch.tensor(translations, device=self._device)
        orientations = torch.tensor(orientations, device=self._device)
        return translations, orientations

    def set_local_poses(
        self,
        translations: torch.Tensor | None = None,
        orientations: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
    ):
        """Set prim poses in the view with respect to the local frame (the prim's parent frame)

        .. warning::

            This method will change (teleport) the prim poses immediately to the indicated value

        Args:
            translations (Optional[Union[np.ndarray, torch.Tensor, wp.array]], optional):
                                                          translations in the local frame of the prims
                                                          (with respect to its parent prim). shape is (M, 3).
                                                          Defaults to None, which means left unchanged.
            orientations (Optional[Union[np.ndarray, torch.Tensor, wp.array]], optional):
                                                          quaternion orientations in the local frame of the prims.
                                                          quaternion is scalar-first (w, x, y, z). shape is (M, 4).
                                                          Defaults to None, which means left unchanged.
            env_ids (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): env_ids to specify which prims
                                                                                 to manipulate. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).

        .. hint::

            This method belongs to the methods used to set the prim state

        Example:

        .. code-block:: python

            >>> # reposition all prims
            >>> positions = np.zeros((num_envs, 3))
            >>> positions[:,0] = np.arange(num_envs)
            >>> orientations = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (num_envs, 1))
            >>> prims.set_local_poses(positions, orientations)
            >>>
            >>> # reposition only the prims for the first, middle and last of the 5 envs
            >>> positions = np.zeros((3, 3))
            >>> positions[:,1] = np.arange(3)
            >>> orientations = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (3, 1))
            >>> prims.set_local_poses(positions, orientations, env_ids=np.array([0, 2, 4]))
        """
        if self._is_valid:
            raise Exception(f"prim view {self._regex_prim_paths} is not a valid view")

        for idx, curr_env_id in enumerate(env_ids.tolist()):
            if translations is not None:
                translation = Gf.Vec3d(*translations[idx])
                xform_op = self._prims[curr_env_id].GetAttribute("xformOp:translate")
                xform_op.Set(translation)
            if orientations is not None:
                xform_op = self._prims[curr_env_id].GetAttribute("xformOp:orient")
                rotq = (
                    Gf.Quatf(*orientations[idx]) if xform_op.GetTypeName() == "quatf" else Gf.Quatd(*orientations[idx])
                )
                xform_op.Set(rotq)

    def get_world_scales(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Get prim scales in the view with respect to the world's frame

        Args:
            env_ids (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): env_ids to specify which prims
                                                                                 to query. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).

        Returns:
            Union[np.ndarray, torch.Tensor, wp.indexedarray]: scales applied to the prim's dimensions in the world frame. shape is (M, 3).
        """
        if self._is_valid:
            raise Exception(f"prim view {self._regex_prim_paths} is not a valid view")

        scales = np.zeros(shape=(env_ids.shape[0], 3), dtype="float32")
        for idx, curr_env_id in enumerate(env_ids.tolist()):
            prim_tf = UsdGeom.Xformable(self._prims[curr_env_id]).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            transform = Gf.Transform()
            transform.SetMatrix(prim_tf)
            scales[idx] = np.array(transform.GetScale(), dtype="float32")
        scales = torch.tensor(scales, device=self._device)
        return scales

    def set_local_scales(
        self,
        scales: torch.Tensor | None,
        env_ids: torch.Tensor | None = None,
    ):
        """Set prim scales in the view with respect to the local frame (the prim's parent frame)

        Args:
            scales (Optional[Union[np.ndarray, torch.Tensor, wp.array]]): scales to be applied to the prim's dimensions in the view.
                                                                shape is (M, 3).
            env_ids (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): env_ids to specify which prims
                                                                                 to manipulate. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).

        Example:

        .. code-block:: python

            >>> # set the scale for all prims. Since there are 5 envs, the scale is repeated 5 times
            >>> scales = np.tile(np.array([1.0, 0.75, 0.5]), (num_envs, 1))
            >>> prims.set_local_scales(scales)
            >>>
            >>> # set the scale for the first, middle and last of the 5 envs
            >>> scales = np.tile(np.array([1.0, 0.75, 0.5]), (3, 1))
            >>> prims.set_local_scales(scales, env_ids=np.array([0, 2, 4]))
        """
        if self._is_valid:
            raise Exception(f"prim view {self._regex_prim_paths} is not a valid view")

        for idx, curr_env_id in enumerate(env_ids.tolist()):
            scale = Gf.Vec3d(*scales[idx].tolist())
            xform_op = self._prims[curr_env_id].GetAttribute("xformOp:scale")
            xform_op.Set(scale)

    def get_local_scales(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Get prim scales in the view with respect to the local frame (the parent's frame).

        Args:
            env_ids (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): env_ids to specify which prims
                                                                                 to query. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).

        Returns:
            Union[np.ndarray, torch.Tensor, wp.indexedarray]: scales applied to the prim's dimensions in the local frame. shape is (M, 3).

        Example:

        .. code-block:: python

            >>> # get all prims scales with respect to the local frame.
            >>> # Returned shape is (5, 3) for the example: 5 envs
            >>> prims.get_local_scales()
            [[1. 1. 1.]
             [1. 1. 1.]
             [1. 1. 1.]
             [1. 1. 1.]
             [1. 1. 1.]]
            >>>
            >>> # get only the prims scales with respect to the local frame for the first, middle and last of the 5 envs.
            >>> # Returned shape is (3, 3) for the example: 3 envs selected
            >>> prims.get_local_scales(env_ids=np.array([0, 2, 4]))
            [[1. 1. 1.]
             [1. 1. 1.]
             [1. 1. 1.]]
        """
        if self._is_valid:
            raise Exception(f"prim view {self._regex_prim_paths} is not a valid view")

        if env_ids is None:
            env_ids = self._ALL_INDICES

        scales = torch.zeros(shape=(env_ids.shape[0], 3))
        for idx, curr_env_id in enumerate(env_ids.tolist()):
            # FIXME: check if it can be directly transformed into a tensor, otherwise go over numpy
            scales[idx] = torch.tensor(self._prims[curr_env_id].GetAttribute("xformOp:scale").Get())

        return scales.to(self._device)

    def reset(self):
        self.set_world_poses(self._positions, self._orientations)

    def _set_xform_properties(self):
        current_positions, current_orientations = self.get_world_poses()
        properties_to_remove = [
            "xformOp:rotateX",
            "xformOp:rotateXZY",
            "xformOp:rotateY",
            "xformOp:rotateYXZ",
            "xformOp:rotateYZX",
            "xformOp:rotateZ",
            "xformOp:rotateZYX",
            "xformOp:rotateZXY",
            "xformOp:rotateXYZ",
            "xformOp:transform",
        ]
        for i in range(self._count):
            prop_names = self._prims[i].GetPropertyNames()
            xformable = UsdGeom.Xformable(self._prims[i])
            xformable.ClearXformOpOrder()
            for prop_name in prop_names:
                if prop_name in properties_to_remove:
                    self._prims[i].RemoveProperty(prop_name)
            if "xformOp:scale" not in prop_names:
                xform_op_scale = xformable.AddXformOp(UsdGeom.XformOp.TypeScale, UsdGeom.XformOp.PrecisionDouble, "")
                xform_op_scale.Set(Gf.Vec3d([1.0, 1.0, 1.0]))
            else:
                xform_op_scale = UsdGeom.XformOp(self._prims[i].GetAttribute("xformOp:scale"))

                if "xformOp:scale:unitsResolve" in prop_names:
                    new_scale = np.array(self._prims[i].GetAttribute("xformOp:scale").Get()) * np.array(
                        self._prims[i].GetAttribute("xformOp:scale:unitsResolve").Get()
                    )
                    self._prims[i].GetAttribute("xformOp:scale").Set(Gf.Vec3d(*list(new_scale)))
                    self._prims[i].RemoveProperty("xformOp:scale:unitsResolve")

            if "xformOp:translate" not in prop_names:
                xform_op_tranlsate = xformable.AddXformOp(
                    UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.PrecisionDouble, ""
                )
            else:
                xform_op_tranlsate = UsdGeom.XformOp(self._prims[i].GetAttribute("xformOp:translate"))

            if "xformOp:orient" not in prop_names:
                xform_op_rot = xformable.AddXformOp(UsdGeom.XformOp.TypeOrient, UsdGeom.XformOp.PrecisionDouble, "")
            else:
                xform_op_rot = UsdGeom.XformOp(self._prims[i].GetAttribute("xformOp:orient"))
            xformable.SetXformOpOrder([xform_op_tranlsate, xform_op_rot, xform_op_scale])
        self.set_world_poses(positions=current_positions, orientations=current_orientations)
        return


def _get_world_pose_transform_w_scale(prim_path):
    # This will return a transformation matrix with translation as the last row and scale included
    usd_prim = prim_utils.get_prim_at_path(prim_path=prim_path, fabric=False)
    local_transform = usdrt.Gf.Matrix4d(UsdGeom.Xformable(usd_prim).GetLocalTransformation(Usd.TimeCode.Default()))
    parent_prim = get_prim_parent(prim_utils.get_prim_at_path(prim_path=prim_path, fabric=False))
    parent_world_transform = usdrt.Gf.Matrix4d(1.0)
    if parent_prim:
        parent_world_transform = _get_world_pose_transform_w_scale(prim_utils.get_prim_path(parent_prim), fabric=False)

    # FIXME: check if can be replaced with this simpler code
    # compute local to world transform
    xform = UsdGeom.Xformable(usd_prim)
    world_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    return local_transform * parent_world_transform
