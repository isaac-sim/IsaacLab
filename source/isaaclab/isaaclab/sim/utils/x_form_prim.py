import re
import weakref
from typing import List, Optional, Tuple, Union

import carb
import isaacsim.core.utils.fabric as fabric_utils
import isaacsim.core.utils.interops as interops_utils
import numpy as np
import torch
import usdrt
import warp as wp
from isaacsim.core.prims.impl.prim import Prim
from isaacsim.core.simulation_manager import IsaacEvents, SimulationManager
from isaacsim.core.utils.prims import (
    get_prim_at_path,
    get_prim_parent,
    is_prim_non_root_articulation_link,
    is_prim_path_valid,
)
from isaacsim.core.utils.types import XFormPrimViewState
from isaacsim.core.utils.xforms import get_local_pose, get_world_pose
from pxr import Gf, Usd, UsdGeom, UsdShade

from isaaclab.sim.utils.stage import get_current_stage


class Prim(object):
    def __init__(self, prim_paths_expr: str, name: str = "prim_view") -> None:
        if not isinstance(prim_paths_expr, list):
            prim_paths_expr = [prim_paths_expr]
        self._prim_paths = []
        self._callbacks = []
        for prim_path_expression in prim_paths_expr:
            self._prim_paths = self._prim_paths + find_matching_prim_paths(prim_path_expression)
        self._is_valid = True
        if len(self._prim_paths) == 0:
            raise Exception(
                "Prim path expression {} is invalid, a prim matching the expression needs to created before wrapping it as view".format(
                    prim_paths_expr
                )
            )
        self._name = name
        self._count = len(self._prim_paths)
        self._prims = []
        self._regex_prim_paths = prim_paths_expr
        for prim_path in self._prim_paths:
            self._prims.append(get_prim_at_path(prim_path))
        self._backend = SimulationManager.get_backend()
        self._device = SimulationManager.get_physics_sim_device()
        self._backend_utils = SimulationManager._get_backend_utils()
        self._callbacks.append(
            SimulationManager.register_callback(self._on_physics_ready, event=IsaacEvents.PHYSICS_READY)
        )
        self._callbacks.append(SimulationManager.register_callback(self._on_post_reset, event=IsaacEvents.POST_RESET))
        self._callbacks.append(
            SimulationManager.register_callback(self._on_prim_deletion, event=IsaacEvents.PRIM_DELETION)
        )
        return

    def __del__(self):
        self.destroy()

    def destroy(self):
        for callback_id in self._callbacks:
            SimulationManager.deregister_callback(callback_id)
        self._callbacks = []
        self._prims = []
        self._prim_paths = []
        self._count = 0
        self._is_valid = False

    @property
    def prim_paths(self) -> List[str]:
        """
        Returns:
            List[str]: list of prim paths in the stage encapsulated in this view.

        Example:

        .. code-block:: python

            >>> prims.prim_paths
            ['/World/envs/env_0', '/World/envs/env_1', '/World/envs/env_2', '/World/envs/env_3', '/World/envs/env_4']
        """
        return self._prim_paths

    @property
    def name(self) -> str:
        """
        Returns:
            str: name given to the prims view when instantiating it.
        """
        return self._name

    @property
    def count(self) -> int:
        """
        Returns:
            int: Number of prims encapsulated in this view.

        Example:

        .. code-block:: python

            >>> prims.count
            5
        """
        return self._count

    @property
    def prims(self) -> List[Usd.Prim]:
        """
        Returns:
            List[Usd.Prim]: List of USD Prim objects encapsulated in this view.

        Example:

        .. code-block:: python

            >>> prims.prims
            [Usd.Prim(</World/envs/env_0>), Usd.Prim(</World/envs/env_1>), Usd.Prim(</World/envs/env_2>),
             Usd.Prim(</World/envs/env_3>), Usd.Prim(</World/envs/env_4>)]
        """
        return self._prims

    @property
    def initialized(self) -> bool:
        """Check if prim view is initialized

        Returns:
            bool: True if the view object was initialized (after the first call of .initialize()). False otherwise.

        Example:

        .. code-block:: python

            >>> # given an initialized articulation view
            >>> prims.initialized
            True
        """
        return SimulationManager.get_physics_sim_view() is not None

    def post_reset(self) -> None:
        """Reset the prims to its default state

        Example:

        .. code-block:: python

            >>> prims.post_reset()
        """
        self._on_post_reset(None)
        return

    def is_valid(self, indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None) -> bool:
        """Check that all prims have a valid USD Prim

        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                 to query. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).

        Returns:
            bool: True if all prim paths specified in the view correspond to a valid prim in stage. False otherwise.

        Example:

        .. code-block:: python

            >>> prims.is_valid()
            True
        """
        return self._is_valid

    def initialize(self, physics_sim_view: omni.physics.tensors.SimulationView = None) -> None:
        """Create a physics simulation view if not passed and set other properties using the PhysX tensor API

        .. note::

            For this particular class, calling this method will do nothing

        Args:
            physics_sim_view (omni.physics.tensors.SimulationView, optional): current physics simulation view. Defaults to None.

        Example:

        .. code-block:: python

            >>> prims.initialize()
        """
        self._on_physics_ready(None)
        return

    def _on_prim_deletion(self, prim_path):
        # TODO: regex matching in c++
        if prim_path == "/":
            self._is_valid = False
            for callback_id in self._callbacks:
                SimulationManager.deregister_callback(callback_id)
            self._callbacks = []
            return
        for regex_prim_paths in self._regex_prim_paths:
            result = re.match(
                pattern="^" + "/".join(regex_prim_paths.split("/")[: prim_path.count("/") + 1]) + "$", string=prim_path
            )
            if result:
                self._is_valid = False
                for callback_id in self._callbacks:
                    SimulationManager.deregister_callback(callback_id)
                self._callbacks = []
                return
        return

    def _on_physics_ready(self, event):
        self._backend = SimulationManager.get_backend()
        self._device = SimulationManager.get_physics_sim_device()
        self._backend_utils = SimulationManager._get_backend_utils()
        return

    def _on_post_reset(self, event) -> None:
        return

    def _remove_callbacks(self) -> None:
        for callback_id in self._callbacks:
            SimulationManager.deregister_callback(callback_id)
        self._callbacks = []
        return




class XFormPrim(Prim):
    """Provides high level functions to deal with a Xform prim view (one or many) and its descendants
    as well as its attributes/properties.

    This class wraps all matching Xforms found at the regex provided at the ``prim_paths_expr`` argument

    .. note::

        Each prim will have ``xformOp:orient``, ``xformOp:translate`` and ``xformOp:scale`` only post-init,
        unless it is a non-root articulation link.

    Args:
        prim_paths_expr (Union[str, List[str]]): prim paths regex to encapsulate all prims that match it.
                                example: "/World/Env[1-5]/Franka" will match /World/Env1/Franka,
                                /World/Env2/Franka..etc.
                                (a non regex prim path can also be used to encapsulate one Xform). Additionally a
                                list of regex can be provided. example ["/World/Env[1-5]/Franka", "/World/Env[10-19]/Franka"].
        name (str, optional): shortname to be used as a key by Scene class.
                                Note: needs to be unique if the object is added to the Scene.
                                Defaults to "xform_prim_view".
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
        usd (bool, optional): True to strictly read/ write from usd. Otherwise False to allow read/ write from Fabric during initialization. Defaults to True.

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
        prim_paths_expr: Union[str, List[str]],
        name: str = "xform_prim_view",
        positions: Optional[Union[np.ndarray, torch.Tensor]] = None,
        translations: Optional[Union[np.ndarray, torch.Tensor]] = None,
        orientations: Optional[Union[np.ndarray, torch.Tensor]] = None,
        scales: Optional[Union[np.ndarray, torch.Tensor]] = None,
        visibilities: Optional[Union[np.ndarray, torch.Tensor]] = None,
        reset_xform_properties: bool = True,
        usd: bool = True,
    ) -> None:
        self._non_root_link = False
        if not isinstance(prim_paths_expr, list):
            prim_paths_expr = [prim_paths_expr]
        Prim.__init__(self, prim_paths_expr, name)
        self._default_state = None
        self._applied_visual_materials = [None] * self._count
        self._binding_apis = [None] * self._count
        self._non_root_link = is_prim_non_root_articulation_link(prim_path=self._prim_paths[0])
        self._usdrt_stage = get_current_stage(fabric=True)
        self._view_index_attr = "isaac_sim:view_index:" + str(hash(self))
        self._view_in_fabric_prepared = False
        self._selection = None
        self._fabric_to_view = None
        self._view_to_fabric = None
        self._default_view_indices = None
        self._fabric_data_dicts = dict()
        self._fabric_data_valid = dict()
        self._fabric_hierarchy = None
        if SimulationManager.get_physics_sim_view() is not None:
            XFormPrim._on_physics_ready(self, None)
        if not self._non_root_link and reset_xform_properties:
            self._set_xform_properties()
        if not self._non_root_link:
            if translations is not None and positions is not None:
                raise Exception("You can not define translation and position at the same time")
            if positions is not None or orientations is not None or translations is not None:
                if translations is not None:
                    self.set_local_poses(translations, orientations)
                else:
                    self.set_world_poses(positions, orientations)
            if scales is not None:
                XFormPrim.set_local_scales(self, scales)
        if visibilities is not None:
            XFormPrim.set_visibilities(self, visibilities=visibilities)
        if not self._non_root_link:
            default_positions, default_orientations = self.get_world_poses(usd=usd)
            if self._backend == "warp":
                self._default_state = XFormPrimViewState(
                    positions=default_positions.data, orientations=default_orientations.data
                )
            else:
                self._default_state = XFormPrimViewState(positions=default_positions, orientations=default_orientations)
        return

    @property
    def is_non_root_articulation_link(self) -> bool:
        """
        Returns:
            bool: True if the prim corresponds to a non root link in an articulation. Otherwise False.
        """
        return self._non_root_link

    def set_visibilities(
        self,
        visibilities: Union[np.ndarray, torch.Tensor, wp.array],
        indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None,
    ) -> None:
        """Set the visibilities of the prims in stage

        Args:
            visibilities (Union[np.ndarray, torch.Tensor, wp.array]): flag to set the visibilities of the usd prims in stage.
                                                            Shape (M,). Where M <= size of the encapsulated prims in the view.
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                 to manipulate. Shape (M,).
                                                                                 Defaults to None (i.e: all prims in the view).

        Example:

        .. code-block:: python

            >>> # make all prims not visible in the stage
            >>> prims.set_visibilities(visibilities=[False] * num_envs)
        """
        if self._is_valid:
            indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
            read_idx = 0
            indices = self._backend_utils.to_list(indices)
            visibilities = self._backend_utils.to_list(visibilities)
            for i in indices:
                imageable = UsdGeom.Imageable(self._prims[i])
                if visibilities[read_idx]:
                    imageable.MakeVisible()
                else:
                    imageable.MakeInvisible()
                read_idx += 1
        else:
            raise Exception("prim view {} is not a valid view".format(self._regex_prim_paths))
        return

    def get_visibilities(
        self, indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None
    ) -> Union[np.ndarray, torch.Tensor, wp.indexedarray]:
        """Returns the current visibilities of the prims in stage.

        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                 to query. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).

        Returns:
            Union[np.ndarray, torch.Tensor, wp.indexedarray]: Shape (M,) with type bool, where each item holds True
                                             if the prim is visible in stage. False otherwise.

        Example:

        .. code-block:: python

            >>> # get all visibilities. Returned shape is (5,) for the example: 5 envs
            >>> prims.get_visibilities()
            [ True  True  True  True  True]
            >>>
            >>> # get the visibilities for the first, middle and last of the 5 envs. Returned shape is (3,)
            >>> prims.get_visibilities(indices=np.array([0, 2, 4]))
            [ True  True  True]
        """
        if self._is_valid:
            indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
            visibilities = np.zeros(shape=indices.shape[0], dtype="bool")
            write_idx = 0
            indices = self._backend_utils.to_list(indices)
            for i in indices:
                visibilities[write_idx] = (
                    UsdGeom.Imageable(self._prims[i]).ComputeVisibility(Usd.TimeCode.Default())
                    != UsdGeom.Tokens.invisible
                )
                write_idx += 1
            visibilities = self._backend_utils.convert(visibilities, dtype="bool", device=self._device, indexed=True)
            return visibilities
        else:
            raise Exception("prim view {} is not a valid view".format(self._regex_prim_paths))

    def get_default_state(self) -> XFormPrimViewState:
        """Get the default states (positions and orientations) defined with the ``set_default_state`` method

        Returns:
            XFormPrimViewState: returns the default state of the prims that is used after each reset.

        Example:

        .. code-block:: python

            >>> state = prims.get_default_state()
            >>> state
            <isaacsim.core.utils.types.XFormPrimViewState object at 0x7f82f73e3070>
            >>> state.positions
            [[ 1.5  -0.75  0.  ]
             [ 1.5   0.75  0.  ]
             [ 0.   -0.75  0.  ]
             [ 0.    0.75  0.  ]
             [-1.5  -0.75  0.  ]]
            >>> state.orientations
            [[1. 0. 0. 0.]
             [1. 0. 0. 0.]
             [1. 0. 0. 0.]
             [1. 0. 0. 0.]
             [1. 0. 0. 0.]]
        """
        if self._is_valid:
            if self._non_root_link:
                carb.log_warn("This view corresponds to non root links that are included in an articulation")
            return self._default_state
        else:
            raise Exception("prim view {} is not a valid view".format(self._regex_prim_paths))

    def set_default_state(
        self,
        positions: Optional[Union[np.ndarray, torch.Tensor, wp.array]] = None,
        orientations: Optional[Union[np.ndarray, torch.Tensor, wp.array]] = None,
        indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None,
    ) -> None:
        """Set the default state of the prims (positions and orientations), that will be used after each reset.

        .. note::

            The default states will be set during post-reset (e.g., calling ``.post_reset()`` or ``world.reset()`` methods)

        Args:
            positions (Optional[Union[np.ndarray, torch.Tensor, wp.array]], optional):  positions in the world frame of the prim. shape is (M, 3).
                                                       Defaults to None, which means left unchanged.
            orientations (Optional[Union[np.ndarray, torch.Tensor, wp.array]], optional): quaternion orientations in the world frame of the prim.
                                                          quaternion is scalar-first (w, x, y, z). shape is (M, 4).
                                                          Defaults to None, which means left unchanged.
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                 to manipulate. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).

        Example:

        .. code-block:: python

            >>> # configure default states for all prims
            >>> positions = np.zeros((num_envs, 3))
            >>> positions[:, 0] = np.arange(num_envs)
            >>> orientations = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (num_envs, 1))
            >>> prims.set_default_state(positions=positions, orientations=orientations)
            >>>
            >>> # set default states during post-reset
            >>> prims.post_reset()
        """
        if self._is_valid:
            if self._non_root_link:
                carb.log_warn("This view corresponds to non root links that are included in an articulation")
                return

            if self._default_state is None:
                self._default_state = XFormPrimViewState(positions=None, orientations=None)

            if positions is not None:
                if indices is None:
                    self._default_state.positions = positions
                else:
                    if self._backend == "warp":
                        indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
                        self._default_state.positions = self._backend_utils.assign(
                            positions, self._default_state.positions, indices
                        )
                    else:
                        self._default_state.positions[indices] = positions
            if orientations is not None:
                if indices is None:
                    self._default_state.orientations = orientations
                else:
                    if self._backend == "warp":
                        indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
                        self._default_state.orientations = self._backend_utils.assign(
                            orientations, self._default_state.orientations, indices
                        )
                    else:
                        self._default_state.orientations[indices] = orientations
            return
        else:
            raise Exception("prim view {} is not a valid view".format(self._regex_prim_paths))

    def apply_visual_materials(
        self,
        visual_materials: Union["VisualMaterial", List["VisualMaterial"]],
        weaker_than_descendants: Optional[Union[bool, List[bool]]] = None,
        indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None,
    ) -> None:
        """Apply visual material to the prims and optionally their prim descendants.

        Args:
            visual_materials (Union[VisualMaterial, List[VisualMaterial]]): visual materials to be applied to the prims. Currently supports
                                                                            PreviewSurface, OmniPBR and OmniGlass. If a list is provided then
                                                                            its size has to be equal the view's size or indices size.
                                                                            If one material is provided it will be applied to all prims in the view.
            weaker_than_descendants (Optional[Union[bool, List[bool]]], optional):  True if the material shouldn't override the descendants
                                                                                    materials, otherwise False. Defaults to False.
                                                                                    If a list of visual materials is provided then a list
                                                                                    has to be provided with the same size for this arg as well.
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                 to manipulate. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).

        Raises:
            Exception: length of visual materials != length of prims indexed
            Exception: length of visual materials != length of weaker descendants bools arg

        Example:

        .. code-block:: python

            >>> from isaacsim.core.api.materials import OmniGlass
            >>>
            >>> # create a dark-red glass visual material
            >>> material = OmniGlass(
            ...     prim_path="/World/material/glass",  # path to the material prim to create
            ...     ior=1.25,
            ...     depth=0.001,
            ...     thin_walled=False,
            ...     color=np.array([0.5, 0.0, 0.0])
            ... )
            >>> prims.apply_visual_materials(material)
        """
        if self._is_valid:
            indices = self._backend_utils.resolve_indices(indices, self.count, self._device)

            if isinstance(visual_materials, list):
                if indices.shape[0] != len(visual_materials):
                    raise Exception("length of visual materials != length of prims indexed")
                if weaker_than_descendants is None:
                    weaker_than_descendants = [False] * len(visual_materials)
                if len(visual_materials) != len(weaker_than_descendants):
                    raise Exception("length of visual materials != length of weaker descendants bools arg")
            if isinstance(visual_materials, list):
                read_idx = 0
                indices = self._backend_utils.to_list(indices)
                for i in indices:
                    if self._binding_apis[i] is None:
                        if self._prims[i].HasAPI(UsdShade.MaterialBindingAPI):
                            self._binding_apis[i] = UsdShade.MaterialBindingAPI(self._prims[i])
                        else:
                            self._binding_apis[i] = UsdShade.MaterialBindingAPI.Apply(self._prims[i])
                    if weaker_than_descendants[read_idx]:
                        self._binding_apis[i].Bind(
                            visual_materials[read_idx].material, bindingStrength=UsdShade.Tokens.weakerThanDescendants
                        )
                    else:
                        self._binding_apis[i].Bind(
                            visual_materials[read_idx].material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
                        )
                    self._applied_visual_materials[i] = visual_materials[read_idx]
                    read_idx += 1
                return
            else:
                if weaker_than_descendants is None:
                    weaker_than_descendants = False
                indices = self._backend_utils.to_list(indices)
                for i in indices:
                    if self._binding_apis[i] is None:
                        if self._prims[i].HasAPI(UsdShade.MaterialBindingAPI):
                            self._binding_apis[i] = UsdShade.MaterialBindingAPI(self._prims[i])
                        else:
                            self._binding_apis[i] = UsdShade.MaterialBindingAPI.Apply(self._prims[i])
                    if weaker_than_descendants:
                        self._binding_apis[i].Bind(
                            visual_materials.material, bindingStrength=UsdShade.Tokens.weakerThanDescendants
                        )
                    else:
                        self._binding_apis[i].Bind(
                            visual_materials.material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
                        )
                    self._applied_visual_materials[i] = visual_materials
            return
        else:
            raise Exception("prim view {} is not a valid view".format(self._regex_prim_paths))

    def get_applied_visual_materials(
        self, indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None
    ) -> List["VisualMaterial"]:
        """Get the current applied visual materials

        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                 to query. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).

        Returns:
            List[VisualMaterial]: a list of the current applied visual materials to the prims if its type is currently supported.

        Example:

        .. code-block:: python

            >>> # get all applied visual materials. Returned size is 5 for the example: 5 envs
            >>> prims.get_applied_visual_materials()
            [<isaacsim.core.api.materials.omni_glass.OmniGlass object at 0x7f829c165de0>,
             <isaacsim.core.api.materials.omni_glass.OmniGlass object at 0x7f829c165de0>,
             <isaacsim.core.api.materials.omni_glass.OmniGlass object at 0x7f829c165de0>,
             <isaacsim.core.api.materials.omni_glass.OmniGlass object at 0x7f829c165de0>,
             <isaacsim.core.api.materials.omni_glass.OmniGlass object at 0x7f829c165de0>]
            >>>
            >>> # get the applied visual materials for the first, middle and last of the 5 envs. Returned size is 3
            >>> prims.get_applied_visual_materials(indices=np.array([0, 2, 4]))
            [<isaacsim.core.api.materials.omni_glass.OmniGlass object at 0x7f829c165de0>,
             <isaacsim.core.api.materials.omni_glass.OmniGlass object at 0x7f829c165de0>,
             <isaacsim.core.api.materials.omni_glass.OmniGlass object at 0x7f829c165de0>]
        """
        if self._is_valid:
            indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
            result = [None] * indices.shape[0]
            write_idx = 0
            indices = self._backend_utils.to_list(indices)
            for i in indices:
                if self._binding_apis[i] is None:
                    if self._prims[i].HasAPI(UsdShade.MaterialBindingAPI):
                        self._binding_apis[i] = UsdShade.MaterialBindingAPI(self._prims[i])
                    else:
                        self._binding_apis[i] = UsdShade.MaterialBindingAPI.Apply(self._prims[i])
                if self._applied_visual_materials[i] is not None:
                    result[write_idx] = self._applied_visual_materials[i]
                    write_idx += 1
                else:
                    visual_binding = self._binding_apis[i].GetDirectBinding()
                    material_path = str(visual_binding.GetMaterialPath())
                    if material_path == "":
                        result[write_idx] = None
                        write_idx += 1
                    else:
                        stage = get_current_stage()
                        material = UsdShade.Material(stage.GetPrimAtPath(material_path))
                        # getting the shader
                        shader_info = material.ComputeSurfaceSource()
                        if shader_info[0].GetPath().pathString != "":
                            shader = shader_info[0]
                        elif is_prim_path_valid(material_path + "/shader"):
                            shader_path = material_path + "/shader"
                            shader = UsdShade.Shader(get_prim_at_path(shader_path))
                        elif is_prim_path_valid(material_path + "/Shader"):
                            shader_path = material_path + "/Shader"
                            shader = UsdShade.Shader(get_prim_at_path(shader_path))
                        else:
                            carb.log_warn("the shader on xform prim {} is not supported".format(self._prim_paths[i]))
                            result[write_idx] = None
                            write_idx += 1
                            continue
                        implementation_source = shader.GetImplementationSource()
                        asset_sub_identifier = shader.GetPrim().GetAttribute("info:mdl:sourceAsset:subIdentifier").Get()
                        shader_id = shader.GetShaderId()
                        if implementation_source == "id" and shader_id == "UsdPreviewSurface":
                            from isaacsim.core.api.materials.preview_surface import PreviewSurface

                            self._applied_visual_materials[i] = PreviewSurface(prim_path=material_path, shader=shader)
                            result[write_idx] = self._applied_visual_materials[i]
                            write_idx += 1
                        elif asset_sub_identifier == "OmniGlass":
                            from isaacsim.core.api.materials.omni_glass import OmniGlass

                            self._applied_visual_materials[i] = OmniGlass(prim_path=material_path, shader=shader)
                            result[write_idx] = self._applied_visual_materials[i]
                            write_idx += 1
                        elif asset_sub_identifier == "OmniPBR":
                            from isaacsim.core.api.materials.omni_pbr import OmniPBR

                            self._applied_visual_materials[i] = OmniPBR(prim_path=material_path, shader=shader)
                            result[write_idx] = self._applied_visual_materials[i]
                            write_idx += 1
                        else:
                            carb.log_warn("the shader on xform prim {} is not supported".format(self._prim_paths[i]))
                            result[write_idx] = None
                            write_idx += 1
            return result
        else:
            raise Exception("prim view {} is not a valid view".format(self._regex_prim_paths))

    def is_visual_material_applied(
        self, indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None
    ) -> List[bool]:
        """Check if there is a visual material applied

        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                 to query. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).

        Returns:
            List[bool]: True if there is a visual material applied is applied to the corresponding prim in the view. False otherwise.

        Example:

        .. code-block:: python

            >>> # given a visual material that is applied only to the first and the last environment
            >>> prims.is_visual_material_applied()
            [True, False, False, False, True]
            >>>
            >>> # check for the first, middle and last of the 5 envs
            >>> prims.is_visual_material_applied(indices=np.array([0, 2, 4]))
            [True, False, True]
        """
        if self._is_valid:
            indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
            result = [None] * indices.shape[0]
            write_idx = 0
            indices = self._backend_utils.to_list(indices)
            for i in indices:
                if self._binding_apis[i] is None:
                    if self._prims[i].HasAPI(UsdShade.MaterialBindingAPI):
                        self._binding_apis[i] = UsdShade.MaterialBindingAPI(self._prims[i])
                    else:
                        self._binding_apis[i] = UsdShade.MaterialBindingAPI.Apply(self._prims[i])
                visual_binding = self._binding_apis[i].GetDirectBinding()
                material_path = str(visual_binding.GetMaterialPath())
                if material_path == "":
                    result[write_idx] = False
                    write_idx += 1
                else:
                    result[write_idx] = True
                    write_idx += 1
            return result
        else:
            raise Exception("prim view {} is not a valid view".format(self._regex_prim_paths))

    def get_world_poses(
        self, indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None, usd: bool = True
    ) -> Union[
        Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor], Tuple[wp.indexedarray, wp.indexedarray]
    ]:
        """Get the poses of the prims in the view with respect to the world's frame

        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indices to specify which prims
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
            >>> positions, orientations = prims.get_world_poses(indices=np.array([0, 2, 4]))
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
            if not usd:
                self._get_fabric_hierarchy().update_world_xforms()
                if not self._view_in_fabric_prepared:
                    self._prepare_view_in_fabric()
                if self._selection is None:
                    self._get_fabric_selection()
                world_matrix = wp.fabricarray(self._selection, "omni:fabric:worldMatrix")
                if indices is None:
                    indices = self._default_view_indices
                else:
                    indices = self._backend2warp(indices, dtype=wp.uint32)
                wp.launch(
                    fabric_utils.decompose_fabric_transformation_matrix_to_warp_arrays,
                    dim=(indices.shape[0]),
                    inputs=[
                        world_matrix,
                        self._fabric_data_dicts["world_position"],
                        self._fabric_data_dicts["world_orientation"],
                        None,
                        indices,
                        self._view_to_fabric,
                    ],
                    device=self._device,
                )
                self._fabric_data_valid["world_position"] = True
                self._fabric_data_valid["world_orientation"] = True
                return self._warp2backend(
                    wp.indexedarray(self._fabric_data_dicts["world_position"], indices=indices.view(wp.int32))
                ), self._warp2backend(
                    wp.indexedarray(self._fabric_data_dicts["world_orientation"], indices=indices.view(wp.int32))
                )
            else:
                indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
                positions = np.zeros((indices.shape[0], 3), dtype=np.float32)
                orientations = np.zeros((indices.shape[0], 4), dtype=np.float32)
                indices = self._backend_utils.to_list(indices)
                write_idx = 0
                for i in indices:
                    positions[write_idx], orientations[write_idx] = get_world_pose(self._prim_paths[i], fabric=False)
                    write_idx += 1
                positions = self._backend_utils.convert(positions, device=self._device, dtype="float32", indexed=True)
                orientations = self._backend_utils.convert(
                    orientations, device=self._device, dtype="float32", indexed=True
                )
                return positions, orientations
        else:
            raise Exception("prim view {} is not a valid view".format(self._regex_prim_paths))

    def set_world_poses(
        self,
        positions: Optional[Union[np.ndarray, torch.Tensor, wp.array]] = None,
        orientations: Optional[Union[np.ndarray, torch.Tensor, wp.array]] = None,
        indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None,
        usd: bool = True,
    ) -> None:
        """Set prim poses in the view with respect to the world's frame

        .. warning::

            This method will change (teleport) the prim poses immediately to the indicated value

        Args:
            positions (Optional[Union[np.ndarray, torch.Tensor, wp.array]], optional): positions in the world frame of the prims. shape is (M, 3).
                                                                             Defaults to None, which means left unchanged.
            orientations (Optional[Union[np.ndarray, torch.Tensor, wp.array]], optional): quaternion orientations in the world frame of the prims.
                                                                                quaternion is scalar-first (w, x, y, z). shape is (M, 4).
                                                                                Defaults to None, which means left unchanged.
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indices to specify which prims
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
            >>> prims.set_world_poses(positions, orientations, indices=np.array([0, 2, 4]))
        """
        if self._is_valid:
            if not usd:
                fabric_hierarchy = self._get_fabric_hierarchy()
                fabric_hierarchy.update_world_xforms()
                if not self._view_in_fabric_prepared:
                    self._prepare_view_in_fabric()
                if indices is None:
                    indices = self._default_view_indices
                else:
                    indices = self._backend2warp(indices, dtype=wp.uint32)
                if positions is not None:
                    positions = self._backend2warp(positions).numpy()
                if orientations is not None:
                    orientations = self._backend2warp(orientations).numpy()
                for i, index in enumerate(indices.numpy()):
                    path = usdrt.Sdf.Path(self._prim_paths[index])
                    matrix = fabric_hierarchy.get_world_xform(path)
                    if positions is not None:
                        matrix.SetTranslateOnly(usdrt.Gf.Vec3d(*positions[i]))
                    if orientations is not None:
                        matrix.SetRotateOnly(usdrt.Gf.Quatd(*orientations[i]))
                        scaling_matrix = (
                            usdrt.Gf.Matrix4d().SetIdentity().SetScale(usdrt.Gf.Transform(matrix).GetScale())
                        )
                        matrix = scaling_matrix * matrix
                    fabric_hierarchy.set_world_xform(path, matrix)
            else:
                if positions is None or orientations is None:
                    current_positions, current_orientations = self.get_world_poses(indices=indices)
                    if positions is None:
                        positions = current_positions
                    if orientations is None:
                        orientations = current_orientations
                indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
                parent_transforms = np.zeros(shape=(indices.shape[0], 4, 4), dtype="float32")
                indices = self._backend_utils.to_list(indices)
                write_idx = 0
                for i in indices:
                    parent_transforms[write_idx] = np.array(
                        UsdGeom.Xformable(get_prim_parent(self._prims[i])).ComputeLocalToWorldTransform(
                            Usd.TimeCode.Default()
                        ),
                        dtype="float32",
                    )
                    write_idx += 1
                parent_transforms = self._backend_utils.convert(parent_transforms, dtype="float32", device=self._device)
                calculated_translations, calculated_orientations = self._backend_utils.get_local_from_world(
                    parent_transforms, positions, orientations, self._device
                )
                XFormPrim.set_local_poses(
                    self, translations=calculated_translations, orientations=calculated_orientations, indices=indices
                )
            return
        else:
            raise Exception("prim view {} is not a valid view".format(self._regex_prim_paths))

    def get_local_poses(
        self, indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None
    ) -> Union[
        Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor], Tuple[wp.indexedarray, wp.indexedarray]
    ]:
        """Get prim poses in the view with respect to the local frame (the prim's parent frame)

        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indices to specify which prims
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
            >>> positions, orientations = prims.get_local_poses(indices=np.array([0, 2, 4]))
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
            indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
            translations = np.zeros(shape=(indices.shape[0], 3), dtype="float32")
            orientations = np.zeros(shape=(indices.shape[0], 4), dtype="float32")
            write_idx = 0
            indices = self._backend_utils.to_list(indices)
            for i in indices:
                translations[write_idx], orientations[write_idx] = get_local_pose(self._prim_paths[i])
                write_idx += 1
            translations = self._backend_utils.convert(translations, dtype="float32", device=self._device, indexed=True)
            orientations = self._backend_utils.convert(orientations, dtype="float32", device=self._device, indexed=True)
            return translations, orientations
        else:
            raise Exception("prim view {} is not a valid view".format(self._regex_prim_paths))

    def set_local_poses(
        self,
        translations: Optional[Union[np.ndarray, torch.Tensor, wp.array]] = None,
        orientations: Optional[Union[np.ndarray, torch.Tensor, wp.array]] = None,
        indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None,
    ) -> None:
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
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indices to specify which prims
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
            >>> prims.set_local_poses(positions, orientations, indices=np.array([0, 2, 4]))
        """
        if self._is_valid:
            indices = self._backend_utils.to_list(
                self._backend_utils.resolve_indices(indices, self.count, self._device)
            )
            if translations is not None:
                translations = self._backend_utils.to_list(translations)
                write_idx = 0
                for i in indices:
                    properties = self._prims[i].GetPropertyNames()
                    translation = Gf.Vec3d(*translations[write_idx])
                    if "xformOp:translate" not in properties:
                        carb.log_error(
                            "Translate property needs to be set for {} before setting its position".format(self.name)
                        )
                    xform_op = self._prims[i].GetAttribute("xformOp:translate")
                    xform_op.Set(translation)
                    write_idx += 1
            if orientations is not None:
                orientations = self._backend_utils.to_list(orientations)
                write_idx = 0
                for i in indices:
                    properties = self._prims[i].GetPropertyNames()
                    if "xformOp:orient" not in properties:
                        carb.log_error(
                            "Orient property needs to be set for {} before setting its orientation".format(self.name)
                        )
                    xform_op = self._prims[i].GetAttribute("xformOp:orient")
                    if xform_op.GetTypeName() == "quatf":
                        rotq = Gf.Quatf(*orientations[write_idx])
                    else:
                        rotq = Gf.Quatd(*orientations[write_idx])
                    xform_op.Set(rotq)
                    write_idx += 1
            return
        else:
            raise Exception("prim view {} is not a valid view".format(self._regex_prim_paths))

    def get_world_scales(
        self, indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None
    ) -> Union[np.ndarray, torch.Tensor, wp.indexedarray]:
        """Get prim scales in the view with respect to the world's frame

        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                 to query. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).

        Returns:
            Union[np.ndarray, torch.Tensor, wp.indexedarray]: scales applied to the prim's dimensions in the world frame. shape is (M, 3).

        Example:

        .. code-block:: python

            >>> # get all prims scales with respect to the world's frame.
            >>> # Returned shape is (5, 3) for the example: 5 envs
            >>> prims.get_world_scales()
            [[1. 1. 1.]
             [1. 1. 1.]
             [1. 1. 1.]
             [1. 1. 1.]
             [1. 1. 1.]]
            >>>
            >>> # get only the prims scales with respect to the world's frame for the first, middle and last of the 5 envs.
            >>> # Returned shape is (3, 3) for the example: 3 envs selected
            >>> prims.get_world_scales(indices=np.array([0, 2, 4]))
            [[1. 1. 1.]
             [1. 1. 1.]
             [1. 1. 1.]]
        """
        if self._is_valid:
            indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
            scales = np.zeros(shape=(indices.shape[0], 3), dtype="float32")
            write_idx = 0
            indices = self._backend_utils.to_list(indices)
            for i in indices:
                prim_tf = UsdGeom.Xformable(self._prims[i]).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                transform = Gf.Transform()
                transform.SetMatrix(prim_tf)
                scales[write_idx] = np.array(transform.GetScale(), dtype="float32")
                write_idx += 1
            scales = self._backend_utils.convert(scales, dtype="float32", device=self._device, indexed=True)
            return scales
        else:
            raise Exception("prim view {} is not a valid view".format(self._regex_prim_paths))

    def set_local_scales(
        self,
        scales: Optional[Union[np.ndarray, torch.Tensor, wp.array]],
        indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None,
    ) -> None:
        """Set prim scales in the view with respect to the local frame (the prim's parent frame)

        Args:
            scales (Optional[Union[np.ndarray, torch.Tensor, wp.array]]): scales to be applied to the prim's dimensions in the view.
                                                                shape is (M, 3).
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indices to specify which prims
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
            >>> prims.set_local_scales(scales, indices=np.array([0, 2, 4]))
        """
        if self._is_valid:
            indices = self._backend_utils.to_list(
                self._backend_utils.resolve_indices(indices, self.count, self._device)
            )
            read_idx = 0
            scales = self._backend_utils.to_list(scales)
            for i in indices:
                scale = Gf.Vec3d(*scales[read_idx])
                properties = self._prims[i].GetPropertyNames()
                if "xformOp:scale" not in properties:
                    carb.log_error("Scale property needs to be set for {} before setting its scale".format(self.name))
                xform_op = self._prims[i].GetAttribute("xformOp:scale")
                xform_op.Set(scale)
                read_idx += 1
            return
        else:
            raise Exception("prim view {} is not a valid view".format(self._regex_prim_paths))

    def get_local_scales(
        self, indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None
    ) -> Union[np.ndarray, torch.Tensor, wp.indexedarray]:
        """Get prim scales in the view with respect to the local frame (the parent's frame).

        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indices to specify which prims
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
            >>> prims.get_local_scales(indices=np.array([0, 2, 4]))
            [[1. 1. 1.]
             [1. 1. 1.]
             [1. 1. 1.]]
        """
        if self._is_valid:
            indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
            scales = np.zeros(shape=(indices.shape[0], 3), dtype="float32")
            write_idx = 0
            indices = self._backend_utils.to_list(indices)
            for i in indices:
                scales[write_idx] = np.array(self._prims[i].GetAttribute("xformOp:scale").Get(), dtype="float32")
                write_idx += 1
            scales = self._backend_utils.convert(scales, dtype="float32", device=self._device, indexed=True)
            return scales
        else:
            raise Exception("prim view {} is not a valid view".format(self._regex_prim_paths))

    def _get_fabric_selection(self) -> None:
        self._selection = self._usdrt_stage.SelectPrims(
            require_attrs=[
                (usdrt.Sdf.ValueTypeNames.UInt, self._view_index_attr, usdrt.Usd.Access.Read),
                (usdrt.Sdf.ValueTypeNames.Matrix4d, "omni:fabric:worldMatrix", usdrt.Usd.Access.Read),
            ],
            device=self._device,
        )
        self._fabric_to_view = wp.fabricarray(self._selection, self._view_index_attr)
        wp.launch(
            fabric_utils.set_view_to_fabric_array,
            dim=(self._fabric_to_view.shape[0]),
            inputs=[self._fabric_to_view, self._view_to_fabric],
            device=self._device,
        )

    def _create_fabric_view_indices(self) -> None:
        for i in range(self.count):
            prim = self._usdrt_stage.GetPrimAtPath(self._prim_paths[i])
            prim.CreateAttribute(self._view_index_attr, usdrt.Sdf.ValueTypeNames.UInt, True)
            prim.GetAttribute(self._view_index_attr).Set(i)
            xformable_prim = usdrt.Rt.Xformable(prim)
            if not xformable_prim.HasWorldXform():
                xformable_prim.SetWorldXformFromUsd()

    def _reset_fabric_selection(self, dt) -> None:
        self._selection = None
        for data_tensor_name in self._fabric_data_valid.keys():
            self._fabric_data_valid[data_tensor_name] = False

    def _warp2backend(self, data) -> Union[wp.indexedarray, torch.Tensor, np.ndarray]:
        if self._backend == "warp":
            return data
        elif self._backend == "torch":
            return interops_utils.warp2torch(data.data)[interops_utils.warp2torch(data.indices[0])]
        elif self._backend == "numpy":
            return interops_utils.warp2numpy(data.data)[interops_utils.warp2numpy(data.indices[0])]
        else:
            raise Exception(
                "utils to convert warp arrays to the specified backend doesn't exist or the data passed is not using the backend specified"
            )

    def _backend2warp(self, data, dtype=None) -> Union[wp.array, torch.Tensor, np.ndarray]:
        if isinstance(data, list):
            result = wp.array(data, dtype=wp.uint32, device=self._device).to(self._device)
        elif self._backend == "warp":
            result = data.to(self._device)
        elif self._backend == "torch":
            result = interops_utils.torch2warp(data).to(self._device)
        elif self._backend == "numpy":
            result = interops_utils.numpy2warp(data).to(self._device)
        elif self._backend == "numpy":
            result = interops_utils.numpy2torch(data).to(self._device)
        else:
            raise Exception("utils to convert the specified backend arrays to warp doesn't exist")
        if dtype is not None:
            return result.view(dtype)
        else:
            return result

    def _prepare_view_in_fabric(self):
        self._create_fabric_view_indices()
        self._view_to_fabric = wp.zeros([self.count], dtype=wp.uint32, device=self._device)
        self._default_view_indices = wp.zeros([self.count], dtype=wp.uint32, device=self._device)
        wp.launch(
            kernel=fabric_utils.arange_k,
            dim=self.count,
            inputs=[self._default_view_indices],
            device=self._device,
        )
        self._get_fabric_selection()
        self._fabric_data_dicts["world_position"] = wp.zeros([self.count, 3], dtype=wp.float32, device=self._device)
        self._fabric_data_valid["world_position"] = False
        self._fabric_data_dicts["world_orientation"] = wp.zeros([self.count, 4], dtype=wp.float32, device=self._device)
        self._fabric_data_valid["world_orientation"] = False
        self._callbacks.append(
            SimulationManager.register_callback(self._reset_fabric_selection, event=IsaacEvents.POST_PHYSICS_STEP)
        )
        self._view_in_fabric_prepared = True

    def _get_fabric_hierarchy(self):
        if self._fabric_hierarchy is None:
            self._fabric_hierarchy = usdrt.hierarchy.IFabricHierarchy().get_fabric_hierarchy(
                self._usdrt_stage.GetFabricId(), self._usdrt_stage.GetStageIdAsStageId()
            )
        return self._fabric_hierarchy

    def _on_post_reset(self, event) -> None:
        if not self._non_root_link:
            self.set_world_poses(self._default_state.positions, self._default_state.orientations)

    def _set_xform_properties(self) -> None:
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


import numpy as np
import usdrt
from isaacsim.core.utils.prims import (
    get_prim_at_path,
    get_prim_attribute_names,
    get_prim_attribute_value,
    get_prim_parent,
    get_prim_path,
    is_prim_path_valid,
)
from pxr import Gf, Usd, UsdGeom
from scipy.spatial.transform import Rotation


def clear_xform_ops(prim: Usd.Prim):
    """Remove all xform ops from input prim.

    Args:
        prim (Usd.Prim): The input USD prim.
    """
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    # Remove any authored transform properties
    authored_prop_names = prim.GetAuthoredPropertyNames()
    for prop_name in authored_prop_names:
        if prop_name.startswith("xformOp:"):
            prim.RemoveProperty(prop_name)


def reset_and_set_xform_ops(
    prim: Usd.Prim, translation: Gf.Vec3d, orientation: Gf.Quatd, scale: Gf.Vec3d = Gf.Vec3d([1.0, 1.0, 1.0])
):
    """Reset xform ops to isaac sim defaults, and set their values

    Args:
        prim (Usd.Prim): Prim to reset
        translation (Gf.Vec3d): translation to set
        orientation (Gf.Quatd): orientation to set
        scale (Gf.Vec3d, optional): scale to set. Defaults to Gf.Vec3d([1.0, 1.0, 1.0]).
    """
    xformable = UsdGeom.Xformable(prim)
    clear_xform_ops(prim)

    xform_op_scale = xformable.AddXformOp(UsdGeom.XformOp.TypeScale, UsdGeom.XformOp.PrecisionDouble, "")
    xform_op_scale.Set(scale)

    xform_op_tranlsate = xformable.AddXformOp(UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.PrecisionDouble, "")
    xform_op_tranlsate.Set(translation)

    xform_op_rot = xformable.AddXformOp(UsdGeom.XformOp.TypeOrient, UsdGeom.XformOp.PrecisionDouble, "")
    xform_op_rot.Set(orientation)

    xformable.SetXformOpOrder([xform_op_tranlsate, xform_op_rot, xform_op_scale])


def reset_xform_ops(prim: Usd.Prim):
    """Reset xform ops for a prim to isaac sim defaults,

    Args:
        prim (Usd.Prim): Prim to reset xform ops on
    """
    properties = prim.GetPropertyNames()
    xformable = UsdGeom.Xformable(prim)
    # get current position and orientation
    T_p_w = xformable.ComputeParentToWorldTransform(Usd.TimeCode.Default())
    T_l_w = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    T_l_p = Gf.Transform()
    T_l_p.SetMatrix(Gf.Matrix4d(np.matmul(T_l_w, np.linalg.inv(T_p_w)).tolist()))
    current_translation = T_l_p.GetTranslation()
    current_orientation = T_l_p.GetRotation().GetQuat()

    reset_and_set_xform_ops(current_translation, current_orientation)


def _get_world_pose_transform_w_scale(prim_path, fabric=False):
    # This will return a transformation matrix with translation as the last row and scale included
    if not is_prim_path_valid(prim_path, fabric=False):
        raise Exception("Prim path is not valid")

    def _get_from_usd_prim(prim_path):
        usd_prim = get_prim_at_path(prim_path=prim_path, fabric=False)
        local_transform = usdrt.Gf.Matrix4d(UsdGeom.Xformable(usd_prim).GetLocalTransformation(Usd.TimeCode.Default()))
        parent_prim = get_prim_parent(get_prim_at_path(prim_path=prim_path, fabric=False))
        parent_world_transform = usdrt.Gf.Matrix4d(1.0)
        if parent_prim:
            parent_world_transform = _get_world_pose_transform_w_scale(get_prim_path(parent_prim), fabric=False)
        return local_transform * parent_world_transform

    if not fabric:
        return _get_from_usd_prim(prim_path)

    fabric_prim = get_prim_at_path(prim_path=prim_path, fabric=True)
    xformable_prim = usdrt.Rt.Xformable(fabric_prim)
    if xformable_prim.HasWorldXform():
        world_pos_attr = xformable_prim.GetWorldPositionAttr()
        if not world_pos_attr.IsValid():
            world_pos = usdrt.Gf.Vec3d(0)
        else:
            world_pos = world_pos_attr.Get(usdrt.Usd.TimeCode.Default())
        world_orientation_attr = xformable_prim.GetWorldOrientationAttr()
        if not world_orientation_attr.IsValid():
            world_orientation = usdrt.Gf.Quatf(1)
        else:
            world_orientation = world_orientation_attr.Get(usdrt.Usd.TimeCode.Default())
        world_scale_attr = xformable_prim.GetWorldScaleAttr()
        if not world_scale_attr.IsValid():
            world_scale = usdrt.Gf.Vec3d(1)
        else:
            world_scale = world_scale_attr.Get(usdrt.Usd.TimeCode.Default())
        scale = usdrt.Gf.Matrix4d()
        rot = usdrt.Gf.Matrix4d()
        scale.SetScale(usdrt.Gf.Vec3d(world_scale))
        rot.SetRotate(usdrt.Gf.Quatd(world_orientation))
        result = scale * rot
        result.SetTranslateOnly(world_pos)
        return result
    elif xformable_prim.HasLocalXform():
        local_transform = xformable_prim.GetLocalMatrixAttr().Get(usdrt.Usd.TimeCode.Default())
        parent_prim = get_prim_parent(get_prim_at_path(prim_path=prim_path, fabric=False))
        parent_world_transform = usdrt.Gf.Matrix4d(1.0)
        if parent_prim:
            parent_world_transform = _get_world_pose_transform_w_scale(get_prim_path(parent_prim))
        return local_transform * parent_world_transform
    else:
        return _get_from_usd_prim(prim_path)


def get_local_pose(prim_path):
    if not is_prim_path_valid(prim_path, fabric=False):
        raise Exception("Prim path is not valid")
    fabric_prim = get_prim_at_path(prim_path=prim_path, fabric=True)
    xformable_prim = usdrt.Rt.Xformable(fabric_prim)
    if xformable_prim.HasWorldXform():
        world_pos = xformable_prim.GetWorldPositionAttr().Get(usdrt.Usd.TimeCode.Default())
        world_orientation = xformable_prim.GetWorldOrientationAttr().Get(usdrt.Usd.TimeCode.Default())
        prim_l_w_transform = usdrt.Gf.Matrix4d()
        prim_l_w_transform.SetRotate(usdrt.Gf.Quatd(world_orientation))
        prim_l_w_transform.SetTranslateOnly(world_pos)
        parent_prim = get_prim_parent(get_prim_at_path(prim_path=prim_path, fabric=False))
        parent_l_w_transform = usdrt.Gf.Matrix4d(1.0)
        if parent_prim:
            parent_l_w_transform = _get_world_pose_transform_w_scale(get_prim_path(parent_prim))
            parent_l_w_transform.Orthonormalize()
            parent_l_w_transform = np.array(parent_l_w_transform)
        result_transform = np.matmul(prim_l_w_transform, np.linalg.inv(parent_l_w_transform))
        result_transform = np.transpose(result_transform)
        r = Rotation.from_matrix(result_transform[:3, :3])
        return result_transform[:3, 3], r.as_quat()[[3, 0, 1, 2]]

    elif xformable_prim.HasLocalXform():
        local_transform = xformable_prim.GetLocalMatrixAttr().Get(usdrt.Usd.TimeCode.Default())
        local_transform.Orthonormalize()
        return (
            np.array(local_transform.ExtractTranslation()),
            np.array(local_transform.ExtractRotationQuat())[[3, 0, 1, 2]],
        )
    else:
        usd_prim = get_prim_at_path(prim_path=prim_path, fabric=False)
        local_transform = usdrt.Gf.Matrix4d(UsdGeom.Xformable(usd_prim).GetLocalTransformation(Usd.TimeCode.Default()))
        local_transform.Orthonormalize()
        return (
            np.array(local_transform.ExtractTranslation()),
            np.array(local_transform.ExtractRotationQuat())[[3, 0, 1, 2]],
        )


def get_world_pose(prim_path, fabric=False):
    result_transform = _get_world_pose_transform_w_scale(prim_path, fabric)
    result_transform.Orthonormalize()
    result_transform = np.transpose(result_transform)
    r = Rotation.from_matrix(result_transform[:3, :3])
    return result_transform[:3, 3], r.as_quat()[[3, 0, 1, 2]]