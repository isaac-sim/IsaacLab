# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from typing import Optional, Sequence

import carb
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.prims import RigidPrim, RigidPrimView

import omni.isaac.orbit.utils.kit as kit_utils

from .rigid_object_cfg import RigidObjectCfg
from .rigid_object_data import RigidObjectData


class RigidObject:
    """Class for handling rigid objects.

    Rigid objects are spawned from USD files and are encapsulated by a single root prim.
    The root prim is used to apply physics material to the rigid body.

    This class wraps around :class:`RigidPrimView` class from Isaac Sim to support the following:

    * Configuring using a single dataclass (struct).
    * Applying physics material to the rigid body.
    * Handling different rigid body views.
    * Storing data related to the rigid object.

    """

    cfg: RigidObjectCfg
    """Configuration class for the rigid object."""
    objects: RigidPrimView
    """Rigid prim view for the rigid object."""

    def __init__(self, cfg: RigidObjectCfg):
        """Initialize the rigid object.

        Args:
            cfg (RigidObjectCfg): An instance of the configuration class.
        """
        # store inputs
        self.cfg = cfg
        # container for data access
        self._data = RigidObjectData()
        # buffer variables (filled during spawn and initialize)
        self._spawn_prim_path: str = None

    """
    Properties
    """

    @property
    def count(self) -> int:
        """Number of prims encapsulated."""
        return self.objects.count

    @property
    def device(self) -> str:
        """Memory device for computation."""
        return self.objects._device

    @property
    def data(self) -> RigidObjectData:
        """Data related to articulation."""
        return self._data

    """
    Operations.
    """

    def spawn(self, prim_path: str, translation: Sequence[float] = None, orientation: Sequence[float] = None):
        """Spawn a rigid object into the stage (loaded from its USD file).

        Note:
            If inputs `translation` or `orientation` are not :obj:`None`, then they override the initial root
            state specified through the configuration class at spawning.

        Args:
            prim_path (str): The prim path for spawning object at.
            translation (Sequence[float], optional): The local position of prim from its parent. Defaults to None.
            orientation (Sequence[float], optional): The local rotation (as quaternion `(w, x, y, z)`
                of the prim from its parent. Defaults to None.
        """
        # use default arguments
        if translation is None:
            translation = self.cfg.init_state.pos
        if orientation is None:
            orientation = self.cfg.init_state.rot

        # -- save prim path for later
        self._spawn_prim_path = prim_path
        # -- spawn asset if it doesn't exist.
        if not prim_utils.is_prim_path_valid(prim_path):
            # add prim as reference to stage
            prim_utils.create_prim(
                self._spawn_prim_path,
                usd_path=self.cfg.meta_info.usd_path,
                translation=translation,
                orientation=orientation,
                scale=self.cfg.meta_info.scale,
            )
        else:
            carb.log_warn(f"A prim already exists at prim path: '{prim_path}'. Skipping...")

        # apply rigid body properties API
        RigidPrim(prim_path=prim_path)
        # -- set rigid body properties
        kit_utils.set_nested_rigid_body_properties(prim_path, **self.cfg.rigid_props.to_dict())
        # apply collision properties
        kit_utils.set_nested_collision_properties(prim_path, **self.cfg.collision_props.to_dict())
        # create physics material
        if self.cfg.physics_material is not None:
            # -- resolve material path
            material_path = self.cfg.physics_material.prim_path
            if not material_path.startswith("/"):
                material_path = prim_path + "/" + prim_path
            # -- create physics material
            material = PhysicsMaterial(
                prim_path=material_path,
                static_friction=self.cfg.physics_material.static_friction,
                dynamic_friction=self.cfg.physics_material.dynamic_friction,
                restitution=self.cfg.physics_material.restitution,
            )
            # -- apply physics material
            kit_utils.apply_nested_physics_material(prim_path, material.prim_path)

    def initialize(self, prim_paths_expr: Optional[str] = None):
        """Initializes the PhysX handles and internal buffers.

        Note:
            PhysX handles are only enabled once the simulator starts playing. Hence, this function needs to be
            called whenever the simulator "plays" from a "stop" state.

        Args:
            prim_paths_expr (Optional[str], optional): The prim path expression for the prims. Defaults to None.

        Raises:
            RuntimeError: When input `prim_paths_expr` is :obj:`None`, the method defaults to using the last
                prim path set when calling the :meth:`spawn()` function. In case, the object was not spawned
                and no valid `prim_paths_expr` is provided, the function throws an error.
        """
        # default prim path if not cloned
        if prim_paths_expr is None:
            if self._is_spawned is not None:
                self._prim_paths_expr = self._spawn_prim_path
            else:
                raise RuntimeError(
                    "Initialize the object failed! Please provide a valid argument for `prim_paths_expr`."
                )
        else:
            self._prim_paths_expr = prim_paths_expr
        # create handles
        # -- object views
        self.objects = RigidPrimView(prim_paths_expr, reset_xform_properties=False)
        self.objects.initialize()
        # set the default state
        self.objects.post_reset()
        # set properties over all instances
        # -- meta-information
        self._process_info_cfg()
        # create buffers
        self._create_buffers()

    def reset_buffers(self, env_ids: Optional[Sequence[int]] = None):
        """Resets all internal buffers.

        Args:
            env_ids (Optional[Sequence[int]], optional): The indices of the object to reset.
                Defaults to None (all instances).
        """
        pass

    def update_buffers(self, dt: float = None):
        """Update the internal buffers.

        The time step ``dt`` is used to compute numerical derivatives of quantities such as joint
        accelerations which are not provided by the simulator.

        Args:
            dt (float, optional): The amount of time passed from last `update_buffers` call. Defaults to None.
        """
        # frame states
        position_w, quat_w = self.objects.get_world_poses(indices=self._ALL_INDICES, clone=False)
        self._data.root_state_w[:, 0:3] = position_w
        self._data.root_state_w[:, 3:7] = quat_w
        self._data.root_state_w[:, 7:] = self.objects.get_velocities(indices=self._ALL_INDICES, clone=False)

    """
    Operations - State.
    """

    def set_root_state(self, root_states: torch.Tensor, env_ids: Optional[Sequence[int]] = None):
        """Sets the root state (pose and velocity) of the actor over selected environment indices.

        Args:
            root_states (torch.Tensor): Input root state for the actor, shape: (len(env_ids), 13).
            env_ids (Optional[Sequence[int]]): Environment indices.
                If :obj:`None`, then all indices are used.
        """
        # resolve all indices
        if env_ids is None:
            env_ids = self._ALL_INDICES
        # set into simulation
        self.objects.set_world_poses(root_states[:, 0:3], root_states[:, 3:7], indices=env_ids)
        self.objects.set_velocities(root_states[:, 7:], indices=env_ids)

        # TODO: Move these to reset_buffers call.
        # note: we need to do this here since tensors are not set into simulation until step.
        # set into internal buffers
        self._data.root_state_w[env_ids] = root_states.clone()

    def get_default_root_state(self, env_ids: Optional[Sequence[int]] = None, clone=True) -> torch.Tensor:
        """Returns the default/initial root state of actor.

        Args:
            env_ids (Optional[Sequence[int]], optional): Environment indices.
                Defaults to None (all environment indices).
            clone (bool, optional): Whether to return a copy or not. Defaults to True.

        Returns:
            torch.Tensor: The default/initial root state of the actor, shape: (len(env_ids), 13).
        """
        # use ellipses object to skip initial indices.
        if env_ids is None:
            env_ids = ...
        # return copy
        if clone:
            return torch.clone(self._default_root_states[env_ids])
        else:
            return self._default_root_states[env_ids]

    """
    Internal helper.
    """

    def _process_info_cfg(self) -> None:
        """Post processing of configuration parameters."""
        # default state
        # -- root state
        # note: we cast to tuple to avoid torch/numpy type mismatch.
        default_root_state = (
            tuple(self.cfg.init_state.pos)
            + tuple(self.cfg.init_state.rot)
            + tuple(self.cfg.init_state.lin_vel)
            + tuple(self.cfg.init_state.ang_vel)
        )
        self._default_root_states = torch.tensor(default_root_state, dtype=torch.float, device=self.device)
        self._default_root_states = self._default_root_states.repeat(self.count, 1)

    def _create_buffers(self):
        """Create buffers for storing data."""
        # constants
        self._ALL_INDICES = torch.arange(self.count, dtype=torch.long, device=self.device)

        # -- frame states
        self._data.root_state_w = torch.zeros(self.count, 13, dtype=torch.float, device=self.device)
