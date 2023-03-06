# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import re
import torch
from typing import Dict, List, Optional, Sequence, Tuple

import carb
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView

import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.orbit.actuators.group import *  # noqa: F403, F401
from omni.isaac.orbit.utils.math import quat_rotate_inverse, sample_uniform, subtract_frame_transforms

from .articulated_object_cfg import ArticulatedObjectCfg
from .articulated_object_data import ArticulatedObjectData


class ArticulatedObject:
    """Class for handling articulated objects.

    This class wraps around :class:`ArticulationView` and :class:`RigidPrimView` classes
    from Isaac Sim to support the following:

    * Configuring the object using a single dataclass (struct).
    * Applying settings to the articulated object from the configuration class.
    * Handling different rigid body views inside the articulated object.
    * Storing data related to the articulated object.

    """

    cfg: ArticulatedObjectCfg
    """Configuration class for the articulated object."""
    articulations: ArticulationView = None
    """Articulation view for the articulated object."""
    site_bodies: Dict[str, RigidPrimView] = None
    """Rigid body view for sites in the articulated object.

    Dictionary with keys as the site names and values as the corresponding rigid body view
    in the articulated object.
    """

    def __init__(self, cfg: ArticulatedObjectCfg):
        """Initialize the articulated object.

        Args:
            cfg (ArticulatedObjectCfg): An instance of the configuration class.
        """
        # store inputs
        self.cfg = cfg
        # container for data access
        self._data = ArticulatedObjectData()
        # buffer variables (filled during spawn and initialize)
        self._spawn_prim_path: str = None

    """
    Properties
    """

    @property
    def count(self) -> int:
        """Number of prims encapsulated."""
        return self.articulations.count

    @property
    def device(self) -> str:
        """Memory device for computation."""
        return self.articulations._device

    @property
    def body_names(self) -> List[str]:
        """Ordered names of links/bodies in articulation."""
        return self.articulations.body_names

    @property
    def dof_names(self) -> List[str]:
        """Ordered names of DOFs in articulation."""
        return self.articulations.dof_names

    @property
    def num_dof(self) -> int:
        """Total number of DOFs in articulation."""
        return self.articulations.num_dof

    @property
    def data(self) -> ArticulatedObjectData:
        """Data related to articulation."""
        return self._data

    """
    Operations.
    """

    def spawn(self, prim_path: str, translation: Sequence[float] = None, orientation: Sequence[float] = None):
        """Spawn an articulated object into the stage (loaded from its USD file).

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
        scale = self.cfg.meta_info.scale

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
                scale=scale,
            )
        else:
            carb.log_warn(f"A prim already exists at prim path: '{prim_path}'. Skipping...")

        # TODO: What if prim already exists in the stage and spawn isn't called?
        # apply rigid body properties
        kit_utils.set_nested_rigid_body_properties(prim_path, **self.cfg.rigid_props.to_dict())
        # apply collision properties
        kit_utils.set_nested_collision_properties(prim_path, **self.cfg.collision_props.to_dict())
        # articulation root settings
        kit_utils.set_articulation_properties(prim_path, **self.cfg.articulation_props.to_dict())

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
                    "Initialize the articulated object failed! Please provide a valid argument for `prim_paths_expr`."
                )
        else:
            self._prim_paths_expr = prim_paths_expr
        # create handles
        # -- articulation views
        self.articulations = ArticulationView(prim_paths_expr, reset_xform_properties=False)
        self.articulations.initialize()
        # set the default state
        self.articulations.post_reset()
        # set properties over all instances
        # -- meta-information
        self._process_info_cfg()
        # create buffers
        self._create_buffers()
        # tracked sites
        if self.sites_indices is not None:
            self.site_bodies: Dict[str, RigidPrimView] = dict()
            for name in self.sites_indices:
                # create rigid body view to track
                site_body = RigidPrimView(prim_paths_expr=f"{prim_paths_expr}/{name}", reset_xform_properties=False)
                site_body.initialize()
                # add to list
                self.site_bodies[name] = site_body
        else:
            self.site_bodies = None

    def reset_buffers(self, env_ids: Optional[Sequence[int]] = None):
        """Resets all internal buffers.

        Args:
            env_ids (Optional[Sequence[int]], optional): The indices of the object to reset.
                Defaults to None (all instances).
        """
        # use ellipses object to skip initial indices.
        if env_ids is None:
            env_ids = ...
        # reset history
        self._previous_dof_vel[env_ids] = 0

    def update_buffers(self, dt: float):
        """Update the internal buffers.

        Args:
            dt (float): The amount of time passed from last `update_buffers` call.

                This is used to compute numerical derivatives of quantities such as joint accelerations
                which are not provided by the simulator.
        """
        # frame states
        # -- root frame in world
        position_w, quat_w = self.articulations.get_world_poses(indices=self._ALL_INDICES, clone=False)
        self._data.root_state_w[:, 0:3] = position_w
        self._data.root_state_w[:, 3:7] = quat_w
        self._data.root_state_w[:, 7:] = self.articulations.get_velocities(indices=self._ALL_INDICES, clone=False)
        # -- tracked sites states
        if self.site_bodies is not None:
            for index, body in enumerate(self.site_bodies.values()):
                # world frame
                position_w, quat_w = body.get_world_poses(indices=self._ALL_INDICES, clone=False)
                self._data.sites_state_w[:, index, 0:3] = position_w
                self._data.sites_state_w[:, index, 3:7] = quat_w
                self._data.sites_state_w[:, index, 7:] = body.get_velocities(indices=self._ALL_INDICES, clone=False)
                # base frame
                position_b, quat_b = subtract_frame_transforms(
                    self._data.root_state_w[:, 0:3],
                    self._data.root_state_w[:, 3:7],
                    self._data.sites_state_w[:, index, 0:3],
                    self._data.sites_state_w[:, index, 3:7],
                )
                self._data.sites_state_b[:, index, 0:3] = position_b
                self._data.sites_state_b[:, index, 3:7] = quat_b
                self._data.sites_state_b[:, index, 7:10] = quat_rotate_inverse(
                    self._data.root_quat_w, self._data.sites_state_w[:, index, 7:10]
                )
                self._data.sites_state_b[:, index, 10:13] = quat_rotate_inverse(
                    self._data.root_quat_w, self._data.sites_state_w[:, index, 10:13]
                )

        # dof states
        self._data.dof_pos[:] = self.articulations.get_joint_positions(indices=self._ALL_INDICES, clone=False)
        self._data.dof_vel[:] = self.articulations.get_joint_velocities(indices=self._ALL_INDICES, clone=False)
        self._data.dof_acc[:] = (self._data.dof_vel - self._previous_dof_vel) / dt
        # update history buffers
        self._previous_dof_vel[:] = self._data.dof_vel[:]

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
        self.articulations.set_world_poses(root_states[:, 0:3], root_states[:, 3:7], indices=env_ids)
        self.articulations.set_velocities(root_states[:, 7:], indices=env_ids)

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

    def set_dof_state(self, dof_pos: torch.Tensor, dof_vel: torch.Tensor, env_ids: Optional[Sequence[int]] = None):
        """Sets the DOF state (position and velocity) of the actor over selected environment indices.

        Args:
            dof_pos (torch.Tensor): Input DOF position for the actor, shape: (len(env_ids), 1).
            dof_vel (torch.Tensor): Input DOF velocity for the actor, shape: (len(env_ids), 1).
            env_ids (torch.Tensor): Environment indices.
                If :obj:`None`, then all indices are used.
        """
        # resolve all indices
        if env_ids is None:
            env_ids = self._ALL_INDICES
        # set into simulation
        self.articulations.set_joint_positions(dof_pos, indices=env_ids)
        self.articulations.set_joint_velocities(dof_vel, indices=env_ids)

        # TODO: Move these to reset_buffers call.
        # note: we need to do this here since tensors are not set into simulation until step.
        # set into internal buffers
        self._data.dof_pos[env_ids] = dof_pos.clone()
        self._data.dof_vel[env_ids] = dof_vel.clone()
        self._data.dof_acc[env_ids] = 0.0

    def get_default_dof_state(
        self, env_ids: Optional[Sequence[int]] = None, clone=True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the default/initial DOF state (position and velocity) of actor.

        Args:
            env_ids (Optional[Sequence[int]], optional): Environment indices.
                Defaults to None (all environment indices).
            clone (bool, optional): Whether to return a copy or not. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The default/initial DOF position and velocity of the actor.
                Each tensor has shape: (len(env_ids), 1).
        """
        # use ellipses object to skip initial indices.
        if env_ids is None:
            env_ids = ...
        # return copy
        if clone:
            return torch.clone(self._default_dof_pos[env_ids]), torch.clone(self._default_dof_vel[env_ids])
        else:
            return self._default_dof_pos[env_ids], self._default_dof_vel[env_ids]

    def get_random_dof_state(
        self, env_ids: Optional[Sequence[int]] = None, lower: float = 0.5, upper: float = 1.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns randomly sampled DOF state (position and velocity) of actor.

        Currently, the following sampling is supported:

        - DOF positions:

          - uniform sampling between `(lower, upper)` times the default DOF position.

        - DOF velocities:

          - zero.

        Args:
            env_ids (Optional[Sequence[int]], optional): Environment indices.
                Defaults to None (all environment indices).
            lower (float, optional): Minimum value for uniform sampling. Defaults to 0.5.
            upper (float, optional): Maximum value for uniform sampling. Defaults to 1.5.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The sampled DOF position and velocity of the actor.
                Each tensor has shape: (len(env_ids), 1).
        """
        # use ellipses object to skip initial indices.
        if env_ids is None:
            env_ids = ...
            actor_count = self.count
        else:
            actor_count = len(env_ids)
        # sample DOF position
        dof_pos = self._default_dof_pos[env_ids] * sample_uniform(
            lower, upper, (actor_count, self.num_dof), device=self.device
        )
        dof_vel = self._default_dof_vel[env_ids]
        # return sampled dof state
        return dof_pos, dof_vel

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
        # -- dof state
        self._default_dof_pos = torch.zeros(self.count, self.num_dof, dtype=torch.float, device=self.device)
        self._default_dof_vel = torch.zeros(self.count, self.num_dof, dtype=torch.float, device=self.device)
        for index, dof_name in enumerate(self.articulations.dof_names):
            # dof pos
            for re_key, value in self.cfg.init_state.dof_pos.items():
                if re.match(re_key, dof_name):
                    self._default_dof_pos[:, index] = value
            # dof vel
            for re_key, value in self.cfg.init_state.dof_vel.items():
                if re.match(re_key, dof_name):
                    self._default_dof_vel[:, index] = value
        # -- tracked sites
        if self.cfg.meta_info.sites_names:
            sites_names = list()
            sites_indices = list()
            for body_index, body_name in enumerate(self.body_names):
                for re_key in self.cfg.meta_info.sites_names:
                    if re.fullmatch(re_key, body_name):
                        sites_names.append(body_name)
                        sites_indices.append(body_index)
            self.sites_indices: Dict[str, int] = dict(zip(sites_names, sites_indices))
        else:
            self.sites_indices = None

    def _create_buffers(self):
        """Create buffers for storing data."""
        # history buffers
        self._previous_dof_vel = torch.zeros(self.count, self.num_dof, dtype=torch.float, device=self.device)
        # constants
        self._ALL_INDICES = torch.arange(self.count, dtype=torch.long, device=self.device)

        # -- frame states
        self._data.root_state_w = torch.zeros(self.count, 13, dtype=torch.float, device=self.device)
        if self.sites_indices is not None:
            self._data.sites_state_w = torch.zeros(self.count, len(self.sites_indices), 13, device=self.device)
            self._data.sites_state_b = torch.zeros_like(self._data.sites_state_w)
        # -- dof states
        self._data.dof_pos = torch.zeros(self.count, self.num_dof, dtype=torch.float, device=self.device)
        self._data.dof_vel = torch.zeros_like(self._data.dof_pos)
        self._data.dof_acc = torch.zeros_like(self._data.dof_pos)
