# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import re
import torch
from typing import Dict, List, Optional, Sequence, Tuple, Union

import carb
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.articulations import ArticulationView

import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.orbit.actuators.group import *  # noqa: F403, F401
from omni.isaac.orbit.actuators.group import ActuatorGroup
from omni.isaac.orbit.utils.math import sample_uniform

from .robot_base_cfg import RobotBaseCfg
from .robot_base_data import RobotBaseData


class RobotBase:
    """Base class for robots.

    The robot class is strictly treated as the physical articulated system which typically
    runs at a high frequency (most commonly 1000 Hz).

    This class wraps around :class:`ArticulationView` class from Isaac Sim to support the following:
    * Configuring using a single dataclass (struct).
    * Handling different rigid body views inside the robot.
    * Handling different actuator groups and models for the robot.
    """

    cfg: RobotBaseCfg
    """Configuration for the robot."""
    articulations: ArticulationView = None
    """Articulation view of the robot."""
    actuator_groups: Dict[str, ActuatorGroup]
    """Mapping between actuator group names and instance."""

    def __init__(self, cfg: RobotBaseCfg):
        """Initialize the robot class.

        Args:
            cfg (RobotBaseCfg): The configuration instance.
        """
        # store inputs
        self.cfg = cfg
        # container for data access
        self._data = RobotBaseData()

        # Flag to check that instance is spawned.
        self._is_spawned = False
        # data for storing actuator group
        self.actuator_groups = dict.fromkeys(self.cfg.actuator_groups.keys())

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
        return self.articulations._device  # noqa: W0212

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
    def num_actions(self) -> int:
        """Dimension of the actions applied."""
        return sum(group.control_dim for group in self.actuator_groups.values())

    @property
    def data(self) -> RobotBaseData:
        """Data related to articulation."""
        return self._data

    """
    Operations.
    """

    def spawn(self, prim_path: str, translation: Sequence[float] = None, orientation: Sequence[float] = None):
        """Spawn a robot into the stage (loaded from its USD file).

        Note:
            If inputs `translation` or `orientation` are not :obj:`None`, then they override the initial root
            state specified through the configuration class at spawning.

        Args:
            prim_path (str): The prim path for spawning robot at.
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
            )
        else:
            carb.log_warn(f"A prim already exists at prim path: '{prim_path}'. Skipping...")

        # apply rigid body properties
        kit_utils.set_nested_rigid_body_properties(prim_path, **self.cfg.rigid_props.to_dict())
        # apply collision properties
        kit_utils.set_nested_collision_properties(prim_path, **self.cfg.collision_props.to_dict())
        # articulation root settings
        kit_utils.set_articulation_properties(prim_path, **self.cfg.articulation_props.to_dict())
        # set spawned to true
        self._is_spawned = True

    def initialize(self, prim_paths_expr: Optional[str] = None):
        """Initializes the PhysX handles and internal buffers.

        Note:
            PhysX handles are only enabled once the simulator starts playing. Hence, this function needs to be
            called whenever the simulator "plays" from a "stop" state.

        Args:
            prim_paths_expr (Optional[str], optional): The prim path expression for robot prims. Defaults to None.

        Raises:
            RuntimeError: When input `prim_paths_expr` is :obj:`None`, the method defaults to using the last
                prim path set when calling the :meth:`spawn()` function. In case, the robot was not spawned
                and no valid `prim_paths_expr` is provided, the function throws an error.
        """
        # default prim path if not cloned
        if prim_paths_expr is None:
            if self._is_spawned is not None:
                self._prim_paths_expr = self._spawn_prim_path
            else:
                raise RuntimeError(
                    "Initialize the robot failed! Please provide a valid argument for `prim_paths_expr`."
                )
        else:
            self._prim_paths_expr = prim_paths_expr
        # create handles
        # -- robot articulation
        self.articulations = ArticulationView(self._prim_paths_expr, reset_xform_properties=False)
        self.articulations.initialize()
        # set the default state
        self.articulations.post_reset()
        # set properties over all instances
        # -- meta-information
        self._process_info_cfg()
        # -- actuation properties
        self._process_actuators_cfg()
        # create buffers
        self._create_buffers()

    def reset_buffers(self, env_ids: Optional[Sequence[int]] = None):
        """Resets all internal buffers.

        Args:
            env_ids (Optional[Sequence[int]], optional): The indices of the robot to reset.
                Defaults to None (all instances).
        """
        # use ellipses object to skip initial indices.
        if env_ids is None:
            env_ids = ...
        # reset history
        self._previous_dof_vel[env_ids] = 0
        # TODO: Reset other cached variables.
        self.articulations.set_joint_efforts(self._ZERO_JOINT_EFFORT[env_ids], indices=self._ALL_INDICES[env_ids])
        # reset actuators
        for group in self.actuator_groups.values():
            group.reset(env_ids)

    def apply_action(self, actions: torch.Tensor):
        """Apply the input action for the robot into the simulator.

        The actions are first processed using actuator groups. Depending on the robot configuration,
        the groups compute the joint level simulation commands and sets them into the PhysX buffers.

        Args:
            actions (torch.Tensor): The input actions to apply.
        """
        # slice actions per actuator group
        group_actions_dims = [group.control_dim for group in self.actuator_groups.values()]
        all_group_actions = torch.split(actions, group_actions_dims, dim=-1)
        # note: we use internal buffers to deal with the resets() as the buffers aren't forwarded
        #   unit the next simulation step.
        dof_pos = self._data.dof_pos
        dof_vel = self._data.dof_vel
        # process actions per group
        for group, group_actions in zip(self.actuator_groups.values(), all_group_actions):
            # compute group dof command
            control_action = group.compute(
                group_actions, dof_pos=dof_pos[:, group.dof_indices], dof_vel=dof_vel[:, group.dof_indices]
            )
            # update targets
            if control_action.joint_positions is not None:
                self._data.dof_pos_targets[:, group.dof_indices] = control_action.joint_positions
            if control_action.joint_velocities is not None:
                self._data.dof_vel_targets[:, group.dof_indices] = control_action.joint_velocities
            if control_action.joint_efforts is not None:
                self._data.dof_effort_targets[:, group.dof_indices] = control_action.joint_efforts
            # update state
            # -- torques
            self._data.computed_torques[:, group.dof_indices] = group.computed_torques
            self._data.applied_torques[:, group.dof_indices] = group.applied_torques
            # -- actuator data
            self._data.gear_ratio[:, group.dof_indices] = group.gear_ratio
            if group.velocity_limit is not None:
                self._data.soft_dof_vel_limits[:, group.dof_indices] = group.velocity_limit
        # silence the physics sim for warnings that make no sense :)
        # note (18.08.2022): Saw a difference of up to 5 ms per step when using Isaac Sim
        #   ArticulationView.apply_action() method compared to direct PhysX calls. Thus,
        #   this function is optimized to apply actions for the whole robot.
        self.articulations._physics_sim_view.enable_warnings(False)
        # apply actions into sim
        if self.sim_dof_control_modes["position"]:
            self.articulations._physics_view.set_dof_position_targets(self._data.dof_pos_targets, self._ALL_INDICES)
        if self.sim_dof_control_modes["velocity"]:
            self.articulations._physics_view.set_dof_velocity_targets(self._data.dof_vel_targets, self._ALL_INDICES)
        if self.sim_dof_control_modes["effort"]:
            self.articulations._physics_view.set_dof_actuation_forces(self._data.dof_effort_targets, self._ALL_INDICES)
        # enable warnings for other unsafe operations ;)
        self.articulations._physics_sim_view.enable_warnings(True)

    def update_buffers(self, dt: float):
        """Update the internal buffers.

        Args:
            dt (float): The amount of time passed from last `update_buffers` call.

                This is used to compute numerical derivatives of quantities such as joint accelerations
                which are not provided by the simulator.
        """
        # TODO: Make function independent of `dt` by using internal clocks that check the rate of `apply_action()` call.
        # frame states
        # -- root frame in world
        position_w, quat_w = self.articulations.get_world_poses(indices=self._ALL_INDICES, clone=False)
        self._data.root_state_w[:, 0:3] = position_w
        self._data.root_state_w[:, 3:7] = quat_w
        self._data.root_state_w[:, 7:] = self.articulations.get_velocities(indices=self._ALL_INDICES, clone=False)
        # -- dof states
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
        self,
        env_ids: Optional[Sequence[int]] = None,
        lower: Union[float, torch.Tensor] = 0.5,
        upper: Union[float, torch.Tensor] = 1.5,
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
            lower (Union[float, torch.Tensor], optional): Minimum value for uniform sampling. Defaults to 0.5.
            upper (Union[float, torch.Tensor], optional): Maximum value for uniform sampling. Defaults to 1.5.

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

    def _process_actuators_cfg(self):
        """Process and apply articulation DOF properties."""
        # sim control mode and dof indices (optimization)
        self.sim_dof_control_modes = {"position": list(), "velocity": list(), "effort": list()}
        # iterate over all actuator configuration
        for group_name, group_cfg in self.cfg.actuator_groups.items():
            # create actuator group
            actuator_group_cls = eval(group_cfg.cls_name)
            actuator_group: ActuatorGroup = actuator_group_cls(cfg=group_cfg, view=self.articulations)
            # store actuator group
            self.actuator_groups[group_name] = actuator_group
            # store the control mode and dof indices (optimization)
            if actuator_group.model_type == "implicit":
                for command in actuator_group.command_types:
                    # resolve name of control mode
                    if "p" in command:
                        command_name = "position"
                    elif "v" in command:
                        command_name = "velocity"
                    elif "t" in command:
                        command_name = "effort"
                    else:
                        continue
                    # store dof indices
                    self.sim_dof_control_modes[command_name].extend(actuator_group.dof_indices)
            else:
                # in explicit mode, we always use the "effort" control mode
                self.sim_dof_control_modes["effort"].extend(actuator_group.dof_indices)

        # perform some sanity checks to ensure actuators are prepared correctly
        total_act_dof = sum(group.num_actuators for group in self.actuator_groups.values())
        if total_act_dof != self.num_dof:
            carb.log_warn(
                f"Not all actuators are configured through groups! Total number of actuated DOFs not equal to number of DOFs available: {total_act_dof} != {self.num_dof}."
            )

    def _create_buffers(self):
        """Create buffers for storing data."""
        # history buffers
        self._previous_dof_vel = torch.zeros(self.count, self.num_dof, dtype=torch.float, device=self.device)
        # constants
        self._ALL_INDICES = torch.arange(self.count, dtype=torch.long, device=self.device)
        self._ZERO_JOINT_EFFORT = torch.zeros(self.count, self.num_dof, dtype=torch.float, device=self.device)

        # -- frame states
        self._data.root_state_w = torch.zeros(self.count, 13, dtype=torch.float, device=self.device)
        # -- dof states
        self._data.dof_pos = torch.zeros(self.count, self.num_dof, dtype=torch.float, device=self.device)
        self._data.dof_vel = torch.zeros_like(self._data.dof_pos)
        self._data.dof_acc = torch.zeros_like(self._data.dof_pos)
        # -- dof commands
        self._data.dof_pos_targets = torch.zeros_like(self._data.dof_pos)
        self._data.dof_vel_targets = torch.zeros_like(self._data.dof_pos)
        self._data.dof_effort_targets = torch.zeros_like(self._data.dof_pos)
        # -- dof commands (explicit)
        self._data.computed_torques = torch.zeros_like(self._data.dof_pos)
        self._data.applied_torques = torch.zeros_like(self._data.dof_pos)
        # -- default actuator offset
        self._data.actuator_pos_offset = torch.zeros_like(self._data.dof_pos)
        self._data.actuator_vel_offset = torch.zeros_like(self._data.dof_pos)
        # -- other data
        self._data.soft_dof_pos_limits = torch.zeros(self.count, self.num_dof, 2, dtype=torch.float, device=self.device)
        self._data.soft_dof_vel_limits = torch.zeros(self.count, self.num_dof, dtype=torch.float, device=self.device)
        self._data.gear_ratio = torch.ones(self.count, self.num_dof, dtype=torch.float, device=self.device)

        # soft dof position limits (recommended not to be too close to limits).
        dof_pos_limits = self.articulations.get_dof_limits()
        dof_pos_mean = (dof_pos_limits[..., 0] + dof_pos_limits[..., 1]) / 2
        dof_pos_range = dof_pos_limits[..., 1] - dof_pos_limits[..., 0]
        soft_limit_factor = self.cfg.meta_info.soft_dof_pos_limit_factor
        # add to data
        self._data.soft_dof_pos_limits[..., 0] = dof_pos_mean - 0.5 * dof_pos_range * soft_limit_factor
        self._data.soft_dof_pos_limits[..., 1] = dof_pos_mean + 0.5 * dof_pos_range * soft_limit_factor

        # store the offset amounts from actuator groups
        for group in self.actuator_groups.values():
            self._data.actuator_pos_offset[:, group.dof_indices] = group.dof_pos_offset
