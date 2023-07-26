# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import re
import torch
from typing import Dict, List, Optional, Sequence

import carb
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.types import ArticulationActions
from pxr import PhysxSchema

import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.orbit.actuators import ActuatorBase, ImplicitActuator

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
    actuators: Dict[str, ActuatorBase]
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
        self.actuators = dict.fromkeys(self.cfg.actuators.keys())

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
        prim_paths = self._body_view.prim_paths[: self.num_bodies]
        return [path.split("/")[-1] for path in prim_paths]

    @property
    def dof_names(self) -> List[str]:
        """Ordered names of DOFs in articulation."""
        return self.articulations.dof_names

    @property
    def num_dof(self) -> int:
        """Total number of DOFs in articulation."""
        return self.articulations.num_dof

    @property
    def num_bodies(self) -> int:
        """Total number of DOFs in articulation."""
        return self.articulations.num_bodies

    @property
    def num_actions(self) -> int:
        """Dimension of the actions applied."""
        return sum(group.control_dim for group in self.actuators.values())

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
        # -- add contact reporting api
        contact_sensor_paths = prim_utils.find_matching_prim_paths(prim_path + "/.*")
        stage = stage_utils.get_current_stage()
        for sensor_path in contact_sensor_paths:
            prim = prim_utils.get_prim_at_path(sensor_path)
            for link_prim in prim.GetChildren() + [prim]:
                if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                    # set sleep threshold to zero
                    rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                    rb.CreateSleepThresholdAttr().Set(0.0)
                    # add contact report API with threshold of zero
                    cr_api = PhysxSchema.PhysxContactReportAPI.Apply(link_prim)
                    cr_api.CreateThresholdAttr().Set(0)
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
            if self._is_spawned is None:
                raise RuntimeError("Failed to initialize robot. Please provide a valid 'prim_paths_expr'.")
            # -- use spawn path
            self._prim_paths_expr = self._spawn_prim_path
        else:
            self._prim_paths_expr = prim_paths_expr
        # create handles
        # -- robot articulation
        self.articulations = ArticulationView(self._prim_paths_expr, reset_xform_properties=False)
        self.articulations.initialize()
        # -- body view only needed to apply external force
        self._body_view = RigidPrimView(
            name="body_view",
            prim_paths_expr=f"{self._prim_paths_expr}/.*",
            reset_xform_properties=False,
            track_contact_forces=False,
            prepare_contact_sensors=False,
        )
        self._body_view.initialize()
        # set the default state
        self.articulations.post_reset()
        # set properties over all instances
        # create buffers
        self._create_buffers()
        # -- meta-information
        self._process_info_cfg()
        # -- actuation properties
        self._process_actuators_cfg()

        # set the default state of the robot (passed from the config)
        self.articulations.set_default_state(self._default_root_states[:, :3], self._default_root_states[:, 3:7])
        self.articulations.set_joints_default_state(self._data.default_dof_pos, self._data.default_dof_vel)
        # buffers for external forces and torque
        self.has_external_force = False
        self._external_force = torch.zeros((self.count, self.num_bodies, 3), device=self.device)
        self._external_torque = torch.zeros((self.count, self.num_bodies, 3), device=self.device)
        self._external_force_indices = torch.arange(self._body_view.count, dtype=torch.int32, device=self.device)

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
        # reset actuators
        for group in self.actuators.values():
            group.reset(env_ids)

    def set_dof_position_targets(self, targets: torch.Tensor, dof_ids: Optional[Sequence[int]] = None) -> None:
        if dof_ids is None:
            dof_ids = ...
        self._data.dof_pos_targets[:, dof_ids] = targets

    def set_dof_velocity_targets(self, targets: torch.Tensor, dof_ids: Optional[Sequence[int]] = None) -> None:
        if dof_ids is None:
            dof_ids = ...
        self._data.dof_pos_targets[:, dof_ids] = targets

    def set_dof_efforts(self, targets: torch.Tensor, dof_ids: Optional[Sequence[int]] = None) -> None:
        if dof_ids is None:
            dof_ids = ...
        self._data.dof_pos_targets[:, dof_ids] = targets

    def write_commands_to_sim(self):
        self._apply_actuator_model()
        # self.articulations._physics_sim_view.enable_warnings(False)
        # apply actions into simluation
        self.articulations._physics_view.set_dof_actuation_forces(self._data.dof_effort_targets, self._ALL_INDICES)
        # position and velocity targets only for implicit actuators
        if self._has_implicit_actuators:
            self.articulations._physics_view.set_dof_position_targets(self._data.dof_pos_targets, self._ALL_INDICES)
            self.articulations._physics_view.set_dof_velocity_targets(self._data.dof_vel_targets, self._ALL_INDICES)

        if self.has_external_force:
            self._body_view._physics_view.apply_forces_and_torques_at_position(
                force_data=self._external_force,
                torque_data=self._external_torque,
                position_data=None,
                indices=self._external_force_indices,
                is_global=False,
            )
        # enable warnings for other unsafe operations ;)
        # self.articulations._physics_sim_view.enable_warnings(True)

    def _apply_actuator_model(self):
        """Processes dof targets of the robot according to the model of actuator.

        The dof targets are first processed using actuator groups. Depending on the robot configuration,
        the groups compute the joint level simulation commands.
        """
        # process actions per group
        for actuator in self.actuators.values():
            if isinstance(actuator, ImplicitActuator):
                # TODO read torque from simulation
                # self._data.computed_torques[:, actuator.dof_indices] = ???
                # self._data.applied_torques[:, actuator.dof_indices] = ???
                continue
            # compute group dof command
            control_action = (
                ArticulationActions(  # TODO : A tensor dict would be nice to do the indexing of all tensors together
                    joint_positions=self._data.dof_pos_targets[:, actuator.dof_indices],
                    joint_velocities=self._data.dof_vel_targets[:, actuator.dof_indices],
                    joint_efforts=self._data.dof_effort_targets[:, actuator.dof_indices],
                )
            )
            control_action = actuator.compute(
                control_action,
                dof_pos=self._data.dof_pos[:, actuator.dof_indices],
                dof_vel=self._data.dof_vel[:, actuator.dof_indices],
            )
            # update targets
            if control_action.joint_positions is not None:
                self._data.dof_pos_targets[:, actuator.dof_indices] = control_action.joint_positions
            if control_action.joint_velocities is not None:
                self._data.dof_vel_targets[:, actuator.dof_indices] = control_action.joint_velocities
            if control_action.joint_efforts is not None:
                self._data.dof_effort_targets[:, actuator.dof_indices] = control_action.joint_efforts
            # update state
            # -- torques
            self._data.computed_torques[:, actuator.dof_indices] = actuator.computed_effort
            self._data.applied_torques[:, actuator.dof_indices] = actuator.applied_effort
            # -- actuator data
            self._data.soft_dof_vel_limits[:, actuator.dof_indices] = actuator.velocity_limit
            if hasattr(actuator, "gear_ratio"):  # TODO find a cleaner way to handle this
                self._data.gear_ratio[:, actuator.dof_indices] = actuator.gear_ratio

    def refresh_sim_data(self, refresh_dofs=True, refresh_bodies=True):
        """Refreshes the internal buffers from the simulator.

        Args:
            refresh_dofs (bool, optional): Whether to refresh the DOF states. Defaults to True.
            refresh_bodies (bool, optional): Whether to refresh the body states. Defaults to True.
        """
        if refresh_dofs:
            self._data.dof_pos[:] = self.articulations._physics_view.get_dof_positions()
            self._data.dof_vel[:] = self.articulations._physics_view.get_dof_velocities()
        if refresh_bodies:
            self._data.body_state_w[:, :, :7] = self._body_view._physics_view.get_transforms().view(self.count, -1, 7)
            self._data.body_state_w[:, :, 3:7] = self._data.body_state_w[:, :, 3:7].roll(1, dims=-1)
            self._data.body_state_w[:, :, 7:13] = self._body_view._physics_view.get_velocities().view(self.count, -1, 6)
            self._data.root_state_w[:] = self._data.body_state_w[:, 0]

    def update_buffers(self, dt: float):
        """Update the internal buffers.

        Args:
            dt (float): The amount of time passed from last `update_buffers` call.

                This is used to compute numerical derivatives of quantities such as joint accelerations
                which are not provided by the simulator.
        """
        # TODO: Make function independent of `dt` by using internal clocks that check the rate of `apply_action()` call.
        # frame states
        # -- dof states
        self._data.dof_acc[:] = (self._data.dof_vel - self._previous_dof_vel) / dt
        # update history buffers
        self._previous_dof_vel[:] = self._data.dof_vel[:]

    def debug_vis(self):
        pass

    """
    Operations - State.
    """

    def set_root_state(self, root_states: torch.Tensor, env_ids: Optional[Sequence[int]] = None):
        """Sets the root state (pose and velocity) of the actor over selected environment indices.

        Args:
            root_states (torch.Tensor): Input root state for the actor, shape: (len(env_ids), 13), where
                `root_states = [pos_x, pos_y, pos_z,
                                quat_w, quat_x, quat_y, quat_z,
                                vel_x, vel_y, vel_z,
                                ang_vel_x, ang_vel_y, ang_vel_z]`.
            env_ids (Optional[Sequence[int]]): Environment indices.
                If :obj:`None`, then all indices are used.
        """
        # resolve all indices
        if env_ids is None:
            env_ids = self._ALL_INDICES

        # note: we need to do this here since tensors are not set into simulation until step.
        # set into internal buffers
        self._data.root_state_w[env_ids] = root_states
        # convert between quaternion conventions
        root_pose_converted = self._data.root_state_w[:, :7].clone()
        root_pose_converted[:, 3:] = root_pose_converted[:, 3:].roll(-1, dims=-1)
        # set into simulation
        self.articulations._physics_view.set_root_transform(root_pose_converted, indices=env_ids)
        self.articulations._physics_view.set_root_velocities(self._data.root_state_w[:, 7:], indices=env_ids)

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

        # note: we need to do this here since tensors are not set into simulation until step.
        # set into internal buffers
        self._data.dof_pos[env_ids] = dof_pos
        self._data.dof_vel[env_ids] = dof_vel
        self._data.dof_acc[env_ids] = 0.0

        self.articulations._physics_view.set_dof_positions(self._data.dof_pos, indices=env_ids)
        self.articulations._physics_view.set_dof_velocities(self._data.dof_vel, indices=env_ids)

    def set_external_force_and_torque(self, forces, torques, env_ids, body_ids):
        if forces.any() or torques.any():
            self.has_external_force = True
            # create global body indices from env_ids and env_body_ids
            if env_ids is None:
                env_ids = torch.arange(self.count, device=self.device)
            bodies_per_env = self._body_view.count // self.count
            indices = torch.tensor(body_ids, dtype=torch.long, device=self.device).repeat(len(env_ids), 1)
            indices += env_ids.unsqueeze(1) * bodies_per_env

            self._external_force.flatten(0, 1)[indices] = forces.unsqueeze(1)
            self._external_torque.flatten(0, 1)[indices] = torques.unsqueeze(1)
        else:
            self.has_external_force = False

    def set_dof_stiffness(self, stiffness, env_ids=None, dof_ids=None):
        if env_ids is None:
            env_ids = ...
        if dof_ids is None:
            dof_ids = ...

        self._data.dof_stiffness[env_ids, dof_ids] = stiffness
        self.articulations._physics_view.set_dof_stiffnesses(
            self._data.dof_stiffness.cpu(), indices=self._ALL_INDICES.cpu()
        )

    def set_dof_damping(self, damping, env_ids=None, dof_ids=None):
        if env_ids is None:
            env_ids = ...
        if dof_ids is None:
            dof_ids = ...

        self._data.dof_damping[env_ids, dof_ids] = damping
        self.articulations._physics_view.set_dof_dampings(self._data.dof_damping.cpu(), indices=self._ALL_INDICES.cpu())

    def set_dof_torque_limit(self, torque_limit, env_ids=None, dof_ids=None):
        if env_ids is None:
            env_ids = ...
        if dof_ids is None:
            dof_ids = ...

        torque_limit_all = self.articulations._physics_view.get_dof_max_forces()
        torque_limit_all[env_ids, dof_ids] = torque_limit
        self.articulations._physics_view.set_dof_max_forces(torque_limit_all.cpu(), self._ALL_INDICES.cpu())

    def find_bodies(self, name_keys):
        idx_list = []
        names_list = []
        if not isinstance(name_keys, list):
            name_keys = [name_keys]
        for i, body_name in enumerate(self.body_names):  # TODO check if we need to sort body names
            for re_name in name_keys:
                if re.match(re_name, body_name):
                    idx_list.append(i)
                    names_list.append(body_name)
                    continue
        return idx_list, names_list

    def find_dofs(self, name_keys, dof_subset=None):
        idx_list = []
        names_list = []
        if dof_subset is None:
            dof_names = self.dof_names
        else:
            dof_names = dof_subset
        if not isinstance(name_keys, list):
            name_keys = [name_keys]
        for i, dof_name in enumerate(dof_names):  # TODO check if we need to sort body names
            for re_name in name_keys:
                if re.match(re_name, dof_name):
                    idx_list.append(i)
                    names_list.append(dof_name)
                    continue
        return idx_list, names_list

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
        for index, dof_name in enumerate(self.articulations.dof_names):
            # dof pos
            for re_key, value in self.cfg.init_state.dof_pos.items():
                if re.match(re_key, dof_name):
                    self._data.default_dof_pos[:, index] = value
            # dof vel
            for re_key, value in self.cfg.init_state.dof_vel.items():
                if re.match(re_key, dof_name):
                    self._data.default_dof_vel[:, index] = value

    def _process_actuators_cfg(self):
        """Process and apply articulation DOF properties."""
        self._has_implicit_actuators = False
        # iterate over all actuator configurations
        for actuator_name, actuator_cfg in self.cfg.actuators.items():
            # create actuator group
            dof_ids, dof_names = self.find_dofs(actuator_cfg.dof_name_expr)
            # for efficiency avoid indexing over all indices
            if len(dof_ids) == self.num_dof:
                dof_ids = ...
            actuator_cls = actuator_cfg.cls
            actuator: ActuatorBase = actuator_cls(
                cfg=actuator_cfg, dof_names=dof_names, dof_ids=dof_ids, num_envs=self.count, device=self.device
            )
            # store actuator group
            self.actuators[actuator_name] = actuator
            # store the control mode and dof indices (optimization)
            # if "P" in actuator.sim_command_type or "V" in actuator.sim_command_type:
            if isinstance(actuator, ImplicitActuator):
                self._has_implicit_actuators = True
                self.set_dof_stiffness(actuator.stiffness, dof_ids=actuator.dof_indices)
                self.set_dof_damping(actuator.damping, dof_ids=actuator.dof_indices)
                self.set_dof_torque_limit(actuator.effort_limit, dof_ids=actuator.dof_indices)
            else:
                # the gains and limits are processed by the actuator model
                # we set gains to zero, and torque limit to a high value in simulation to avoid any interference
                self.set_dof_stiffness(0.0, dof_ids=actuator.dof_indices)
                self.set_dof_damping(0.0, dof_ids=actuator.dof_indices)
                self.set_dof_torque_limit(1.0e9, dof_ids=actuator.dof_indices)

        # perform some sanity checks to ensure actuators are prepared correctly
        total_act_dof = sum(group.num_actuators for group in self.actuators.values())
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
        # -- properties
        self._data.dof_names = self.articulations.dof_names
        # -- frame states
        self._data.root_state_w = torch.zeros(self.count, 13, dtype=torch.float, device=self.device)
        self._data.body_state_w = torch.zeros(self.count, self.num_bodies, 13, dtype=torch.float, device=self.device)
        # -- dof states
        self._data.dof_pos = torch.zeros(self.count, self.num_dof, dtype=torch.float, device=self.device)
        self._data.dof_vel = torch.zeros_like(self._data.dof_pos)
        self._data.dof_acc = torch.zeros_like(self._data.dof_pos)
        self._data.default_dof_pos = torch.zeros_like(self._data.dof_pos)
        self._data.default_dof_vel = torch.zeros_like(self._data.dof_pos)
        # -- dof commands
        self._data.dof_pos_targets = torch.zeros_like(self._data.dof_pos)
        self._data.dof_vel_targets = torch.zeros_like(self._data.dof_pos)
        self._data.dof_effort_targets = torch.zeros_like(self._data.dof_pos)
        self._data.dof_stiffness = torch.zeros_like(self._data.dof_pos)
        self._data.dof_damping = torch.zeros_like(self._data.dof_pos)
        # -- dof commands (explicit)
        self._data.computed_torques = torch.zeros_like(self._data.dof_pos)
        self._data.applied_torques = torch.zeros_like(self._data.dof_pos)
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
