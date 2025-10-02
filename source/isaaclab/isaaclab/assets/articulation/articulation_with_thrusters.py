# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Flag for pyright to ignore type errors in this file.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import os
os.environ["OMNI_PHYSX_TENSORS_WARNINGS"] = "0"

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from .articulation import Articulation

import omni.log
from omni.physics.tensors import JointType
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.version import get_version
from pxr import UsdPhysics

import isaaclab.sim as sim_utils

import isaaclab.utils.string as string_utils
from isaaclab.utils.types import ArticulationThrustActions
from isaaclab.actuators import ThrusterCfg, Thruster

from isaaclab.assets.articulation.articulation_data_thrusters import ArticulationDataWithThrusters

if TYPE_CHECKING:
    from .articulation_cfg import ArticulationWithThrustersCfg


class ArticulationWithThrusters(Articulation):
    """An articulation asset class with thrusters.

    This class extends the base articulation class to include thruster actuators.
    """

    cfg: ArticulationWithThrustersCfg
    """Configuration instance for the articulations."""

    actuators: dict[str, Thruster]
    """Dictionary of actuator instances for the articulation.

    The keys are the actuator names and the values are the actuator instances. The actuator instances
    are initialized based on the actuator configurations specified in the :attr:`ArticulationCfg.actuators`
    attribute. They are used to compute the joint commands during the :meth:`write_data_to_sim` function.
    """

    def __init__(self, cfg: ArticulationWithThrustersCfg):
        """Initialize the articulation.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)

    """
    Properties
    """
    
    @property
    def joint_names(self) -> list[str]:
        """Ordered names of thrusters in articulation."""
        joint_types = self.root_physx_view.shared_metatype.joint_types
        joint_names = self.root_physx_view.shared_metatype.joint_names
        # TODO change this such that the the articulation can have fixed joints that are not thruster
        fixed_joints = [name for name, jtype in zip(joint_names, joint_types) if jtype == JointType.Fixed]
        return fixed_joints

    @property
    def num_joints(self) -> int:
        """Number of thrusters in articulation."""
        return len(self.joint_names)

    def _create_buffers(self):
        # constants
        self._ALL_INDICES = torch.arange(self.num_instances, dtype=torch.long, device=self.device)
        self._allocation_matrix = torch.tensor(self.cfg.allocation_matrix, device=self.device)

        # external forces and torques
        self.has_external_wrench = False
        self.uses_external_wrench_positions = False
        self._external_force_b = torch.zeros((self.num_instances, self.num_bodies, 3), device=self.device)
        self._external_torque_b = torch.zeros_like(self._external_force_b)
        self._external_wrench_positions_b = torch.zeros_like(self._external_force_b)
        self._use_global_wrench_frame = False

        # asset named data
        self._data.thruster_names = self.joint_names
        self._data.body_names = self.body_names
        # tendon names are set in _process_tendons function

        # -- joint properties
        self._data.default_joint_pos_limits = self.root_physx_view.get_dof_limits().to(self.device).clone()
        self._data.default_joint_stiffness = self.root_physx_view.get_dof_stiffnesses().to(self.device).clone()
        self._data.default_joint_damping = self.root_physx_view.get_dof_dampings().to(self.device).clone()
        self._data.default_joint_armature = self.root_physx_view.get_dof_armatures().to(self.device).clone()
        if int(get_version()[2]) < 5:
            self._data.default_joint_friction_coeff = (
                self.root_physx_view.get_dof_friction_coefficients().to(self.device).clone()
            )
            self._data.default_joint_dynamic_friction_coeff = torch.zeros_like(self._data.default_joint_friction_coeff)
            self._data.default_joint_viscous_friction_coeff = torch.zeros_like(self._data.default_joint_friction_coeff)
        else:
            friction_props = self.root_physx_view.get_dof_friction_properties()
            self._data.default_joint_friction_coeff = friction_props[:, :, 0].to(self.device).clone()
            self._data.default_joint_dynamic_friction_coeff = friction_props[:, :, 1].to(self.device).clone()
            self._data.default_joint_viscous_friction_coeff = friction_props[:, :, 2].to(self.device).clone()

        self._data.joint_pos_limits = self._data.default_joint_pos_limits.clone()
        self._data.joint_vel_limits = self.root_physx_view.get_dof_max_velocities().to(self.device).clone()
        self._data.joint_effort_limits = self.root_physx_view.get_dof_max_forces().to(self.device).clone()
        self._data.joint_stiffness = self._data.default_joint_stiffness.clone()
        self._data.joint_damping = self._data.default_joint_damping.clone()
        self._data.joint_armature = self._data.default_joint_armature.clone()
        self._data.joint_friction_coeff = self._data.default_joint_friction_coeff.clone()
        self._data.joint_dynamic_friction_coeff = self._data.default_joint_dynamic_friction_coeff.clone()
        self._data.joint_viscous_friction_coeff = self._data.default_joint_viscous_friction_coeff.clone()

        # -- body properties
        self._data.default_mass = self.root_physx_view.get_masses().clone()
        self._data.default_inertia = self.root_physx_view.get_inertias().clone()

        # -- joint commands (sent to the actuator from the user)
        self._data.joint_pos_target = torch.zeros(self.num_instances, self.num_joints, device=self.device)
        self._data.joint_vel_target = torch.zeros_like(self._data.joint_pos_target)
        self._data.joint_effort_target = torch.zeros_like(self._data.joint_pos_target)
        self._data.thrust_target = torch.zeros(self.num_instances, self.num_joints, device=self.device)
        # -- joint commands (sent to the simulation after actuator processing)
        self._joint_pos_target_sim = torch.zeros_like(self._data.joint_pos_target)
        self._joint_vel_target_sim = torch.zeros_like(self._data.joint_pos_target)
        self._joint_effort_target_sim = torch.zeros_like(self._data.joint_pos_target)
        self._thrust_target_sim = torch.zeros_like(self._data.thrust_target)
        # self._torque_target_sim = torch.zeros_like(self._data.torque_target)
        self._internal_wrench_target_sim = torch.zeros(self.num_instances, 6, device=self.device)
        self._internal_force_target_sim = torch.zeros_like(self._external_force_b)
        self._internal_torque_target_sim = torch.zeros_like(self._external_torque_b)

        # -- computed joint efforts from the actuator models
        self._data.computed_torque = torch.zeros_like(self._data.joint_pos_target)
        self._data.applied_torque = torch.zeros_like(self._data.joint_pos_target)
        self._data.computed_thrust = torch.zeros_like(self._data.thrust_target)
        self._data.applied_thrust = torch.zeros_like(self._data.thrust_target)
        

        # -- other data that are filled based on explicit actdef _validate_cfg(self):
        """Validate the configuration after processing.

        Note:
            This function should be called only after the configuration has been processed and the buffers have been
            created. Otherwise, some settings that are altered during processing may not be validated.
            For instance, the actuator models may change the joint max velocity limits.
        """
        # check that the default values are within the limits
        # joint_pos_limits = self.root_physx_view.get_dof_limits()[0].to(self.device)
        # out_of_range = self._data.default_joint_pos[0] < joint_pos_limits[:, 0]
        # out_of_range |= self._data.default_joint_pos[0] > joint_pos_limits[:, 1]
        # violated_indices = torch.nonzero(out_of_range, as_tuple=False).squeeze(-1)
        # throw error if any of the default joint positions are out of the limits
        # if len(violated_indices) > 0:
        #     # prepare message for violated joints
        #     msg = "The following joints have default positions out of the limits: \n"
        #     for idx in violated_indices:
        #         joint_name = self.data.joint_names[idx]
        #         joint_limit = joint_pos_limits[idx]
        #         joint_pos = self.data.default_joint_pos[0, idx]
        #         # add to message
        #         msg += f"\t- '{joint_name}': {joint_pos:.3f} not in [{joint_limit[0]:.3f}, {joint_limit[1]:.3f}]\n"
        #     raise ValueError(msg)

        # check that the default joint velocities are within the limits
        # joint_max_vel = self.root_physx_view.get_dof_max_velocities()[0].to(self.device)
        # out_of_range = torch.abs(self._data.default_joint_vel[0]) > joint_max_vel
        # violated_indices = torch.nonzero(out_of_range, as_tuple=False).squeeze(-1)
        # if len(violated_indices) > 0:
        #     # prepare message for violated joints
        #     msg = "The following joints have default velocities out of the limits: \n"
        #     for idx in violated_indices:
        #         joint_name = self.data.joint_names[idx]
        #         joint_limit = [-joint_max_vel[idx], joint_max_vel[idx]]
        #         joint_vel = self.data.default_joint_vel[0, idx]
        #         # add to message
        #         msg += f"\t- '{joint_name}': {joint_vel:.3f} not in [{joint_limit[0]:.3f}, {joint_limit[1]:.3f}]\n"
        #     raise ValueError(msg)
        # self._data.soft_joint_vel_limits = torch.zeros(self.num_instances, self.num_joints, device=self.device)
        # self._data.gear_ratio = torch.ones(self.num_instances, self.num_joints, device=self.device)

        # add to data
        self._data.soft_joint_pos_limits = torch.zeros(self.num_instances, self.num_joints, 2, device=self.device)


    def set_thrust_target(
        self, target: torch.Tensor, joint_ids: Sequence[int] | slice | None = None, env_ids: Sequence[int] | None = None
    ):
        
        # resolve indices
        if env_ids is None:
            env_ids = slice(None)
        if joint_ids is None:
            joint_ids = slice(None)
        # broadcast env_ids if needed to allow double indexing
        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        # set targets
        self._data.thrust_target[env_ids, joint_ids] = target
    

    def write_data_to_sim(self):
        """Write external wrenches and joint commands to the simulation.

        If any explicit actuators are present, then the actuator models are used to compute the
        joint commands. Otherwise, the joint commands are directly set into the simulation.

        Note:
            We write external wrench to the simulation here since this function is called before the simulation step.
            This ensures that the external wrench is applied at every simulation step.
        """
        # # write external wrench
        # if self.has_external_wrench:
        #     if self.uses_external_wrench_positions:
        #         self.root_physx_view.apply_forces_and_torques_at_position(
        #             force_data=self._external_force_b.view(-1, 3),
        #             torque_data=self._external_torque_b.view(-1, 3),
        #             position_data=self._external_wrench_positions_b.view(-1, 3),
        #             indices=self._ALL_INDICES,        
        #             is_global=self._use_global_wrench_frame,
        #         )
        #     else:
        #         self.root_physx_view.apply_forces_and_torques_at_position(
        #             force_data=self._external_force_b.view(-1, 3),
        #             torque_data=self._external_torque_b.view(-1, 3),
        #             position_data=None,
        #             indices=self._ALL_INDICES,
        #             is_global=self._use_global_wrench_frame,
        #         )

        # apply actuator models
        self._apply_actuator_model()
        # write actions into simulation
        # TODO @mihirk @welfr @greg fince a nice way to only apply this when needed.
        # self.root_physx_view.set_dof_actuation_forces(self._joint_effort_target_sim, self._ALL_INDICES)
        # apply thruster actions
        self._combine_thrusts()
        # TODO make sure force is applied to center of gravity (important for arbitrary robots)
        # TODO @mihirk check and ask what is the correct way to include all indices of the robots.
        
        root_body_id = 0
        all_indices = torch.arange(self._internal_force_target_sim.shape[0], device=self.device)

        self.root_physx_view.apply_forces_and_torques_at_position(force_data=self._internal_force_target_sim.view(-1,3),
                                                                 torque_data = self._internal_torque_target_sim.view(-1,3),
                                                                 position_data=None,
                                                                 indices = all_indices,
                                                                 is_global = False)
        
        # position and velocity targets only for implicit actuators
        # if self._has_implicit_actuators:
        #     self.root_physx_view.set_dof_position_targets(self._joint_pos_target_sim, self._ALL_INDICES)
        #     self.root_physx_view.set_dof_velocity_targets(self._joint_vel_target_sim, self._ALL_INDICES)
            
    def _initialize_impl(self):
        # obtain global simulation view
        self._physics_sim_view = SimulationManager.get_physics_sim_view()

        if self.cfg.articulation_root_prim_path is not None:
            # The articulation root prim path is specified explicitly, so we can just use this.
            root_prim_path_expr = self.cfg.prim_path + self.cfg.articulation_root_prim_path
        else:
            # No articulation root prim path was specified, so we need to search
            # for it. We search for this in the first environment and then
            # create a regex that matches all environments.
            first_env_matching_prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
            if first_env_matching_prim is None:
                raise RuntimeError(f"Failed to find prim for expression: '{self.cfg.prim_path}'.")
            first_env_matching_prim_path = first_env_matching_prim.GetPath().pathString

            # Find all articulation root prims in the first environment.
            first_env_root_prims = sim_utils.get_all_matching_child_prims(
                first_env_matching_prim_path,
                predicate=lambda prim: prim.HasAPI(UsdPhysics.ArticulationRootAPI),
            )
            if len(first_env_root_prims) == 0:
                raise RuntimeError(
                    f"Failed to find an articulation when resolving '{first_env_matching_prim_path}'."
                    " Please ensure that the prim has 'USD ArticulationRootAPI' applied."
                )
            if len(first_env_root_prims) > 1:
                raise RuntimeError(
                    f"Failed to find a single articulation when resolving '{first_env_matching_prim_path}'."
                    f" Found multiple '{first_env_root_prims}' under '{first_env_matching_prim_path}'."
                    " Please ensure that there is only one articulation in the prim path tree."
                )

            # Now we convert the found articulation root from the first
            # environment back into a regex that matches all environments.
            first_env_root_prim_path = first_env_root_prims[0].GetPath().pathString
            root_prim_path_relative_to_prim_path = first_env_root_prim_path[len(first_env_matching_prim_path) :]
            root_prim_path_expr = self.cfg.prim_path + root_prim_path_relative_to_prim_path

        # -- articulation
        self._root_physx_view = self._physics_sim_view.create_articulation_view(root_prim_path_expr.replace(".*", "*"))
        
        # check if the articulation was created
        if self._root_physx_view._backend is None:
            raise RuntimeError(f"Failed to create articulation at: {root_prim_path_expr}. Please check PhysX logs.")

        if int(get_version()[2]) < 5:
            omni.log.warn(
                "Spatial tendons are not supported in Isaac Sim < 5.0: patching spatial-tendon getter"
                " and setter to use dummy value"
            )
            self._root_physx_view.max_spatial_tendons = 0
            self._root_physx_view.get_spatial_tendon_stiffnesses = lambda: torch.empty(0, device=self.device)
            self._root_physx_view.get_spatial_tendon_dampings = lambda: torch.empty(0, device=self.device)
            self._root_physx_view.get_spatial_tendon_limit_stiffnesses = lambda: torch.empty(0, device=self.device)
            self._root_physx_view.get_spatial_tendon_offsets = lambda: torch.empty(0, device=self.device)
            self._root_physx_view.set_spatial_tendon_properties = lambda *args, **kwargs: omni.log.warn(
                "Spatial tendons are not supported in Isaac Sim < 5.0: Calling"
                " set_spatial_tendon_properties has no effect"
            )

        # log information about the articulation
        omni.log.info(f"Articulation initialized at: {self.cfg.prim_path} with root '{root_prim_path_expr}'.")
        omni.log.info(f"Is fixed root: {self.is_fixed_base}")
        omni.log.info(f"Number of bodies: {self.num_bodies}")
        omni.log.info(f"Body names: {self.body_names}")
        omni.log.info(f"Number of joints: {self.num_joints}")
        omni.log.info(f"Thruster names: {self.joint_names}")
        omni.log.info(f"Number of fixed tendons: {self.num_fixed_tendons}")

        # container for data access
        self._data = ArticulationDataWithThrusters(self.root_physx_view, self.device)

        # create buffers
        self._create_buffers()
        # process configuration
        self._process_cfg()
        self._process_actuators_cfg()
        self._process_tendons()
        # validate configuration
        self._validate_cfg()
        # update the robot data
        self.update(0.0)
        # log joint information
        self._log_articulation_info()
        
        
    def _process_cfg(self):
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
        default_root_state = torch.tensor(default_root_state, dtype=torch.float, device=self.device)
        self._data.default_root_state = default_root_state.repeat(self.num_instances, 1)
        self._data.default_thruster_rps= torch.zeros(self.num_instances, self.num_joints, device=self.device)
        indices_list, _, values_list = string_utils.resolve_matching_names_values(
            self.cfg.init_state.rps, self.joint_names
        )

        self._data.default_thruster_rps[:, indices_list] = torch.tensor(values_list, device=self.device)

    """
    Internal helpers -- Actuators.
    """

    def _process_actuators_cfg(self):
        """Process and apply articulation joint properties."""
        # create actuators
        self.actuators = dict()
        # flag for implicit actuators
        # if this is false, we by-pass certain checks when doing actuator-related operations

        # iterate over all actuator configurations
        for actuator_name, actuator_cfg in self.cfg.actuators.items():
            # type annotation for type checkers
            
            actuator_cfg: ThrusterCfg
            # create actuator group
            thruster_ids, thruster_names = self.find_thrusters(actuator_cfg.thruster_names_expr)
            # check if any joints are found
            if len(thruster_names) == 0:
                raise ValueError(
                    f"No joints found for actuator group: {actuator_name} with joint name expression:"
                    f" {actuator_cfg.thruster_names_expr}."
                )
            # resolve joint indices
            # we pass a slice if all joints are selected to avoid indexing overhead
            if len(thruster_names) == self.num_joints:
                thruster_ids = slice(None)
            else:
                thruster_ids = torch.tensor(thruster_ids, device=self.device)
            # create actuator collection
            # note: for efficiency avoid indexing when over all indices
            actuator: Thruster = actuator_cfg.class_type(
                cfg=actuator_cfg,
                thruster_names=thruster_names,
                thruster_ids=thruster_ids,
                num_envs=self.num_instances,
                device=self.device,
                init_thruster_rps=self._data.default_thruster_rps
            )
            # log information on actuator groups
            omni.log.info(
                f"Actuator collection: {actuator_name} with model '{actuator_cfg.class_type.__name__}'"
                f" (thruster names: {thruster_names} [{thruster_ids}]."
            )
            # store actuator group
            self.actuators[actuator_name] = actuator


    def _apply_actuator_model(self):
        """Processes joint commands for the articulation by forwarding them to the actuators.

        The actions are first processed using actuator models. Depending on the robot configuration,
        the actuator models compute the joint level simulation commands and sets them into the PhysX buffers.
        """
        # process actions per group
        for actuator in self.actuators.values():
            # prepare input for actuator model based on cached data
            control_action = ArticulationThrustActions(
                thrusts=self._data.thrust_target[:, actuator.thruster_indices],
                thruster_indices=actuator.thruster_indices,
            )
            # compute joint command from the actuator model
            control_action = actuator.compute(
                control_action
            )
            # update targets (these are set into the simulation)
            if control_action.thrusts is not None:
                self._thrust_target_sim[:, actuator.thruster_indices] = control_action.thrusts
            # update state of the actuator model
            # -- thrusts
            self._data.computed_thrust[:, actuator.thruster_indices] = actuator.computed_thrust
            self._data.applied_thrust[:, actuator.thruster_indices] = actuator.applied_thrust


    def _combine_thrusts(self):
        """Combine individual thrusts into a wrench vector."""
        thrusts = self._thrust_target_sim
        self._internal_wrench_target_sim = (self._allocation_matrix @ thrusts.T).T
        self._internal_force_target_sim[:,0,:] = self._internal_wrench_target_sim[:,:3]
        self._internal_torque_target_sim[:,0,:] = self._internal_wrench_target_sim[:,3:]

    def find_thrusters(
        self, name_keys: str | Sequence[str], thruster_subset: list[str] | None = None, preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        """Find thrusters in the articulation based on the name keys.

        Please see the :func:`isaaclab.utils.string.resolve_matching_names` function for more information
        on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the thruster names.
            joint_subset: A subset of thrusters to search for. Defaults to None, which means all thrusters
                in the articulation are searched.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the thruster indices and names.
        """
        if thruster_subset is None:
            thruster_subset = self.joint_names
        # find thrusters
        return string_utils.resolve_matching_names(name_keys, thruster_subset, preserve_order)
    
    def _validate_cfg(self):
        """Validate the configuration after processing.

        Note:
            This function should be called only after the configuration has been processed and the buffers have been
            created. Otherwise, some settings that are altered during processing may not be validated.
            For instance, the actuator models may change the joint max velocity limits.
        """
        # check that the default values are within the limits
        for actuator_name in self.actuators:
            initial_thrust = self.actuators[actuator_name].curr_thrust
            # check that the initial thrust is within the limits
            thrust_limits = self.actuators[actuator_name].cfg.thrust_range
            if torch.any(initial_thrust < thrust_limits[0]) or torch.any(initial_thrust > thrust_limits[1]):
                raise ValueError(f"Initial thrust for actuator '{actuator_name}' is out of bounds: {initial_thrust} not in {thrust_limits}")
            
            
    def _log_articulation_info(self):
        # TODO implement logging for articulations with thruster 
        pass