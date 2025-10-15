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
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.utils.types import ArticulationThrustActions
from isaaclab.actuators import ThrusterCfg, Thruster, ActuatorBase
from isaaclab.assets.articulation.multirotor_data import MultirotorData

if TYPE_CHECKING:
    from .multirotor_cfg import MultirotorCfg


class Multirotor(Articulation):
    """A multirotor articulation asset class.

    This class extends the base articulation class to support multirotor vehicles
    with thruster actuators that can be applied at specific body locations. It maintains 
    full compatibility with the base articulation functionality while adding 
    multirotor-specific features.

    The class supports various multirotor configurations (quadcopter, hexacopter, etc.)
    and applies individual thruster forces at their respective body locations rather
    than using aggregated wrench approaches.
    """

    cfg: MultirotorCfg
    """Configuration instance for the multirotor."""

    actuators: dict[str, Thruster]
    """Dictionary of thruster actuator instances for the multirotor.

    The keys are the actuator names and the values are the actuator instances. The actuator instances
    are initialized based on the actuator configurations specified in the :attr:`MultirotorCfg.actuators`
    attribute. They are used to compute the thruster commands during the :meth:`write_data_to_sim` function.
    """

    def __init__(self, cfg: MultirotorCfg):
        """Initialize the multirotor articulation.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)

    """
    Properties
    """

    @property
    def thruster_names(self) -> list[str]:
        """Ordered names of thrusters in the multirotor."""
        if not hasattr(self, 'actuators') or not self.actuators:
            return []
        
        thruster_names = []
        for actuator in self.actuators.values():
            if hasattr(actuator, 'thruster_names'):
                thruster_names.extend(actuator.thruster_names)
        
        return thruster_names

    @property
    def num_thrusters(self) -> int:
        """Number of thrusters in the multirotor."""
        return len(self.thruster_names)
    
    @property
    def _allocation_matrix(self) -> torch.Tensor:
        """Allocation matrix for control allocation."""
        return torch.tensor(self.cfg.allocation_matrix, device=self.device, dtype=torch.float32)
    
    """
    Operations 
    """

    def find_thrusters(
        self, 
        name_keys: str | Sequence[str], 
        thruster_subset: list[str] | None = None, 
        preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        """Find thrusters in the multirotor based on the name keys.

        This method identifies thrusters based on body names containing 'thruster', 'prop', 
        'motor', 'rotor', or explicit naming patterns common in multirotor configurations.

        Args:
            name_keys: A regular expression or list of regular expressions to match thruster names.
            thruster_subset: A subset of thrusters to search for. Defaults to None.
            preserve_order: Whether to preserve the order of name keys in output. Defaults to False.

        Returns:
            A tuple of lists containing the thruster indices and names.

        Raises:
            ValueError: If no thrusters are found matching the criteria.
        """
        # Identify potential thruster bodies by common multirotor naming patterns
        if thruster_subset is None:
            # Look for bodies with multirotor-related keywords
            multirotor_keywords = ["thruster", "prop", "motor", "rotor"]
            thruster_subset = [
                name for name in self.body_names 
                if any(keyword in name.lower() for keyword in multirotor_keywords)
            ]
            # If no bodies found, raise an error instead of falling back to joints
            if not thruster_subset:
                raise ValueError(
                    f"No thruster bodies found in body names: {self.body_names}. "
                    f"Thruster bodies should contain one of: {multirotor_keywords}"
                )
        
        return string_utils.resolve_matching_names(name_keys, thruster_subset, preserve_order)

    def set_thrust_target(
        self, 
        target: torch.Tensor, 
        thruster_ids: Sequence[int] | slice | None = None, 
        env_ids: Sequence[int] | None = None
    ):
        """Set target thrust values for thrusters.

        Args:
            target: Target thrust values. Shape is (num_envs, num_thrusters) or (num_envs,).
            thruster_ids: Indices of thrusters to set. Defaults to None (all thrusters).
            env_ids: Environment indices to set. Defaults to None (all environments).
        """
        # resolve indices
        if env_ids is None:
            env_ids = slice(None)
        if thruster_ids is None:
            thruster_ids = slice(None)
        
        # broadcast env_ids if needed to allow double indexing
        if env_ids != slice(None) and thruster_ids != slice(None):
            env_ids = env_ids[:, None]
        
        # set targets
        self._data.thrust_target[env_ids, thruster_ids] = target

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset the multirotor to default state.

        Args:
            env_ids: Environment indices to reset. Defaults to None (all environments).
        """
        # call parent reset
        super().reset(env_ids)
        
        # reset multirotor-specific data
        if env_ids is None:
            env_ids = torch.arange(self.num_instances, device=self.device)
        else:
            env_ids = torch.tensor(env_ids, device=self.device)
        
        # reset thruster targets to default values
        if self._data.thrust_target is not None and self._data.default_thruster_rps is not None:
            self._data.thrust_target[env_ids] = self._data.default_thruster_rps[env_ids]

    def write_data_to_sim(self):
        """Write thrust and torque commands to the simulation..
        """
        self._apply_actuator_model()
        # apply thruster forces at individual locations
        self._apply_thruster_forces_and_torques()


    def _initialize_impl(self):
        """Initialize the multirotor implementation."""
        # call parent initialization
        super()._initialize_impl()
        
        # Replace data container with MultirotorData
        self._data = MultirotorData(self.root_physx_view, self.device)
        
        # Create thruster buffers
        self._create_thruster_buffers()
        
        # Process thruster configuration (this replaces the parent's actuator processing)
        self._process_thruster_cfg()
        
        # Process configuration
        self._process_cfg()
        
        # Update the robot data
        self.update(0.0)
        
        # Log multirotor information
        self._log_multirotor_info()
    
    def _process_actuators_cfg(self):
        """Override parent method to do nothing - we handle thrusters separately."""
        # Do nothing - we handle thruster processing in _process_thruster_cfg() otherwise this
        # gives issues with joint name expressions
        pass

    def _create_thruster_buffers(self):
        """Create thruster buffers before actuators are processed."""
        # Create placeholder buffers with reasonable defaults
        num_instances = self.num_instances
        num_thrusters = 4  # Reasonable default for most multirotors
        
        # Create placeholder thruster data tensors
        self._data.default_thruster_rps = torch.zeros(num_instances, num_thrusters, device=self.device)
        self._data.thrust_target = torch.zeros(num_instances, num_thrusters, device=self.device)
        self._data.computed_thrust = torch.zeros(num_instances, num_thrusters, device=self.device)
        self._data.applied_thrust = torch.zeros(num_instances, num_thrusters, device=self.device)
        
        # FIX: Combined wrench buffers with correct shape
        self._thrust_target_sim = torch.zeros_like(self._data.thrust_target)
        self._internal_wrench_target_sim = torch.zeros(num_instances, 6, device=self.device)
        # Shape should match the external force buffers: (num_envs, num_bodies, 3)
        self._internal_force_target_sim = torch.zeros(num_instances, self.num_bodies, 3, device=self.device)
        self._internal_torque_target_sim = torch.zeros(num_instances, self.num_bodies, 3, device=self.device)
        
        # Placeholder thruster names
        self._data.thruster_names = [f"thruster_{i}" for i in range(num_thrusters)]

    def _update_thruster_buffers(self):
        """Update thruster buffers with actual data after actuators are created."""
        # Get actual number of thrusters
        num_instances = self.num_instances
        num_thrusters = self.num_thrusters
        
        # Resize buffers to match actual thruster count
        self._data.default_thruster_rps = torch.zeros(num_instances, num_thrusters, device=self.device)
        self._data.thrust_target = torch.zeros(num_instances, num_thrusters, device=self.device)
        self._data.computed_thrust = torch.zeros(num_instances, num_thrusters, device=self.device)
        self._data.applied_thrust = torch.zeros(num_instances, num_thrusters, device=self.device)
        
        # Set actual thruster names
        self._data.thruster_names = self.thruster_names

    def _process_cfg(self):
        """Post processing of multirotor configuration parameters."""
        # Handle root state (like parent does)
        default_root_state = (
            tuple(self.cfg.init_state.pos)
            + tuple(self.cfg.init_state.rot)
            + tuple(self.cfg.init_state.lin_vel)
            + tuple(self.cfg.init_state.ang_vel)
        )
        default_root_state = torch.tensor(default_root_state, dtype=torch.float, device=self.device)
        self._data.default_root_state = default_root_state.repeat(self.num_instances, 1)
        
        # Handle thruster-specific initial state
        if hasattr(self._data, 'default_thruster_rps') and hasattr(self.cfg.init_state, 'rps'):
            # Match against thruster names (not body names!)
            indices_list, _, values_list = string_utils.resolve_matching_names_values(
                self.cfg.init_state.rps, self.thruster_names  # ← Use thruster_names, not body_names
            )
            if indices_list:
                rps_values = torch.tensor(values_list, device=self.device)
                self._data.default_thruster_rps[:, indices_list] = rps_values
                self._data.thrust_target[:, indices_list] = rps_values

    def _process_thruster_cfg(self):
        """Process and apply multirotor thruster properties.
        
        This method only handles thruster actuators. Mixed configurations with
        both thrusters and regular joints are not supported.
        """
        # create actuators
        self.actuators = dict()
        # flag for implicit actuators (not used for thrusters)
        self._has_implicit_actuators = False

        # Check for mixed configurations
        has_thrusters = False
        has_joints = False
        
        for actuator_name, actuator_cfg in self.cfg.actuators.items():
            if hasattr(actuator_cfg, 'thruster_names_expr'):
                has_thrusters = True
            elif hasattr(actuator_cfg, 'joint_names_expr'):
                has_joints = True
        
        # Throw error for mixed configurations
        if has_thrusters and has_joints:
            raise ValueError(
                "Mixed configurations with both thrusters and regular joints are not supported. "
                "Please use either pure thruster configuration or pure joint configuration."
            )
        
        # If we have regular joints, throw error (not supported in Multirotor class)
        if has_joints:
            raise ValueError(
                "Regular joint actuators are not supported in Multirotor class. "
                "Please use the base Articulation class for joint-based systems."
            )
            
        # Store the body-to-thruster mapping at the class level
        self._thruster_body_mapping = {}
        
        for actuator_name, actuator_cfg in self.cfg.actuators.items():
            # Find thruster bodies
            body_indices, thruster_names = self.find_bodies(actuator_cfg.thruster_names_expr)
            
            # Create 0-based thruster array indices
            thruster_array_indices = list(range(len(body_indices)))
            
            # Store the mapping
            self._thruster_body_mapping[actuator_name] = {
                'body_indices': body_indices,  # [1, 2, 3, 4]
                'array_indices': thruster_array_indices,  # [0, 1, 2, 3]
                'thruster_names': thruster_names
            }
            
            # Create thruster actuator
            actuator: Thruster = actuator_cfg.class_type(
                cfg=actuator_cfg,
                thruster_names=thruster_names,
                thruster_ids=thruster_array_indices,
                num_envs=self.num_instances,
                device=self.device,
                init_thruster_rps=self._data.default_thruster_rps
            )
            
            # Store actuator
            self.actuators[actuator_name] = actuator
            
            # Log information
            omni.log.info(
                f"Thruster actuator: {actuator_name} with model '{actuator_cfg.class_type.__name__}'"
                f" (thruster names: {thruster_names} [{body_indices}])."
            )

        # Log summary
        omni.log.info(f"Initialized {len(self.actuators)} thruster actuator(s) for multirotor.")

    def _apply_actuator_model(self):
        """Processes thruster commands for the multirotor by forwarding them to the actuators.

        The actions are first processed using actuator models. The thruster actuator models
        compute the thruster level simulation commands and sets them into the PhysX buffers.
        """
        
        # process thruster actions per group
        for actuator in self.actuators.values():
            if not isinstance(actuator, Thruster):
                continue
            
            # prepare input for actuator model based on cached data
            control_action = ArticulationThrustActions(
                thrusts=self._data.thrust_target[:, actuator.thruster_indices],
                thruster_indices=actuator.thruster_indices,
            )

            # compute thruster command from the actuator model
            control_action = actuator.compute(control_action)
            
            # update targets (these are set into the simulation)
            if control_action.thrusts is not None:
                self._data.thrust_target[:, actuator.thruster_indices] = control_action.thrusts
            
            # update state of the actuator model
            self._data.computed_thrust[:, actuator.thruster_indices] = actuator.computed_thrust
            self._data.applied_thrust[:, actuator.thruster_indices] = actuator.applied_thrust

    def _apply_thruster_forces_and_torques(self):
        """Apply thruster forces based on the configured mode."""
        if self.cfg.force_application_mode == "individual":
            self._apply_individual_thruster_forces_and_torques()
        elif self.cfg.force_application_mode == "combined":
            self._apply_combined_wrench()
        else:
            raise ValueError(f"Unknown force application mode: {self.cfg.force_application_mode}")
    
    def _apply_individual_thruster_forces_and_torques(self):
        """Apply individual thruster forces at their respective body locations."""
        # get current body poses 
        body_poses = self.data.body_link_pose_w
        
        # Create full force/torque tensors for ALL bodies across ALL environments
        # Shape: (num_envs * num_bodies, 3) = (48 * 5, 3) = (240, 3)
        all_forces = torch.zeros(self.num_instances * self.num_bodies, 3, device=self.device)
        all_torques = torch.zeros(self.num_instances * self.num_bodies, 3, device=self.device)
        all_positions = torch.zeros(self.num_instances * self.num_bodies, 3, device=self.device)
        
        # apply forces for each thruster actuator group
        for actuator_name, actuator in self.actuators.items():
            if not isinstance(actuator, Thruster):
                continue
                
            # Get the mapping for this actuator
            mapping = self._thruster_body_mapping[actuator_name]
            body_indices = mapping['body_indices']  # [1, 2, 3, 4]
            
            # get thruster forces from actuator
            thruster_forces = actuator.applied_thrust  # Shape: (num_envs, 4)
            
            # apply forces at each thruster location
            for i, thruster_name in enumerate(actuator.thruster_names):
                body_idx = body_indices[i]  # Use the stored body index from mapping
                        
                # get body pose and orientation
                body_pos = body_poses[:, body_idx, :3]  # (num_envs, 3)
                body_quat = body_poses[:, body_idx, 3:7]  # (num_envs, 4)
                
                # get thruster force magnitude
                force_magnitude = thruster_forces[:, i]  # (num_envs,)
                torque_magnitude = force_magnitude * actuator.cfg.torque_to_thrust_ratio  # (num_envs,)
                
                # transform force direction to world frame (IsaacLab math pattern)
                local_force_dir = torch.tensor(
                    self.cfg.thruster_force_direction, device=self.device
                ).expand(self.num_instances, -1)  # Shape: (num_envs, 3)
                
                world_force_dir = math_utils.quat_apply(body_quat, local_force_dir)  # (num_envs, 3)
                
                # calculate world force
                world_force = world_force_dir * force_magnitude.unsqueeze(-1)  # (num_envs, 3)
                
                # Get rotor direction for this thruster
                rotor_direction = 1.0  # default counter-clockwise
                if self.cfg.rotor_directions is not None and i < len(self.cfg.rotor_directions):
                    rotor_direction = float(self.cfg.rotor_directions[i])
                
                # transform torque direction to world frame (same as force direction)
                world_torque_dir = world_force_dir
                world_torque = world_torque_dir * torque_magnitude.unsqueeze(-1) * rotor_direction  # (num_envs, 3)
                
                # Calculate the flattened indices for this body across all environments
                # Format: [env0_body_idx, env1_body_idx, env2_body_idx, ...]
                indices = torch.arange(self.num_instances, device=self.device) * self.num_bodies + body_idx
                
                # Set forces only for this specific body across all environments
                all_forces[indices] = world_force
                all_torques[indices] = world_torque
                all_positions[indices] = body_pos
        
        # Apply all forces at once using ALL_INDICES
        self.root_physx_view.apply_forces_and_torques_at_position(
            force_data=all_forces,  # Shape: (240, 3)
            torque_data=all_torques,  # Shape: (240, 3)
            position_data=all_positions,  # Shape: (240, 3)
            indices=self._ALL_INDICES,  # Use ALL_INDICES like IsaacLab examples
            is_global=True  # position is in world frame
        )
    
    def _apply_combined_wrench(self):
        """Apply combined wrench to the base link (like articulation_with_thrusters.py)."""
        # Combine individual thrusts into a wrench vector
        self._combine_thrusts()
        
        # Apply the combined wrench to the base link (body index 0)
        # Use ALL_INDICES like the articulation_with_thrusters.py example
        self.root_physx_view.apply_forces_and_torques_at_position(
            force_data=self._internal_force_target_sim.view(-1, 3),  # Shape: (num_envs * num_bodies, 3)
            torque_data=self._internal_torque_target_sim.view(-1, 3),  # Shape: (num_envs * num_bodies, 3)
            position_data=None,  # Apply at center of mass
            indices=self._ALL_INDICES,  # Use ALL_INDICES
            is_global=False  # Forces are in local frame
        )
    
    def _combine_thrusts(self):
        """Combine individual thrusts into a wrench vector."""
        thrusts = self._thrust_target_sim
        self._internal_wrench_target_sim = (self._allocation_matrix @ thrusts.T).T
        # Apply forces to base link (body index 0) only
        self._internal_force_target_sim[:, 0, :] = self._internal_wrench_target_sim[:, :3]
        self._internal_torque_target_sim[:, 0, :] = self._internal_wrench_target_sim[:, 3:]
    
    def _validate_cfg(self):
        """Validate the multirotor configuration after processing.

        Note:
            This function should be called only after the configuration has been processed and the buffers have been
            created. Otherwise, some settings that are altered during processing may not be validated.
        """
        # Only validate if actuators have been created
        if hasattr(self, 'actuators') and self.actuators:
            # Validate thruster-specific configuration
            for actuator_name in self.actuators:
                if isinstance(self.actuators[actuator_name], Thruster):
                    initial_thrust = self.actuators[actuator_name].curr_thrust
                    # check that the initial thrust is within the limits
                    thrust_limits = self.actuators[actuator_name].cfg.thrust_range
                    if torch.any(initial_thrust < thrust_limits[0]) or torch.any(initial_thrust > thrust_limits[1]):
                        raise ValueError(
                            f"Initial thrust for actuator '{actuator_name}' is out of bounds: "
                            f"{initial_thrust} not in {thrust_limits}"
                        )

    def _log_multirotor_info(self):
        """Log multirotor-specific information."""
        omni.log.info(f"Multirotor initialized with {self.num_thrusters} thrusters")
        omni.log.info(f"Thruster names: {self.thruster_names}")
        omni.log.info(f"Thruster force direction: {self.cfg.thruster_force_direction}")