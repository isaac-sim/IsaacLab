# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Flag for pyright to ignore type errors in this file.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation

from isaaclab_contrib.actuators import Thruster
from isaaclab_contrib.utils.types import MultiRotorActions

from .multirotor_data import MultirotorData

if TYPE_CHECKING:
    from .multirotor_cfg import MultirotorCfg

# import logger
logger = logging.getLogger(__name__)


class Multirotor(Articulation):
    """A multirotor articulation asset class.

    This class extends the base articulation class to support multirotor vehicles
    with thruster actuators that can be applied at specific body locations.
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
        if not hasattr(self, "actuators") or not self.actuators:
            return []

        thruster_names = []
        for actuator in self.actuators.values():
            if hasattr(actuator, "thruster_names"):
                thruster_names.extend(actuator.thruster_names)
            else:
                raise ValueError("Non thruster actuator found in multirotor actuators. Not supported at the moment.")

        return thruster_names

    @property
    def num_thrusters(self) -> int:
        """Number of thrusters in the multirotor."""
        return len(self.thruster_names)

    @property
    def allocation_matrix(self) -> torch.Tensor:
        """Allocation matrix for control allocation."""
        return torch.tensor(self.cfg.allocation_matrix, device=self.device, dtype=torch.float32)

    """
    Operations
    """

    def set_thrust_target(
        self,
        target: torch.Tensor,
        thruster_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
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
            env_ids = self._ALL_INDICES
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)

        # reset thruster targets to default values
        if self._data.thrust_target is not None and self._data.default_thruster_rps is not None:
            self._data.thrust_target[env_ids] = self._data.default_thruster_rps[env_ids]

    def write_data_to_sim(self):
        """Write thrust and torque commands to the simulation."""
        self._apply_actuator_model()
        # apply thruster forces at individual locations
        self._apply_combined_wrench()

    def _initialize_impl(self):
        """Initialize the multirotor implementation."""
        # call parent initialization
        super()._initialize_impl()

        # Replace data container with MultirotorData
        self._data = MultirotorData(self.root_physx_view, self.device)

        # Create thruster buffers with correct size (SINGLE PHASE)
        self._create_thruster_buffers()

        # Process thruster configuration
        self._process_thruster_cfg()

        # Process configuration
        self._process_cfg()

        # Update the robot data
        self.update(0.0)

        # Log multirotor information
        self._log_multirotor_info()

    def _count_thrusters_from_config(self) -> int:
        """Count total number of thrusters from actuator configuration.

        Returns:
            Total number of thrusters across all actuator groups.
        """
        total_thrusters = 0

        for actuator_name, actuator_cfg in self.cfg.actuators.items():
            if not hasattr(actuator_cfg, "thruster_names_expr"):
                continue

            # Use find_bodies to count thrusters for this actuator
            body_indices, thruster_names = self.find_bodies(actuator_cfg.thruster_names_expr)
            total_thrusters += len(body_indices)

        if total_thrusters == 0:
            raise ValueError(
                "No thrusters found in actuator configuration. "
                "Please check thruster_names_expr in your MultirotorCfg.actuators."
            )

        return total_thrusters

    def _create_thruster_buffers(self):
        """Create thruster buffers with correct size."""
        num_instances = self.num_instances
        num_thrusters = self._count_thrusters_from_config()

        # Create thruster data tensors with correct size
        self._data.default_thruster_rps = torch.zeros(num_instances, num_thrusters, device=self.device)
        # thrust after controller and allocation is applied
        self._data.thrust_target = torch.zeros(num_instances, num_thrusters, device=self.device)
        self._data.computed_thrust = torch.zeros(num_instances, num_thrusters, device=self.device)
        self._data.applied_thrust = torch.zeros(num_instances, num_thrusters, device=self.device)

        # Combined wrench buffers
        self._thrust_target_sim = torch.zeros_like(self._data.thrust_target)  # thrust after actuator model is applied
        # wrench target for combined mode
        self._internal_wrench_target_sim = torch.zeros(num_instances, 6, device=self.device)
        # internal force/torque targets per body for combined mode
        self._internal_force_target_sim = torch.zeros(num_instances, self.num_bodies, 3, device=self.device)
        self._internal_torque_target_sim = torch.zeros(num_instances, self.num_bodies, 3, device=self.device)

        # Placeholder thruster names (will be filled during actuator creation)
        self._data.thruster_names = [f"thruster_{i}" for i in range(num_thrusters)]

    def _process_actuators_cfg(self):
        """Override parent method to do nothing - we handle thrusters separately."""
        # Do nothing - we handle thruster processing in _process_thruster_cfg() otherwise this
        # gives issues with joint name expressions
        pass

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
        if hasattr(self._data, "default_thruster_rps") and hasattr(self.cfg.init_state, "rps"):
            # Match against thruster names
            indices_list, _, values_list = string_utils.resolve_matching_names_values(
                self.cfg.init_state.rps, self.thruster_names
            )
            if indices_list:
                rps_values = torch.tensor(values_list, device=self.device)
                self._data.default_thruster_rps[:, indices_list] = rps_values
                self._data.thrust_target[:, indices_list] = rps_values

    def _process_thruster_cfg(self):
        """Process and apply multirotor thruster properties."""
        # create actuators
        self.actuators = dict()
        self._has_implicit_actuators = False

        # Check for mixed configurations (same as before)
        has_thrusters = False
        has_joints = False

        for actuator_name, actuator_cfg in self.cfg.actuators.items():
            if hasattr(actuator_cfg, "thruster_names_expr"):
                has_thrusters = True
            elif hasattr(actuator_cfg, "joint_names_expr"):
                has_joints = True

        if has_thrusters and has_joints:
            raise ValueError("Mixed configurations with both thrusters and regular joints are not supported.")

        if has_joints:
            raise ValueError("Regular joint actuators are not supported in Multirotor class.")

        # Store the body-to-thruster mapping
        self._thruster_body_mapping = {}

        # Track thruster names as we create actuators
        all_thruster_names = []

        for actuator_name, actuator_cfg in self.cfg.actuators.items():
            body_indices, thruster_names = self.find_bodies(actuator_cfg.thruster_names_expr)

            # Create 0-based thruster array indices starting from current count
            start_idx = len(all_thruster_names)
            thruster_array_indices = list(range(start_idx, start_idx + len(body_indices)))

            # Track all thruster names
            all_thruster_names.extend(thruster_names)

            # Store the mapping
            self._thruster_body_mapping[actuator_name] = {
                "body_indices": body_indices,
                "array_indices": thruster_array_indices,
                "thruster_names": thruster_names,
            }

            # Create thruster actuator
            actuator: Thruster = actuator_cfg.class_type(
                cfg=actuator_cfg,
                thruster_names=thruster_names,
                thruster_ids=thruster_array_indices,
                num_envs=self.num_instances,
                device=self.device,
                init_thruster_rps=self._data.default_thruster_rps[:, thruster_array_indices],
            )

            # Store actuator
            self.actuators[actuator_name] = actuator

            # Log information
            logger.info(
                f"Thruster actuator: {actuator_name} with model '{actuator_cfg.class_type.__name__}'"
                f" (thruster names: {thruster_names} [{body_indices}])."
            )

        # Update thruster names in data container
        self._data.thruster_names = all_thruster_names

        # Log summary
        logger.info(f"Initialized {len(self.actuators)} thruster actuator(s) for multirotor.")

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
            control_action = MultiRotorActions(
                thrusts=self._data.thrust_target[:, actuator.thruster_indices],
                thruster_indices=actuator.thruster_indices,
            )

            # compute thruster command from the actuator model
            control_action = actuator.compute(control_action)

            # update targets (these are set into the simulation)
            if control_action.thrusts is not None:
                self._thrust_target_sim[:, actuator.thruster_indices] = control_action.thrusts

            # update state of the actuator model
            self._data.computed_thrust[:, actuator.thruster_indices] = actuator.computed_thrust
            self._data.applied_thrust[:, actuator.thruster_indices] = actuator.applied_thrust

    def _apply_combined_wrench(self):
        """Apply combined wrench to the base link (like articulation_with_thrusters.py)."""
        # Combine individual thrusts into a wrench vector
        self._combine_thrusts()

        self.root_physx_view.apply_forces_and_torques_at_position(
            force_data=self._internal_force_target_sim.view(-1, 3),  # Shape: (num_envs * num_bodies, 3)
            torque_data=self._internal_torque_target_sim.view(-1, 3),  # Shape: (num_envs * num_bodies, 3)
            position_data=None,  # Apply at center of mass
            indices=self._ALL_INDICES,
            is_global=False,  # Forces are in local frame
        )

    def _combine_thrusts(self):
        """Combine individual thrusts into a wrench vector."""
        thrusts = self._thrust_target_sim
        self._internal_wrench_target_sim = (self.allocation_matrix @ thrusts.T).T
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
        if hasattr(self, "actuators") and self.actuators:
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
        logger.info(f"Multirotor initialized with {self.num_thrusters} thrusters")
        logger.info(f"Thruster names: {self.thruster_names}")
        logger.info(f"Thruster force direction: {self.cfg.thruster_force_direction}")
