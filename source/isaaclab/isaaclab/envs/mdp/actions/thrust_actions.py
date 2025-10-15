# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# from __future__ import annotations

# from typing import TYPE_CHECKING
# import torch
# from .joint_actions import JointAction

# if TYPE_CHECKING:
#     from isaaclab.envs import ManagerBasedEnv

#     from . import actions_cfg

# from isaaclab.controllers.lee_velocity_control import LeeVelController

# class ThrustAction(JointAction):
#     """Joint action term that applies the processed actions as thrust commands."""

#     cfg: actions_cfg.ThrustActionCfg
#     """The configuration of the action term."""
    

#     def __init__(self, cfg: actions_cfg.ThrustActionCfg, env: ManagerBasedEnv):
#         super().__init__(cfg, env)

#     def apply_actions(self):
#         # set joint thrust targets
#         # TODO still inherits Articulation instead of ArticulationWithThrusters as default. Is overwritten so doesnt matter but gives ugly warnings in VSCode. Fix in future.
#         self._asset.set_thrust_target(self.processed_actions, joint_ids=self._joint_ids)

# class NavigationAction(JointAction):
#     """Joint action term that applies the processed actions as velocity commands."""

#     cfg: actions_cfg.NavigationActionCfg
#     """The configuration of the action term."""

#     def __init__(self, cfg: actions_cfg.NavigationActionCfg, env: ManagerBasedEnv):
#         super().__init__(cfg, env)
#         if self.cfg.command_type not in ["vel", "pos", "acc"]:
#             raise ValueError(f"Unsupported command_type {self.cfg.command_type}. Supported types are 'vel', 'pos', 'acc'.")
#         elif self.cfg.command_type == "pos":
#             raise NotImplementedError("Position command type is not implemented yet.")
#         elif self.cfg.command_type == "vel":
#             pass
#         elif self.cfg.command_type == "acc":
#             raise NotImplementedError("Acceleration command type is not implemented yet.")
        
#         self._lvc = LeeVelController(cfg=self.cfg.controller_cfg, asset=self._asset, num_envs=self.num_envs, device=self.device)
        
        
#     def apply_actions(self):
#         """Apply the processed actions as velocity commands."""
#         # process the actions to be in the correct range
#         clamped_action = torch.clamp(self.processed_actions, min=-1.0, max=1.0)
#         processed_actions = torch.zeros(self.num_envs, 4, device=self.device)
#         max_speed = 2.0  # [m/s]
#         max_yawrate = torch.pi / 3.0  # [rad/s]
#         max_inclination_angle = torch.pi / 4.0  # [rad]
        
#         clamped_action[:, 0] += 1.0  # only allow positive thrust commands [0, 2]

#         processed_actions[:, 0] = (
#             clamped_action[:, 0]
#             * torch.cos(max_inclination_angle * clamped_action[:, 1])
#             * max_speed
#             / 2.0
#         )
#         processed_actions[:, 1] = 0.0  # set lateral thrust command to 0
#         processed_actions[:, 2] = (
#             clamped_action[:, 0]
#             * torch.sin(max_inclination_angle * clamped_action[:, 1])
#             * max_speed
#             / 2.0
#         )
#         processed_actions[:, 3] = clamped_action[:, 2] * max_yawrate

#         wrench_command = self._lvc.compute(processed_actions)
#         thrust_commands = (torch.pinverse(self._asset._allocation_matrix) @ wrench_command.T).T
#         self._asset.set_thrust_target(thrust_commands, joint_ids=self._joint_ids)

#     def reset(self, env_ids: torch.Tensor):
#         """Reset the controller internal states for the given environments."""
#         super().reset(env_ids)
#         self._lvc.reset_idx(env_ids)  # reset the controller internal states


# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Multirotor
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

    from . import actions_cfg

from isaaclab.controllers.lee_velocity_control import LeeVelController


class ThrustAction(ActionTerm):
    """Thrust action term that applies the processed actions as thrust commands."""

    cfg: actions_cfg.ThrustActionCfg
    """The configuration of the action term."""
    _asset: Multirotor
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor | float
    """The scaling factor applied to the input action."""
    _offset: torch.Tensor | float
    """The offset applied to the input action."""
    _clip: torch.Tensor
    """The clip applied to the input action."""

    def __init__(self, cfg: actions_cfg.ThrustActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)
        
        thruster_names_expr = self._asset.actuators["thrusters"].cfg.thruster_names_expr

        # resolve the thrusters over which the action term is applied
        self._thruster_ids, self._thruster_names = self._asset.find_bodies(
            thruster_names_expr, preserve_order=self.cfg.preserve_order
        )
        self._num_thrusters = len(self._thruster_ids)
        # log the resolved thruster names for debugging
        omni.log.info(
            f"Resolved thruster names for the action term {self.__class__.__name__}:"
            f" {self._thruster_names} [{self._thruster_ids}]"
        )

        # Avoid indexing across all thrusters for efficiency
        if self._num_thrusters == self._asset.num_thrusters and not self.cfg.preserve_order:
            self._thruster_ids = slice(None)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # parse scale
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.scale, self._thruster_names)
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")
        
        # parse offset
        if isinstance(cfg.offset, (float, int)):
            self._offset = float(cfg.offset)
        elif isinstance(cfg.offset, dict):
            self._offset = torch.zeros_like(self._raw_actions)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.offset, self._thruster_names)
            self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported offset type: {type(cfg.offset)}. Supported types are float and dict.")
        
        # parse clip 
        if cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, self.action_dim, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._thruster_names)
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")

        # Handle use_default_offset
        if cfg.use_default_offset:
            # Use default thruster RPS as offset
            self._offset = self._asset.data.default_thruster_rps[:, self._thruster_ids].clone()
        
    @property
    def action_dim(self) -> int:
        return self._num_thrusters

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:
        """The IO descriptor of the action term."""
        super().IO_descriptor
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "ThrustAction"
        self._IO_descriptor.thruster_names = self._thruster_names
        self._IO_descriptor.scale = self._scale
        if isinstance(self._offset, torch.Tensor):
            self._IO_descriptor.offset = self._offset[0].detach().cpu().numpy().tolist()
        else:
            self._IO_descriptor.offset = self._offset
        if self.cfg.clip is not None:
            if isinstance(self._clip, torch.Tensor):
                self._IO_descriptor.clip = self._clip[0].detach().cpu().numpy().tolist()
            else:
                self._IO_descriptor.clip = self._clip
        else:
            self._IO_descriptor.clip = None
        return self._IO_descriptor

    def process_actions(self, actions: torch.Tensor):
        """Process actions EXACTLY like JointAction."""
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        self._processed_actions = self._raw_actions * self._scale + self._offset
        # clip actions
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def apply_actions(self):
        """Apply the processed actions as thrust commands."""
        # Set thrust targets using thruster IDs 
        self._asset.set_thrust_target(self.processed_actions, thruster_ids=self._thruster_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset EXACTLY like JointAction."""
        self._raw_actions[env_ids] = 0.0


class NavigationAction(ThrustAction):
    """Navigation action term that applies velocity commands to multirotors."""

    cfg: actions_cfg.NavigationActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.NavigationActionCfg, env: ManagerBasedEnv) -> None:
        
        # Initialize parent class (this handles all the thruster setup)
        super().__init__(cfg, env)
        
        # Validate command type
        if self.cfg.command_type not in ["vel", "pos", "acc"]:
            raise ValueError(f"Unsupported command_type {self.cfg.command_type}. Supported types are 'vel', 'pos', 'acc'.")
        elif self.cfg.command_type == "pos":
            raise NotImplementedError("Position command type is not implemented yet.")
        elif self.cfg.command_type == "vel":
            pass
        elif self.cfg.command_type == "acc":
            raise NotImplementedError("Acceleration command type is not implemented yet.")

        # Initialize controller
        self._lvc = LeeVelController(
            cfg=self.cfg.controller_cfg, 
            asset=self._asset, 
            num_envs=self.num_envs, 
            device=self.device
        )

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:
        """The IO descriptor of the action term."""
        # Get parent IO descriptor
        descriptor = super().IO_descriptor
        # Override action type for navigation
        descriptor.action_type = "NavigationAction"
        return descriptor

    def apply_actions(self):
        """Apply the processed actions as velocity commands."""
        # process the actions to be in the correct range
        clamped_action = torch.clamp(self.processed_actions, min=-1.0, max=1.0)
        processed_actions = torch.zeros(self.num_envs, 4, device=self.device)
        max_speed = 2.0  # [m/s]
        max_yawrate = torch.pi / 3.0  # [rad/s]
        max_inclination_angle = torch.pi / 4.0  # [rad]
        
        clamped_action[:, 0] += 1.0  # only allow positive thrust commands [0, 2]

        processed_actions[:, 0] = (
            clamped_action[:, 0]
            * torch.cos(max_inclination_angle * clamped_action[:, 1])
            * max_speed
            / 2.0
        )
        processed_actions[:, 1] = 0.0  # set lateral thrust command to 0
        processed_actions[:, 2] = (
            clamped_action[:, 0]
            * torch.sin(max_inclination_angle * clamped_action[:, 1])
            * max_speed
            / 2.0
        )
        processed_actions[:, 3] = clamped_action[:, 2] * max_yawrate

        # Compute wrench command using controller
        wrench_command = self._lvc.compute(processed_actions)
        
        # Convert wrench to thrust commands using allocation matrix
        thrust_commands = (torch.pinverse(self._asset._allocation_matrix) @ wrench_command.T).T
        
        # Apply thrust commands using thruster IDs
        self._asset.set_thrust_target(thrust_commands, thruster_ids=self._thruster_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset EXACTLY like JointAction."""
        # Call parent reset
        super().reset(env_ids)
        # Reset controller internal states
        self._lvc.reset_idx(env_ids)