# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.string as string_utils
from isaaclab.managers.action_manager import ActionTerm

from isaaclab_contrib.assets import Multirotor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

    from . import thrust_actions_cfg

# import logger
logger = logging.getLogger(__name__)

from isaaclab_contrib.controllers.lee_acceleration_control import LeeAccController
from isaaclab_contrib.controllers.lee_position_control import LeePosController
from isaaclab_contrib.controllers.lee_velocity_control import LeeVelController


class ThrustAction(ActionTerm):
    """Thrust action term that applies the processed actions as thrust commands.

    This action term is designed specifically for controlling multirotor vehicles by mapping
    action inputs to thruster commands. It provides flexible preprocessing of actions through:

    - **Scaling**: Multiply actions by a scale factor to adjust command magnitudes
    - **Offset**: Add an offset to center actions around a baseline (e.g., hover thrust)
    - **Clipping**: Constrain actions to valid ranges to prevent unsafe commands

    The action term integrates with Isaac Lab's :class:`~isaaclab.managers.ActionManager`
    framework and is specifically designed to work with :class:`~isaaclab_contrib.assets.Multirotor`
    assets.

    Key Features:
        - Supports per-thruster or uniform scaling and offsets
        - Optional automatic offset computation based on hover thrust
        - Action clipping for safety and constraint enforcement
        - Regex-based thruster selection for flexible control schemes

    Example:
        .. code-block:: python

            from isaaclab.envs import ManagerBasedRLEnvCfg
            from isaaclab_contrib.mdp.actions import ThrustActionCfg


            @configclass
            class MyEnvCfg(ManagerBasedRLEnvCfg):
                # ... other configuration ...

                @configclass
                class ActionsCfg:
                    # Direct thrust control (normalized actions)
                    thrust = ThrustActionCfg(
                        asset_name="robot",
                        scale=5.0,  # Convert [-1, 1] to [-5, 5] N
                        use_default_offset=True,  # Add hover thrust as offset
                        clip={".*": (-2.0, 8.0)},  # Clip to safe thrust range
                    )

    """

    cfg: thrust_actions_cfg.ThrustActionCfg
    """The configuration of the action term."""
    _asset: Multirotor
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor | float
    """The scaling factor applied to the input action."""
    _offset: torch.Tensor | float
    """The offset applied to the input action."""
    _clip: torch.Tensor
    """The clip applied to the input action."""

    def __init__(self, cfg: thrust_actions_cfg.ThrustActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        thruster_names_expr = self._asset.actuators["thrusters"].cfg.thruster_names_expr

        # resolve the thrusters over which the action term is applied
        self._thruster_ids, self._thruster_names = self._asset.find_bodies(
            thruster_names_expr, preserve_order=self.cfg.preserve_order
        )
        self._num_thrusters = len(self._thruster_ids)
        # log the resolved thruster names for debugging
        logger.info(
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
            index_list, _, value_list = string_utils.resolve_matching_names_values(
                self.cfg.offset, self._thruster_names
            )
            self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported offset type: {type(cfg.offset)}. Supported types are float and dict.")

        # parse clip
        if cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, self.action_dim, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(
                    self.cfg.clip, self._thruster_names
                )
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")

        # Handle use_default_offset
        if cfg.use_default_offset:
            # Use default thruster RPS as offset
            self._offset = self._asset.data.default_thruster_rps[:, self._thruster_ids].clone()

    """
    Properties
    """

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

    """
    Methods
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the action term.

        This method resets the raw actions to zero for the specified environments.
        The processed actions will be recomputed during the next :meth:`process_actions` call.

        Args:
            env_ids: Environment indices to reset. Defaults to None (all environments).
        """
        self._raw_actions[env_ids] = 0.0

    def process_actions(self, actions: torch.Tensor):
        r"""Process actions by applying scaling, offset, and clipping.

        This method transforms raw policy actions into thrust commands through
        an affine transformation followed by optional clipping. The transformation is:

        .. math::
            \text{processed} = \text{raw} \times \text{scale} + \text{offset}

        If clipping is configured, the processed actions are then clamped:

        .. math::
            \text{processed} = \text{clamp}(\text{processed}, \text{min}, \text{max})

        Args:
            actions: Raw action tensor from the policy. Shape is ``(num_envs, action_dim)``.
                Typically in the range [-1, 1] for normalized policies.

        Note:
            The processed actions are stored internally and applied during the next
            :meth:`apply_actions` call.
        """
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
        """Apply the processed actions as thrust commands.

        This method sets the processed actions as thrust targets on the multirotor
        asset. The thrust targets are then used by the thruster actuator models
        to compute actual thrust forces during the simulation step.

        The method calls :meth:`~isaaclab_contrib.assets.Multirotor.set_thrust_target`
        on the multirotor asset with the appropriate thruster IDs.
        """
        # Set thrust targets using thruster IDs
        self._asset.set_thrust_target(self.processed_actions, thruster_ids=self._thruster_ids)


class NavigationAction(ThrustAction):
    """Navigation action term that converts high-level navigation commands to thrust commands
    using a geometric tracking controller.

    This action term extends `ThrustAction` by adding a controller layer that computes wrench
    (force and torque) commands from navigation setpoints, then allocates those wrenches to
    individual thruster commands using the multirotor's allocation matrix.

    The controller type is automatically determined based on the `controller_cfg` type:
        - LeeVelControllerCfg: Velocity tracking controller
        - LeePosControllerCfg: Position tracking controller
        - LeeAccControllerCfg: Acceleration tracking controller

    The control pipeline:
        1. Process raw actions (scale, offset, clip) using parent `ThrustAction`
        2. Transform processed actions into setpoints constrained within camera FOV
        3. Compute 6-DOF wrench command using the selected Lee controller
        4. Solve thrust allocation: thrust_cmd = pinv(allocation_matrix) @ wrench_cmd
        5. Apply thrust commands to thrusters

    Attributes:
        cfg: Configuration for the navigation action term, including controller config.
        _lc: Lee controller instance (LeeVelController, LeePosController, or LeeAccController).

    Action Space:
        The action dimension is always 3D: (forward_magnitude, pitch_angle, yaw_rate)

        Actions are clipped in range [-1, 1] and are transformed to controller commands:
        - Forward position/velocity/acceleration:
            [0, max_magnitude] via (action[0] + 1) * cos(pitch) * max_magnitude / 2
        - Lateral position/velocity/acceleration:
            Always 0.0 (constrained to camera FOV)
        - Vertical position/velocity/acceleration:
            [0, max_magnitude] via (action[0] + 1) * sin(pitch) * max_magnitude / 2
        - Yaw rate: [-max_yawrate, max_yawrate] via action[2] * max_yawrate

        Where:
        - pitch angle is computed as: action[1] * max_inclination_angle

    Parameters (from cfg):
        max_magnitude: Maximum translational magnitude for position/velocity/acceleration commands.
        max_yawrate: Maximum yaw rate in rad/s.
        max_inclination_angle: Maximum pitch angle in rad.

    Notes:
        - The controller's internal states (e.g., integral terms) are reset when `reset()` is called.
        - Lateral term is constrained to 0.0 to keep commands within camera FOV.
        - The x and z components are derived from magnitude and inclination angle.
        - Requires the multirotor asset to have a valid `allocation_matrix` attribute.

    Example:
        ```python
        cfg = NavigationActionCfg(
            controller_cfg=LeeVelControllerCfg(...),
            asset_name="robot",
            max_magnitude=2.0,
            max_yawrate=1.047,  # pi/3
            max_inclination_angle=0.785,  # pi/4
        )
        nav_action = NavigationAction(cfg, env)
        ```
    """

    cfg: thrust_actions_cfg.NavigationActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: thrust_actions_cfg.NavigationActionCfg, env: ManagerBasedEnv) -> None:
        # Initialize parent class (this handles all the thruster setup)
        super().__init__(cfg, env)

        # Initialize controller based on controller_cfg type
        from isaaclab_contrib.controllers import LeeAccControllerCfg, LeePosControllerCfg, LeeVelControllerCfg

        if isinstance(self.cfg.controller_cfg, LeeVelControllerCfg):
            self._lc = LeeVelController(
                cfg=self.cfg.controller_cfg, asset=self._asset, num_envs=self.num_envs, device=self.device
            )
        elif isinstance(self.cfg.controller_cfg, LeePosControllerCfg):
            self._lc = LeePosController(
                cfg=self.cfg.controller_cfg, asset=self._asset, num_envs=self.num_envs, device=self.device
            )
        elif isinstance(self.cfg.controller_cfg, LeeAccControllerCfg):
            self._lc = LeeAccController(
                cfg=self.cfg.controller_cfg, asset=self._asset, num_envs=self.num_envs, device=self.device
            )
        else:
            raise ValueError(
                f"Unsupported controller_cfg type: {type(self.cfg.controller_cfg)}. "
                f"Supported types are LeeVelControllerCfg, LeePosControllerCfg, LeeAccControllerCfg."
            )

        # Add buffer to store velocity commands for observations)
        self._commands = torch.zeros(self.num_envs, 4, device=self.device)
        self._prev_commands = torch.zeros(self.num_envs, 4, device=self.device)

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def prev_commands(self) -> torch.Tensor:
        return self._prev_commands

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:
        """The IO descriptor of the action term."""
        # Get parent IO descriptor
        descriptor = super().IO_descriptor
        # Override action type for navigation
        descriptor.action_type = "NavigationAction"
        return descriptor

    def process_actions(self, actions: torch.Tensor):
        """Process actions by applying scaling, offset, and clipping."""
        # Call parent to handle basic processing
        super().process_actions(actions)

        self._has_actions_updated = False

    def apply_actions(self):
        """Apply the processed actions as velocity commands."""
        # process the actions to be in the correct range
        clamped_action = torch.clamp(self.processed_actions, min=-1.0, max=1.0)
        processed_actions = torch.zeros(self.num_envs, 4, device=self.device)

        clamped_action[:, 0] += 1.0  # only allow positive thrust commands [0, 2]
        processed_actions[:, 0] = (
            clamped_action[:, 0]
            * torch.cos(self.cfg.max_inclination_angle * clamped_action[:, 1])
            * self.cfg.max_magnitude
            / 2.0
        )
        processed_actions[:, 1] = 0.0  # set lateral thrust command to 0
        processed_actions[:, 2] = (
            clamped_action[:, 0]
            * torch.sin(self.cfg.max_inclination_angle * clamped_action[:, 1])
            * self.cfg.max_magnitude
            / 2.0
        )
        processed_actions[:, 3] = clamped_action[:, 2] * self.cfg.max_yawrate

        # Store velocity commands for observations
        if not self._has_actions_updated:
            self._prev_commands[:] = self._commands
            self._commands[:] = processed_actions
            self._has_actions_updated = True

        # Compute wrench command using controller
        wrench_command = self._lc.compute(processed_actions)

        # Convert wrench to thrust commands using allocation matrix
        thrust_commands = (torch.pinverse(self._asset.allocation_matrix) @ wrench_command.T).T

        # Apply thrust commands using thruster IDs
        self._asset.set_thrust_target(thrust_commands, thruster_ids=self._thruster_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # Call parent reset
        super().reset(env_ids)
        # Reset controller internal states
        self._lc.reset_idx(env_ids)

        if env_ids is not None:
            self._commands[env_ids] = 0.0
            self._prev_commands[env_ids] = 0.0
        else:
            self._commands[:] = 0.0
            self._prev_commands[:] = 0.0
