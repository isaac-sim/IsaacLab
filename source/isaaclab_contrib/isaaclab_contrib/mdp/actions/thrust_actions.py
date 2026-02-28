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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

    from isaaclab_contrib.assets import Multirotor

    from . import thrust_actions_cfg

# import logger
logger = logging.getLogger(__name__)


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
