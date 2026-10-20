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
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.controllers import AckermannController
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

    from . import actions_cfg


logger = logging.getLogger(__name__)


class AckermannAction(ActionTerm):
    r"""Map vehicle commands to physical steering and wheel joints.

    The raw two-dimensional action contains longitudinal speed :math:`v_{B,x}` [m/s] at the center of the
    non-steerable axle, expressed in the vehicle body frame, and virtual steering angle :math:`\delta` [rad]. This
    reference point applies to both front- and rear-steering vehicles. The configured
    :class:`~isaaclab.controllers.AckermannController` computes two steering-joint position targets [rad] and one
    angular-velocity target [rad/s] per wheel.

    Steering joints are ordered left then right. Wheel joints are ordered as the left and right steerable-wheel
    spin joints followed by non-steerable-wheel spin joints in the order of
    :attr:`~isaaclab.controllers.AckermannControllerCfg.non_steerable_wheel_offsets`. Joint direction multipliers
    make this logical order independent of articulation joint-axis signs.
    """

    cfg: actions_cfg.AckermannActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""

    def __init__(self, cfg: actions_cfg.AckermannActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        self._steering_joint_ids, self._steering_joint_names = self._asset.find_joints(
            self.cfg.steering_joint_names, preserve_order=True
        )
        self._wheel_joint_ids, self._wheel_joint_names = self._asset.find_joints(
            self.cfg.wheel_joint_names, preserve_order=True
        )
        expected_num_wheels = 2 + len(self.cfg.controller.non_steerable_wheel_offsets)
        if len(self._steering_joint_ids) != 2:
            raise ValueError(
                "steering_joint_names must resolve exactly two joints ordered left then right, "
                f"got {self._steering_joint_names}."
            )
        if len(self._wheel_joint_ids) != expected_num_wheels:
            raise ValueError(
                f"wheel_joint_names must resolve exactly {expected_num_wheels} joints ordered as the left and right "
                "steerable-wheel spin joints followed by the configured non-steerable wheels, "
                f"got {self._wheel_joint_names}."
            )
        if len(set(self._steering_joint_ids)) != len(self._steering_joint_ids):
            raise ValueError(f"steering_joint_names resolved duplicate joints: {self._steering_joint_names}.")
        if len(set(self._wheel_joint_ids)) != len(self._wheel_joint_ids):
            raise ValueError(f"wheel_joint_names resolved duplicate joints: {self._wheel_joint_names}.")
        if set(self._steering_joint_ids).intersection(self._wheel_joint_ids):
            raise ValueError("Steering position joints and wheel spin joints must be distinct.")

        self._validate_joint_directions(self.cfg.steering_joint_directions, 2, "steering_joint_directions")
        self._validate_joint_directions(self.cfg.wheel_joint_directions, expected_num_wheels, "wheel_joint_directions")
        self._steering_joint_directions = torch.tensor(
            self.cfg.steering_joint_directions, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        self._wheel_joint_directions = torch.tensor(
            self.cfg.wheel_joint_directions, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        controller_type = self.cfg.controller.class_type
        self._controller: AckermannController = controller_type(
            cfg=self.cfg.controller, num_envs=self.num_envs, device=self.device
        )

        self._raw_actions = torch.zeros((self.num_envs, self.action_dim), dtype=torch.float32, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._steering_joint_targets = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self._wheel_joint_targets = torch.zeros(
            (self.num_envs, expected_num_wheels), dtype=torch.float32, device=self.device
        )
        self._scale = torch.tensor(self.cfg.scale, dtype=torch.float32, device=self.device).unsqueeze(0)
        self._offset = torch.tensor(self.cfg.offset, dtype=torch.float32, device=self.device).unsqueeze(0)
        self._clip: torch.Tensor | None = None
        if self.cfg.clip is not None:
            if not isinstance(self.cfg.clip, dict):
                raise TypeError(f"clip must be a dictionary, got {type(self.cfg.clip)}.")
            self._clip = torch.tensor(
                [[[-torch.inf, torch.inf], [-torch.inf, torch.inf]]], dtype=torch.float32, device=self.device
            )
            index_list, _, value_list = string_utils.resolve_matching_names_values(
                self.cfg.clip, ["linear_speed", "steering_angle"]
            )
            self._clip[:, index_list] = torch.tensor(value_list, dtype=torch.float32, device=self.device)

        logger.info(
            f"Resolved steering joints for {self.__class__.__name__}: "
            f"{self._steering_joint_names} [{self._steering_joint_ids}]"
        )
        logger.info(
            f"Resolved wheel joints for {self.__class__.__name__}: {self._wheel_joint_names} [{self._wheel_joint_ids}]"
        )

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """Dimension of the action term."""
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        """Unprocessed non-steerable-axle speed [m/s] and virtual steering-angle [rad] actions."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Processed non-steerable-axle speed [m/s] and virtual steering-angle [rad] actions."""
        return self._processed_actions

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:
        """Return the action IO descriptor."""
        super().IO_descriptor
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "AckermannAction"
        self._IO_descriptor.units = ["m/s", "rad"]
        self._IO_descriptor.element_names = ["linear_speed", "steering_angle"]
        self._IO_descriptor.scale = self._scale
        self._IO_descriptor.offset = self._offset
        self._IO_descriptor.clip = self._clip
        self._IO_descriptor.steering_joint_names = self._steering_joint_names
        self._IO_descriptor.wheel_joint_names = self._wheel_joint_names
        return self._IO_descriptor

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor) -> None:
        """Process raw vehicle commands.

        Args:
            actions: Longitudinal speed at the non-steerable axle center [m/s] and virtual steering angle [rad],
                shape ``(N, 2)``.
        """
        self._raw_actions.copy_(actions)
        torch.mul(self._raw_actions, self._scale, out=self._processed_actions)
        self._processed_actions.add_(self._offset)
        finite_rows = torch.isfinite(self._processed_actions).all(dim=1)
        if self._clip is not None:
            torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
                out=self._processed_actions,
            )
        # Clipping would otherwise turn an infinite command into finite full-speed/full-lock motion. Preserve the
        # controller's fail-safe policy by zeroing the complete command pair when either processed component was
        # non-finite before clipping.
        self._processed_actions[~finite_rows] = 0.0

    def apply_actions(self) -> None:
        """Compute and apply steering-position [rad] and wheel-velocity [rad/s] targets."""
        steering_targets, wheel_targets = self._controller.compute(self._processed_actions)
        torch.mul(steering_targets, self._steering_joint_directions, out=self._steering_joint_targets)
        torch.mul(wheel_targets, self._wheel_joint_directions, out=self._wheel_joint_targets)
        self._asset.set_joint_position_target_index(
            target=self._steering_joint_targets, joint_ids=self._steering_joint_ids
        )
        self._asset.set_joint_velocity_target_index(target=self._wheel_joint_targets, joint_ids=self._wheel_joint_ids)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Reset commands for selected environments.

        Args:
            env_ids: Environment indices to reset. Defaults to all environments.
        """
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._controller.reset(env_ids)

    """
    Helper methods.
    """

    @staticmethod
    def _validate_joint_directions(directions: tuple[float, ...], expected_length: int, name: str) -> None:
        """Validate ordered joint-axis direction multipliers."""
        if len(directions) != expected_length:
            raise ValueError(f"{name} must contain {expected_length} entries, got {len(directions)}.")
        if not all(direction in (-1.0, 1.0) for direction in directions):
            raise ValueError(f"{name} entries must be either -1.0 or 1.0, got {directions}.")
