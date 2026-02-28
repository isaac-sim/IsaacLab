# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions specific to the in-hand dexterous manipulation environments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
import warp as wp

from isaaclab.assets import Articulation
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class reset_joints_within_limits_range(ManagerTermBase):
    """Reset an articulation's joints to a random position in the given limit ranges.

    This function samples random values for the joint position and velocities from the given limit ranges.
    The values are then set into the physics simulation.

    The parameters to the function are:

    * :attr:`position_range` - a dictionary of position ranges for each joint. The keys of the dictionary are the
      joint names (or regular expressions) of the asset.
    * :attr:`velocity_range` - a dictionary of velocity ranges for each joint. The keys of the dictionary are the
      joint names (or regular expressions) of the asset.
    * :attr:`use_default_offset` - a boolean flag to indicate if the ranges are offset by the default joint state.
      Defaults to False.
    * :attr:`asset_cfg` - the configuration of the asset to reset. Defaults to the entity named "robot" in the scene.
    * :attr:`operation` - whether the ranges are scaled values of the joint limits, or absolute limits.
       Defaults to "abs".

    The dictionary values are a tuple of the form ``(a, b)``. Based on the operation, these values are
    interpreted differently:

    * If the operation is "abs", the values are the absolute minimum and maximum values for the joint, i.e.
      the joint range becomes ``[a, b]``.
    * If the operation is "scale", the values are the scaling factors for the joint limits, i.e. the joint range
      becomes ``[a * min_joint_limit, b * max_joint_limit]``.

    If the ``a`` or the ``b`` value is ``None``, the joint limits are used instead.

    Note:
        If the dictionary does not contain a key, the joint position or joint velocity is set to the default value for
        that joint.

    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)

        # check if the cfg has the required parameters
        if "position_range" not in cfg.params or "velocity_range" not in cfg.params:
            raise ValueError(
                "The term 'reset_joints_within_range' requires parameters: 'position_range' and 'velocity_range'."
                f" Received: {list(cfg.params.keys())}."
            )

        # parse the parameters
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        use_default_offset = cfg.params.get("use_default_offset", False)
        operation = cfg.params.get("operation", "abs")
        # check if the operation is valid
        if operation not in ["abs", "scale"]:
            raise ValueError(
                f"For event 'reset_joints_within_limits_range', unknown operation: '{operation}'."
                " Please use 'abs' or 'scale'."
            )

        # extract the used quantities (to enable type-hinting)
        self._asset: Articulation = env.scene[asset_cfg.name]
        default_joint_pos = wp.to_torch(self._asset.data.default_joint_pos)[0]
        default_joint_vel = wp.to_torch(self._asset.data.default_joint_vel)[0]

        # create buffers to store the joint position range
        self._pos_ranges = wp.to_torch(self._asset.data.soft_joint_pos_limits)[0].clone()
        # parse joint position ranges
        pos_joint_ids = []
        for joint_name, joint_range in cfg.params["position_range"].items():
            # find the joint ids
            joint_ids = self._asset.find_joints(joint_name)[0]
            pos_joint_ids.extend(joint_ids)

            # set the joint position ranges based on the given values
            if operation == "abs":
                if joint_range[0] is not None:
                    self._pos_ranges[joint_ids, 0] = joint_range[0]
                if joint_range[1] is not None:
                    self._pos_ranges[joint_ids, 1] = joint_range[1]
            elif operation == "scale":
                if joint_range[0] is not None:
                    self._pos_ranges[joint_ids, 0] *= joint_range[0]
                if joint_range[1] is not None:
                    self._pos_ranges[joint_ids, 1] *= joint_range[1]
            else:
                raise ValueError(
                    f"Unknown operation: '{operation}' for joint position ranges. Please use 'abs' or 'scale'."
                )
            # add the default offset
            if use_default_offset:
                self._pos_ranges[joint_ids] += default_joint_pos[joint_ids].unsqueeze(1)

        # store the joint pos ids (used later to sample the joint positions)
        self._pos_joint_ids = torch.tensor(pos_joint_ids, device=self._pos_ranges.device)
        self._pos_ranges = self._pos_ranges[self._pos_joint_ids]

        # create buffers to store the joint velocity range
        soft_joint_vel_limits_torch = wp.to_torch(self._asset.data.soft_joint_vel_limits)[0]
        self._vel_ranges = torch.stack([-soft_joint_vel_limits_torch, soft_joint_vel_limits_torch], dim=1)
        # parse joint velocity ranges
        vel_joint_ids = []
        for joint_name, joint_range in cfg.params["velocity_range"].items():
            # find the joint ids
            joint_ids = self._asset.find_joints(joint_name)[0]
            vel_joint_ids.extend(joint_ids)

            # set the joint position ranges based on the given values
            if operation == "abs":
                if joint_range[0] is not None:
                    self._vel_ranges[joint_ids, 0] = joint_range[0]
                if joint_range[1] is not None:
                    self._vel_ranges[joint_ids, 1] = joint_range[1]
            elif operation == "scale":
                if joint_range[0] is not None:
                    self._vel_ranges[joint_ids, 0] = joint_range[0] * self._vel_ranges[joint_ids, 0]
                if joint_range[1] is not None:
                    self._vel_ranges[joint_ids, 1] = joint_range[1] * self._vel_ranges[joint_ids, 1]
            else:
                raise ValueError(
                    f"Unknown operation: '{operation}' for joint velocity ranges. Please use 'abs' or 'scale'."
                )
            # add the default offset
            if use_default_offset:
                self._vel_ranges[joint_ids] += default_joint_vel[joint_ids].unsqueeze(1)

        # store the joint vel ids (used later to sample the joint positions)
        self._vel_joint_ids = torch.tensor(vel_joint_ids, device=self._vel_ranges.device)
        self._vel_ranges = self._vel_ranges[self._vel_joint_ids]

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        position_range: dict[str, tuple[float | None, float | None]],
        velocity_range: dict[str, tuple[float | None, float | None]],
        use_default_offset: bool = False,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        operation: Literal["abs", "scale"] = "abs",
    ):
        # get default joint state
        joint_pos = wp.to_torch(self._asset.data.default_joint_pos)[env_ids].clone()
        joint_vel = wp.to_torch(self._asset.data.default_joint_vel)[env_ids].clone()

        # sample random joint positions for each joint
        if len(self._pos_joint_ids) > 0:
            joint_pos_shape = (len(env_ids), len(self._pos_joint_ids))
            joint_pos[:, self._pos_joint_ids] = sample_uniform(
                self._pos_ranges[:, 0], self._pos_ranges[:, 1], joint_pos_shape, device=joint_pos.device
            )
            # clip the joint positions to the joint limits
            joint_pos_limits = wp.to_torch(self._asset.data.soft_joint_pos_limits)[0, self._pos_joint_ids]
            joint_pos = joint_pos.clamp(joint_pos_limits[:, 0], joint_pos_limits[:, 1])

        # sample random joint velocities for each joint
        if len(self._vel_joint_ids) > 0:
            joint_vel_shape = (len(env_ids), len(self._vel_joint_ids))
            joint_vel[:, self._vel_joint_ids] = sample_uniform(
                self._vel_ranges[:, 0], self._vel_ranges[:, 1], joint_vel_shape, device=joint_vel.device
            )
            # clip the joint velocities to the joint limits
            joint_vel_limits = wp.to_torch(self._asset.data.soft_joint_vel_limits)[0, self._vel_joint_ids]
            joint_vel = joint_vel.clamp(-joint_vel_limits, joint_vel_limits)

        # set into the physics simulation
        self._asset.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self._asset.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
