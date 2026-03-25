# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import NormalVelocityCommandCfg, UniformVelocityCommandCfg

# import logger
logger = logging.getLogger(__name__)
_VIS_DEBUG_ENABLED = os.getenv("ISAACLAB_VIS_DEBUG", "0").lower() in {"1", "true", "yes", "on"}


class UniformVelocityCommand(CommandTerm):
    r"""Command generator that generates a velocity command in SE(2) from uniform distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    If the :attr:`cfg.heading_command` flag is set to True, the angular velocity is computed from the heading
    error similar to doing a proportional control on the heading error. The target heading is sampled uniformly
    from the provided range. Otherwise, the angular velocity is sampled uniformly from the provided range.

    Mathematically, the angular velocity is computed as follows from the heading command:

    .. math::

        \omega_z = \frac{1}{2} \text{wrap_to_pi}(\theta_{\text{target}} - \theta_{\text{current}})

    """

    cfg: UniformVelocityCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # check configuration
        if self.cfg.heading_command and self.cfg.ranges.heading is None:
            raise ValueError(
                "The velocity command has heading commands active (heading_command=True) but the `ranges.heading`"
                " parameter is set to None."
            )
        if self.cfg.ranges.heading and not self.cfg.heading_command:
            logger.warning(
                f"The velocity command has the 'ranges.heading' attribute set to '{self.cfg.ranges.heading}'"
                " but the heading command is not active. Consider setting the flag for the heading command to True."
            )

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # crete buffers to store the command
        # -- command: x vel, y vel, yaw vel, heading
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)
        self._debug_vis_callback_count = 0
        self._debug_env0_probe_body_path: str | None = None
        self._debug_env0_probe_status: str = "uninitialized"
        # -- metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        self.metrics["error_vel_xy"] += (
            torch.linalg.norm(self.vel_command_b[:, :2] - wp.to_torch(self.robot.data.root_lin_vel_b)[:, :2], dim=-1)
            / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - wp.to_torch(self.robot.data.root_ang_vel_b)[:, 2]) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- ang vel yaw - rotation around z
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        # Compute angular velocity from heading direction
        if self.cfg.heading_command:
            # resolve indices of heading envs
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            # compute angular velocity
            heading_error = math_utils.wrap_to_pi(
                self.heading_target[env_ids] - wp.to_torch(self.robot.data.heading_w)[env_ids]
            )
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
        # Enforce standing (i.e., zero velocity command) for standing envs
        # TODO: check if conversion is needed
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = wp.to_torch(self.robot.data.root_pos_w).clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(
            wp.to_torch(self.robot.data.root_lin_vel_b)[:, :2]
        )
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)
        self._debug_vis_callback_count += 1
        if _VIS_DEBUG_ENABLED and (
            self._debug_vis_callback_count <= 3 or self._debug_vis_callback_count % 120 == 0
        ):
            mean_cmd = float(torch.linalg.norm(self.command[:, :2], dim=-1).mean().item())
            mean_vel = float(torch.linalg.norm(wp.to_torch(self.robot.data.root_lin_vel_b)[:, :2], dim=-1).mean().item())
            tensor_root_pos = wp.to_torch(self.robot.data.root_pos_w)[0].detach().cpu()
            fabric_root_pos = self._get_env0_fabric_probe_position()
            root_view_pos = self._get_env0_root_view_position()
            robot_root = self._resolve_env0_robot_root_path()
            fabric_base_pos = self._get_fabric_position_for_path(f"{robot_root}/base") if robot_root else None
            fabric_trunk_pos = self._get_fabric_position_for_path(f"{robot_root}/trunk") if robot_root else None
            if fabric_root_pos is not None:
                fabric_root_pos_t = torch.tensor(fabric_root_pos, dtype=tensor_root_pos.dtype)
                delta = torch.linalg.norm(tensor_root_pos - fabric_root_pos_t).item()
                root_view_delta = None
                if root_view_pos is not None:
                    root_view_pos_t = torch.tensor(root_view_pos, dtype=tensor_root_pos.dtype)
                    root_view_delta = torch.linalg.norm(tensor_root_pos - root_view_pos_t).item()
                logger.warning(
                    "[VIS-DEBUG][UniformVelocityCommand._debug_vis_callback] count=%d mean_cmd_xy=%.4f "
                    "mean_vel_xy=%.4f env0_tensor_root=(%.3f, %.3f, %.3f) env0_fabric_root=(%.3f, %.3f, %.3f) "
                    "tensor_fabric_delta=%.5f root_view_delta=%s fabric_base=%s fabric_trunk=%s "
                    "probe_path=%s probe_status=%s",
                    self._debug_vis_callback_count,
                    mean_cmd,
                    mean_vel,
                    float(tensor_root_pos[0]),
                    float(tensor_root_pos[1]),
                    float(tensor_root_pos[2]),
                    float(fabric_root_pos_t[0]),
                    float(fabric_root_pos_t[1]),
                    float(fabric_root_pos_t[2]),
                    float(delta),
                    f"{root_view_delta:.5f}" if root_view_delta is not None else "<unavailable>",
                    (
                        f"({fabric_base_pos[0]:.3f}, {fabric_base_pos[1]:.3f}, {fabric_base_pos[2]:.3f})"
                        if fabric_base_pos is not None
                        else "<unavailable>"
                    ),
                    (
                        f"({fabric_trunk_pos[0]:.3f}, {fabric_trunk_pos[1]:.3f}, {fabric_trunk_pos[2]:.3f})"
                        if fabric_trunk_pos is not None
                        else "<unavailable>"
                    ),
                    self._debug_env0_probe_body_path or "<unresolved>",
                    self._debug_env0_probe_status,
                )
            else:
                logger.warning(
                    "[VIS-DEBUG][UniformVelocityCommand._debug_vis_callback] count=%d mean_cmd_xy=%.4f "
                    "mean_vel_xy=%.4f env0_tensor_root=(%.3f, %.3f, %.3f) env0_fabric_root=<unavailable> "
                    "root_view=%s fabric_base=%s fabric_trunk=%s probe_path=%s probe_status=%s",
                    self._debug_vis_callback_count,
                    mean_cmd,
                    mean_vel,
                    float(tensor_root_pos[0]),
                    float(tensor_root_pos[1]),
                    float(tensor_root_pos[2]),
                    (
                        f"({root_view_pos[0]:.3f}, {root_view_pos[1]:.3f}, {root_view_pos[2]:.3f})"
                        if root_view_pos is not None
                        else "<unavailable>"
                    ),
                    (
                        f"({fabric_base_pos[0]:.3f}, {fabric_base_pos[1]:.3f}, {fabric_base_pos[2]:.3f})"
                        if fabric_base_pos is not None
                        else "<unavailable>"
                    ),
                    (
                        f"({fabric_trunk_pos[0]:.3f}, {fabric_trunk_pos[1]:.3f}, {fabric_trunk_pos[2]:.3f})"
                        if fabric_trunk_pos is not None
                        else "<unavailable>"
                    ),
                    self._debug_env0_probe_body_path or "<unresolved>",
                    self._debug_env0_probe_status,
                )

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = wp.to_torch(self.robot.data.root_quat_w)
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat

    def _resolve_env0_robot_root_path(self) -> str | None:
        """Best-effort resolve env_0 robot root path from articulation cfg prim path."""
        prim_path = getattr(self.robot.cfg, "prim_path", None)
        if not isinstance(prim_path, str) or not prim_path:
            return None
        path = prim_path
        if "{ENV_REGEX_NS}" in path:
            path = path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
        path = path.replace("env_.*", "env_0")
        path = re.sub(r"\.\*$", "0", path)
        return path

    def _resolve_env0_fabric_probe_body_path(self) -> str | None:
        """Resolve a rigid-body prim path for env_0 suitable for Fabric world-matrix probing."""
        if self._debug_env0_probe_body_path is not None:
            return self._debug_env0_probe_body_path
        try:
            from pxr import UsdPhysics

            stage = self._env.sim.stage
            if stage is None:
                self._debug_env0_probe_status = "no_stage"
                return None
            robot_root = self._resolve_env0_robot_root_path()
            if not robot_root:
                self._debug_env0_probe_status = "no_robot_root_path"
                return None
            root_prim = stage.GetPrimAtPath(robot_root)
            if not root_prim.IsValid():
                self._debug_env0_probe_status = f"invalid_robot_root:{robot_root}"
                return None

            # First try common body names directly under Robot to avoid brittle traversal assumptions.
            fallback_candidates = [
                f"{robot_root}/base",
                f"{robot_root}/base_link",
                f"{robot_root}/trunk",
                f"{robot_root}/pelvis",
            ]
            for candidate in fallback_candidates:
                prim = stage.GetPrimAtPath(candidate)
                if prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    self._debug_env0_probe_body_path = candidate
                    self._debug_env0_probe_status = "resolved_fallback_candidate"
                    return self._debug_env0_probe_body_path

            preferred_names = {"base", "base_link", "trunk", "pelvis"}
            for prim in root_prim.Traverse():
                if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    continue
                if prim.GetName() in preferred_names:
                    self._debug_env0_probe_body_path = prim.GetPath().pathString
                    self._debug_env0_probe_status = "resolved_traverse_preferred"
                    return self._debug_env0_probe_body_path
            for prim in root_prim.Traverse():
                if prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    self._debug_env0_probe_body_path = prim.GetPath().pathString
                    self._debug_env0_probe_status = "resolved_traverse_first_rigidbody"
                    return self._debug_env0_probe_body_path
            self._debug_env0_probe_status = "no_rigidbody_under_robot_root"
        except Exception:
            self._debug_env0_probe_status = "resolve_exception"
            return None
        return None

    def _get_env0_fabric_probe_position(self) -> tuple[float, float, float] | None:
        """Read env_0 probe body world translation from Fabric backend."""
        body_path = self._resolve_env0_fabric_probe_body_path()
        if not body_path:
            return None
        return self._get_fabric_position_for_path(body_path)

    def _get_fabric_position_for_path(self, body_path: str) -> tuple[float, float, float] | None:
        """Read a body world translation from Fabric backend."""
        try:
            import isaacsim.core.experimental.utils.backend as backend_utils
            import isaacsim.core.experimental.utils.prim as prim_utils
            import numpy as np

            with backend_utils.use_backend("fabric"):
                world_matrix = prim_utils.get_prim_attribute_value(body_path, "omni:fabric:worldMatrix")
            translation = np.array(world_matrix.ExtractTranslation(), dtype=float)
            self._debug_env0_probe_status = "fabric_read_ok"
            return (float(translation[0]), float(translation[1]), float(translation[2]))
        except Exception:
            self._debug_env0_probe_status = "fabric_read_exception"
            return None

    def _get_env0_root_view_position(self) -> tuple[float, float, float] | None:
        """Read env_0 articulation root position directly from PhysX root_view."""
        try:
            root_transforms = self.robot.root_view.get_root_transforms()
            root_transforms_t = wp.to_torch(root_transforms)
            if root_transforms_t.ndim >= 2 and root_transforms_t.shape[1] >= 3 and root_transforms_t.shape[0] > 0:
                return (
                    float(root_transforms_t[0, 0].item()),
                    float(root_transforms_t[0, 1].item()),
                    float(root_transforms_t[0, 2].item()),
                )
        except Exception:
            return None
        return None


class NormalVelocityCommand(UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from a normal distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    The command is sampled from a normal distribution with mean and standard deviation specified in
    the configuration. With equal probability, the sign of the individual components is flipped.
    """

    cfg: NormalVelocityCommandCfg
    """The command generator configuration."""

    def __init__(self, cfg: NormalVelocityCommandCfg, env: ManagerBasedEnv):
        """Initializes the command generator.

        Args:
            cfg: The command generator configuration.
            env: The environment.
        """
        super().__init__(cfg, env)
        # create buffers for zero commands envs
        self.is_zero_vel_x_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_zero_vel_y_env = torch.zeros_like(self.is_zero_vel_x_env)
        self.is_zero_vel_yaw_env = torch.zeros_like(self.is_zero_vel_x_env)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "NormalVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    def _resample_command(self, env_ids):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.normal_(mean=self.cfg.ranges.mean_vel[0], std=self.cfg.ranges.std_vel[0])
        self.vel_command_b[env_ids, 0] *= torch.where(r.uniform_(0.0, 1.0) <= 0.5, 1.0, -1.0)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.normal_(mean=self.cfg.ranges.mean_vel[1], std=self.cfg.ranges.std_vel[1])
        self.vel_command_b[env_ids, 1] *= torch.where(r.uniform_(0.0, 1.0) <= 0.5, 1.0, -1.0)
        # -- angular velocity - yaw direction
        self.vel_command_b[env_ids, 2] = r.normal_(mean=self.cfg.ranges.mean_vel[2], std=self.cfg.ranges.std_vel[2])
        self.vel_command_b[env_ids, 2] *= torch.where(r.uniform_(0.0, 1.0) <= 0.5, 1.0, -1.0)

        # update element wise zero velocity command
        # TODO what is zero prob ?
        self.is_zero_vel_x_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.ranges.zero_prob[0]
        self.is_zero_vel_y_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.ranges.zero_prob[1]
        self.is_zero_vel_yaw_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.ranges.zero_prob[2]

        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        """Sets velocity command to zero for standing envs."""
        # Enforce standing (i.e., zero velocity command) for standing envs
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()  # TODO check if conversion is needed
        self.vel_command_b[standing_env_ids, :] = 0.0

        # Enforce zero velocity for individual elements
        # TODO: check if conversion is needed
        zero_vel_x_env_ids = self.is_zero_vel_x_env.nonzero(as_tuple=False).flatten()
        zero_vel_y_env_ids = self.is_zero_vel_y_env.nonzero(as_tuple=False).flatten()
        zero_vel_yaw_env_ids = self.is_zero_vel_yaw_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[zero_vel_x_env_ids, 0] = 0.0
        self.vel_command_b[zero_vel_y_env_ids, 1] = 0.0
        self.vel_command_b[zero_vel_yaw_env_ids, 2] = 0.0
