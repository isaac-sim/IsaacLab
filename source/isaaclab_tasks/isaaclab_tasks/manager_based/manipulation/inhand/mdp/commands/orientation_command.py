# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for 3D orientation goals for objects."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.leapp import POSE7_ELEMENT_NAMES

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands_cfg import InHandReOrientationCommandCfg


class InHandReOrientationCommand(CommandTerm):
    """Command term that generates 3D pose commands for in-hand manipulation task.

    This command term generates 3D orientation commands for the object. The orientation commands
    are sampled uniformly from the 3D orientation space. The position commands are the default
    root state of the object.

    The constant position commands is to encourage that the object does not move during the task.
    For instance, the object should not fall off the robot's palm.

    Unlike typical command terms, where the goals are resampled based on time, this command term
    does not resample the goals based on time. Instead, the goals are resampled when the object
    reaches the goal orientation. The goal orientation is considered to be reached when the
    orientation error is below a certain threshold.
    """

    cfg: InHandReOrientationCommandCfg
    """Configuration for the command term."""

    def __init__(self, cfg: InHandReOrientationCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command term class.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # object
        self.object: RigidObject = env.scene[cfg.asset_name]

        # create buffers to store the command
        # -- command: (x, y, z)
        init_pos_offset = torch.tensor(cfg.init_pos_offset, dtype=torch.float, device=self.device)
        self.pos_command_e = self.object.data.default_root_pose.torch[:, :3] + init_pos_offset
        self.pos_command_w = self.pos_command_e + self._env.scene.env_origins
        # -- orientation: (x, y, z, w)
        self.quat_command_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.quat_command_w[:, 3] = 1.0  # set the scalar component to 1.0

        # -- unit vectors
        self._X_UNIT_VEC = torch.tensor([1.0, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self._Y_UNIT_VEC = torch.tensor([0, 1.0, 0], device=self.device).repeat((self.num_envs, 1))
        self._Z_UNIT_VEC = torch.tensor([0, 0, 1.0], device=self.device).repeat((self.num_envs, 1))

        # -- metrics
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["consecutive_success"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["success_rate"] = torch.zeros(self.num_envs, device=self.device)
        # -- per-attempt success accounting: each success-driven resample completes one attempt;
        #    the trailing attempt at episode end counts as one unsuccessful attempt.
        self._completed_attempts = torch.zeros(self.num_envs, device=self.device)

        # adds (optional) cmd kind and element names for leapp export
        # during export, semantic data about this command will be used to annotate the command input
        self.cfg.cmd_kind = self.cfg.cmd_kind or "command/body/pose"
        self.cfg.element_names = self.cfg.element_names or POSE7_ELEMENT_NAMES

    def __str__(self) -> str:
        msg = "InHandManipulationCommandGenerator:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired goal pose in the environment frame. Shape is (num_envs, 7)."""
        return torch.cat((self.pos_command_e, self.quat_command_w), dim=-1)

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # logs data
        # -- compute the orientation error
        self.metrics["orientation_error"] = math_utils.quat_error_magnitude(
            self.object.data.root_quat_w.torch, self.quat_command_w
        )
        # -- compute the position error
        self.metrics["position_error"] = torch.linalg.norm(
            self.object.data.root_pos_w.torch - self.pos_command_w, dim=1
        )
        # -- compute the number of consecutive successes
        successes = self.metrics["orientation_error"] < self.cfg.orientation_success_threshold
        self.metrics["consecutive_success"] += successes.float()

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        # Snapshot per-attempt success rate BEFORE the base class logs and zeros metrics.
        # success_rate = completed_attempts / (completed_attempts + 1 trailing in-progress).
        if env_ids is None:
            env_ids = slice(None)
        completed = self._completed_attempts[env_ids]
        self.metrics["success_rate"][env_ids] = completed / (completed + 1.0)
        extras = super().reset(env_ids)
        # super().reset() invoked _resample_command for the new initial goal, which
        # incremented _completed_attempts; zero it back out so the new episode starts clean.
        self._completed_attempts[env_ids] = 0.0
        # Route success_rate to the unified ``Metrics/success_rate`` path (shared TensorBoard
        # card across tasks); pop it from the returned dict so CommandManager does not
        # additionally log it under ``Metrics/<term_name>/success_rate``.
        self._env.extras.setdefault("log", {})["Metrics/success_rate"] = extras.pop("success_rate")
        return extras

    def _resample_command(self, env_ids: Sequence[int]):
        # Each call corresponds to a success-driven (or initial) resample; count it as a
        # completed attempt. The post-reset increment is cleared by ``reset()`` afterwards.
        self._completed_attempts[env_ids] += 1.0
        # sample new orientation targets
        rand_floats = 2.0 * torch.rand((len(env_ids), 2), device=self.device) - 1.0
        # rotate randomly about x-axis and then y-axis
        quat = math_utils.quat_mul(
            math_utils.quat_from_angle_axis(rand_floats[:, 0] * torch.pi, self._X_UNIT_VEC[env_ids]),
            math_utils.quat_from_angle_axis(rand_floats[:, 1] * torch.pi, self._Y_UNIT_VEC[env_ids]),
        )
        # make sure the quaternion real-part is always positive
        self.quat_command_w[env_ids] = math_utils.quat_unique(quat) if self.cfg.make_quat_unique else quat

    def _update_command(self):
        # update the command if goal is reached
        if self.cfg.update_goal_on_success:
            # compute the goal resets
            goal_resets = self.metrics["orientation_error"] < self.cfg.orientation_success_threshold
            goal_reset_ids = goal_resets.nonzero(as_tuple=False).squeeze(-1)
            # resample the goals
            self._resample(goal_reset_ids)

    def _set_debug_vis_impl(self, debug_vis: TYPE_CHECKING):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            # set visibility
            self.goal_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # add an offset to the marker position to visualize the goal
        marker_pos = self.pos_command_w + torch.tensor(self.cfg.marker_pos_offset, device=self.device)
        marker_quat = self.quat_command_w
        # visualize the goal marker
        self.goal_pose_visualizer.visualize(translations=marker_pos, orientations=marker_quat)
