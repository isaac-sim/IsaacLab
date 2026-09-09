# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command term for 3D orientation goals of in-hand reorientation objects."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.leapp import POSE7_ELEMENT_NAMES

from isaaclab_tasks.core.reorient.utils import SuccessTracker

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands_cfg import ReorientCommandCfg


class ReorientCommand(CommandTerm):
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

    cfg: ReorientCommandCfg
    """Configuration for the command term."""

    def __init__(self, cfg: ReorientCommandCfg, env: ManagerBasedRLEnv):
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

        # -- metrics
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["consecutive_success"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["success_rate"] = torch.zeros(self.num_envs, device=self.device)
        self._success = SuccessTracker(self.num_envs, self.device)
        self._fixed_marker_pos_w: torch.Tensor | None = None

        # adds (optional) cmd kind and element names for leapp export
        # during export, semantic data about this command will be used to annotate the command input
        self.cfg.cmd_kind = self.cfg.cmd_kind or "command/body/pose"
        self.cfg.element_names = self.cfg.element_names or POSE7_ELEMENT_NAMES

    def __str__(self) -> str:
        msg = "ReorientCommand:\n"
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
        # snapshot the goal count before the base class logs and zeros the metrics
        if env_ids is None:
            env_ids = slice(None)
        # Reaching a goal draws a replacement, so exactly one goal is outstanding when
        # the episode ends: the episode presented ``goals + 1`` and completed ``goals``.
        goals = self._success.snapshot(env_ids)
        self.metrics["success_rate"][env_ids] = goals / (goals + 1.0)
        extras = super().reset(env_ids)
        # only an auto-reset lands mid-step, with the new goal evaluated before any
        # physics runs against it; an explicit reset is followed by a full step
        self._success.clear(env_ids, skip_next_update=self._env.reset_buf[env_ids])
        # Route success_rate to the unified ``Metrics/success_rate`` path (shared TensorBoard
        # card across tasks); pop it from the returned dict so CommandManager does not
        # additionally log it under ``Metrics/<term_name>/success_rate``.
        self._env.extras.setdefault("log", {})["Metrics/success_rate"] = extras.pop("success_rate")
        return extras

    def _resample_command(self, env_ids: Sequence[int]):
        self._success.record_goal_reached(env_ids)
        # The shared sampler covers SO(3) uniformly. Composing a rotation about x with one about y, as
        # this did, reaches only a two-axis subset and needs a unit-axis buffer per axis to do it.
        quat = math_utils.random_orientation(len(env_ids), device=self.device)
        # make sure the quaternion real-part is always positive
        self.quat_command_w[env_ids] = math_utils.quat_unique(quat) if self.cfg.make_quat_unique else quat

    def _update_command(self):
        if not self.cfg.update_goal_on_success:
            return
        reached = self.metrics["orientation_error"] < self.cfg.orientation_success_threshold
        goal_resets = self._success.earned(reached)
        self._resample(goal_resets.nonzero(as_tuple=False).squeeze(-1))

    def _set_debug_vis_impl(self, debug_vis: bool):
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
        if self.cfg.fixed_marker_pos is None:
            marker_pos = self.pos_command_w + torch.tensor(self.cfg.marker_pos_offset, device=self.device)
        else:
            if self._fixed_marker_pos_w is None:
                # constant per run; cached to avoid a host-to-device allocation every render frame
                self._fixed_marker_pos_w = (
                    torch.tensor(self.cfg.fixed_marker_pos, device=self.device).repeat(self.num_envs, 1)
                    + self._env.scene.env_origins
                )
            marker_pos = self._fixed_marker_pos_w
        self.goal_pose_visualizer.visualize(
            translations=marker_pos,
            orientations=self.quat_command_w,
            environment_ids=self._env.scene._ALL_INDICES,
        )
