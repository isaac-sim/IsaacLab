# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab_contrib.controllers import Ik7dControllerCfg

    from .ik_7d_actions import Ik7dAction


@configclass
class Ik7dActionCfg(ActionTermCfg):
    """Configuration for the ik_7d task-space action term.

    The action is ``8`` elements per arm -- ``[x, y, z, qx, qy, qz, qw, swivel]``,
    quaternion in **xyzw** -- concatenated in the order of
    :attr:`Ik7dControllerCfg.arms`.

    Example:
        .. code-block:: python

            from isaaclab_contrib.controllers import Ik7dControllerCfg
            from isaaclab_contrib.mdp import Ik7dActionCfg

            arm_action = Ik7dActionCfg(
                asset_name="robot",
                controller=Ik7dControllerCfg(
                    robot_name="G2_t2_crs",
                    urdf_path="/path/to/G2_t2_crs.urdf",
                    arms=["left", "right"],
                ),
                pose_frame="base",
            )

    .. seealso::
        - :class:`~isaaclab_contrib.mdp.actions.Ik7dAction`: the action term itself
        - :class:`~isaaclab_contrib.controllers.Ik7dController`: the solver wrapper
    """

    class_type: type[Ik7dAction] | str = "{DIR}.ik_7d_actions:Ik7dAction"

    asset_name: str = MISSING
    """Name of the articulation the action is applied to, e.g. ``"robot"``."""

    controller: Ik7dControllerCfg = MISSING
    """Configuration of the ik_7d solver, including which arms to drive."""

    pose_frame: Literal["base", "world"] = "base"
    """Frame the commanded end-effector poses are expressed in.

    * ``"base"`` (default) -- poses are already in the robot's base-link frame,
      which is what ik_7d works in natively and what ``fk7d`` returns. Use this
      for scripted trajectories built from forward kinematics.
    * ``"world"`` -- poses are in world coordinates and are pulled back through
      the robot's measured root pose (env origins subtracted first, so tiled
      multi-environment scenes are handled). Use this for a teleoperation device,
      whose poses are anchored to the operator rather than the robot.
    """

    debug_vis_stats: bool = True
    """Whether to print the resolved joint mapping and per-arm swivel geometry once at startup.

    Cheap and one-shot, and it is the only place the *measured* free swivel
    direction becomes visible -- that is probed at construction and is easy to
    get silently wrong when moving to another arm variant.
    """
