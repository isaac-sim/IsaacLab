# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reach task module: track a 6-D end-effector pose command."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import ManagerTermBaseCfg as TermCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_contrib.tasks.manipulation.multitask import mdp
from isaaclab_contrib.tasks.manipulation.multitask.mdp.commands_cfg import PoseCommandRanges

from ._base import TaskModuleCfg

if TYPE_CHECKING:
    from isaaclab.managers import EventTermCfg

    from ..robots._base import RobotModuleCfg


_TABLE_SPAWN = UsdFileCfg(
    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
)


@dataclass
class ReachTaskCfg(TaskModuleCfg):
    """End-effector pose-tracking task.

    No task objects are spawned.  A :class:`~...mdp.commands_cfg.PoseCommandCfg`
    sets a target 6-D pose; rewards encourage both position and orientation
    tracking.
    """

    command_ranges: PoseCommandRanges = field(
        default_factory=lambda: PoseCommandRanges(
            pos_x=(0.35, 0.65),
            pos_y=(-0.2, 0.2),
            pos_z=(0.15, 0.5),
            roll=(0.0, 0.0),
            pitch=(math.pi, math.pi),
            yaw=(-3.14, 3.14),
        )
    )
    """Workspace pose ranges for the reach target [m, rad]."""

    resampling_time_range: tuple[float, float] = (3.0, 3.0)
    """Min/max seconds between goal resamples [s]."""

    pos_tracking_weight: float = -0.2
    """Reward weight for L2 position-tracking error."""

    pos_tracking_fine_weight: float = 0.1
    """Reward weight for tanh position-tracking bonus."""

    pos_tracking_fine_std: float = 0.1
    """Standard deviation for the tanh bonus [m]."""

    ori_tracking_weight: float = -0.1
    """Reward weight for orientation-tracking error."""

    @property
    def name(self) -> str:
        return "reach"

    # ------------------------------------------------------------------
    # Scene assets
    # ------------------------------------------------------------------

    def scene_assets(self, group: str, robot: RobotModuleCfg) -> dict[str, object]:
        table_name = f"{robot.name}_reach_table"
        return {
            table_name: AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{robot.name.title()}_ReachTable",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.0, 0.0, 0.707, 0.707)),
                spawn=_TABLE_SPAWN,
            ),
        }

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def command_terms(self, group: str, robot: RobotModuleCfg) -> dict[str, object]:
        # Determine the ee body and asset name from the robot
        _ROBOT_EE = {
            "franka": ("franka_robot", "panda_hand"),
            "openarm": ("openarm_robot", "openarm_hand"),
            "ur10": ("ur10_robot", "ee_link"),
        }
        asset_name, ee_body = _ROBOT_EE.get(robot.name, (f"{robot.name}_robot", "ee_link"))
        cmd_name = f"{group}_{self.name}"

        return {
            cmd_name: mdp.PoseCommandCfg(
                asset_cfg=SceneEntityCfg(asset_name, body_names=[ee_body], selector=group),
                ranges=self.command_ranges,
                resampling_time_range=self.resampling_time_range,
                debug_vis=True,
            ),
        }

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def task_obs_terms(self, group: str, robot: RobotModuleCfg) -> dict[str, ObsTerm]:
        # Reach has no additional task-local obs beyond ee_pose (already in robot scatter obs)
        return {}

    def scatter_obs_terms(self, group: str, robot: RobotModuleCfg) -> dict[str, tuple[int | None, TermCfg]]:
        _ROBOT_EE = {
            "franka": ("franka_robot", "panda_hand"),
            "openarm": ("openarm_robot", "openarm_hand"),
            "ur10": ("ur10_robot", "ee_link"),
        }
        asset_name, ee_body = _ROBOT_EE.get(robot.name, (f"{robot.name}_robot", "ee_link"))
        cmd_name = f"{group}_{self.name}"

        return {
            "commands": (
                7,
                TermCfg(
                    func=mdp.generated_commands,
                    params={
                        "asset_cfg": SceneEntityCfg(asset_name, selector=group),
                        "command_name": cmd_name,
                    },
                ),
            ),
            "ee_pos_error": (
                3,
                TermCfg(
                    func=mdp.ee_pos_error,
                    params={
                        "asset_cfg": SceneEntityCfg(asset_name, body_names=[ee_body], selector=group),
                        "command_name": cmd_name,
                    },
                ),
            ),
        }

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def reward_terms(self, group: str, robot: RobotModuleCfg) -> dict[str, RewTerm]:
        _ROBOT_EE = {
            "franka": ("franka_robot", "panda_hand"),
            "openarm": ("openarm_robot", "openarm_hand"),
            "ur10": ("ur10_robot", "ee_link"),
        }
        asset_name, ee_body = _ROBOT_EE.get(robot.name, (f"{robot.name}_robot", "ee_link"))
        cmd_name = f"{group}_{self.name}"
        asset_cfg = SceneEntityCfg(asset_name, body_names=[ee_body], selector=group)

        return {
            f"{group}_ee_pos_tracking": RewTerm(
                func=mdp.position_command_error,
                weight=self.pos_tracking_weight,
                params={"asset_cfg": asset_cfg, "command_name": cmd_name},
            ),
            f"{group}_ee_pos_tracking_fine": RewTerm(
                func=mdp.position_command_error_tanh,
                weight=self.pos_tracking_fine_weight,
                params={
                    "std": self.pos_tracking_fine_std,
                    "asset_cfg": asset_cfg,
                    "command_name": cmd_name,
                },
            ),
            f"{group}_ee_ori_tracking": RewTerm(
                func=mdp.orientation_command_error,
                weight=self.ori_tracking_weight,
                params={"asset_cfg": asset_cfg, "command_name": cmd_name},
            ),
        }

    # ------------------------------------------------------------------
    # Terminations  (reach has none beyond global timeout)
    # ------------------------------------------------------------------

    def termination_terms(self, group: str, robot: RobotModuleCfg) -> dict[str, DoneTerm]:
        return {}

    # ------------------------------------------------------------------
    # Reset events  (reach has none beyond robot reset)
    # ------------------------------------------------------------------

    def reset_events(self, group: str, robot: RobotModuleCfg) -> dict[str, EventTermCfg]:
        return {}


# ---------------------------------------------------------------------------
# Pre-built instances with common workspace ranges
# ---------------------------------------------------------------------------

REACH_TASK = ReachTaskCfg()
"""Standard reach task with Franka/UR10-friendly workspace ranges."""

REACH_TASK_OPENARM = ReachTaskCfg(
    command_ranges=PoseCommandRanges(
        pos_x=(0.25, 0.35),
        pos_y=(-0.2, 0.2),
        pos_z=(0.3, 0.4),
        roll=(-math.pi / 6, math.pi / 6),
        pitch=(math.pi / 2, math.pi / 2),
        yaw=(-math.pi / 9, math.pi / 9),
    )
)
"""Reach task with workspace ranges tuned for the OpenArm robot."""
