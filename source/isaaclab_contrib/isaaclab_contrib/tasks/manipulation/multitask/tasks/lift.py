# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lift task module: pick up a cube and move it to a target pose."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBaseCfg as TermCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_contrib.tasks.manipulation.multitask import mdp
from isaaclab_contrib.tasks.manipulation.multitask.mdp.commands_cfg import PoseCommandRanges

from ._base import TaskModuleCfg

if TYPE_CHECKING:
    from ..robots._base import RobotModuleCfg

_CUBE_SPAWN = UsdFileCfg(
    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
    scale=(0.8, 0.8, 0.8),
    rigid_props=RigidBodyPropertiesCfg(
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=5.0,
        disable_gravity=False,
    ),
)

_TABLE_SPAWN = UsdFileCfg(
    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
)


# Map robot → (asset_name, ee_body_name, ee_frame_name) for lift-specific MDP terms
_ROBOT_EE = {
    "franka": ("franka_robot", "panda_hand", "franka_ee_frame"),
    "openarm": ("openarm_robot", "openarm_hand", "openarm_ee_frame"),
    "ur10": ("ur10_robot", "ee_link", None),
}


@dataclass
class LiftTaskCfg(TaskModuleCfg):
    """Cube-lifting task.

    Spawns a DexCube for the robot to grasp and carry to a pose command target.
    The cube name in the scene is ``f"{robot.name}_lift_object"``.
    """

    cube_init_pos: tuple[float, float, float] = (0.4, 0.0, 0.055)
    """Default cube spawn position relative to env origin [m]."""

    object_spawn_range: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)}
    )
    """Uniform randomisation range for cube position at reset [m]."""

    command_ranges: PoseCommandRanges = field(
        default_factory=lambda: PoseCommandRanges(
            pos_x=(0.4, 0.6),
            pos_y=(-0.25, 0.25),
            pos_z=(0.25, 0.5),
            roll=(0.0, 0.0),
            pitch=(math.pi, math.pi),
            yaw=(-3.14, 3.14),
        )
    )
    """Workspace pose ranges for the lift goal [m, rad]."""

    resampling_time_range: tuple[float, float] = (5.0, 5.0)
    """Min/max seconds between goal resamples [s]."""

    minimal_height: float = 0.04
    """Minimum cube height above its initial spawn to count as lifted [m]."""

    reaching_weight: float = 1.0
    """Reward weight for ee-to-object distance."""

    lifting_weight: float = 15.0
    """Reward weight for object-lifted bonus."""

    goal_tracking_weight: float = 16.0
    """Reward weight for coarse object-goal distance."""

    goal_tracking_fine_weight: float = 5.0
    """Reward weight for fine object-goal distance."""

    @property
    def name(self) -> str:
        return "lift"

    def _cube_name(self, robot: RobotModuleCfg) -> str:
        return f"{robot.name}_lift_object"

    def _table_name(self, robot: RobotModuleCfg) -> str:
        return f"{robot.name}_lift_table"

    def _cmd_name(self, group: str) -> str:
        return f"{group}_{self.name}"

    # ------------------------------------------------------------------
    # Scene assets
    # ------------------------------------------------------------------

    def scene_assets(self, group: str, robot: RobotModuleCfg) -> dict[str, object]:
        return {
            self._table_name(robot): AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{robot.name.title()}_LiftTable",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.0, 0.0, 0.707, 0.707)),
                spawn=_TABLE_SPAWN,
            ),
            self._cube_name(robot): RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{robot.name.title()}_LiftObject",
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=self.cube_init_pos,
                    rot=(0.0, 0.0, 0.0, 1.0),
                ),
                spawn=_CUBE_SPAWN,
            ),
        }

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def command_terms(self, group: str, robot: RobotModuleCfg) -> dict[str, object]:
        asset_name, ee_body, _ = _ROBOT_EE.get(robot.name, (f"{robot.name}_robot", "ee_link", None))
        cmd_name = self._cmd_name(group)
        return {
            cmd_name: mdp.PoseCommandCfg(
                asset_cfg=SceneEntityCfg(asset_name, body_names=[ee_body], selector=group),
                ranges=self.command_ranges,
                resampling_time_range=self.resampling_time_range,
                debug_vis=False,
            ),
        }

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def task_obs_terms(self, group: str, robot: RobotModuleCfg) -> dict[str, ObsTerm]:
        asset_name, _, ee_frame_name = _ROBOT_EE.get(robot.name, (f"{robot.name}_robot", "ee_link", None))
        cube_name = self._cube_name(robot)
        cmd_name = self._cmd_name(group)

        terms: dict[str, ObsTerm] = {
            "object_pos": ObsTerm(
                func=mdp.object_pos_in_robot_frame,
                params={
                    "robot_cfg": SceneEntityCfg(asset_name, selector=group),
                    "object_cfg": SceneEntityCfg(cube_name, selector=group),
                },
            ),
            "object_target_pos_error": ObsTerm(
                func=mdp.object_target_pos_error,
                params={
                    "robot_cfg": SceneEntityCfg(asset_name, selector=group),
                    "object_cfg": SceneEntityCfg(cube_name, selector=group),
                    "command_name": cmd_name,
                },
            ),
        }
        if ee_frame_name is not None:
            terms["ee_object_pos_error"] = ObsTerm(
                func=mdp.ee_object_pos_error,
                params={
                    "robot_cfg": SceneEntityCfg(asset_name, selector=group),
                    "object_cfg": SceneEntityCfg(cube_name, selector=group),
                    "ee_frame_cfg": SceneEntityCfg(ee_frame_name, selector=group),
                },
            )
        return terms

    def scatter_obs_terms(self, group: str, robot: RobotModuleCfg) -> dict[str, tuple[int | None, TermCfg]]:
        asset_name, ee_body, _ = _ROBOT_EE.get(robot.name, (f"{robot.name}_robot", "ee_link", None))
        cmd_name = self._cmd_name(group)
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
        asset_name, _, ee_frame_name = _ROBOT_EE.get(robot.name, (f"{robot.name}_robot", "ee_link", None))
        cube_name = self._cube_name(robot)
        cmd_name = self._cmd_name(group)

        terms: dict[str, RewTerm] = {}

        if ee_frame_name is not None:
            terms[f"{group}_reaching_object"] = RewTerm(
                func=mdp.object_ee_distance,
                weight=self.reaching_weight,
                params={
                    "std": 0.1,
                    "object_cfg": SceneEntityCfg(cube_name, selector=group),
                    "ee_frame_cfg": SceneEntityCfg(ee_frame_name, selector=group),
                },
            )

        terms[f"{group}_lifting_object"] = RewTerm(
            func=mdp.object_is_lifted,
            weight=self.lifting_weight,
            params={
                "minimal_height": self.minimal_height,
                "object_cfg": SceneEntityCfg(cube_name, selector=group),
            },
        )
        terms[f"{group}_object_goal_tracking"] = RewTerm(
            func=mdp.object_goal_distance,
            weight=self.goal_tracking_weight,
            params={
                "std": 0.3,
                "minimal_height": self.minimal_height,
                "command_name": cmd_name,
                "robot_cfg": SceneEntityCfg(asset_name, selector=group),
                "object_cfg": SceneEntityCfg(cube_name, selector=group),
            },
        )
        terms[f"{group}_object_goal_tracking_fine"] = RewTerm(
            func=mdp.object_goal_distance,
            weight=self.goal_tracking_fine_weight,
            params={
                "std": 0.05,
                "minimal_height": self.minimal_height,
                "command_name": cmd_name,
                "robot_cfg": SceneEntityCfg(asset_name, selector=group),
                "object_cfg": SceneEntityCfg(cube_name, selector=group),
            },
        )
        return terms

    # ------------------------------------------------------------------
    # Terminations
    # ------------------------------------------------------------------

    def termination_terms(self, group: str, robot: RobotModuleCfg) -> dict[str, DoneTerm]:
        cube_name = self._cube_name(robot)
        return {
            f"{group}_object_dropping": DoneTerm(
                func=mdp.object_height_below_minimum,
                params={
                    "minimum_height": -0.05,
                    "object_cfg": SceneEntityCfg(cube_name, selector=group),
                },
            ),
        }

    # ------------------------------------------------------------------
    # Reset events
    # ------------------------------------------------------------------

    def reset_events(self, group: str, robot: RobotModuleCfg) -> dict[str, EventTerm]:
        cube_name = self._cube_name(robot)
        # Only reset the task object here; the robot module owns its own reset events.
        return {
            f"{group}_reset_lift_object_default": EventTerm(
                func=mdp.reset_to_default,
                mode="reset",
                params={
                    "asset_cfgs": [SceneEntityCfg(cube_name, selector=group)],
                },
            ),
            f"{group}_reset_lift_object_uniform": EventTerm(
                func=mdp.reset_object_uniform,
                mode="reset",
                params={
                    "pose_range": self.object_spawn_range,
                    "velocity_range": {},
                    "object_cfg": SceneEntityCfg(cube_name, selector=group),
                },
            ),
        }


# ---------------------------------------------------------------------------
# Pre-built instances
# ---------------------------------------------------------------------------

LIFT_TASK = LiftTaskCfg()
"""Standard cube-lifting task (Franka/UR10 workspace)."""

LIFT_TASK_OPENARM = LiftTaskCfg(
    cube_init_pos=(0.4, 0.0, 0.055),
    command_ranges=PoseCommandRanges(
        pos_x=(0.2, 0.4),
        pos_y=(-0.2, 0.2),
        pos_z=(0.15, 0.4),
        roll=(-math.pi / 6, math.pi / 6),
        pitch=(math.pi / 2, math.pi / 2),
        yaw=(-math.pi / 9, math.pi / 9),
    ),
)
"""Cube-lifting task with workspace ranges tuned for the OpenArm robot."""
