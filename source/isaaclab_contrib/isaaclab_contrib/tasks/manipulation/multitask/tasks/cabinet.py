# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cabinet task module: open the top drawer of a Sektion cabinet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBaseCfg as TermCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_contrib.tasks.manipulation.multitask import mdp
from isaaclab_contrib.tasks.manipulation.multitask.mdp.obs import MultiTaskObsTerm

from ._base import TaskModuleCfg

if TYPE_CHECKING:
    from ..robots._base import RobotModuleCfg

# Map robot → (robot_asset_name, finger_joint_pattern, ee_frame_name)
_ROBOT_EE = {
    "franka": ("franka_robot", "panda_finger.*", "franka_ee_frame"),
    "openarm": ("openarm_robot", "openarm_finger_joint.*", "openarm_ee_frame"),
}


@dataclass
class CabinetTaskCfg(TaskModuleCfg):
    """Sektion-cabinet drawer-opening task.

    Spawns a Sektion cabinet and a :class:`FrameTransformerCfg` sensor on
    the drawer handle.  Rewards guide the robot through approach → align →
    grasp → pull phases.  The names of cabinet assets in the scene are
    ``f"{robot.name}_cabinet"`` and ``f"{robot.name}_cabinet_frame"``.

    .. note::
        Currently supports Franka and OpenArm robots.  Other robots must
        have a gripper; providing a robot without a gripper will raise
        a ``KeyError`` at build time.
    """

    cabinet_init_pos: tuple[float, float, float] = (0.8, 0.0, 0.4)
    """Cabinet spawn position relative to env origin [m]."""

    success_threshold: float = 0.39
    """Drawer displacement [m] that triggers the success termination."""

    @property
    def name(self) -> str:
        return "cabinet"

    def _cabinet_name(self, robot: RobotModuleCfg) -> str:
        return f"{robot.name}_cabinet"

    def _frame_name(self, robot: RobotModuleCfg) -> str:
        return f"{robot.name}_cabinet_frame"

    # ------------------------------------------------------------------
    # Scene assets
    # ------------------------------------------------------------------

    def scene_assets(self, group: str, robot: RobotModuleCfg) -> dict[str, object]:
        cab_marker_cfg = FRAME_MARKER_CFG.copy()
        cab_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        cab_marker_cfg.prim_path = f"/Visuals/{group}_CabinetFrame"

        cabinet_name = self._cabinet_name(robot)
        frame_name = self._frame_name(robot)
        title = robot.name.title()

        return {
            cabinet_name: ArticulationCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{title}_Cabinet",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
                    activate_contact_sensors=False,
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=self.cabinet_init_pos,
                    rot=(0.0, 0.0, 1.0, 0.0),
                    joint_pos={
                        "door_left_joint": 0.0,
                        "door_right_joint": 0.0,
                        "drawer_bottom_joint": 0.0,
                        "drawer_top_joint": 0.0,
                    },
                ),
                actuators={
                    "drawers": ImplicitActuatorCfg(
                        joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                        effort_limit_sim=87.0,
                        stiffness=10.0,
                        damping=1.0,
                    ),
                    "doors": ImplicitActuatorCfg(
                        joint_names_expr=["door_left_joint", "door_right_joint"],
                        effort_limit_sim=87.0,
                        stiffness=10.0,
                        damping=2.5,
                    ),
                },
            ),
            frame_name: FrameTransformerCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{title}_Cabinet/sektion",
                debug_vis=True,
                visualizer_cfg=cab_marker_cfg,
                target_frames=[
                    FrameTransformerCfg.FrameCfg(
                        prim_path=f"{{ENV_REGEX_NS}}/{title}_Cabinet/drawer_handle_top",
                        name="drawer_handle_top",
                        offset=OffsetCfg(
                            pos=(0.305, 0.0, 0.01),
                            rot=(0.5, -0.5, -0.5, 0.5),
                        ),
                    ),
                ],
            ),
        }

    # ------------------------------------------------------------------
    # Commands  (cabinet has no pose command — goal is the drawer state)
    # ------------------------------------------------------------------

    def command_terms(self, group: str, robot: RobotModuleCfg) -> dict[str, object]:
        return {}

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def task_obs_terms(self, group: str, robot: RobotModuleCfg) -> dict[str, ObsTerm]:
        cabinet_name = self._cabinet_name(robot)
        frame_name = self._frame_name(robot)
        _, _, ee_frame_name = _ROBOT_EE.get(robot.name, (None, None, None))
        if ee_frame_name is None:
            raise KeyError(
                f"Cabinet task requires a robot with a gripper and EE frame sensor. "
                f"Robot '{robot.name}' is not supported."
            )

        return {
            "cabinet_joint_pos": MultiTaskObsTerm(
                dim=1,
                func=mdp.cabinet_joint_pos,
                params={
                    "cabinet_asset_cfg": SceneEntityCfg(
                        cabinet_name,
                        joint_names=["drawer_top_joint"],
                        selector=group,
                    )
                },
            ),
            "cabinet_joint_vel": MultiTaskObsTerm(
                dim=1,
                func=mdp.cabinet_joint_vel,
                params={
                    "cabinet_asset_cfg": SceneEntityCfg(
                        cabinet_name,
                        joint_names=["drawer_top_joint"],
                        selector=group,
                    )
                },
            ),
            "cabinet_handle_error": ObsTerm(
                func=mdp.cabinet_rel_ee_drawer_distance,
                params={
                    "ee_frame_cfg": SceneEntityCfg(ee_frame_name, selector=group),
                    "cabinet_frame_cfg": SceneEntityCfg(frame_name, selector=group),
                },
            ),
        }

    def scatter_obs_terms(self, group: str, robot: RobotModuleCfg) -> dict[str, tuple[int | None, TermCfg]]:
        # Cabinet contributes no scatter obs (no goal command)
        return {}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def reward_terms(self, group: str, robot: RobotModuleCfg) -> dict[str, RewTerm]:
        robot_name, finger_joint_pattern, ee_frame_name = _ROBOT_EE.get(robot.name, (None, None, None))
        if robot_name is None:
            raise KeyError(
                f"Cabinet task: robot '{robot.name}' is not in the supported-robot map. "
                "Add it to `_ROBOT_EE` in tasks/cabinet.py."
            )
        cabinet_name = self._cabinet_name(robot)
        frame_name = self._frame_name(robot)
        ee_frame_cfg = SceneEntityCfg(ee_frame_name, selector=group)
        cab_frame_cfg = SceneEntityCfg(frame_name, selector=group)
        cab_asset_cfg = SceneEntityCfg(cabinet_name, joint_names=["drawer_top_joint"], selector=group)

        return {
            f"{group}_approach_ee_handle": RewTerm(
                func=mdp.cabinet_approach_ee_handle,
                weight=2.0,
                params={"threshold": 0.2, "ee_frame_cfg": ee_frame_cfg, "cabinet_frame_cfg": cab_frame_cfg},
            ),
            f"{group}_align_ee_handle": RewTerm(
                func=mdp.cabinet_align_ee_handle,
                weight=0.5,
                params={"ee_frame_cfg": ee_frame_cfg, "cabinet_frame_cfg": cab_frame_cfg},
            ),
            f"{group}_approach_gripper_handle": RewTerm(
                func=mdp.cabinet_approach_gripper_handle,
                weight=5.0,
                params={"offset": 0.04, "ee_frame_cfg": ee_frame_cfg, "cabinet_frame_cfg": cab_frame_cfg},
            ),
            f"{group}_align_grasp_around_handle": RewTerm(
                func=mdp.cabinet_align_grasp_around_handle,
                weight=0.125,
                params={"ee_frame_cfg": ee_frame_cfg, "cabinet_frame_cfg": cab_frame_cfg},
            ),
            f"{group}_grasp_handle": RewTerm(
                func=mdp.cabinet_grasp_handle,
                weight=0.5,
                params={
                    "threshold": 0.03,
                    "open_joint_pos": 0.04,
                    "asset_cfg": SceneEntityCfg(robot_name, joint_names=[finger_joint_pattern], selector=group),
                    "ee_frame_cfg": ee_frame_cfg,
                    "cabinet_frame_cfg": cab_frame_cfg,
                },
            ),
            f"{group}_open_drawer_bonus": RewTerm(
                func=mdp.cabinet_open_drawer_bonus,
                weight=7.5,
                params={
                    "ee_frame_cfg": ee_frame_cfg,
                    "cabinet_frame_cfg": cab_frame_cfg,
                    "cabinet_asset_cfg": cab_asset_cfg,
                },
            ),
            f"{group}_multi_stage_open_drawer": RewTerm(
                func=mdp.cabinet_multi_stage_open_drawer,
                weight=1.0,
                params={
                    "ee_frame_cfg": ee_frame_cfg,
                    "cabinet_frame_cfg": cab_frame_cfg,
                    "cabinet_asset_cfg": cab_asset_cfg,
                },
            ),
        }

    # ------------------------------------------------------------------
    # Terminations
    # ------------------------------------------------------------------

    def termination_terms(self, group: str, robot: RobotModuleCfg) -> dict[str, DoneTerm]:
        cabinet_name = self._cabinet_name(robot)
        return {
            f"{group}_cabinet_success": DoneTerm(
                func=mdp.cabinet_drawer_opened,
                params={
                    "threshold": self.success_threshold,
                    "cabinet_asset_cfg": SceneEntityCfg(
                        cabinet_name,
                        joint_names=["drawer_top_joint"],
                        selector=group,
                    ),
                },
            ),
        }

    # ------------------------------------------------------------------
    # Reset events
    # ------------------------------------------------------------------

    def reset_events(self, group: str, robot: RobotModuleCfg) -> dict[str, EventTerm]:
        cabinet_name = self._cabinet_name(robot)
        return {
            f"{group}_reset_cabinet": EventTerm(
                func=mdp.reset_to_default,
                mode="reset",
                params={"asset_cfgs": [SceneEntityCfg(cabinet_name, selector=group)]},
            ),
        }


# ---------------------------------------------------------------------------
# Pre-built instance
# ---------------------------------------------------------------------------

CABINET_TASK = CabinetTaskCfg()
"""Standard Sektion-cabinet drawer-opening task."""
