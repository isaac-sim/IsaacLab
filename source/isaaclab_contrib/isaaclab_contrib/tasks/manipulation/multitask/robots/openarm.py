# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OpenArm robot module for multitask environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
    RelativeJointPositionActionCfg,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBaseCfg as TermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg

from isaaclab_contrib.tasks.manipulation.multitask import mdp

from isaaclab_assets.robots.openarm import OPENARM_UNI_HIGH_PD_CFG

from ._base import RobotModuleCfg

_IK_CTRL = DifferentialIKControllerCfg(
    command_type="pose",
    use_relative_mode=True,
    ik_method="dls",
)


@dataclass
class OpenArmRobotCfg(RobotModuleCfg):
    """OpenArm Uni robot module with selectable arm control mode.

    Supports two arm action modes selected by :attr:`arm_action_type`:

    * ``"ik"`` — Differential IK (6-D pose delta), column key ``"arm"``.
    * ``"relative_joint"`` — Relative joint position (6-D), column key ``"openarm_arm"``.

    The gripper always uses a binary open/close action (1-D).
    """

    arm_action_type: Literal["ik", "relative_joint"] = "ik"
    """Arm control mode."""

    ik_scale: float = 0.5
    """Scale factor applied to IK pose-delta commands [--]."""

    joint_scale: float = 0.1
    """Scale factor applied to relative joint-position commands [--]."""

    reset_joint_pos_range: tuple[float, float] = (0.5, 1.25)
    """Uniform range for joint-position reset multiplier [--]."""

    @property
    def name(self) -> str:
        return "openarm"

    @property
    def all_joint_names(self) -> list[str]:
        return ["openarm_joint.*", "openarm_finger_joint.*"]

    # ------------------------------------------------------------------
    # Scene assets
    # ------------------------------------------------------------------

    def scene_assets(self, group: str) -> dict[str, object]:
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = f"/Visuals/{group}_OpenArmEEFrame"

        return {
            "openarm_robot": OPENARM_UNI_HIGH_PD_CFG.replace(
                prim_path="{ENV_REGEX_NS}/OpenArm_Robot",
            ),
            "openarm_ee_frame": FrameTransformerCfg(
                prim_path="{ENV_REGEX_NS}/OpenArm_Robot/openarm_link0",
                debug_vis=False,
                visualizer_cfg=marker_cfg,
                target_frames=[
                    FrameTransformerCfg.FrameCfg(
                        prim_path="{ENV_REGEX_NS}/OpenArm_Robot/openarm_ee_tcp",
                        name="end_effector",
                    ),
                ],
            ),
        }

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_specs(self, group: str | None = None) -> dict[str, tuple[int, object]]:
        if self.arm_action_type == "ik":
            arm = DifferentialInverseKinematicsActionCfg(
                asset_name="openarm_robot",
                joint_names=["openarm_joint.*"],
                body_name="openarm_hand",
                controller=_IK_CTRL,
                scale=self.ik_scale,
            )
            arm_dim = 6
            arm_key = "arm"
        else:
            arm = RelativeJointPositionActionCfg(
                asset_name="openarm_robot",
                joint_names=["openarm_joint.*"],
                scale=self.joint_scale,
            )
            arm_dim = 6
            arm_key = "openarm_arm"

        gripper = BinaryJointPositionActionCfg(
            asset_name="openarm_robot",
            joint_names=["openarm_finger_joint.*"],
            open_command_expr={"openarm_finger_joint.*": 0.044},
            close_command_expr={"openarm_finger_joint.*": 0.0},
        )
        return {
            arm_key: (arm_dim, arm),
            "gripper": (1, gripper),
        }

    # ------------------------------------------------------------------
    # Observations (robot-side scatter contributions)
    # ------------------------------------------------------------------

    def scatter_obs_terms(self, group: str) -> dict[str, tuple[int | None, TermCfg]]:
        return {
            "ee_pose": (
                7,
                TermCfg(
                    func=mdp.ee_pose,
                    params={"asset_cfg": SceneEntityCfg("openarm_robot", body_names=["openarm_hand"], selector=group)},
                ),
            ),
        }

    # ------------------------------------------------------------------
    # Reset events
    # ------------------------------------------------------------------

    def reset_events(self, group: str) -> dict[str, EventTerm]:
        return {
            "openarm_reset_to_default": EventTerm(
                func=mdp.reset_to_default,
                mode="reset",
                params={
                    "reset_joint_targets": True,
                    "asset_cfgs": [SceneEntityCfg("openarm_robot", selector=group)],
                },
            ),
            "openarm_reset_joints": EventTerm(
                func=mdp.reset_joints,
                mode="reset",
                params={
                    "position_range": self.reset_joint_pos_range,
                    "velocity_range": (0.0, 0.0),
                    "asset_cfg": SceneEntityCfg(
                        "openarm_robot",
                        joint_names=["openarm_joint.*", "openarm_finger_joint.*"],
                        selector=group,
                    ),
                },
            ),
        }


# ---------------------------------------------------------------------------
# Pre-built instances
# ---------------------------------------------------------------------------

OPENARM_IK = OpenArmRobotCfg()
"""OpenArm Uni with DiffIK arm control (6-D pose delta, column key ``"arm"``)."""

OPENARM_JOINT = OpenArmRobotCfg(arm_action_type="relative_joint")
"""OpenArm Uni with relative joint-position arm control (6-D, column key ``"openarm_arm"``)."""
