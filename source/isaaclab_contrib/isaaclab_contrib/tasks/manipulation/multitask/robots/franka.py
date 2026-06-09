# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Panda robot module for multitask environments."""

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
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

from isaaclab_contrib.tasks.manipulation.multitask import mdp

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from ._base import RobotModuleCfg

_IK_CTRL = DifferentialIKControllerCfg(
    command_type="pose",
    use_relative_mode=True,
    ik_method="dls",
)


@dataclass
class FrankaRobotCfg(RobotModuleCfg):
    """Franka Panda robot module.

    Supports two arm action modes selected by :attr:`arm_action_type`:

    * ``"ik"`` — Differential IK (6-D pose delta), suitable for reach and lift.
    * ``"relative_joint"`` — Relative joint position (7-D), useful when joint-space
      control is preferred (e.g. cabinet opening with direct joint actuation).

    The gripper always uses a binary open/close action (1-D).
    """

    arm_action_type: Literal["ik", "relative_joint"] = "ik"
    """Arm control mode.  ``"ik"`` for DiffIK (dim=6), ``"relative_joint"`` for
    relative joint position (dim=7)."""

    ik_scale: float = 0.5
    """Scale factor applied to IK pose-delta commands [--]."""

    joint_scale: float = 0.1
    """Scale factor applied to relative joint-position commands [--]."""

    reset_joint_pos_range: tuple[float, float] = (0.5, 1.25)
    """Uniform range for joint-position reset multiplier [--]."""

    @property
    def name(self) -> str:
        return "franka"

    @property
    def all_joint_names(self) -> list[str]:
        return ["panda_joint.*", "panda_finger.*"]

    # ------------------------------------------------------------------
    # Scene assets
    # ------------------------------------------------------------------

    def scene_assets(self, group: str) -> dict[str, object]:
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = f"/Visuals/{group}_FrankaEEFrame"

        return {
            "franka_robot": FRANKA_PANDA_HIGH_PD_CFG.replace(
                prim_path="{ENV_REGEX_NS}/Franka_Robot",
            ),
            "franka_ee_frame": FrameTransformerCfg(
                prim_path="{ENV_REGEX_NS}/Franka_Robot/panda_link0",
                debug_vis=False,
                visualizer_cfg=marker_cfg,
                target_frames=[
                    FrameTransformerCfg.FrameCfg(
                        prim_path="{ENV_REGEX_NS}/Franka_Robot/panda_hand",
                        name="ee_tcp",
                        offset=OffsetCfg(pos=(0.0, 0.0, 0.1034)),
                    ),
                    FrameTransformerCfg.FrameCfg(
                        prim_path="{ENV_REGEX_NS}/Franka_Robot/panda_leftfinger",
                        name="tool_leftfinger",
                        offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
                    ),
                    FrameTransformerCfg.FrameCfg(
                        prim_path="{ENV_REGEX_NS}/Franka_Robot/panda_rightfinger",
                        name="tool_rightfinger",
                        offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
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
                asset_name="franka_robot",
                joint_names=["panda_joint.*"],
                body_name="panda_hand",
                controller=_IK_CTRL,
                scale=self.ik_scale,
                body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
            )
            arm_dim = 6
            arm_key = "arm"
        else:
            arm = RelativeJointPositionActionCfg(
                asset_name="franka_robot",
                joint_names=["panda_joint.*"],
                scale=self.joint_scale,
            )
            arm_dim = 7
            arm_key = "franka_arm"  # separate column for heterogeneous dims

        gripper = BinaryJointPositionActionCfg(
            asset_name="franka_robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
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
                    params={"asset_cfg": SceneEntityCfg("franka_robot", body_names=["panda_hand"], selector=group)},
                ),
            ),
        }

    # ------------------------------------------------------------------
    # Reset events
    # ------------------------------------------------------------------

    def reset_events(self, group: str) -> dict[str, EventTerm]:
        return {
            "franka_reset_to_default": EventTerm(
                func=mdp.reset_to_default,
                mode="reset",
                params={
                    "reset_joint_targets": True,
                    "asset_cfgs": [SceneEntityCfg("franka_robot", selector=group)],
                },
            ),
            "franka_reset_joints": EventTerm(
                func=mdp.reset_joints,
                mode="reset",
                params={
                    "position_range": self.reset_joint_pos_range,
                    "velocity_range": (0.0, 0.0),
                    "asset_cfg": SceneEntityCfg(
                        "franka_robot",
                        joint_names=["panda_joint.*", "panda_finger.*"],
                        selector=group,
                    ),
                },
            ),
        }


# ---------------------------------------------------------------------------
# Pre-built instances
# ---------------------------------------------------------------------------

FRANKA_IK = FrankaRobotCfg()
"""Franka Panda with DiffIK arm control (6-D pose delta, column key ``"arm"``)."""

FRANKA_JOINT = FrankaRobotCfg(arm_action_type="relative_joint")
"""Franka Panda with relative joint-position arm control (7-D, column key ``"franka_arm"``)."""
