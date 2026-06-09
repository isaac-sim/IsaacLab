# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR10 robot module for multitask environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg, RelativeJointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBaseCfg as TermCfg
from isaaclab.managers import SceneEntityCfg

from isaaclab_contrib.tasks.manipulation.multitask import mdp

from isaaclab_assets.robots.universal_robots import UR10_CFG

from ._base import RobotModuleCfg

_IK_CTRL = DifferentialIKControllerCfg(
    command_type="pose",
    use_relative_mode=True,
    ik_method="dls",
)


@dataclass
class UR10RobotCfg(RobotModuleCfg):
    """UR10 robot module with selectable arm control mode.

    Supports two arm action modes selected by :attr:`arm_action_type`:

    * ``"ik"`` — Differential IK (6-D pose delta), column key ``"arm"``.
    * ``"relative_joint"`` — Relative joint position (6-D), column key ``"ur10_arm"``.

    The UR10 has no gripper; the ``"gripper"`` column is not registered.
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
        return "ur10"

    @property
    def all_joint_names(self) -> list[str]:
        return [".*"]

    # ------------------------------------------------------------------
    # Scene assets
    # ------------------------------------------------------------------

    def scene_assets(self, group: str) -> dict[str, object]:
        return {
            "ur10_robot": UR10_CFG.replace(
                prim_path="{ENV_REGEX_NS}/UR10_Robot",
            ),
        }

    # ------------------------------------------------------------------
    # Actions  (arm only — no gripper)
    # ------------------------------------------------------------------

    def action_specs(self, group: str | None = None) -> dict[str, tuple[int, object]]:
        if self.arm_action_type == "ik":
            arm = DifferentialInverseKinematicsActionCfg(
                asset_name="ur10_robot",
                joint_names=[".*"],
                body_name="ee_link",
                controller=_IK_CTRL,
                scale=self.ik_scale,
            )
            return {"arm": (6, arm)}
        else:
            arm = RelativeJointPositionActionCfg(
                asset_name="ur10_robot",
                joint_names=[".*"],
                scale=self.joint_scale,
            )
            return {"ur10_arm": (6, arm)}

    # ------------------------------------------------------------------
    # Observations (robot-side scatter contributions)
    # ------------------------------------------------------------------

    def scatter_obs_terms(self, group: str) -> dict[str, tuple[int | None, TermCfg]]:
        return {
            "ee_pose": (
                7,
                TermCfg(
                    func=mdp.ee_pose,
                    params={"asset_cfg": SceneEntityCfg("ur10_robot", body_names=["ee_link"], selector=group)},
                ),
            ),
        }

    # ------------------------------------------------------------------
    # Reset events
    # ------------------------------------------------------------------

    def reset_events(self, group: str) -> dict[str, EventTerm]:
        return {
            "ur10_reset_to_default": EventTerm(
                func=mdp.reset_to_default,
                mode="reset",
                params={
                    "reset_joint_targets": True,
                    "asset_cfgs": [SceneEntityCfg("ur10_robot", selector=group)],
                },
            ),
            "ur10_reset_joints": EventTerm(
                func=mdp.reset_joints,
                mode="reset",
                params={
                    "position_range": self.reset_joint_pos_range,
                    "velocity_range": (0.0, 0.0),
                    "asset_cfg": SceneEntityCfg("ur10_robot", joint_names=[".*"], selector=group),
                },
            ),
        }


# ---------------------------------------------------------------------------
# Pre-built instances
# ---------------------------------------------------------------------------

UR10_IK = UR10RobotCfg()
"""UR10 with DiffIK arm control (6-D pose delta, column key ``"arm"``)."""

UR10_JOINT = UR10RobotCfg(arm_action_type="relative_joint")
"""UR10 with relative joint-position arm control (6-D, column key ``"ur10_arm"``)."""
