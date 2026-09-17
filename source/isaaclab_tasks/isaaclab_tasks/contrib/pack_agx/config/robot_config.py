# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab articulation presets for the Unitree H2 + Sharpa Wave embodiment."""

from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.unitree import H2_SHARPA_CFG

from .metadata import H2_ACTION_JOINT_ORDER, POLICY_58_ORDER

#: Joint angles the articulation resets to; the presets merge task overrides onto these.
H2_DEFAULT_JOINT_POS: dict[str, float] = dict(H2_SHARPA_CFG.init_state.joint_pos)


def make_h2_sharpa_cfg(
    *,
    prim_path: str = "/World/envs/env_.*/Robot",
    init_pos: tuple[float, float, float] | None = None,
    init_rot: tuple[float, float, float, float] | None = None,
    custom_joint_pos: dict[str, float] | None = None,
    base_config: ArticulationCfg = H2_SHARPA_CFG,
) -> ArticulationCfg:
    """H2 + Sharpa cfg with per-task pose overrides merged onto ``H2_DEFAULT_JOINT_POS``.

    Args:
        prim_path: Scene path the articulation spawns at.
        init_pos: Base position [m] in the world frame. Defaults to the pose ``base_config``
            was authored with.
        init_rot: Base orientation as an ``(x, y, z, w)`` quaternion. Defaults to the pose
            ``base_config`` was authored with.
        custom_joint_pos: Per-task joint angles [rad] merged onto ``H2_DEFAULT_JOINT_POS``.
        base_config: Articulation the pose is applied to.
    """
    joint_pos = dict(H2_DEFAULT_JOINT_POS)
    if custom_joint_pos:
        joint_pos.update(custom_joint_pos)
    return base_config.replace(
        prim_path=prim_path,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=base_config.init_state.pos if init_pos is None else init_pos,
            rot=base_config.init_state.rot if init_rot is None else init_rot,
            joint_pos=joint_pos,
            joint_vel={".*": 0.0},
        ),
    )


def h2_body_joint_offsets(custom_joint_pos: dict[str, float] | None = None) -> dict[str, float]:
    """Action offsets that park the joints GR00T does not predict at the robot's default pose.

    The policy emits only ``POLICY_58_ORDER`` (arms and hands); the legs, waist and head entries of
    the action vector are zero-filled by the GR00T action converter. Without an offset those joints
    are driven to 0 rad, which tips the head up from its 0.6 rad default and points the front
    camera at the wall instead of the table.

    Args:
        custom_joint_pos: Per-task overrides merged onto ``H2_DEFAULT_JOINT_POS``, matching the
            ``custom_joint_pos`` passed to :func:`make_h2_sharpa_cfg`.

    Returns:
        Mapping from body joint name to its default position [rad].
    """
    joint_pos = dict(H2_DEFAULT_JOINT_POS)
    if custom_joint_pos:
        joint_pos.update(custom_joint_pos)
    return {name: joint_pos[name] for name in H2_ACTION_JOINT_ORDER if name not in POLICY_58_ORDER}


@configclass
class H2RobotPresets:
    """H2 robot presets."""

    @classmethod
    def h2_sharpa_base_fix(
        cls,
        init_pos: tuple[float, float, float] | None = None,
        init_rot: tuple[float, float, float, float] | None = None,
        custom_joint_pos: dict[str, float] | None = None,
    ) -> ArticulationCfg:
        """H2 + Sharpa Wave, base-fixed. A ``None`` pose keeps the articulation's authored one."""
        return make_h2_sharpa_cfg(init_pos=init_pos, init_rot=init_rot, custom_joint_pos=custom_joint_pos)
