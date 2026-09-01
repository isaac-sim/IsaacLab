# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCaster
from isaaclab.utils.configclass import configclass

from .rough_env_cfg import G1RoughEnvCfg

MINIMUM_PELVIS_HEIGHT = 0.4
"""Height above the terrain below which the episode ends [m].

Not a tuning knob -- the task has no other signal for a robot that has stopped standing. Its only
termination is torso contact, and on this asset that never fires: the robot rests weight on the
pelvis and knee colliders, keeps the torso clear of the ground, and rides out the full episode in a
crouch. Measured on flat ground, ``success_rate`` ends at 0.010 without a height term and 1.000
with one. A tilt termination does not substitute -- the torso stays vertical throughout, so 30
degrees is never reached and the score only reaches 0.385.
"""

##
# Pre-defined configs
##
from isaaclab_assets import G1_29DOF_VELOCITY_CFG  # isort: skip


def retarget_g1_rewards_to_29dof(cfg: G1RoughEnvCfg) -> None:
    """Respell every joint name in ``cfg``'s rewards for the 29-DoF G1.

    ``G1_MINIMAL_CFG`` is built from a superseded URDF whose waist is a single ``torso_joint``
    and whose forearm is ``elbow_pitch`` / ``elbow_roll``. The current robot names those
    ``waist_yaw``/``waist_roll``/``waist_pitch`` and ``elbow``/``wrist_roll``, and adds wrist
    pitch and yaw. Body names are unchanged -- ``pelvis``, ``torso_link`` and
    ``.*_ankle_roll_link`` exist on both -- so the height scanner, the base-contact termination
    and the base-mass and external-force events need no adjustment.

    Args:
        cfg: The environment configuration to retarget, modified in place.
    """
    cfg.rewards.joint_deviation_arms.params["asset_cfg"] = SceneEntityCfg(
        "robot",
        joint_names=[
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
            ".*_elbow_joint",
            ".*_wrist_roll_joint",
            ".*_wrist_pitch_joint",
            ".*_wrist_yaw_joint",
        ],
    )
    # Dex3 naming: ``.*_hand_{thumb,middle,index}_N_joint``.
    cfg.rewards.joint_deviation_fingers.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=[".*_hand_.*_joint"])
    # Waist roll and pitch are new, and go unpenalized without being named here.
    cfg.rewards.joint_deviation_torso.params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    )


def pelvis_below_terrain_clearance(
    env,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
) -> torch.Tensor:
    """Terminate when the root sits less than ``minimum_height`` above the ground beneath it.

    :func:`~isaaclab.envs.mdp.terminations.root_height_below_minimum` compares against world z and
    its docstring restricts it to flat ground; on generated terrain it ends episodes for standing in
    a dip -- 89% of them, measured. The reference here is the median ray hit rather than the mean,
    because the scanner spans 1.6 x 1.0 m and on broken ground a few rays land on a ledge, which
    drags a mean far enough to end healthy episodes.

    Args:
        env: The environment.
        minimum_height: Clearance below which the episode ends [m].
        asset_cfg: Articulation whose root height is checked.
        sensor_cfg: Ray caster defining the ground beneath the robot.

    Returns:
        Boolean tensor, one entry per environment.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene[sensor_cfg.name]
    hits = sensor.data.ray_hits_w.torch[..., 2]
    ground = torch.nan_to_num(hits, nan=0.0, posinf=0.0, neginf=0.0).median(dim=1).values
    return asset.data.root_pos_w.torch[:, 2] - ground < minimum_height


@configclass
class G129DofRoughEnvCfg(G1RoughEnvCfg):
    """Rough-terrain velocity tracking for the Unitree G1 on the current 29-DoF asset."""

    def __post_init__(self):
        super().__post_init__()

        # scene
        self.scene.robot = G1_29DOF_VELOCITY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # rewards
        retarget_g1_rewards_to_29dof(self)
        # terminations
        self.terminations.base_height = DoneTerm(
            func=pelvis_below_terrain_clearance,
            params={
                "minimum_height": MINIMUM_PELVIS_HEIGHT,
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )
