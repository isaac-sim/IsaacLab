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


HARDWARE_EFFORT_LIMITS = {
    "hip": 88.0,
    "knee": 139.0,
    "waist_yaw": 88.0,
    "ankle": 50.0,
    "waist_roll_pitch": 50.0,
    "shoulder_elbow": 25.0,
    "wrist_roll": 25.0,
    "wrist_pitch_yaw": 5.0,
    "hand": 2.45,
}
"""Per-joint torque ceilings [N*m] from the official MJCF's ``ctrlrange``.

:data:`~isaaclab_assets.robots.unitree.G1_29DOF_VELOCITY_CFG` inherits ``G1_CFG``'s blanket 300 for
the legs, 300 for the arms and 20 for the feet. All three are wrong against the hardware, and the
ankle is wrong in the direction that matters: a policy trained against a 20 N*m ankle never learns
it has 50, and 20 is not enough to turn on rough terrain -- the default task ends at
``error_vel_yaw`` 1.06 against a 0.4 success threshold while linear tracking is already fine at
0.31.
"""


def _apply_hardware_efforts(cfg, ankle_only: bool = False) -> None:
    """Replace the blanket torque ceilings with the hardware's, in place.

    Args:
        cfg: Environment config whose robot actuators are rewritten.
        ankle_only: Change the ankle alone, leaving every other group on the blanket value. Isolates
            the ankle as a cause rather than changing nine numbers at once.
    """
    actuators = cfg.scene.robot.actuators
    actuators["feet"].effort_limit_sim = HARDWARE_EFFORT_LIMITS["ankle"]
    if ankle_only:
        return
    actuators["legs"].effort_limit_sim = {
        ".*_hip_.*": HARDWARE_EFFORT_LIMITS["hip"],
        ".*_knee_joint": HARDWARE_EFFORT_LIMITS["knee"],
        "waist_yaw_joint": HARDWARE_EFFORT_LIMITS["waist_yaw"],
    }
    actuators["waist"].effort_limit_sim = HARDWARE_EFFORT_LIMITS["waist_roll_pitch"]
    actuators["arms"].effort_limit_sim = {
        ".*_shoulder_.*": HARDWARE_EFFORT_LIMITS["shoulder_elbow"],
        ".*_elbow_joint": HARDWARE_EFFORT_LIMITS["shoulder_elbow"],
        ".*_wrist_roll_joint": HARDWARE_EFFORT_LIMITS["wrist_roll"],
        ".*_wrist_pitch_joint": HARDWARE_EFFORT_LIMITS["wrist_pitch_yaw"],
        ".*_wrist_yaw_joint": HARDWARE_EFFORT_LIMITS["wrist_pitch_yaw"],
    }
    actuators["hands"].effort_limit_sim = HARDWARE_EFFORT_LIMITS["hand"]


@configclass
class G129DofRoughRealAnkleEnvCfg(G129DofRoughEnvCfg):
    """One variable changed against :class:`G129DofRoughEnvCfg`: the ankle torque ceiling."""

    def __post_init__(self):
        super().__post_init__()
        _apply_hardware_efforts(self, ankle_only=True)


@configclass
class G129DofRoughRealTorqueEnvCfg(G129DofRoughEnvCfg):
    """Every torque ceiling taken from the hardware, as the sim-to-real DR29 task does."""

    def __post_init__(self):
        super().__post_init__()
        _apply_hardware_efforts(self)


_AIR_TIME_WEIGHTS = {"x4": 1.0, "x8": 2.0}
"""Multiples of the stock ``feet_air_time`` weight of 0.25, bracketing rather than guessing.

``G129DofRoughRealAnkleEnvCfg`` walks in short shuffling steps: ``feet_air_time_positive_biped``
returns the single-stance duration clamped at its 0.4 s threshold, so the term saturates at 0.1 per
step under the stock weight, and that policy earns 0.0052 -- about five per cent of what is on
offer. The term is live; the policy simply is not paid enough to lengthen its stride.
"""


@configclass
class G129DofRoughRealAnkleAirTime4EnvCfg(G129DofRoughRealAnkleEnvCfg):
    """The ankle fix plus four times the stock air-time weight."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.feet_air_time.weight = _AIR_TIME_WEIGHTS["x4"]


@configclass
class G129DofRoughRealAnkleAirTime8EnvCfg(G129DofRoughRealAnkleEnvCfg):
    """The ankle fix plus eight times the stock air-time weight."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.feet_air_time.weight = _AIR_TIME_WEIGHTS["x8"]
