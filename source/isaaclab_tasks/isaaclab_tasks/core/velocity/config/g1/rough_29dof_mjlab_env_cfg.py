# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Three things mjlab's G1 velocity task does differently, each as its own arm.

The gait this line produces still hesitates on the left foot, and the numbers that would show it --
``success_rate``, waist pitch, pelvis height -- are all fine. mjlab (``src/mjlab/tasks/velocity``)
trains the same robot on the same task and does three things we do not:

1. **It does not reward air time at all.** ``air_time`` is weight 0 on G1; foot behaviour is priced
   as ``foot_clearance`` -2.0, ``foot_swing_height`` -0.25 and ``foot_slip`` -0.1 instead. Paying for
   time spent airborne rewards a foot that lingers; a clearance *target* does not.
2. **Its action scale is per joint**, ``0.25 * effort_limit / stiffness`` -- the joint offset that
   one unit of action asks for is a quarter of the motor's stall torque, everywhere. We use a
   blanket 0.5, which on this robot means the ankle and the knee get the same authority per unit.
3. **Posture is one reward, not five penalties.** ``variable_posture`` is
   ``exp(-mean(err^2 / std^2))`` with a per-joint ``std`` -- 0.1 on ankle roll, 0.35 on the knee --
   so each joint gets its own tolerance, against our L1/L2 deviation penalties which price every
   radian of every joint the same way.

Each arm changes exactly one of those against :class:`G129DofRoughStandUpEnvCfg`, plus one arm that
combines the first two. Everything else -- the plate-foot rung, waist L2, the height-termination
warm-up, the pelvis-height penalty -- is untouched, so a difference is attributable.

Not ported, deliberately: mjlab's foot is seven capsules where ours is one plate, its gains come
from motor specs, and its default pose is deeper. Those are asset-side and belong in their own
comparison. ``foot_slip`` is also not ported -- our ``feet_slide`` at -0.1 is the same term in a
different form (``|v|`` rather than ``v^2``, ungated), and adding a second would double-price it.
"""

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensor, RayCaster, RayCasterCfg, patterns
from isaaclab.utils.configclass import configclass

from .rough_29dof_standup_env_cfg import G129DofRoughStandUpEnvCfg

_SOLE_OFFSET = 0.035
"""Distance from the ankle-roll body origin down to the sole [m].

The ``a1_feet`` plate sits at z = -0.0252 with a half-thickness of 0.0093, so its underside is
0.0345 below the body origin. mjlab measures from a site at -0.037 on the same link. Rounded to
0.035: the two assets put the sole in the same place to within 2 mm.
"""

_SWING_TARGET = 0.10
"""Sole clearance the swing foot is asked for [m]. mjlab's ``target_height`` for both foot terms."""

_COMMAND_THRESHOLD = 0.05
"""Total commanded speed below which the foot terms are switched off [m/s + rad/s].

Standing still is not a gait, and a foot term that stays on while the command is zero pays the
policy to shuffle in place.
"""

_FOOT_SCANNERS = ("left_foot_scanner", "right_foot_scanner")
_FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")


def _as_tensor(value):
    """Return ``value`` as a plain tensor, unwrapping this repo's ``ProxyArray`` where present."""
    return value.torch if hasattr(value, "torch") else value


def _sole_clearance(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Height of each sole above the terrain directly beneath it, shape ``(num_envs, 2)``.

    One ray caster per foot rather than the pelvis scanner: on generated terrain the ground under
    the pelvis is up to the terrain amplitude away from the ground under the swing foot, which is
    the whole quantity being measured.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    feet_z = _as_tensor(asset.data.body_pos_w)[:, asset_cfg.body_ids, 2]
    grounds = []
    for name in _FOOT_SCANNERS:
        sensor: RayCaster = env.scene[name]
        hits = _as_tensor(sensor.data.ray_hits_w)[..., 2]
        grounds.append(torch.nan_to_num(hits, nan=0.0, posinf=0.0, neginf=0.0).median(dim=1).values)
    return feet_z - torch.stack(grounds, dim=1) - _SOLE_OFFSET


def _command_active(env, command_name: str) -> torch.Tensor:
    """1.0 where the commanded speed is above :data:`_COMMAND_THRESHOLD`, else 0.0."""
    command = env.command_manager.get_command(command_name)
    speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    return (speed > _COMMAND_THRESHOLD).float()


def feet_clearance(
    env,
    command_name: str,
    target_height: float = _SWING_TARGET,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=list(_FOOT_BODIES)),
) -> torch.Tensor:
    """Clearance error weighted by how fast the foot is travelling.

    ``sum_feet |h - target| * ||v_xy||``: a foot parked on the ground costs nothing however far from
    the target it is, and a foot swinging at the wrong height costs in proportion to the swing. This
    is mjlab's ``feet_clearance``.

    Args:
        env: The environment.
        command_name: Velocity command term, used to switch the penalty off when standing.
        target_height: Sole clearance the swing foot is asked for [m].
        asset_cfg: Articulation and the two foot bodies.

    Returns:
        Per-environment penalty magnitude, shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    velocity = _as_tensor(asset.data.body_lin_vel_w)[:, asset_cfg.body_ids, :2].norm(dim=-1)
    error = torch.abs(_sole_clearance(env, asset_cfg) - target_height)
    return torch.sum(error * velocity, dim=1) * _command_active(env, command_name)


class feet_swing_height(ManagerTermBase):
    """Squared relative error of the *peak* swing height, charged once at touchdown.

    ``feet_clearance`` prices the whole trajectory and so is satisfied by a foot that crosses the
    target height briefly; this one remembers the highest point each foot reached while airborne and
    charges ``(peak / target - 1)^2`` on the step it lands. mjlab runs both together for that reason.
    """

    def __init__(self, cfg: RewTerm, env):
        super().__init__(cfg, env)
        self._peak = torch.zeros((env.num_envs, len(_FOOT_BODIES)), device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self._peak[:] = 0.0
        else:
            self._peak[env_ids] = 0.0

    def __call__(
        self,
        env,
        command_name: str,
        target_height: float = _SWING_TARGET,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=list(_FOOT_BODIES)),
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=list(_FOOT_BODIES)),
    ) -> torch.Tensor:
        contacts: ContactSensor = env.scene[sensor_cfg.name]
        air_time = contacts.data.current_air_time[:, sensor_cfg.body_ids]
        clearance = _sole_clearance(env, asset_cfg)
        self._peak = torch.where(air_time > 0.0, torch.maximum(self._peak, clearance), self._peak)
        landed = contacts.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids].bool()
        error = self._peak / target_height - 1.0
        cost = torch.sum(error.square() * landed.float(), dim=1) * _command_active(env, command_name)
        self._peak = torch.where(landed, torch.zeros_like(self._peak), self._peak)
        return cost


_POSTURE_STD_STANDING = {".*": 0.05}
_POSTURE_STD_WALKING = {
    ".*hip_pitch.*": 0.3,
    ".*hip_roll.*": 0.15,
    ".*hip_yaw.*": 0.15,
    ".*knee.*": 0.35,
    ".*ankle_pitch.*": 0.25,
    ".*ankle_roll.*": 0.1,
    ".*waist_yaw.*": 0.2,
    ".*waist_roll.*": 0.08,
    ".*waist_pitch.*": 0.1,
    ".*shoulder_pitch.*": 0.15,
    ".*shoulder_roll.*": 0.15,
    ".*shoulder_yaw.*": 0.1,
    ".*elbow.*": 0.15,
    ".*wrist.*": 0.3,
    ".*hand.*": 0.3,
}
"""Per-joint tolerance while walking [rad], copied from mjlab's G1 config.

Tight where deviation is a fault (ankle roll 0.1, waist roll 0.08) and loose where it is the stride
(knee 0.35, hip pitch 0.3). ``.*hand.*`` is ours: this asset has finger joints that mjlab's does not,
and every joint must match a pattern.
"""


class variable_posture(ManagerTermBase):
    """``exp(-mean(err^2 / std^2))`` against the default pose, with a per-joint ``std``.

    A positive reward rather than a penalty, and one term for all 43 joints instead of five
    deviation penalties. The tolerance widens once the command asks for motion, so holding still is
    priced strictly and striding is not.
    """

    def __init__(self, cfg: RewTerm, env):
        super().__init__(cfg, env)
        from isaaclab.utils.string import resolve_matching_names_values  # noqa: PLC0415

        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self._ids, names = asset.find_joints(".*")
        stds = []
        for table in (_POSTURE_STD_STANDING, _POSTURE_STD_WALKING):
            _, _, values = resolve_matching_names_values(table, names)
            stds.append(torch.tensor(values, device=env.device, dtype=torch.float32))
        self._std_standing, self._std_walking = stds

    def __call__(
        self,
        env,
        command_name: str,
        walking_threshold: float = 0.05,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
        walking = (speed >= walking_threshold).float().unsqueeze(1)
        std = self._std_walking * walking + self._std_standing * (1.0 - walking)
        error = _as_tensor(asset.data.joint_pos)[:, self._ids] - _as_tensor(asset.data.default_joint_pos)[:, self._ids]
        return torch.exp(-torch.mean(error.square() / std.square(), dim=1))


_MJLAB_ACTION_SCALE = {
    ".*_hip_yaw_joint": 0.1467,
    ".*_hip_roll_joint": 0.1467,
    ".*_hip_pitch_joint": 0.11,
    ".*_knee_joint": 0.1738,
    "waist_yaw_joint": 0.11,
    "waist_roll_joint": 0.0625,
    "waist_pitch_joint": 0.0625,
    ".*_ankle_pitch_joint": 0.625,
    ".*_ankle_roll_joint": 0.625,
    ".*_shoulder_.*_joint": 0.1563,
    ".*_elbow_joint": 0.1563,
    ".*_wrist_roll_joint": 0.1563,
    ".*_wrist_pitch_joint": 0.0313,
    ".*_wrist_yaw_joint": 0.0313,
    ".*_hand_.*_joint": 0.0153,
}
"""``0.25 * effort_limit / stiffness`` per joint, mjlab's rule applied to this robot.

Torque ceilings from :data:`~.rough_29dof_env_cfg.HARDWARE_EFFORT_LIMITS` (the MJCF's ``ctrlrange``)
and stiffness from :data:`~isaaclab_assets.robots.unitree.G1_29DOF_VELOCITY_CFG`. The hardware value
is used rather than ``effort_limit_sim``, which is a blanket 300 on the legs and arms and carries no
information; the simulated ceilings are left alone, so this arm changes the action scale and nothing
else.

Against the blanket 0.5 this is 3-4x *smaller* everywhere except the ankle, which goes to 0.625 --
the ankle is the joint with the weakest motor against the stiffest gain, and it is the one that
places the foot.
"""


def _add_foot_scanners(cfg) -> None:
    """Attach a downward ray caster to each foot, six rays over the sole footprint."""
    for name, body in zip(_FOOT_SCANNERS, _FOOT_BODIES):
        setattr(
            cfg.scene,
            name,
            RayCasterCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{body}",
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
                ray_alignment="yaw",
                pattern_cfg=patterns.GridPatternCfg(resolution=0.06, size=[0.12, 0.06]),
                debug_vis=False,
                mesh_prim_paths=["/World/ground"],
                global_world_only=True,
            ),
        )


def _add_mjlab_gait(cfg) -> None:
    """Swap the air-time reward for mjlab's clearance pair."""
    _add_foot_scanners(cfg)
    cfg.rewards.feet_air_time.weight = 0.0
    # ``asset_cfg``/``sensor_cfg`` have to be passed explicitly: the manager only resolves names to
    # indices for parameters that appear in ``params``, and a default left in the signature resolves
    # to every body on the robot.
    foot_asset = SceneEntityCfg("robot", body_names=list(_FOOT_BODIES))
    foot_contacts = SceneEntityCfg("contact_forces", body_names=list(_FOOT_BODIES))
    cfg.rewards.feet_clearance = RewTerm(
        func=feet_clearance,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "target_height": _SWING_TARGET,
            "asset_cfg": foot_asset,
        },
    )
    cfg.rewards.feet_swing_height = RewTerm(
        func=feet_swing_height,
        weight=-0.25,
        params={
            "command_name": "base_velocity",
            "target_height": _SWING_TARGET,
            "asset_cfg": foot_asset,
            "sensor_cfg": foot_contacts,
        },
    )


def _add_mjlab_posture(cfg) -> None:
    """Replace the deviation penalties with the single tolerance-weighted posture reward."""
    for term in ("joint_deviation_hip", "joint_deviation_arms", "joint_deviation_fingers", "joint_deviation_torso"):
        setattr(cfg.rewards, term, None)
    cfg.rewards.posture = RewTerm(
        func=variable_posture,
        weight=1.0,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class G129DofRoughMjlabGaitEnvCfg(G129DofRoughStandUpEnvCfg):
    """One change: air time unpriced, foot clearance and swing height priced instead."""

    def __post_init__(self):
        super().__post_init__()
        _add_mjlab_gait(self)


@configclass
class G129DofRoughMjlabScaleEnvCfg(G129DofRoughStandUpEnvCfg):
    """One change: the per-joint action scale."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.scale = dict(_MJLAB_ACTION_SCALE)


_MJLAB_ACTION_SCALE_HIPKNEE = {
    **_MJLAB_ACTION_SCALE,
    ".*_hip_pitch_joint": 0.5,
    ".*_knee_joint": 0.5,
}
"""The mjlab table with hip pitch and knee handed back the blanket 0.5.

mjlab's rule is ``0.25 * effort_limit / stiffness``, and on this robot the ankle carries kp 20
against 200 at the hip pitch and knee, so the rule gives the ankle 0.625 and the hip pitch 0.110 --
the ankle ends up with 5.7 times the per-unit angular authority of the hip. Measured on the depth
student of that arm, on flat ground under a pinned command: at 0.5 m/s it walks (single-stance
0.889) but at 0.1 m/s it stops stepping altogether, 95% of the time on both feet, and on hardware it
goes over backwards at low speed because there is no swing leg to catch the lean. The blanket-0.5
student in the same environment keeps stepping at 0.1 m/s (single-stance 0.601) and stands 6 cm
taller.

This arm keeps the ankle's 0.625 -- that is the part of mjlab's table that plausibly earns the
crisper gait -- and restores 0.5 on the two joints that swing the leg. One variable against
``yms``: nothing else in the environment differs.
"""


@configclass
class G129DofRoughMjlabScaleHipKneeEnvCfg(G129DofRoughStandUpEnvCfg):
    """One change against ``ms``: hip pitch and knee go back to the blanket 0.5."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.scale = dict(_MJLAB_ACTION_SCALE_HIPKNEE)


@configclass
class G129DofRoughMjlabGaitScaleEnvCfg(G129DofRoughMjlabGaitEnvCfg):
    """Both of the above, to see whether they need each other."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.scale = dict(_MJLAB_ACTION_SCALE)


@configclass
class G129DofRoughMjlabPostureEnvCfg(G129DofRoughStandUpEnvCfg):
    """One change: the five deviation penalties replaced by ``variable_posture``."""

    def __post_init__(self):
        super().__post_init__()
        _add_mjlab_posture(self)


@configclass
class G129DofRoughMjlabScaleHwTorqueEnvCfg(G129DofRoughMjlabScaleEnvCfg):
    """The per-joint action scale, with the torque ceilings it was computed from actually enforced.

    The scale in :data:`_MJLAB_ACTION_SCALE` is ``0.25 * effort_limit / stiffness`` using the
    hardware's ``ctrlrange``, but ``effort_limit_sim`` is left at ``G1_29DOF_VELOCITY_CFG``'s blanket
    placeholders -- 300 on the legs and arms, 20 on the ankles, 300 on the hands. So the rule and the
    ceiling disagree, and they disagree worst exactly where it matters: **the simulated ankle is
    capped at 20 N*m against the hardware's 50**, which is the joint that places the foot and sets
    the body roll, and the one the per-joint scale was supposed to give more authority to.

    This arm enforces the hardware ceilings, which makes one unit of action a quarter of the motor's
    real capacity *and* the motor's real capacity the limit. Everything else is ``ms``.

    The ankle ceiling has been measured to matter on its own before: at 20 N*m the stock task ends at
    ``error_vel_yaw`` 1.06 against a 0.4 success threshold while linear tracking is already fine.
    """

    def __post_init__(self):
        super().__post_init__()

        from .rough_29dof_env_cfg import _apply_hardware_efforts  # noqa: PLC0415

        _apply_hardware_efforts(self)


_LOCOMOTION_JOINTS = [
    ".*_hip_pitch_joint",
    ".*_hip_roll_joint",
    ".*_hip_yaw_joint",
    ".*_knee_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
    ".*_wrist_roll_joint",
    ".*_wrist_pitch_joint",
    ".*_wrist_yaw_joint",
]
"""The 29 joints mjlab's G1 has: everything except the fourteen finger joints."""


@configclass
class G129DofRoughMjlabScaleNoFingersEnvCfg(G129DofRoughMjlabScaleEnvCfg):
    """``ms`` with the fingers out of the action and observation spaces, as mjlab's G1 has them.

    mjlab trains ``g1_29dof_rev_1_0``: 29 joints, no hand. This asset carries 43, and the extra
    fourteen are not free:

    * **They are saturated.** Mean ``|action|`` on the hands is 26 against 0.37 on the legs, so with
      ``scale`` and ``use_default_offset`` the finger targets sit at roughly +-13 rad against +-1.75
      rad limits -- permanently against the stops, flipping every step. Under the blanket 0.5 scale
      one unit of action was 816% of a finger motor's 2.45 N*m; they never had a usable range.
    * **They dominate the penalties that are summed over all joints.** ``action_rate_l2``,
      ``dof_torques_l2`` and ``dof_acc_l2`` are computed across the whole action or joint vector, so
      the term meant to smooth the legs is spending most of itself on fingers. It is also what made
      a first pass at measuring the camera's influence read 20x too small.
    * **They are dropped at deployment anyway.** ``g1_deploy`` maps all fourteen to motor index -1;
      the robot has 29 motors.

    The fingers keep their actuators and stay at their default pose, which is zero for all fourteen,
    so nothing about the robot's mass or contact set changes -- only what the policy writes and
    reads. Action dimension 43 -> 29; the observation loses 42 values (fingers in ``joint_pos``,
    ``joint_vel`` and ``actions``).
    """

    def __post_init__(self):
        super().__post_init__()

        self.actions.joint_pos.joint_names = list(_LOCOMOTION_JOINTS)
        # The per-joint scale is resolved against the action's own joint list, so an entry for a
        # joint that is no longer an action is an unmatched pattern rather than a harmless extra.
        self.actions.joint_pos.scale = {
            pattern: value for pattern, value in _MJLAB_ACTION_SCALE.items() if "_hand_" not in pattern
        }
        for term in ("joint_pos", "joint_vel"):
            getattr(self.observations.policy, term).params["asset_cfg"] = SceneEntityCfg(
                "robot", joint_names=list(_LOCOMOTION_JOINTS)
            )
