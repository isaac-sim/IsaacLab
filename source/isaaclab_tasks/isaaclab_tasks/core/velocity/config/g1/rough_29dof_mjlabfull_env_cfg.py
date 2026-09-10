# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""mjlab's whole reward and termination set, ported onto this robot.

The arms so far have each moved one thing from mjlab's G1 velocity task into ours. This one moves
the whole reward set and the whole termination set at once, so that "would their recipe work here"
is answered rather than inferred from four one-term ablations.

What is ported, term for term, from ``mjlab/tasks/velocity/velocity_env_cfg.py`` and its G1 overrides:

=========================  =======  ===================================================
term                       weight   note
=========================  =======  ===================================================
track_linear_velocity       +2.0    ``exp(-(|cmd_xy - v_xy|^2 + v_z^2) / 0.25)``
track_angular_velocity      +2.0    ``exp(-((cmd_z - w_z)^2 + |w_xy|^2) / 0.5)``
upright                     +1.0    ``exp(-|g_xy|^2 / 0.2)`` on ``torso_link``
pose                        +1.0    ``exp(-mean(err^2 / std^2))``, per-joint tolerance
body_ang_vel                -0.05   ``|w_xy|^2`` of ``torso_link``
angular_momentum            -0.02   squared whole-body angular momentum about the COM
dof_pos_limits              -1.0    unchanged from ours
action_rate_l2              -0.1    ours was -0.005
air_time                     0.0    mjlab does not reward air time on the G1
foot_clearance              -2.0    ``|h - 0.1| * |v_xy|`` per foot
foot_swing_height           -0.25   ``(peak / 0.1 - 1)^2`` at touchdown
foot_slip                   -0.1    ``|v_xy|^2`` while in contact
soft_landing                -1e-5   contact force at touchdown
=========================  =======  ===================================================

Terminations become ``time_out``, ``fell_over`` (``bad_orientation`` at 70 degrees) and
``out_of_terrain_bounds``. Everything of ours goes: the terrain-relative base-height termination and
its warm-up, the torso contact termination, the -200 termination penalty, the pelvis-height
penalty, all four ``joint_deviation`` terms, ``flat_orientation_l2``, ``feet_slide``,
``dof_torques_l2``, ``dof_acc_l2``, ``lin_vel_z_l2`` and ``ang_vel_xy_l2``.

**One term is not ported: ``self_collisions`` (-1.0).** It needs a robot-against-robot contact
matrix that this scene's contact sensor does not produce, and enabling self-collision on this asset
is not free -- on the DR29 line it cost 17 cm of pelvis height while buying no success. It is the
only gap, and it is stated here rather than left to be discovered in the numbers.

**What this is not.** The environment is still ours: our asset with the plate-foot rung, our PD
gains, our terrain generator, our command ranges and our observation set. mjlab derives its gains
from motor specs and starts from a knees-bent keyframe; changing those as well would be a different
robot, and the point of this arm is to isolate the reward and termination design.
"""

import math

import torch

from isaaclab.assets import Articulation
from isaaclab.envs.mdp import bad_orientation
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensor
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse

from isaaclab_tasks.core.velocity.mdp.terminations import terrain_out_of_bounds

from .rough_29dof_mjlab_env_cfg import (
    _FOOT_BODIES,
    G129DofRoughMjlabScaleEnvCfg,
    _add_foot_scanners,
    feet_clearance,
    feet_swing_height,
    variable_posture,
)

_LIN_STD = math.sqrt(0.25)
_ANG_STD = math.sqrt(0.5)
_UPRIGHT_STD = math.sqrt(0.2)
_FELL_OVER_ANGLE = math.radians(70.0)
_COMMAND_THRESHOLD = 0.05


def _as_tensor(value):
    """Return ``value`` as a plain tensor, unwrapping this repo's ``ProxyArray`` where present."""
    return value.torch if hasattr(value, "torch") else value


def _command_active(env, command_name: str) -> torch.Tensor:
    """1.0 where the commanded speed is above :data:`_COMMAND_THRESHOLD`, else 0.0."""
    command = env.command_manager.get_command(command_name)
    speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    return (speed > _COMMAND_THRESHOLD).float()


def track_lin_vel_mjlab(env, command_name: str, std: float = _LIN_STD) -> torch.Tensor:
    """``exp(-(|cmd_xy - v_xy|^2 + v_z^2) / std^2)``.

    Differs from ours in two ways: the vertical velocity is part of the tracking error rather than a
    separate ``lin_vel_z_l2`` penalty, and the whole thing is one bounded reward.

    Args:
        env: The environment.
        command_name: Velocity command term.
        std: Tolerance; ``std**2`` divides the squared error.

    Returns:
        Per-environment reward in ``(0, 1]``.
    """
    asset: Articulation = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    velocity = _as_tensor(asset.data.root_lin_vel_b)
    error = torch.sum(torch.square(command[:, :2] - velocity[:, :2]), dim=1) + torch.square(velocity[:, 2])
    return torch.exp(-error / std**2)


def track_ang_vel_mjlab(env, command_name: str, std: float = _ANG_STD) -> torch.Tensor:
    """``exp(-((cmd_z - w_z)^2 + |w_xy|^2) / std^2)``.

    Args:
        env: The environment.
        command_name: Velocity command term.
        std: Tolerance; ``std**2`` divides the squared error.

    Returns:
        Per-environment reward in ``(0, 1]``.
    """
    asset: Articulation = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    omega = _as_tensor(asset.data.root_ang_vel_b)
    error = torch.square(command[:, 2] - omega[:, 2]) + torch.sum(torch.square(omega[:, :2]), dim=1)
    return torch.exp(-error / std**2)


def upright_body_exp(
    env, std: float = _UPRIGHT_STD, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """``exp(-|g_xy|^2 / std^2)`` for a named body, mjlab's ``upright`` on ``torso_link``.

    Args:
        env: The environment.
        std: Tolerance on the projected-gravity magnitude.
        asset_cfg: Articulation and the single body whose tilt is rewarded.

    Returns:
        Per-environment reward in ``(0, 1]``, one when the body is vertical.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    quat = _as_tensor(asset.data.body_quat_w)[:, asset_cfg.body_ids, :].squeeze(1)
    # The articulation exposes ``projected_gravity_b`` for the root only, so the world down vector
    # is written out and rotated into the body's frame here.
    gravity = torch.tensor([0.0, 0.0, -1.0], device=quat.device).expand(quat.shape[0], 3)
    projected = quat_apply_inverse(quat, gravity)
    return torch.exp(-torch.sum(torch.square(projected[:, :2]), dim=1) / std**2)


def body_ang_vel_xy_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Squared world-frame roll/pitch angular velocity of a named body.

    Args:
        env: The environment.
        asset_cfg: Articulation and the single body whose angular velocity is penalized.

    Returns:
        Per-environment penalty magnitude.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    omega = _as_tensor(asset.data.body_ang_vel_w)[:, asset_cfg.body_ids, :].squeeze(1)
    return torch.sum(torch.square(omega[:, :2]), dim=1)


def angular_momentum_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Squared magnitude of the whole-body angular momentum about the centre of mass.

    mjlab reads MuJoCo's ``subtreeangmom`` sensor; there is no equivalent here, so it is assembled
    from the body states: the orbital term ``sum m_i (r_i - r_com) x (v_i - v_com)`` plus the spin
    term ``sum R_i I_i R_i^T w_i``. Its purpose in mjlab is to buy natural arm swing -- a gait that
    swings its arms against its legs cancels its own angular momentum.

    Args:
        env: The environment.
        asset_cfg: The articulation.

    Returns:
        Per-environment penalty magnitude.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    pos = _as_tensor(asset.data.body_com_pos_w)
    vel = _as_tensor(getattr(asset.data, "body_com_lin_vel_w", asset.data.body_lin_vel_w))
    omega = _as_tensor(asset.data.body_ang_vel_w)
    quat = _as_tensor(asset.data.body_quat_w)
    mass = _as_tensor(asset.data.default_mass).to(pos.device)
    total = mass.sum(dim=1, keepdim=True).unsqueeze(-1)
    weights = mass.unsqueeze(-1)
    com = (pos * weights).sum(dim=1, keepdim=True) / total
    com_vel = (vel * weights).sum(dim=1, keepdim=True) / total
    orbital = torch.cross(pos - com, (vel - com_vel) * weights, dim=-1).sum(dim=1)

    inertia = _as_tensor(asset.data.default_inertia).to(pos.device).reshape(*mass.shape, 3, 3)
    rot = matrix_from_quat(quat.reshape(-1, 4)).reshape(*mass.shape, 3, 3)
    world_inertia = rot @ inertia @ rot.transpose(-1, -2)
    spin = (world_inertia @ omega.unsqueeze(-1)).squeeze(-1).sum(dim=1)
    return torch.sum(torch.square(orbital + spin), dim=1)


def feet_slip_l2(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=list(_FOOT_BODIES)),
) -> torch.Tensor:
    """``sum_feet |v_xy|^2`` while the foot is in contact, off when standing.

    Ours (``feet_slide``) uses ``|v_xy|`` and is ungated; mjlab squares it and switches it off below
    a commanded speed, so a foot planted while standing still costs nothing.

    Args:
        env: The environment.
        command_name: Velocity command term, used to gate the penalty.
        sensor_cfg: Contact sensor and the two foot bodies.
        asset_cfg: Articulation and the two foot bodies.

    Returns:
        Per-environment penalty magnitude.
    """
    contacts: ContactSensor = env.scene[sensor_cfg.name]
    in_contact = (contacts.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0).float()
    asset: Articulation = env.scene[asset_cfg.name]
    velocity = _as_tensor(asset.data.body_lin_vel_w)[:, asset_cfg.body_ids, :2]
    return torch.sum(torch.sum(torch.square(velocity), dim=-1) * in_contact, dim=1) * _command_active(env, command_name)


def soft_landing(env, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Contact-force magnitude summed over the feet that touch down this step.

    Args:
        env: The environment.
        command_name: Velocity command term, used to gate the penalty.
        sensor_cfg: Contact sensor and the two foot bodies.

    Returns:
        Per-environment penalty magnitude [N].
    """
    contacts: ContactSensor = env.scene[sensor_cfg.name]
    forces = contacts.data.net_forces_w[:, sensor_cfg.body_ids, :]
    landed = contacts.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids].bool()
    return torch.sum(torch.norm(forces, dim=-1) * landed.float(), dim=1) * _command_active(env, command_name)


@configclass
class G129DofRoughMjlabFullEnvCfg(G129DofRoughMjlabScaleEnvCfg):
    """mjlab's reward set and terminations, on our robot and our environment."""

    def __post_init__(self):
        super().__post_init__()

        rewards = self.rewards
        foot_asset = SceneEntityCfg("robot", body_names=list(_FOOT_BODIES))
        foot_contacts = SceneEntityCfg("contact_forces", body_names=list(_FOOT_BODIES))
        torso = SceneEntityCfg("robot", body_names=["torso_link"])

        # Everything of ours that mjlab does not have.
        for term in (
            "termination_penalty",
            "track_lin_vel_xy_exp",
            "track_ang_vel_z_exp",
            "lin_vel_z_l2",
            "ang_vel_xy_l2",
            "dof_torques_l2",
            "dof_acc_l2",
            "feet_air_time",
            "flat_orientation_l2",
            "feet_slide",
            "undesired_contacts",
            "joint_deviation_hip",
            "joint_deviation_arms",
            "joint_deviation_fingers",
            "joint_deviation_torso",
            "pelvis_height",
        ):
            if hasattr(rewards, term):
                setattr(rewards, term, None)

        _add_foot_scanners(self)
        rewards.track_linear_velocity = RewTerm(
            func=track_lin_vel_mjlab, weight=2.0, params={"command_name": "base_velocity", "std": _LIN_STD}
        )
        rewards.track_angular_velocity = RewTerm(
            func=track_ang_vel_mjlab, weight=2.0, params={"command_name": "base_velocity", "std": _ANG_STD}
        )
        rewards.upright = RewTerm(func=upright_body_exp, weight=1.0, params={"std": _UPRIGHT_STD, "asset_cfg": torso})
        rewards.pose = RewTerm(
            func=variable_posture,
            weight=1.0,
            params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
        )
        rewards.body_ang_vel = RewTerm(func=body_ang_vel_xy_l2, weight=-0.05, params={"asset_cfg": torso})
        rewards.angular_momentum = RewTerm(
            func=angular_momentum_l2, weight=-0.02, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        rewards.action_rate_l2.weight = -0.1
        rewards.dof_pos_limits.weight = -1.0
        rewards.foot_clearance = RewTerm(
            func=feet_clearance,
            weight=-2.0,
            params={"command_name": "base_velocity", "target_height": 0.10, "asset_cfg": foot_asset},
        )
        rewards.foot_swing_height = RewTerm(
            func=feet_swing_height,
            weight=-0.25,
            params={
                "command_name": "base_velocity",
                "target_height": 0.10,
                "asset_cfg": foot_asset,
                "sensor_cfg": foot_contacts,
            },
        )
        rewards.foot_slip = RewTerm(
            func=feet_slip_l2,
            weight=-0.1,
            params={"command_name": "base_velocity", "sensor_cfg": foot_contacts, "asset_cfg": foot_asset},
        )
        rewards.soft_landing = RewTerm(
            func=soft_landing,
            weight=-1.0e-5,
            params={"command_name": "base_velocity", "sensor_cfg": foot_contacts},
        )

        # Terminations: mjlab keeps three, and none of ours.
        for term in ("base_contact", "base_height"):
            if hasattr(self.terminations, term):
                setattr(self.terminations, term, None)
        self.terminations.fell_over = DoneTerm(func=bad_orientation, params={"limit_angle": _FELL_OVER_ANGLE})
        self.terminations.out_of_terrain_bounds = DoneTerm(func=terrain_out_of_bounds, time_out=True)
