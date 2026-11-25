# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ANYmal D velocity tracking on flat ground, with its learned actuator network."""

from isaaclab_experimental import mdp2
from isaaclab_experimental.mdp2 import mdp

from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass, replace

from isaaclab_tasks.core.velocity.velocity_env_cfg import MySceneCfg

from isaaclab_assets.robots.anymal import ANYMAL_D_CFG

from ..velocity_flat import (
    FLAT_TERRAIN_SCENE,
    FLAT_VELOCITY_TASK,
    FlatVelocityEnvCfg,
    FlatVelocityPhysicsCfg,
    contact_forces,
    joints,
    robot,
    root_state,
    scene,
    velocity_tracking,
)

TASK = replace(
    FLAT_VELOCITY_TASK,
    actions={
        "joint_pos": mdp2.DirectAction(robot.joint_pos_target, scale=0.5, offset=robot.default_joint_pos, subset=joints)
    },
    rewards={
        "track_lin_vel_xy_exp": mdp2.Reward(
            mdp.track_lin_vel_xy_exp, inputs=velocity_tracking, params={"std": 0.5}, weight=1.0
        ),
        "track_ang_vel_z_exp": mdp2.Reward(
            mdp.track_ang_vel_z_exp, inputs=velocity_tracking, params={"std": 0.5}, weight=0.5
        ),
        "lin_vel_z_l2": mdp2.Reward(mdp.lin_vel_z_l2, inputs=root_state, weight=-2.0),
        "ang_vel_xy_l2": mdp2.Reward(mdp.ang_vel_xy_l2, inputs=root_state, weight=-0.05),
        "dof_torques_l2": mdp2.Reward(
            mdp.joint_torques_l2, inputs={"applied_effort": robot.applied_effort, "joint_ids": joints}, weight=-2.5e-5
        ),
        "dof_acc_l2": mdp2.Reward(
            mdp.joint_acc_l2, inputs={"joint_acc": robot.joint_acc, "joint_ids": joints}, weight=-2.5e-7
        ),
        "action_rate_l2": mdp2.Reward(mdp.action_rate_l2, inputs={"actions": mdp2.last_action}, weight=-0.01),
        "feet_air_time": mdp2.Reward(
            mdp.feet_air_time,
            inputs={
                "current_contact_time": contact_forces.current_contact_time,
                "last_air_time": contact_forces.last_air_time,
                "command": mdp2.commands.base_velocity,
                "body_ids": contact_forces.bodies(".*FOOT"),
            },
            params={"threshold": 0.5, "step_dt": mdp2.step_dt},
            weight=0.5,
        ),
        "undesired_contacts": mdp2.Reward(
            mdp.undesired_contacts,
            inputs={
                "net_normal_forces_w_history": contact_forces.net_normal_forces_w_history,
                "body_ids": contact_forces.bodies(".*THIGH"),
            },
            params={"threshold": 1.0},
            weight=-1.0,
        ),
        "flat_orientation_l2": mdp2.Reward(
            mdp.flat_orientation_l2,
            inputs={"root_link_pose_w": robot.root_link_pose_w, "gravity_w": scene.gravity_w},
            weight=-5.0,
        ),
    },
)
"""The MDP of the ANYmal D task."""

_FLAT_DEFAULTS = FlatVelocityEnvCfg()
_FLAT_MJWARP = _FLAT_DEFAULTS.sim.physics.newton_mjwarp
_ANYMAL_D_MJWARP = replace(
    _FLAT_MJWARP, solver_cfg=replace(_FLAT_MJWARP.solver_cfg, njmax=60, nconmax=25, cone="elliptic", impratio=100.0)
)


@configclass
class AnymalDFlatMDP2EnvCfg(FlatVelocityEnvCfg):
    """Flat-ground velocity tracking with ANYmal D."""

    sim: SimulationCfg = replace(
        _FLAT_DEFAULTS.sim, physics=FlatVelocityPhysicsCfg(newton_mjwarp=_ANYMAL_D_MJWARP, default=_ANYMAL_D_MJWARP)
    )
    scene: MySceneCfg = replace(
        FLAT_TERRAIN_SCENE,
        robot=replace(
            ANYMAL_D_CFG,
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=replace(ANYMAL_D_CFG.spawn, variants={"Colliders": "minimal"}),
            init_state=replace(ANYMAL_D_CFG.init_state, pos=(0.0, 0.0, 0.65)),
        ),
    )
    task: mdp2.TaskCfg = TASK
