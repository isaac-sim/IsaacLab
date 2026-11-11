# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree Go2 velocity tracking: follow a commanded base velocity and heading on flat ground."""

import math

from isaaclab_experimental import mdp2
from isaaclab_experimental.envs import MDP2RLEnvCfg
from isaaclab_experimental.mdp2 import mdp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass, replace

from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

SIM_DT = 0.005
DECIMATION = 4
GROUND_MATERIAL = sim_utils.PhysxRigidBodyMaterialCfg(
    friction_combine_mode="multiply", restitution_combine_mode="multiply", static_friction=1.0, dynamic_friction=1.0
)

robot = mdp2.Entity("robot")
contact_forces = mdp2.Entity("contact_forces")
scene = mdp2.Entity("scene")
joints = robot.joints(".*")
bodies = robot.bodies(".*")
base = robot.bodies("base")
feet = contact_forces.bodies(".*_foot")
base_contact = contact_forces.bodies("base")

root_state = {"root_link_pose_w": robot.root_link_pose_w, "root_com_vel_w": robot.root_com_vel_w}
velocity_tracking = {**root_state, "command": mdp2.commands.base_velocity}


@configclass
class UnitreeGo2FlatSceneCfg(InteractiveSceneCfg):
    """A Unitree Go2 on a ground plane, with contact sensors on all its bodies, lit by the sky."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=GROUND_MATERIAL,
    )
    robot: ArticulationCfg = replace(UNITREE_GO2_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/[^/]*", history_length=3, track_air_time=True, update_period=SIM_DT
    )
    sky_light = AssetBaseCfg(prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=750.0))


@configclass
class UnitreeGo2FlatPhysicsCfg(PresetCfg):
    """Newton physics presets of the Unitree Go2 flat velocity task."""

    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=65,
            nconmax=35,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
        num_substeps=2,
        debug_mode=False,
        default_shape_cfg=NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0),
    )
    default: NewtonCfg = newton_mjwarp


TASK = mdp2.TaskCfg(
    actions={
        "joint_pos": mdp2.DirectAction(robot.joint_pos_target, scale=0.25, offset=robot.default_joint_pos),
    },
    commands={
        "base_velocity": mdp2.Command(
            mdp.heading_velocity_command,
            inputs={"root_link_pose_w": robot.root_link_pose_w},
            resampling_time_range=(10.0, 10.0),
            params={
                "lin_vel_x": (-1.0, 1.0),
                "lin_vel_y": (-1.0, 1.0),
                "ang_vel_z": (-1.0, 1.0),
                "heading": (-math.pi, math.pi),
                "rel_standing_envs": 0.02,
                "rel_heading_envs": 1.0,
                "heading_control_stiffness": 0.5,
            },
        ),
    },
    observations={
        "policy": mdp2.ObservationGroup(
            base_lin_vel=mdp2.Observation(mdp.base_lin_vel, inputs=root_state, noise=mdp2.UniformNoise(-0.1, 0.1)),
            base_ang_vel=mdp2.Observation(mdp.base_ang_vel, inputs=root_state, noise=mdp2.UniformNoise(-0.2, 0.2)),
            projected_gravity=mdp2.Observation(
                mdp.projected_gravity,
                inputs={"root_link_pose_w": robot.root_link_pose_w, "gravity_w": scene.gravity_w},
                noise=mdp2.UniformNoise(-0.05, 0.05),
            ),
            velocity_commands=mdp2.Observation(mdp.generated_commands, inputs={"command": mdp2.commands.base_velocity}),
            joint_pos=mdp2.Observation(
                mdp.joint_pos_rel,
                inputs={
                    "joint_pos": robot.joint_pos,
                    "default_joint_pos": robot.default_joint_pos,
                    "joint_ids": joints,
                },
                noise=mdp2.UniformNoise(-0.01, 0.01),
            ),
            joint_vel=mdp2.Observation(
                mdp.joint_vel_rel,
                inputs={
                    "joint_vel": robot.joint_vel,
                    "default_joint_vel": robot.default_joint_vel,
                    "joint_ids": joints,
                },
                noise=mdp2.UniformNoise(-1.5, 1.5),
            ),
            actions=mdp2.Observation(mdp.last_action, inputs={"actions": mdp2.last_action}),
        )
    },
    rewards={
        "track_lin_vel_xy_exp": mdp2.Reward(
            mdp.track_lin_vel_xy_exp, inputs=velocity_tracking, params={"std": math.sqrt(0.25)}, weight=1.5
        ),
        "track_ang_vel_z_exp": mdp2.Reward(
            mdp.track_ang_vel_z_exp, inputs=velocity_tracking, params={"std": math.sqrt(0.25)}, weight=0.75
        ),
        "lin_vel_z_l2": mdp2.Reward(mdp.lin_vel_z_l2, inputs=root_state, weight=-2.0),
        "ang_vel_xy_l2": mdp2.Reward(mdp.ang_vel_xy_l2, inputs=root_state, weight=-0.05),
        "dof_torques_l2": mdp2.Reward(
            mdp.joint_torques_l2, inputs={"applied_effort": robot.applied_effort, "joint_ids": joints}, weight=-2.0e-4
        ),
        "dof_acc_l2": mdp2.Reward(
            mdp.joint_acc_l2, inputs={"joint_acc": robot.joint_acc, "joint_ids": joints}, weight=-2.5e-7
        ),
        "action_rate_l2": mdp2.Reward(mdp.action_rate_l2, inputs={"actions": mdp2.last_action}, weight=-0.005),
        "feet_air_time": mdp2.Reward(
            mdp.feet_air_time,
            inputs={
                "current_contact_time": contact_forces.current_contact_time,
                "last_air_time": contact_forces.last_air_time,
                "command": mdp2.commands.base_velocity,
                "body_ids": feet,
            },
            params={"threshold": 0.5, "step_dt": mdp2.step_dt},
            weight=0.25,
        ),
        "flat_orientation_l2": mdp2.Reward(
            mdp.flat_orientation_l2,
            inputs={"root_link_pose_w": robot.root_link_pose_w, "gravity_w": scene.gravity_w},
            weight=-2.5,
        ),
        # Go2 has colliders on its legs, so without this term it can crouch on them, its base level.
        "base_height_l2": mdp2.Reward(
            mdp.base_height_l2,
            inputs={"root_link_pose_w": robot.root_link_pose_w},
            params={"target_height": 0.4},
            weight=-30.0,
        ),
    },
    terminations={
        "time_out": mdp2.TimeLimit(episode_length_s=20.0),
        "base_contact": mdp2.Termination(
            mdp.illegal_contact,
            inputs={
                "net_normal_forces_w_history": contact_forces.net_normal_forces_w_history,
                "body_ids": base_contact,
            },
            params={"threshold": 1.0},
        ),
    },
    events={
        # Newton has one friction coefficient, so MDP1's dynamic friction of 0.6 does not apply.
        "physics_material": mdp2.Event(
            mdp.randomize_rigid_body_material,
            mode=mdp2.EventMode.STARTUP,
            inputs={"body_ids": bodies},
            params={"friction_range": (0.8, 0.8), "restitution_range": (0.0, 0.0)},
        ),
        "add_base_mass": mdp2.Event(
            mdp.randomize_rigid_body_mass,
            mode=mdp2.EventMode.STARTUP,
            inputs={"body_ids": base},
            params={
                "mass_distribution_params": (1 / 1.25, 1.25),
                "operation": "scale",
                "distribution": "log_uniform",
            },
        ),
        "base_com": mdp2.Event(
            mdp.randomize_rigid_body_com,
            mode=mdp2.EventMode.STARTUP,
            inputs={"body_ids": base},
            params={"com_range": ((-0.05, 0.05), (-0.05, 0.05), (-0.01, 0.01))},
        ),
        "reset_base": mdp2.Event(
            mdp.reset_root_state_uniform,
            inputs={
                "default_root_pose": robot.default_root_pose,
                "default_root_vel": robot.default_root_vel,
                "env_origins": scene.env_origins,
            },
            outputs=root_state,
            params={
                "pose_range": ((-0.5, 0.5), (-0.5, 0.5), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (-3.14, 3.14)),
                "velocity_range": ((-0.5, 0.5),) * 6,
            },
        ),
        "reset_robot_joints": mdp2.Event(
            mdp.reset_joints_by_scale,
            inputs={
                "default_joint_pos": robot.default_joint_pos,
                "default_joint_vel": robot.default_joint_vel,
                "soft_joint_pos_limits": robot.soft_joint_pos_limits,
                "soft_joint_vel_limits": robot.soft_joint_vel_limits,
                "joint_ids": joints,
            },
            outputs={"joint_pos": robot.joint_pos, "joint_vel": robot.joint_vel},
            params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
        ),
        "push_robot": mdp2.Event(
            mdp.push_by_setting_velocity,
            outputs={"root_com_vel_w": robot.root_com_vel_w},
            mode=mdp2.EventMode.INTERVAL,
            interval_range_s=(10.0, 15.0),
            params={"velocity_range": ((-0.5, 0.5), (-0.5, 0.5), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))},
        ),
    },
)
"""The MDP of the task."""


@configclass
class UnitreeGo2FlatEnvCfg(MDP2RLEnvCfg):
    """Configuration of the Unitree Go2 flat velocity environment."""

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=SIM_DT,
        render_interval=DECIMATION,
        physics=UnitreeGo2FlatPhysicsCfg(),
        physics_material=GROUND_MATERIAL,
        use_newton_actuators=True,
    )
    scene: UnitreeGo2FlatSceneCfg = UnitreeGo2FlatSceneCfg(num_envs=4096, env_spacing=2.5)
    task: mdp2.TaskCfg = TASK
    decimation: int = DECIMATION
