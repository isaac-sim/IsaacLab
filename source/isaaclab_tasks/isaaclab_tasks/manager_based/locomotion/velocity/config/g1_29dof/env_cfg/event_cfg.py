# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
# import isaaclab_tasks.manager_based.locomotion.velocity.mdp as vel_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.config.g1_29dof.mdp as g1_mdp


@configclass
class G1EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.4, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass, 
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    # base_com = EventTerm(
    #     func=mdp.randomize_rigid_body_com, 
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
    #         "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
    #     },
    # )
    
    # scale_actuator_gains = EventTerm(
    #     func=mdp.randomize_actuator_gains, # type: ignore
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=".*"
    #         ),
    #         "operation": "scale",
    #         "distribution": "uniform",
    #         # "stiffness_distribution_params": (0.9, 1.1),
    #         # "damping_distribution_params": (0.9, 1.1),
    #         "stiffness_distribution_params": (0.75, 1.25),
    #         "damping_distribution_params": (0.75, 1.25),
    #     },
    # )

    """
    reset
    """
    # base_external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
    #         "force_range": (0.0, 0.0),
    #         "torque_range": (-0.0, 0.0),
    #     },
    # )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform, 
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale, 
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )
    
    scale_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains, # type: ignore
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=".*"
            ),
            "operation": "scale",
            "distribution": "uniform",
            # "stiffness_distribution_params": (0.9, 1.1),
            # "damping_distribution_params": (0.9, 1.1),
            "stiffness_distribution_params": (0.75, 1.25),
            "damping_distribution_params": (0.75, 1.25),
        },
    )

    """
    interval
    """
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )