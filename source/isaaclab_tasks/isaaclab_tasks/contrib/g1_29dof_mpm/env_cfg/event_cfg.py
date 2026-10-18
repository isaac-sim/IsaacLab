# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Events for the G1 29-DoF granular locomotion task.

Robot-side randomization is kept as in the soft-contact task. Terrain-parameter randomization is
dropped: the granular properties live in the MPM material, and randomizing them per reset would
require rebuilding the particle material. The bed itself is restored on reset instead.
"""

from isaaclab_newton.envs.mdp import randomize_mpm_material

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.velocity.mdp as mdp


@configclass
class G1EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.2, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,  # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (0.0, 5.0),
            "operation": "add",
        },
    )

    """
    reset
    """

    # restore the flat bed before the robot is placed on it

    # The robot starts on the rigid approach platform, so the yaw spread is narrow enough that a
    # forward command carries it onto the bed rather than off the side of the platform.
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.6, 0.6)},
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
        func=mdp.randomize_actuator_gains,  # type: ignore
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "operation": "scale",
            "distribution": "uniform",
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
        },
    )

    """
    interval
    """

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )

    # Sample each material property once per environment at reset.
    mpm_material = EventTerm(
        func=randomize_mpm_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("sand"),
            "parameter_ranges": {"friction": (0.3, 0.9), "density": (1000.0, 3000.0), "poisson_ratio": (0.3, 0.3)},
            "distribution": "uniform",
        },
    )
    mpm_material_log = EventTerm(
        func=randomize_mpm_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("sand"),
            "parameter_ranges": {"young_modulus": (1.0e7, 1.0e9), "yield_pressure": (1.0e8, 1.0e9)},
            "distribution": "log_uniform",
        },
    )
