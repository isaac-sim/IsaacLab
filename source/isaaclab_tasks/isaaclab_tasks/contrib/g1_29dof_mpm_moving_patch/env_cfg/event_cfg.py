# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Events for the G1 29-DoF granular locomotion task.

Robot-side randomization follows the soft-contact task. MPM material ranges can be
configured through the reset-time ``mpm_material`` term.
"""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.velocity.mdp as mdp

from .. import mdp as mpm_mdp


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

    # Randomize the reset pose around the configured sand-surface spawn position.
    reset_base = EventTerm(
        func=mpm_mdp.reset_root_state_on_terrain,  # type: ignore
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
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

    mpm_material = EventTerm(
        func=mpm_mdp.randomize_mpm_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("sand"),
            "parameter_ranges": {
                "friction": (0.3, 0.9),
                "density": (1000.0, 3000.0),
                "poisson_ratio": (0.3, 0.3),
                # "hardening": (0.0, 0.0),
                # "dilatancy": (0.0, 0.0),
                # "viscosity": (0.0, 0.0),
            },
            "distribution": "uniform",
        },
    )

    mpm_material_log = EventTerm(
        func=mpm_mdp.randomize_mpm_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("sand"),
            "parameter_ranges": {
                "young_modulus": (1.0e7, 1.0e9),
                "yield_pressure": (1.0e8, 1.0e9),
                # "tensile_yield_ratio": (1.0e-9, 1.0), # TODO: investigate instability in this parameters
                # "yield_stress": (1.0, 1.0e9), # TODO: ablate
            },
            "distribution": "log_uniform",
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
