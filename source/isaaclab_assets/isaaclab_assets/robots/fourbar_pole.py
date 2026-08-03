# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a parallel four-bar linkage with an inverted pendulum pole on the coupler."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import NEWTON_ASSET_DIR, retrieve_git_asset_path

_FOURBAR_POLE_USD = retrieve_git_asset_path(NEWTON_ASSET_DIR, "fourbar_pole/usd/fourbar_pole.usda")

##
# Configuration
##

FOURBAR_POLE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_FOURBAR_POLE_USD,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.5),
        joint_pos={
            "ground_to_crank": 0.0,
            "crank_to_coupler": 0.0,
            "coupler_to_rocker": 0.0,
            "coupler_to_pole": 0.0,
        },
    ),
    actuators={
        "fourbar_actuator": ImplicitActuatorCfg(
            joint_names_expr=["ground_to_crank"],
            effort_limit_sim=400.0,
            stiffness=0.0,
            damping=10.0,
        ),
        # Add an actuator to the pole even though it is not actuated by the user.
        # This makes the Kamino forward kinematics solver treat the pole as
        # an actuated joint and preserve its user-specified joint position on resets.
        "pole_actuator": ImplicitActuatorCfg(
            joint_names_expr=["coupler_to_pole"],
            effort_limit_sim=400.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
"""Configuration for a parallel four-bar linkage with a pole on the coupler midpoint."""
