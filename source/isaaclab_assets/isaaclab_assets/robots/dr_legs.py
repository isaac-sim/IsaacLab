# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Disney DR Legs closed-loop biped.

DR Legs is a parallel-linkage bipedal lower body with 30 joints (12 actuated + 18 passive
linkage DOFs) plus 6 loop-closing joints. The Kamino solver enforces the loop-closing joints
as bilateral constraints in maximal coordinates.

The following configuration is available:

* :data:`DR_LEGS_IMPLICIT_PD_CFG`: DR Legs with implicit (solver-side) PD on the
  12 actuated joints and zero-PD on the 18 passive linkage joints.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import NEWTON_ASSET_DIR, retrieve_git_asset_path

_DR_LEGS_USD_PATH = retrieve_git_asset_path(NEWTON_ASSET_DIR, "disneyresearch/dr_legs/usd/dr_legs.usda")

DR_LEGS_JOINT_ORDER: list[str] = [
    "j1_l_i",
    "j2_l_i",
    "j3_l_i",
    "j4_l_i",
    "j6_l_i",
    "j7_l_i",
    "j8_l_i",
    "j1_l_o",
    "j2_l_o",
    "j3_l_o",
    "j4_l_o",
    "j5_l_o",
    "j6_l_o",
    "j7_l_o",
    "j8_l_o",
    "j1_r_i",
    "j2_r_i",
    "j3_r_i",
    "j4_r_i",
    "j6_r_i",
    "j7_r_i",
    "j8_r_i",
    "j1_r_o",
    "j2_r_o",
    "j3_r_o",
    "j4_r_o",
    "j5_r_o",
    "j6_r_o",
    "j7_r_o",
    "j8_r_o",
]
"""Canonical ordering of the 30 (6 loop-closure joints are excluded) DR Legs joints."""

DR_LEGS_ACTUATED_JOINTS: list[str] = [
    "j1_l_i",
    "j2_l_i",
    "j6_l_i",
    "j7_l_i",
    "j2_l_o",
    "j7_l_o",
    "j1_r_i",
    "j2_r_i",
    "j6_r_i",
    "j7_r_i",
    "j2_r_o",
    "j7_r_o",
]
"""The 12 servo-driven joints (6 per leg)."""

DR_LEGS_PASSIVE_JOINTS: list[str] = [j for j in DR_LEGS_JOINT_ORDER if j not in DR_LEGS_ACTUATED_JOINTS]
"""The 18 closed-loop linkage DOFs that are not driven by an actuator on real hardware."""


DR_LEGS_IMPLICIT_PD_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_DR_LEGS_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.NewtonArticulationRootPropertiesCfg(self_collision_enabled=True),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.27),
        # Closed-loop FK is only valid at the assembled reference (all joint coords zero).
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    articulation_root_prim_path="/pelvis",
    actuators={
        "driven_joints": ImplicitActuatorCfg(
            joint_names_expr=DR_LEGS_ACTUATED_JOINTS,
            stiffness=5.0,
            damping=0.2,
            effort_limit_sim=3.1,
            velocity_limit_sim=100.0,
        ),
        # Linkage DOFs are undriven: explicit zeros so the solver ignores USD drive defaults.
        "passive_joints": ImplicitActuatorCfg(
            joint_names_expr=DR_LEGS_PASSIVE_JOINTS,
            stiffness=0.0,
            damping=0.0,
            armature=0.0,
            friction=0.0,
            effort_limit_sim=400.0,
            velocity_limit_sim=100.0,
        ),
    },
)
"""DR Legs with implicit PD on the 12 actuated joints (Kamino solver, closed-loop)."""
