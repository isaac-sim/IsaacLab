# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

"""
Sub-module containing configuration of Spot.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from rai.eval_sim.utils import ASSETS_DIR

from .actuators import DelayedPDActuatorCfg, RemotizedPDActuatorCfg

##
# Configuration
##

SPOT_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=False,
        make_instanceable=False,
        link_density=1.0e-8,
        asset_path=f"{ASSETS_DIR}/spot/urdf/spot.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            "[fh]l_hx": 0.1,  # all left hip_x
            "[fh]r_hx": -0.1,  # all right hip_x
            "f[rl]_hy": 0.9,  # front hip_y
            "h[rl]_hy": 1.1,  # hind hip_y
            ".*_kn": -1.5,  # all knees
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "spot_hip": DelayedPDActuatorCfg(
            joint_names_expr=[".*_h[xy]"],
            effort_limit=45.0,
            stiffness=60.0,
            damping=1.5,
            min_num_time_lags=0,  # physics time steps (max: 2.0*0=0.0ms)
            max_num_time_lags=4,  # physics time steps (max: 2.0*4=8.0ms)
        ),
        "spot_knee": RemotizedPDActuatorCfg(
            joint_names_expr=[".*_kn"],
            effort_limit=None,  # torque limits are handled based experimental data (:meth:`RemotizedPDActuatorCfg.data`)
            stiffness=60.0,
            damping=1.5,
            min_num_time_lags=0,  # physics time steps (max: 2.0*0=0.0ms)
            max_num_time_lags=4,  # physics time steps (max: 2.0*4=8.0ms)
        ),
    },
)
