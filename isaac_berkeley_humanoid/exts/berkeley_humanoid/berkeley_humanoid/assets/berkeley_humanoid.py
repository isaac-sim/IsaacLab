# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from berkeley_humanoid.actuators import IdentifiedActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from berkeley_humanoid.assets import ISAAC_ASSET_DIR

INWHEEL_UP_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=[".*upper_leg"],
    effort_limit=20.88,
    velocity_limit=31.0,
    saturation_effort=443, #갑자기 토크
    stiffness={".*": 1.5}, # 강성 : 관절이 외부 힘에 더 잘 저항
    damping={".*": 0.016},    # 감쇠 : 진동과 불안정한 움직임을 줄임임
    armature={".*": 6.9e-5 * 81},
    friction_static=0.3,
    activation_vel=0.2,
    friction_dynamic=0.2,
)


INWHEEL_WHEEL_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=[".*wheel"],
    effort_limit=2.13,
    velocity_limit=39.58,
    saturation_effort=443,
    stiffness={".*": 1.0},
    damping={".*": 0.1},
    armature={".*": 6.9e-5 * 81},
    friction_static=0.3,
    activation_vel=0.05,
    friction_dynamic=0.2,
)


BERKELEY_HUMANOID_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=r"C:\Users\Hansol\IsaacLab\isaac_berkeley_humanoid\exts\berkeley_humanoid\berkeley_humanoid\assets\Robots\robot.usd",
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
        pos=(0.0, 0.0, 0.40),
        joint_pos={
            'upper_leg': 0.0,
            'lower_leg': 0.0,
        },
    ),
    actuators={"upper_leg": INWHEEL_UP_ACTUATOR_CFG, "wheel": INWHEEL_WHEEL_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.75,
)