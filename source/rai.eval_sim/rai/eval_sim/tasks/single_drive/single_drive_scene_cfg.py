# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from rai.eval_sim.utils import ASSETS_DIR

# Single Drive articulation configuration
SingleDriveArticulationCfg = ArticulationCfg(
    # USD file configuration
    spawn=sim_utils.UsdFileCfg(
        # Location of USD file
        usd_path=str(ASSETS_DIR / "single_drive" / "usd" / "single_drive_rect.usd"),
        # Rigid body properties
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        # Articulation root properties
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    # Initial state definition
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={"Motor_Joint": 0.0}, joint_vel={"Motor_Joint": 0.0}
    ),
    # Actuators definition
    actuators={
        "single_drive_actuator": ImplicitActuatorCfg(
            joint_names_expr=["Motor_Joint"], effort_limit=400.0, velocity_limit=100.0, stiffness=40.0, damping=10.0
        ),
    },
)


@configclass
class SingleDriveSceneCfg(InteractiveSceneCfg):
    """Configuration for a single drive scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    light_1 = AssetBaseCfg(
        prim_path="/World/Distance",
        spawn=sim_utils.DistantLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=3000.0,
        ),
    )
    light_2 = AssetBaseCfg(
        prim_path="/World/Dome",
        spawn=sim_utils.DomeLightCfg(
            color=(0.13, 0.13, 0.13),
            intensity=1000.0,
        ),
    )

    # articulation
    robot: ArticulationCfg = SingleDriveArticulationCfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # interactive scene configurations
    num_envs: int = 1024
    env_spacing: float = 1.5


@configclass
class SingleDriveSceneVerticalCfg(SingleDriveSceneCfg):
    """Configuration for a single drive scene oriented vertically, exposed to gravity."""

    # articulation
    robot: ArticulationCfg = SingleDriveArticulationCfg.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.65),
            rot=(0.5, 0.5, 0.5, 0.5),
            joint_pos={"Motor_Joint": 0.0},
            joint_vel={"Motor_Joint": 0.0},
        ),
    )
