# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_ovphysx.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.reorient.reorient_task_base import (
    ALLEGRO_ACTUATED_JOINT_NAMES,
    ALLEGRO_FINGERTIP_BODY_NAMES,
)
from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG


@configclass
class ObjectCfg(PresetCfg):
    physx = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
            scale=(1.2, 1.2, 1.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.17, 0.56), rot=(0.0, 0.0, 0.0, 1.0)),
    )
    newton_mjwarp = ArticulationCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
            scale=(1.2, 1.2, 1.2),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, -0.17, 0.565), rot=(0.0, 0.0, 0.0, 1.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
        articulation_root_prim_path="",
    )
    ovphysx = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
            scale=(1.2, 1.2, 1.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.17, 0.56), rot=(0.0, 0.0, 0.0, 1.0)),
    )
    default = physx


@configclass
class PhysicsCfg(PresetCfg):
    physx = PhysxCfg(
        bounce_threshold_velocity=0.2,
    )
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=80,
            nconmax=70,
            impratio=10.0,
            cone="elliptic",
            update_data_interval=2,
            iterations=100,
            # save_to_mjcf="AllegroHand.xml",
        ),
        num_substeps=2,
        debug_mode=False,
    )
    ovphysx = OvPhysxCfg()
    default = physx


# Scene pieces shared verbatim by the manager-based variant.
ROBOT_CFG = ALLEGRO_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot")
OBJECT_CFG = ObjectCfg()
GOAL_OBJECT_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/goal_marker",
    markers={
        "goal": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(1.2, 1.2, 1.2),
        )
    },
)


# Simulation settings shared by the Direct and manager variants (configclass
# deep-copies these defaults per cfg instance). The solver-common base material
# is sufficient: only friction values are set, so no PhysX-specific
# ``physxMaterial`` attributes are authored.
@configclass
class AllegroHandTaskCfgBase:
    """Shared Allegro task settings inherited by both the Direct and the manager configurations."""

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=4,
        physics_material=RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0),
        physics=PhysicsCfg(),
    )


@configclass
class AllegroHandEnvCfg(AllegroHandTaskCfgBase, DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 10.0
    action_space = 16
    observation_space = 124  # (full)
    state_space = 0
    asymmetric_obs = False
    obs_type = "full"
    # robot
    robot_cfg: ArticulationCfg = ROBOT_CFG

    actuated_joint_names = ALLEGRO_ACTUATED_JOINT_NAMES
    fingertip_body_names = ALLEGRO_FINGERTIP_BODY_NAMES

    # in-hand object
    object_cfg: ObjectCfg = OBJECT_CFG
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = GOAL_OBJECT_CFG
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192,
        env_spacing=0.75,
        replicate_physics=True,
        clone_in_fabric=True,
    )
    # reset
    reset_position_noise = 0.01  # range of position at reset
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset
    # reward scales
    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.0002
    reach_goal_bonus = 250.0
    fall_penalty = 0.0
    fall_dist = 0.24
    vel_obs_scale = 0.2
    success_tolerance = 0.2
    max_consecutive_success = 0
    success_count_threshold: int = 1
    """Minimum number of goals reached in an episode to count it as a successful episode."""
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0
