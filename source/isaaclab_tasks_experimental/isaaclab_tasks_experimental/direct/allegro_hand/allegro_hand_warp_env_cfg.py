# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg  # , RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim._impl.newton_manager_cfg import NewtonCfg
from isaaclab.sim._impl.solvers_cfg import MJWarpSolverCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


@configclass
class AllegroHandWarpEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    action_space = 16
    observation_space = 124  # (full)
    state_space = 0
    asymmetric_obs = False
    obs_type = "full"

    solver_cfg = MJWarpSolverCfg(
        solver="newton",
        integrator="implicit",
        njmax=70,
        nconmax=70,
        impratio=10.0,
        cone="elliptic",
        update_data_interval=2,
        iterations=100,
        ls_iterations=15,
        ls_parallel=True,
        # save_to_mjcf="AllegroHand.xml",
    )

    newton_cfg = NewtonCfg(
        solver_cfg=solver_cfg,
        num_substeps=2,
        debug_mode=False,
    )
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        newton_cfg=newton_cfg,
    )
    # robot
    robot_cfg: ArticulationCfg = ALLEGRO_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    actuated_joint_names = [
        "index_joint_0",
        "middle_joint_0",
        "ring_joint_0",
        "thumb_joint_0",
        "index_joint_1",
        "index_joint_2",
        "index_joint_3",
        "middle_joint_1",
        "middle_joint_2",
        "middle_joint_3",
        "ring_joint_1",
        "ring_joint_2",
        "ring_joint_3",
        "thumb_joint_1",
        "thumb_joint_2",
        "thumb_joint_3",
    ]
    fingertip_body_names = [
        "index_link_3",
        "middle_link_3",
        "ring_link_3",
        "thumb_link_3",
    ]

    # in-hand object
    object_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
            scale=(1.2, 1.2, 1.2),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, -0.17, 0.56), rot=(0.0, 0.0, 0.0, 1.0)),
        actuators={},
        articulation_root_prim_path="",
    )
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(1.2, 1.2, 1.2),
            )
        },
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192, env_spacing=0.75, replicate_physics=True, clone_in_fabric=True
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
    reach_goal_bonus = 250
    fall_penalty = 0
    fall_dist = 0.24
    vel_obs_scale = 0.2
    success_tolerance = 0.2
    max_consecutive_success = 0
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0
