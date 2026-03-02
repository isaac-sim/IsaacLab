# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from dataclasses import MISSING
import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns, ImuCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg, EventCfg, TerminationsCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
##
# Pre-defined configs
##
from isaaclab_assets import G1_CFG, G1_MINIMAL_CFG, UNITREE_G1_23DOF_CFG, UNITREE_G1_29DOF_CFG
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG # isort: skip
# import os
from pathlib import Path

##### Board #######

#####################################################
########### board for ArticulationCfg ###############
#####################################################


script_dir = Path(__file__).parent
usd_path = str(script_dir / "separeted_model/board/board_articulation.usdz")
BOARD_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/object_balanceboard",
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_path,
        scale=(1.0, 1.0, 1.0),

        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
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

        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0025,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.365, 0.136),
        rot=(0.7071, 0.7071, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={},
)

#####################################################
########### board for RigidCfg ######################
#####################################################
# script_dir = Path(__file__).parent
# usd_path = str(script_dir / "separeted_model/board/board_rigid.usda")
# BOARD_CFG = RigidObjectCfg(
#     prim_path="{ENV_REGEX_NS}/object_balanceboard",
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=usd_path,
#         activate_contact_sensors=True,
#         scale=(1.0, 1.0, 1.0),
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             rigid_body_enabled=True,
#             kinematic_enabled=False,
#             disable_gravity=False,
#             enable_gyroscopic_forces=True,
#             solver_position_iteration_count=8,
#             solver_velocity_iteration_count=0,
#             sleep_threshold=0.005,
#             stabilization_threshold=0.0025,
#             max_depenetration_velocity=1000.0,
#         ),
#         mass_props=sim_utils.MassPropertiesCfg(density=400.0),
#     ),
#     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.405, 0.136), rot=(0.7071, 0.7071, 0.0, 0.0)),
# )


##### Plane #######
PLANE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/static_plane",
    spawn=sim_utils.CuboidCfg(
        size = [2.5,2.5,0.01],
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True,),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.005), rot=(1.0, 0.0, 0.0, 0.0)),
)

###### Roller ######
script_dir = Path(__file__).parent
usd_path = str(script_dir / "separeted_model/roller/roller.usda")
ROLLER_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/object_roller",
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_path,
        # activate_contact_sensors=True,
        scale=(1.0, 1.0, 1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
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
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.2, 0.0, 0.075), rot=(math.sqrt(2) / 2, 0.0, math.sqrt(2) / 2, 0.0)),
)
@configclass
class G1SceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",  # "plane", "generator"
        env_spacing=2.5,
        debug_vis=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
    )

    # robots
    robot: ArticulationCfg = MISSING
    board: ArticulationCfg = BOARD_CFG
    static_plane: RigidObjectCfg = PLANE_CFG
    cylinder: RigidObjectCfg = ROLLER_CFG

    # sensor
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=6, track_air_time=True, debug_vis=False,
                                    # filter_prim_paths_expr=["{ENV_REGEX_NS}/static_plane"]
                                    # filter_prim_paths_expr=["/World/ground/terrain/mesh"]
                                      )
    contact_forces_LF  = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link", history_length=6, track_air_time=True, debug_vis=False,
                                    filter_prim_paths_expr=["{ENV_REGEX_NS}/static_plane"]
                                      )
    contact_forces_RF  = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link", history_length=6, track_air_time=True, debug_vis=False,
                                    filter_prim_paths_expr=["{ENV_REGEX_NS}/static_plane"]
                                      )

    contact_forces_B = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/object_balanceboard/ROOT", history_length=64,
                                         track_air_time=True, debug_vis=False,
                                         filter_prim_paths_expr=["{ENV_REGEX_NS}/static_plane"]
                                         )
    
    imu_acc = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/torso_link", gravity_bias=(0, 0, 0), debug_vis=True)

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

@configclass
class G1Terminations(TerminationsCfg):
    """Termination terms for the MDP."""
    # height_detection = DoneTerm(
    #     func=mdp.terminate_on_z_height,
    #     params={"asset_cfg": SceneEntityCfg("board", body_names="ROOT/geometry_0_005"), "height_threshold": 0.5},
    # )
    
    plane_contact_detection_left  = DoneTerm(
        func=mdp.ground_contact_termination,
        params={"sensor_cfg": SceneEntityCfg("contact_forces_LF", body_names="left_ankle_roll_link"), "threshold": 0.0},
         )
    plane_contact_detection_right  = DoneTerm(
        func=mdp.ground_contact_termination,
        params={"sensor_cfg": SceneEntityCfg("contact_forces_RF", body_names="right_ankle_roll_link"), "threshold": 0.0},
         )
    plane_contact_detection_board  = DoneTerm(
        func=mdp.ground_contact_termination,
        params={"sensor_cfg": SceneEntityCfg("contact_forces_B", body_names="ROOT"), "threshold": 0.0},
         )



@configclass
class G1Rewards:
    ## First Stage
    # alive and termination
    alive_reward = RewTerm(func=mdp.is_alive, weight=1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)
    # torso_orientation_penalty
    torso_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-10.0,
        params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                }
    )
    # torso_height_penalty
    base_height_penalty = RewTerm(
        func=mdp.base_height_l2,
        weight=-100.0,
        params={
            "target_height": 0.93,
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
        }
    )

    ## Second Stage
    # board_balance
    platform_stabilization_penalty = RewTerm(
        func=mdp.reward_platform_stabilization,
        weight=1000.0,
        params={"std": 0.5,
                "asset_cfg": SceneEntityCfg("board"),
                }
    )

    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.10)

    com_acceleration_penalty = RewTerm(
        func=mdp.reward_com_acceleration,
        weight=10,
        params={"std": 0.5,
                "sensor_cfg": SceneEntityCfg("imu_acc")}
    )

    ## Third Stage
    joint_torques_penalty = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

    joint_velocities_penalty = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

    action_rate_penalty = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01
    )

    default_pose_reward = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,

    )

@configclass
class G1ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.0, n_max=0.0))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.0, n_max=0.0))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.00, n_max=0.00),
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.0, n_max=0.0))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.0, n_max=0.0))
        actions = ObsTerm(func=mdp.last_action)
        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(-1.0, 1.0),
        # )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class G1CommandsCfg:
    """Command specifications for the MDP."""

    null = mdp.NullCommandCfg()

@configclass
class G1BalanceEnvCfg(LocomotionVelocityRoughEnvCfg):
    ## scene settings
    scene: G1SceneCfg = G1SceneCfg(num_envs = 4096, env_spacing = 2.5)
    ## basic settings
    observations: G1ObservationsCfg = G1ObservationsCfg()
    rewards: G1Rewards = G1Rewards()
    ## MDP settings
    terminations: G1Terminations = G1Terminations()
    commands: G1CommandsCfg = G1CommandsCfg()


    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Scene
        self.scene.robot = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos=(0.0, 0.0, 0.945)
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        self.scene.imu_acc.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        self.scene.height_scanner = None

        # Randomization
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.push_robot = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"
        

