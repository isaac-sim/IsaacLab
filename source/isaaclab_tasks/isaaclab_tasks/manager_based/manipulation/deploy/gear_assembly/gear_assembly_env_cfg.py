# Copyright (c) 2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
import os

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.simulation_cfg import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import ResetSampledNoiseModelCfg, UniformNoiseCfg
import isaaclab_tasks.manager_based.manipulation.deploy.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.deploy.reach.reach_env_cfg import ReachEnvCfg

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Get the directory where this configuration file is located
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(CONFIG_DIR, "assets")

##
# Environment configuration
##

@configclass
class GearAssemblySceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""
    
    # Disable scene replication to allow USD-level randomization
    replicate_physics = False

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    factory_gear_base = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/FactoryGearBase",
        # TODO: change to common isaac sim directory
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"omniverse://isaac-dev.ov.nvidia.com/Projects/isaac_ros_gear_insertion/Factory/Gears_1.5x/factory_gear_base.usd",
            # usd_path=os.path.join(ASSETS_DIR, "Factory/Gears_1.5x/factory_gear_base.usd"),
            # usd_path=f"omniverse://isaac-dev.ov.nvidia.com/Users/ashwinvk@nvidia.com/props/factory_gear_base.usd",
            # usd_path=f"omniverse://isaac-dev.ov.nvidia.com/Projects/isaacsim/Props/gear_assembly/gear_base_scale_1.5_usd/gear_base_scale_1.5.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                kinematic_enabled=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=196,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=None),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            # usd_path=f"omniverse://isaac-dev.ov.nvidia.com/Projects/isaacsim/Props/gear_assembly/gear_base_scale_1.5_usd/gear_base_scale_1.5.usd",
            # rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0200, -0.2100, -0.1), rot=(-0.70711, 0.0, 0.0, 0.70711)),
    )

    # factory_gear_small = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/FactoryGearSmall",
    #     # TODO: change to common isaac sim directory
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"omniverse://isaac-dev.ov.nvidia.com/Projects/isaac_ros_gear_insertion/Factory/Gears_1.5x/factory_gear_small.usd",
    #         # usd_path=os.path.join(ASSETS_DIR, "Factory/Gears_1.5x/factory_gear_small.usd"),
    #         # usd_path=f"omniverse://isaac-dev.ov.nvidia.com/Users/ashwinvk@nvidia.com/props/factory_gear_small.usd",
    #         # usd_path=f"omniverse://isaac-dev.ov.nvidia.com/Projects/isaacsim/Props/gear_assembly/small_gear_scale_1p5_usd/small_gear_scale_1p5.usd",
    #         activate_contact_sensors=True,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             disable_gravity=False,
    #             kinematic_enabled=False,
    #             max_depenetration_velocity=5.0,
    #             linear_damping=0.0,
    #             angular_damping=0.0,
    #             max_linear_velocity=1000.0,
    #             max_angular_velocity=3666.0,
    #             enable_gyroscopic_forces=True,
    #             solver_position_iteration_count=196,
    #             solver_velocity_iteration_count=1,
    #             max_contact_impulse=1e32,
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
    #         collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0200, -0.2100, -0.1), rot=(-0.70711, 0.0, 0.0, 0.70711)),
    # )

    factory_gear_medium = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/FactoryGearMedium",
        # TODO: change to common isaac sim directory
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"omniverse://isaac-dev.ov.nvidia.com/Projects/isaac_ros_gear_insertion/Factory/Gears_1.5x/factory_gear_medium.usd",
            # usd_path=os.path.join(ASSETS_DIR, "Factory/Gears_1.5x/factory_gear_medium.usd"),
            # usd_path=f"omniverse://isaac-dev.ov.nvidia.com/Users/ashwinvk@nvidia.com/props/factory_gear_medium.usd",
            # usd_path=f"omniverse://isaac-dev.ov.nvidia.com/Projects/isaacsim/Props/gear_assembly/medium_gear_scale_1p5_usd/medium_gear_scale_1p5.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                kinematic_enabled=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=196,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0200, -0.2100, -0.1), rot=(-0.70711, 0.0, 0.0, 0.70711)),
    )

    # factory_gear_large = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/FactoryGearLarge",
    #     # TODO: change to common isaac sim directory
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"omniverse://isaac-dev.ov.nvidia.com/Projects/isaac_ros_gear_insertion/Factory/Gears_1.5x/factory_gear_large.usd",
    #         # usd_path=f"omniverse://isaac-dev.ov.nvidia.com/Users/ashwinvk@nvidia.com/props/factory_gear_large.usd",
    #         # usd_path=f"omniverse://isaac-dev.ov.nvidia.com/Projects/isaacsim/Props/gear_assembly/large_gear_scale_1p5_usd/large_gear_scale_1p5.usd",
    #         activate_contact_sensors=True,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             disable_gravity=False,
    #             kinematic_enabled=False,
    #             max_depenetration_velocity=5.0,
    #             linear_damping=0.0,
    #             angular_damping=0.0,
    #             max_linear_velocity=1000.0,
    #             max_angular_velocity=3666.0,
    #             enable_gyroscopic_forces=True,
    #             solver_position_iteration_count=196,
    #             solver_velocity_iteration_count=1,
    #             max_contact_impulse=1e32,
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
    #         collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0200, -0.2100, -0.1), rot=(-0.70711, 0.0, 0.0, 0.70711)),
    # )
    

    # robots
    robot: ArticulationCfg = MISSING


    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = MISSING
    gripper_action: ActionTerm | None = None


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos,
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
        joint_vel = ObsTerm(func=mdp.joint_vel,
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
        gear_shaft_pos = ObsTerm(func=mdp.gear_shaft_pos_w, noise=ResetSampledNoiseModelCfg(
            noise_cfg=UniformNoiseCfg(n_min=-0.005, n_max=0.005, operation="add")
        ))
        gear_shaft_quat = ObsTerm(func=mdp.gear_shaft_quat_w)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    # @configclass
    # class CriticCfg(ObsGroup):
    #     """Observations for policy group."""

    #     # observation terms (order preserved)
    #     joint_pos = ObsTerm(func=mdp.joint_pos,
    #                         params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
    #     joint_vel = ObsTerm(func=mdp.joint_vel,
    #                         params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
    #     # gear_shaft_pos = ObsTerm(func=mdp.gear_shaft_pos_w,
    #     #                          noise=AdditiveUniformNoiseModelCfg(n_min=-0.0025, n_max=0.0025))
    #     # gear_shaft_pos = ObsTerm(func=mdp.gear_shaft_pos_w, noise=Unoise(n_min=-0.005, n_max=0.005, operation="add"))
    #     # gear_shaft_pos = ObsTerm(func=mdp.gear_shaft_pos_w)
    #     gear_shaft_pos = ObsTerm(func=mdp.gear_shaft_pos_w)
    #     gear_shaft_quat = ObsTerm(func=mdp.gear_shaft_quat_w)

    #     gear_pos = ObsTerm(func=mdp.gear_pos_w)
    #     gear_quat = ObsTerm(func=mdp.gear_quat_w)

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    # critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_gear = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.05, 0.05],
                "y": [-0.05, 0.05],
                "z": [0.1, 0.15],
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("factory_gear_small"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    end_effector_gear_keypoint_tracking = RewTerm(
        func=mdp.keypoint_entity_error,
        weight=-1.5,
        params={
            "asset_cfg_1": SceneEntityCfg("factory_gear_base"), 
            "keypoint_scale": 0.15,
        },
    )

    end_effector_gear_keypoint_tracking_exp = RewTerm(
        func=mdp.keypoint_entity_error_exp,
        weight=1.5,
        params={
            "asset_cfg_1": SceneEntityCfg("factory_gear_base"), 
            "kp_exp_coeffs": [(50, 0.0001), (300, 0.0001)],
            "kp_use_sum_of_exps": False,
            "keypoint_scale": 0.15,
        },
    )

    action_rate = RewTerm(func=mdp.action_rate, weight=-5.0e-06)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

@configclass
class GearAssemblyEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: GearAssemblySceneCfg = GearAssemblySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    sim: SimulationCfg = SimulationCfg(

        physx=PhysxCfg(
            gpu_collision_stack_size=2**31,  # Important to prevent collisionStackSize buffer overflow in contact-rich environments.
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23
        ),
    )
    
    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.episode_length_s = 6.66
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.sim.dt = 1.0 / 120.0

        self.hand_close_pos = 0.7

        self.gear_offsets = {'gear_small': [0.076125, 0.0, 0.0],
                            'gear_medium': [0.030375, 0.0, 0.0],
                            'gear_large': [-0.045375, 0.0, 0.0]}
        
        self.gear_offsets_grasp = {'gear_small': [0.0, 0.076125, -0.26],
                            'gear_medium': [0.0, 0.030375, -0.26],
                            'gear_large': [0.0, -0.045375, -0.26]}

        self.hand_grasp_pos = {"gear_small": 0.67,
                               "gear_medium": 0.54,
                               "gear_large": 0.52}
