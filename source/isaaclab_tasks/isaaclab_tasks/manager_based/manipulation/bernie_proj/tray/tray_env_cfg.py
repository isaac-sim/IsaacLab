# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
# from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp

from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

##
# Scene definition
##


@configclass
class TraySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a humanoid robot."""

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        debug_vis=False,
    )

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[1, 0.0, 0.75], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path="/home/davin123/IsaacLab/assets/bernie_proj/table.usd"),
    )

    # Tray object in Scene
    tray = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/tray",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[1, 0, 1], rot=[0.70710678, 0.0, 0.0, 0.70710678]),
        spawn=UsdFileCfg(
            usd_path="/home/davin123/IsaacLab/assets/bernie_proj/tray.usd",
            scale=(0.01, 0.01, 0.05),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.56, 0.93, 0.56))
        ),
    )

    # RBY1 robot
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/davin123/IsaacLab/assets/RBY1.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=5.0,
                disable_gravity=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
            copy_from_source=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "left_arm_0": 0.0,
                "left_arm_1": 0.0,
                "left_arm_2": 0.0,
                "left_arm_3": 0.0,
                "left_arm_4": 0.0,
                "left_arm_5": 0.0,
                "left_arm_6": 0.0,
                "right_arm_0": 0.0,
                "right_arm_1": 0.0,
                "right_arm_2": 0.0,
                "right_arm_3": 0.0,
                "right_arm_4": 0.0,
                "right_arm_5": 0.0,
                "right_arm_6": 0.0,
                "gripper_finger_l1": 0.0,
                "gripper_finger_l2": 0.0,
                "gripper_finger_r1": 0.0,
                "gripper_finger_r2": 0.0,
                "torso_0": 0.0,
                "torso_1": 0.0,
                "torso_2": 0.0,
                "torso_3": 0.0,
                "torso_4": 0.0,
                "torso_5": 0.0,
                "head_0": 0.0,
                "head_1": 0.0,
            },
        ),
        actuators={
            "rby1_leftarm": ImplicitActuatorCfg(
                joint_names_expr=["left_arm_[0-6]"],
                effort_limit_sim=87.0,
                velocity_limit_sim=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "rby1_rightarm": ImplicitActuatorCfg(
                joint_names_expr=["right_arm_[0-6]"],
                effort_limit_sim=87.0,
                velocity_limit_sim=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "rby1_leftgripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper_finger_l.*"],
                effort_limit_sim=200.0,
                velocity_limit_sim=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
            "rby1_rightgripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper_finger_r.*"],
                effort_limit_sim=200.0,
                velocity_limit_sim=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
            "rby1_torso": ImplicitActuatorCfg(
                joint_names_expr=["torso_[0-5]"],
                effort_limit=0.0,
                velocity_limit=0.0,
                effort_limit_sim=0.0,
                velocity_limit_sim=0.0,
                stiffness=1e7,
                damping=1e7,
            ),
            "rby1_head": ImplicitActuatorCfg(
                joint_names_expr=["head_[0-1]"],
                effort_limit=0.0,
                velocity_limit=0.0,
                effort_limit_sim=0.0,
                velocity_limit_sim=0.0,
                stiffness=1e7,
                damping=1e7,
            ),
            "rby1_wheel": ImplicitActuatorCfg(
                joint_names_expr=["left_wheel", "right_wheel"],
                effort_limit=0.0,
                velocity_limit=0.0,
                effort_limit_sim=0.0,
                velocity_limit_sim=0.0,
                stiffness=1e7,
                damping=1e7,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # for the actions, only set it for the joints that are 
    # for the arm and the gripper only and not for everything
    # maybe later on, set it for everything once there is tasks
    # that require the movement of the robot
    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=["left_arm_[0-6]", "right_arm_[0-6]", "gripper_finger_l.*", "gripper_finger_r.*"],
        scale={},
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # start up positin for the robot

    # start and reset position for the pot
    reset_box = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range":
            {
                "x": (-0.35, -0.05),
                "y": (0, 0.5),
                "z": (0.75, 1.25),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("tray"),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}},
    ) 


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # TODO: the following will be generated via a LLM
    # action_l2 = RewTerm(func=mdp.action_l2, weight=-0.01)
    pass


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length is exceeded
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class TrayEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the MuJoCo-style Humanoid walking environment."""

    # Scene settings
    scene: TraySceneCfg = TraySceneCfg(num_envs=4096, env_spacing=5.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 3
        # simulation settings
        self.sim.dt = 1 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        # default friction material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0
