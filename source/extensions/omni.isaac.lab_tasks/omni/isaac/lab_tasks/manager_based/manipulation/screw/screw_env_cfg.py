# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import ActionTermCfg as ActionTerm
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.markers.config import (  # isort: skip
    DEFORMABLE_TARGET_MARKER_CFG,
    FRAME_MARKER_CFG,
    RED_ARROW_X_MARKER_CFG,
)
from omni.isaac.lab.markers.visualization_markers import VisualizationMarkersCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import MassPropertiesCfg, RigidBodyPropertiesCfg
from omni.isaac.lab.sim.simulation_cfg import PhysxCfg, SimulationCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import omni.isaac.lab_tasks.manager_based.manipulation.screw.mdp as mdp
from omegaconf import OmegaConf
import copy
from typing import Literal
##
# Scene definition
##
FRAME_MARKER_SMALL_CFG = copy.deepcopy(FRAME_MARKER_CFG)
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.008, 0.008, 0.008)
RED_PLATE_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "height": sim_utils.CylinderCfg(
            radius=0.01,
            height=0.001,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0, 0), opacity=0.5),
        )
    }
)
BLUE_PLATE_MARKER_CFG = RED_PLATE_MARKER_CFG.copy()
BLUE_PLATE_MARKER_CFG.markers["height"].visual_material = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0, 0, 1.0), opacity=0.5
)
PLATE_ARROW_CFG = VisualizationMarkersCfg(
    markers={
        "frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.008, 0.008, 0.008),
        ),
    }
)

asset_factory = {
    "m8_loose": {
        "nut_path": f"{ISAAC_NUCLEUS_DIR}/Props/Factory/factory_nut_m8_loose/factory_nut_m8_loose.usd",
        "bolt_path": f"{ISAAC_NUCLEUS_DIR}/Props/Factory/factory_bolt_m8_loose/factory_bolt_m8_loose.usd",
        "nut_frame_offset": (0.0, 0.0, 0.011),
        "bolt_bottom_offset": (0.0, 0.0, 0.0),
    },
    "m8_tight": {
        "nut_path": f"{ISAAC_NUCLEUS_DIR}/Props/Factory/factory_nut_m8_tight/factory_nut_m8_tight.usd",
         "bolt_path": f"{ISAAC_NUCLEUS_DIR}/Props/Factory/factory_bolt_m8_tight/factory_bolt_m8_tight.usd",
         "nut_frame_offset": (0.0, 0.0, 0.011),
         "bolt_bottom_offset": (0.0, 0.0, 0.0),
         },
    "m16_tight": {},
    "m16_loose": {},
}

@configclass
class ScrewSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""
    screw_type : Literal["m8_loose", "m8_tight", "m16_loose", "m16_tight"] = "m8_tight"
    def __post_init__(self):
        screw_dict = asset_factory[self.screw_type]
        # world
        self.ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        )
        self.origin = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Origin",
        spawn=sim_utils.SphereCfg(
            radius=1e-3,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        )

        self.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
                # usd_path=f"/home/zixuanh/force_tool/assets/table_instanceable.usd",
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
        )

        # robots: will be populated by agent env cfg
        self.robot: ArticulationCfg = MISSING

        # objects
        self.nut: RigidObjectCfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Nut",
            spawn=sim_utils.UsdFileCfg(
                usd_path=screw_dict["nut_path"],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 0.0065)),
        )

        self.bolt: RigidObjectCfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Bolt",
            spawn=sim_utils.UsdFileCfg(
                usd_path=screw_dict["bolt_path"],
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.63, 0.0, 0.0)),
        )

        # lights
        self.light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
        )

        self.nut_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Origin",
            debug_vis=True,
            visualizer_cfg=PLATE_ARROW_CFG.replace(prim_path="/Visuals/Nut"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Nut/factory_nut",
                    name="nut",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.011)),
                )
            ],
        )

        self.bolt_frame: FrameTransformerCfg = MISSING


##
# MDP settings
##


@configclass
class BaseActionsCfg:
    """Action specifications for the MDP."""

    nut_action: ActionTerm | None = None
    arm_action: ActionTerm | None = None
    gripper_action: ActionTerm | None = None


@configclass
class BaseObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.01, n_max=0.01))
        # joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-0.01, n_max=0.01))
        bolt_pose = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("bolt")})
        nut_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("nut")})
        nut_quat = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("nut")})
        nut_lin_vel = ObsTerm(func=mdp.root_lin_vel_w, params={"asset_cfg": SceneEntityCfg("nut")})
        nut_ang_vel = ObsTerm(func=mdp.root_ang_vel_w, params={"asset_cfg": SceneEntityCfg("nut")})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # nut_physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("nut"),
    #         "static_friction_range": (0.8, 0.8),
    #         "dynamic_friction_range": (0.6, 0.6),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 64,
    #     },
    # )

    # bolt_physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("bolt"),
    #         "static_friction_range": (0.8, 0.8),
    #         "dynamic_friction_range": (0.6, 0.6),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 64,
    #     },
    # )

    reset_default = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )

    # reset_robot_joints = EventTerm(
    #     func=mdp.reset_joints_by_scale,
    #     mode="reset",
    #     params={
    #         "position_range": (0.5, 1.5),
    #         "velocity_range": (0.0, 0.0),
    #     },
    # )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # action_rate = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500}
    # )

    # joint_vel = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 4500}
    # )


##
# Environment configuration
##


@configclass
class BaseScrewEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the screw end-effector pose tracking environment."""
    
    scene: ScrewSceneCfg = MISSING
    # Basic settings
    observations: BaseObservationsCfg = BaseObservationsCfg()
    actions: BaseActionsCfg = BaseActionsCfg()
    # MDP settings
    rewards = MISSING
    terminations = MISSING
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 60.0,
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_collision_stack_size=2**31,
            gpu_heap_capacity=2**31,
            gpu_temp_buffer_capacity=2**30,
            gpu_max_rigid_patch_count=2**24,
        ),
    )
    def get_default_env_params(self):
        """Set default environment parameters."""
        self.env_params.scene = self.env_params.get("scene", OmegaConf.create())
        self.env_params.sim = self.env_params.get("sim", OmegaConf.create())
        self.env_params.scene.screw_type = self.env_params.scene.get("screw_type", "m8_tight")
        self.env_params.scene.nut = self.env_params.scene.get("nut", OmegaConf.create())
        self.env_params.sim.dt = self.env_params.sim.get("dt", 1.0 / 60.0)
    
    def __post_init__(self):
        """Post initialization."""
        self.get_default_env_params()
        self.scene = ScrewSceneCfg(num_envs=4096, env_spacing=2.5, screw_type=self.env_params.scene.screw_type)
        # general settings
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.sim.dt = self.env_params.sim.dt
        self.episode_length_s = 24.0
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "bolt"
        self.viewer.eye = (0.1, 0, 0.04)
        self.viewer.lookat = (0, 0, 0.02)
        



###################################
#           Nut Tighten           #
def nut_tighten_reward_forge(env: ManagerBasedRLEnv, a: float = 100, b: float = 0, tol: float = 0):
    diff = mdp.rel_nut_bolt_bottom_distance(env)
    rewards = mdp.forge_kernel(diff, a, b, tol)
    return rewards


@configclass
class NutTightenRewardsCfg:
    """Reward terms for the MDP."""

    # task terms
    coarse_nut = RewTerm(func=nut_tighten_reward_forge, params={"a": 100, "b": 2}, weight=1.0)
    fine_nut = RewTerm(
        func=nut_tighten_reward_forge,
        params={
            "a": 500,
            "b": 0,
        },
        weight=1.0,
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.00001)


@configclass
class NutTightenTerminationsCfg:
    """Termination terms for screw tightening."""

    nut_screwed = DoneTerm(func=mdp.nut_fully_screwed, params={"threshold": 1e-4})
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class BaseNutTightenEnvCfg(BaseScrewEnvCfg):
    rewards: NutTightenRewardsCfg = NutTightenRewardsCfg()
    terminations: NutTightenTerminationsCfg = NutTightenTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.nut.init_state.pos = (6.3000e-01, 2.0661e-06, 3.0895e-03)
        self.scene.nut.init_state.rot = (-2.1609e-01, 6.6671e-05, -6.6467e-05, 9.7637e-01)
        self.scene.bolt_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Origin",
            debug_vis=True,
            visualizer_cfg=RED_PLATE_MARKER_CFG.replace(prim_path="/Visuals/Bolt"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Bolt/factory_bolt",
                    name="bolt_bottom",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.012)),  # strict 0.011 + 0.001
                ),
            ],
        )


###################################
#           Nut Thread           #
def nut_thread_reward_forge(env: ManagerBasedRLEnv, a: float = 100, b: float = 0, tol: float = 0):
    diff = mdp.rel_nut_bolt_tip_distance(env)
    rewards = mdp.forge_kernel(diff, a, b, tol)
    return rewards


@configclass
class NutThreadRewardsCfg:
    """Reward terms for the MDP."""

    # task terms
    coarse_nut = RewTerm(func=nut_thread_reward_forge, params={"a": 100, "b": 2}, weight=1.0)
    fine_nut = RewTerm(
        func=nut_thread_reward_forge,
        params={
            "a": 500,
            "b": 0,
        },
        weight=1.0,
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.00001)


@configclass
class NutThreadTerminationsCfg:
    """Termination terms for screw tightening."""

    nut_screwed = DoneTerm(func=mdp.nut_successfully_threaded, params={"threshold": 1e-4})
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class BaseNutThreadEnvCfg(BaseScrewEnvCfg):
    rewards: NutThreadRewardsCfg = NutThreadRewardsCfg()
    terminations: NutThreadTerminationsCfg = NutThreadTerminationsCfg()

    def __init__(self):
        super().__init__()  # Call the parent class's __init__ method

    def __post_init__(self):
        super().__post_init__()
        self.scene.nut.init_state.pos = (6.3000e-01, 4.0586e-06, 0.02)
        self.scene.nut.init_state.rot = (9.9833e-01, 1.2417e-04, -1.2629e-05, 5.7803e-02)

        self.scene.nut_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Origin",
            debug_vis=True,
            visualizer_cfg=BLUE_PLATE_MARKER_CFG.replace(prim_path="/Visuals/Nut"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Nut/factory_nut",
                    name="nut",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.011)),
                )
            ],
        )
        self.scene.bolt_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Origin",
            debug_vis=True,
            visualizer_cfg=RED_PLATE_MARKER_CFG.replace(prim_path="/Visuals/Bolt"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Bolt/factory_bolt",
                    name="bolt_tip",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0261)),  # 0.011 + 0.0161
                )
            ],
        )


