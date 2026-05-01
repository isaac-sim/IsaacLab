# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from dataclasses import MISSING

from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.simulation_cfg import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg

import isaaclab_tasks.manager_based.manipulation.deploy.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.deploy.mdp.noise_models import ResetSampledConstantNoiseModelCfg

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
# Use the FLATTENED USDs (produced by ``scripts/tools/flatten_cable_usd.py``).
# Flattening bakes any per-prim ``xformOp`` translations/rotations into the mesh
# vertices and recentres the collidable mesh so that ``bbox_center == (0, 0, 0)``
# in the rigid-body frame. This is required for stable grasping because the
# IK-based grasp event places the gripper at the rigid-body origin via
# ``grasp_offset`` -- if the visible/collision mesh is offset from the body
# origin (as in the un-flattened source USDs), the fingertips end up beside the
# plug rather than around it, the gripper closes on empty space, and the plug
# slides out (which is what we observed in the un-flattened smoke tests).
ASSETS_DIR = os.path.join(CONFIG_DIR, "cable_insertion_assets", "flattened")

##
# Asset Configurations
##


@configclass
class GB300Plug(RigidObjectCfg):
    """Configuration for GB300 Plug (held asset)."""

    prim_path = "{ENV_REGEX_NS}/GB300Plug"
    spawn = sim_utils.UsdFileCfg(
        usd_path=os.path.join(ASSETS_DIR, "plug_A_no_snapfit_latch_no_bulge_collision_mesh.usd"),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            kinematic_enabled=False,
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=128,
            solver_velocity_iteration_count=1,
            max_contact_impulse=1e32,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    )
    init_state = RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.1), rot=(0.0, 0.0, 0.0, 1.0))


@configclass
class GB300Socket(RigidObjectCfg):
    """Configuration for GB300 Socket (fixed asset)."""

    prim_path = "{ENV_REGEX_NS}/GB300Socket"
    spawn = sim_utils.UsdFileCfg(
        usd_path=os.path.join(ASSETS_DIR, "socket_A_simplified_minimal_transformed.usd"),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            kinematic_enabled=True,
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=128,
            solver_velocity_iteration_count=1,
            max_contact_impulse=1e32,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=None),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    )
    init_state = RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.0, 0.1), rot=(0.0, 0.0, 0.0, 1.0))


##
# Environment configuration
##


@configclass
class CableInsertionSceneCfg(InteractiveSceneCfg):
    """Configuration for the cable insertion scene."""

    replicate_physics = False

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    gb300_plug = GB300Plug()
    gb300_socket = GB300Socket()

    robot: ArticulationCfg = MISSING

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

        joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
        joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
        socket_pos = ObsTerm(
            func=mdp.rigid_object_pos_w,
            params={"asset_cfg": SceneEntityCfg("gb300_socket"), "offset": [0.0, 0.0, 0.0]},
            noise=ResetSampledConstantNoiseModelCfg(
                noise_cfg=UniformNoiseCfg(n_min=-0.01, n_max=0.01, operation="add")
            ),
        )
        socket_quat = ObsTerm(
            func=mdp.rigid_object_quat_w,
            params={"asset_cfg": SceneEntityCfg("gb300_socket")},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
        joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
        socket_pos = ObsTerm(
            func=mdp.rigid_object_pos_w,
            params={"asset_cfg": SceneEntityCfg("gb300_socket"), "offset": [0.0, 0.0, 0.0]},
        )
        socket_quat = ObsTerm(
            func=mdp.rigid_object_quat_w,
            params={"asset_cfg": SceneEntityCfg("gb300_socket")},
        )
        plug_pos = ObsTerm(
            func=mdp.rigid_object_pos_w,
            params={"asset_cfg": SceneEntityCfg("gb300_plug")},
        )
        plug_quat = ObsTerm(
            func=mdp.rigid_object_quat_w,
            params={"asset_cfg": SceneEntityCfg("gb300_plug")},
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    plug_socket_keypoint_tracking = RewTerm(
        func=mdp.keypoint_two_body_error,
        weight=-1.5,
        params={
            "asset_cfg_1": SceneEntityCfg("gb300_socket"),
            "asset_cfg_2": SceneEntityCfg("gb300_plug"),
            "keypoint_scale": 0.15,
        },
    )

    plug_socket_keypoint_tracking_exp = RewTerm(
        func=mdp.keypoint_two_body_error_exp,
        weight=3.0,
        params={
            "asset_cfg_1": SceneEntityCfg("gb300_socket"),
            "asset_cfg_2": SceneEntityCfg("gb300_plug"),
            "kp_exp_coeffs": [(10, 0.0001), (50, 0.0001), (150, 0.0001), (300, 0.0001)],
            "kp_use_sum_of_exps": False,
            "keypoint_scale": 0.15,
        },
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-5.0e-06)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CableInsertionEnvCfg(ManagerBasedRLEnvCfg):
    scene: CableInsertionSceneCfg = CableInsertionSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    sim: SimulationCfg = SimulationCfg(
        physics=PhysxCfg(
            gpu_collision_stack_size=2**30,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
        ),
    )

    def __post_init__(self):
        """Post initialization."""
        self.episode_length_s = 6.66
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.decimation = 33
        self.sim.render_interval = self.decimation
        self.sim.dt = 1.0 / 1000.0
