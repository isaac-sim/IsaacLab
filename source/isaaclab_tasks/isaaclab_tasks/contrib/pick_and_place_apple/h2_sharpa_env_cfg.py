# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""H2 + Sharpa Wave env for the pick-and-place apple task (PhysX, rigid apple)."""

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.h2_sharpa.metadata import H2_ACTION_JOINT_ORDER

from . import mdp
from .config import (
    APPLE_USD,
    BACKGROUND_USD,
    H2_PNP_APPLE_CUSTOM_JOINT_POS,
    H2_PNP_APPLE_INIT_POS,
    H2_PNP_APPLE_INIT_ROT,
    PLATE_USD,
    TABLE_USD,
    CameraPresets,
    H2RobotPresets,
)

# Compatibility alias; action order differs from Isaac articulation order.
h2_joint_names = H2_ACTION_JOINT_ORDER


@configclass
class H2PnpAppleSceneCfg(InteractiveSceneCfg):
    """H2 scene with front, left-wrist, and right-wrist cameras."""

    robot = H2RobotPresets.h2_sharpa_base_fix(
        init_pos=H2_PNP_APPLE_INIT_POS,
        init_rot=H2_PNP_APPLE_INIT_ROT,
        custom_joint_pos=H2_PNP_APPLE_CUSTOM_JOINT_POS,
    )

    background = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/background",
        spawn=sim_utils.UsdFileCfg(
            usd_path=BACKGROUND_USD,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-1.30, -0.30, 2.0),
            rot=(0.0, 0.0, 0.70710678, 0.70710678),
        ),
    )

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
        ),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=TABLE_USD,
            scale=(1.0, 1.0, 1.20),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.42, 0, 0.565),
            rot=(0, 0, 0.70710678, 0.70710678),
        ),
    )

    apple = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Apple",
        spawn=sim_utils.UsdFileCfg(
            usd_path=APPLE_USD,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.95,
                dynamic_friction=0.95,
                restitution=0.0,
            ),
            # 13/20 of the original (1.5885, 1.5885, 1.4274).
            scale=(1.1357775, 1.1357775, 1.120591),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.48, 0.09, 1.081),
            rot=(0, 0, 0, 1),
        ),
    )

    plate = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Plate",
        spawn=sim_utils.UsdFileCfg(
            usd_path=PLATE_USD,
            scale=(0.9, 0.9, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.5, -0.265, 1.055),
            rot=(0, 0, 0, 1),
        ),
    )

    # Dome and distant lights are infinite: a copy per env would light every other
    # env as well, so N envs render brighter and flatter than the N=1 the SFT data
    # was collected at. Only the local cylinder strip below is cloned per env.
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(1.0, 0.90, 0.80),
            intensity=420.0,
        ),
    )

    top_light_1 = AssetBaseCfg(
        prim_path="/World/top_light_1",
        spawn=sim_utils.DistantLightCfg(
            color=(1.0, 0.92, 0.84),
            intensity=1500.0,
            angle=5.0,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            rot=(0.8259431, 0.1503529, 0.4768585, 0.2604189),
        ),
    )

    top_light_2 = AssetBaseCfg(
        prim_path="/World/top_light_2",
        spawn=sim_utils.DistantLightCfg(
            color=(1.0, 0.95, 0.88),
            intensity=1000.0,
            angle=5.0,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            rot=(0.8259431, -0.1503529, 0.4768585, -0.2604189),
        ),
    )

    overhead_strip = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/overhead_strip",
        spawn=sim_utils.CylinderLightCfg(
            color=(1.0, 0.96, 0.90),
            intensity=2200.0,
            length=1.1,
            radius=0.025,
            treat_as_line=True,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.92, 0.0, 2.35),
            rot=(0.70710678, 0.70710678, 0.0, 0.0),
        ),
    )

    front_camera = CameraPresets.h2_front_fisheye_camera(height=240, width=320)
    left_wrist_camera = CameraPresets.left_shf3l_fisheye_camera(height=240, width=320)
    right_wrist_camera = CameraPresets.right_shf3l_fisheye_camera(height=240, width=320)


@configclass
class H2ActionsCfg:
    """Direct joint angle control."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=h2_joint_names,
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )


@configclass
class H2RLActionsCfg:
    """RL/Eval joint control with PhysX gravity feed-forward on both arms."""

    joint_pos = mdp.JointPositionActionCfg(
        class_type=mdp.H2GravityCompensatedJointPositionAction,
        asset_name="robot",
        joint_names=h2_joint_names,
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )


@configclass
class H2ObservationsCfg:
    """Joint state and three camera observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        robot_joint_state = ObsTerm(func=mdp.get_robot_joint_states)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class CameraImagesCfg(ObsGroup):
        front_camera = ObsTerm(
            func=mdp.warm_rgb_image,
            params={"sensor_cfg": SceneEntityCfg("front_camera"), "data_type": "rgb", "normalize": False},
        )
        left_wrist_camera = ObsTerm(
            func=mdp.warm_rgb_image,
            params={"sensor_cfg": SceneEntityCfg("left_wrist_camera"), "data_type": "rgb", "normalize": False},
        )
        right_wrist_camera = ObsTerm(
            func=mdp.warm_rgb_image,
            params={"sensor_cfg": SceneEntityCfg("right_wrist_camera"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    camera_images: CameraImagesCfg = CameraImagesCfg()


@configclass
class H2TerminationsCfg:
    """Time-out + success (apple placed on plate and right hand released)."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.apple_on_plate_and_released, time_out=False)


@configclass
class H2EventCfg:
    """Reset the scene, then randomize the apple in XY."""

    # Also latch PD targets (esp. head_pitch=0.6). Controllers that only
    # command a subset of joints otherwise leave head target at 0 and IdealPD
    # lifts the camera into the wall.
    reset_scene = EventTermCfg(
        func=base_mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )

    # Must follow ``reset_scene`` so the default reset does not overwrite it.
    randomize_apple_xy = EventTermCfg(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.015, 0.015), "y": (-0.015, 0.015)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("apple"),
        },
    )


@configclass
class H2PnpAppleEnvCfg(ManagerBasedRLEnvCfg):
    """Unitree H2 + Sharpa Wave pick-and-place apple env (PhysX)."""

    scene: H2PnpAppleSceneCfg = H2PnpAppleSceneCfg(
        num_envs=1,
        # Prevent replicated background rooms from intersecting.
        env_spacing=10.0,
        # Envs carry per-env assets, so clone prims rather than replicate physics.
        replicate_physics=False,
    )

    viewer: ViewerCfg = ViewerCfg(
        eye=(0.6, -0.2, 2.4),
        lookat=(-0.45, 0.0, 1.22),
        cam_prim_path="/OmniverseKit_Persp",
    )

    observations: H2ObservationsCfg = H2ObservationsCfg()
    actions: H2ActionsCfg = H2ActionsCfg()
    terminations: H2TerminationsCfg = H2TerminationsCfg()
    events: H2EventCfg = H2EventCfg()
    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        from isaaclab_physx.physics import PhysxCfg

        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 1 / 120
        if self.sim.physics is None:
            self.sim.physics = PhysxCfg()
        # self.sim.physics.enable_external_forces_every_iteration = True
        # self.sim.physics.solve_articulation_contact_last = True
        self.sim.physics.gpu_max_num_partitions = 32
        self.sim.render_interval = 2
        # SimulationCfg.render only exists on IsaacLab builds that ship RenderCfg;
        # it is absent from the pinned checkout here, where the bare attribute
        # access raised AttributeError before the environment was ever built.
        if hasattr(self.sim, "render"):
            self.sim.render.antialiasing_mode = "DLAA"


# ---------------------------------------------------------------------------
# RLinf variant: sparse stage rewards + stage-based success termination.
#
# The base ``H2PnpAppleEnvCfg`` above is shared by eval / replay / mimic (all
# with ``rewards = None``); this subclass adds the reward + termination +
# stage-tracking machinery for RL post-training / sim-real co-training, plus a
# 58-D policy-ordered joint-state observation the RLinf YAML slices into GR00T
# state keys. Registered as ``Isaac-PNP-Apple-H2-Sharpa-RLinf-v0``.
# ---------------------------------------------------------------------------
@configclass
class H2RLObservationsCfg(H2ObservationsCfg):
    """Policy observations for GR00T / RLinf (58-D positions in policy order)."""

    @configclass
    class PolicyCfg(H2ObservationsCfg.PolicyCfg):
        robot_policy_joint_pos = ObsTerm(func=mdp.get_robot_policy_joint_positions)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class H2RLTerminationsCfg:
    """Timeout, stage success, and apple-drop failure."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    task_success = DoneTerm(
        func=mdp.task_success_termination,
        time_out=False,
        params={"success_stage": 4, "print_log": False},
    )
    # apple_drop = DoneTerm(
    #     func=mdp.apple_drop_termination,
    #     time_out=True,
    #     params={"apple_cfg": SceneEntityCfg("apple"), "drop_margin": 0.05},
    # )


@configclass
class H2RLRewardsCfg:
    """Sparse stage rewards: left lift, right catch, place, then release."""

    left_grasp_lift = RewTerm(
        func=mdp.left_grasp_lift_reward,
        weight=1.0,
        params={
            "apple_cfg": SceneEntityCfg("apple"),
            "plate_cfg": SceneEntityCfg("plate"),
            "robot_cfg": SceneEntityCfg("robot"),
            "lift_z": 0.10,
            "grasp_dist": 0.26,
            "release_dist": 0.08,
            "place_grasp_dist": 0.30,
            # Broad placement band relative to the plate root. XY proximity
            # and right-hand control remain required to reject unrelated drops.
            "xy_radius": 0.10,
            "z_above": 0.025,
            "z_window": 0.075,
            "hold_steps": 1,
            "print_log": False,
        },
    )
    handover_to_right = RewTerm(
        func=mdp.handover_to_right_reward,
        weight=1.0,
        params={"print_log": False},
    )
    place_on_plate = RewTerm(
        func=mdp.place_on_plate_reward,
        weight=1.0,
        params={"print_log": False},
    )
    release_on_plate = RewTerm(
        func=mdp.release_on_plate_reward,
        weight=1.0,
        params={"print_log": False},
    )


@configclass
class H2RLEventCfg(H2EventCfg):
    """Scene reset, apple randomization, and stage tracker reset."""

    reset_task_stage = EventTermCfg(
        func=mdp.reset_task_stage,
        mode="reset",
        params={"apple_cfg": SceneEntityCfg("apple"), "print_log": True},
    )


@configclass
class H2PnpAppleRLEnvCfg(H2PnpAppleEnvCfg):
    """Unitree H2 + Sharpa pick-and-place apple RL / co-train environment."""

    actions: H2RLActionsCfg = H2RLActionsCfg()
    observations: H2RLObservationsCfg = H2RLObservationsCfg()
    terminations: H2RLTerminationsCfg = H2RLTerminationsCfg()
    events: H2RLEventCfg = H2RLEventCfg()
    rewards: H2RLRewardsCfg = H2RLRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
