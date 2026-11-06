# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-robot demonstrations on platforms, ramps, and granular terrain."""

import math

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_newton.sim.schemas import MujocoCollisionCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxCollisionCfg, PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.utils import PresetCfg


@configclass
class DepthPolicyCfg:
    """Input and action contract of the pretrained depth policy."""

    checkpoint: str = (
        f"{ISAAC_NUCLEUS_DIR}/IsaacLab/PretrainedCheckpoints/rsl_rl/"
        "IsaacContrib-Velocity-Rough-G1_Terrain_newtonmjwarp_none_rsl_rl.pt"
    )
    joint_names = (
        [
            f"{side}_{joint}_joint"
            for side in ("left", "right")
            for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
        ]
        + [f"waist_{axis}_joint" for axis in ("yaw", "roll", "pitch")]
        + [
            f"{side}_{joint}_joint"
            for side in ("left", "right")
            for joint in (
                "shoulder_pitch",
                "shoulder_roll",
                "shoulder_yaw",
                "elbow",
                "wrist_roll",
                "wrist_pitch",
                "wrist_yaw",
                "hand_index_0",
                "hand_index_1",
                "hand_middle_0",
                "hand_middle_1",
                "hand_thumb_0",
                "hand_thumb_1",
                "hand_thumb_2",
            )
        ]
    )
    default_joint_pos = {
        ".*_hip_pitch_joint": -0.2,
        ".*_knee_joint": 0.42,
        ".*_ankle_pitch_joint": -0.23,
        ".*_elbow_joint": 0.87,
        ".*_shoulder_pitch_joint": 0.35,
        "left_shoulder_roll_joint": 0.16,
        "right_shoulder_roll_joint": -0.16,
    }
    action_scale = {
        ".*_hip_yaw_joint": 0.1467,
        ".*_hip_roll_joint": 0.1467,
        ".*_hip_pitch_joint": 0.11,
        ".*_knee_joint": 0.1738,
        "waist_yaw_joint": 0.11,
        "waist_roll_joint": 0.0625,
        "waist_pitch_joint": 0.0625,
        ".*_ankle_pitch_joint": 0.625,
        ".*_ankle_roll_joint": 0.625,
        ".*_shoulder_.*_joint": 0.1563,
        ".*_elbow_joint": 0.1563,
        ".*_wrist_roll_joint": 0.1563,
        ".*_wrist_pitch_joint": 0.0313,
        ".*_wrist_yaw_joint": 0.0313,
        ".*_hand_.*_joint": 0.0153,
    }
    history_length = 5
    depth_shape = (3, 38, 64)
    depth_max_range = 3.0


@configclass
class TerrainPhysicsCfg(PresetCfg):
    """Rigid-body backends at a 2.5 ms solver cadence."""

    isaacsim_physx = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    physx = PhysxAutoCfg(isaacsim_physx=isaacsim_physx)
    newton_mjwarp = NewtonCfg(
        num_substeps=2,
        solver_cfg=MJWarpSolverCfg(
            njmax=300, nconmax=300, cone="pyramidal", impratio=1.0, integrator="implicitfast", use_mujoco_contacts=False
        ),
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
        default_shape_cfg=NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0),
    )
    default = newton_mjwarp


@configclass
class CourseCfg:
    """Surface appearance and static collision properties of the terrain courses."""

    ramp_angle_deg: float = 10.0
    granular_top: float = 2.5 * math.tan(math.radians(10.0))
    platform_heights = (
        (0.51, 0.56, 0.50),
        (0.56, 0.51, 0.58),
        (0.50, 0.56, 0.53),
        (0.55, 0.50, 0.58),
        (0.50, 0.57, 0.51),
        (0.54, 0.51, 0.56),
    )
    visual_materials = {
        "white": sim_utils.PreviewSurfaceCfg(diffuse_color=(0.73, 0.75, 0.75), roughness=0.9),
        "steel": sim_utils.PreviewSurfaceCfg(diffuse_color=(0.26, 0.28, 0.29), roughness=0.9),
        "black": sim_utils.PreviewSurfaceCfg(diffuse_color=(0.055, 0.06, 0.065), roughness=0.9),
        "wood": sim_utils.PreviewSurfaceCfg(diffuse_color=(0.62, 0.43, 0.20), roughness=0.9),
        "wall": sim_utils.PreviewSurfaceCfg(diffuse_color=(0.68, 0.67, 0.64), roughness=0.9),
        "floor": sim_utils.PreviewSurfaceCfg(diffuse_color=(0.39, 0.39, 0.37), roughness=0.9),
        "gravel": sim_utils.PreviewSurfaceCfg(diffuse_color=(0.32, 0.32, 0.30), roughness=0.9),
        "approach": sim_utils.PreviewSurfaceCfg(diffuse_color=(0.055, 0.065, 0.08), roughness=0.96),
        "end": sim_utils.PreviewSurfaceCfg(diffuse_color=(0.055, 0.065, 0.08), roughness=0.96),
        "gravel_floor_riser": sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.19, 0.20), roughness=0.96),
        "gravel_particles": sim_utils.PreviewSurfaceCfg(diffuse_color=(0.38, 0.37, 0.34), roughness=0.96),
        "sand_particles": sim_utils.PreviewSurfaceCfg(diffuse_color=(0.70, 0.55, 0.32), roughness=0.96),
    }
    textures = {"wood": "wood.png", "gravel": "gravel.png", "approach": "asphalt.png", "end": "asphalt.png"}
    physics_material = UsdPhysicsRigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
    tray_material = UsdPhysicsRigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.6, restitution=0.0)
    collision_props = [sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True), MujocoCollisionCfg(priority=1)]
    mesh_collision_props = sim_utils.UsdPhysicsMeshCollisionCfg(mesh_approximation_name="none")
    tray_collision_props = [
        sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
        PhysxCollisionCfg(contact_offset=0.001, rest_offset=0.0),
    ]


@configclass
class TerrainSceneCfg(InteractiveSceneCfg):
    """G1 with plate sole colliders and the deployment depth camera."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path="",
        visual_material=None,
        collision_group=-1,
        physics_material=sim_utils.PhysxRigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        ),
    )
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/IsaacLab/Robots/Unitree/G1/g1_29dof.usd",
            rigid_props=PhysxRigidBodyCfg(
                disable_gravity=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=PhysxArticulationCfg(
                enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos=DepthPolicyCfg().default_joint_pos,
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_.*", ".*_knee_joint", "waist_yaw_joint"],
                joint_effort_limit={".*_hip_.*": 88.0, ".*_knee_joint": 139.0, "waist_yaw_joint": 88.0},
                stiffness={
                    ".*_hip_pitch_joint": 200.0,
                    ".*_hip_roll_joint": 150.0,
                    ".*_hip_yaw_joint": 150.0,
                    ".*_knee_joint": 200.0,
                    "waist_yaw_joint": 200.0,
                },
                damping=5.0,
                armature=0.01,
            ),
            "feet": ImplicitActuatorCfg(
                joint_names_expr=[".*_ankle_.*"], joint_effort_limit=50.0, stiffness=20.0, damping=2.0, armature=0.01
            ),
            "waist": ImplicitActuatorCfg(
                joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
                joint_effort_limit=50.0,
                stiffness=200.0,
                damping=5.0,
                armature=0.01,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_.*"],
                joint_effort_limit={
                    ".*_shoulder_.*": 25.0,
                    ".*_elbow_joint": 25.0,
                    ".*_wrist_roll_joint": 25.0,
                    ".*_wrist_pitch_joint": 5.0,
                    ".*_wrist_yaw_joint": 5.0,
                },
                stiffness=40.0,
                damping=10.0,
                armature=0.01,
            ),
            "hands": ImplicitActuatorCfg(
                joint_names_expr=[".*_hand_.*"], joint_effort_limit=2.45, stiffness=40.0, damping=10.0, armature=0.001
            ),
        },
    )
    depth_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/depth_camera",
        offset=CameraCfg.OffsetCfg(pos=(0.0576, 0.0175, 0.4299), rot=(0.915, 0.0, -0.4035, 0.0), convention="world"),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=1.93, horizontal_aperture=3.663003227519435, clipping_range=(0.05, 10.0)
        ),
        width=DepthPolicyCfg().depth_shape[2],
        height=DepthPolicyCfg().depth_shape[1],
        data_types=["distance_to_image_plane"],
        update_latest_camera_pose=True,
        renderer_cfg=NewtonWarpRendererCfg(),
    )
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1300.0))


@configclass
class ActionsCfg:
    """Absolute joint position targets in radians."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=False)


@configclass
class ObservationsCfg:
    """Robot state available to example controllers."""

    @configclass
    class StateCfg(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(func=mdp.joint_pos)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel)

    state: StateCfg = StateCfg()


@configclass
class EventsCfg:
    """Nominal root pose and joint positions within the soft joint limits."""

    reset_base = EventTermCfg(
        func=mdp.reset_root_state_uniform, mode="reset", params={"pose_range": {}, "velocity_range": {}}
    )
    reset_joints = EventTermCfg(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )


@configclass
class G1PlatformsEnvCfg(ManagerBasedRLEnvCfg):
    """Depth-based locomotion over uneven white platforms."""

    class_type = "{DIR}.terrain_env:G1TerrainEnv"
    policy: DepthPolicyCfg = DepthPolicyCfg()
    course: CourseCfg = CourseCfg()
    robot_visual_materials = {
        "light": sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7), roughness=0.5),
        "dark": sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2), roughness=0.5),
    }
    warmup_height: float | None = 1.38
    terrain_name: str = "platforms"
    duration: float = 7.76
    heading: float = math.pi
    camera_eye: tuple[float, float, float] = (3.5777457457175097, -0.8435508432753542, 2.1149220372215085)
    camera_target: tuple[float, float, float] = (1.1391558425933481, 3.3213482163422823, 0.8084400788397561)
    camera_vfov: float = 45.571061887967865
    sim = sim_utils.SimulationCfg(
        dt=0.005,
        render_interval=4,
        physics=TerrainPhysicsCfg(),
        use_newton_actuators=False,
        physics_material=sim_utils.PhysxRigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        ),
    )
    scene: TerrainSceneCfg = TerrainSceneCfg(num_envs=1, env_spacing=20.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    decimation = 4
    seed = 43
    rewards = {}
    terminations = {}
    commands = {}
    episode_length_s = 1000.0

    def __post_init__(self):
        self.scene.robot.init_state.pos = (3.25, 2.15, 1.3400013446807861)
        self.scene.robot.init_state.rot = (0.0, 0.0, 1.0, 0.0)


@configclass
class G1RampsEnvCfg(G1PlatformsEnvCfg):
    """Depth-based locomotion over wooden ramps."""

    terrain_name = "ramps"
    warmup_height = 1.363555205562947
    duration = 8.68
    heading = 0.0
    camera_eye = (6.230492542938942, -3.910147672506915, 3.1521005517604688)
    camera_target = (2.7742890567655323, -0.7747420522966051, 1.3565813903711013)
    camera_vfov = 20.408707665145386

    def __post_init__(self):
        self.scene.robot.init_state.pos = (0.22, -0.3, 1.308342456817627)
        self.scene.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)


@configclass
class G1GranularEnvCfg(G1PlatformsEnvCfg):
    """SONIC traversal of gravel and sand using MPM or PhysX PBD."""

    policy_repo: str = "nvidia/GEAR-SONIC"
    policy_revision: str = "6733128a3d8a523b1418b06bca3cdf61c8b0987f"
    terrain_name = "granular"
    warmup_height = None
    duration = 23.04
    heading = 0.0
    camera_eye = (12.640227474658403, -1.9951440805141667, 1.529737129878094)
    camera_target = (6.525763654021182, 1.2626086601986954, 0.5295475146069122)
    camera_vfov = 30.974660215824073
    seed = 42

    def __post_init__(self):
        self.scene.robot.init_state.pos = (5.1, 0.0, 1.2338175773620605)
        self.scene.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)
        self.scene.depth_camera = None
        self.scene.light.spawn.intensity = 750.0
        self.scene.light.spawn.texture_file = (
            f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"
        )
        self.events.physics_material = EventTermCfg(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.8, 0.8),
                "dynamic_friction_range": (0.6, 0.6),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
            },
        )


@configclass
class G1TerrainPresetsCfg(PresetCfg):
    """Fixed courses for the pretrained depth and SONIC controllers."""

    platforms = G1PlatformsEnvCfg()
    ramps = G1RampsEnvCfg()
    granular = G1GranularEnvCfg()
    default = platforms
