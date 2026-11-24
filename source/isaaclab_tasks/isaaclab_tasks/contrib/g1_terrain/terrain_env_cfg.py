# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-robot demonstrations on platforms, ramps, and granular terrain."""

import math

from isaaclab_isaacsim.physics import PhysxCfg
from isaaclab_isaacsim.sim.schemas import PhysxArticulationCfg
from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.physics import MJWarpSolverCfg, MPMSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_newton.physics.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMParticleMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass, replace
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.string import resolve_matching_names

from ...core.velocity.config.g1.rough_env_cfg import (
    DEPTH_HEIGHT,
    DEPTH_MAX_RANGE,
    DEPTH_STACK,
    DEPTH_WIDTH,
    G1SceneCfg,
    G1StudentObservationsCfg,
)
from ...utils import PresetCfg, preset
from .sonic_parameters import DAMPING, JOINT_NAMES, STIFFNESS

TERRAIN_ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Contrib/G1Terrain"
GRANULAR_TOP = 2.5 * math.tan(math.radians(10.0))


def _sonic_actuators(actuators: dict[str, ImplicitActuatorCfg]) -> dict[str, ImplicitActuatorCfg]:
    """Apply SONIC gains and effort limits to the existing actuator groups."""
    actuators = {
        name: replace(actuator, joint_effort_limit={"feet": 20.0, "waist": 50.0}.get(name, 300.0))
        for name, actuator in actuators.items()
    }
    for name, actuator in actuators.items():
        if name == "hands":
            continue
        indices, _ = resolve_matching_names(actuator.joint_names_expr, JOINT_NAMES)
        actuator.stiffness = {JOINT_NAMES[i]: float(STIFFNESS[i]) for i in indices}
        actuator.damping = {JOINT_NAMES[i]: float(DAMPING[i]) for i in indices}
    return actuators


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
    default_joint_pos = G1SceneCfg().robot.init_state.joint_pos
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
    history_length = G1StudentObservationsCfg.ProprioHistoryCfg().history_length
    depth_shape = (DEPTH_STACK, DEPTH_HEIGHT, DEPTH_WIDTH)
    depth_max_range = DEPTH_MAX_RANGE


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
class TerrainSceneCfg(InteractiveSceneCfg):
    """G1 with plate sole colliders and the deployment depth camera."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=f"{TERRAIN_ASSET_DIR}/platforms.usdc",
        use_terrain_origins=False,
        visual_material=None,
        collision_group=-1,
        physics_material=sim_utils.PhysxRigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        ),
    )
    robot = G1SceneCfg().robot
    robot.spawn.variants = {"Colliders": "full"}
    robot.spawn.activate_contact_sensors = False
    robot.spawn.rigid_props.retain_accelerations = None
    robot.spawn.articulation_props = PhysxArticulationCfg(
        enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
    )
    robot.actuators["legs"].joint_effort_limit = {".*_hip_.*": 88.0, ".*_knee_joint": 139.0, "waist_yaw_joint": 88.0}
    robot.actuators["feet"].joint_effort_limit = 50.0
    robot.actuators["arms"].joint_effort_limit = {
        ".*_shoulder_.*": 25.0,
        ".*_elbow_joint": 25.0,
        ".*_wrist_roll_joint": 25.0,
        ".*_wrist_pitch_joint": 5.0,
        ".*_wrist_yaw_joint": 5.0,
    }
    robot.actuators["hands"].joint_effort_limit = 2.45
    depth_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/depth_camera",
        offset=CameraCfg.OffsetCfg(pos=(0.0576, 0.0175, 0.4299), rot=(0.915, 0.0, -0.4035, 0.0), convention="world"),
        spawn=sim_utils.PinholeCameraCfg(focal_length=1.93, horizontal_aperture=3.663, clipping_range=(0.05, 10.0)),
        width=DepthPolicyCfg().depth_shape[2],
        height=DepthPolicyCfg().depth_shape[1],
        data_types=["distance_to_image_plane"],
        update_latest_camera_pose=True,
        renderer_cfg=NewtonWarpRendererCfg(),
    )
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1300.0))


@configclass
class GranularSceneCfg(TerrainSceneCfg):
    """Authored PhysX particle beds or standard Newton MPM spawners, with SONIC gains."""

    terrain = replace(
        TerrainSceneCfg().terrain,
        usd_path=f"{TERRAIN_ASSET_DIR}/granular.usdc",
        class_type="isaaclab.terrains.terrain_importer:TerrainImporter",
    )
    terrain = preset(
        default=terrain,
        isaacsim_physx=replace(
            terrain,
            usd_path=f"{TERRAIN_ASSET_DIR}/granular_physx.usdc",
            class_type=f"{__package__}.terrain_env:ParticleTerrainImporter",
        ),
        physx=replace(
            terrain,
            usd_path=f"{TERRAIN_ASSET_DIR}/granular_physx.usdc",
            class_type=f"{__package__}.terrain_env:ParticleTerrainImporter",
        ),
    )
    robot = TerrainSceneCfg().robot
    robot.actuators = _sonic_actuators(robot.actuators)
    depth_camera = None
    sand = preset(
        default=MPMObjectCfg(
            prim_path="{ENV_REGEX_NS}/Sand",
            spawn=MPMGridCfg(
                lower=(8.25, -0.6, GRANULAR_TOP - 0.03),
                upper=(11.5, 0.6, GRANULAR_TOP),
                voxel_size=0.01,
                particles_per_cell=2,
                particle_placement="cell_center",
                jitter=0.002,
                material=MPMParticleMaterialCfg(density=1600.0, friction=0.6, yield_pressure=1e5),
                visual_color=(0.70, 0.55, 0.32),
            ),
        ),
        isaacsim_physx=None,
        physx=None,
    )
    gravel = preset(
        default=MPMObjectCfg(
            prim_path="{ENV_REGEX_NS}/Gravel",
            spawn=MPMGridCfg(
                lower=(5.5, -0.6, GRANULAR_TOP - 0.03),
                upper=(8.25, 0.6, GRANULAR_TOP),
                voxel_size=0.01,
                particles_per_cell=2,
                particle_placement="cell_center",
                jitter=0.002,
                material=MPMParticleMaterialCfg(density=1800.0, friction=0.85, yield_pressure=1e6),
                visual_color=(0.38, 0.37, 0.34),
            ),
        ),
        isaacsim_physx=None,
        physx=None,
    )


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
    robot_visual_materials = {
        "light": sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7), roughness=0.5),
        "dark": sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2), roughness=0.5),
    }
    warmup_height: float | None = 1.38
    duration: float = 7.76
    heading: float = math.pi
    camera_eye: tuple[float, float, float] = (3.578, -0.844, 2.115)
    camera_target: tuple[float, float, float] = (1.139, 3.321, 0.808)
    camera_vfov: float = 45.571
    sim = sim_utils.SimulationCfg(
        dt=preset(default=0.005, isaacsim_physx=0.0025, physx=0.0025),
        render_interval=preset(default=4, isaacsim_physx=8, physx=8),
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
    decimation = preset(default=4, isaacsim_physx=8, physx=8)
    seed = 43
    rewards = {}
    terminations = {}
    commands = {}
    episode_length_s = 1000.0

    def __post_init__(self) -> None:
        self.scene.robot.init_state.pos = (3.25, 2.15, 1.340)
        self.scene.robot.init_state.rot = (0.0, 0.0, 1.0, 0.0)


@configclass
class G1RampsEnvCfg(G1PlatformsEnvCfg):
    """Depth-based locomotion over wooden ramps."""

    warmup_height = 1.364
    duration = 8.68
    heading = 0.0
    camera_eye = (6.230, -3.910, 3.152)
    camera_target = (2.774, -0.775, 1.357)
    camera_vfov = 20.409
    scene: TerrainSceneCfg = G1PlatformsEnvCfg().scene
    scene.terrain.usd_path = f"{TERRAIN_ASSET_DIR}/ramps.usdc"

    def __post_init__(self) -> None:
        self.scene.robot.init_state.pos = (0.22, -0.3, 1.308)
        self.scene.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)


@configclass
class G1GranularEnvCfg(G1PlatformsEnvCfg):
    """SONIC traversal of gravel and sand using MPM or PhysX PBD."""

    policy_repo: str = "nvidia/GEAR-SONIC"
    policy_revision: str = "6733128a3d8a523b1418b06bca3cdf61c8b0987f"
    warmup_height = None
    duration = 23.04
    heading = 0.0
    camera_eye = (12.640, -1.995, 1.530)
    camera_target = (6.526, 1.263, 0.530)
    camera_vfov = 30.975
    seed = 42
    scene: GranularSceneCfg = GranularSceneCfg(num_envs=1, env_spacing=20.0)
    sim = G1PlatformsEnvCfg().sim
    sim.physics.newton_mjwarp = replace(
        sim.physics.newton_mjwarp,
        solver_cfg=CouplerProxyCfg(
            entries=[
                CouplerEntryCfg(
                    name="robot",
                    solver_cfg=sim.physics.newton_mjwarp.solver_cfg,
                    bodies=[r"{ENV_REGEX_NS}/Robot"],
                    include_static_shapes=True,
                ),
                CouplerEntryCfg(
                    name="media",
                    solver_cfg=MPMSolverCfg(
                        voxel_size=0.01,
                        grid_type="sparse",
                        grid_padding=0,
                        max_active_cell_count=1 << 19,
                        max_leaf_node_count=1 << 15,
                        max_lower_node_count=1024,
                        max_upper_node_count=32,
                        max_iterations=50,
                        strain_basis="P0",
                        critical_fraction=0.0,
                        transfer_scheme="pic",
                    ),
                    all_particles=True,
                    in_place=True,
                ),
            ],
            proxies=[
                CouplerProxyMappingCfg(
                    source="robot",
                    destination="media",
                    bodies=[r"{ENV_REGEX_NS}/Robot"],
                    mode="lagged",
                    collision_pipeline=None,
                ),
            ],
            iterations=1,
        ),
        collision_cfg=replace(sim.physics.newton_mjwarp.collision_cfg, soft_contact_max=0),
    )
    sim.physics.default = sim.physics.newton_mjwarp

    def __post_init__(self) -> None:
        self.scene.robot.init_state.pos = (5.1, 0.0, 1.234)
        self.scene.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)
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
