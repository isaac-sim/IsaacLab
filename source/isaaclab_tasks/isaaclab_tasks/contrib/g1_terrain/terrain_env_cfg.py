# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-robot reconstructions of uneven platforms and ramps."""

import math

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg

from .depth_parameters import DEFAULT_JOINT_POS, DEPTH_SHAPE


@configclass
class TerrainPhysicsCfg(PresetCfg):
    """Rigid-body backends at a 2.5 ms solver cadence."""

    isaacsim_physx = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
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
            usd_path="omniverse://isaac-dev.ov.nvidia.com/Isaac/IsaacLab/Robots/Unitree/G1/g1_29dof.usd",
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
            joint_pos=DEFAULT_JOINT_POS,
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
        width=DEPTH_SHAPE[2],
        height=DEPTH_SHAPE[1],
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
    policy_path: str = (
        "omniverse://isaac-dev.ov.nvidia.com/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/"
        "IsaacContrib-Velocity-Rough-G1_Terrain_newtonmjwarp_none_rsl_rl.pt"
    )
    warmup_height: float = 1.38
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
