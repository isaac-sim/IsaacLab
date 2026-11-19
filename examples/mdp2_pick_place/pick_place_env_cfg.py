# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pick-and-place of a cube onto a plate by a Franka, for evaluating vision-language-action policies.

An example for the review of the MDP2 Arena support, removed before merging; see ``README.md``.

The observations and actions follow the DROID contract that policies such as openpi's pi0 read:

- ``policy``: ``joint_pos`` (7 arm joints [rad]), ``gripper_pos`` (0 open to 1 closed), ``robot_joint_pos``, and
  ``actions`` (the last action);
- ``camera_obs``: ``external_camera_rgb`` and ``wrist_camera_rgb``, ``uint8`` images of shape (height, width, 3);
- actions: 7 absolute arm joint positions [rad], then a gripper command, closed above 0.5.

The task is composed from three partial configs, as an evaluation framework composes a scene, an embodiment, and a
task: :data:`SCENE`, :data:`EMBODIMENT`, and :data:`TASK`.
"""

import pick_place_mdp as mdp
from isaaclab_experimental import mdp2
from isaaclab_experimental.envs import MDP2RLEnvCfg, ResetStep
from isaaclab_experimental.mdp2 import mdp as mdp2_mdp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass, replace

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

robot = mdp2.Entity("robot")
cube = mdp2.Entity("cube")
plate = mdp2.Entity("plate")
scene = mdp2.Entity("scene")
external_camera = mdp2.Entity("external_camera")
wrist_camera = mdp2.Entity("wrist_camera")

ARM_JOINTS = robot.joints("panda_joint[1-7]")
GRIPPER_JOINT = robot.joints("panda_finger_joint1")
GRIPPER_OPEN, GRIPPER_CLOSED = 0.04, 0.0  # Finger joint positions [m]
CUBE_SIZE = 0.05  # [m]
PLATE_SIZE = (0.16, 0.16, 0.01)  # [m]

##
# Scene
##

CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/ExternalCamera",
    data_types=["rgb"],
    width=320,
    height=180,
    spawn=sim_utils.PinholeCameraCfg(focal_length=2.1, horizontal_aperture=5.376, clipping_range=(0.01, 10.0)),
    renderer_cfg=NewtonWarpRendererCfg(),
)


@configclass
class PickPlaceSceneCfg(InteractiveSceneCfg):
    """A Franka at the edge of a table with a cube and a plate on it, an external camera, and a wrist camera."""

    robot: ArticulationCfg = replace(FRANKA_PANDA_HIGH_PD_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.9, 1.2, 0.04),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.42, 0.3)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.0, -0.02)),
    )
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE,) * 3,
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            mass_props=sim_utils.MassCfg(mass=0.1),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, -0.15, CUBE_SIZE / 2)),
    )
    plate = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Plate",
        spawn=sim_utils.CuboidCfg(
            size=PLATE_SIZE,
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            mass_props=sim_utils.MassCfg(mass=0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.3, 0.8)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.15, PLATE_SIZE[2] / 2)),
    )
    # The external camera looks at the table from the side, like DROID's exterior camera; the wrist camera rides on
    # the hand, so its pose is refreshed for every render.
    external_camera = replace(
        CAMERA_CFG,
        offset=CameraCfg.OffsetCfg(pos=(0.05, -0.57, 0.66), rot=(-0.1572, 0.3244, 0.4068, 0.8394), convention="world"),
    )
    wrist_camera = replace(
        CAMERA_CFG,
        prim_path=(
            "{ENV_REGEX_NS}/Robot/Geometry/panda_link0/panda_link1/panda_link2/panda_link3/panda_link4/panda_link5"
            "/panda_link6/panda_link7/panda_hand/WristCamera"
        ),
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(focal_length=2.8, horizontal_aperture=5.376, clipping_range=(0.01, 10.0)),
        # Beside the hand, tilted 25 degrees toward the fingers, which show at the bottom of the image.
        offset=CameraCfg.OffsetCfg(pos=(0.12, 0.0, -0.02), rot=(-0.153, -0.153, 0.6903, 0.6903), convention="ros"),
    )
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.8)),
        spawn=sim_utils.GroundPlaneCfg(),
    )
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1000.0))

    def __post_init__(self):
        super().__post_init__()
        self.robot.spawn.variants["Physics"] = "mujoco"


##
# Task, from a scene, an embodiment, and a task
##

SCENE = mdp2.TaskCfg(
    events={
        "reset_layout": mdp2.Event(
            mdp.reset_layout,
            inputs={"env_origins": scene.env_origins},
            outputs={
                "object_pose": cube.root_link_pose_w,
                "object_vel": cube.root_com_vel_w,
                "target_pose": plate.root_link_pose_w,
                "target_vel": plate.root_com_vel_w,
            },
            params={
                "pool_size": 256,
                "object_range": ((0.4, 0.65), (-0.3, 0.0)),
                "target_range": ((0.4, 0.65), (0.0, 0.3)),
                "min_distance": 0.2,
                "seed": 0,
                "object_height": CUBE_SIZE / 2,
                "target_height": PLATE_SIZE[2] / 2,
            },
        ),
    },
)
"""What the scene contributes: a layout of the cube and the plate, drawn from a pool at every reset."""

EMBODIMENT = mdp2.TaskCfg(
    actions={
        "arm": mdp2.DirectAction(robot.joint_pos_target, subset=ARM_JOINTS),
        "gripper": mdp2.Action(
            mdp2_mdp.binary_joint_position,
            inputs={"joint_ids": GRIPPER_JOINT},
            outputs={"joint_pos_target": robot.joint_pos_target},
            params={
                "open_position": GRIPPER_OPEN,
                "close_position": GRIPPER_CLOSED,
                "threshold": 0.5,
                "close_above": True,
            },
        ),
    },
    observations={
        "policy": mdp2.ObservationGroup(
            terms={
                "actions": mdp2.Observation(mdp2_mdp.last_action, inputs={"actions": mdp2.last_action}),
                "robot_joint_pos": mdp2.Observation(
                    mdp2_mdp.joint_pos, inputs={"joint_pos": robot.joint_pos, "joint_ids": robot.joints(".*")}
                ),
                "joint_pos": mdp2.Observation(
                    mdp2_mdp.joint_pos, inputs={"joint_pos": robot.joint_pos, "joint_ids": ARM_JOINTS}
                ),
                "gripper_pos": mdp2.Observation(
                    mdp2_mdp.joint_pos_normalized,
                    inputs={"joint_pos": robot.joint_pos, "joint_ids": GRIPPER_JOINT},
                    params={"zero": GRIPPER_OPEN, "one": GRIPPER_CLOSED},
                ),
            },
            concatenate=False,
        ),
        "camera_obs": mdp2.ObservationGroup(
            terms={
                "external_camera_rgb": mdp2.Observation(mdp2_mdp.image_rgb, inputs={"image": external_camera.rgb}),
                "wrist_camera_rgb": mdp2.Observation(mdp2_mdp.image_rgb, inputs={"image": wrist_camera.rgb}),
            },
            concatenate=False,
        ),
    },
    events={
        "reset_arm": mdp2.Event(
            mdp2_mdp.reset_joints_by_offset,
            inputs={
                "default_joint_pos": robot.default_joint_pos,
                "default_joint_vel": robot.default_joint_vel,
                "soft_joint_pos_limits": robot.soft_joint_pos_limits,
                "soft_joint_vel_limits": robot.soft_joint_vel_limits,
                "joint_ids": ARM_JOINTS,
            },
            outputs={"joint_pos": robot.joint_pos, "joint_vel": robot.joint_vel},
            params={"position_range": (-0.02, 0.02), "velocity_range": (0.0, 0.0)},
        ),
    },
)
"""What the robot contributes: its actions, its observations with the camera images, and the reset of its arm."""

TASK = mdp2.TaskCfg(
    quantities={
        "lifted": mdp2.Quantity(
            mdp.lifted, inputs={"root_link_pose_w": cube.root_link_pose_w}, params={"minimum_height": 0.1}
        ),
        "placed": mdp2.Quantity(
            mdp.placed,
            inputs={
                "object_pose": cube.root_link_pose_w,
                "object_vel": cube.root_com_vel_w,
                "target_pose": plate.root_link_pose_w,
            },
            params={"radius": 0.06, "maximum_height": CUBE_SIZE, "maximum_speed": 0.01},
        ),
        # Lifted, then placed: success, with the subtask reached so far as its state "stage".
        "success": mdp2.Quantity(
            mdp2_mdp.sequence,
            inputs={"lift": mdp2.quantities.lifted, "place": mdp2.quantities.placed},
            log=True,
            record=True,
        ),
        "object_moved": mdp2.Quantity(
            mdp.moved, inputs={"root_com_vel_w": cube.root_com_vel_w}, params={"speed": 0.05}, log=True, record=True
        ),
    },
    terminations={
        "time_out": mdp2.TimeLimit(episode_length_s=30.0),
        "success": mdp2.Termination(mdp2_mdp.is_positive, inputs={"value": mdp2.quantities.success}),
        "object_dropped": mdp2.Termination(
            mdp2_mdp.root_height_below_minimum,
            inputs={"root_link_pose_w": cube.root_link_pose_w},
            params={"minimum_height": -0.05},
        ),
    },
)
"""What the task contributes: the conditions of success, the metrics it records, and when episodes end."""


@configclass
class PickPlacePhysicsCfg(NewtonCfg):
    """Newton physics of the task, as for the lift tasks."""

    solver_cfg: MJWarpSolverCfg = MJWarpSolverCfg(
        solver="newton",
        integrator="implicitfast",
        njmax=300,
        nconmax=200,
        impratio=1.0,
        cone="pyramidal",
        iterations=100,
        ls_iterations=50,
        use_mujoco_contacts=False,
    )
    collision_cfg: NewtonCollisionPipelineCfg = NewtonCollisionPipelineCfg()
    default_shape_cfg: NewtonShapeCfg = NewtonShapeCfg()
    num_substeps: int = 2
    use_cuda_graph: bool = True


@configclass
class PickPlaceEnvCfg(MDP2RLEnvCfg):
    """Configuration of the pick-and-place environment. Control at 15 Hz, as for DROID."""

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=1 / 120, render_interval=8, physics=PickPlacePhysicsCfg())
    scene: PickPlaceSceneCfg = PickPlaceSceneCfg(num_envs=4, env_spacing=3.0)
    task: mdp2.TaskCfg = mdp2.compose(SCENE, EMBODIMENT, TASK)
    decimation: int = 8
    # Render again after resets, so that the first images of an episode show its new layout.
    reset_step = [ResetStep.EVENTS, ResetStep.COMMIT, ResetStep.RENDER, ResetStep.OBSERVE]
