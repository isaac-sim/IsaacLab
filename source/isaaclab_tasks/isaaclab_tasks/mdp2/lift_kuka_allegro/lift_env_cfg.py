# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kuka Allegro lift: grasp an object from a table and carry it to a commanded position."""

from isaaclab_experimental import mdp2
from isaaclab_experimental.envs import MDP2RLEnvCfg
from isaaclab_experimental.mdp2 import mdp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_physx.sim.schemas import PhysxCollisionCfg, PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import MeshCapsuleCfg, MeshConeCfg, MeshCuboidCfg, MeshSphereCfg, RigidBodyMaterialBaseCfg
from isaaclab.utils import configclass, replace

from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.kuka_allegro import KUKA_ALLEGRO_CFG

from . import mdp as lift
from .clearance import mesh_clearance, slab_clearance
from .reset_bank import reset_from_bank

SIM_DT = 1 / 120
DECIMATION = 4  # 30 Hz control
OBJECT_PHYSICS = {
    "physics_material": RigidBodyMaterialBaseCfg(static_friction=0.5),
    "collision_props": [PhysxCollisionCfg(contact_offset=0.002)],
}
"""Physics properties shared by the object shapes."""

robot = mdp2.Entity("robot")
target = mdp2.Entity("object")
fingertip_contacts = mdp2.Entity("fingertip_contacts")
scene = mdp2.Entity("scene")
joints = robot.joints(".*")
hand = robot.bodies("palm_link", ".*_tip")
thumb = fingertip_contacts.bodies("thumb_link_3")
fingers = fingertip_contacts.bodies("(index|middle|ring)_link_3")
fingertips = fingertip_contacts.bodies(".*")

GOAL_RANGE = ((-0.7, -0.3), (-0.25, 0.25), (0.55, 0.95))
"""Ranges of the commanded object position x, y, and z in the robot's root frame [m]."""

# The object is grasped when the thumb and another finger press it.
grasp = {"thumb_force": mdp2.quantities.thumb_force, "finger_force": mdp2.quantities.finger_force}
goal = {"goal_error": mdp2.quantities.goal_error, **grasp}
robot_state = {"joint_pos": robot.joint_pos, "joint_vel": robot.joint_vel}
object_state = {"object_pose_w": target.root_link_pose_w, "object_vel_w": target.root_com_vel_w}
joint_limits = {
    "default_joint_pos": robot.default_joint_pos,
    "default_joint_vel": robot.default_joint_vel,
    "soft_joint_pos_limits": robot.soft_joint_pos_limits,
    "soft_joint_vel_limits": robot.soft_joint_vel_limits,
}

# The curriculum's progress in [0, 1]: observation noise grows from zero with it, and bounds and gravity move with it.
progress = mdp2.term_state("events.adr", "progress")


def noise(bound: float) -> mdp2.UniformNoise:
    """Return noise in ``[-bound, bound]`` at full curriculum progress, and none before the curriculum starts."""
    return mdp2.UniformNoise(-bound, bound, scale=progress)


@configclass
class KukaAllegroLiftSceneCfg(InteractiveSceneCfg):
    """A Kuka arm with an Allegro hand next to a table with one of 16 objects, and object contacts on the fingertips."""

    robot: ArticulationCfg = replace(KUKA_ALLEGRO_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                MeshCuboidCfg(size=(0.05, 0.1, 0.1), **OBJECT_PHYSICS),
                MeshCuboidCfg(size=(0.05, 0.05, 0.1), **OBJECT_PHYSICS),
                MeshCuboidCfg(size=(0.025, 0.1, 0.1), **OBJECT_PHYSICS),
                MeshCuboidCfg(size=(0.025, 0.05, 0.1), **OBJECT_PHYSICS),
                MeshCuboidCfg(size=(0.025, 0.025, 0.1), **OBJECT_PHYSICS),
                MeshCuboidCfg(size=(0.01, 0.1, 0.1), **OBJECT_PHYSICS),
                MeshSphereCfg(radius=0.05, **OBJECT_PHYSICS),
                MeshSphereCfg(radius=0.025, **OBJECT_PHYSICS),
                MeshCapsuleCfg(radius=0.04, height=0.025, **OBJECT_PHYSICS),
                MeshCapsuleCfg(radius=0.04, height=0.01, **OBJECT_PHYSICS),
                MeshCapsuleCfg(radius=0.04, height=0.1, **OBJECT_PHYSICS),
                MeshCapsuleCfg(radius=0.025, height=0.1, **OBJECT_PHYSICS),
                MeshCapsuleCfg(radius=0.025, height=0.2, **OBJECT_PHYSICS),
                MeshCapsuleCfg(radius=0.01, height=0.2, **OBJECT_PHYSICS),
                MeshConeCfg(radius=0.05, height=0.1, **OBJECT_PHYSICS),
                MeshConeCfg(radius=0.025, height=0.1, **OBJECT_PHYSICS),
            ],
            rigid_props=PhysxRigidBodyCfg(
                solver_position_iteration_count=16, solver_velocity_iteration_count=0, disable_gravity=False
            ),
            collision_props=[
                sim_utils.UsdPhysicsCollisionCfg(),
                sim_utils.UsdPhysicsMeshCollisionCfg(mesh_approximation_name="convexHull"),
            ],
            mass_props=sim_utils.MassCfg(mass=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.55, 0.1, 0.35)),
    )
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 1.5, 0.04),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.15, 0.15)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.55, 0.0, 0.235), rot=(0.0, 0.0, 0.0, 1.0)),
    )
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg(color=(1.0, 1.0, 1.0)), collision_group=-1
    )
    # One sensor for the four fingertips, where MDP1 has one per fingertip.
    fingertip_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ee_link/(index|middle|ring|thumb)_link_3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    sky_light = AssetBaseCfg(prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=750.0))


@configclass
class KukaAllegroLiftPhysicsCfg(PresetCfg):
    """Newton physics presets of the Kuka Allegro lift task."""

    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=300,
            nconmax=200,
            impratio=1.0,
            cone="pyramidal",
            update_data_interval=2,
            iterations=100,
            ls_iterations=50,
            use_mujoco_contacts=False,
            ccd_iterations=35,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(rigid_contact_max=4000000),
        default_shape_cfg=NewtonShapeCfg(),
        num_substeps=2,
        debug_mode=False,
    )
    default: NewtonCfg = newton_mjwarp


CANDIDATES = mdp2.TaskCfg(
    observations={
        "descriptor": mdp2.ObservationGroup(
            grasp_travel_distance=mdp2.Observation(
                lift.grasp_travel_distance,
                inputs={
                    "body_link_pose_w": robot.body_link_pose_w,
                    "body_ids": hand,
                    "root_link_pose_w": robot.root_link_pose_w,
                    "object_pose_w": target.root_link_pose_w,
                },
                params={"goal_center": tuple((low + high) / 2 for low, high in GOAL_RANGE)},
            ),
        ),
    },
    terminations={
        "object_robot_clearance": mdp2.Termination(
            mesh_clearance,
            inputs={
                "body_link_pose_w": robot.body_link_pose_w,
                "body_ids": robot.bodies(".*"),
                "object_pose_w": target.root_link_pose_w,
            },
            params={"min_clearance": 0.0},
        ),
        "robot_table_clearance": mdp2.Termination(
            slab_clearance,
            inputs={
                "body_link_pose_w": robot.body_link_pose_w,
                # Everything but the arm's base, which stands on the ground; finger links are also *_link_N.
                "body_ids": robot.bodies("(?!iiwa7_link_0$).*"),
                "object_pose_w": target.root_link_pose_w,
                "env_origins": scene.env_origins,
            },
            params={
                "obstacle_slabs": (((-0.95, -0.15), (-0.75, 0.75), 0.255), (None, None, 0.0)),
                "min_clearance": 0.02,
            },
        ),
    },
    events={
        "reset_robot_joints": mdp2.Event(
            mdp.reset_joints_by_offset,
            inputs={**joint_limits, "joint_ids": joints},
            outputs=robot_state,
            params={"position_range": (-0.5, 0.5), "velocity_range": (0.0, 0.0)},
        ),
        "reset_robot_wrist_joint": mdp2.Event(
            mdp.reset_joints_by_offset,
            inputs={**joint_limits, "joint_ids": robot.joints("iiwa7_joint_7")},
            outputs=robot_state,
            params={"position_range": (-3.0, 3.0), "velocity_range": (0.0, 0.0)},
        ),
        "reset_object": mdp2.Event(
            mdp.reset_root_state_uniform,
            inputs={
                "default_root_pose": target.default_root_pose,
                "default_root_vel": target.default_root_vel,
                "env_origins": scene.env_origins,
            },
            outputs={"root_link_pose_w": target.root_link_pose_w, "root_com_vel_w": target.root_com_vel_w},
            params={
                "pose_range": ((-0.2, 0.2), (-0.2, 0.2), (0.0, 0.4), (-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14)),
                "velocity_range": ((0.0, 0.0),) * 6,
            },
        ),
        # A quarter of the candidates start with the object at the palm, whose pose the joint resets changed.
        "reset_object_to_target": mdp2.Event(
            lift.reset_to_target,
            inputs={"body_link_pose_w": robot.body_link_pose_w, "body_ids": robot.bodies("palm_link")},
            outputs=object_state,
            params={"position_range": ((0.03, 0.07), (-0.04, 0.04), (0.02, 0.08)), "probability": 0.25},
            commit=True,
        ),
    },
)
"""Candidate start states of the reset bank: how they are drawn, which are rejected, and what spreads them."""

TASK = mdp2.TaskCfg(
    actions={
        # Targets relative to the current joint positions, re-based before every physics step as in MDP1.
        "action": mdp2.DirectAction(robot.joint_pos_target, scale=0.1, offset=robot.joint_pos),
    },
    commands={
        "object_pose": mdp2.Command(
            mdp.uniform_pose_command,
            resampling_time_range=(4.0, 6.0),
            params={
                "pose_range": (*GOAL_RANGE, (-3.14, 3.14), (-3.14, 3.14), (0.0, 0.0)),
                "make_quat_unique": False,
            },
        ),
    },
    quantities={
        "thumb_force": mdp2.Quantity(
            lift.max_contact_force,
            inputs={"normal_force_matrix_w": fingertip_contacts.normal_force_matrix_w, "body_ids": thumb},
        ),
        "finger_force": mdp2.Quantity(
            lift.max_contact_force,
            inputs={"normal_force_matrix_w": fingertip_contacts.normal_force_matrix_w, "body_ids": fingers},
        ),
        "goal_error": mdp2.Quantity(
            lift.goal_error,
            inputs={
                "command": mdp2.commands.object_pose,
                "root_link_pose_w": robot.root_link_pose_w,
                "object_pose_w": target.root_link_pose_w,
            },
        ),
        "success": mdp2.Quantity(
            lift.succeeded, inputs=goal, params={"distance": 0.05, "contact_threshold": 0.01}, log=True
        ),
        "abnormal_robot": mdp2.Quantity(
            lift.abnormal_robot,
            inputs={
                "joint_vel": robot.joint_vel,
                "soft_joint_vel_limits": robot.soft_joint_vel_limits,
                "joint_ids": joints,
            },
        ),
        "adr_progress": mdp2.Quantity(mdp.value, inputs={"value": progress}, log=True),
    },
    observations={
        "policy": mdp2.ObservationGroup(
            object_quat_b=mdp2.Observation(
                lift.object_quat_b,
                inputs={"root_link_pose_w": robot.root_link_pose_w, "object_pose_w": target.root_link_pose_w},
                noise=noise(0.03),
                history_length=5,
            ),
            target_object_pose_b=mdp2.Observation(
                mdp.generated_commands, inputs={"command": mdp2.commands.object_pose}, history_length=5
            ),
            actions=mdp2.Observation(mdp.last_action, inputs={"actions": mdp2.last_action}, history_length=5),
        ),
        "proprio": mdp2.ObservationGroup(
            joint_pos=mdp2.Observation(
                mdp.joint_pos,
                inputs={"joint_pos": robot.joint_pos, "joint_ids": joints},
                noise=noise(0.1),
                history_length=5,
            ),
            joint_vel=mdp2.Observation(
                mdp.joint_vel,
                inputs={"joint_vel": robot.joint_vel, "joint_ids": joints},
                noise=noise(0.2),
                history_length=5,
            ),
            # Poses only: body velocities are the most solver-sensitive observables, and the history carries them.
            hand_tips_state_b=mdp2.Observation(
                lift.body_state_b,
                inputs={
                    "body_link_pose_w": robot.body_link_pose_w,
                    "root_link_pose_w": robot.root_link_pose_w,
                    "body_ids": hand,
                },
                noise=noise(0.01),
                clip=(-2.0, 2.0),
                history_length=5,
            ),
            contact=mdp2.Observation(
                lift.fingers_contact_force_b,
                inputs={
                    "normal_force_matrix_w": fingertip_contacts.normal_force_matrix_w,
                    "root_link_pose_w": robot.root_link_pose_w,
                    "body_ids": fingertips,
                },
                clip=(-20.0, 20.0),  # Fingertip contact forces stay under 20 N normally.
                history_length=5,
            ),
        ),
        "perception": mdp2.ObservationGroup(
            object_point_cloud=mdp2.Observation(
                lift.object_point_cloud_b,
                inputs={"root_link_pose_w": robot.root_link_pose_w, "object_pose_w": target.root_link_pose_w},
                params={"num_points": 64},
                noise=noise(0.01),
                clip=(-2.0, 2.0),
                history_length=5,
            ),
        ),
    },
    rewards={
        "action_l2": mdp2.Reward(mdp.action_l2, inputs={"actions": mdp2.last_action}, weight=-0.01),
        "fingers_to_object": mdp2.Reward(
            lift.object_ee_distance,
            inputs={
                "body_link_pose_w": robot.body_link_pose_w,
                "body_ids": hand,
                "object_pose_w": target.root_link_pose_w,
                **grasp,
            },
            params={"std": 0.4, "contact_threshold": 1.0},
            weight=0.05,
        ),
        "position_tracking": mdp2.Reward(
            lift.position_command_progress,
            inputs={"command": mdp2.commands.object_pose, **goal},
            params={"min_improvement": 0.0025, "contact_threshold": 0.1},
            weight=5.0,
        ),
        "success": mdp2.Reward(
            lift.success_reward, inputs=goal, params={"pos_std": 0.05, "contact_threshold": 0.01}, weight=10.0
        ),
        "early_termination": mdp2.Reward(mdp.value, inputs={"value": mdp2.quantities.abnormal_robot}, weight=-50.0),
        "good_finger_contact": mdp2.Reward(lift.contacts, inputs=grasp, params={"threshold": 0.01}, weight=1.0),
        "contact_count": mdp2.Reward(
            lift.contact_count,
            inputs={"normal_force_matrix_w": fingertip_contacts.normal_force_matrix_w, "body_ids": fingertips},
            params={"threshold": 0.01},
            weight=0.1,
        ),
    },
    terminations={
        "time_out": mdp2.TimeLimit(episode_length_s=12.0),
        "object_out_of_bound": mdp2.Termination(
            lift.out_of_bound,
            inputs={
                "object_pose_w": target.root_link_pose_w,
                "object_vel_w": target.root_com_vel_w,
                "env_origins": scene.env_origins,
                "progress": progress,
            },
            # The curriculum lowers the floor from above the table to the ground.
            params={
                "in_bound_range": ((-1.5, 0.5), (-2.0, 2.0), (0.3, 2.0)),
                "final_in_bound_range": ((-1.5, 0.5), (-2.0, 2.0), (0.0, 2.0)),
            },
        ),
        "abnormal_robot": mdp2.Termination(mdp.is_positive, inputs={"value": mdp2.quantities.abnormal_robot}),
    },
    events={
        # Newton has one friction coefficient, so the dynamic friction and MDP1's buckets do not apply.
        "robot_physics_material": mdp2.Event(
            mdp.randomize_rigid_body_material,
            mode=mdp2.EventMode.STARTUP,
            inputs={"body_ids": robot.bodies(".*")},
            params={"friction_range": (0.5, 1.0), "restitution_range": (0.0, 0.0)},
        ),
        "object_physics_material": mdp2.Event(
            mdp.randomize_rigid_body_material,
            mode=mdp2.EventMode.STARTUP,
            inputs={"body_ids": target.bodies(".*")},
            params={"friction_range": (0.5, 1.0), "restitution_range": (0.0, 0.0)},
        ),
        "object_physics_inertia": mdp2.Event(
            mdp.randomize_rigid_body_inertia,
            mode=mdp2.EventMode.STARTUP,
            inputs={"body_ids": target.bodies(".*")},
            params={"inertia_distribution_params": (0.01, 0.01), "operation": "add", "diagonal_only": True},
        ),
        "joint_stiffness_and_damping": mdp2.Event(
            mdp.randomize_actuator_gains,
            mode=mdp2.EventMode.STARTUP,
            inputs={"joint_ids": joints},
            params={
                "stiffness_distribution_params": (0.5, 2.0),
                "damping_distribution_params": (0.5, 2.0),
                "operation": "scale",
            },
        ),
        "joint_friction": mdp2.Event(
            mdp.randomize_joint_parameters,
            mode=mdp2.EventMode.STARTUP,
            inputs={"joint_ids": joints},
            params={"friction_distribution_params": (0.0, 5.0), "operation": "scale"},
        ),
        "object_scale_mass": mdp2.Event(
            mdp.randomize_rigid_body_mass,
            mode=mdp2.EventMode.STARTUP,
            inputs={"body_ids": target.bodies(".*")},
            params={"mass_distribution_params": (0.2, 2.5), "operation": "scale"},
        ),
        # The armature sets the effort-to-inertia ratio, so this randomizes how fast the fingers close.
        "finger_closing_speed": mdp2.Event(
            mdp.randomize_joint_parameters,
            mode=mdp2.EventMode.STARTUP,
            inputs={"joint_ids": robot.joints("(index|middle|ring|thumb)_joint_(0|1|2|3)")},
            params={
                "armature_distribution_params": (0.1, 1.0),
                "operation": "scale",
                "distribution": "log_uniform",
            },
        ),
        # Promoted on success: observation noise, the object's floor, and gravity move with the progress.
        "adr": mdp2.Event(
            mdp.difficulty_scheduler,
            inputs={"succeeded": mdp2.quantities.success},
            params={"min_difficulty": 0, "max_difficulty": 10},
        ),
        # Starting without gravity makes learning smoother and removes the need for a lifting reward.
        "variable_gravity": mdp2.Event(
            lift.curriculum_gravity,
            inputs={"progress": progress},
            outputs={"gravity_w": scene.gravity_w},
            params={"initial_gravity": (0.0, 0.0, 0.0), "final_gravity": (0.0, 0.0, -9.81)},
        ),
        # Restart from the valid start states that the policy solves about half the time.
        "reset": mdp2.Event(
            reset_from_bank,
            inputs={"succeeded": mdp2.quantities.success},
            outputs={**robot_state, **object_state},
            params={
                "candidates": CANDIDATES,
                "bank_size": 16,
                "history_length": 10,
                "target_success_rate": 0.5,
                "kappa": 1.0,
            },
        ),
    },
)
"""The MDP of the task."""


@configclass
class KukaAllegroLiftEnvCfg(MDP2RLEnvCfg):
    """Configuration of the Kuka Allegro lift environment."""

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=SIM_DT, render_interval=DECIMATION, physics=KukaAllegroLiftPhysicsCfg()
    )
    scene: KukaAllegroLiftSceneCfg = KukaAllegroLiftSceneCfg(num_envs=4096, env_spacing=3)
    task: mdp2.TaskCfg = TASK
    decimation: int = DECIMATION
