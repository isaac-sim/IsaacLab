# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_ovphysx.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.handover.handover_common import (
    ACTUATED_JOINT_NAMES,
    FINGERTIP_BODY_NAMES,
    GOAL_MARKER_CFG,
    OBJECT_RADIUS,
)
from isaaclab_tasks.utils import PresetCfg, preset

from isaaclab_assets.robots.shadow_hand import SHADOW_HAND_CFG, SHADOW_HAND_NEWTON_CFG


@configclass
class EventCfg:
    """Configuration for randomization (PhysX path).

    Note: this config is currently not wired into ``HandoverEnvCfg.events`` -
    it is kept as a reference for future event-randomization work. The event
    terms here use PhysX-only APIs (rigid-body materials, fixed tendons), so
    they would need a Newton variant before being enabled in the env.
    """

    # -- robot
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("right_hand"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("right_hand", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    robot_joint_pos_limits = EventTerm(
        func=mdp.randomize_joint_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("right_hand", joint_names=".*"),
            "lower_limit_distribution_params": (0.00, 0.01),
            "upper_limit_distribution_params": (0.00, 0.01),
            "operation": "add",
            "distribution": "gaussian",
        },
    )
    robot_tendon_properties = EventTerm(
        func=mdp.randomize_fixed_tendon_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("right_hand", fixed_tendon_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    # -- object
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # -- scene
    reset_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",
        is_global_time=True,
        interval_range_s=(36.0, 36.0),  # time_s = num_steps * (decimation * dt)
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
            "operation": "add",
            "distribution": "gaussian",
        },
    )


def _shadow_hand_cfg(
    prim_path: str,
    init_pos: tuple[float, float, float],
    init_rot: tuple[float, float, float, float],
) -> PresetCfg:
    """Per-hand Shadow Hand preset (PhysX, Newton MJWarp, and OVPhysX variants).

    All variants are placed at *prim_path* with the same init pose; per-hand
    differences (right vs left) come from the caller's *prim_path* / *init_pos* /
    *init_rot* — the gain tuning is identical on both hands.

    The Newton variant reuses the shared actuator defined on ``SHADOW_HAND_NEWTON_CFG``,
    raising its gains to ``20.0`` / ``2.0`` -- the catch needs more joint authority than
    reorientation. The scalar replaces the per-joint gain mapping, so it applies to the whole
    actuator group.
    """
    physx_cfg = SHADOW_HAND_CFG.replace(prim_path=prim_path).replace(
        init_state=ArticulationCfg.InitialStateCfg(pos=init_pos, rot=init_rot, joint_pos={".*": 0.0})
    )
    # Newton's importer bakes the asset's root orientation into the root joint (see the note on
    # SHADOW_HAND_NEWTON_CFG.init_state), so the task rotation must compose with it rather than
    # replace it — replacing leaves both palms rotated 90 degrees.
    newton_rot = tuple(
        math_utils.quat_mul(
            torch.tensor(init_rot, dtype=torch.float64),
            torch.tensor(SHADOW_HAND_NEWTON_CFG.init_state.rot, dtype=torch.float64),
        ).tolist()
    )
    newton_mjwarp_cfg = SHADOW_HAND_NEWTON_CFG.replace(
        prim_path=prim_path,
        init_state=SHADOW_HAND_NEWTON_CFG.init_state.replace(pos=init_pos, rot=newton_rot),
        actuators={
            **SHADOW_HAND_NEWTON_CFG.actuators,
            "fingers": SHADOW_HAND_NEWTON_CFG.actuators["fingers"].replace(stiffness=20.0, damping=2.0),
        },
    )
    ovphysx_cfg = SHADOW_HAND_CFG.replace(
        prim_path=prim_path,
        # OVPhysX does not expose the fixed-tendon runtime API, so spawn without tendon overrides.
        spawn=SHADOW_HAND_CFG.spawn.replace(fixed_tendons_props=None),
        init_state=SHADOW_HAND_CFG.init_state.replace(pos=init_pos, rot=init_rot),
    )
    return preset(
        default=newton_mjwarp_cfg,
        physx=physx_cfg,
        isaacsim_physx=physx_cfg,
        newton_mjwarp=newton_mjwarp_cfg,
        ovphysx=ovphysx_cfg,
    )


# Per-hand presets shared by the Direct environment and the manager scene.
RIGHT_HAND_CFG = _shadow_hand_cfg(
    prim_path="/World/envs/env_.*/RightRobot",
    init_pos=(0.0, 0.0, 0.5),
    init_rot=(0.0, 0.0, 0.0, 1.0),
)
LEFT_HAND_CFG = _shadow_hand_cfg(
    prim_path="/World/envs/env_.*/LeftRobot",
    init_pos=(0.0, -1.0, 0.5),
    init_rot=(0.0, 0.0, 1.0, 0.0),
)


@configclass
class ObjectCfg(PresetCfg):
    """Hand-over object preset.

    Both backends spawn the same procedural sphere as a free rigid body:
    Newton's :class:`~isaaclab_newton.assets.RigidObject` resolves the
    asset via the ``UsdPhysics.RigidBodyAPI`` that
    :class:`~isaaclab.sim.RigidBodyPropertiesCfg` applies. The Newton
    variant drops PhysX-only knobs (per-shape solver iterations, sleep
    thresholds, max depenetration velocity, custom physics material).
    """

    physx = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.SphereCfg(
            radius=OBJECT_RADIUS,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 1.0, 0.0)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.7),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(density=500.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.39, 0.54), rot=(0.0, 0.0, 0.0, 1.0)),
    )
    newton_mjwarp = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.SphereCfg(
            radius=OBJECT_RADIUS,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 1.0, 0.0)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(density=500.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.39, 0.54), rot=(0.0, 0.0, 0.0, 1.0)),
    )
    ovphysx = physx
    isaacsim_physx = physx
    default = newton_mjwarp


@configclass
class PhysicsCfg(PresetCfg):
    """Physics-backend preset (PhysX vs Newton/MJWarp).

    Newton settings mirror the single-agent ShadowHand Newton port: elliptic
    cone, ``impratio=10`` (favors normal contacts over friction), 100 solver
    iterations, 2 substeps. Empirically converges on the single-agent ShadowHand
    tasks; tuning may be needed for handover-specific contact dynamics.
    """

    isaacsim_physx = PhysxCfg(
        bounce_threshold_velocity=0.2,
        gpu_max_rigid_contact_count=2**23,
        gpu_max_rigid_patch_count=2**23,
    )
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=200,
            nconmax=70,
            impratio=10.0,
            cone="elliptic",
            update_data_interval=4,
            ccd_iterations=50,  # bumped from default 35 for multi-finger contact geometry
        ),
        # 4 substeps (vs reorient's 2): sustained ball-palm contact drives a small fraction of
        # envs to NaN at 2.
        num_substeps=4,
        debug_mode=False,
    )
    ovphysx = OvPhysxCfg()
    physx = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    default = newton_mjwarp


@configclass
class HandoverEnvCfg(DirectMARLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 7.5
    possible_agents = ["right_hand", "left_hand"]
    action_spaces = {"right_hand": 20, "left_hand": 20}
    observation_spaces = {"right_hand": 157, "left_hand": 157}
    state_space = 290

    # simulation — values mirrored by the manager cfg (guarded by the value-parity test)
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0),
        physics=PhysicsCfg(),
    )

    # robot
    right_robot_cfg: PresetCfg = RIGHT_HAND_CFG
    left_robot_cfg: PresetCfg = LEFT_HAND_CFG
    actuated_joint_names = ACTUATED_JOINT_NAMES
    fingertip_body_names = FINGERTIP_BODY_NAMES

    # in-hand object
    object_cfg: ObjectCfg = ObjectCfg()
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = GOAL_MARKER_CFG
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=1.5, replicate_physics=True)

    # reset
    reset_position_noise = 0.01  # range of position at reset
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset
    # scales and constants
    fall_dist = 0.24
    vel_obs_scale = 0.2
    act_moving_average = 1.0
    # success criteria
    success_distance_threshold: float = 0.1
    """Object-to-goal distance below which the handover is considered successful [m]."""
    # reward-related scales
    dist_reward_scale = 20.0
