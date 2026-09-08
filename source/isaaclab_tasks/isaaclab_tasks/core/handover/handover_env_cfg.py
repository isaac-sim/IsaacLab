# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.core.handover.handover_common import GOAL_MARKER_CFG, OBJECT_RADIUS
from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.shadow_hand import (
    FINGERTIP_NAMES,
    JOINT_NAMES,
    SHADOW_HAND_NEWTON_CFG,
    SHADOW_HAND_PHYSX_CFG,
    TENDON_NAMES,
    TENDON_POSITION_LIMITS,
)


def _hand_cfg(
    base: ArticulationCfg,
    prim_path: str,
    init_pos: tuple[float, float, float],
    init_rot: tuple[float, float, float, float],
) -> ArticulationCfg:
    """Place one engine's Shadow Hand at this task's pose for one hand.

    The catch needs more joint authority than reorientation, but the hand's gains belong to the
    hand, so both tasks take them as the asset configuration supplies them. This task used to raise
    every actuator to stiffness 20 / damping 2, which also drove the tendon-coupled joints -- they
    take no position command, and MEASURED, giving them one costs the tendon most of its travel:
    11.1 rad falls to 1.0 rad.

    Args:
        base: The hand on the engine's asset variant.
        prim_path: Scene path the hand spawns at.
        init_pos: Spawn position [m].
        init_rot: Spawn orientation as ``(w, x, y, z)``.

    Returns:
        That configuration at *prim_path* with the given pose.
    """
    # The asset's own spawn rotation is shared by both engines, so the per-hand rotation COMPOSES
    # with it rather than replacing it -- replacing leaves both palms turned 90 degrees. See
    # SHADOW_HAND_PHYSX_CFG's init_state for why the asset carries that rotation.
    hand_rot = tuple(
        math_utils.quat_mul(
            torch.tensor(init_rot, dtype=torch.float64),
            torch.tensor(base.init_state.rot, dtype=torch.float64),
        ).tolist()
    )
    return base.replace(
        prim_path=prim_path,
        init_state=base.init_state.replace(pos=init_pos, rot=hand_rot),
    )


# Per-hand poses. The rotations are composed with the asset's own; they are unchanged from the
# previous Newton asset, which the two assets being identical geometry makes valid.
_RIGHT_POSE = ("{ENV_REGEX_NS}/RightRobot", (0.0, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0))
_LEFT_POSE = ("{ENV_REGEX_NS}/LeftRobot", (0.0, -1.0, 0.5), (0.0, 0.0, 1.0, 0.0))


@configclass
class RightHandCfg(PresetCfg):
    """The right hand on every engine; only the asset's physics variant differs."""

    newton_mjwarp = _hand_cfg(SHADOW_HAND_NEWTON_CFG, *_RIGHT_POSE)
    isaacsim_physx = _hand_cfg(SHADOW_HAND_PHYSX_CFG, *_RIGHT_POSE)
    physx = isaacsim_physx
    ovphysx = isaacsim_physx
    default = newton_mjwarp


@configclass
class LeftHandCfg(PresetCfg):
    """The left hand on every engine; only the asset's physics variant differs."""

    newton_mjwarp = _hand_cfg(SHADOW_HAND_NEWTON_CFG, *_LEFT_POSE)
    isaacsim_physx = _hand_cfg(SHADOW_HAND_PHYSX_CFG, *_LEFT_POSE)
    physx = isaacsim_physx
    ovphysx = isaacsim_physx
    default = newton_mjwarp


BALL_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/object",
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
"""Hand-over ball, thrown from one Shadow hand to the other."""


@configclass
class PhysicsCfg(PresetCfg):
    """Physics-backend preset (PhysX vs Newton/MJWarp).

    Newton mirrors the single-agent Shadow Hand Newton port: an elliptic friction
    cone with ``impratio=10``, which weights normal contacts over friction, 100
    solver iterations and 2 substeps.
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

    # simulation — values mirrored by the manager cfg
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0),
        physics=PhysicsCfg(),
        # Frame both hands and the object between them. Without this the visualizer looks at the
        # origin from its default 4 m away, which renders the pair a few pixels wide.
        default_visualizer_cfg=VisualizerCfg(eye=(1.15, -1.65, 1.15), lookat=(0.0, -0.5, 0.55), focal_length=35.0),
    )

    # robot
    right_robot_cfg: RightHandCfg = RightHandCfg()
    left_robot_cfg: LeftHandCfg = LeftHandCfg()
    actuated_joint_names = JOINT_NAMES
    actuated_tendon_names = TENDON_NAMES
    actuated_tendon_position_limits = TENDON_POSITION_LIMITS
    fingertip_body_names = FINGERTIP_NAMES

    # in-hand object
    object_cfg: RigidObjectCfg = BALL_CFG
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
