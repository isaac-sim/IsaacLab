# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectMARLEnvCfg
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


def _shadow_hand_cfg(
    prim_path: str,
    init_pos: tuple[float, float, float],
    init_rot: tuple[float, float, float, float],
) -> PresetCfg:
    """Build the per-hand Shadow Hand preset for each supported backend.

    Args:
        prim_path: Scene path the hand spawns at.
        init_pos: Spawn position [m].
        init_rot: Spawn orientation as ``(w, x, y, z)``.

    Returns:
        A preset carrying the PhysX, Newton MJWarp and OvPhysX variants, each at
        *prim_path* with the given pose. The two hands differ only in these arguments.
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
    prim_path="{ENV_REGEX_NS}/RightRobot",
    init_pos=(0.0, 0.0, 0.5),
    init_rot=(0.0, 0.0, 0.0, 1.0),
)
LEFT_HAND_CFG = _shadow_hand_cfg(
    prim_path="{ENV_REGEX_NS}/LeftRobot",
    init_pos=(0.0, -1.0, 0.5),
    init_rot=(0.0, 0.0, 1.0, 0.0),
)


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
    )

    # robot
    right_robot_cfg: PresetCfg = RIGHT_HAND_CFG
    left_robot_cfg: PresetCfg = LEFT_HAND_CFG
    actuated_joint_names = ACTUATED_JOINT_NAMES
    fingertip_body_names = FINGERTIP_BODY_NAMES

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
