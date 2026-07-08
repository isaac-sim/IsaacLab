# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG
from isaaclab_tasks.utils import PresetCfg

ALLEGRO_ROTATE_DIR = Path(__file__).resolve().parent
ALLEGRO_ROTATE_ASSET_DIR = ALLEGRO_ROTATE_DIR / "assets"
ALLEGRO_HAND_USD_PATH = str(ALLEGRO_ROTATE_ASSET_DIR / "allegro_hand_inst.usd")
# Optional pose-authoring USD path. Disabled for submission because the tuned
# ready pose is copied into cfg constants below.
# ALLEGRO_GRASP_REFERENCE_USD_PATH = "<pose_authoring_usd>"
ALLEGRO_GRASP_REFERENCE_USD_PATH = ""
ALLEGRO_GRASP_REFERENCE_PRIM_PATH = "/allegro"
ALLEGRO_GRASP_OBJECT_REFERENCE_PRIM_PATH = "/allegro/cylinder/object/Cylinder"
ALLEGRO_CYLINDER_USD_PATH = str(ALLEGRO_ROTATE_ASSET_DIR / "cylinder_nv_logo.usd")
# This task is an in-hand rolling baseline, not the table/nut-thread task.
# The object is cradled by the palm/base geometry while several fingertips
# maintain side contact and roll it.
# These ready-pose values were copied from the tuned Allegro hold USD so the
# submitted task does not depend on loading a separate pose-authoring file.
HAND_ROOT_POS = (0.0, 0.0, 0.5)
HAND_ROOT_ROT = (-0.065605408, 0.550657485, -0.451303169, 0.699140697)
CYLINDER_INIT_ROT = (0.0, 0.0, 0.0, 1.0)
CYLINDER_INIT_POS = (0.043066784, -0.058802739, 0.690463225)
ALLEGRO_READY_JOINT_POS = {
    "index_joint_0": 0.171042270,
    "index_joint_1": 1.398008704,
    "index_joint_2": 0.261799388,
    "index_joint_3": -0.130899694,
    "middle_joint_0": -0.057595864,
    "middle_joint_1": 0.982620356,
    "middle_joint_2": 0.432841674,
    "middle_joint_3": 0.024434609,
    "ring_joint_0": 0.141371676,
    "ring_joint_1": 1.469567310,
    "ring_joint_2": 0.150098322,
    "ring_joint_3": 0.008726646,
    "thumb_joint_0": 1.033234930,
    "thumb_joint_1": 0.907571211,
    "thumb_joint_2": 0.652753167,
    "thumb_joint_3": 0.073303834,
}

# Legacy placement helpers kept only for debugging placement regressions. The
# normal Allegro grasp/cache path uses the explicit ready-pose constants above.
ALLEGRO_PALM_LINK_LOCAL_POS = (-0.00822, -0.02063, 0.08086)
ALLEGRO_INDEX_BASE_LOCAL_POS = (0.02545, -0.03832, 0.13229)
ALLEGRO_MIDDLE_BASE_LOCAL_POS = (-0.00644, -0.00643, 0.13460)
ALLEGRO_RING_BASE_LOCAL_POS = (-0.03832, 0.02546, 0.13229)
ALLEGRO_INDEX_FINGERTIP_LOCAL_POS = (0.03116, -0.04402, 0.22434)
ALLEGRO_MIDDLE_FINGERTIP_LOCAL_POS = (-0.00642, -0.00644, 0.22700)
ALLEGRO_RING_FINGERTIP_LOCAL_POS = (-0.04401, 0.03114, 0.22434)
ALLEGRO_THUMB_FINGERTIP_LOCAL_POS = (0.09008, -0.08429, 0.03480)
ALLEGRO_FINGERTIP_CAGE_WORLD_Z_LIFT = 0.075
ALLEGRO_CRADLE_OBJECT_WORLD_Z_LIFT = 0.055


def _quat_xyzw_to_matrix(quat: tuple[float, float, float, float]) -> tuple[tuple[float, float, float], ...]:
    x, y, z, w = quat
    norm = (x * x + y * y + z * z + w * w) ** 0.5
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )


def _transform_hand_local_to_world(local_pos: tuple[float, float, float]) -> tuple[float, float, float]:
    rot = _quat_xyzw_to_matrix(HAND_ROOT_ROT)
    return tuple(
        HAND_ROOT_POS[row] + sum(rot[row][col] * local_pos[col] for col in range(3)) for row in range(3)
    )


def _mean_pos(positions: tuple[tuple[float, float, float], ...]) -> tuple[float, float, float]:
    return tuple(sum(pos[axis] for pos in positions) / len(positions) for axis in range(3))


def _allegro_fingertip_cage_object_init_pos() -> tuple[float, float, float]:
    fingertip_world_positions = tuple(
        _transform_hand_local_to_world(pos)
        for pos in (
            ALLEGRO_INDEX_FINGERTIP_LOCAL_POS,
            ALLEGRO_MIDDLE_FINGERTIP_LOCAL_POS,
            ALLEGRO_RING_FINGERTIP_LOCAL_POS,
            ALLEGRO_THUMB_FINGERTIP_LOCAL_POS,
        )
    )
    fingertip_center_world = _mean_pos(fingertip_world_positions)
    return (
        fingertip_center_world[0],
        fingertip_center_world[1],
        max(pos[2] for pos in fingertip_world_positions) + ALLEGRO_FINGERTIP_CAGE_WORLD_Z_LIFT,
    )


def _allegro_cradle_object_init_pos() -> tuple[float, float, float]:
    """Legacy palm/base seed kept for debugging placement regressions."""
    palm_world = _transform_hand_local_to_world(ALLEGRO_PALM_LINK_LOCAL_POS)
    finger_base_world = _transform_hand_local_to_world(
        _mean_pos(
            (
                ALLEGRO_INDEX_BASE_LOCAL_POS,
                ALLEGRO_MIDDLE_BASE_LOCAL_POS,
                ALLEGRO_RING_BASE_LOCAL_POS,
            )
        )
    )
    finger_weight = 0.70
    palm_weight = 1.0 - finger_weight
    return (
        palm_weight * palm_world[0] + finger_weight * finger_base_world[0],
        palm_weight * palm_world[1] + finger_weight * finger_base_world[1],
        max(palm_world[2], finger_base_world[2]) + ALLEGRO_CRADLE_OBJECT_WORLD_Z_LIFT,
    )


def allegro_grasp_cache_path(cache_prefix: str, scale_range: list[float]) -> str:
    """Return the Allegro grasp-cache path for a prefix and scale range."""
    if cache_prefix.endswith(".npy"):
        return cache_prefix
    return f"{cache_prefix}_{scale_range[0]}-{scale_range[1]}-{scale_range[2]}.npy"


def _allegro_hand_cfg() -> ArticulationCfg:
    cfg = ALLEGRO_HAND_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=HAND_ROOT_POS,
            rot=HAND_ROOT_ROT,
            joint_pos=ALLEGRO_READY_JOINT_POS.copy(),
        ),
    )
    cfg.spawn.usd_path = ALLEGRO_HAND_USD_PATH
    cfg.spawn.activate_contact_sensors = True
    return cfg


def _cylinder_object_cfg(pos=CYLINDER_INIT_POS, rot=CYLINDER_INIT_ROT) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ALLEGRO_CYLINDER_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                linear_damping=0.02,
                angular_damping=0.001,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
                sleep_threshold=0.0,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            scale=(0.8, 0.8, 0.8),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.004, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
    )


@configclass
class PhysicsCfg(PresetCfg):
    physx = PhysxCfg(
        bounce_threshold_velocity=0.2,
        gpu_collision_stack_size=2**28,
        gpu_found_lost_pairs_capacity=2**23,
        gpu_found_lost_aggregate_pairs_capacity=2**23,
        gpu_total_aggregate_pairs_capacity=2**23,
    )
    newton = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=80,
            nconmax=70,
            impratio=10.0,
            cone="elliptic",
            update_data_interval=2,
            iterations=100,
        ),
        num_substeps=2,
        debug_mode=False,
    )
    default = physx


@configclass
class AllegroRotateEnvCfg(DirectRLEnvCfg):
    """Milestone A: Allegro rolls a free rigid cylinder in a palm-supported hand cradle."""

    # env
    decimation = 4
    episode_length_s = 20.0
    action_space = 16
    # 3 frames of joint_pos + joint_target history: 16 * 2 * 3 = 96.
    # Current sensor/object state: fingertip rel pos 12 + contact force 4
    # + object pos error 3 + linvel 3 + angvel 3 = 25.
    observation_space = 121
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        gravity=(0.0, 0.0, -0.05),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.2,
            dynamic_friction=1.0,
        ),
        physics=PhysicsCfg(),
    )

    # robot
    robot_cfg: ArticulationCfg = _allegro_hand_cfg()

    actuated_joint_names = [
        "index_joint_0",
        "middle_joint_0",
        "ring_joint_0",
        "thumb_joint_0",
        "index_joint_1",
        "index_joint_2",
        "index_joint_3",
        "middle_joint_1",
        "middle_joint_2",
        "middle_joint_3",
        "ring_joint_1",
        "ring_joint_2",
        "ring_joint_3",
        "thumb_joint_1",
        "thumb_joint_2",
        "thumb_joint_3",
    ]
    fingertip_body_names = [
        "index_link_3",
        "middle_link_3",
        "ring_link_3",
        "thumb_link_3",
    ]
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*_link_[123]",
        history_length=3,
        update_period=0.0,
    )

    # free cylinder object
    object_init_pos = CYLINDER_INIT_POS
    object_init_rot = CYLINDER_INIT_ROT
    object_cfg: RigidObjectCfg = _cylinder_object_cfg(object_init_pos, object_init_rot)
    object_init_from_fingertip_center = False
    object_fingertip_center_offset = (0.0, 0.0, 0.0)
    log_reset_fingertip_center = False
    reset_fingertip_center_log_interval = 40
    object_init_from_pinch_center = False
    object_pinch_center_body_names = [
        "index_link_3",
        "middle_link_3",
        "ring_link_3",
        "thumb_link_3",
    ]
    object_pinch_center_offset = (0.0, 0.0, 0.0)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=0.75, replicate_physics=True, clone_in_fabric=True
    )

    # observation/action
    history_len = 3
    action_scale = 0.035
    fingertip_rel_pos_obs_scale = 10.0
    contact_obs_force_scale = 0.2
    object_linvel_obs_scale = 1.0
    object_angvel_obs_scale = 0.25
    reset_dof_pos_noise = 0.02
    reset_position_noise = 0.002
    # Lower the configured cylinder seed into the fingers for non-cache
    # probe/cache generation.
    object_reset_pos_offset = (0.0, 0.0, 0.0)
    object_reset_z_offset = -0.02
    # Cache rows are saved after the grasp generator has already settled the
    # object. Do not shift cached object poses again unless explicitly debugging.
    cache_object_reset_pos_offset = (0.0, 0.0, 0.0)
    cache_object_reset_z_offset = 0.0
    # Ready pose is submitted as cfg constants above. Keep the old USD read
    # path documented but disabled so local pose-authoring files are optional.
    # hand_init_usd_path = ALLEGRO_GRASP_REFERENCE_USD_PATH
    # hand_init_usd_prim_path = ALLEGRO_GRASP_REFERENCE_PRIM_PATH
    # hand_init_usd_object_prim_path = ALLEGRO_GRASP_OBJECT_REFERENCE_PRIM_PATH
    hand_init_usd_path = ""
    hand_init_usd_prim_path = ""
    hand_init_usd_object_prim_path = ""
    hand_init_usd_world_offset = (0.0, 0.0, 0.0)
    # Rotate training starts from saved stable grasp states. Keep the task
    # strict so a missing cache is not silently hidden.
    scale_range = [0.8, 0.8, 1]
    grasp_cache_path = "cache/allegro_grasp_linspace"
    require_grasp_cache = True

    # reward
    target_axis = (0.0, 0.0, 1.0)
    angvel_clip_min = -0.5
    angvel_clip_max = 0.5
    rotate_reward_scale = 2.5
    object_linvel_penalty_scale = -0.3
    off_axis_angvel_penalty_scale = 0.0
    pos_diff_penalty_scale = -0.4
    torque_penalty_scale = -0.1
    work_penalty_scale = -0.5
    object_pos_reward_scale = 0.003
    proximity_std = 0.08
    contact_gate_dist = 0.12
    contact_gate_width = 0.06
    contact_force_threshold = 0.05
    contact_force_width = 0.25
    side_wall_z_std = 0.035
    # Contact metrics are logged for diagnosis, but the training reward remains
    # rotation-focused: no explicit finger/contact shaping.
    rotate_support_gate_floor = 0.30
    pinch_center_reward_std = 0.06
    thumb_escape_dist = 0.11
    thumb_escape_width = 0.04
    min_train_contact_count = 4
    success_rotation_count = 0.25
    gravity_curriculum = True
    gravity_curriculum_start_step = 1000
    gravity_curriculum_height_reset_threshold = 0.02
    gravity_curriculum_increment = 0.05
    gravity_curriculum_max = 10.0

    # termination
    drop_dist = 0.12
    min_object_height = 0.50
    max_object_height = 0.62
    reset_height_lower = 0.52
    reset_height_upper = 0.60
