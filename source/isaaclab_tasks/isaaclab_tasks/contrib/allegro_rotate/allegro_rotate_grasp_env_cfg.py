# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.utils.configclass import configclass

from .allegro_rotate_env_cfg import (
    CYLINDER_INIT_POS,
    CYLINDER_INIT_ROT,
    AllegroRotateEnvCfg,
    _allegro_hand_cfg,
    _cylinder_object_cfg,
)

GRASP_CYLINDER_INIT_POS = CYLINDER_INIT_POS


def _allegro_grasp_hand_cfg() -> ArticulationCfg:
    cfg = _allegro_hand_cfg()
    actuator = cfg.actuators["fingers"]
    actuator.effort_limit_sim = 2.0
    actuator.effort_limit = 2.0
    actuator.stiffness = 12.0
    actuator.damping = 1.0
    # Reset-time joint targets come from ALLEGRO_READY_JOINT_POS in cfg.
    return cfg


@configclass
class AllegroRotateGraspEnvCfg(AllegroRotateEnvCfg):
    """Grasp-cache generation stage for Allegro in-hand rotation."""

    scale_range = [0.8, 0.8, 1]
    require_grasp_cache = False
    grasp_cache_path = ""
    grasp_output_path = "source/isaaclab_tasks/isaaclab_tasks/contrib/allegro_rotate/cache/allegro_grasp_linspace"
    grasp_cache_target = 50000
    reset_dof_pos_noise = 0.15
    reset_position_noise = 0.0
    hand_init_usd_path = ""
    hand_init_usd_prim_path = ""
    hand_init_usd_object_prim_path = ""
    hand_init_usd_world_offset = (0.0, 0.0, 0.0)
    torque_control = False
    randomize_pd_gains = False
    randomize_friction = False
    randomize_com = False
    force_scale = 0.0
    gravity_curriculum = False
    grasp_gravity_switch_interval = 40
    binary_contact = False
    enable_contact_pos = False
    robot_cfg: ArticulationCfg = _allegro_grasp_hand_cfg()
    object_init_pos = GRASP_CYLINDER_INIT_POS
    object_cfg: RigidObjectCfg = _cylinder_object_cfg(object_init_pos, CYLINDER_INIT_ROT)
    object_init_from_fingertip_center = False
    object_fingertip_center_offset = (0.0, 0.0, 0.015)
    log_reset_fingertip_center = False
    object_init_from_pinch_center = False
    object_pinch_center_body_names = [
        "index_link_3",
        "middle_link_3",
        "ring_link_3",
        "thumb_link_3",
    ]
    # Disabled while using the configured cylinder seed; kept for quick
    # debugging if pinch-center placement is explicitly turned on.
    object_pinch_center_offset = (0.0, 0.0, 0.005)
    reset_height_lower = GRASP_CYLINDER_INIT_POS[2] - 0.015
    reset_height_upper = GRASP_CYLINDER_INIT_POS[2] + 0.015
    grasp_probe_init_pose = False
    grasp_probe_drop_hold_steps = 60
    joint_jitter_window_steps = 100
    joint_jitter_warn_threshold_deg = 0.5
    pinch_center_metric_threshold = 0.02

    grasp_contact_body_names = [
        "index_link_1",
        "index_link_2",
        "index_link_3",
        "middle_link_1",
        "middle_link_2",
        "middle_link_3",
        "ring_link_1",
        "ring_link_2",
        "ring_link_3",
        "thumb_link_1",
        "thumb_link_2",
        "thumb_link_3",
    ]
    grasp_fingertip_dist_threshold = 0.13
    grasp_min_near_fingers = 3
    grasp_contact_force_threshold = 0.005
    grasp_min_contact_bodies = 3
    grasp_min_contact_fingers = 3
    grasp_require_thumb_contact = True
    grasp_max_pinch_center_dist = 0.04
    grasp_max_object_pos_diff = 0.08
    grasp_reset_angle_diff = 30.0 / 180.0 * math.pi
    grasp_cache_status_interval = 40
