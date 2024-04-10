# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import omni.isaac.core.utils.prims as prim_utils
from pxr import Gf, Usd, UsdPhysics
from warp.utils import quat_rpy

from omni.isaac.orbit.sim import schemas
from omni.isaac.orbit.sim.utils import clone, safe_set_attribute_on_usd_schema

if TYPE_CHECKING:
    from . import race_quadcopter_cfg


@clone
def spawn_race_quadcopter(
    prim_path: str,
    cfg: race_quadcopter_cfg.RaceQuadcopterCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    # body frame FLU-xyz
    prim_utils.create_prim(
        prim_path,
        prim_type="Xform",
        translation=translation,
        orientation=orientation,
    )

    # visual shapes
    visual_prim_path = prim_path + "/visual"
    prim_utils.create_prim(visual_prim_path, prim_type="Xform")

    # central body cube
    prim_utils.create_prim(
        visual_prim_path + "/central_body",
        prim_type="Cube",
        attributes={"size": 1.0},
        scale=(
            cfg.central_body_length_x,
            cfg.central_body_length_y,
            cfg.central_body_length_z,
        ),
        translation=(
            cfg.central_body_center_x,
            cfg.central_body_center_y,
            cfg.central_body_center_z,
        ),
    )

    # arms
    arm_move_length = (cfg.arm_length_front + cfg.arm_length_rear) / 2 - cfg.arm_length_rear

    # arms 1, 4
    qx, qy, qz, qw = quat_rpy(0.0, 0.0, cfg.arm_front_angle / 2)
    prim_utils.create_prim(
        visual_prim_path + "/arm_1_4",
        prim_type="Cube",
        attributes={"size": 1.0},
        scale=(
            cfg.arm_length_front + cfg.arm_length_rear,
            cfg.motor_diameter,
            cfg.arm_thickness,
        ),
        orientation=(qw, qx, qy, qz),
        translation=(
            math.cos(cfg.arm_front_angle / 2) * arm_move_length,
            math.sin(cfg.arm_front_angle / 2) * arm_move_length,
            -cfg.arm_thickness / 2,
        ),
    )

    # arms 2, 3
    qx, qy, qz, qw = quat_rpy(0.0, 0.0, -cfg.arm_front_angle / 2)
    prim_utils.create_prim(
        visual_prim_path + "/arm_2_3",
        prim_type="Cube",
        attributes={"size": 1.0},
        scale=(
            cfg.arm_length_front + cfg.arm_length_rear,
            cfg.motor_diameter,
            cfg.arm_thickness,
        ),
        orientation=(qw, qx, qy, qz),
        translation=(
            math.cos(cfg.arm_front_angle / 2) * arm_move_length,
            -math.sin(cfg.arm_front_angle / 2) * arm_move_length,
            -cfg.arm_thickness / 2,
        ),
    )

    # rotors
    rotor_angles = [
        cfg.arm_front_angle / 2 + math.pi,
        -cfg.arm_front_angle / 2,
        -cfg.arm_front_angle / 2 + math.pi,
        cfg.arm_front_angle / 2,
    ]
    for i in [1, 2, 3, 4]:
        arm_length = None
        if i == 1 or i == 3:
            arm_length = cfg.arm_length_rear
        else:
            arm_length = cfg.arm_length_front
        prim_utils.create_prim(
            visual_prim_path + "/motor_" + str(i),
            prim_type="Cylinder",
            attributes={
                "radius": cfg.motor_diameter / 2,
                "height": cfg.motor_height + cfg.arm_thickness,
            },
            translation=(
                math.cos(rotor_angles[i - 1]) * arm_length,
                math.sin(rotor_angles[i - 1]) * arm_length,
                (cfg.motor_height + cfg.arm_thickness) / 2 - cfg.arm_thickness,
            ),
        )
    for i in [1, 2, 3, 4]:
        arm_length = None
        if i == 1 or i == 3:
            arm_length = cfg.arm_length_rear
        else:
            arm_length = cfg.arm_length_front
        prim_utils.create_prim(
            visual_prim_path + "/propeller_" + str(i),
            prim_type="Cylinder",
            attributes={
                "radius": cfg.propeller_diameter / 2,
                "height": cfg.propeller_height,
            },
            translation=(
                math.cos(rotor_angles[i - 1]) * arm_length,
                math.sin(rotor_angles[i - 1]) * arm_length,
                cfg.propeller_height / 2 + cfg.motor_height,
            ),
        )

    # collision box
    collision_prim_path = prim_path + "/collision"
    prim_utils.create_prim(collision_prim_path, prim_type="Xform")

    # collision box size
    positive_x_extend = max(
        cfg.central_body_center_x + cfg.central_body_length_x / 2,
        cfg.arm_length_front * math.cos(cfg.arm_front_angle / 2) + cfg.propeller_diameter / 2,
    )
    negative_x_extend = -min(
        cfg.central_body_center_x - cfg.central_body_length_x / 2,
        cfg.arm_length_rear * math.cos(cfg.arm_front_angle / 2 + math.pi) - cfg.propeller_diameter / 2,
    )
    collision_bbox_length_x = positive_x_extend + negative_x_extend
    collision_bbox_center_x = -negative_x_extend + collision_bbox_length_x / 2

    positive_y_extend = max(
        cfg.central_body_center_y + cfg.central_body_length_y / 2,
        cfg.arm_length_front * math.sin(cfg.arm_front_angle / 2) + cfg.propeller_diameter / 2,
        cfg.arm_length_rear * math.sin(math.pi - cfg.arm_front_angle / 2) + cfg.propeller_diameter / 2,
    )
    collision_bbox_length_y = 2 * positive_y_extend
    collision_bbox_center_y = 0.0

    positive_z_extend = max(
        cfg.central_body_center_z + cfg.central_body_length_z / 2,
        cfg.motor_height + cfg.propeller_height,
    )
    negative_z_extend = -min(cfg.central_body_center_z - cfg.central_body_length_z / 2, -cfg.arm_thickness)
    collision_bbox_length_z = positive_z_extend + negative_z_extend
    collision_bbox_center_z = -negative_z_extend + collision_bbox_length_z / 2

    prim_utils.create_prim(
        collision_prim_path + "/bbox",
        prim_type="Cube",
        attributes={"size": 1.0, "purpose": "guide"},
        scale=(
            collision_bbox_length_x,
            collision_bbox_length_y,
            collision_bbox_length_z,
        ),
        translation=(
            collision_bbox_center_x,
            collision_bbox_center_y,
            collision_bbox_center_z,
        ),
    )
    schemas.define_collision_properties(collision_prim_path + "/bbox", cfg.collision_props)

    # other properties
    if cfg.mass_props is not None:
        schemas.define_mass_properties(prim_path, cfg.mass_props)
    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props)

    # additional mass properties
    mass_api = UsdPhysics.MassAPI(prim_utils.get_prim_at_path(prim_path))
    safe_set_attribute_on_usd_schema(mass_api, "center_of_mass", cfg.center_of_mass, True)
    safe_set_attribute_on_usd_schema(mass_api, "diagonal_inertia", cfg.diagonal_inertia, True)
    safe_set_attribute_on_usd_schema(
        mass_api,
        "principal_axes",
        Gf.Quatf(
            cfg.principal_axes_rotation[0],
            cfg.principal_axes_rotation[1],
            cfg.principal_axes_rotation[2],
            cfg.principal_axes_rotation[3],
        ),
        True,
    )

    # instance-able
    visual = prim_utils.get_prim_at_path(visual_prim_path)
    collision = prim_utils.get_prim_at_path(collision_prim_path)
    visual.SetInstanceable(True)
    collision.SetInstanceable(True)

    return prim_utils.get_prim_at_path(prim_path)
