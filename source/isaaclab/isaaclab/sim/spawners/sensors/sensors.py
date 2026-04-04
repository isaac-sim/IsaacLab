# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pxr import Sdf, Usd

from isaaclab.sim.utils import change_prim_property, clone, create_prim, get_current_stage
from isaaclab.utils import to_camel_case

if TYPE_CHECKING:
    from . import sensors_cfg

# import logger
logger = logging.getLogger(__name__)

CUSTOM_PINHOLE_CAMERA_ATTRIBUTES = {
    "projection_type": ("cameraProjectionType", Sdf.ValueTypeNames.Token),
}
"""Custom attributes for pinhole camera model.

The dictionary maps the attribute name in the configuration to the attribute name in the USD prim.
"""


CUSTOM_FISHEYE_CAMERA_ATTRIBUTES = {
    "projection_type": ("cameraProjectionType", Sdf.ValueTypeNames.Token),
    "fisheye_nominal_width": ("fthetaWidth", Sdf.ValueTypeNames.Float),
    "fisheye_nominal_height": ("fthetaHeight", Sdf.ValueTypeNames.Float),
    "fisheye_optical_centre_x": ("fthetaCx", Sdf.ValueTypeNames.Float),
    "fisheye_optical_centre_y": ("fthetaCy", Sdf.ValueTypeNames.Float),
    "fisheye_max_fov": ("fthetaMaxFov", Sdf.ValueTypeNames.Float),
    "fisheye_polynomial_a": ("fthetaPolyA", Sdf.ValueTypeNames.Float),
    "fisheye_polynomial_b": ("fthetaPolyB", Sdf.ValueTypeNames.Float),
    "fisheye_polynomial_c": ("fthetaPolyC", Sdf.ValueTypeNames.Float),
    "fisheye_polynomial_d": ("fthetaPolyD", Sdf.ValueTypeNames.Float),
    "fisheye_polynomial_e": ("fthetaPolyE", Sdf.ValueTypeNames.Float),
    "fisheye_polynomial_f": ("fthetaPolyF", Sdf.ValueTypeNames.Float),
}
"""Custom attributes for fisheye camera model.

The dictionary maps the attribute name in the configuration to the attribute name in the USD prim.
"""


@clone
def spawn_camera(
    prim_path: str,
    cfg: sensors_cfg.PinholeCameraCfg | sensors_cfg.FisheyeCameraCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a USD camera prim with given projection type.

    The function creates various attributes on the camera prim that specify the camera's properties.
    These are later used by ``omni.replicator.core`` to render the scene with the given camera.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which case
            this is set to the origin.
        orientation: The orientation in (x, y, z, w) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The created prim.

    Raises:
        ValueError: If a prim already exists at the given path.
    """
    # obtain stage handle
    stage = get_current_stage()

    # spawn camera if it doesn't exist.
    if not stage.GetPrimAtPath(prim_path).IsValid():
        create_prim(prim_path, "Camera", translation=translation, orientation=orientation, stage=stage)
    else:
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    # lock camera from viewport (this disables viewport movement for camera)
    if cfg.lock_camera:
        change_prim_property(
            prop_path=f"{prim_path}.omni:kit:cameraLock",
            value=True,
            stage=stage,
            type_to_create_if_not_exist=Sdf.ValueTypeNames.Bool,
        )
    # decide the custom attributes to add
    if cfg.projection_type == "pinhole":
        attribute_types = CUSTOM_PINHOLE_CAMERA_ATTRIBUTES
    else:
        attribute_types = CUSTOM_FISHEYE_CAMERA_ATTRIBUTES

    # TODO: Adjust to handle aperture offsets once supported by omniverse
    #   Internal ticket from rendering team: OM-42611
    if cfg.horizontal_aperture_offset > 1e-4 or cfg.vertical_aperture_offset > 1e-4:
        logger.warning("Camera aperture offsets are not supported by Omniverse. These parameters will be ignored.")

    # custom attributes in the config that are not USD Camera parameters
    non_usd_cfg_param_names = [
        "func",
        "copy_from_source",
        "lock_camera",
        "visible",
        "semantic_tags",
        "from_intrinsic_matrix",
        "spawn_path",
    ]
    # get camera prim
    prim = stage.GetPrimAtPath(prim_path)
    # create attributes for the fisheye camera model
    # note: for pinhole those are already part of the USD camera prim
    for attr_name, attr_type in attribute_types.values():
        # check if attribute does not exist
        if prim.GetAttribute(attr_name).Get() is None:
            # create attribute based on type
            prim.CreateAttribute(attr_name, attr_type)
    # set attribute values
    for param_name, param_value in cfg.__dict__.items():
        # check if value is valid
        if param_value is None or param_name in non_usd_cfg_param_names:
            continue
        # obtain prim property name
        if param_name in attribute_types:
            # check custom attributes
            prim_prop_name = attribute_types[param_name][0]
        else:
            # convert attribute name in prim to cfg name
            prim_prop_name = to_camel_case(param_name, to="cC")
        # get attribute from the class
        prim.GetAttribute(prim_prop_name).Set(param_value)
    # return the prim
    return prim
