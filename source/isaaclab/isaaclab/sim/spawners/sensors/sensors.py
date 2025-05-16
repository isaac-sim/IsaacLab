# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import isaacsim.core.utils.prims as prim_utils
import omni.kit.commands
import omni.log
from pxr import Gf, Sdf, Usd

from isaaclab.sim.utils import clone
from isaaclab.utils import to_camel_case

if TYPE_CHECKING:
    from . import sensors_cfg


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
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.

    Returns:
        The created prim.

    Raises:
        ValueError: If a prim already exists at the given path.
    """
    # spawn camera if it doesn't exist.
    if not prim_utils.is_prim_path_valid(prim_path):
        prim_utils.create_prim(prim_path, "Camera", translation=translation, orientation=orientation)
    else:
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    # lock camera from viewport (this disables viewport movement for camera)
    if cfg.lock_camera:
        omni.kit.commands.execute(
            "ChangePropertyCommand",
            prop_path=Sdf.Path(f"{prim_path}.omni:kit:cameraLock"),
            value=True,
            prev=None,
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
        omni.log.warn("Camera aperture offsets are not supported by Omniverse. These parameters will be ignored.")

    # custom attributes in the config that are not USD Camera parameters
    non_usd_cfg_param_names = [
        "func",
        "copy_from_source",
        "lock_camera",
        "visible",
        "semantic_tags",
        "from_intrinsic_matrix",
    ]
    # get camera prim
    prim = prim_utils.get_prim_at_path(prim_path)
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
    return prim_utils.get_prim_at_path(prim_path)


@clone
def spawn_lidar(
    prim_path: str,
    cfg: sensors_cfg.LidarCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> Usd.Prim:
    """Create a USD Camera prim used for RTX Lidar models.
    This function creates an RTX lidar model attached to a USD Camera prim. The RTX lidar model is configured from json
    files located within ``omni.isaac.core``.
    Custom configurations for the RTX lidar are passed in via the :class:`LidarCfg`. By setting the `lidar_type` to
    custom and providing a `sensor_profile` dictionary a json configuration file will be created and passed to the
    kit commands responsible for the RTX lidar creation.
    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which case
            this is set to the origin.
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.
    Returns:
        The created prim.
    Raises:
        ValueError: If the LidarCfg.sensor_profile is None when LidarCfg.lidar_type == "Custom"
        RuntimeError: If the creation of the RTXLidar fails
        ValueError: If a prim already exists at the given path.
    """

    if cfg.lidar_type == "Custom":
        if cfg.sensor_profile is None:
            raise ValueError("LidarCfg sensor_profile cannot be none for lidar_type: Custom")

        # make directories
        if not os.path.isdir(cfg.sensor_profile_temp_dir):
            os.makedirs(cfg.sensor_profile_temp_dir)

        # create file path
        file_name = cfg.sensor_profile_temp_prefix + ".json"
        file_path = os.path.join(cfg.sensor_profile_temp_dir, file_name)

        # Check for tempfiles and remove
        while os.path.isfile(file_path):
            os.remove(file_path)

        # Write to file
        with open(file_path, "w") as outfile:
            json.dump(cfg.sensor_profile, outfile)

        print("Custom: ")
        config = file_path.split("/")[-1].split(".")[0]
        print(file_path)
    else:
        config = cfg.lidar_type

    if not prim_utils.is_prim_path_valid(prim_path):
        # prim_utils.create_prim(prim_path, "Camera", translation=translation, orientation=orientation)
        _, sensor = omni.kit.commands.execute(
            "IsaacSensorCreateRtxLidar",
            path=prim_path,
            parent=None,
            config=config,
            translation=translation,
            orientation=Gf.Quatd(orientation[0], orientation[1], orientation[2], orientation[3]),
        )
        if not sensor.IsValid():
            raise RuntimeError("IsaacSensorCreateRtxLidar failed to create a USD.Prim")

    else:
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    return sensor
