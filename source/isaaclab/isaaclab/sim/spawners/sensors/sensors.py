# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pxr import Gf, Sdf, Usd

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


# OpenCV lens-distortion models authored as the ``omni:lensdistortion:*`` USD API. The RTX/OVRTX
# renderer honors these attributes natively; they are read back into ``camera.data.intrinsic_matrices``
# by :meth:`~isaaclab.sensors.camera.Camera._update_intrinsic_matrices`.
_OPENCV_DISTORTION_API_SCHEMAS = {
    "opencvPinhole": "OmniLensDistortionOpenCvPinholeAPI",
    "opencvFisheye": "OmniLensDistortionOpenCvFisheyeAPI",
}
"""Maps an OpenCV distortion model discriminator to its applied USD API schema name."""

_OPENCV_DISTORTION_COEFFS = {
    "opencvPinhole": ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2", "s1", "s2", "s3", "s4"),
    "opencvFisheye": ("k1", "k2", "k3", "k4"),
}
"""Maps an OpenCV distortion model discriminator to its distortion-coefficient field names."""


def _author_opencv_distortion(prim: Usd.Prim, cfg: sensors_cfg.OpenCvDistortionCfg) -> None:
    """Author an OpenCV lens-distortion model on a camera prim as the ``omni:lensdistortion:*`` API.

    The attributes are authored explicitly (not through the generic camelCase loop of
    :func:`spawn_camera`) because their names are namespaced (e.g. ``omni:lensdistortion:opencvPinhole:k1``)
    and cannot be produced by :func:`~isaaclab.utils.to_camel_case`. Applying the schema only edits prim
    metadata, so it survives ``stage.ExportToString()`` and does not require the schema to be registered.

    Args:
        prim: The camera prim to author the distortion model on.
        cfg: The OpenCV distortion configuration.

    Raises:
        ValueError: If the distortion ``model`` is not a supported OpenCV model.
    """
    if cfg.model not in _OPENCV_DISTORTION_API_SCHEMAS:
        raise ValueError(
            f"Unsupported OpenCV distortion model: '{cfg.model}'. Supported models are:"
            f" {list(_OPENCV_DISTORTION_API_SCHEMAS)}."
        )
    prefix = f"omni:lensdistortion:{cfg.model}"

    # apply the schema and set the model discriminator token
    prim.AddAppliedSchema(_OPENCV_DISTORTION_API_SCHEMAS[cfg.model])

    def _set_attr(name: str, type_name: Sdf.ValueTypeName, value) -> None:
        attr = prim.GetAttribute(name) or prim.CreateAttribute(name, type_name)
        attr.Set(value)

    _set_attr("omni:lensdistortion:model", Sdf.ValueTypeNames.Token, cfg.model)
    _set_attr(
        f"{prefix}:imageSize",
        Sdf.ValueTypeNames.Int2,
        Gf.Vec2i(int(cfg.image_size[0]), int(cfg.image_size[1])),
    )
    for name in ("fx", "fy", "cx", "cy"):
        _set_attr(f"{prefix}:{name}", Sdf.ValueTypeNames.Float, float(getattr(cfg, name)))
    # coefficients are muted (authored as zero) unless apply_lens_distortion is set
    for name in _OPENCV_DISTORTION_COEFFS[cfg.model]:
        value = float(getattr(cfg, name)) if cfg.apply_lens_distortion else 0.0
        _set_attr(f"{prefix}:{name}", Sdf.ValueTypeNames.Float, value)


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
        "distortion",
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
    # author the OpenCV lens-distortion model (renderer-agnostic; RTX/OVRTX honors it natively)
    if cfg.distortion is not None:
        _author_opencv_distortion(prim, cfg.distortion)
    # return the prim
    return prim


@clone
def spawn_sensor_frame(
    prim_path: str,
    cfg: sensors_cfg.SensorFrameCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a plain USD Xform prim as a sensor attachment frame.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern.

    Args:
        prim_path: The prim path or pattern to spawn the asset at.
        cfg: The configuration instance.
        translation: Local translation (x, y, z) [m] w.r.t. the parent prim. Defaults to None
            (origin).
        orientation: Local orientation as quaternion (x, y, z, w) w.r.t. the parent prim.
            Defaults to None (identity).
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The created USD prim.

    Raises:
        ValueError: If a prim already exists at the given path.
    """
    stage = get_current_stage()
    if not stage.GetPrimAtPath(prim_path).IsValid():
        prim = create_prim(
            prim_path,
            "Xform",
            translation=translation,
            orientation=orientation,
            stage=stage,
        )
    else:
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")
    return prim
