# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for working with USD transform (xform) operations.

This module provides utilities for manipulating USD transform operations (xform ops) on prims.
Transform operations in USD define how geometry is positioned, oriented, and scaled in 3D space.

The utilities in this module help standardize transform stacks, clear operations, and manipulate
transforms in a consistent way across different USD assets.
"""

from __future__ import annotations

__all__ = ["standardize_xform_ops"]

import logging
from pxr import Gf, Usd, UsdGeom

# import logger
logger = logging.getLogger(__name__)

_INVALID_XFORM_OPS = [
    "xformOp:rotateX",
    "xformOp:rotateXZY",
    "xformOp:rotateY",
    "xformOp:rotateYXZ",
    "xformOp:rotateYZX",
    "xformOp:rotateZ",
    "xformOp:rotateZYX",
    "xformOp:rotateZXY",
    "xformOp:rotateXYZ",
    "xformOp:transform",
]
"""List of invalid xform ops that should be removed."""


def standardize_xform_ops(
    prim: Usd.Prim,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    scale: tuple[float, float, float] | None = None,
) -> bool:
    """Standardize and normalize the transform operations on a USD prim.

    This function standardizes a prim's transform stack to use the common transform operation
    order: translate, orient (quaternion rotation), and scale. It performs the following:

    1. Captures the current local pose of the prim (relative to parent)
    2. Clears all existing transform operations
    3. Removes deprecated/non-standard transform operations (e.g., rotateXYZ, transform matrix)
    4. Establishes the standard transform stack: [translate, orient, scale]
    5. Handles unit resolution for scale attributes
    6. Restores the original local pose using the new standardized operations

    This is particularly useful when importing assets from different sources that may use
    various transform operation conventions, ensuring a consistent and predictable transform
    stack across all prims in the scene.

    .. note::
        The standard transform operation order follows USD best practices:
        ``xformOp:translate``, ``xformOp:orient``, ``xformOp:scale``. This order is
        compatible with most USD tools and workflows.

    .. warning::
        This function modifies the prim's transform stack in place. While it preserves
        the local pose by default, any animation or time-sampled transform data will be lost
        as only the current (default) time code values are preserved.

    Args:
        prim: The USD prim to standardize transform operations for. Must be a valid
            prim that supports the Xformable schema.
        translation: Optional translation (x, y, z) to set. If None, preserves current
            local translation. Defaults to None.
        orientation: Optional orientation quaternion (w, x, y, z) to set. If None, preserves
            current local orientation. Defaults to None.
        scale: Optional scale (x, y, z) to set. If None, preserves current scale or uses
            (1.0, 1.0, 1.0) if no scale exists. Defaults to None.

    Returns:
        True if the transform operations were standardized successfully, False otherwise.

    Raises:
        ValueError: If the prim is not valid or does not support transform operations.

    Example:
        >>> import isaaclab.sim as sim_utils
        >>>
        >>> # Get a prim with non-standard transform operations
        >>> prim = stage.GetPrimAtPath("/World/Asset")
        >>> # Standardize its transform stack while preserving pose
        >>> sim_utils.standardize_xform_ops(prim)
        >>> # The prim now uses: translate, orient, scale in that order
        >>>
        >>> # Or standardize and set new transform values
        >>> sim_utils.standardize_xform_ops(
        ...     prim,
        ...     translation=(1.0, 2.0, 3.0),
        ...     orientation=(1.0, 0.0, 0.0, 0.0),
        ...     scale=(2.0, 2.0, 2.0)
        ... )
    """
    # Validate prim
    if not prim.IsValid():
        raise ValueError(f"Prim at path '{prim.GetPath()}' is not valid.")

    # Check if prim is an Xformable
    if not prim.IsA(UsdGeom.Xformable):
        logger.error(f"Prim at path '{prim.GetPath()}' is not an Xformable.")
        return False

    # Create xformable interface
    xformable = UsdGeom.Xformable(prim)
    # Get current property names
    prop_names = prim.GetPropertyNames()

    # Obtain current local transformations
    tf = Gf.Transform(xformable.GetLocalTransformation())
    xform_pos = Gf.Vec3d(tf.GetTranslation())
    xform_quat = Gf.Quatd(tf.GetRotation().GetQuat())
    xform_scale = Gf.Vec3d(tf.GetScale())

    if translation is not None:
        xform_pos = Gf.Vec3d(*translation)
    if orientation is not None:
        xform_quat = Gf.Quatd(*orientation)

    # Handle scale resolution
    if scale is not None:
        # User provided scale
        xform_scale = scale
    elif "xformOp:scale" in prop_names:
        # Handle unit resolution for scale if present
        # This occurs when assets are imported with different unit scales
        # Reference: Omniverse Metrics Assembler
        if "xformOp:scale:unitsResolve" in prop_names:
            units_resolve = prim.GetAttribute("xformOp:scale:unitsResolve").Get()
            for i in range(3):
                xform_scale[i] = xform_scale[i] * units_resolve[i]
        # Convert to tuple
        xform_scale = tuple(xform_scale)
    else:
        # No scale exists, use default uniform scale
        xform_scale = Gf.Vec3d(1.0, 1.0, 1.0)

    # Clear the existing transform operation order
    has_reset = xformable.GetResetXformStack()
    for prop_name in prop_names:
        if prop_name in _INVALID_XFORM_OPS:
            prim.RemoveProperty(prop_name)

    # Remove unitsResolve attribute if present (already handled in scale resolution above)
    if "xformOp:scale:unitsResolve" in prop_names:
        prim.RemoveProperty("xformOp:scale:unitsResolve")

    # Set up or retrieve scale operation
    xform_op_scale = UsdGeom.XformOp(prim.GetAttribute("xformOp:scale"))
    if not xform_op_scale:
        xform_op_scale = xformable.AddXformOp(UsdGeom.XformOp.TypeScale, UsdGeom.XformOp.PrecisionDouble, "")

    # Set up or retrieve translate operation
    xform_op_translate = UsdGeom.XformOp(prim.GetAttribute("xformOp:translate"))
    if not xform_op_translate:
        xform_op_translate = xformable.AddXformOp(UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.PrecisionDouble, "")

    # Set up or retrieve orient (quaternion rotation) operation
    xform_op_orient = UsdGeom.XformOp(prim.GetAttribute("xformOp:orient"))
    if not xform_op_orient:
        xform_op_orient = xformable.AddXformOp(UsdGeom.XformOp.TypeOrient, UsdGeom.XformOp.PrecisionDouble, "")

    # Set the transform operation order: translate -> orient -> scale
    # This is the standard USD convention and ensures consistent behavior
    xformable.SetXformOpOrder([xform_op_translate, xform_op_orient, xform_op_scale], has_reset)

    # Set the transform values using the new standardized transform operations
    # Convert tuples to Gf types for USD
    xform_op_translate.Set(xform_pos)
    xform_op_orient.Set(xform_quat)
    xform_op_scale.Set(xform_scale)

    return True
