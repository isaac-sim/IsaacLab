# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pxr import Gf, Sdf, Usd, UsdGeom  # noqa: F401

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

_STANDARD_XFORM_OP_ORDER = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
"""Canonical xform op order enforced by :func:`standardize_xform_ops`."""


def standardize_xform_ops(
    prim: Usd.Prim,
    translation: tuple[float, ...] | None = None,
    orientation: tuple[float, ...] | None = None,
    scale: tuple[float, ...] | None = None,
) -> bool:
    """Standardize the transform operation stack on a USD prim to a canonical form.

    This function converts a prim's transform stack to use the standard USD transform operation
    order: [translate, orient, scale]. The function performs the following operations:

    1. Validates that the prim is Xformable
    2. Captures the current local transform (translation, rotation, scale)
    3. Resolves and bakes unit scale conversions (xformOp:scale:unitsResolve)
    4. Creates or reuses standard transform operations (translate, orient, scale)
    5. Sets the transform operation order to [translate, orient, scale]
    6. Applies the preserved or user-specified transform values

    The entire modification is performed within an ``Sdf.ChangeBlock`` for optimal performance
    when processing multiple prims.

    .. note::
        **Standard Transform Order:** The function enforces the USD best practice order:
        ``xformOp:translate``, ``xformOp:orient``, ``xformOp:scale``. This order is
        compatible with most USD tools and workflows, and uses quaternions for rotation
        (avoiding gimbal lock issues).

    .. note::
        **Pose Preservation:** By default, the function preserves the prim's local transform
        (relative to its parent). The world-space position of the prim remains unchanged
        unless explicit ``translation``, ``orientation``, or ``scale`` values are provided.

    .. warning::
        **Animation Data Loss:** This function only preserves transform values at the default
        time code (``Usd.TimeCode.Default()``). Any animation or time-sampled transform data
        will be lost. Use this function during asset import or preparation, not on animated prims.

    .. warning::
        **Unit Scale Resolution:** If the prim has a ``xformOp:scale:unitsResolve`` attribute
        (common in imported assets with unit mismatches), it will be baked into the scale
        and removed. For example, a scale of (1, 1, 1) with unitsResolve of (100, 100, 100)
        becomes a final scale of (100, 100, 100).

    Args:
        prim: The USD prim to standardize. Must be a valid prim that supports the
            UsdGeom.Xformable schema (e.g., Xform, Mesh, Cube, etc.). Material and
            Shader prims are not Xformable and will return False.
        translation: Optional translation vector (x, y, z) in local space. If provided,
            overrides the prim's current translation. If None, preserves the current
            local translation. Defaults to None.
        orientation: Optional orientation quaternion (x, y, z, w) in local space. If provided,
            overrides the prim's current orientation. If None, preserves the current
            local orientation. Defaults to None.
        scale: Optional scale vector (x, y, z). If provided, overrides the prim's current scale.
            If None, preserves the current scale (after unit resolution) or uses (1, 1, 1)
            if no scale exists. Defaults to None.

    Returns:
        bool: True if the transform operations were successfully standardized. False if the
            prim is not Xformable (e.g., Material, Shader prims). The function will log an
            error message when returning False.

    Raises:
        ValueError: If the prim is not valid (i.e., does not exist or is an invalid prim).

    Example:
        >>> import isaaclab.sim as sim_utils
        >>>
        >>> # Standardize a prim with non-standard transform operations
        >>> prim = stage.GetPrimAtPath("/World/ImportedAsset")
        >>> result = sim_utils.standardize_xform_ops(prim)
        >>> if result:
        ...     print("Transform stack standardized successfully")
        >>> # The prim now uses: [translate, orient, scale] in that order
        >>>
        >>> # Standardize and set new transform values
        >>> sim_utils.standardize_xform_ops(
        ...     prim,
        ...     translation=(1.0, 2.0, 3.0),
        ...     orientation=(0.0, 0.0, 0.0, 1.0),  # identity rotation (x, y, z, w)
        ...     scale=(2.0, 2.0, 2.0),
        ... )
        >>>
        >>> # Batch processing for performance
        >>> prims_to_standardize = [stage.GetPrimAtPath(p) for p in prim_paths]
        >>> for prim in prims_to_standardize:
        ...     sim_utils.standardize_xform_ops(prim)  # Each call uses Sdf.ChangeBlock
    """
    from pxr import Gf, Sdf, UsdGeom  # noqa: PLC0415

    if not prim.IsValid():
        raise ValueError(f"Prim at path '{prim.GetPath()}' is not valid.")
    if not prim.IsA(UsdGeom.Xformable):
        logger.error(
            f"Prim at path '{prim.GetPath().pathString}' is of type '{prim.GetTypeName()}', "
            "which is not an Xformable. Transform operations will not be standardized. "
            "This is expected for material, shader, and scope prims."
        )
        return False

    xformable = UsdGeom.Xformable(prim)
    prop_names = prim.GetPropertyNames()

    # current local transform, overridden by any explicitly provided values
    tf = Gf.Transform(xformable.GetLocalTransformation())
    xform_pos = Gf.Vec3d(tf.GetTranslation())
    xform_quat = Gf.Quatd(tf.GetRotation().GetQuat())
    xform_scale = Gf.Vec3d(tf.GetScale())

    if translation is not None:
        xform_pos = Gf.Vec3d(*translation)
    if orientation is not None:
        # orientation is (x, y, z, w), Gf.Quatd expects (w, x, y, z)
        xform_quat = Gf.Quatd(orientation[3], orientation[0], orientation[1], orientation[2])

    if scale is not None:
        xform_scale = Gf.Vec3d(*scale)
    elif "xformOp:scale" in prop_names:
        # bake the unit conversion authored by the Omniverse Metrics Assembler into the scale
        if "xformOp:scale:unitsResolve" in prop_names:
            units_resolve = prim.GetAttribute("xformOp:scale:unitsResolve").Get()
            for i in range(3):
                xform_scale[i] = xform_scale[i] * units_resolve[i]
    else:
        xform_scale = Gf.Vec3d(1.0, 1.0, 1.0)

    has_reset = xformable.GetResetXformStack()

    # Ensure the prim has an "over" spec on the edit target layer. Prims from
    # referenced USD files may only exist in the reference layer with no spec on
    # the edit target. Inside an Sdf.ChangeBlock, AddXformOp calls CreateAttribute
    # which needs an existing prim spec on the edit target — it cannot create one
    # while stage recomposition is deferred.
    edit_layer = prim.GetStage().GetEditTarget().GetLayer()
    if not edit_layer.GetPrimAtPath(prim.GetPath()):
        for prefix in prim.GetPath().GetPrefixes():
            if not edit_layer.GetPrimAtPath(prefix):
                parent_spec = edit_layer.GetPrimAtPath(prefix.GetParentPath()) or edit_layer.pseudoRoot
                Sdf.PrimSpec(parent_spec, prefix.name, Sdf.SpecifierOver)

    with Sdf.ChangeBlock():
        for prop_name in prop_names:
            if prop_name in _INVALID_XFORM_OPS or prop_name == "xformOp:scale:unitsResolve":
                prim.RemoveProperty(prop_name)

        # reuse existing standard ops so their precision is preserved; create missing ones as double
        xform_ops = []
        for op_type, value in (
            (UsdGeom.XformOp.TypeTranslate, xform_pos),
            (UsdGeom.XformOp.TypeOrient, xform_quat),
            (UsdGeom.XformOp.TypeScale, xform_scale),
        ):
            xform_op = UsdGeom.XformOp(prim.GetAttribute(f"xformOp:{UsdGeom.XformOp.GetOpTypeToken(op_type)}"))
            if not xform_op:
                xform_op = xformable.AddXformOp(op_type, UsdGeom.XformOp.PrecisionDouble, "")
            # cast to the op's existing value type so float-precision ops stay float
            current_value = xform_op.Get()
            xform_op.Set(type(current_value)(value) if current_value is not None else value)
            xform_ops.append(xform_op)

        xformable.SetXformOpOrder(xform_ops, has_reset)

    return True


def validate_standard_xform_ops(prim: Usd.Prim) -> bool:
    """Validate if the transform operations on a prim are standardized.

    This function checks if the transform operations on a prim are standardized to the canonical form:
    [translate, orient, scale].

    Args:
        prim: The USD prim to validate.
    """
    from pxr import UsdGeom  # noqa: PLC0415

    if not prim.IsValid():
        logger.error(f"Prim at path '{prim.GetPath().pathString}' is not valid.")
        return False
    if not prim.IsA(UsdGeom.Xformable):
        logger.error(f"Prim at path '{prim.GetPath().pathString}' is not an xformable.")
        return False
    xform_op_order = [op.GetOpName() for op in UsdGeom.Xformable(prim).GetOrderedXformOps()]
    if xform_op_order != _STANDARD_XFORM_OP_ORDER:
        logger.error(
            f"Xform operation order for prim at path '{prim.GetPath().pathString}' is not the canonical form."
            f" Received order: {xform_op_order} Expected order: {_STANDARD_XFORM_OP_ORDER}"
        )
        return False
    return True


def resolve_prim_pose(
    prim: Usd.Prim, ref_prim: Usd.Prim | None = None
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Resolve the pose of a prim with respect to another prim.

    Note:
        This function ignores scale and skew by orthonormalizing the transformation
        matrix at the final step. However, if any ancestor prim in the hierarchy
        has non-uniform scale, that scale will still affect the resulting position
        and orientation of the prim (because it's baked into the transform before
        scale removal).

        In other words: scale **is not removed hierarchically**. If you need
        completely scale-free poses, you must walk the transform chain and strip
        scale at each level. Please open an issue if you need this functionality.

    Args:
        prim: The USD prim to resolve the pose for.
        ref_prim: The USD prim to compute the pose with respect to.
            Defaults to None, in which case the world frame is used.

    Returns:
        A tuple containing the position (as a 3D vector) and the quaternion orientation
        in the (x, y, z, w) format.

    Raises:
        ValueError: If the prim or ref prim is not valid.

    Example:
        >>> import isaaclab.sim as sim_utils
        >>> from pxr import Usd, UsdGeom
        >>>
        >>> # Get prim
        >>> stage = sim_utils.get_current_stage()
        >>> prim = stage.GetPrimAtPath("/World/ImportedAsset")
        >>>
        >>> # Resolve pose
        >>> pos, quat = sim_utils.resolve_prim_pose(prim)
        >>> print(f"Position: {pos}")
        >>> print(f"Orientation: {quat}")
        >>>
        >>> # Resolve pose with respect to another prim
        >>> ref_prim = stage.GetPrimAtPath("/World/Reference")
        >>> pos, quat = sim_utils.resolve_prim_pose(prim, ref_prim)
        >>> print(f"Position: {pos}")
        >>> print(f"Orientation: {quat}")
    """
    from pxr import Sdf, Usd, UsdGeom  # noqa: PLC0415

    if not prim.IsValid():
        raise ValueError(f"Prim at path '{prim.GetPath().pathString}' is not valid.")
    prim_tf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    # orthonormalize so the extracted quaternion is normalized
    prim_tf.Orthonormalize()

    # the root frame is the world frame, so only a non-root reference changes the result
    if ref_prim is not None and ref_prim.GetPath() != Sdf.Path.absoluteRootPath:
        ref_tf = UsdGeom.Xformable(ref_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        ref_tf.Orthonormalize()
        prim_tf = prim_tf * ref_tf.GetInverse()

    quat = prim_tf.ExtractRotationQuat()
    return tuple(prim_tf.ExtractTranslation()), (*quat.imaginary, quat.real)


def resolve_prim_scale(prim: Usd.Prim) -> tuple[float, float, float]:
    """Resolve the scale of a prim in the world frame.

    At an attribute level, a USD prim's scale is a scaling transformation applied to the prim with
    respect to its parent prim. This function resolves the scale of the prim in the world frame,
    by computing the local to world transform of the prim. This is equivalent to traversing up
    the prim hierarchy and accounting for the rotations and scales of the prims.

    For instance, if a prim has a scale of (1, 2, 3) and it is a child of a prim with a scale of (4, 5, 6),
    then the scale of the prim in the world frame is (4, 10, 18).

    Args:
        prim: The USD prim to resolve the scale for.

    Returns:
        The scale of the prim in the x, y, and z directions in the world frame.

    Raises:
        ValueError: If the prim is not valid.

    Example:
        >>> import isaaclab.sim as sim_utils
        >>> from pxr import Usd, UsdGeom
        >>>
        >>> # Get prim
        >>> stage = sim_utils.get_current_stage()
        >>> prim = stage.GetPrimAtPath("/World/ImportedAsset")
        >>>
        >>> # Resolve scale
        >>> scale = sim_utils.resolve_prim_scale(prim)
        >>> print(f"Scale: {scale}")
    """
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    if not prim.IsValid():
        raise ValueError(f"Prim at path '{prim.GetPath().pathString}' is not valid.")
    world_transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return tuple(v.GetLength() for v in world_transform.ExtractRotationMatrix())


def convert_world_pose_to_local(
    position: tuple[float, ...],
    orientation: tuple[float, ...] | None,
    ref_prim: Usd.Prim,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float] | None]:
    """Convert a world-space pose to local-space pose relative to a reference prim.

    This function takes a position and orientation in world space and converts them to local space
    relative to the given reference prim. This is useful when creating or positioning prims where you
    know the desired world position but need to set local transform attributes relative to another prim.

    The conversion uses the standard USD transformation math:
    ``local_transform = world_transform * inverse(ref_world_transform)``

    .. note::
        If the reference prim is the root prim ("/"), the position and orientation are returned
        unchanged, as they are already effectively in local/world space.

    Args:
        position: The world-space position as (x, y, z).
        orientation: The world-space orientation as quaternion (x, y, z, w). If None, only position is converted
            and None is returned for orientation.
        ref_prim: The reference USD prim to compute the local transform relative to. If this is
            the root prim ("/"), the world pose is returned unchanged.

    Returns:
        A tuple of (local_translation, local_orientation) where:

        - local_translation is a tuple of (x, y, z) in local space relative to ref_prim
        - local_orientation is a tuple of (x, y, z, w) in local space relative to ref_prim,
          or None if no orientation was provided

    Raises:
        ValueError: If the reference prim is not a valid USD prim.

    Example:
        >>> import isaaclab.sim as sim_utils
        >>> from pxr import Usd, UsdGeom
        >>>
        >>> # Get reference prim
        >>> stage = sim_utils.get_current_stage()
        >>> ref_prim = stage.GetPrimAtPath("/World/Reference")
        >>>
        >>> # Convert world pose to local (relative to ref_prim)
        >>> world_pos = (10.0, 5.0, 0.0)
        >>> world_quat = (0.0, 0.0, 0.0, 1.0)  # identity rotation (x, y, z, w)
        >>> local_pos, local_quat = sim_utils.convert_world_pose_to_local(world_pos, world_quat, ref_prim)
        >>> print(f"Local position: {local_pos}")
        >>> print(f"Local orientation: {local_quat}")
    """
    from pxr import Gf, Sdf, Usd, UsdGeom  # noqa: PLC0415

    if not ref_prim.IsValid():
        raise ValueError(f"Reference prim at path '{ref_prim.GetPath().pathString}' is not valid.")
    if ref_prim.GetPath() == Sdf.Path.absoluteRootPath:
        return position, orientation  # type: ignore

    ref_world_tf = UsdGeom.Xformable(ref_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    desired_world_tf = Gf.Matrix4d()
    desired_world_tf.SetTranslateOnly(Gf.Vec3d(*position))
    if orientation is not None:
        # orientation is (x, y, z, w), Gf.Quatd expects (w, x, y, z)
        desired_world_tf.SetRotateOnly(Gf.Quatd(orientation[3], orientation[0], orientation[1], orientation[2]))

    # local = world * inv(ref_world)
    local_transform = Gf.Transform(desired_world_tf * ref_world_tf.GetInverse())
    local_translation = tuple(local_transform.GetTranslation())

    local_orientation = None
    if orientation is not None:
        quat = local_transform.GetRotation().GetQuat()
        local_orientation = (*quat.GetImaginary(), quat.GetReal())

    return local_translation, local_orientation
