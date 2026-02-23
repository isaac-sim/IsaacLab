# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for creating and manipulating USD prims."""

from __future__ import annotations

import functools
import inspect
import logging
import re
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import torch

import omni.kit.commands
import omni.usd
from isaacsim.core.cloner import Cloner
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils

from isaaclab.utils.string import to_camel_case
from isaaclab.utils.version import get_isaac_sim_version

from .queries import find_matching_prim_paths
from .semantics import add_labels
from .stage import get_current_stage, get_current_stage_id, resolve_paths
from .transforms import convert_world_pose_to_local, standardize_xform_ops

if TYPE_CHECKING:
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg

# import logger
logger = logging.getLogger(__name__)


"""
General Utils
"""


def create_prim(
    prim_path: str,
    prim_type: str = "Xform",
    position: Any | None = None,
    translation: Any | None = None,
    orientation: Any | None = None,
    scale: Any | None = None,
    usd_path: str | None = None,
    semantic_label: str | None = None,
    semantic_type: str = "class",
    attributes: dict | None = None,
    stage: Usd.Stage | None = None,
) -> Usd.Prim:
    """Creates a prim in the provided USD stage.

    The method applies the specified transforms, the semantic label and sets the specified attributes.
    The transform can be specified either in world space (using ``position``) or local space (using
    ``translation``).

    The function determines the coordinate system of the transform based on the provided arguments.

    * If ``position`` is provided, it is assumed the orientation is provided in the world frame as well.
    * If ``translation`` is provided, it is assumed the orientation is provided in the local frame as well.

    The scale is always applied in the local frame.

    The function handles various sequence types (list, tuple, numpy array, torch tensor)
    and converts them to properly-typed tuples for operations on the prim.

    .. note::
        Transform operations are standardized to the USD convention: translate, orient (quaternion),
        and scale, in that order. See :func:`standardize_xform_ops` for more details.

    Args:
        prim_path:
            The path of the new prim.
        prim_type:
            Prim type name. Defaults to "Xform", in which case a simple Xform prim is created.
        position:
            Prim position in world space as (x, y, z). If the prim has a parent, this is
            automatically converted to local space relative to the parent. Cannot be used with
            ``translation``. Defaults to None, in which case no position is applied.
        translation:
            Prim translation in local space as (x, y, z). This is applied directly without
            any coordinate transformation. Cannot be used with ``position``. Defaults to None,
            in which case no translation is applied.
        orientation:
            Prim rotation as a quaternion (x, y, z, w). When used with ``position``, the
            orientation is also converted from world space to local space. When used with ``translation``,
            it is applied directly as local orientation. Defaults to None.
        scale:
            Scaling factor in x, y, z. Applied in local space. Defaults to None,
            in which case a uniform scale of 1.0 is applied.
        usd_path:
            Path to the USD file that this prim will reference. Defaults to None.
        semantic_label:
            Semantic label to apply to the prim. Defaults to None, in which case no label is added.
        semantic_type:
            Semantic type for the label. Defaults to "class".
        attributes:
            Key-value pairs of prim attributes to set. Defaults to None, in which case no attributes are set.
        stage:
            The stage to create the prim in. Defaults to None, in which case the current stage is used.

    Returns:
        The created USD prim.

    Raises:
        ValueError: If there is already a prim at the provided prim path.
        ValueError: If both position and translation are provided.

    Example:
        >>> import isaaclab.sim as sim_utils
        >>>
        >>> # Create a cube at world position (1.0, 0.5, 0.0)
        >>> sim_utils.create_prim(
        ...     prim_path="/World/Parent/Cube",
        ...     prim_type="Cube",
        ...     position=(1.0, 0.5, 0.0),
        ...     attributes={"size": 2.0},
        ... )
        Usd.Prim(</World/Parent/Cube>)
        >>>
        >>> # Create a sphere with local translation relative to its parent
        >>> sim_utils.create_prim(
        ...     prim_path="/World/Parent/Sphere",
        ...     prim_type="Sphere",
        ...     translation=(0.5, 0.0, 0.0),
        ...     scale=(2.0, 2.0, 2.0),
        ... )
        Usd.Prim(</World/Parent/Sphere>)
    """
    # Ensure that user doesn't provide both position and translation
    if position is not None and translation is not None:
        raise ValueError("Cannot provide both position and translation. Please provide only one.")

    # obtain stage handle
    stage = get_current_stage() if stage is None else stage

    # check if prim already exists
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    # create prim in stage
    prim = stage.DefinePrim(prim_path, prim_type)
    if not prim.IsValid():
        raise ValueError(f"Failed to create prim at path: '{prim_path}' of type: '{prim_type}'.")
    # apply attributes into prim
    if attributes is not None:
        for k, v in attributes.items():
            prim.GetAttribute(k).Set(v)
    # add reference to USD file
    if usd_path is not None:
        add_usd_reference(prim_path=prim_path, usd_path=usd_path, stage=stage)
    # add semantic label to prim
    if semantic_label is not None:
        add_labels(prim, labels=[semantic_label], instance_name=semantic_type)

    # check if prim type is Xformable
    if not prim.IsA(UsdGeom.Xformable):
        logger.debug(
            f"Prim at path '{prim.GetPath().pathString}' is of type '{prim.GetTypeName()}', "
            "which is not an Xformable. Transform operations will not be standardized. "
            "This is expected for material, shader, and scope prims."
        )
        return prim

    # convert input arguments to tuples
    position = _to_tuple(position) if position is not None else None
    translation = _to_tuple(translation) if translation is not None else None
    orientation = _to_tuple(orientation) if orientation is not None else None
    scale = _to_tuple(scale) if scale is not None else None

    # convert position and orientation to translation and orientation
    # world --> local
    if position is not None:
        # this means that user provided pose in the world frame
        translation, orientation = convert_world_pose_to_local(position, orientation, ref_prim=prim.GetParent())

    # standardize the xform ops
    standardize_xform_ops(prim, translation, orientation, scale)

    return prim


def delete_prim(prim_path: str | Sequence[str], stage: Usd.Stage | None = None) -> bool:
    """Removes the USD Prim and its descendants from the scene if able.

    Args:
        prim_path: The path of the prim to delete. If a list of paths is provided,
            the function will delete all the prims in the list.
        stage: The stage to delete the prim in. Defaults to None, in which case the current stage is used.

    Returns:
        True if the prim or prims were deleted successfully, False otherwise.

    Example:
        >>> import isaaclab.sim as sim_utils
        >>>
        >>> sim_utils.delete_prim("/World/Cube")
    """
    # convert prim_path to list if it is a string
    if isinstance(prim_path, str):
        prim_path = [prim_path]
    # get stage handle
    stage = get_current_stage() if stage is None else stage
    # FIXME: We should not need to cache the stage here. It should
    # happen at the creation of the stage.
    # the prim command looks for the stage ID in the stage cache
    # so we need to ensure the stage is cached
    stage_cache = UsdUtils.StageCache.Get()
    stage_id = stage_cache.GetId(stage).ToLongInt()
    if stage_id < 0:
        stage_id = stage_cache.Insert(stage).ToLongInt()
    # delete prims
    success, _ = omni.kit.commands.execute(
        "DeletePrimsCommand",
        paths=prim_path,
        stage=stage,
    )
    return success


"""
USD Prim properties and attributes.
"""


def make_uninstanceable(prim_path: str | Sdf.Path, stage: Usd.Stage | None = None):
    """Check if a prim and its descendants are instanced and make them uninstanceable.

    This function checks if the prim at the specified prim path and its descendants are instanced.
    If so, it makes the respective prim uninstanceable by disabling instancing on the prim.

    This is useful when we want to modify the properties of a prim that is instanced. For example, if we
    want to apply a different material on an instanced prim, we need to make the prim uninstanceable first.

    Args:
        prim_path: The prim path to check.
        stage: The stage where the prim exists. Defaults to None, in which case the current stage is used.

    Raises:
        ValueError: If the prim path is not global (i.e: does not start with '/').
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # make paths str type if they aren't already
    prim_path = str(prim_path)
    # check if prim path is global
    if not prim_path.startswith("/"):
        raise ValueError(f"Prim path '{prim_path}' is not global. It must start with '/'.")
    # get prim
    prim = stage.GetPrimAtPath(prim_path)
    # check if prim is valid
    if not prim.IsValid():
        raise ValueError(f"Prim at path '{prim_path}' is not valid.")
    # iterate over all prims under prim-path
    all_prims = [prim]
    while len(all_prims) > 0:
        # get current prim
        child_prim = all_prims.pop(0)
        # check if prim is instanced
        if child_prim.IsInstance():
            # make the prim uninstanceable
            child_prim.SetInstanceable(False)
        # add children to list
        all_prims += child_prim.GetFilteredChildren(Usd.TraverseInstanceProxies())


def set_prim_visibility(prim: Usd.Prim, visible: bool) -> None:
    """Sets the visibility of the prim in the opened stage.

    .. note::

        The method does this through the USD API.

    Args:
        prim: the USD prim
        visible: flag to set the visibility of the usd prim in stage.

    Example:
        >>> import isaaclab.sim as sim_utils
        >>>
        >>> # given the stage: /World/Cube. Make the Cube not visible
        >>> prim = sim_utils.get_prim_at_path("/World/Cube")
        >>> sim_utils.set_prim_visibility(prim, False)
    """
    imageable = UsdGeom.Imageable(prim)
    if visible:
        imageable.MakeVisible()
    else:
        imageable.MakeInvisible()


def safe_set_attribute_on_usd_schema(schema_api: Usd.APISchemaBase, name: str, value: Any, camel_case: bool):
    """Set the value of an attribute on its USD schema if it exists.

    A USD API schema serves as an interface or API for authoring and extracting a set of attributes.
    They typically derive from the :class:`pxr.Usd.SchemaBase` class. This function checks if the
    attribute exists on the schema and sets the value of the attribute if it exists.

    Args:
        schema_api: The USD schema to set the attribute on.
        name: The name of the attribute.
        value: The value to set the attribute to.
        camel_case: Whether to convert the attribute name to camel case.

    Raises:
        TypeError: When the input attribute name does not exist on the provided schema API.
    """
    # if value is None, do nothing
    if value is None:
        return
    # convert attribute name to camel case
    if camel_case:
        attr_name = to_camel_case(name, to="CC")
    else:
        attr_name = name
    # retrieve the attribute
    # reference: https://openusd.org/dev/api/_usd__page__common_idioms.html#Usd_Create_Or_Get_Property
    attr = getattr(schema_api, f"Create{attr_name}Attr", None)
    # check if attribute exists
    if attr is not None:
        attr().Set(value)
    else:
        # think: do we ever need to create the attribute if it doesn't exist?
        #   currently, we are not doing this since the schemas are already created with some defaults.
        logger.error(f"Attribute '{attr_name}' does not exist on prim '{schema_api.GetPath()}'.")
        raise TypeError(f"Attribute '{attr_name}' does not exist on prim '{schema_api.GetPath()}'.")


def safe_set_attribute_on_usd_prim(prim: Usd.Prim, attr_name: str, value: Any, camel_case: bool):
    """Set the value of a attribute on its USD prim.

    The function creates a new attribute if it does not exist on the prim. This is because in some cases (such
    as with shaders), their attributes are not exposed as USD prim properties that can be altered. This function
    allows us to set the value of the attributes in these cases.

    Args:
        prim: The USD prim to set the attribute on.
        attr_name: The name of the attribute.
        value: The value to set the attribute to.
        camel_case: Whether to convert the attribute name to camel case.
    """
    # if value is None, do nothing
    if value is None:
        return
    # convert attribute name to camel case
    if camel_case:
        attr_name = to_camel_case(attr_name, to="cC")
    # resolve sdf type based on value
    if isinstance(value, bool):
        sdf_type = Sdf.ValueTypeNames.Bool
    elif isinstance(value, int):
        sdf_type = Sdf.ValueTypeNames.Int
    elif isinstance(value, float):
        sdf_type = Sdf.ValueTypeNames.Float
    elif isinstance(value, str):
        sdf_type = Sdf.ValueTypeNames.String
    elif isinstance(value, (tuple, list)) and len(value) == 3 and any(isinstance(v, float) for v in value):
        sdf_type = Sdf.ValueTypeNames.Float3
    elif isinstance(value, (tuple, list)) and len(value) == 2 and any(isinstance(v, float) for v in value):
        sdf_type = Sdf.ValueTypeNames.Float2
    else:
        raise NotImplementedError(
            f"Cannot set attribute '{attr_name}' with value '{value}'. Please modify the code to support this type."
        )

    # change property using the change_prim_property function
    change_prim_property(
        prop_path=f"{prim.GetPath()}.{attr_name}",
        value=value,
        stage=prim.GetStage(),
        type_to_create_if_not_exist=sdf_type,
    )


def change_prim_property(
    prop_path: str | Sdf.Path,
    value: Any,
    stage: Usd.Stage | None = None,
    type_to_create_if_not_exist: Sdf.ValueTypeNames | None = None,
    is_custom: bool = False,
) -> bool:
    """Change or create a property value on a USD prim.

    This is a simplified property setter that works with the current edit target. If you need
    complex layer management, use :class:`omni.kit.commands.ChangePropertyCommand` instead.

    By default, this function changes the value of the property when it exists. If the property
    doesn't exist, :attr:`type_to_create_if_not_exist` must be provided to create it.

    Note:
        The attribute :attr:`value` must be the correct type for the property.
        For example, if the property is a float, the value must be a float.
        If it is supposed to be a RGB color, the value must be of type :class:`Gf.Vec3f`.

    Args:
        prop_path: Property path in the format ``/World/Prim.propertyName``.
        value: Value to set. If None, the attribute value goes to its default value.
            If the attribute has no default value, it is a silent no-op.
        stage: The USD stage. Defaults to None, in which case the current stage is used.
        type_to_create_if_not_exist: If not None and property doesn't exist, a new property will
            be created with the given type and value. Defaults to None.
        is_custom: If the property is created, specify if it is a custom property (not part of
            the schema). Defaults to False.

    Returns:
        True if the property was successfully changed, False otherwise.

    Raises:
        ValueError: If the prim does not exist at the specified path.

    Example:
        >>> import isaaclab.sim as sim_utils
        >>> from pxr import Sdf
        >>>
        >>> # Change an existing property
        >>> sim_utils.change_prim_property(prop_path="/World/Cube.size", value=2.0)
        True
        >>>
        >>> # Create a new custom property
        >>> sim_utils.change_prim_property(
        ...     prop_path="/World/Cube.customValue",
        ...     value=42,
        ...     type_to_create_if_not_exist=Sdf.ValueTypeNames.Int,
        ...     is_custom=True,
        ... )
        True
    """
    # get stage handle
    stage = get_current_stage() if stage is None else stage

    # convert to Sdf.Path if needed
    prop_path = Sdf.Path(prop_path) if isinstance(prop_path, str) else prop_path

    # get the prim path
    prim_path = prop_path.GetAbsoluteRootOrPrimPath()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise ValueError(f"Prim does not exist at path: '{prim_path}'")

    # get or create the property
    prop = stage.GetPropertyAtPath(prop_path)

    if not prop:
        if type_to_create_if_not_exist is not None:
            # create new attribute on the prim
            prop = prim.CreateAttribute(prop_path.name, type_to_create_if_not_exist, is_custom)
        else:
            logger.error(f"Property {prop_path} does not exist and 'type_to_create_if_not_exist' was not provided.")
            return False

    if not prop:
        logger.error(f"Failed to get or create property at path: '{prop_path}'")
        return False

    # set the value
    if value is None:
        return bool(prop.Clear())
    else:
        return bool(prop.Set(value, Usd.TimeCode.Default()))


"""
Exporting.
"""


def export_prim_to_file(
    path: str | Sdf.Path,
    source_prim_path: str | Sdf.Path,
    target_prim_path: str | Sdf.Path | None = None,
    stage: Usd.Stage | None = None,
):
    """Exports a prim from a given stage to a USD file.

    The function creates a new layer at the provided path and copies the prim to the layer.
    It sets the copied prim as the default prim in the target layer. Additionally, it updates
    the stage up-axis and meters-per-unit to match the current stage.

    Args:
        path: The filepath path to export the prim to.
        source_prim_path: The prim path to export.
        target_prim_path: The prim path to set as the default prim in the target layer.
            Defaults to None, in which case the source prim path is used.
        stage: The stage where the prim exists. Defaults to None, in which case the
            current stage is used.

    Raises:
        ValueError: If the prim paths are not global (i.e: do not start with '/').
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # automatically casting to str in case args
    # are path types
    path = str(path)
    source_prim_path = str(source_prim_path)
    if target_prim_path is not None:
        target_prim_path = str(target_prim_path)

    if not source_prim_path.startswith("/"):
        raise ValueError(f"Source prim path '{source_prim_path}' is not global. It must start with '/'.")
    if target_prim_path is not None and not target_prim_path.startswith("/"):
        raise ValueError(f"Target prim path '{target_prim_path}' is not global. It must start with '/'.")

    # get root layer
    source_layer = stage.GetRootLayer()

    # only create a new layer if it doesn't exist already
    target_layer = Sdf.Find(path)
    if target_layer is None:
        target_layer = Sdf.Layer.CreateNew(path)
    # open the target stage
    target_stage = Usd.Stage.Open(target_layer)

    # update stage data
    UsdGeom.SetStageUpAxis(target_stage, UsdGeom.GetStageUpAxis(stage))
    UsdGeom.SetStageMetersPerUnit(target_stage, UsdGeom.GetStageMetersPerUnit(stage))

    # specify the prim to copy
    source_prim_path = Sdf.Path(source_prim_path)
    if target_prim_path is None:
        target_prim_path = source_prim_path

    # copy the prim
    Sdf.CreatePrimInLayer(target_layer, target_prim_path)
    Sdf.CopySpec(source_layer, source_prim_path, target_layer, target_prim_path)
    # set the default prim
    target_layer.defaultPrim = Sdf.Path(target_prim_path).name
    # resolve paths so asset references remain valid from the new location
    resolve_paths(source_layer.identifier, target_layer.identifier)
    # save the stage
    target_layer.Save()


"""
Decorators
"""


def apply_nested(func: Callable) -> Callable:
    """Decorator to apply a function to all prims under a specified prim-path.

    The function iterates over the provided prim path and all its children to apply input function
    to all prims under the specified prim path.

    If the function succeeds to apply to a prim, it will not look at the children of that prim.
    This is based on the physics behavior that nested schemas are not allowed. For example, a parent prim
    and its child prim cannot both have a rigid-body schema applied on them, or it is not possible to
    have nested articulations.

    While traversing the prims under the specified prim path, the function will throw a warning if it
    does not succeed to apply the function to any prim. This is because the user may have intended to
    apply the function to a prim that does not have valid attributes, or the prim may be an instanced prim.

    Args:
        func: The function to apply to all prims under a specified prim-path. The function
            must take the prim-path and other arguments. It should return a boolean indicating whether
            the function succeeded or not.

    Returns:
        The wrapped function that applies the function to all prims under a specified prim-path.

    Raises:
        ValueError: If the prim-path does not exist on the stage.
    """

    @functools.wraps(func)
    def wrapper(prim_path: str | Sdf.Path, *args, **kwargs):
        # map args and kwargs to function signature so we can get the stage
        # note: we do this to check if stage is given in arg or kwarg
        sig = inspect.signature(func)
        bound_args = sig.bind(prim_path, *args, **kwargs)
        # get current stage
        stage = bound_args.arguments.get("stage")
        if stage is None:
            stage = get_current_stage()

        # get USD prim
        prim: Usd.Prim = stage.GetPrimAtPath(prim_path)
        # check if prim is valid
        if not prim.IsValid():
            raise ValueError(f"Prim at path '{prim_path}' is not valid.")
        # add iterable to check if property was applied on any of the prims
        count_success = 0
        instanced_prim_paths = []
        # iterate over all prims under prim-path
        all_prims = [prim]
        while len(all_prims) > 0:
            # get current prim
            child_prim = all_prims.pop(0)
            child_prim_path = child_prim.GetPath().pathString  # type: ignore
            # check if prim is a prototype
            if child_prim.IsInstance():
                instanced_prim_paths.append(child_prim_path)
                continue
            # set properties
            success = func(child_prim_path, *args, **kwargs)
            # if successful, do not look at children
            # this is based on the physics behavior that nested schemas are not allowed
            if not success:
                all_prims += child_prim.GetChildren()
            else:
                count_success += 1
        # check if we were successful in applying the function to any prim
        if count_success == 0:
            logger.warning(
                f"Could not perform '{func.__name__}' on any prims under: '{prim_path}'."
                " This might be because of the following reasons:"
                "\n\t(1) The desired attribute does not exist on any of the prims."
                "\n\t(2) The desired attribute exists on an instanced prim."
                f"\n\t\tDiscovered list of instanced prim paths: {instanced_prim_paths}"
            )

    return wrapper


def clone(func: Callable) -> Callable:
    """Decorator for cloning a prim based on matching prim paths of the prim's parent.

    The decorator checks if the parent prim path matches any prim paths in the stage. If so, it clones the
    spawned prim at each matching prim path. For example, if the input prim path is: ``/World/Table_[0-9]/Bottle``,
    the decorator will clone the prim at each matching prim path of the parent prim: ``/World/Table_0/Bottle``,
    ``/World/Table_1/Bottle``, etc.

    Note:
        For matching prim paths, the decorator assumes that valid prims exist for all matching prim paths.
        In case no matching prim paths are found, the decorator raises a ``RuntimeError``.

    Args:
        func: The function to decorate.

    Returns:
        The decorated function that spawns the prim and clones it at each matching prim path.
        It returns the spawned source prim, i.e., the first prim in the list of matching prim paths.
    """

    @functools.wraps(func)
    def wrapper(prim_path: str | Sdf.Path, cfg: SpawnerCfg, *args, **kwargs):
        # get stage handle
        stage = get_current_stage()

        # cast prim_path to str type in case its an Sdf.Path
        prim_path = str(prim_path)
        # check prim path is global
        if not prim_path.startswith("/"):
            raise ValueError(f"Prim path '{prim_path}' is not global. It must start with '/'.")
        # resolve: {SPAWN_NS}/AssetName
        # note: this assumes that the spawn namespace already exists in the stage
        root_path, asset_path = prim_path.rsplit("/", 1)
        # check if input is a regex expression
        # note: a valid prim path can only contain alphanumeric characters, underscores, and forward slashes
        is_regex_expression = re.match(r"^[a-zA-Z0-9/_]+$", root_path) is None

        # resolve matching prims for source prim path expression
        if is_regex_expression and root_path != "":
            source_prim_paths = find_matching_prim_paths(root_path)
            # if no matching prims are found, raise an error
            if len(source_prim_paths) == 0:
                raise RuntimeError(
                    f"Unable to find source prim path: '{root_path}'. Please create the prim before spawning."
                )
        else:
            source_prim_paths = [root_path]

        # resolve prim paths for spawning and cloning
        prim_paths = [f"{source_prim_path}/{asset_path}" for source_prim_path in source_prim_paths]
        # spawn single instance
        prim = func(prim_paths[0], cfg, *args, **kwargs)
        # set the prim visibility
        if hasattr(cfg, "visible"):
            imageable = UsdGeom.Imageable(prim)
            if cfg.visible:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()
        # set the semantic annotations
        if hasattr(cfg, "semantic_tags") and cfg.semantic_tags is not None:
            # note: taken from replicator scripts.utils.utils.py
            for semantic_type, semantic_value in cfg.semantic_tags:
                # deal with spaces by replacing them with underscores
                semantic_type_sanitized = semantic_type.replace(" ", "_")
                semantic_value_sanitized = semantic_value.replace(" ", "_")
                # add labels to the prim
                add_labels(
                    prim, labels=[semantic_value_sanitized], instance_name=semantic_type_sanitized, overwrite=False
                )
        # activate rigid body contact sensors (lazy import to avoid circular import with schemas)
        if hasattr(cfg, "activate_contact_sensors") and cfg.activate_contact_sensors:  # type: ignore
            from ..schemas import schemas as _schemas

            _schemas.activate_contact_sensors(prim_paths[0])
        # clone asset using cloner API
        if len(prim_paths) > 1:
            cloner = Cloner(stage=stage)
            # check version of Isaac Sim to determine whether clone_in_fabric is valid
            if get_isaac_sim_version().major < 5:
                # clone the prim
                cloner.clone(
                    prim_paths[0], prim_paths[1:], replicate_physics=False, copy_from_source=cfg.copy_from_source
                )
            else:
                # clone the prim
                clone_in_fabric = kwargs.get("clone_in_fabric", False)
                replicate_physics = kwargs.get("replicate_physics", False)
                cloner.clone(
                    prim_paths[0],
                    prim_paths[1:],
                    replicate_physics=replicate_physics,
                    copy_from_source=cfg.copy_from_source,
                    clone_in_fabric=clone_in_fabric,
                )
        # return the source prim
        return prim

    return wrapper


"""
Material bindings.
"""


@apply_nested
def bind_visual_material(
    prim_path: str | Sdf.Path,
    material_path: str | Sdf.Path,
    stage: Usd.Stage | None = None,
    stronger_than_descendants: bool = True,
):
    """Bind a visual material to a prim.

    This function is a wrapper around the USD command `BindMaterialCommand`_.

    .. note::
        The function is decorated with :meth:`apply_nested` to allow applying the function to a prim path
        and all its descendants.

    .. _BindMaterialCommand: https://docs.omniverse.nvidia.com/kit/docs/omni.usd/latest/omni.usd.commands/omni.usd.commands.BindMaterialCommand.html

    Args:
        prim_path: The prim path where to apply the material.
        material_path: The prim path of the material to apply.
        stage: The stage where the prim and material exist.
            Defaults to None, in which case the current stage is used.
        stronger_than_descendants: Whether the material should override the material of its descendants.
            Defaults to True.

    Raises:
        ValueError: If the provided prim paths do not exist on stage.
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # check if prim and material exists
    if not stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"Target prim '{material_path}' does not exist.")
    if not stage.GetPrimAtPath(material_path).IsValid():
        raise ValueError(f"Visual material '{material_path}' does not exist.")

    # resolve token for weaker than descendants
    # bind material command expects a string token
    if stronger_than_descendants:
        binding_strength = "strongerThanDescendants"
    else:
        binding_strength = "weakerThanDescendants"
    # obtain material binding API
    # note: we prefer using the command here as it is more robust than the USD API
    success, _ = omni.kit.commands.execute(
        "BindMaterialCommand",
        prim_path=prim_path,
        material_path=material_path,
        strength=binding_strength,
        stage=stage,
    )
    # return success
    return success


@apply_nested
def bind_physics_material(
    prim_path: str | Sdf.Path,
    material_path: str | Sdf.Path,
    stage: Usd.Stage | None = None,
    stronger_than_descendants: bool = True,
):
    """Bind a physics material to a prim.

    `Physics material`_ can be applied only to a prim with physics-enabled on them. This includes having
    collision APIs, or deformable body APIs, or being a particle system. In case the prim does not have
    any of these APIs, the function will not apply the material and return False.

    .. note::
        The function is decorated with :meth:`apply_nested` to allow applying the function to a prim path
        and all its descendants.

    .. _Physics material: https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.html#isaaclab.sim.SimulationCfg.physics_material

    Args:
        prim_path: The prim path where to apply the material.
        material_path: The prim path of the material to apply.
        stage: The stage where the prim and material exist.
            Defaults to None, in which case the current stage is used.
        stronger_than_descendants: Whether the material should override the material of its descendants.
            Defaults to True.

    Raises:
        ValueError: If the provided prim paths do not exist on stage.
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # check if prim and material exists
    if not stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"Target prim '{material_path}' does not exist.")
    if not stage.GetPrimAtPath(material_path).IsValid():
        raise ValueError(f"Physics material '{material_path}' does not exist.")
    # get USD prim
    prim = stage.GetPrimAtPath(prim_path)
    # check if prim has collision applied on it
    applied = prim.GetAppliedSchemas()
    has_physics_scene_api = "PhysxSceneAPI" in applied
    has_collider = prim.HasAPI(UsdPhysics.CollisionAPI)
    has_deformable_body = "PhysxDeformableBodyAPI" in applied
    has_particle_system = prim.GetTypeName() == "PhysxParticleSystem"
    if not (has_physics_scene_api or has_collider or has_deformable_body or has_particle_system):
        logger.debug(
            f"Cannot apply physics material '{material_path}' on prim '{prim_path}'. It is neither a"
            " PhysX scene, collider, a deformable body, nor a particle system."
        )
        return False

    # obtain material binding API
    if prim.HasAPI(UsdShade.MaterialBindingAPI):
        material_binding_api = UsdShade.MaterialBindingAPI(prim)
    else:
        material_binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
    # obtain the material prim

    material = UsdShade.Material(stage.GetPrimAtPath(material_path))
    # resolve token for weaker than descendants
    if stronger_than_descendants:
        binding_strength = UsdShade.Tokens.strongerThanDescendants
    else:
        binding_strength = UsdShade.Tokens.weakerThanDescendants
    # apply the material
    material_binding_api.Bind(material, bindingStrength=binding_strength, materialPurpose="physics")  # type: ignore
    # return success
    return True


"""
USD References and Variants.
"""


def add_usd_reference(
    prim_path: str, usd_path: str, prim_type: str = "Xform", stage: Usd.Stage | None = None
) -> Usd.Prim:
    """Adds a USD reference at the specified prim path on the provided stage.

    This function adds a reference to an external USD file at the specified prim path on the provided stage.
    If the prim does not exist, it will be created with the specified type.

    The function also handles stage units verification to ensure compatibility. For instance,
    if the current stage is in meters and the referenced USD file is in centimeters, the function will
    convert the units to match. This is done using the :mod:`omni.metrics.assembler` functionality.

    Args:
        prim_path: The prim path where the reference will be attached.
        usd_path: The path to USD file to reference.
        prim_type: The type of prim to create if it doesn't exist. Defaults to "Xform".
        stage: The stage to add the reference to. Defaults to None, in which case the current stage is used.

    Returns:
        The USD prim at the specified prim path.

    Raises:
        FileNotFoundError: When the input USD file is not found at the specified path.
    """
    # get current stage
    stage = get_current_stage() if stage is None else stage
    # get prim at path
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        prim = stage.DefinePrim(prim_path, prim_type)

    def _add_reference_to_prim(prim: Usd.Prim) -> Usd.Prim:
        """Helper function to add a reference to a prim."""
        success_bool = prim.GetReferences().AddReference(usd_path)
        if not success_bool:
            raise RuntimeError(
                f"Unable to add USD reference to the prim at path: {prim_path} from the USD file at path: {usd_path}"
            )
        return prim

    # Compatibility with Isaac Sim 4.5 where omni.metrics is not available
    if get_isaac_sim_version().major < 5:
        return _add_reference_to_prim(prim)

    # check if the USD file is valid and add reference to the prim
    sdf_layer = Sdf.Layer.FindOrOpen(usd_path)
    if not sdf_layer:
        raise FileNotFoundError(f"Unable to open the usd file at path: {usd_path}")

    # import metrics assembler interface
    # note: this is only available in Isaac Sim 5.0 and above
    from omni.metrics.assembler.core import get_metrics_assembler_interface

    # obtain the stage ID
    stage_id = get_current_stage_id()
    # check if the layers are compatible (i.e. the same units)
    ret_val = get_metrics_assembler_interface().check_layers(
        stage.GetRootLayer().identifier, sdf_layer.identifier, stage_id
    )
    # log that metric assembler did not detect any issues
    if ret_val["ret_val"]:
        logger.info(
            "Metric assembler detected no issues between the current stage and the referenced USD file at path:"
            f" {usd_path}"
        )
    # add reference to the prim
    return _add_reference_to_prim(prim)


def get_usd_references(prim_path: str, stage: Usd.Stage | None = None) -> list[str]:
    """Gets the USD references at the specified prim path on the provided stage.

    Args:
        prim_path: The prim path to get the USD references from.
        stage: The stage to get the USD references from. Defaults to None, in which case the current stage is used.

    Returns:
        A list of USD reference paths.

    Raises:
        ValueError: If the prim at the specified path is not valid.
    """
    # get stage handle
    stage = get_current_stage() if stage is None else stage
    # get prim at path
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path '{prim_path}' is not valid.")
    # get USD references
    references = []
    for prim_spec in prim.GetPrimStack():
        for ref in prim_spec.referenceList.prependedItems:
            references.append(str(ref.assetPath))
    return references


def select_usd_variants(prim_path: str, variants: object | dict[str, str], stage: Usd.Stage | None = None):
    """Sets the variant selections from the specified variant sets on a USD prim.

    `USD Variants`_ are a very powerful tool in USD composition that allows prims to have different options on
    a single asset. This can be done by modifying variations of the same prim parameters per variant option in a set.
    This function acts as a script-based utility to set the variant selections for the specified variant sets on a
    USD prim.

    The function takes a dictionary or a config class mapping variant set names to variant selections. For instance,
    if we have a prim at ``"/World/Table"`` with two variant sets: "color" and "size", we can set the variant
    selections as follows:

    .. code-block:: python

        select_usd_variants(
            prim_path="/World/Table",
            variants={
                "color": "red",
                "size": "large",
            },
        )

    Alternatively, we can use a config class to define the variant selections:

    .. code-block:: python

        @configclass
        class TableVariants:
            color: Literal["blue", "red"] = "red"
            size: Literal["small", "large"] = "large"


        select_usd_variants(
            prim_path="/World/Table",
            variants=TableVariants(),
        )

    Args:
        prim_path: The path of the USD prim.
        variants: A dictionary or config class mapping variant set names to variant selections.
        stage: The USD stage. Defaults to None, in which case, the current stage is used.

    Raises:
        ValueError: If the prim at the specified path is not valid.

    .. _USD Variants: https://graphics.pixar.com/usd/docs/USD-Glossary.html#USDGlossary-Variant
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # Obtain prim
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path '{prim_path}' is not valid.")
    # Convert to dict if we have a configclass object.
    if not isinstance(variants, dict):
        variants = variants.to_dict()  # type: ignore

    existing_variant_sets = prim.GetVariantSets()
    for variant_set_name, variant_selection in variants.items():  # type: ignore
        # Check if the variant set exists on the prim.
        if not existing_variant_sets.HasVariantSet(variant_set_name):
            logger.warning(f"Variant set '{variant_set_name}' does not exist on prim '{prim_path}'.")
            continue

        variant_set = existing_variant_sets.GetVariantSet(variant_set_name)
        # Only set the variant selection if it is different from the current selection.
        if variant_set.GetVariantSelection() != variant_selection:
            variant_set.SetVariantSelection(variant_selection)
            logger.info(
                f"Setting variant selection '{variant_selection}' for variant set '{variant_set_name}' on"
                f" prim '{prim_path}'."
            )


"""
Internal Helpers.
"""


def _to_tuple(value: Any) -> tuple[float, ...]:
    """Convert various sequence types to a Python tuple of floats.

    This function provides robust conversion from different array-like types (list, tuple, numpy array,
    torch tensor) to Python tuples. It handles edge cases like malformed sequences, CUDA tensors,
    and arrays with singleton dimensions.

    Args:
        value: A sequence-like object containing floats. Supported types include:
            - Python list or tuple
            - NumPy array (any device)
            - PyTorch tensor (CPU or CUDA)
            - Mixed sequences with numpy/torch scalar items and float values

    Returns:
        A one-dimensional tuple of floats.

    Raises:
        ValueError: If the input value is not one-dimensional after squeezing singleton dimensions.

    Example:
        >>> import torch
        >>> import numpy as np
        >>>
        >>> _to_tuple([1.0, 2.0, 3.0])
        (1.0, 2.0, 3.0)
        >>> _to_tuple(torch.tensor([[1.0, 2.0]]))  # Squeezes first dimension
        (1.0, 2.0)
        >>> _to_tuple(np.array([1.0, 2.0, 3.0]))
        (1.0, 2.0, 3.0)
        >>> _to_tuple((1.0, 2.0, 3.0))
        (1.0, 2.0, 3.0)

    """
    # Normalize to tensor if value is a plain sequence (list with mixed types, etc.)
    # This handles cases like [np.float32(1.0), 2.0, torch.tensor(3.0)]
    if not hasattr(value, "tolist"):
        value = torch.tensor(value, device="cpu", dtype=torch.float)

    # Remove leading singleton dimension if present (e.g., shape (1, 3) -> (3,))
    # This is common when batched operations produce single-item batches
    if value.ndim != 1:
        value = value.squeeze()
    # Validate that the result is one-dimensional
    if value.ndim != 1:
        raise ValueError(f"Input value is not one dimensional: {value.shape}")

    # Convert to tuple - works for both numpy arrays and torch tensors
    return tuple(value.tolist())
