# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module with USD-related utilities."""

from __future__ import annotations

import contextlib
import functools
import inspect
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import carb
import isaacsim.core.utils.stage as stage_utils
import omni
import omni.kit.commands
import omni.log
from isaacsim.core.cloner import Cloner
from isaacsim.core.utils.carb import get_carb_setting
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.version import get_version
from pxr import PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils

# from Isaac Sim 4.2 onwards, pxr.Semantics is deprecated
try:
    import Semantics
except ModuleNotFoundError:
    from pxr import Semantics

from isaaclab.utils.string import to_camel_case

from . import schemas

if TYPE_CHECKING:
    from .spawners.spawner_cfg import SpawnerCfg

"""
Attribute - Setters.
"""


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
        omni.log.error(f"Attribute '{attr_name}' does not exist on prim '{schema_api.GetPath()}'.")
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
    elif isinstance(value, (tuple, list)) and len(value) == 3 and any(isinstance(v, float) for v in value):
        sdf_type = Sdf.ValueTypeNames.Float3
    elif isinstance(value, (tuple, list)) and len(value) == 2 and any(isinstance(v, float) for v in value):
        sdf_type = Sdf.ValueTypeNames.Float2
    else:
        raise NotImplementedError(
            f"Cannot set attribute '{attr_name}' with value '{value}'. Please modify the code to support this type."
        )

    # early attach stage to usd context if stage is in memory
    # since stage in memory is not supported by the "ChangePropertyCommand" kit command
    attach_stage_to_usd_context(attaching_early=True)

    # change property
    omni.kit.commands.execute(
        "ChangePropertyCommand",
        prop_path=Sdf.Path(f"{prim.GetPath()}.{attr_name}"),
        value=value,
        prev=None,
        type_to_create_if_not_exist=sdf_type,
        usd_context_name=prim.GetStage(),
    )


"""
Decorators.
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
            omni.log.warn(
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
                # set the semantic API for the instance
                instance_name = f"{semantic_type_sanitized}_{semantic_value_sanitized}"
                sem = Semantics.SemanticsAPI.Apply(prim, instance_name)
                # create semantic type and data attributes
                sem.CreateSemanticTypeAttr()
                sem.CreateSemanticDataAttr()
                sem.GetSemanticTypeAttr().Set(semantic_type)
                sem.GetSemanticDataAttr().Set(semantic_value)
        # activate rigid body contact sensors
        if hasattr(cfg, "activate_contact_sensors") and cfg.activate_contact_sensors:
            schemas.activate_contact_sensors(prim_paths[0], cfg.activate_contact_sensors)
        # clone asset using cloner API
        if len(prim_paths) > 1:
            cloner = Cloner(stage=stage)
            # check version of Isaac Sim to determine whether clone_in_fabric is valid
            isaac_sim_version = float(".".join(get_version()[2]))
            if isaac_sim_version < 5:
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

    .. _Physics material: https://docs.omniverse.nvidia.com/extensions/latest/ext_physics/simulation-control/physics-settings.html#physics-materials

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
    has_physics_scene_api = prim.HasAPI(PhysxSchema.PhysxSceneAPI)
    has_collider = prim.HasAPI(UsdPhysics.CollisionAPI)
    has_deformable_body = prim.HasAPI(PhysxSchema.PhysxDeformableBodyAPI)
    has_particle_system = prim.IsA(PhysxSchema.PhysxParticleSystem)
    if not (has_physics_scene_api or has_collider or has_deformable_body or has_particle_system):
        omni.log.verbose(
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
    # resolve all paths relative to layer path
    omni.usd.resolve_paths(source_layer.identifier, target_layer.identifier)
    # save the stage
    target_layer.Save()


"""
USD Prim properties.
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
    prim: Usd.Prim = stage.GetPrimAtPath(prim_path)
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
        all_prims += child_prim.GetChildren()


"""
USD Stage traversal.
"""


def get_first_matching_child_prim(
    prim_path: str | Sdf.Path, predicate: Callable[[Usd.Prim], bool], stage: Usd.Stage | None = None
) -> Usd.Prim | None:
    """Recursively get the first USD Prim at the path string that passes the predicate function

    Args:
        prim_path: The path of the prim in the stage.
        predicate: The function to test the prims against. It takes a prim as input and returns a boolean.
        stage: The stage where the prim exists. Defaults to None, in which case the current stage is used.

    Returns:
        The first prim on the path that passes the predicate. If no prim passes the predicate, it returns None.

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
        # check if prim passes predicate
        if predicate(child_prim):
            return child_prim
        # add children to list
        all_prims += child_prim.GetChildren()
    return None


def get_all_matching_child_prims(
    prim_path: str | Sdf.Path,
    predicate: Callable[[Usd.Prim], bool] = lambda _: True,
    depth: int | None = None,
    stage: Usd.Stage | None = None,
) -> list[Usd.Prim]:
    """Performs a search starting from the root and returns all the prims matching the predicate.

    Args:
        prim_path: The root prim path to start the search from.
        predicate: The predicate that checks if the prim matches the desired criteria. It takes a prim as input
            and returns a boolean. Defaults to a function that always returns True.
        depth: The maximum depth for traversal, should be bigger than zero if specified.
            Defaults to None (i.e: traversal happens till the end of the tree).
        stage: The stage where the prim exists. Defaults to None, in which case the current stage is used.

    Returns:
        A list containing all the prims matching the predicate.

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
    # check if depth is valid
    if depth is not None and depth <= 0:
        raise ValueError(f"Depth must be bigger than zero, got {depth}.")

    # iterate over all prims under prim-path
    # list of tuples (prim, current_depth)
    all_prims_queue = [(prim, 0)]
    output_prims = []
    while len(all_prims_queue) > 0:
        # get current prim
        child_prim, current_depth = all_prims_queue.pop(0)
        # check if prim passes predicate
        if predicate(child_prim):
            output_prims.append(child_prim)
        # add children to list
        if depth is None or current_depth < depth:
            all_prims_queue += [(child, current_depth + 1) for child in child_prim.GetChildren()]

    return output_prims


def find_first_matching_prim(prim_path_regex: str, stage: Usd.Stage | None = None) -> Usd.Prim | None:
    """Find the first matching prim in the stage based on input regex expression.

    Args:
        prim_path_regex: The regex expression for prim path.
        stage: The stage where the prim exists. Defaults to None, in which case the current stage is used.

    Returns:
        The first prim that matches input expression. If no prim matches, returns None.

    Raises:
        ValueError: If the prim path is not global (i.e: does not start with '/').
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # check prim path is global
    if not prim_path_regex.startswith("/"):
        raise ValueError(f"Prim path '{prim_path_regex}' is not global. It must start with '/'.")
    # need to wrap the token patterns in '^' and '$' to prevent matching anywhere in the string
    pattern = f"^{prim_path_regex}$"
    compiled_pattern = re.compile(pattern)
    # obtain matching prim (depth-first search)
    for prim in stage.Traverse():
        # check if prim passes predicate
        if compiled_pattern.match(prim.GetPath().pathString) is not None:
            return prim
    return None


def find_matching_prims(prim_path_regex: str, stage: Usd.Stage | None = None) -> list[Usd.Prim]:
    """Find all the matching prims in the stage based on input regex expression.

    Args:
        prim_path_regex: The regex expression for prim path.
        stage: The stage where the prim exists. Defaults to None, in which case the current stage is used.

    Returns:
        A list of prims that match input expression.

    Raises:
        ValueError: If the prim path is not global (i.e: does not start with '/').
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # check prim path is global
    if not prim_path_regex.startswith("/"):
        raise ValueError(f"Prim path '{prim_path_regex}' is not global. It must start with '/'.")
    # need to wrap the token patterns in '^' and '$' to prevent matching anywhere in the string
    tokens = prim_path_regex.split("/")[1:]
    tokens = [f"^{token}$" for token in tokens]
    # iterate over all prims in stage (breath-first search)
    all_prims = [stage.GetPseudoRoot()]
    output_prims = []
    for index, token in enumerate(tokens):
        token_compiled = re.compile(token)
        for prim in all_prims:
            for child in prim.GetAllChildren():
                if token_compiled.match(child.GetName()) is not None:
                    output_prims.append(child)
        if index < len(tokens) - 1:
            all_prims = output_prims
            output_prims = []
    return output_prims


def find_matching_prim_paths(prim_path_regex: str, stage: Usd.Stage | None = None) -> list[str]:
    """Find all the matching prim paths in the stage based on input regex expression.

    Args:
        prim_path_regex: The regex expression for prim path.
        stage: The stage where the prim exists. Defaults to None, in which case the current stage is used.

    Returns:
        A list of prim paths that match input expression.

    Raises:
        ValueError: If the prim path is not global (i.e: does not start with '/').
    """
    # obtain matching prims
    output_prims = find_matching_prims(prim_path_regex, stage)
    # convert prims to prim paths
    output_prim_paths = []
    for prim in output_prims:
        output_prim_paths.append(prim.GetPath().pathString)
    return output_prim_paths


def find_global_fixed_joint_prim(
    prim_path: str | Sdf.Path, check_enabled_only: bool = False, stage: Usd.Stage | None = None
) -> UsdPhysics.Joint | None:
    """Find the fixed joint prim under the specified prim path that connects the target to the simulation world.

    A joint is a connection between two bodies. A fixed joint is a joint that does not allow relative motion
    between the two bodies. When a fixed joint has only one target body, it is considered to attach the body
    to the simulation world.

    This function finds the fixed joint prim that has only one target under the specified prim path. If no such
    fixed joint prim exists, it returns None.

    Args:
        prim_path: The prim path to search for the fixed joint prim.
        check_enabled_only: Whether to consider only enabled fixed joints. Defaults to False.
            If False, then all joints (enabled or disabled) are considered.
        stage: The stage where the prim exists. Defaults to None, in which case the current stage is used.

    Returns:
        The fixed joint prim that has only one target. If no such fixed joint prim exists, it returns None.

    Raises:
        ValueError: If the prim path is not global (i.e: does not start with '/').
        ValueError: If the prim path does not exist on the stage.
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # check prim path is global
    if not prim_path.startswith("/"):
        raise ValueError(f"Prim path '{prim_path}' is not global. It must start with '/'.")

    # check if prim exists
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path '{prim_path}' is not valid.")

    fixed_joint_prim = None
    # we check all joints under the root prim and classify the asset as fixed base if there exists
    # a fixed joint that has only one target (i.e. the root link).
    for prim in Usd.PrimRange(prim):
        # note: ideally checking if it is FixedJoint would have been enough, but some assets use "Joint" as the
        # schema name which makes it difficult to distinguish between the two.
        joint_prim = UsdPhysics.Joint(prim)
        if joint_prim:
            # if check_enabled_only is True, we only consider enabled joints
            if check_enabled_only and not joint_prim.GetJointEnabledAttr().Get():
                continue
            # check body 0 and body 1 exist
            body_0_exist = joint_prim.GetBody0Rel().GetTargets() != []
            body_1_exist = joint_prim.GetBody1Rel().GetTargets() != []
            # if either body 0 or body 1 does not exist, we have a fixed joint that connects to the world
            if not (body_0_exist and body_1_exist):
                fixed_joint_prim = joint_prim
                break

    return fixed_joint_prim


"""
Stage management.
"""


def attach_stage_to_usd_context(attaching_early: bool = False):
    """Attaches stage in memory to usd context.

    This function should be called during or after scene is created and before stage is simulated or rendered.

    Note:
        If the stage is not in memory or rendering is not enabled, this function will return without attaching.

    Args:
        attaching_early: Whether to attach the stage to the usd context before stage is created. Defaults to False.
    """

    from isaacsim.core.simulation_manager import SimulationManager

    from isaaclab.sim.simulation_context import SimulationContext

    # if Isaac Sim version is less than 5.0, stage in memory is not supported
    isaac_sim_version = float(".".join(get_version()[2]))
    if isaac_sim_version < 5:
        return

    # if stage is not in memory, we can return early
    if not is_current_stage_in_memory():
        return

    # attach stage to physx
    stage_id = get_current_stage_id()
    physx_sim_interface = omni.physx.get_physx_simulation_interface()
    physx_sim_interface.attach_stage(stage_id)

    # this carb flag is equivalent to if rendering is enabled
    carb_setting = carb.settings.get_settings()
    is_rendering_enabled = get_carb_setting(carb_setting, "/physics/fabricUpdateTransformations")

    # if rendering is not enabled, we don't need to attach it
    if not is_rendering_enabled:
        return

    # early attach warning msg
    if attaching_early:
        omni.log.warn(
            "Attaching stage in memory to USD context early to support an operation which doesn't support stage in"
            " memory."
        )

    # skip this callback to avoid wiping the stage after attachment
    SimulationContext.instance().skip_next_stage_open_callback()

    # disable stage open callback to avoid clearing callbacks
    SimulationManager.enable_stage_open_callback(False)

    # enable physics fabric
    SimulationContext.instance()._physics_context.enable_fabric(True)

    # attach stage to usd context
    omni.usd.get_context().attach_stage_with_callback(stage_id)

    # attach stage to physx
    physx_sim_interface = omni.physx.get_physx_simulation_interface()
    physx_sim_interface.attach_stage(stage_id)

    # re-enable stage open callback
    SimulationManager.enable_stage_open_callback(True)


def is_current_stage_in_memory() -> bool:
    """This function checks if the current stage is in memory.

    Compares the stage id of the current stage with the stage id of the context stage.

    Returns:
        If the current stage is in memory.
    """

    # grab current stage id
    stage_id = get_current_stage_id()

    # grab context stage id
    context_stage = omni.usd.get_context().get_stage()
    with use_stage(context_stage):
        context_stage_id = get_current_stage_id()

    # check if stage ids are the same
    return stage_id != context_stage_id


"""
USD Variants.
"""


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
        variants = variants.to_dict()

    existing_variant_sets = prim.GetVariantSets()
    for variant_set_name, variant_selection in variants.items():
        # Check if the variant set exists on the prim.
        if not existing_variant_sets.HasVariantSet(variant_set_name):
            omni.log.warn(f"Variant set '{variant_set_name}' does not exist on prim '{prim_path}'.")
            continue

        variant_set = existing_variant_sets.GetVariantSet(variant_set_name)
        # Only set the variant selection if it is different from the current selection.
        if variant_set.GetVariantSelection() != variant_selection:
            variant_set.SetVariantSelection(variant_selection)
            omni.log.info(
                f"Setting variant selection '{variant_selection}' for variant set '{variant_set_name}' on"
                f" prim '{prim_path}'."
            )


"""
Isaac Sim stage utils wrappers to enable backwards compatibility to Isaac Sim 4.5
"""


@contextlib.contextmanager
def use_stage(stage: Usd.Stage) -> None:
    """Context manager that sets a thread-local stage, if supported.

    In Isaac Sim < 5.0, this is a no-op to maintain compatibility.

    Args:
        stage (Usd.Stage): The stage to set temporarily.
    """
    isaac_sim_version = float(".".join(get_version()[2]))
    if isaac_sim_version < 5:
        omni.log.warn("[Compat] Isaac Sim < 5.0 does not support thread-local stage contexts. Skipping use_stage().")
        yield  # no-op
    else:
        with stage_utils.use_stage(stage):
            yield


def create_new_stage_in_memory() -> Usd.Stage:
    """Create a new stage in memory, if supported.

    Returns:
        The new stage.
    """
    isaac_sim_version = float(".".join(get_version()[2]))
    if isaac_sim_version < 5:
        omni.log.warn(
            "[Compat] Isaac Sim < 5.0 does not support creating a new stage in memory. Falling back to creating a new"
            " stage attached to USD context."
        )
        return stage_utils.create_new_stage()
    else:
        return stage_utils.create_new_stage_in_memory()


def get_current_stage_id() -> int:
    """Get the current open stage id.

    Reimplementation of stage_utils.get_current_stage_id() for Isaac Sim < 5.0.

    Returns:
        int: The stage id.
    """
    stage = get_current_stage()
    stage_cache = UsdUtils.StageCache.Get()
    stage_id = stage_cache.GetId(stage).ToLongInt()
    if stage_id < 0:
        stage_id = stage_cache.Insert(stage).ToLongInt()
    return stage_id
