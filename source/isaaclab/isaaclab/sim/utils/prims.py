# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
import inspect
import logging
import re
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import omni
import omni.kit.commands
import omni.usd
import usdrt  # noqa: F401
from isaacsim.core.cloner import Cloner
from isaacsim.core.version import get_version
from omni.usd.commands import DeletePrimsCommand, MovePrimCommand
from pxr import PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from isaaclab.utils.string import to_camel_case

from .semantics import add_labels
from .stage import attach_stage_to_usd_context, get_current_stage, get_current_stage_id

if TYPE_CHECKING:
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg

# from Isaac Sim 4.2 onwards, pxr.Semantics is deprecated
try:
    import Semantics
except ModuleNotFoundError:
    from pxr import Semantics

# import logger
logger = logging.getLogger(__name__)


"""
General Utils
"""


def create_prim(
    prim_path: str,
    prim_type: str = "Xform",
    position: Sequence[float] | None = None,
    translation: Sequence[float] | None = None,
    orientation: Sequence[float] | None = None,
    scale: Sequence[float] | None = None,
    usd_path: str | None = None,
    semantic_label: str | None = None,
    semantic_type: str = "class",
    attributes: dict | None = None,
    stage: Usd.Stage | None = None,
) -> Usd.Prim:
    """Create a prim into current USD stage.

    The method applies specified transforms, the semantic label and set specified attributes.

    Args:
        prim_path: The path of the new prim.
        prim_type: Prim type name
        position: prim position (applied last)
        translation: prim translation (applied last)
        orientation: prim rotation as quaternion
        scale: scaling factor in x, y, z.
        usd_path: Path to the USD that this prim will reference.
        semantic_label: Semantic label.
        semantic_type: set to "class" unless otherwise specified.
        attributes: Key-value pairs of prim attributes to set.
        stage: The stage to create the prim in. Defaults to None, in which case the current stage is used.

    Raises:
        Exception: If there is already a prim at the prim_path

    Returns:
        The created USD prim.

    Example:

    .. code-block:: python

        >>> import numpy as np
        >>> import isaaclab.sim as sim_utils
        >>>
        >>> # create a cube (/World/Cube) of size 2 centered at (1.0, 0.5, 0.0)
        >>> sim_utils.create_prim(
        ...     prim_path="/World/Cube",
        ...     prim_type="Cube",
        ...     position=np.array([1.0, 0.5, 0.0]),
        ...     attributes={"size": 2.0}
        ... )
        Usd.Prim(</World/Cube>)

    .. code-block:: python

        >>> import isaaclab.sim as sim_utils
        >>>
        >>> # load an USD file (franka.usd) to the stage under the path /World/panda
        >>> sim_utils.create_prim(
        ...     prim_path="/World/panda",
        ...     prim_type="Xform",
        ...     usd_path="/home/<user>/Documents/Assets/Robots/FrankaRobotics/FrankaPanda/franka.usd"
        ... )
        Usd.Prim(</World/panda>)
    """
    # Note: Imported here to prevent cyclic dependency in the module.
    from isaacsim.core.prims import XFormPrim

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
        add_usd_reference(usd_path=usd_path, prim_path=prim_path, stage=stage)
    # add semantic label to prim
    if semantic_label is not None:
        add_labels(prim, labels=[semantic_label], instance_name=semantic_type)

    # apply the transformations
    from isaacsim.core.api.simulation_context.simulation_context import SimulationContext

    if SimulationContext.instance() is None:
        # FIXME: remove this, we should never even use backend utils  especially not numpy ones
        import isaacsim.core.utils.numpy as backend_utils

        device = "cpu"
    else:
        backend_utils = SimulationContext.instance().backend_utils
        device = SimulationContext.instance().device
    if position is not None:
        position = backend_utils.expand_dims(backend_utils.convert(position, device), 0)
    if translation is not None:
        translation = backend_utils.expand_dims(backend_utils.convert(translation, device), 0)
    if orientation is not None:
        orientation = backend_utils.expand_dims(backend_utils.convert(orientation, device), 0)
    if scale is not None:
        scale = backend_utils.expand_dims(backend_utils.convert(scale, device), 0)
    XFormPrim(prim_path, positions=position, translations=translation, orientations=orientation, scales=scale)

    return prim


def delete_prim(prim_path: str | list[str]) -> None:
    """Remove the USD Prim and its descendants from the scene if able.

    Args:
        prim_path: The path of the prim to delete. If a list of paths is provided,
            the function will delete all the prims in the list.

    Example:

    .. code-block:: python

        >>> import isaaclab.sim as sim_utils
        >>>
        >>> sim_utils.delete_prim("/World/Cube")
    """
    # convert prim_path to list if it is a string
    if isinstance(prim_path, str):
        prim_path = [prim_path]
    # delete prims
    DeletePrimsCommand(prim_path).do()


def from_prim_path_get_type_name(prim_path: str, fabric: bool = False) -> str:
    """Get the TypeName of the USD Prim at the path if it is valid

    Args:
        prim_path: path of the prim in the stage
        fabric: True for fabric stage and False for USD stage. Defaults to False.

    Returns:
        The TypeName of the USD Prim at the path string
    """
    stage = get_current_stage(fabric=fabric)

    if not stage.GetPrimAtPath(prim_path).IsValid():
        raise Exception(f"A prim does not exist at prim path: {prim_path}")

    prim = stage.GetPrimAtPath(prim_path)

    # TODO: Check if GetTypeName is directly available in USD API.
    if fabric:
        return prim.GetTypeName()
    else:
        return prim.GetPrimTypeInfo().GetTypeName()


def move_prim(path_from: str, path_to: str) -> None:
    """Run the Move command to change a prims USD Path in the stage

    Args:
        path_from: Path of the USD Prim you wish to move
        path_to: Final destination of the prim

    Example:

    .. code-block:: python

        >>> import isaaclab.sim as sim_utils
        >>>
        >>> # given the stage: /World/Cube. Move the prim Cube outside the prim World
        >>> sim_utils.move_prim("/World/Cube", "/Cube")
    """
    MovePrimCommand(path_from=path_from, path_to=path_to).do()


"""
USD Stage traversal.
"""


def get_next_free_prim_path(path: str, stage: Usd.Stage | None = None) -> str:
    """Gets a new prim path that doesn't exist in the stage given a base path.

    If the given path doesn't exist in the stage already, it returns the given path. Otherwise,
    it appends a suffix with an incrementing number to the given path.

    Args:
        path: The base prim path to check.
        stage: The stage to check. Defaults to the current stage.

    Returns:
        A new path that is guaranteed to not exist on the current stage

    Example:

    .. code-block:: python

        >>> import isaaclab.sim as sim_utils
        >>>
        >>> # given the stage: /World/Cube, /World/Cube_01.
        >>> # Get the next available path for /World/Cube
        >>> sim_utils.get_next_free_prim_path("/World/Cube")
        /World/Cube_02
    """
    # get current stage
    stage = get_current_stage() if stage is None else stage
    # get next free path
    return omni.usd.get_stage_next_free_path(stage, path, True)


def get_first_matching_ancestor_prim(
    prim_path: str | Sdf.Path,
    predicate: Callable[[Usd.Prim], bool],
    stage: Usd.Stage | None = None,
) -> Usd.Prim | None:
    """Gets the first ancestor prim that passes the predicate function.

    This function walks up the prim hierarchy starting from the target prim and returns the first ancestor prim
    that passes the predicate function. This includes the prim itself if it passes the predicate.

    Args:
        prim_path: The path of the prim in the stage.
        predicate: The function to test the prims against. It takes a prim as input and returns a boolean.
        stage: The stage where the prim exists. Defaults to None, in which case the current stage is used.

    Returns:
        The first ancestor prim that passes the predicate. If no ancestor prim passes the predicate, it returns None.

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

    # walk up to find the first matching ancestor prim
    ancestor_prim = prim
    while ancestor_prim and ancestor_prim.IsValid():
        # check if prim passes predicate
        if predicate(ancestor_prim):
            return ancestor_prim
        # get parent prim
        ancestor_prim = ancestor_prim.GetParent()

    # If no ancestor prim passes the predicate, return None
    return None


def get_first_matching_child_prim(
    prim_path: str | Sdf.Path,
    predicate: Callable[[Usd.Prim], bool],
    stage: Usd.Stage | None = None,
    traverse_instance_prims: bool = True,
) -> Usd.Prim | None:
    """Recursively get the first USD Prim at the path string that passes the predicate function.

    This function performs a depth-first traversal of the prim hierarchy starting from
    :attr:`prim_path`, returning the first prim that satisfies the provided :attr:`predicate`.
    It optionally supports traversal through instance prims, which are normally skipped in standard USD
    traversals.

    USD instance prims are lightweight copies of prototype scene structures and are not included
    in default traversals unless explicitly handled. This function allows traversing into instances
    when :attr:`traverse_instance_prims` is set to :attr:`True`.

    .. versionchanged:: 2.3.0

        Added :attr:`traverse_instance_prims` to control whether to traverse instance prims.
        By default, instance prims are now traversed.

    Args:
        prim_path: The path of the prim in the stage.
        predicate: The function to test the prims against. It takes a prim as input and returns a boolean.
        stage: The stage where the prim exists. Defaults to None, in which case the current stage is used.
        traverse_instance_prims: Whether to traverse instance prims. Defaults to True.

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
        if traverse_instance_prims:
            all_prims += child_prim.GetFilteredChildren(Usd.TraverseInstanceProxies())
        else:
            all_prims += child_prim.GetChildren()
    return None


def get_all_matching_child_prims(
    prim_path: str | Sdf.Path,
    predicate: Callable[[Usd.Prim], bool] = lambda _: True,
    depth: int | None = None,
    stage: Usd.Stage | None = None,
    traverse_instance_prims: bool = True,
) -> list[Usd.Prim]:
    """Performs a search starting from the root and returns all the prims matching the predicate.

    This function performs a depth-first traversal of the prim hierarchy starting from
    :attr:`prim_path`, returning all prims that satisfy the provided :attr:`predicate`. It optionally
    supports traversal through instance prims, which are normally skipped in standard USD traversals.

    USD instance prims are lightweight copies of prototype scene structures and are not included
    in default traversals unless explicitly handled. This function allows traversing into instances
    when :attr:`traverse_instance_prims` is set to :attr:`True`.

    .. versionchanged:: 2.3.0

        Added :attr:`traverse_instance_prims` to control whether to traverse instance prims.
        By default, instance prims are now traversed.

    Args:
        prim_path: The root prim path to start the search from.
        predicate: The predicate that checks if the prim matches the desired criteria. It takes a prim as input
            and returns a boolean. Defaults to a function that always returns True.
        depth: The maximum depth for traversal, should be bigger than zero if specified.
            Defaults to None (i.e: traversal happens till the end of the tree).
        stage: The stage where the prim exists. Defaults to None, in which case the current stage is used.
        traverse_instance_prims: Whether to traverse instance prims. Defaults to True.

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
            # resolve prims under the current prim
            if traverse_instance_prims:
                children = child_prim.GetFilteredChildren(Usd.TraverseInstanceProxies())
            else:
                children = child_prim.GetChildren()
            # add children to list
            all_prims_queue += [(child, current_depth + 1) for child in children]

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
    prim_path_regex = _normalize_legacy_wildcard_pattern(prim_path_regex)
    # need to wrap the token patterns in '^' and '$' to prevent matching anywhere in the string
    pattern = f"^{prim_path_regex}$"
    compiled_pattern = re.compile(pattern)
    # obtain matching prim (depth-first search)
    for prim in stage.Traverse():
        # check if prim passes predicate
        if compiled_pattern.match(prim.GetPath().pathString) is not None:
            return prim
    return None


def _normalize_legacy_wildcard_pattern(prim_path_regex: str) -> str:
    """Convert legacy '*' wildcard usage to '.*' and warn users."""
    fixed_regex = re.sub(r"(?<![\\\.])\*", ".*", prim_path_regex)
    if fixed_regex != prim_path_regex:
        logger.warning(
            "Using '*' as a wildcard in prim path regex is deprecated; automatically converting '%s' to '%s'. "
            "Please update your pattern to use '.*' explicitly.",
            prim_path_regex,
            fixed_regex,
        )
    return fixed_regex


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
    prim_path_regex = _normalize_legacy_wildcard_pattern(prim_path_regex)
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
        in the (w, x, y, z) format.

    Raises:
        ValueError: If the prim or ref prim is not valid.
    """
    # check if prim is valid
    if not prim.IsValid():
        raise ValueError(f"Prim at path '{prim.GetPath().pathString}' is not valid.")
    # get prim xform
    xform = UsdGeom.Xformable(prim)
    prim_tf = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    # sanitize quaternion
    # this is needed, otherwise the quaternion might be non-normalized
    prim_tf = prim_tf.GetOrthonormalized()

    if ref_prim is not None:
        # check if ref prim is valid
        if not ref_prim.IsValid():
            raise ValueError(f"Ref prim at path '{ref_prim.GetPath().pathString}' is not valid.")
        # get ref prim xform
        ref_xform = UsdGeom.Xformable(ref_prim)
        ref_tf = ref_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        # make sure ref tf is orthonormal
        ref_tf = ref_tf.GetOrthonormalized()
        # compute relative transform to get prim in ref frame
        prim_tf = prim_tf * ref_tf.GetInverse()

    # extract position and orientation
    prim_pos = [*prim_tf.ExtractTranslation()]
    prim_quat = [prim_tf.ExtractRotationQuat().real, *prim_tf.ExtractRotationQuat().imaginary]
    return tuple(prim_pos), tuple(prim_quat)


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
    """
    # check if prim is valid
    if not prim.IsValid():
        raise ValueError(f"Prim at path '{prim.GetPath().pathString}' is not valid.")
    # compute local to world transform
    xform = UsdGeom.Xformable(prim)
    world_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    # extract scale
    return tuple([*(v.GetLength() for v in world_transform.ExtractRotationMatrix())])


def set_prim_visibility(prim: Usd.Prim, visible: bool) -> None:
    """Sets the visibility of the prim in the opened stage.

    .. note::

        The method does this through the USD API.

    Args:
        prim: the USD prim
        visible: flag to set the visibility of the usd prim in stage.

    Example:

    .. code-block:: python

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
                # set the semantic API for the instance
                instance_name = f"{semantic_type_sanitized}_{semantic_value_sanitized}"
                sem = Semantics.SemanticsAPI.Apply(prim, instance_name)
                # create semantic type and data attributes
                sem.CreateSemanticTypeAttr()
                sem.CreateSemanticDataAttr()
                sem.GetSemanticTypeAttr().Set(semantic_type)
                sem.GetSemanticDataAttr().Set(semantic_value)
        # activate rigid body contact sensors (lazy import to avoid circular import with schemas)
        if hasattr(cfg, "activate_contact_sensors") and cfg.activate_contact_sensors:
            from ..schemas import schemas as _schemas

            _schemas.activate_contact_sensors(prim_paths[0])
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
    has_physics_scene_api = prim.HasAPI(PhysxSchema.PhysxSceneAPI)
    has_collider = prim.HasAPI(UsdPhysics.CollisionAPI)
    has_deformable_body = prim.HasAPI(PhysxSchema.PhysxDeformableBodyAPI)
    has_particle_system = prim.IsA(PhysxSchema.PhysxParticleSystem)
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
    usd_path: str, prim_path: str, prim_type: str = "Xform", stage: Usd.Stage | None = None
) -> Usd.Prim:
    """Adds a USD reference at the specified prim path on the provided stage.

    This function adds a reference to an external USD file at the specified prim path on the provided stage.
    If the prim does not exist, it will be created with the specified type.

    The function also handles stage units verification to ensure compatibility. For instance,
    if the current stage is in meters and the referenced USD file is in centimeters, the function will
    convert the units to match. This is done using the :mod:`omni.metrics.assembler` functionality.

    Args:
        usd_path: The path to USD file to reference.
        prim_path: The prim path where the reference will be attached.
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

    # get isaac sim version
    isaac_sim_version = float(".".join(get_version()[2]))
    # Compatibility with Isaac Sim 4.5 where omni.metrics is not available
    if isaac_sim_version < 5:
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
    if ret_val["ret_val"]:
        try:
            import omni.metrics.assembler.ui

            omni.kit.commands.execute(
                "AddReference", stage=stage, prim_path=prim.GetPath(), reference=Sdf.Reference(usd_path)
            )

            return prim
        except Exception:
            return _add_reference_to_prim(prim)
    else:
        return _add_reference_to_prim(prim)


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
