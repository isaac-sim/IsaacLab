# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for querying the USD stage."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

from isaaclab.cloner.cloner_utils import resolve_clone_plan_source
from isaaclab.sim.simulation_context import SimulationContext

from .stage import get_current_stage

if TYPE_CHECKING:
    from pxr import Sdf, Usd, UsdPhysics  # noqa: F401

# import logger
logger = logging.getLogger(__name__)


def get_next_free_prim_path(path: str, stage: Usd.Stage | None = None) -> str:
    """Gets a new prim path that doesn't exist in the stage given a base path.

    If the given path doesn't exist in the stage already, it returns the given path. Otherwise,
    it appends a suffix with an incrementing number to the given path.

    Args:
        path: The base prim path to check.
        stage: The stage to check. Defaults to the current stage.

    Returns:
        A new path that is guaranteed to not exist on the current stage

    Raises:
        ValueError: If the path is not a valid prim path string.

    Example:
        >>> import isaaclab.sim as sim_utils
        >>>
        >>> # given the stage: /World/Cube, /World/Cube_01.
        >>> # Get the next available path for /World/Cube
        >>> sim_utils.get_next_free_prim_path("/World/Cube")
        /World/Cube_02
    """
    from pxr import Sdf  # noqa: PLC0415

    # get current stage
    stage = get_current_stage() if stage is None else stage

    # validate and convert path
    if not Sdf.Path.IsValidPathString(path):
        raise ValueError(f"'{path}' is not a valid prim path")
    sdf_path = Sdf.Path(path)

    # ensure path is absolute
    corrected_path = sdf_path.MakeAbsolutePath(Sdf.Path.absoluteRootPath)
    if sdf_path != corrected_path:
        logger.warning(f"Path '{sdf_path}' auto-corrected to '{corrected_path}'.")
        sdf_path = corrected_path

    # prepend default prim if needed
    if stage.HasDefaultPrim():
        default_prim = stage.GetDefaultPrim()
        if default_prim and not (sdf_path.HasPrefix(default_prim.GetPath()) and sdf_path != default_prim.GetPath()):
            sdf_path = sdf_path.ReplacePrefix(Sdf.Path.absoluteRootPath, default_prim.GetPath())

    def _increment_path(path_str: str) -> str:
        match = re.search(r"_(\d+)$", path_str)
        if match:
            new_num = int(match.group(1)) + 1
            return re.sub(r"_(\d+)$", f"_{new_num:02d}", path_str)
        return path_str + "_01"

    path_string = sdf_path.pathString
    while stage.GetPrimAtPath(path_string).IsValid():
        path_string = _increment_path(path_string)

    return path_string


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
    from pxr import Usd  # noqa: PLC0415

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
    expected_num_matches: int | None = None,
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
        expected_num_matches: Expected number of matching prims. If specified, a :class:`RuntimeError` is raised when
            the number of matches differs. Defaults to None, which disables count validation.

    Returns:
        A list containing all the prims matching the predicate.

    Raises:
        ValueError: If the prim path is not global (i.e: does not start with '/').
        ValueError: If :attr:`expected_num_matches` is negative.
        RuntimeError: If :attr:`expected_num_matches` is specified and the number of matches differs.
    """
    from pxr import Usd  # noqa: PLC0415

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
    if expected_num_matches is not None and expected_num_matches < 0:
        raise ValueError(f"Expected number of matches must be non-negative, got {expected_num_matches}.")

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

    if expected_num_matches is not None and len(output_prims) != expected_num_matches:
        matched = [prim.GetPath().pathString for prim in output_prims]
        predicate_name = getattr(predicate, "__name__", None)
        predicate_msg = ""
        if predicate_name is not None and predicate_name != "<lambda>":
            predicate_msg = f" matching predicate '{predicate_name}'"
        actual_num_matches = len(output_prims)
        raise RuntimeError(
            f"Expected {expected_num_matches} prims under '{prim_path}'{predicate_msg}, "
            f"found {actual_num_matches}: {matched}."
        )
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


def matches_path_expr_prefix(path_expr: str, prim_path: str) -> bool:
    """Return whether ``prim_path`` matches ``path_expr`` up to ``prim_path`` depth."""
    prefix_expr = "/".join(path_expr.split("/")[: prim_path.count("/") + 1])
    return re.match(f"^{_normalize_legacy_wildcard_pattern(prefix_expr)}$", prim_path) is not None


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

    # normalize legacy wildcard pattern
    prim_path_regex = _normalize_legacy_wildcard_pattern(prim_path_regex)

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


def resolve_matching_prims_from_source(
    path_expr: str,
    predicate: Callable[[Usd.Prim], bool] | None = None,
    expected_num_matches: int | None = None,
    env_regex_ns: str = "/World/envs/env_.*",
    raise_if_no_matches: bool = True,
    traverse_instance_prims: bool = True,
) -> list[tuple[Usd.Prim, str]]:
    """Resolve matching prims from a single(source) instance when multiple instances are present.

    The returned prims come from the stage source instance, while each destination expression
    keeps the multi-instance pattern that callers can pass to simulation views.

    Args:
        path_expr: Prim path expression to resolve. It may contain regex wildcards.
        predicate: Optional descendant filter; returned expressions include the matching descendant suffix.
        expected_num_matches: Optional exact result count.
        env_regex_ns: Namespace pattern that marks one instance root when no clone plan applies.
        raise_if_no_matches: Whether to raise if no prim matches ``path_expr``. Defaults to True.
        traverse_instance_prims: Whether to traverse instance prims when applying ``predicate``.

    Returns:
        A list of ``(source_prim, destination_expr)`` pairs. Empty only when
        ``raise_if_no_matches`` is False.

    Raises:
        RuntimeError: If no prim matches ``path_expr`` and ``raise_if_no_matches`` is True.
    """
    plan = SimulationContext.instance().get_clone_plan()
    resolved = resolve_clone_plan_source(path_expr, plan) if plan is not None else None
    if resolved is not None:
        source_path, dest_glob, asset_suffix = resolved
        walk_root = source_path + asset_suffix
        results = [
            (prim, dest_glob + prim.GetPath().pathString[len(source_path) :]) for prim in find_matching_prims(walk_root)
        ]
    else:
        # No clone plan, or ``path_expr`` is not owned by any plan row. Resolve from the stage
        # in two phases (mirroring the clone-plan branch above): (1) locate ONE instance root to
        # search from, (2) collect the bodies of interest within just that instance and map each
        # back to the multi-instance pattern. Phase 1 stops at the first match and phase 2 walks
        # under a concrete instance prefix, so only a single instance subtree is traversed.
        segments = path_expr.strip("/").split("/")
        ns_segments = env_regex_ns.strip("/").split("/")
        # Instance ("env") boundary. Assume the standard namespace ``env_regex_ns`` and put the
        # boundary at its depth when ``path_expr`` sits under it -- literal ns segments must
        # match, wildcard ns segments (e.g. ``env_.*``) accept any segment. Otherwise fall back
        # to the first regex segment of ``path_expr`` (treat the first ``.*`` as the env id),
        # covering ad-hoc roots like ``/World/Table_.*/Object``. A layout the fallback would
        # mis-split (e.g. multiple wildcard levels) must pass ``env_regex_ns``.
        under_ns = len(segments) >= len(ns_segments) and all(
            ns_seg == seg or not ns_seg.isidentifier() for ns_seg, seg in zip(ns_segments, segments)
        )
        if under_ns:
            instance_seg = len(ns_segments) - 1
        else:
            instance_seg = next((i for i, seg in enumerate(segments) if not seg.isidentifier()), None)
        first = find_first_matching_prim(path_expr)
        if first is None:
            results = []
        elif instance_seg is None:
            # Fully concrete path: a single instance, mapped to itself.
            results = [(first, first.GetPath().pathString)]
        else:
            instance_expr = "/" + "/".join(segments[: instance_seg + 1])
            match_segments = first.GetPath().pathString.strip("/").split("/")
            instance_root = "/" + "/".join(match_segments[: instance_seg + 1])
            trailing = segments[instance_seg + 1 :]
            walk_root = instance_root + ("/" + "/".join(trailing) if trailing else "")
            results = [
                (prim, instance_expr + prim.GetPath().pathString[len(instance_root) :])
                for prim in find_matching_prims(walk_root)
                if prim.GetPath().pathString == instance_root
                or prim.GetPath().pathString.startswith(instance_root + "/")
            ]
    if predicate is not None:
        results = [
            (child, dest + child.GetPath().pathString[len(source.GetPath().pathString) :])
            for source, dest in results
            for child in get_all_matching_child_prims(
                source.GetPath(), predicate, traverse_instance_prims=traverse_instance_prims
            )
        ]

    if expected_num_matches is not None and len(results) != expected_num_matches:
        raise RuntimeError(f"Expected {expected_num_matches} prims at '{path_expr}', found {len(results)}.")
    if raise_if_no_matches and not results:
        raise RuntimeError(f"No prim found at '{path_expr}'.")
    return results


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
    from pxr import Usd, UsdPhysics  # noqa: PLC0415

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
