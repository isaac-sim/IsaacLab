# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for legacy functionality.

This sub-module contains legacy functions from Isaac Sim that are no longer
required for Isaac Lab. Most functions are simple wrappers around USD APIs
and are provided mainly for convenience.

It is recommended to use the USD APIs directly whenever possible.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from pxr import Usd, UsdGeom

from .prims import add_usd_reference
from .queries import get_next_free_prim_path
from .stage import get_current_stage

# import logger
logger = logging.getLogger(__name__)


"""
Stage utilities.
"""


def add_reference_to_stage(usd_path: str, path: str, prim_type: str = "Xform") -> Usd.Prim:
    """Adds a USD reference to the stage at the specified prim path.

    .. deprecated:: 2.3.0
        This function is deprecated. Please use the :func:`isaaclab.sim.utils.prims.add_usd_reference` function instead.

    Args:
        usd_path: The path to the USD file to reference.
        path: The prim path to add the reference to.
        prim_type: The type of prim to create if it doesn't exist. Defaults to "Xform".

    Returns:
        The USD prim at the specified prim path.
    """
    logger.warning("Function 'add_reference_to_stage' is deprecated. Please use 'add_usd_reference' instead.")
    return add_usd_reference(prim_path=path, usd_path=usd_path, prim_type=prim_type)


def get_stage_up_axis() -> str:
    """Gets the up axis of the stage.

    .. deprecated:: 2.3.0
        This function is deprecated. Please use the USD APIs directly instead.

        >>> import isaaclab.sim as sim_utils
        >>> from pxr import UsdGeom
        >>>
        >>> UsdGeom.GetStageUpAxis(sim_utils.get_current_stage())
        'Z'
    """
    msg = """Function 'get_stage_up_axis' is deprecated. Please use the USD APIs directly instead.

    Example:
        >>> import isaaclab.sim as sim_utils
        >>> from pxr import UsdGeom
        >>>
        >>> UsdGeom.GetStageUpAxis(sim_utils.get_current_stage())
        'Z'
    """
    logger.warning(msg)
    return UsdGeom.GetStageUpAxis(get_current_stage())


def traverse_stage(fabric: bool = False) -> Iterable[Usd.Prim]:
    """Traverses the stage and returns all the prims.

    .. deprecated:: 2.3.0
        This function is deprecated. Please use the USD APIs directly instead.

        >>> import isaaclab.sim as sim_utils
        >>>
        >>> stage = sim_utils.get_current_stage()
        >>> for prim in stage.Traverse():
        >>>     print(prim)
        Usd.Prim(</World>)
        Usd.Prim(</World/Cube>)
        Usd.Prim(</World/Cube_01>)
        Usd.Prim(</World/Cube_02>)

    Args:
        fabric: True for fabric stage and False for USD stage. Defaults to False.

    Returns:
        An iterable of all the prims in the stage.
    """
    msg = """Function 'traverse_stage' is deprecated. Please use the USD APIs directly instead.

    Example:
        >>> import isaaclab.sim as sim_utils
        >>>
        >>> stage = sim_utils.get_current_stage()
        >>> for prim in stage.Traverse():
        >>>     print(prim)
    """
    logger.warning(msg)
    # get current stage
    stage = get_current_stage(fabric=fabric)
    # traverse stage
    return stage.Traverse()


"""
Prims utilities.
"""


def get_prim_at_path(prim_path: str, fabric: bool = False) -> Usd.Prim | None:
    """Gets the USD prim at the specified path.

    .. deprecated:: 2.3.0
        This function is deprecated. Please use the USD APIs directly instead.

        >>> import isaaclab.sim as sim_utils
        >>>
        >>> stage = sim_utils.get_current_stage()
        >>> stage.GetPrimAtPath("/World/Cube")
        Usd.Prim(</World/Cube>)

    Args:
        prim_path: The path of the prim to get.
        fabric: Whether to get the prim from the fabric stage. Defaults to False.

    Returns:
        The USD prim at the specified path. If stage is not found, returns None.
    """
    msg = """Function 'get_prim_at_path' is deprecated. Please use the USD APIs directly instead.

    Example:
        >>> import isaaclab.sim as sim_utils
        >>>
        >>> stage = sim_utils.get_current_stage()
        >>> stage.GetPrimAtPath("/World/Cube")
        Usd.Prim(</World/Cube>)
    """
    logger.warning(msg)
    # get current stage
    stage = get_current_stage(fabric=fabric)
    if stage is not None:
        return stage.GetPrimAtPath(prim_path)
    return None


def get_prim_path(prim: Usd.Prim) -> str:
    """Gets the path of the specified USD prim.

    .. deprecated:: 2.3.0
        This function is deprecated. Please use the USD APIs directly instead.

        >>> import isaaclab.sim as sim_utils
        >>>
        >>> stage = sim_utils.get_current_stage()
        >>> prim = stage.GetPrimAtPath("/World/Cube")
        >>> prim.GetPath().pathString
        "/World/Cube"

    Args:
        prim: The USD prim to get the path of.

    Returns:
        The path of the specified USD prim.
    """
    msg = """Function 'get_prim_path' is deprecated. Please use the USD APIs directly instead.

    Example:
        >>> import isaaclab.sim as sim_utils
        >>>
        >>> stage = sim_utils.get_current_stage()
        >>> prim = stage.GetPrimAtPath("/World/Cube")
        >>> prim.GetPath().pathString
        "/World/Cube"
    """
    logger.warning(msg)
    return prim.GetPath().pathString if prim.IsValid() else ""


def is_prim_path_valid(prim_path: str, fabric: bool = False) -> bool:
    """Check if a path has a valid USD Prim on the specified stage.

    .. deprecated:: 2.3.0
        This function is deprecated. Please use the USD APIs directly instead.

        >>> import isaaclab.sim as sim_utils
        >>>
        >>> stage = sim_utils.get_current_stage()
        >>> prim = stage.GetPrimAtPath("/World/Cube")
        >>> prim.IsValid()
        True

    Args:
        prim_path: path of the prim in the stage
        fabric: True for fabric stage and False for USD stage. Defaults to False.

    Returns:
        True if the path points to a valid prim. False otherwise.
    """
    msg = """Function 'is_prim_path_valid' is deprecated. Please use the USD APIs directly instead.

    Example:
        >>> import isaaclab.sim as sim_utils
        >>>
        >>> stage = sim_utils.get_current_stage()
        >>> prim = stage.GetPrimAtPath("/World/Cube")
        >>> prim.IsValid()
        True
    """
    logger.warning(msg)
    # get prim at path
    prim = get_prim_at_path(prim_path, fabric=fabric)
    # return validity
    return prim.IsValid() if prim else False


def define_prim(prim_path: str, prim_type: str = "Xform", fabric: bool = False) -> Usd.Prim:
    """Create a USD Prim at the given prim_path of type prim type unless one already exists.

    This function creates a prim of the specified type in the specified path. To apply a
    transformation (position, orientation, scale), set attributes or load an USD file while
    creating the prim use the :func:`isaaclab.sim.utils.prims.create_prim` function.

    .. deprecated:: 2.3.0
        This function is deprecated. Please use the USD APIs directly instead.
        In case, a new prim is needed, use the :func:`isaaclab.sim.utils.prims.create_prim`
        function instead.

        >>> import isaaclab.sim as sim_utils
        >>>
        >>> stage = sim_utils.get_current_stage()
        >>> stage.DefinePrim("/World/Shapes", "Xform")
        Usd.Prim(</World/Shapes>)

    Args:
        prim_path: path of the prim in the stage
        prim_type: The type of the prim to create. Defaults to "Xform".
        fabric: True for fabric stage and False for USD stage. Defaults to False.

    Returns:
        The created USD prim.

    Raises:
        ValueError: If there is already a prim at the prim_path
    """
    msg = """Function 'define_prim' is deprecated. Please use the USD APIs directly instead.

    Example:
        >>> import isaaclab.sim as sim_utils
        >>>
        >>> stage = sim_utils.get_current_stage()
        >>> stage.DefinePrim("/World/Shapes", "Xform")
        Usd.Prim(</World/Shapes>)
    """
    logger.warning(msg)
    # get current stage
    stage = get_current_stage(fabric=fabric)
    # check if prim path is valid
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at prim path: {prim_path}")
    # define prim
    return stage.DefinePrim(prim_path, prim_type)


def get_prim_type_name(prim_path: str | Usd.Prim, fabric: bool = False) -> str:
    """Get the type name of the USD Prim at the provided path.

    .. deprecated:: 2.3.0
        This function is deprecated. Please use the USD APIs directly instead.

        >>> import isaaclab.sim as sim_utils
        >>>
        >>> stage = sim_utils.get_current_stage()
        >>> prim = stage.GetPrimAtPath("/World/Cube")
        >>> prim.GetTypeName()
        "Cube"

    Args:
        prim_path: path of the prim in the stage or the prim itself
        fabric: True for fabric stage and False for USD stage. Defaults to False.

    Returns:
        The type name of the USD Prim at the provided path.

    Raises:
        ValueError: If there is not a valid prim at the provided path
    """
    msg = """Function 'get_prim_type_name' is deprecated. Please use the USD APIs directly instead.

    Example:
        >>> import isaaclab.sim as sim_utils
        >>>
        >>> stage = sim_utils.get_current_stage()
        >>> prim = stage.GetPrimAtPath("/World/Cube")
        >>> prim.GetTypeName()
        "Cube"
    """
    logger.warning(msg)
    # check if string
    if isinstance(prim_path, str):
        stage = get_current_stage(fabric=fabric)
        prim = stage.GetPrimAtPath(prim_path)
    else:
        prim = prim_path
    # check if prim is valid
    if not prim.IsValid():
        raise ValueError(f"A prim does not exist at prim path: {prim_path}")
    # return type name
    return prim.GetTypeName()


"""
Queries utilities.
"""


def get_next_free_path(path: str) -> str:
    """Gets a new prim path that doesn't exist in the stage given a base path.

    .. deprecated:: 2.3.0
        This function is deprecated. Please use the
        :func:`isaaclab.sim.utils.queries.get_next_free_prim_path` function instead.

    Args:
        path: The base prim path to check.
        stage: The stage to check. Defaults to the current stage.

    Returns:
        A new path that is guaranteed to not exist on the current stage
    """
    logger.warning("Function 'get_next_free_path' is deprecated. Please use 'get_next_free_prim_path' instead.")
    return get_next_free_prim_path(path)
