# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for operating on the USD stage."""

from __future__ import annotations

import builtins
import contextlib
import logging
import os
import threading
from collections.abc import Callable, Generator

from pxr import Sdf, Usd, UsdUtils

from isaaclab.utils.version import get_isaac_sim_version, has_kit

# import logger
logger = logging.getLogger(__name__)
_context = threading.local()  # thread-local storage to handle nested contexts and concurrent access

# Kit-dependent imports (only available when running with Kit/Isaac Sim)
if has_kit():
    import omni.kit.app


def _check_ancestral(prim: Usd.Prim) -> bool:
    """Check if a prim is brought into composition by its ancestor (an ancestral prim).

    This is a pure USD implementation of ``omni.usd.check_ancestral``.

    An ancestral prim is one that exists due to a reference, payload, or other composition arc
    on an ancestor prim. Such prims cannot be directly deleted because they are "opinions" from
    the referenced asset, not locally authored prims.

    Args:
        prim: The USD prim to check.

    Returns:
        True if the prim is an ancestral prim, False otherwise.
    """
    if not prim or not prim.IsValid():
        return False

    def _check_ancestral_node(node) -> bool:
        """Recursively check if any composition node is due to an ancestor."""
        if node.IsDueToAncestor():
            return True
        return any(_check_ancestral_node(child) for child in node.children)

    prim_index = prim.GetPrimIndex()
    if not prim_index:
        return False

    return _check_ancestral_node(prim_index.rootNode)


def resolve_paths(
    src_layer_identifier: str,
    dst_layer_identifier: str,
    store_relative_path: bool = True,
) -> None:
    """Resolve external asset paths in a destination layer relative to a source layer.

    When content is copied from one USD layer to another (e.g., via ``Sdf.CopySpec`` or
    ``layer.TransferContent``), relative asset paths that were valid from the source
    layer's location may become invalid from the destination layer's location. This
    function recalculates those paths.

    This uses USD's built-in ``UsdUtils.ModifyAssetPaths`` to update all external references
    (sublayers, references, payloads, asset paths) in the destination layer.

    Args:
        src_layer_identifier: The identifier (path) of the source layer.
        dst_layer_identifier: The identifier (path) of the destination layer.
        store_relative_path: Whether to store paths as relative. Defaults to True.

    Example:
        >>> from pxr import Sdf
        >>> import isaaclab.sim as sim_utils
        >>>
        >>> # After copying content to a new layer
        >>> source_layer = stage.GetRootLayer()
        >>> target_layer = Sdf.Layer.CreateNew("/path/to/output.usd")
        >>> target_layer.TransferContent(source_layer)
        >>> sim_utils.resolve_paths(source_layer.identifier, target_layer.identifier)
        >>> target_layer.Save()
    """
    src_layer = Sdf.Layer.FindOrOpen(src_layer_identifier)
    dst_layer = Sdf.Layer.FindOrOpen(dst_layer_identifier)

    if not src_layer:
        logger.warning(f"Source layer not found: {src_layer_identifier}")
        return
    if not dst_layer:
        logger.warning(f"Destination layer not found: {dst_layer_identifier}")
        return

    dst_dir = os.path.dirname(dst_layer.realPath or dst_layer.identifier)

    def _modify_path(asset_path: str) -> str:
        if not asset_path:
            return asset_path
        resolved = src_layer.ComputeAbsolutePath(asset_path)
        if store_relative_path and resolved and dst_dir:
            try:
                return os.path.relpath(resolved, dst_dir)
            except ValueError:
                return resolved
        return resolved or asset_path

    UsdUtils.ModifyAssetPaths(dst_layer, _modify_path)


# ##############################################################################
# Public API
# ##############################################################################


try:
    # _context is a singleton design in isaacsim and for that reason
    #  until we fully replace all modules that references the singleton(such as XformPrim, Prim ....), we have to point
    #  that singleton to this _context
    from isaacsim.core.utils import stage as sim_stage

    sim_stage._context = _context  # type: ignore
except ImportError:
    pass


def create_new_stage() -> Usd.Stage:
    """Create a new in-memory USD stage.

    Creates a new stage using pure USD (``Usd.Stage.CreateInMemory()``).

    If Kit is running and Kit extensions need to discover this stage (e.g.
    PhysX, ``isaacsim.core.prims.Articulation``), call
    :func:`attach_stage_to_usd_context` after scene setup.

    Returns:
        Usd.Stage: The created USD stage.

    Example:
        >>> import isaaclab.sim as sim_utils
        >>>
        >>> sim_utils.create_new_stage()
        Usd.Stage.Open(rootLayer=Sdf.Find('anon:0x7fba6c04f840:World7.usd'),
                       sessionLayer=Sdf.Find('anon:0x7fba6c01c5c0:World7-session.usda'),
                       pathResolverContext=<invalid repr>)
    """
    stage: Usd.Stage = Usd.Stage.CreateInMemory()
    _context.stage = stage
    UsdUtils.StageCache.Get().Insert(stage)
    return stage


def is_current_stage_in_memory() -> bool:
    """Checks if the current stage is NOT attached to the USD context.

    This function compares the current stage (from :func:`get_current_stage`) with
    the stage attached to Kit's ``omni.usd`` context. If they are different,
    the current stage is considered "in memory" - meaning it's not the stage
    that the viewport/UI displays.

    This is useful for determining if we're working with a separate in-memory
    stage created via :func:`create_new_stage_in_memory` with
    ``SimulationCfg(create_stage_in_memory=True)``.

    In kitless mode (no USD context), this always returns True.

    Returns:
        True if the current stage is different from (not attached to) the context stage.
        Also returns True if there is no context stage at all.
    """
    if not has_kit():
        return True

    import omni.usd

    context = omni.usd.get_context()
    if context is None:
        return True

    context_stage = context.get_stage()
    if context_stage is None:
        return True

    return get_current_stage() is not context_stage


def open_stage(usd_path: str) -> Usd.Stage:
    """Open the given USD file.

    Opens a USD file using pure USD (``Usd.Stage.Open()``). If Kit is available and
    context attachment is needed for viewport/UI display, use
    :func:`attach_stage_to_usd_context` after opening the stage.

    Args:
        usd_path: The path to the USD file to open.

    Returns:
        The opened USD stage.

    Raises:
        ValueError: When input path is not a supported file type by USD.
        RuntimeError: When failed to open the stage.
    """
    if not Usd.Stage.IsSupportedFile(usd_path):
        raise ValueError(f"The USD file at path '{usd_path}' is not supported.")

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage at path '{usd_path}'.")
    # Set as current stage so get_current_stage() can find it
    _context.stage = stage
    return stage


@contextlib.contextmanager
def use_stage(stage: Usd.Stage) -> Generator[None, None, None]:
    """Context manager that sets a thread-local stage, if supported.

    This function binds the stage to the thread-local context for the duration of the context manager.
    During the context manager, any call to :func:`get_current_stage` will return the stage specified
    in the context manager. After the context manager is exited, the stage is restored to the default
    stage attached to the USD context.

    .. versionadded:: 2.3.0
        This function is available in Isaac Sim 5.0 and later. For backwards
        compatibility, it falls back to a no-op context manager in Isaac Sim < 5.0.

    Args:
        stage: The stage to set in the context.

    Returns:
        A context manager that sets the stage in the context.

    Raises:
        AssertionError: If the stage is not a USD stage instance.

    Example:
        >>> from pxr import Usd
        >>> import isaaclab.sim as sim_utils
        >>>
        >>> stage_in_memory = Usd.Stage.CreateInMemory()
        >>> with sim_utils.use_stage(stage_in_memory):
        ...     # operate on the specified stage
        ...     pass
        >>> # operate on the default stage attached to the USD context
    """
    if get_isaac_sim_version().major < 5:
        logger.warning("Isaac Sim < 5.0 does not support thread-local stage contexts. Skipping use_stage().")
        yield  # no-op
    else:
        # check stage
        if not isinstance(stage, Usd.Stage):
            raise TypeError(f"Expected a USD stage instance, got: {type(stage)}")
        # store previous context value if it exists
        previous_stage = getattr(_context, "stage", None)
        # set new context value
        try:
            _context.stage = stage
            yield
        # remove context value or restore previous one if it exists
        finally:
            if previous_stage is None:
                delattr(_context, "stage")
            else:
                _context.stage = previous_stage


def update_stage() -> None:
    """Triggers a full application update cycle to process USD stage changes.

    This function calls ``omni.kit.app.get_app_interface().update()`` which triggers
    a complete application update including:

    * Physics simulation step (if ``/app/player/playSimulations`` is True)
    * Rendering (RTX path tracing, viewport updates)
    * UI updates (widgets, windows)
    * Timeline events and callbacks
    * Extension updates
    * USD/Fabric synchronization

    When to Use:
        * **After creating a new stage**: ``create_new_stage()`` → ``update_stage()``
        * **After spawning prims**: ``cfg.func("/World/Robot", cfg)`` → ``update_stage()``
        * **After USD authoring**: Creating materials, lights, meshes, etc.
        * **Before simulation starts**: During setup phase, before ``sim.reset()``
        * **In test fixtures**: To ensure consistent state before each test

    When NOT to Use:
        * **During active simulation** (after ``sim.play()``): Can interfere with
          physics stepping and cause double-stepping or timing issues.
        * **During sensor updates**: Can reset RTX renderer state mid-cycle,
          causing incorrect sensor outputs (e.g., ``inf`` depth values).
        * **Inside physics/render callbacks**: Can cause recursion or timing issues.
        * **Inside ``sim.step()`` or ``sim.render()``**: These already perform
          app updates internally with proper safeguards.

    For rendering during simulation without physics stepping, use::

        sim.set_setting("/app/player/playSimulations", False)
        omni.kit.app.get_app().update()
        sim.set_setting("/app/player/playSimulations", True)

    Example:
        >>> import isaaclab.sim as sim_utils
        >>>
        >>> # Setup phase - safe to use
        >>> sim_utils.create_new_stage()
        >>> robot_cfg.func("/World/Robot", robot_cfg)
        >>> sim_utils.update_stage()  # Commit USD changes
        >>>
        >>> # Simulation phase - DO NOT use update_stage()
        >>> sim.reset()
        >>> sim.play()
        >>> for _ in range(100):
        ...     sim.step()  # Handles updates internally
    """
    omni.kit.app.get_app_interface().update()


def save_stage(usd_path: str, save_and_reload_in_place: bool = True) -> bool:
    """Saves contents of the root layer of the current stage to the specified USD file.

    If the file already exists, it will be overwritten.

    Args:
        usd_path: The file path to save the current stage to
        save_and_reload_in_place: Whether to open the saved USD file in place. Defaults to True.

    Returns:
        True if operation is successful, otherwise False.

    Raises:
        ValueError: When input path is not a supported file type by USD.
        RuntimeError: When layer creation or save operation fails.
    """
    # check if USD file is supported
    if not Usd.Stage.IsSupportedFile(usd_path):
        raise ValueError(f"The USD file at path '{usd_path}' is not supported.")

    # create new layer
    layer = Sdf.Layer.CreateNew(usd_path)
    if layer is None:
        raise RuntimeError(f"Failed to create new USD layer at path '{usd_path}'.")

    # get root layer
    root_layer = get_current_stage().GetRootLayer()
    # transfer content from root layer to new layer
    layer.TransferContent(root_layer)

    # resolve paths so asset references remain valid from the new location
    resolve_paths(root_layer.identifier, layer.identifier)

    # save layer
    result = layer.Save()
    if not result:
        logger.error(f"Failed to save USD layer to path '{usd_path}'.")

    # if requested, open the saved USD file in place
    if save_and_reload_in_place and result:
        open_stage(usd_path)

    return result


def close_stage() -> bool:
    """Closes the current USD stage by clearing the stage cache.

    .. note::

        Once the stage is closed, it is necessary to open a new stage or create a
        new one in order to work on it.

    Returns:
        True if operation is successful.

    Example:
        >>> import isaaclab.sim as sim_utils
        >>>
        >>> sim_utils.close_stage()
        True
    """
    stage_cache = UsdUtils.StageCache.Get()
    stage_cache.Clear()
    _context.stage = None
    return True


def _is_prim_deletable(prim: Usd.Prim) -> bool:
    """Check if a prim can be safely deleted.

    This function checks various conditions to determine if a prim should be deleted:
    - Root prim ("/") cannot be deleted
    - Prims under "/Render" namespace are preserved
    - Prims with "no_delete" metadata are preserved
    - Prims hidden in stage window are preserved
    - Ancestral prims (from USD references) cannot be deleted

    Args:
        prim: The USD prim to check.

    Returns:
        True if the prim can be deleted, False otherwise.
    """
    prim_path = prim.GetPath().pathString
    if prim_path == "/":
        return False
    if prim_path.startswith("/Render"):
        return False
    if prim.GetMetadata("no_delete"):
        return False
    if prim.GetMetadata("hide_in_stage_window"):
        return False
    # Check ancestral prims (from USD references) using pure USD helper
    if _check_ancestral(prim):
        return False
    return True


def clear_stage(predicate: Callable[[Usd.Prim], bool] | None = None) -> None:
    """Deletes all prims in the stage without populating the undo command buffer.

    The function will delete all prims in the stage that satisfy the predicate. If the predicate
    is None, a default predicate will be used that deletes all prims. The default predicate deletes
    all prims that are not the root prim, are not under the /Render namespace, have the ``no_delete``
    metadata, are not ancestral to any other prim, and are not hidden in the stage window.

    Args:
        predicate: A user defined function that takes the USD prim as an argument and
            returns a boolean indicating if the prim should be deleted. If the predicate is None,
            a default predicate will be used that deletes all prims.

    Example:
        >>> import isaaclab.sim as sim_utils
        >>>
        >>> # clear the whole stage
        >>> sim_utils.clear_stage()
        >>>
        >>> # given the stage: /World/Cube, /World/Cube_01, /World/Cube_02.
        >>> # Delete only the prims of type Cube
        >>> predicate = lambda _prim: _prim.GetTypeName() == "Cube"
        >>> sim_utils.clear_stage(predicate)  # after the execution the stage will be /World
    """
    # Note: Need to import this here to prevent circular dependencies.
    from .prims import delete_prim
    from .queries import get_all_matching_child_prims

    def _predicate_from_path(prim: Usd.Prim) -> bool:
        if predicate is None:
            return _is_prim_deletable(prim)
        # Custom predicate must also pass the deletable check
        return predicate(prim) and _is_prim_deletable(prim)

    # get all prims to delete
    prims = get_all_matching_child_prims("/", _predicate_from_path)
    # convert prims to prim paths
    prim_paths_to_delete = [prim.GetPath().pathString for prim in prims]
    # delete prims
    delete_prim(prim_paths_to_delete)

    if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:  # type: ignore
        omni.kit.app.get_app_interface().update()


def get_current_stage(fabric: bool = False) -> Usd.Stage:
    """Get the current open USD or Fabric stage

    Args:
        fabric: True to get the fabric stage. False to get the USD stage. Defaults to False.

    Returns:
        The USD or Fabric stage as specified by the input arg fabric.

    Example:
        >>> import isaaclab.sim as sim_utils
        >>>
        >>> sim_utils.get_current_stage()
        Usd.Stage.Open(rootLayer=Sdf.Find('anon:0x7fba6c04f840:World7.usd'),
                       sessionLayer=Sdf.Find('anon:0x7fba6c01c5c0:World7-session.usda'),
                       pathResolverContext=<invalid repr>)
    """
    # First check thread-local context for an in-memory stage
    stage = getattr(_context, "stage", None)
    if stage is not None:
        if fabric:
            import usdrt

            # Get stage ID and attach to Fabric stage
            stage_id = get_current_stage_id()
            return usdrt.Usd.Stage.Attach(stage_id)
        return stage

    return stage


def get_current_stage_id() -> int:
    """Get the current open stage ID.

    Returns:
        The current open stage id.

    Example:
        >>> import isaaclab.sim as sim_utils
        >>>
        >>> sim_utils.get_current_stage_id()
        1234567890
    """
    # get current stage
    stage = get_current_stage()
    if stage is None:
        raise RuntimeError("No current stage available. Did you create a stage?")

    # retrieve stage ID from stage cache
    stage_cache = UsdUtils.StageCache.Get()
    stage_id = stage_cache.GetId(stage).ToLongInt()
    # if stage ID is not found, insert it into the stage cache
    if stage_id < 0:
        # Ensure stage has a valid root layer before inserting
        if not stage.GetRootLayer():
            raise RuntimeError("Stage has no root layer - cannot cache an incomplete stage.")
        stage_id = stage_cache.Insert(stage).ToLongInt()
    # return stage ID
    return stage_id
