# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for operating on the USD stage."""

from __future__ import annotations

import builtins
import contextlib
import logging
import threading
from collections.abc import Callable, Generator

import omni.kit.app
import omni.usd
from isaacsim.core.utils import stage as sim_stage
from pxr import Sdf, Usd, UsdUtils

from isaaclab.utils.version import get_isaac_sim_version

# import logger
logger = logging.getLogger(__name__)
_context = threading.local()  # thread-local storage to handle nested contexts and concurrent access

# _context is a singleton design in isaacsim and for that reason
#  until we fully replace all modules that references the singleton(such as XformPrim, Prim ....), we have to point
#  that singleton to this _context
sim_stage._context = _context  # type: ignore


def create_new_stage() -> Usd.Stage:
    """Create a new stage attached to the USD context.

    Returns:
        Usd.Stage: The created USD stage.

    Raises:
        RuntimeError: When failed to create a new stage.

    Example:
        >>> import isaaclab.sim as sim_utils
        >>>
        >>> sim_utils.create_new_stage()
        Usd.Stage.Open(rootLayer=Sdf.Find('anon:0x7fba6c04f840:World7.usd'),
                       sessionLayer=Sdf.Find('anon:0x7fba6c01c5c0:World7-session.usda'),
                       pathResolverContext=<invalid repr>)
    """
    result = omni.usd.get_context().new_stage()
    if result:
        return omni.usd.get_context().get_stage()
    else:
        raise RuntimeError("Failed to create a new stage. Please check if the USD context is valid.")


def create_new_stage_in_memory() -> Usd.Stage:
    """Creates a new stage in memory, if supported.

    .. versionadded:: 2.3.0
        This function is available in Isaac Sim 5.0 and later. For backwards
        compatibility, it falls back to creating a new stage attached to the USD context.

    Returns:
        The new stage in memory.

    Example:
        >>> import isaaclab.sim as sim_utils
        >>>
        >>> sim_utils.create_new_stage_in_memory()
        Usd.Stage.Open(rootLayer=Sdf.Find('anon:0xf7b00e0:tmp.usda'),
                       sessionLayer=Sdf.Find('anon:0xf7cd2e0:tmp-session.usda'),
                       pathResolverContext=<invalid repr>)
    """
    if get_isaac_sim_version().major < 5:
        logger.warning(
            "Isaac Sim < 5.0 does not support creating a new stage in memory. Falling back to creating a new"
            " stage attached to USD context."
        )
        return create_new_stage()
    else:
        return Usd.Stage.CreateInMemory()


def get_context_stage() -> Usd.Stage | None:
    """Get the stage attached to the USD context, if any.

    The "context stage" is the USD stage attached to the Omniverse application's
    UsdContext. This is the stage that:

    * The viewport renders
    * The Stage panel in the UI displays
    * Most Isaac Sim/Omniverse systems operate on by default

    This is different from an "in-memory stage" created via
    :func:`create_new_stage_in_memory`, which exists only in RAM. Note that when
    using ``SimulationCfg(create_stage_in_memory=True)``, the in-memory stage is
    automatically attached to the USD context at ``SimulationContext`` creation.

    Returns:
        The stage attached to the USD context, or None if no stage is attached.

    Example:
        >>> import isaaclab.sim as sim_utils
        >>>
        >>> stage = sim_utils.get_context_stage()
        >>> if stage is not None:
        ...     print("Context has a stage attached")
    """
    context = omni.usd.get_context()
    if context is None:
        return None
    return context.get_stage()


def is_current_stage_in_memory() -> bool:
    """Checks if the current stage is NOT attached to the USD context.

    This function compares the current stage (from :func:`get_current_stage`) with
    the context stage (from :func:`get_context_stage`). If they are different,
    the current stage is considered "in memory" - meaning it's not the stage
    that the viewport/UI displays.

    This is useful for determining if we're working with a separate in-memory
    stage created via :func:`create_new_stage_in_memory` with
    ``SimulationCfg(create_stage_in_memory=True)``.

    Returns:
        True if the current stage is different from (not attached to) the context stage.
        Also returns True if there is no context stage at all.
    """
    # Get current stage
    current_stage = get_current_stage()
    context_stage = get_context_stage()

    # If no context stage exists, current stage is definitely not attached to it
    if context_stage is None:
        return True

    # Compare by identity - are they the same stage object?
    # Note: We can't just compare IDs because different stage objects could
    # theoretically have the same ID if one was closed and another opened.
    return current_stage is not context_stage


def open_stage(usd_path: str) -> bool:
    """Open the given usd file and replace currently opened stage.

    Args:
        usd_path: The path to the USD file to open.

    Returns:
        True if operation is successful, otherwise False.

    Raises:
        ValueError: When input path is not a supported file type by USD.
    """
    # check if USD file is supported
    if not Usd.Stage.IsSupportedFile(usd_path):
        raise ValueError(f"The USD file at path '{usd_path}' is not supported.")

    # get USD context
    usd_context = omni.usd.get_context()
    # disable save to recent files
    usd_context.disable_save_to_recent_files()
    # open stage
    result = usd_context.open_stage(usd_path)
    # enable save to recent files
    usd_context.enable_save_to_recent_files()
    # return result
    return result


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
    # resolve paths
    omni.usd.resolve_paths(root_layer.identifier, layer.identifier)
    # save layer
    result = layer.Save()
    if not result:
        logger.error(f"Failed to save USD layer to path '{usd_path}'.")

    # if requested, open the saved USD file in place
    if save_and_reload_in_place and result:
        open_stage(usd_path)

    return result


def close_stage(callback_fn: Callable[[bool, str], None] | None = None) -> bool:
    """Closes the current USD stage.

    .. note::

        Once the stage is closed, it is necessary to open a new stage or create a
        new one in order to work on it.

    Args:
        callback_fn: A callback function to call while closing the stage.
            The function should take two arguments: a boolean indicating whether the stage is closing
            and a string indicating the error message if the stage closing fails. Defaults to None,
            in which case the stage will be closed without a callback.

    Returns:
        True if operation is successful, otherwise False.

    Example:
        >>> import isaaclab.sim as sim_utils
        >>>
        >>> sim_utils.close_stage()
        True
        >>>

    Example with callback function:
        >>> import isaaclab.sim as sim_utils
        >>>
        >>> def callback(*args, **kwargs):
        ...     print("callback:", args, kwargs)
        >>> sim_utils.close_stage(callback)
        True
        >>> sim_utils.close_stage(callback)
        callback: (False, 'Stage opening or closing already in progress!!') {}
        False
    """
    if callback_fn is None:
        result = omni.usd.get_context().close_stage()
    else:
        result = omni.usd.get_context().close_stage_with_callback(callback_fn)
    return result


def is_prim_deletable(prim: Usd.Prim) -> bool:
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
    if omni.usd.check_ancestral(prim):
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
            return is_prim_deletable(prim)
        # Custom predicate must also pass the deletable check
        return predicate(prim) and is_prim_deletable(prim)

    # get all prims to delete
    prims = get_all_matching_child_prims("/", _predicate_from_path)
    # convert prims to prim paths
    prim_paths_to_delete = [prim.GetPath().pathString for prim in prims]
    # delete prims
    delete_prim(prim_paths_to_delete)

    if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:  # type: ignore
        omni.kit.app.get_app_interface().update()


def is_stage_loading() -> bool:
    """Convenience function to see if any files are being loaded.

    Returns:
        True if loading, False otherwise

    Example:
        >>> import isaaclab.sim as sim_utils
        >>>
        >>> sim_utils.is_stage_loading()
        False
    """
    context = omni.usd.get_context()
    if context is None:
        return False
    else:
        _, _, loading = context.get_stage_loading_status()
        return loading > 0


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
    stage = getattr(_context, "stage", omni.usd.get_context().get_stage())

    if fabric:
        import usdrt

        # Get stage ID and attach to Fabric stage
        stage_id = get_current_stage_id()
        return usdrt.Usd.Stage.Attach(stage_id)

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
