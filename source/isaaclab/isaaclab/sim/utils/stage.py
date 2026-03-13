# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for operating on the USD stage."""

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


def is_current_stage_in_memory() -> bool:
    """Checks if the current stage is in memory.

    This function compares the stage id of the current USD stage with the stage id of the USD context stage.

    Returns:
        Whether the current stage is in memory.
    """
    # grab current stage id
    stage_id = get_current_stage_id()

    # grab context stage id
    context_stage = omni.usd.get_context().get_stage()
    with use_stage(context_stage):
        context_stage_id = get_current_stage_id()

    # check if stage ids are the same
    return stage_id != context_stage_id


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
    """Updates the current stage by triggering an application update cycle.

    This function triggers a single update cycle of the application interface, which
    in turn updates the stage and all associated systems (rendering, physics, etc.).
    This is necessary to ensure that changes made to the stage are properly processed
    and reflected in the simulation.

    Note:
        This function calls the application update interface rather than directly
        updating the stage because the stage update is part of the broader
        application update cycle that includes rendering, physics, and other systems.

    Example:
        >>> import isaaclab.sim as sim_utils
        >>>
        >>> sim_utils.update_stage()
    """
    # TODO: Why is this updating the simulation and not the stage?
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

    def _default_predicate(prim: Usd.Prim) -> bool:
        """Check if the prim should be deleted."""
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

    def _predicate_from_path(prim: Usd.Prim) -> bool:
        if predicate is None:
            return _default_predicate(prim)
        return predicate(prim)

    # get all prims to delete
    if predicate is None:
        prims = get_all_matching_child_prims("/", _default_predicate)
    else:
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
    # retrieve stage ID from stage cache
    stage_cache = UsdUtils.StageCache.Get()
    stage_id = stage_cache.GetId(stage).ToLongInt()
    # if stage ID is not found, insert it into the stage cache
    if stage_id < 0:
        stage_id = stage_cache.Insert(stage).ToLongInt()
    # return stage ID
    return stage_id


def attach_stage_to_usd_context(attaching_early: bool = False):
    """Attaches the current USD stage in memory to the USD context.

    This function should be called during or after scene is created and before stage is simulated or rendered.
    If the stage is not in memory or rendering is not enabled, this function will return without attaching.

    .. versionadded:: 2.3.0
        This function is available in Isaac Sim 5.0 and later. For backwards
        compatibility, it returns without attaching to the USD context.

    Args:
        attaching_early: Whether to attach the stage to the usd context before stage is created. Defaults to False.
    """

    import carb
    import omni.physx
    import omni.usd
    from isaacsim.core.simulation_manager import SimulationManager

    from isaaclab.sim.simulation_context import SimulationContext

    # if Isaac Sim version is less than 5.0, stage in memory is not supported
    if get_isaac_sim_version().major < 5:
        return

    # if stage is not in memory, we can return early
    if not is_current_stage_in_memory():
        return

    # attach stage to physx
    stage_id = get_current_stage_id()
    physx_sim_interface = omni.physx.get_physx_simulation_interface()
    physx_sim_interface.attach_stage(stage_id)

    # this carb flag is equivalent to if rendering is enabled
    carb_setting = carb.settings.get_settings()  # type: ignore
    is_rendering_enabled = carb_setting.get("/physics/fabricUpdateTransformations")

    # if rendering is not enabled, we don't need to attach it
    if not is_rendering_enabled:
        return

    # early attach warning msg
    if attaching_early:
        logger.warning(
            "Attaching stage in memory to USD context early to support an operation which"
            " does not support stage in memory."
        )

    # skip this callback to avoid wiping the stage after attachment
    SimulationContext.instance().skip_next_stage_open_callback()

    # disable stage open callback to avoid clearing callbacks
    SimulationManager.enable_stage_open_callback(False)

    # enable physics fabric
    SimulationContext.instance()._physics_context.enable_fabric(True)  # type: ignore

    # attach stage to usd context
    omni.usd.get_context().attach_stage_with_callback(stage_id)

    # attach stage to physx
    physx_sim_interface = omni.physx.get_physx_simulation_interface()
    physx_sim_interface.attach_stage(stage_id)

    # re-enable stage open callback
    SimulationManager.enable_stage_open_callback(True)
