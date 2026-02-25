# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import logging
import threading
import typing
from collections.abc import Generator

from pxr import Sdf, Usd, UsdGeom, UsdUtils

logger = logging.getLogger(__name__)
_context = threading.local()  # thread-local storage to handle nested contexts and concurrent access

# Import for remote file handling
from isaaclab.utils.assets import check_file_path, retrieve_file_path

AXES_TOKEN = {
    "X": UsdGeom.Tokens.x,
    "x": UsdGeom.Tokens.x,
    "Y": UsdGeom.Tokens.y,
    "y": UsdGeom.Tokens.y,
    "Z": UsdGeom.Tokens.z,
    "z": UsdGeom.Tokens.z,
}
"""Mapping from axis name to axis USD token

    >>> import isaacsim.core.utils.constants as constants_utils
    >>>
    >>> # get the x-axis USD token
    >>> constants_utils.AXES_TOKEN['x']
    X
    >>> constants_utils.AXES_TOKEN['X']
    X
"""


def attach_stage_to_usd_context(attaching_early: bool = False):
    """Attaches the current USD stage in memory to the USD context.

    This function should be called during or after scene is created and before stage is simulated or rendered.

    Note:
        If the stage is not in memory or rendering is not enabled, this function will return without attaching.

    Args:
        attaching_early: Whether to attach the stage to the usd context before stage is created. Defaults to False.
    """

    from isaaclab.sim.simulation_context import SimulationContext

    # if stage is not in memory, we can return early
    if not is_current_stage_in_memory():
        return

    import carb
    import omni.physx

    # attach stage to physx
    stage_id = get_current_stage_id()
    physx_sim_interface = omni.physx.get_physx_simulation_interface()
    physx_sim_interface.attach_stage(stage_id)

    # this carb flag is equivalent to if rendering is enabled
    carb_setting = carb.settings.get_settings()
    is_rendering_enabled = carb_setting.get("/physics/fabricUpdateTransformations")

    # if rendering is not enabled, we don't need to attach it
    if not is_rendering_enabled:
        return

    # early attach warning msg
    if attaching_early:
        logger.warning(
            "Attaching stage in memory to USD context early to support an operation which doesn't support stage in"
            " memory."
        )

    # skip this callback to avoid wiping the stage after attachment
    SimulationContext.instance().skip_next_stage_open_callback()

    # enable physics fabric
    SimulationContext.instance()._physics_context.enable_fabric(True)

    import omni.usd

    # attach stage to usd context
    omni.usd.get_context().attach_stage_with_callback(stage_id)

    # attach stage to physx
    physx_sim_interface = omni.physx.get_physx_simulation_interface()
    physx_sim_interface.attach_stage(stage_id)


def is_current_stage_in_memory() -> bool:
    """Checks if the current stage is in memory.

    This function compares the stage id of the current USD stage with the stage id of the USD context stage.

    Returns:
        Whether the current stage is in memory.
    """

    # grab current stage id
    stage_id = get_current_stage_id()

    import omni.usd

    # grab context stage id
    context_stage = omni.usd.get_context().get_stage()
    with use_stage(context_stage):
        context_stage_id = get_current_stage_id()

    # check if stage ids are the same
    return stage_id != context_stage_id


@contextlib.contextmanager
def use_stage(stage: Usd.Stage) -> Generator[None, None, None]:
    """Context manager that sets a thread-local stage.

    This allows different threads or nested contexts to use different USD stages
    without interfering with each other.

    Args:
        stage: The stage to set temporarily.

    Raises:
        AssertionError: If the stage is not a USD stage instance.

    Example:

    .. code-block:: python

        >>> from pxr import Usd
        >>> import isaaclab.sim.utils.stage as stage_utils
        >>>
        >>> stage_in_memory = Usd.Stage.CreateInMemory()
        >>> with stage_utils.use_stage(stage_in_memory):
        ...    # operate on the specified stage
        ...    pass
        >>> # operate on the default stage attached to the USD context
    """
    # check stage
    assert isinstance(stage, Usd.Stage), f"Expected a USD stage instance, got: {type(stage)}"
    # store previous context value if it exists
    previous_stage = getattr(_context, "stage", None)
    # set new context value
    try:
        _context.stage = stage
        yield
    # remove context value or restore previous one if it exists
    finally:
        if previous_stage is None:
            if hasattr(_context, "stage"):
                delattr(_context, "stage")
        else:
            _context.stage = previous_stage


def get_current_stage(fabric: bool = False) -> Usd.Stage:
    """Get the current open USD stage.

    This function retrieves the current USD stage using the following priority:
    1. Thread-local stage context (set via use_stage context manager)
    2. SimulationContext singleton's stage (if initialized)
    3. USD StageCache (standard USD way to track open stages)

    Args:
        fabric: Deprecated parameter, kept for backward compatibility. Defaults to False.

    Raises:
        RuntimeError: If no USD stage is currently open.

    Returns:
        The USD stage.

    Example:

    .. code-block:: python

        >>> import isaaclab.sim.utils.stage as stage_utils
        >>>
        >>> stage_utils.get_current_stage()
        Usd.Stage.Open(rootLayer=Sdf.Find('anon:0x7fba6c04f840:World7.usd'),
                        sessionLayer=Sdf.Find('anon:0x7fba6c01c5c0:World7-session.usda'),
                        pathResolverContext=<invalid repr>)
    """

    # Try to get stage from thread-local context first
    # stage = getattr(_context, "stage", None)
    from isaaclab.sim.simulation_context import SimulationContext

    sim_context = SimulationContext.instance()
    stage = sim_context.stage if sim_context is not None else None

    if fabric:
        import usdrt

        stage_cache = UsdUtils.StageCache.Get()
        stage_id = stage_cache.GetId(stage).ToLongInt()
        if stage_id < 0:
            # Try to get from omni.usd if it's the context stage
            try:
                import omni.usd

                context_stage = omni.usd.get_context().get_stage()
                if (
                    context_stage is not None
                    and stage.GetRootLayer().identifier == context_stage.GetRootLayer().identifier
                ):
                    stage_id = omni.usd.get_context().get_stage_id()
                else:
                    stage_id = stage_cache.Insert(stage).ToLongInt()
            except (ImportError, AttributeError):
                stage_id = stage_cache.Insert(stage).ToLongInt()
        return usdrt.Usd.Stage.Attach(stage_id)

    if stage is not None:
        return stage

    # Fall back to omni.usd when SimulationContext is not available
    try:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is not None:
            return stage
    except (ImportError, AttributeError):
        pass

    raise RuntimeError("No USD stage is currently open. Please create a stage first.")


def get_current_stage_id() -> int:
    """Get the current open stage id

    Returns:
        The current open stage id.

    Example:

    .. code-block:: python

        >>> import isaaclab.sim.utils.stage as stage_utils
        >>>
        >>> stage_utils.get_current_stage_id()
        1234567890
    """
    stage = get_current_stage()
    stage_cache = UsdUtils.StageCache.Get()
    stage_id = stage_cache.GetId(stage).ToLongInt()

    # If stage is already in the cache, return its ID
    if stage_id >= 0:
        return stage_id

    # Stage not in cache - try to get from omni.usd if it's the context stage
    # This handles stages managed by omni.usd that may have a different Python wrapper type
    try:
        import omni.usd

        context_stage = omni.usd.get_context().get_stage()
        # Compare by root layer identifier to verify it's the same stage
        if context_stage is not None and stage.GetRootLayer().identifier == context_stage.GetRootLayer().identifier:
            return omni.usd.get_context().get_stage_id()
    except (ImportError, AttributeError):
        pass

    # Fall back to inserting into StageCache (works for in-memory stages we created)
    stage_id = stage_cache.Insert(stage).ToLongInt()
    return stage_id


def update_stage() -> None:
    """Update the current USD stage.

    Example:

    .. code-block:: python

        >>> import isaaclab.sim.utils.stage as stage_utils
        >>>
        >>> stage_utils.update_stage()
    """
    import omni.kit.app

    omni.kit.app.get_app_interface().update()


# TODO: make a generic util for setting all layer properties
def set_stage_up_axis(axis: str = "z") -> None:
    """Change the up axis of the current stage

    Args:
        axis (UsdGeom.Tokens, optional): valid values are ``"x"``, ``"y"`` and ``"z"``

    Example:

    .. code-block:: python

        >>> import isaaclab.sim.utils.stage as stage_utils
        >>>
        >>> # set stage up axis to Y-up
        >>> stage_utils.set_stage_up_axis("y")
    """
    stage = get_current_stage()
    if stage is None:
        raise Exception("There is no stage currently opened")
    rootLayer = stage.GetRootLayer()
    rootLayer.SetPermissionToEdit(True)
    with Usd.EditContext(stage, rootLayer):
        UsdGeom.SetStageUpAxis(stage, AXES_TOKEN[axis])


def get_stage_up_axis() -> str:
    """Get the current up-axis of USD stage.

    Returns:
        str: The up-axis of the stage.

    Example:

    .. code-block:: python

        >>> import isaaclab.sim.utils.stage as stage_utils
        >>>
        >>> stage_utils.get_stage_up_axis()
        Z
    """
    stage = get_current_stage()
    return UsdGeom.GetStageUpAxis(stage)


def clear_stage(predicate: typing.Callable[[str], bool] | None = None) -> None:
    """Deletes all prims in the stage without populating the undo command buffer

    Args:
        predicate: user defined function that takes a prim_path (str) as input and returns True/False if the prim
            should/shouldn't be deleted. If predicate is None, a default is used that deletes all prims

    Example:

    .. code-block:: python

        >>> import isaaclab.sim.utils.stage as stage_utils
        >>>
        >>> # clear the whole stage
        >>> stage_utils.clear_stage()
        >>>
        >>> # given the stage: /World/Cube, /World/Cube_01, /World/Cube_02.
        >>> # Delete only the prims of type Cube
        >>> predicate = lambda path: prims_utils.from_prim_path_get_type_name(path) == "Cube"
        >>> stage_utils.clear_stage(predicate)  # after the execution the stage will be /World
    """
    # Note: Need to import this here to prevent circular dependencies.
    from .prims import get_all_matching_child_prims

    def default_predicate(prim: Usd.Prim) -> bool:
        prim_path = prim.GetPath().pathString
        if prim_path == "/":
            return False
        if prim_path.startswith("/Render"):
            return False
        if prim.GetMetadata("no_delete"):
            return False
        if prim.GetMetadata("hide_in_stage_window"):
            return False
        # Check if any ancestor has references (ancestral check)
        current = prim
        while current and current.GetPath() != Sdf.Path("/"):
            if current.HasAuthoredReferences():
                return False
            current = current.GetParent()
        return True

    def predicate_from_path(prim: Usd.Prim) -> bool:
        if predicate is None:
            return default_predicate(prim)
        return predicate(prim.GetPath().pathString)

    if predicate is None:
        prims = get_all_matching_child_prims("/", default_predicate)
    else:
        prims = get_all_matching_child_prims("/", predicate_from_path)
    prim_paths_to_delete = [prim.GetPath().pathString for prim in prims]

    # Delete prims using USD API directly
    stage = get_current_stage()
    for prim_path in prim_paths_to_delete:
        stage.RemovePrim(prim_path)


def print_stage_prim_paths(fabric: bool = False) -> None:
    """Traverses the stage and prints all prim (hidden or not) paths.

    Args:
        fabric: Deprecated parameter, kept for backward compatibility. Defaults to False.

    Example:

    .. code-block:: python

        >>> import isaaclab.sim.utils.stage as stage_utils
        >>>
        >>> # given the stage: /World/Cube, /World/Cube_01, /World/Cube_02.
        >>> stage_utils.print_stage_prim_paths()
        /Render
        /World
        /World/Cube
        /World/Cube_01
        /World/Cube_02
        /OmniverseKit_Persp
        /OmniverseKit_Front
        /OmniverseKit_Top
        /OmniverseKit_Right
    """
    # Note: Need to import this here to prevent circular dependencies.
    from .prims import get_prim_path

    for prim in traverse_stage(fabric=fabric):
        prim_path = get_prim_path(prim)
        print(prim_path)


def add_reference_to_stage(usd_path: str, prim_path: str, prim_type: str = "Xform") -> Usd.Prim:
    """Add USD reference to the opened stage at specified prim path.

    Adds a reference to an external USD file at the specified prim path on the current stage.
    If the prim does not exist, it will be created with the specified type.

    This function supports both local and remote USD files (HTTP, HTTPS, S3). Remote files
    are automatically downloaded to a local cache before being referenced.

    Note:
        This is a USD-core only implementation that does not use omni APIs for metrics checking.

    Args:
        usd_path: The path to USD file to reference. Can be a local path or remote URL (HTTP, HTTPS, S3).
        prim_path: The prim path where the reference will be attached.
        prim_type: The type of prim to create if it doesn't exist. Defaults to "Xform".

    Returns:
        The USD prim at the specified prim path.

    Raises:
        FileNotFoundError: When the input USD file is not found at the specified path.

    Example:

    .. code-block:: python

        >>> import isaaclab.sim.utils.stage as stage_utils
        >>>
        >>> # load a local USD file
        >>> prim = stage_utils.add_reference_to_stage(
        ...     usd_path="/home/<user>/Documents/Assets/Robots/FrankaRobotics/FrankaPanda/franka.usd",
        ...     prim_path="/World/panda"
        ... )
        >>> prim
        Usd.Prim(</World/panda>)
        >>>
        >>> # load a remote USD file from S3
        >>> prim = stage_utils.add_reference_to_stage(
        ...     usd_path="https://s3.amazonaws.com/bucket/robot.usd",
        ...     prim_path="/World/robot"
        ... )
    """
    # Store original path for error messages
    original_usd_path = usd_path

    # Check if file is remote and download if needed
    # check_file_path returns: 0 (not found), 1 (local), 2 (remote)
    file_status = check_file_path(usd_path)

    if file_status == 0:
        raise FileNotFoundError(f"USD file not found at path: {usd_path}")

    # Download remote files to local cache
    if file_status == 2:
        logger.info(f"Downloading remote USD file: {original_usd_path}")
        try:
            usd_path = retrieve_file_path(usd_path, force_download=False)
            logger.info(f"  Downloaded to: {usd_path}")
        except Exception as e:
            raise FileNotFoundError(f"Failed to download USD file from {original_usd_path}: {e}")

    # Verify the local file exists and can be opened
    import os

    if not os.path.exists(usd_path):
        raise FileNotFoundError(f"USD file does not exist at local path: {usd_path} (original: {original_usd_path})")

    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        prim = stage.DefinePrim(prim_path, prim_type)

    # Try to open the USD layer with the local path
    sdf_layer = Sdf.Layer.FindOrOpen(usd_path)
    if not sdf_layer:
        raise FileNotFoundError(f"Could not open USD file at {usd_path} (original: {original_usd_path})")

    # Attempt to use omni metrics assembler for unit checking (if available)
    try:
        from omni.metrics.assembler.core import get_metrics_assembler_interface

        stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        ret_val = get_metrics_assembler_interface().check_layers(
            stage.GetRootLayer().identifier, sdf_layer.identifier, stage_id
        )
        if ret_val["ret_val"]:
            # Metrics check passed, add reference using pure USD API
            success_bool = prim.GetReferences().AddReference(usd_path)
            if not success_bool:
                raise FileNotFoundError(f"Failed to add reference to {usd_path}")
        else:
            # Metrics check didn't pass, use pure USD
            success_bool = prim.GetReferences().AddReference(usd_path)
            if not success_bool:
                raise FileNotFoundError(f"Failed to add reference to {usd_path}")
    except (ImportError, Exception):
        # Omni APIs not available or failed, fall back to pure USD implementation
        success_bool = prim.GetReferences().AddReference(usd_path)
        if not success_bool:
            raise FileNotFoundError(f"Failed to add reference to {usd_path}")

    return prim


def create_new_stage() -> Usd.Stage:
    """Create a new stage attached to the USD context.

    If omni.usd is not available (e.g., running without Isaac Sim), this function
    falls back to creating a stage in memory using USD core APIs.

    Note:
        When using omni.usd, this function calls app.update() to ensure the async
        stage creation completes before returning.

    Returns:
        Usd.Stage: The created USD stage.

    Example:

    .. code-block:: python

        >>> import isaaclab.sim.utils.stage as stage_utils
        >>>
        >>> stage_utils.create_new_stage()
        Usd.Stage.Open(rootLayer=Sdf.Find('anon:0x7fba6c04f840:World7.usd'),
                        sessionLayer=Sdf.Find('anon:0x7fba6c01c5c0:World7-session.usda'),
                        pathResolverContext=<invalid repr>)
    """
    try:
        import omni.kit.app
        import omni.usd

        result = omni.usd.get_context().new_stage()
        # new_stage() is an async operation - need to update app to complete it
        omni.kit.app.get_app().update()
        return result
    except (ImportError, AttributeError):
        # Fall back to in-memory stage when omni.usd is not available
        return create_new_stage_in_memory()


def create_new_stage_in_memory() -> Usd.Stage:
    """Creates a new stage in memory using USD core APIs.

    This function creates a stage in memory and adds it to the USD StageCache
    so it can be properly tracked and retrieved by get_current_stage().

    The stage is configured with Z-up axis and meters as the unit, matching
    Isaac Sim's default configuration.

    Returns:
        The new stage in memory.

    Example:

    .. code-block:: python

        >>> import isaaclab.sim.utils.stage as stage_utils
        >>>
        >>> stage_utils.create_new_stage_in_memory()
        Usd.Stage.Open(rootLayer=Sdf.Find('anon:0xf7b00e0:tmp.usda'),
                        sessionLayer=Sdf.Find('anon:0xf7cd2e0:tmp-session.usda'),
                        pathResolverContext=<invalid repr>)
    """
    # Create a new stage in memory using USD core API
    stage = Usd.Stage.CreateInMemory()

    # Configure stage to match Isaac Sim defaults:
    # - Z-up axis (USD defaults to Y-up)
    # - Meters as the unit (USD defaults to centimeters)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Add to StageCache so it can be found by get_current_stage()
    # This ensures the stage is properly tracked and can be retrieved later
    stage_cache = UsdUtils.StageCache.Get()
    stage_cache.Insert(stage)

    return stage


def open_stage(usd_path: str) -> bool:
    """Open the given usd file and replace currently opened stage.

    Args:
        usd_path (str): Path to the USD file to open.

    Raises:
        ValueError: When input path is not a supported file type by USD.

    Returns:
        bool: True if operation is successful, otherwise false.

    Example:

    .. code-block:: python

        >>> import isaaclab.sim.utils.stage as stage_utils
        >>>
        >>> stage_utils.open_stage("/home/<user>/Documents/Assets/Robots/FrankaRobotics/FrankaPanda/franka.usd")
        True
    """
    import omni.usd

    if not Usd.Stage.IsSupportedFile(usd_path):
        raise ValueError("Only USD files can be loaded with this method")
    usd_context = omni.usd.get_context()
    usd_context.disable_save_to_recent_files()
    result = omni.usd.get_context().open_stage(usd_path)
    usd_context.enable_save_to_recent_files()
    return result


def save_stage(usd_path: str, save_and_reload_in_place=True) -> bool:
    """Save usd file to path, it will be overwritten with the current stage

    Args:
        usd_path (str): File path to save the current stage to
        save_and_reload_in_place (bool, optional): use ``save_as_stage`` to save and reload the root layer in place. Defaults to True.

    Raises:
        ValueError: When input path is not a supported file type by USD.

    Returns:
        bool: True if operation is successful, otherwise false.

    Example:

    .. code-block:: python

        >>> import isaaclab.sim.utils.stage as stage_utils
        >>>
        >>> stage_utils.save_stage("/home/<user>/Documents/Save/stage.usd")
        True
    """
    if not Usd.Stage.IsSupportedFile(usd_path):
        raise ValueError("Only USD files can be saved with this method")

    import omni.usd

    layer = Sdf.Layer.CreateNew(usd_path)
    root_layer = get_current_stage().GetRootLayer()
    layer.TransferContent(root_layer)
    omni.usd.resolve_paths(root_layer.identifier, layer.identifier)
    result = layer.Save()
    if save_and_reload_in_place:
        open_stage(usd_path)

    return result


def close_stage(callback_fn: typing.Callable | None = None) -> bool:
    """Closes the current opened USD stage.

    .. note::

        Once the stage is closed, it is necessary to open a new stage or create a new one in order to work on it.

    Args:
        callback_fn: Callback function to call while closing. Defaults to None.

    Returns:
        bool: True if operation is successful, otherwise false.

    Example:

    .. code-block:: python

        >>> import isaaclab.sim.utils.stage as stage_utils
        >>>
        >>> stage_utils.close_stage()
        True

    .. code-block:: python

        >>> import isaaclab.sim.utils.stage as stage_utils
        >>>
        >>> def callback(*args, **kwargs):
        ...     print("callback:", args, kwargs)
        ...
        >>> stage_utils.close_stage(callback)
        True
        >>> stage_utils.close_stage(callback)
        callback: (False, 'Stage opening or closing already in progress!!') {}
        False
    """
    import omni.usd

    if callback_fn is None:
        result = omni.usd.get_context().close_stage()
    else:
        result = omni.usd.get_context().close_stage_with_callback(callback_fn)
    return result


def traverse_stage(fabric=False) -> typing.Iterable:
    """Traverse through prims (hidden or not) in the opened Usd stage.

    Args:
        fabric: Deprecated parameter, kept for backward compatibility. Defaults to False.

    Returns:
        Generator which yields prims from the stage in depth-first-traversal order.

    Example:

    .. code-block:: python

        >>> import isaaclab.sim.utils.stage as stage_utils
        >>>
        >>> # given the stage: /World/Cube, /World/Cube_01, /World/Cube_02.
        >>> # Traverse through prims in the stage
        >>> for prim in stage_utils.traverse_stage():
        >>>     print(prim)
        Usd.Prim(</World>)
        Usd.Prim(</World/Cube>)
        Usd.Prim(</World/Cube_01>)
        Usd.Prim(</World/Cube_02>)
        Usd.Prim(</OmniverseKit_Persp>)
        Usd.Prim(</OmniverseKit_Front>)
        Usd.Prim(</OmniverseKit_Top>)
        Usd.Prim(</OmniverseKit_Right>)
        Usd.Prim(</Render>)
    """
    return get_current_stage(fabric=fabric).Traverse()


def is_stage_loading() -> bool:
    """Convenience function to see if any files are being loaded.

    Returns:
        bool: True if loading, False otherwise

    Example:

    .. code-block:: python

        >>> import isaaclab.sim.utils.stage as stage_utils
        >>>
        >>> stage_utils.is_stage_loading()
        False
    """
    import omni.usd

    context = omni.usd.get_context()
    if context is None:
        return False
    else:
        _, _, loading = context.get_stage_loading_status()
        return loading > 0


def set_stage_units(stage_units_in_meters: float) -> None:
    """Set the stage meters per unit

    The most common units and their values are listed in the following table:

    +------------------+--------+
    | Unit             | Value  |
    +==================+========+
    | kilometer (km)   | 1000.0 |
    +------------------+--------+
    | meters (m)       | 1.0    |
    +------------------+--------+
    | inch (in)        | 0.0254 |
    +------------------+--------+
    | centimeters (cm) | 0.01   |
    +------------------+--------+
    | millimeter (mm)  | 0.001  |
    +------------------+--------+

    Args:
        stage_units_in_meters (float): units for stage

    Example:

    .. code-block:: python

        >>> import isaaclab.sim.utils.stage as stage_utils
        >>>
        >>> stage_utils.set_stage_units(1.0)
    """
    if get_current_stage() is None:
        raise Exception("There is no stage currently opened, init_stage needed before calling this func")
    with Usd.EditContext(get_current_stage(), get_current_stage().GetRootLayer()):
        UsdGeom.SetStageMetersPerUnit(get_current_stage(), stage_units_in_meters)


def get_stage_units() -> float:
    """Get the stage meters per unit currently set

    The most common units and their values are listed in the following table:

    +------------------+--------+
    | Unit             | Value  |
    +==================+========+
    | kilometer (km)   | 1000.0 |
    +------------------+--------+
    | meters (m)       | 1.0    |
    +------------------+--------+
    | inch (in)        | 0.0254 |
    +------------------+--------+
    | centimeters (cm) | 0.01   |
    +------------------+--------+
    | millimeter (mm)  | 0.001  |
    +------------------+--------+

    Returns:
        float: current stage meters per unit

    Example:

    .. code-block:: python

        >>> import isaaclab.sim.utils.stage as stage_utils
        >>>
        >>> stage_utils.get_stage_units()
        1.0
    """
    return UsdGeom.GetStageMetersPerUnit(get_current_stage())


def get_next_free_path(path: str, parent: str = None) -> str:
    """Returns the next free usd path for the current stage

    Args:
        path (str): path we want to check
        parent (str, optional): Parent prim for the given path. Defaults to None.

    Returns:
        str: a new path that is guaranteed to not exist on the current stage

    Example:

    .. code-block:: python

        >>> import isaaclab.sim.utils.stage as stage_utils
        >>>
        >>> # given the stage: /World/Cube, /World/Cube_01.
        >>> # Get the next available path for /World/Cube
        >>> stage_utils.get_next_free_path("/World/Cube")
        /World/Cube_02
    """
    stage = get_current_stage()

    # Build the full path if parent is provided
    if parent is not None:
        full_path = parent.rstrip("/") + "/" + path.lstrip("/")
    else:
        full_path = path

    # Check if the original path is free
    if not stage.GetPrimAtPath(full_path).IsValid():
        return full_path

    # Find the next available path with a numeric suffix
    counter = 1
    while True:
        candidate_path = f"{full_path}_{counter:02d}"
        if not stage.GetPrimAtPath(candidate_path).IsValid():
            return candidate_path
        counter += 1


def remove_deleted_references():
    """Clean up deleted references in the current USD stage.

    Removes any deleted items from both payload and references lists
    for all prims in the stage's root layer. Prints information about
    any deleted items that were cleaned up.

    Example:

    .. code-block:: python

        >>> import isaaclab.sim.utils.stage as stage_utils
        >>> stage_utils.remove_deleted_references()
        Removed 2 deleted payload items from </World/Robot>
        Removed 1 deleted reference items from </World/Scene>
    """
    stage = get_current_stage()
    deleted_count = 0

    for prim in stage.Traverse():
        prim_spec = stage.GetRootLayer().GetPrimAtPath(prim.GetPath())
        if not prim_spec:
            continue

        # Clean payload references
        payload_list = prim_spec.GetInfo("payload")
        if payload_list.deletedItems:
            deleted_payload_count = len(payload_list.deletedItems)
            print(f"Removed {deleted_payload_count} deleted payload items from {prim.GetPath()}")
            payload_list.deletedItems = []
            prim_spec.SetInfo("payload", payload_list)
            deleted_count += deleted_payload_count

        # Clean prim references
        references_list = prim_spec.GetInfo("references")
        if references_list.deletedItems:
            deleted_ref_count = len(references_list.deletedItems)
            print(f"Removed {deleted_ref_count} deleted reference items from {prim.GetPath()}")
            references_list.deletedItems = []
            prim_spec.SetInfo("references", references_list)
            deleted_count += deleted_ref_count

    if deleted_count == 0:
        print("No deleted references or payloads found in the stage.")
