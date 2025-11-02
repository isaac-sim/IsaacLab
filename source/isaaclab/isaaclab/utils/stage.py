import builtins
import contextlib
import threading

# python
import typing

# omniverse
import omni
# import omni.log
import omni.kit.app
import usdrt

# isaacsim
from isaacsim.core.utils.constants import AXES_TOKEN
from omni.kit.usd import layers
from omni.metrics.assembler.core import get_metrics_assembler_interface
from omni.usd.commands import DeletePrimsCommand
from pxr import Sdf, Usd, UsdGeom, UsdUtils

_context = threading.local()  # thread-local storage to handle nested contexts and concurrent access


@contextlib.contextmanager
def use_stage(stage: Usd.Stage) -> None:
    """Context manager that sets a thread-local stage.

    Args:
        stage: The stage to set in the context.

    Raises:
        AssertionError: If the stage is not a USD stage instance.

    Example:

    .. code-block:: python

        >>> from pxr import Usd
        >>> import isaaclab.utils.stage as stage_utils
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
            delattr(_context, "stage")
        else:
            _context.stage = previous_stage


def get_current_stage(fabric: bool = False) -> typing.Union[Usd.Stage, usdrt.Usd._Usd.Stage]:
    """Get the current open USD or Fabric stage

    Args:
        fabric (bool, optional): True to get the fabric stage. False to get the USD stage. Defaults to False.

    Returns:
        typing.Union[Usd.Stage, usdrt.Usd._Usd.Stage]: The USD or Fabric stage as specified by the input arg fabric.

    Example:

    .. code-block:: python

        >>> import isaaclab.utils.stage as stage_utils
        >>>
        >>> stage_utils.get_current_stage()
        Usd.Stage.Open(rootLayer=Sdf.Find('anon:0x7fba6c04f840:World7.usd'),
                        sessionLayer=Sdf.Find('anon:0x7fba6c01c5c0:World7-session.usda'),
                        pathResolverContext=<invalid repr>)
    """
    stage = getattr(_context, "stage", omni.usd.get_context().get_stage())
    if fabric:
        stage_cache = UsdUtils.StageCache.Get()
        stage_id = stage_cache.GetId(stage).ToLongInt()
        if stage_id < 0:
            stage_id = stage_cache.Insert(stage).ToLongInt()
        return usdrt.Usd.Stage.Attach(stage_id)
    return stage


def get_current_stage_id() -> int:
    """Get the current open stage id

    Returns:
        int: The stage id.

    Example:

    .. code-block:: python

        >>> import isaaclab.utils.stage as stage_utils
        >>>
        >>> stage_utils.get_current_stage_id()
        1234567890
    """
    stage = get_current_stage()
    stage_cache = UsdUtils.StageCache.Get()
    stage_id = stage_cache.GetId(stage).ToLongInt()
    if stage_id < 0:
        stage_id = stage_cache.Insert(stage).ToLongInt()
    return stage_id


def update_stage() -> None:
    """Update the current USD stage.

    Example:

    .. code-block:: python

        >>> import isaaclab.utils.stage as stage_utils
        >>>
        >>> stage_utils.update_stage()
    """
    omni.kit.app.get_app_interface().update()


async def update_stage_async() -> None:
    """Update the current USD stage (asynchronous version).

    Example:

    .. code-block:: python

        >>> import asyncio
        >>> import isaaclab.utils.stage as stage_utils
        >>> from omni.kit.async_engine import run_coroutine
        >>>
        >>> async def task():
        ...     await stage_utils.update_stage_async()
        ...
        >>> run_coroutine(task())
    """
    await omni.kit.app.get_app().next_update_async()


# TODO: make a generic util for setting all layer properties
def set_stage_up_axis(axis: str = "z") -> None:
    """Change the up axis of the current stage

    Args:
        axis (UsdGeom.Tokens, optional): valid values are ``"x"``, ``"y"`` and ``"z"``

    Example:

    .. code-block:: python

        >>> import isaaclab.utils.stage as stage_utils
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

        >>> import isaaclab.utils.stage as stage_utils
        >>>
        >>> stage_utils.get_stage_up_axis()
        Z
    """
    stage = get_current_stage()
    return UsdGeom.GetStageUpAxis(stage)


def clear_stage(predicate: typing.Optional[typing.Callable[[str], bool]] = None) -> None:
    """Deletes all prims in the stage without populating the undo command buffer

    Args:
        predicate (typing.Optional[typing.Callable[[str], bool]], optional):
            user defined function that takes a prim_path (str) as input and returns True/False if the prim
            should/shouldn't be deleted. If predicate is None, a default is used that deletes all prims

    Example:

    .. code-block:: python

        >>> import isaaclab.utils.stage as stage_utils
        >>>
        >>> # clear the whole stage
        >>> stage_utils.clear_stage()
        >>>
        >>> # given the stage: /World/Cube, /World/Cube_01, /World/Cube_02.
        >>> # Delete only the prims of type Cube
        >>> predicate = lambda path: prims_utils.get_prim_type_name(path) == "Cube"
        >>> stage_utils.clear_stage(predicate)  # after the execution the stage will be /World
    """
    # Note: Need to import this here to prevent circular dependencies.
    from isaaclab.utils.prims import (
        get_all_matching_child_prims,
        get_prim_path,
        is_prim_ancestral,
        is_prim_hidden_in_stage,
        is_prim_no_delete,
    )

    def default_predicate(prim_path: str):
        # prim = get_prim_at_path(prim_path)
        # skip prims that we cannot delete
        if is_prim_no_delete(prim_path):
            return False
        if is_prim_hidden_in_stage(prim_path):
            return False
        if is_prim_ancestral(prim_path):
            return False
        if prim_path == "/":
            return False
        if prim_path.startswith("/Render"):
            return False
        return True

    if predicate is None:
        prims = get_all_matching_child_prims("/", default_predicate)
        prim_paths_to_delete = [get_prim_path(prim) for prim in prims]
        DeletePrimsCommand(prim_paths_to_delete).do()
    else:
        prims = get_all_matching_child_prims("/", predicate)
        prim_paths_to_delete = [get_prim_path(prim) for prim in prims]
        DeletePrimsCommand(prim_paths_to_delete).do()

    if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
        omni.kit.app.get_app_interface().update()


def print_stage_prim_paths(fabric: bool = False) -> None:
    """Traverses the stage and prints all prim (hidden or not) paths.

    Example:

    .. code-block:: python

        >>> import isaaclab.utils.stage as stage_utils
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
    from isaaclab.utils.prims import get_prim_path

    for prim in traverse_stage(fabric=fabric):
        prim_path = get_prim_path(prim)
        print(prim_path)


def add_reference_to_stage(usd_path: str, prim_path: str, prim_type: str = "Xform") -> Usd.Prim:
    """Add USD reference to the opened stage at specified prim path.

    Adds a reference to an external USD file at the specified prim path on the current stage.
    If the prim does not exist, it will be created with the specified type.
    This function also handles stage units verification to ensure compatibility.

    Args:
        usd_path: The path to USD file to reference.
        prim_path: The prim path where the reference will be attached.
        prim_type: The type of prim to create if it doesn't exist. Defaults to "Xform".

    Returns:
        The USD prim at the specified prim path.

    Raises:
        FileNotFoundError: When the input USD file is not found at the specified path.

    Example:

    .. code-block:: python

        >>> import isaaclab.utils.stage as stage_utils
        >>>
        >>> # load an USD file (franka.usd) to the stage under the path /World/panda
        >>> prim = stage_utils.add_reference_to_stage(
        ...     usd_path="/home/<user>/Documents/Assets/Robots/FrankaRobotics/FrankaPanda/franka.usd",
        ...     prim_path="/World/panda"
        ... )
        >>> prim
        Usd.Prim(</World/panda>)
    """
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        prim = stage.DefinePrim(prim_path, prim_type)
    # omni.log.info("Loading Asset from path {} ".format(usd_path))
    # Handle units
    sdf_layer = Sdf.Layer.FindOrOpen(usd_path)
    if not sdf_layer:
        pass
        # omni.log.info(f"Could not get Sdf layer for {usd_path}")
    else:
        stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        ret_val = get_metrics_assembler_interface().check_layers(
            stage.GetRootLayer().identifier, sdf_layer.identifier, stage_id
        )
        if ret_val["ret_val"]:
            try:
                import omni.metrics.assembler.ui

                payref = Sdf.Reference(usd_path)
                omni.kit.commands.execute("AddReference", stage=stage, prim_path=prim.GetPath(), reference=payref)
            except Exception as exc:
                # omni.log.warn(
                #     f"The USD file {usd_path} used for a reference does have divergent units, please either enable omni.usd.metrics.assembler.ui or convert the file into right units."
                # )
                success_bool = prim.GetReferences().AddReference(usd_path)
                if not success_bool:
                    raise FileNotFoundError("The usd file at path {} provided wasn't found".format(usd_path))
        else:
            success_bool = prim.GetReferences().AddReference(usd_path)
            if not success_bool:
                raise FileNotFoundError("The usd file at path {} provided wasn't found".format(usd_path))

    return prim


def create_new_stage() -> Usd.Stage:
    """Create a new stage attached to the USD context.

    Returns:
        Usd.Stage: The created USD stage.

    Example:

    .. code-block:: python

        >>> import isaaclab.utils.stage as stage_utils
        >>>
        >>> stage_utils.create_new_stage()
        Usd.Stage.Open(rootLayer=Sdf.Find('anon:0x7fba6c04f840:World7.usd'),
                        sessionLayer=Sdf.Find('anon:0x7fba6c01c5c0:World7-session.usda'),
                        pathResolverContext=<invalid repr>)
    """
    return omni.usd.get_context().new_stage()


def create_new_stage_in_memory() -> Usd.Stage:
    """Create a new stage in memory.

    Returns:
        Usd.Stage: The created USD stage.

    Example:

    .. code-block:: python

        >>> import isaaclab.utils.stage as stage_utils
        >>>
        >>> stage_utils.create_new_stage_in_memory()
        Usd.Stage.Open(rootLayer=Sdf.Find('anon:0xf7b00e0:tmp.usda'),
                        sessionLayer=Sdf.Find('anon:0xf7cd2e0:tmp-session.usda'),
                        pathResolverContext=<invalid repr>)
    """
    return Usd.Stage.CreateInMemory()


async def create_new_stage_async() -> None:
    """Create a new stage (asynchronous version).

    Example:

    .. code-block:: python

        >>> import asyncio
        >>> import isaaclab.utils.stage as stage_utils
        >>> from omni.kit.async_engine import run_coroutine
        >>>
        >>> async def task():
        ...     await stage_utils.create_new_stage_async()
        ...
        >>> run_coroutine(task())
    """
    await omni.usd.get_context().new_stage_async()
    await omni.kit.app.get_app().next_update_async()


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

        >>> import isaaclab.utils.stage as stage_utils
        >>>
        >>> stage_utils.open_stage("/home/<user>/Documents/Assets/Robots/FrankaRobotics/FrankaPanda/franka.usd")
        True
    """
    if not Usd.Stage.IsSupportedFile(usd_path):
        raise ValueError("Only USD files can be loaded with this method")
    usd_context = omni.usd.get_context()
    usd_context.disable_save_to_recent_files()
    result = omni.usd.get_context().open_stage(usd_path)
    usd_context.enable_save_to_recent_files()
    return result


async def open_stage_async(usd_path: str) -> typing.Tuple[bool, int]:
    """Open the given usd file and replace currently opened stage (asynchronous version).

    Args:
        usd_path (str): Path to the USD file to open.

    Raises:
        ValueError: When input path is not a supported file type by USD.

    Returns:
        bool: True if operation is successful, otherwise false.

    Example:

    .. code-block:: python

        >>> import asyncio
        >>> import isaaclab.utils.stage as stage_utils
        >>> from omni.kit.async_engine import run_coroutine
        >>>
        >>> async def task():
        ...     await stage_utils.open_stage_async("/home/<user>/Documents/Assets/Robots/FrankaRobotics/FrankaPanda/franka.usd")
        ...
        >>> run_coroutine(task())
    """
    if not Usd.Stage.IsSupportedFile(usd_path):
        raise ValueError("Only USD files can be loaded with this method")
    usd_context = omni.usd.get_context()
    usd_context.disable_save_to_recent_files()
    (result, error) = await omni.usd.get_context().open_stage_async(usd_path)
    usd_context.enable_save_to_recent_files()
    return (result, error)


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

        >>> import isaaclab.utils.stage as stage_utils
        >>>
        >>> stage_utils.save_stage("/home/<user>/Documents/Save/stage.usd")
        True
    """
    if not Usd.Stage.IsSupportedFile(usd_path):
        raise ValueError("Only USD files can be saved with this method")

    layer = Sdf.Layer.CreateNew(usd_path)
    root_layer = get_current_stage().GetRootLayer()
    layer.TransferContent(root_layer)
    omni.usd.resolve_paths(root_layer.identifier, layer.identifier)
    result = layer.Save()
    if save_and_reload_in_place:
        open_stage(usd_path)

    return result


def close_stage(callback_fn: typing.Callable = None) -> bool:
    """Closes the current opened USD stage.

    .. note::

        Once the stage is closed, it is necessary to open a new stage or create a new one in order to work on it.

    Args:
        callback_fn (typing.Callable, optional): Callback function to call while closing. Defaults to None.

    Returns:
        bool: True if operation is successful, otherwise false.

    Example:

    .. code-block:: python

        >>> import isaaclab.utils.stage as stage_utils
        >>>
        >>> stage_utils.close_stage()
        True

    .. code-block:: python

        >>> import isaaclab.utils.stage as stage_utils
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
    if callback_fn is None:
        result = omni.usd.get_context().close_stage()
    else:
        result = omni.usd.get_context().close_stage_with_callback(callback_fn)
    return result


def set_livesync_stage(usd_path: str, enable: bool) -> bool:
    """Save the stage and set the Live Sync mode for real-time live editing of shared files on a Nucleus server

    Args:
        usd_path (str): path to enable live sync for, it will be overwritten with the current stage
        enable (bool): True to enable livesync, false to disable livesync

    Returns:
        bool: True if operation is successful, otherwise false.

    Example:

    .. code-block:: python

        >>> import isaaclab.utils.stage as stage_utils
        >>>
        >>> stage_utils.set_livesync_stage("omniverse://localhost/Users/Live/stage.usd", enable=True)
        server omniverse://localhost: ConnectionStatus.CONNECTING
        server omniverse://localhost: ConnectionStatus.CONNECTED
        True
    """
    # TODO: Check that the provided usd_path exists
    # TODO: Check that the provided usd_path exists
    if save_stage(usd_path):
        if enable:
            usd_path_split = usd_path.split("/")
            live_session = layers.get_live_syncing().find_live_session_by_name(usd_path, "Default")
            if live_session is None:
                live_session = layers.get_live_syncing().create_live_session(name="Default")
            result = layers.get_live_syncing().join_live_session(live_session)
            return True
        else:
            layers.get_live_syncing().stop_live_session(usd_path)
            return True
    else:
        return False


def traverse_stage(fabric=False) -> typing.Iterable:
    """Traverse through prims (hidden or not) in the opened Usd stage.

    Returns:
        typing.Iterable: Generator which yields prims from the stage in depth-first-traversal order.

    Example:

    .. code-block:: python

        >>> import isaaclab.utils.stage as stage_utils
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

        >>> import isaaclab.utils.stage as stage_utils
        >>>
        >>> stage_utils.is_stage_loading()
        False
    """
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

        >>> import isaaclab.utils.stage as stage_utils
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

        >>> import isaaclab.utils.stage as stage_utils
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

        >>> import isaaclab.utils.stage as stage_utils
        >>>
        >>> # given the stage: /World/Cube, /World/Cube_01.
        >>> # Get the next available path for /World/Cube
        >>> stage_utils.get_next_free_path("/World/Cube")
        /World/Cube_02
    """
    if parent is not None:
        # remove trailing slash from parent and leading slash from path
        path = omni.usd.get_stage_next_free_path(
            get_current_stage(), parent.rstrip("/") + "/" + path.lstrip("/"), False
        )
    else:
        path = omni.usd.get_stage_next_free_path(get_current_stage(), path, True)
    return path


def remove_deleted_references():
    """Clean up deleted references in the current USD stage.

    Removes any deleted items from both payload and references lists
    for all prims in the stage's root layer. Prints information about
    any deleted items that were cleaned up.

    Example:

    .. code-block:: python

        >>> import isaaclab.utils.stage as stage_utils
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
