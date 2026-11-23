# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import os
import pathlib
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, search_simready_usd_paths

from ..from_files import UsdFileCfg
from ..spawner_cfg import DeformableObjectSpawnerCfg, RigidObjectSpawnerCfg, SpawnerCfg


@configclass
class MultiAssetSpawnerCfg(RigidObjectSpawnerCfg, DeformableObjectSpawnerCfg):
    """Configuration parameters for loading multiple assets from their individual configurations.

    Specifying values for any properties at the configuration level will override the settings of
    individual assets' configuration. For instance if the attribute
    :attr:`MultiAssetSpawnerCfg.mass_props` is specified, its value will overwrite the values of the
    mass properties in each configuration inside :attr:`assets_cfg` (wherever applicable).
    This is done to simplify configuring similar properties globally. By default, all properties are set to None.

    The following is an exception to the above:

    * :attr:`visible`: This parameter is ignored. Its value for the individual assets is used.
    * :attr:`semantic_tags`: If specified, it will be appended to each individual asset's semantic tags.

    """

    func: str = "{DIR}.wrappers:spawn_multi_asset"

    assets_cfg: list[SpawnerCfg] = MISSING
    """List of asset configurations to spawn."""

    spawn_paths: list[str | None] | None = None
    """Optional concrete spawn paths, one per asset configuration.

    When set, :func:`spawn_multi_asset` uses these paths instead of deriving
    sibling paths from the input ``prim_path``. Entries set to ``None`` are
    skipped.
    """

    static: bool = True
    """Keep each clone's initial asset choice. Set False to request reset-time mesh variants."""

    random_choice: bool = True
    """ This parameter is ignored.
    See ``clone_strategy`` in :class:`~isaaclab.cloner.CloneCfg` for details.

    .. warning::

        This attribute is deprecated. Use
        ``clone_strategy`` in :class:`~isaaclab.cloner.CloneCfg` instead.
    """


@configclass
class MultiUsdFileCfg(UsdFileCfg):
    """Configuration parameters for loading multiple USD files.

    Specifying values for any properties at the configuration level is applied to all the assets
    imported from their USD files.

    .. tip::
        It is recommended that all the USD based assets follow a similar prim-hierarchy.

    """

    func: str = "{DIR}.wrappers:spawn_multi_usd_file"

    usd_path: str | list[str] = MISSING
    """Path or a list of paths to the USD files to spawn asset from."""

    spawn_paths: list[str | None] | None = None
    """Optional concrete spawn paths, one per USD path.

    When set, :func:`spawn_multi_usd_file` uses these paths instead of deriving
    sibling paths from the input ``prim_path``. Entries set to ``None`` are
    skipped.
    """

    static: bool = True
    """Keep each clone's initial USD choice. Set False to request reset-time mesh variants."""

    random_choice: bool = True
    """Whether to randomly select an asset configuration. Default is True.

    If False, the asset configurations are spawned in the order they are provided in the list.
    If True, a random asset configuration is selected for each spawn.

    .. warning::

        This attribute is deprecated. Use
        ``clone_strategy`` in :class:`~isaaclab.cloner.CloneCfg` instead.
    """


@configclass
class SimReadyUsdFileCfg(MultiUsdFileCfg):
    """Configuration parameters for loading the USD files found by a SimReady search.

    The USD paths are resolved when the configuration is created, since the scene counts the files before spawning
    them. They are read from :attr:`cache_file` when it holds the same search; otherwise the search is run through
    :func:`~isaaclab.utils.assets.search_simready_usd_paths` and :attr:`cache_file` is rewritten with its result.
    Edit :attr:`search` in the configuration: a later change, such as a command-line override, is not searched again.
    """

    search: dict = MISSING
    """Keyword arguments of :func:`~isaaclab.utils.assets.search_simready_usd_paths`."""

    cache_file: str = MISSING
    """JSON file holding the search and its USD paths, rewritten when it holds another search."""

    move_rigid_body_to_root: bool = True
    """Whether to move each file's single rigid body onto the spawned prim. Defaults to True.

    SimReady props nest their rigid body at a file-specific path, such as ``/Geometry/socket_01_obj_00``,
    which files spawned together cannot share. Set it to False for files without exactly one rigid body,
    such as static props.
    """

    def __post_init__(self):
        root = f"{ISAAC_NUCLEUS_DIR}/SimReady/"
        # 1. the search as the cache stores it (tuples become lists), so that an unchanged search matches it
        search = json.loads(json.dumps(self.search))
        cache = pathlib.Path(self.cache_file)
        entry = json.loads(cache.read_text()) if cache.exists() else {}
        # 2. on another search, search again and cache the paths relative to the SimReady folder
        if entry.get("search") != search:
            paths = search_simready_usd_paths(**search)
            entry = {"search": search, "usd_paths": [path.removeprefix(root) for path in paths]}
            # written through a temporary file, so that processes loading the configuration together never read it
            # half written
            temp = cache.with_name(f"{cache.name}.{os.getpid()}")
            temp.write_text(json.dumps(entry, indent=2) + "\n")
            temp.replace(cache)
        # 3. the paths under the configured asset root
        self.usd_path = [root + path for path in entry["usd_paths"]]
