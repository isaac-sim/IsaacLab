# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.sim.spawners.wrappers import MultiUsdFileCfg
from isaaclab.utils.configclass import configclass

from .simready import SIMREADY_SEARCH_SERVICE_ENDPOINT, search_simready_usd_paths


@configclass
class SimReadyUsdFileCfg(UsdFileCfg):
    """Configuration parameters for spawning the top-ranked SimReady asset for a search query.

    This is a variant of :class:`~isaaclab.sim.spawners.from_files.UsdFileCfg` whose
    :attr:`usd_path` is resolved from a SimReady semantic search query when the configuration is
    instantiated. The top-ranked result is used. The service is queried exactly once: by the time
    the spawner runs, :attr:`usd_path` is a plain string and no network calls are made during
    simulation. When :attr:`usd_path` is already set (for instance, on a copy of a resolved
    configuration), the search is skipped.

    Please check :func:`~isaaclab.sim.spawners.simready.search_simready_usd_paths` for the
    required optional dependency and the service credentials.

    Example usage::

        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObjectCfg

        object_cfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/FoodBox",
            spawn=sim_utils.SimReadyUsdFileCfg(query="food box"),
        )
    """

    query: str = MISSING
    """SimReady free-text search phrase."""

    min_relevance: float = 0.0
    """Minimum relevance score for a match to be considered. Defaults to 0.0."""

    filter_profiles: list[str] | None = None
    """SimReady profile names that the match must have. Defaults to None."""

    filter_features: list[str] | None = None
    """SimReady feature names that the match must have. Defaults to None."""

    service_endpoint: str = SIMREADY_SEARCH_SERVICE_ENDPOINT
    """URL of the SimReady USD-Search service."""

    # note: redeclared so that __post_init__ can fill it from the search results.
    usd_path: str = MISSING
    """Path to the USD file to spawn asset from. Resolved from :attr:`query` when not set."""

    def __post_init__(self):
        # skip when already resolved: copy() and replace() re-run __post_init__ with resolved values.
        # note: configclass default factories deepcopy MISSING, so we match by type, not identity.
        if not isinstance(self.usd_path, type(MISSING)):
            return
        if isinstance(self.query, type(MISSING)):
            raise ValueError("Expected 'query' to be set when 'usd_path' is not provided.")
        self.usd_path = search_simready_usd_paths(
            self.query,
            top_k=1,
            min_relevance=self.min_relevance,
            filter_profiles=self.filter_profiles,
            filter_features=self.filter_features,
            service_endpoint=self.service_endpoint,
        )[0]


@configclass
class SimReadyMultiUsdFileCfg(MultiUsdFileCfg):
    """Configuration parameters for spawning multiple SimReady assets from a search query.

    This is a variant of :class:`~isaaclab.sim.spawners.wrappers.MultiUsdFileCfg` whose
    :attr:`usd_path` list is resolved from a SimReady semantic search query when the configuration
    is instantiated. The service is queried exactly once: after that, :attr:`usd_path` contains a
    frozen list of asset paths and the spawner behaves identically to a manually specified
    :class:`~isaaclab.sim.spawners.wrappers.MultiUsdFileCfg`. No network calls are made during
    simulation. When :attr:`usd_path` is already set (for instance, on a copy of a resolved
    configuration), the search is skipped.

    Since the resolved asset list depends on the state of the SimReady asset index, re-running a
    script may resolve a different list. The resolved paths are recorded in the environment
    configuration dump of a training run; pin them via :attr:`usd_path` to reproduce a run exactly.

    Please check :func:`~isaaclab.sim.spawners.simready.search_simready_usd_paths` for the
    required optional dependency and the service credentials.

    Example usage::

        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObjectCfg

        clutter_cfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Clutter_.*",
            spawn=sim_utils.SimReadyMultiUsdFileCfg(
                query="food box",
                top_k=15,
                mass_props=sim_utils.MassPropertiesCfg(mass=0.4),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
        )
    """

    query: str = MISSING
    """SimReady free-text search phrase."""

    top_k: int = 20
    """Maximum number of search results to include. Defaults to 20."""

    min_relevance: float = 0.0
    """Minimum relevance score for a match to be included. Defaults to 0.0."""

    filter_profiles: list[str] | None = None
    """SimReady profile names that every match must have. Defaults to None."""

    filter_features: list[str] | None = None
    """SimReady feature names that every match must have. Defaults to None."""

    service_endpoint: str = SIMREADY_SEARCH_SERVICE_ENDPOINT
    """URL of the SimReady USD-Search service."""

    # note: redeclared so that __post_init__ can fill it from the search results.
    usd_path: str | list[str] = MISSING
    """Paths to the USD files to spawn assets from. Resolved from :attr:`query` when not set."""

    def __post_init__(self):
        # skip when already resolved: copy() and replace() re-run __post_init__ with resolved values.
        # note: configclass default factories deepcopy MISSING, so we match by type, not identity.
        if not isinstance(self.usd_path, type(MISSING)):
            return
        if isinstance(self.query, type(MISSING)):
            raise ValueError("Expected 'query' to be set when 'usd_path' is not provided.")
        self.usd_path = search_simready_usd_paths(
            self.query,
            top_k=self.top_k,
            min_relevance=self.min_relevance,
            filter_profiles=self.filter_profiles,
            filter_features=self.filter_features,
            service_endpoint=self.service_endpoint,
        )
