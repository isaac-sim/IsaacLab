# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions to resolve USD asset paths from the SimReady USD-Search service."""

from __future__ import annotations

SIMREADY_SEARCH_SERVICE_ENDPOINT = "https://search.dev.simready.omniverse.nvidia.com/"
"""Default SimReady USD-Search service endpoint.

The service requires HTTP Basic authentication. The credentials are read by the ``simready-search``
package from the ``USD_SEARCH_USERNAME`` and ``USD_SEARCH_PASSWORD`` environment variables, which
must be set before the first search call.
"""


def search_simready_usd_paths(
    query: str | None = None,
    top_k: int = 20,
    min_relevance: float = 0.0,
    filter_profiles: list[str] | None = None,
    filter_features: list[str] | None = None,
    service_endpoint: str = SIMREADY_SEARCH_SERVICE_ENDPOINT,
) -> list[str]:
    """Search the SimReady USD-Search service and return the top-ranked USD asset paths.

    The results are ordered by descending relevance score. Ties are broken by the asset path so
    that repeated calls (for instance, one per process in distributed training) resolve the same
    ordering for identical service responses.

    This function requires the optional ``simready-search`` package. Install it with
    ``./isaaclab.sh -i simready`` or ``pip install simready-search``. The service credentials are
    read from the ``USD_SEARCH_USERNAME`` and ``USD_SEARCH_PASSWORD`` environment variables when
    the package is first imported.

    Args:
        query: Free-text search phrase (e.g. ``"food box"``). At least one of :paramref:`query`,
            :paramref:`filter_profiles`, or :paramref:`filter_features` must be provided.
        top_k: Maximum number of asset paths to return.
        min_relevance: Minimum relevance score for a match to be included.
        filter_profiles: SimReady profile names that every match must have
            (e.g. ``"Prop-Robotics-Isaac"``).
        filter_features: SimReady feature names that every match must have
            (e.g. ``"FET004_BASE_PHYSX"`` for rigid-body physics readiness).
        service_endpoint: URL of the USD-Search service.

    Returns:
        USD asset paths ordered by descending relevance, at most :paramref:`top_k` entries.

    Raises:
        ImportError: If the optional ``simready-search`` package is not installed.
        ValueError: If no search criterion is provided, ``top_k`` is not positive, or the search
            returned no results.
    """
    try:
        from simready.search import (
            AssetLibrary,
            SearchFilterFeature,
            SearchFilterPhrase,
            SearchFilterProfile,
            SearchFilterRelevance,
        )
    except ImportError as ex:
        raise ImportError(
            "SimReady asset search requires the optional 'simready-search' package. Install it with"
            " './isaaclab.sh -i simready' or 'pip install simready-search'."
        ) from ex

    if top_k < 1:
        raise ValueError(f"Expected 'top_k' to be positive, got: {top_k}.")
    if query is None and not filter_profiles and not filter_features:
        raise ValueError("At least one of 'query', 'filter_profiles', or 'filter_features' must be provided.")

    # assemble the search filters. a match must pass all of them.
    filters = []
    if query is not None:
        filters.append(SearchFilterPhrase(query))
    if min_relevance > 0.0:
        filters.append(SearchFilterRelevance(minimum=min_relevance))
    for profile in filter_profiles or []:
        filters.append(SearchFilterProfile(profile))
    for feature in filter_features or []:
        filters.append(SearchFilterFeature(feature))

    # raise on network/auth failures instead of the library default of returning empty results
    library = AssetLibrary(raise_on_network_error=True)
    library.add_service_source(service_endpoint)
    matches = library.search(include_all=filters, max_count=top_k)
    if not matches:
        raise ValueError(
            f"SimReady search returned no results for query: {query!r}. Check the query phrase,"
            " the filters, and the service endpoint."
        )
    # deterministic ordering: sort by descending relevance, breaking ties by asset path
    matches = sorted(matches, key=lambda match: (-(match.relevance_score or 0.0), match.asset_path))
    return [match.asset_path for match in matches]
