# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the SimReady USD-Search spawner configurations.

The tests install a fake ``simready.search`` module so they run without the optional
``simready-search`` package and without network access.
"""

import sys
import types

import pytest

from isaaclab.sim.spawners.simready import SimReadyMultiUsdFileCfg, SimReadyUsdFileCfg, search_simready_usd_paths

pytestmark = pytest.mark.unit


@pytest.fixture
def fake_simready(monkeypatch):
    """Install a fake ``simready.search`` module and return its call-recording state."""
    state = types.SimpleNamespace(results=[], calls=[], endpoint=None, raise_on_network_error=None)

    class FakeFilter:
        def __init__(self, value="", **kwargs):
            self.value = value
            self.kwargs = kwargs

    class FakeAssetLibrary:
        def __init__(self, log_func=None, raise_on_network_error=False):
            state.raise_on_network_error = raise_on_network_error

        def add_service_source(self, endpoint):
            state.endpoint = endpoint

        def search(self, include_all=None, max_count=None, **kwargs):
            state.calls.append({"include_all": include_all, "max_count": max_count})
            matches = state.results
            if max_count is not None:
                matches = matches[:max_count]
            return list(matches)

    module = types.ModuleType("simready.search")
    module.AssetLibrary = FakeAssetLibrary
    for name in ("SearchFilterFeature", "SearchFilterPhrase", "SearchFilterProfile", "SearchFilterRelevance"):
        setattr(module, name, type(name, (FakeFilter,), {}))
    package = types.ModuleType("simready")
    package.search = module
    monkeypatch.setitem(sys.modules, "simready", package)
    monkeypatch.setitem(sys.modules, "simready.search", module)
    return state


def _match(asset_path, relevance_score):
    return types.SimpleNamespace(asset_path=asset_path, relevance_score=relevance_score)


def test_search_orders_by_relevance_with_path_tiebreak(fake_simready):
    """Results are sorted by descending score; ties and missing scores are deterministic."""
    fake_simready.results = [
        _match("omniverse://b_box.usd", 0.5),
        _match("omniverse://no_score.usd", None),
        _match("omniverse://best.usd", 0.9),
        _match("omniverse://a_box.usd", 0.5),
    ]
    paths = search_simready_usd_paths("food box")
    assert paths == [
        "omniverse://best.usd",
        "omniverse://a_box.usd",
        "omniverse://b_box.usd",
        "omniverse://no_score.usd",
    ]
    # network and auth failures must raise instead of looking like an empty result
    assert fake_simready.raise_on_network_error is True


def test_search_top_k_limits_result_count(fake_simready):
    fake_simready.results = [_match(f"omniverse://asset_{i}.usd", 1.0 - 0.1 * i) for i in range(5)]
    paths = search_simready_usd_paths("food box", top_k=2)
    assert len(paths) == 2
    assert fake_simready.calls[0]["max_count"] == 2


def test_search_applies_optional_filters(fake_simready):
    fake_simready.results = [_match("omniverse://asset.usd", 1.0)]
    search_simready_usd_paths(
        "food box", min_relevance=0.5, filter_profiles=["Prop-Robotics-Isaac"], filter_features=["FET004_BASE_PHYSX"]
    )
    filter_names = [type(f).__name__ for f in fake_simready.calls[0]["include_all"]]
    assert filter_names == [
        "SearchFilterPhrase",
        "SearchFilterRelevance",
        "SearchFilterProfile",
        "SearchFilterFeature",
    ]


def test_search_raises_on_empty_results(fake_simready):
    fake_simready.results = []
    with pytest.raises(ValueError, match="no results.*food box"):
        search_simready_usd_paths("food box")


def test_search_requires_a_criterion(fake_simready):
    with pytest.raises(ValueError, match="At least one of"):
        search_simready_usd_paths()


def test_search_raises_helpful_error_without_package(monkeypatch):
    """A missing 'simready-search' package produces an error with install instructions."""
    monkeypatch.setitem(sys.modules, "simready", None)
    monkeypatch.setitem(sys.modules, "simready.search", None)
    with pytest.raises(ImportError, match="simready-search"):
        search_simready_usd_paths("food box")


def test_usd_file_cfg_resolves_top_result(fake_simready):
    fake_simready.results = [_match("omniverse://best.usd", 0.9)]
    cfg = SimReadyUsdFileCfg(query="food box")
    assert cfg.usd_path == "omniverse://best.usd"
    assert fake_simready.calls[0]["max_count"] == 1


def test_usd_file_cfg_copy_does_not_requery(fake_simready):
    fake_simready.results = [_match("omniverse://best.usd", 0.9)]
    cfg = SimReadyUsdFileCfg(query="food box")
    cfg_copy = cfg.copy()
    assert cfg_copy.usd_path == cfg.usd_path
    assert len(fake_simready.calls) == 1


def test_usd_file_cfg_requires_query(fake_simready):
    with pytest.raises(ValueError, match="query"):
        SimReadyUsdFileCfg()


def test_multi_usd_file_cfg_resolves_path_list(fake_simready):
    fake_simready.results = [_match(f"omniverse://asset_{i}.usd", 1.0 - 0.1 * i) for i in range(5)]
    cfg = SimReadyMultiUsdFileCfg(query="food box", top_k=3)
    assert cfg.usd_path == [f"omniverse://asset_{i}.usd" for i in range(3)]
    assert len(fake_simready.calls) == 1


def test_multi_usd_file_cfg_replace_does_not_requery(fake_simready):
    fake_simready.results = [_match("omniverse://asset.usd", 1.0)]
    cfg = SimReadyMultiUsdFileCfg(query="food box")
    replaced = cfg.replace(random_choice=False)
    assert replaced.usd_path == cfg.usd_path
    assert len(fake_simready.calls) == 1


def test_multi_usd_file_cfg_keeps_explicit_usd_path(fake_simready):
    """An explicitly provided usd_path skips the search entirely."""
    cfg = SimReadyMultiUsdFileCfg(query="food box", usd_path=["omniverse://pinned.usd"])
    assert cfg.usd_path == ["omniverse://pinned.usd"]
    assert cfg.query == "food box"
    assert len(fake_simready.calls) == 0
