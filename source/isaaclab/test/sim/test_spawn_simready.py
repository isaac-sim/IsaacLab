# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the SimReady USD-Search spawner configurations.

The tests stub :func:`isaaclab.utils.assets.search_simready_usd_paths` so they run without the
optional ``simready-search`` package and without network access. The resolver itself is tested in
``test/utils/test_assets.py``.
"""

import types

import pytest

import isaaclab.sim.spawners.simready.simready_cfg as simready_cfg_module
from isaaclab.sim.spawners.simready import SimReadyMultiUsdFileCfg, SimReadyUsdFileCfg

pytestmark = pytest.mark.unit


@pytest.fixture
def fake_search(monkeypatch):
    """Stub the SimReady search resolver and return its call-recording state."""
    state = types.SimpleNamespace(results=[], calls=[])

    def _fake_search(query=None, top_k=20, **kwargs):
        state.calls.append({"query": query, "top_k": top_k, **kwargs})
        return list(state.results[:top_k])

    # patch the name bound at import time in the cfg module, not the utils.assets original
    monkeypatch.setattr(simready_cfg_module, "search_simready_usd_paths", _fake_search)
    return state


def test_usd_file_cfg_resolves_top_result(fake_search):
    fake_search.results = ["omniverse://best.usd", "omniverse://second.usd"]
    cfg = SimReadyUsdFileCfg(query="food box")
    assert cfg.usd_path == "omniverse://best.usd"
    assert fake_search.calls[0]["top_k"] == 1


def test_usd_file_cfg_forwards_search_parameters(fake_search):
    fake_search.results = ["omniverse://best.usd"]
    cfg = SimReadyUsdFileCfg(
        query="food box",
        min_relevance=0.5,
        filter_profiles=["Prop-Robotics-Isaac"],
        filter_features=["FET004_BASE_PHYSX"],
    )
    assert cfg.usd_path == "omniverse://best.usd"
    assert fake_search.calls[0] == {
        "query": "food box",
        "top_k": 1,
        "min_relevance": 0.5,
        "filter_profiles": ["Prop-Robotics-Isaac"],
        "filter_features": ["FET004_BASE_PHYSX"],
        "service_endpoint": simready_cfg_module.SIMREADY_SEARCH_SERVICE_ENDPOINT,
    }


def test_usd_file_cfg_copy_does_not_requery(fake_search):
    fake_search.results = ["omniverse://best.usd"]
    cfg = SimReadyUsdFileCfg(query="food box")
    cfg_copy = cfg.copy()
    assert cfg_copy.usd_path == cfg.usd_path
    assert len(fake_search.calls) == 1


def test_usd_file_cfg_requires_query(fake_search):
    with pytest.raises(ValueError, match="query"):
        SimReadyUsdFileCfg()


def test_multi_usd_file_cfg_resolves_path_list(fake_search):
    fake_search.results = [f"omniverse://asset_{i}.usd" for i in range(5)]
    cfg = SimReadyMultiUsdFileCfg(query="food box", top_k=3)
    assert cfg.usd_path == [f"omniverse://asset_{i}.usd" for i in range(3)]
    assert len(fake_search.calls) == 1


def test_multi_usd_file_cfg_replace_does_not_requery(fake_search):
    fake_search.results = ["omniverse://asset.usd"]
    cfg = SimReadyMultiUsdFileCfg(query="food box")
    replaced = cfg.replace(random_choice=False)
    assert replaced.usd_path == cfg.usd_path
    assert len(fake_search.calls) == 1


def test_multi_usd_file_cfg_keeps_explicit_usd_path(fake_search):
    """An explicitly provided usd_path skips the search entirely."""
    cfg = SimReadyMultiUsdFileCfg(query="food box", usd_path=["omniverse://pinned.usd"])
    assert cfg.usd_path == ["omniverse://pinned.usd"]
    assert cfg.query == "food box"
    assert len(fake_search.calls) == 0
