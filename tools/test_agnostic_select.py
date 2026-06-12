# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the agnostic-drop plugin (``tools/_agnostic_select.py``).

The plugin deselects every collected test that is not parametrized over
``device``. It carries no mask logic; the executor injects it only for units
whose mask lacks cpu (mgpu shards and the cuda unit of a can't-mix split), where
device-agnostic tests must not run.
"""

from __future__ import annotations

from _agnostic_select import pytest_collection_modifyitems


class _FakeCallspec:
    def __init__(self, params):
        self.params = params


class _FakeItem:
    """Stand-in for a pytest item. Omitting ``params`` means no callspec at all."""

    def __init__(self, name, params=None):
        self.name = name
        if params is not None:
            self.callspec = _FakeCallspec(params)


class _FakeHook:
    def __init__(self):
        self.deselected = []

    def pytest_deselected(self, items):
        self.deselected.extend(items)


class _FakeConfig:
    def __init__(self):
        self.hook = _FakeHook()


def test_drops_items_without_a_device_param():
    cpu = _FakeItem("t[cpu]", {"device": "cpu"})
    gpu = _FakeItem("t[cuda:0]", {"device": "cuda:0"})
    agnostic = _FakeItem("t_logic")  # no callspec
    items = [cpu, gpu, agnostic]
    config = _FakeConfig()

    pytest_collection_modifyitems(config, items)

    assert items == [cpu, gpu]
    assert config.hook.deselected == [agnostic]


def test_drops_parametrized_item_that_lacks_a_device_key():
    # Parametrized over something else (num_envs) but not device -> still agnostic.
    cpu = _FakeItem("t[cpu]", {"device": "cpu"})
    other = _FakeItem("t[2]", {"num_envs": 2})
    items = [cpu, other]
    config = _FakeConfig()

    pytest_collection_modifyitems(config, items)

    assert items == [cpu]
    assert config.hook.deselected == [other]


def test_keeps_all_when_every_item_has_a_device_param():
    cpu = _FakeItem("t[cpu]", {"device": "cpu"})
    gpu = _FakeItem("t[cuda:1]", {"device": "cuda:1"})
    items = [cpu, gpu]
    config = _FakeConfig()

    pytest_collection_modifyitems(config, items)

    assert items == [cpu, gpu]
    assert config.hook.deselected == []  # nothing deselected


def test_empty_item_list_is_a_no_op():
    items = []
    config = _FakeConfig()

    pytest_collection_modifyitems(config, items)

    assert items == []
    assert config.hook.deselected == []
