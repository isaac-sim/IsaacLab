# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Tests for recursive _process_term_cfg_at_play / _resolve_param_value.

These tests exercise ManagerBase's parameter resolution logic and do NOT
require an Isaac Sim launch, so they can run without AppLauncher.
"""

from collections import namedtuple
from collections.abc import Sequence
from unittest.mock import MagicMock

import pytest
import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ManagerTermBase, ManagerTermBaseCfg
from isaaclab.managers.manager_base import ManagerBase


DummyEnv = namedtuple("ManagerBasedRLEnv", ["num_envs", "dt", "device", "sim", "dummy1", "dummy2"])
"""Dummy environment for testing."""


class SimpleManager(ManagerBase):
    """Minimal concrete ManagerBase for testing term resolution."""

    def __init__(self, cfg: dict, env):
        self._term_cfgs: list[tuple[str, ManagerTermBaseCfg]] = []
        super().__init__(cfg, env)

    @property
    def active_terms(self) -> list[str]:
        return [name for name, _ in self._term_cfgs]

    def _prepare_terms(self):
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        for term_name, term_cfg in cfg_items:
            if term_cfg is None:
                continue
            self._resolve_common_term_cfg(term_name, term_cfg, min_argc=2)
            self._term_cfgs.append((term_name, term_cfg))

    def apply(self, env_ids: torch.Tensor):
        """Call each registered term with the environment."""
        for _, term_cfg in self._term_cfgs:
            term_cfg.func(self._env, env_ids, **term_cfg.params)


def increment_dummy1_by_one(env, env_ids: torch.Tensor):
    env.dummy1[env_ids] += 1


def change_dummy1_by_value(env, env_ids: torch.Tensor, value: int):
    env.dummy1[env_ids] += value


def reset_dummy2_to_zero(env, env_ids: torch.Tensor):
    env.dummy2[env_ids] = 0


class reset_dummy2_to_zero_class(ManagerTermBase):
    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        pass

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
    ) -> None:
        env.dummy2[env_ids] = 0


class chained_terms_class(ManagerTermBase):
    """A class-based term whose params contain nested ManagerTermBaseCfg dicts."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.sub_terms: dict[str, ManagerTermBaseCfg] = cfg.params["terms"]

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        pass

    def __call__(self, env: ManagerBasedEnv, env_ids: torch.Tensor, terms: dict) -> None:
        for term_cfg in terms.values():
            term_cfg.func(env, env_ids, **term_cfg.params)


class list_terms_class(ManagerTermBase):
    """A class-based term whose params contain a list of nested ManagerTermBaseCfg."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.sub_terms: list[ManagerTermBaseCfg] = cfg.params["term_list"]

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        pass

    def __call__(self, env: ManagerBasedEnv, env_ids: torch.Tensor, term_list: list) -> None:
        for term_cfg in term_list:
            term_cfg.func(env, env_ids, **term_cfg.params)


@pytest.fixture
def env():
    num_envs = 2
    device = "cpu"
    dummy1 = torch.zeros((num_envs, 2), device=device)
    dummy2 = torch.zeros((num_envs, 10), device=device)
    sim = MagicMock()
    sim.is_playing.return_value = True
    return DummyEnv(num_envs, 0.01, device, sim, dummy1, dummy2)


def test_nested_term_cfg_in_dict_params(env):
    """Test that nested ManagerTermBaseCfg inside a dict param are recursively resolved."""
    cfg = {
        "chained": ManagerTermBaseCfg(
            func=chained_terms_class,
            params={
                "terms": {
                    "step_a": ManagerTermBaseCfg(func=increment_dummy1_by_one),
                    "step_b": ManagerTermBaseCfg(func=change_dummy1_by_value, params={"value": 5}),
                }
            },
        ),
    }
    manager = SimpleManager(cfg, env)

    # The outer chained_terms_class should be instantiated (class -> instance).
    outer_cfg = manager._term_cfgs[0][1]
    assert isinstance(outer_cfg.func, chained_terms_class)

    # Inner terms should have their func resolved to callables (not strings).
    inner_terms = outer_cfg.params["terms"]
    assert callable(inner_terms["step_a"].func), "Nested func should be resolved to a callable"
    assert callable(inner_terms["step_b"].func), "Nested func should be resolved to a callable"
    assert inner_terms["step_a"].func is increment_dummy1_by_one
    assert inner_terms["step_b"].func is change_dummy1_by_value

    # Functionally: applying the chained term should run both inner terms.
    manager.apply(torch.arange(env.num_envs, device=env.device))
    # increment_dummy1_by_one adds 1, then change_dummy1_by_value adds 5 -> total 6
    torch.testing.assert_close(env.dummy1, 6 * torch.ones_like(env.dummy1))


def test_nested_term_cfg_in_list_params(env):
    """Test that nested ManagerTermBaseCfg inside a list param are recursively resolved."""
    cfg = {
        "list_chained": ManagerTermBaseCfg(
            func=list_terms_class,
            params={
                "term_list": [
                    ManagerTermBaseCfg(func=increment_dummy1_by_one),
                    ManagerTermBaseCfg(func=change_dummy1_by_value, params={"value": 3}),
                ]
            },
        ),
    }
    manager = SimpleManager(cfg, env)

    # Inner terms in the list should have callable funcs.
    outer_cfg = manager._term_cfgs[0][1]
    term_list = outer_cfg.params["term_list"]
    assert isinstance(term_list, list)
    assert callable(term_list[0].func)
    assert callable(term_list[1].func)

    # Apply and verify: +1 then +3 -> total 4
    manager.apply(torch.arange(env.num_envs, device=env.device))
    torch.testing.assert_close(env.dummy1, 4 * torch.ones_like(env.dummy1))


def test_string_func_in_nested_term_cfg(env):
    """Test that string-based func references inside nested term cfgs are resolved."""
    this_module = __name__

    cfg = {
        "chained": ManagerTermBaseCfg(
            func=chained_terms_class,
            params={
                "terms": {
                    "step_a": ManagerTermBaseCfg(
                        func=f"{this_module}:increment_dummy1_by_one",
                    ),
                    "step_b": ManagerTermBaseCfg(
                        func=f"{this_module}:change_dummy1_by_value",
                        params={"value": 10},
                    ),
                }
            },
        ),
    }
    manager = SimpleManager(cfg, env)

    # String funcs in nested terms should be resolved to actual callables.
    outer_cfg = manager._term_cfgs[0][1]
    inner_terms = outer_cfg.params["terms"]
    assert inner_terms["step_a"].func is increment_dummy1_by_one
    assert inner_terms["step_b"].func is change_dummy1_by_value

    # Apply and verify: +1 then +10 -> 11
    manager.apply(torch.arange(env.num_envs, device=env.device))
    torch.testing.assert_close(env.dummy1, 11 * torch.ones_like(env.dummy1))


def test_string_func_top_level_class_term(env):
    """Test that a top-level string-based func pointing to a class is properly instantiated."""
    this_module = __name__

    cfg = {
        "term_class_str": ManagerTermBaseCfg(
            func=f"{this_module}:reset_dummy2_to_zero_class",
        ),
    }
    manager = SimpleManager(cfg, env)

    # The string func should be resolved and the class instantiated.
    outer_cfg = manager._term_cfgs[0][1]
    assert isinstance(outer_cfg.func, reset_dummy2_to_zero_class)

    # Verify it works: set dummy2 to non-zero, apply, check it's zero.
    env.dummy2[:] = 42.0
    manager.apply(torch.arange(env.num_envs, device=env.device))
    torch.testing.assert_close(env.dummy2, torch.zeros_like(env.dummy2))


def test_deeply_nested_dict_in_params(env):
    """Test that term cfgs are resolved even when nested inside dict values."""
    cfg = {
        "chained": ManagerTermBaseCfg(
            func=chained_terms_class,
            params={
                "terms": {
                    "only": ManagerTermBaseCfg(
                        func=change_dummy1_by_value,
                        params={"value": 7},
                    ),
                }
            },
        ),
    }
    manager = SimpleManager(cfg, env)

    outer_cfg = manager._term_cfgs[0][1]
    inner_cfg = outer_cfg.params["terms"]["only"]
    assert callable(inner_cfg.func)
    assert inner_cfg.params == {"value": 7}

    manager.apply(torch.arange(env.num_envs, device=env.device))
    torch.testing.assert_close(env.dummy1, 7 * torch.ones_like(env.dummy1))


def test_chained_containing_chained_and_list(env):
    """Test multi-level nesting: a chained term whose children are chained and list terms."""
    cfg = {
        "outer": ManagerTermBaseCfg(
            func=chained_terms_class,
            params={
                "terms": {
                    "inner_chain": ManagerTermBaseCfg(
                        func=chained_terms_class,
                        params={
                            "terms": {
                                "add_1": ManagerTermBaseCfg(func=increment_dummy1_by_one),
                                "add_2": ManagerTermBaseCfg(func=change_dummy1_by_value, params={"value": 2}),
                            }
                        },
                    ),
                    "inner_list": ManagerTermBaseCfg(
                        func=list_terms_class,
                        params={
                            "term_list": [
                                ManagerTermBaseCfg(func=change_dummy1_by_value, params={"value": 10}),
                                ManagerTermBaseCfg(func=change_dummy1_by_value, params={"value": 20}),
                            ]
                        },
                    ),
                }
            },
        ),
    }
    manager = SimpleManager(cfg, env)

    # Outer term should be instantiated.
    outer_cfg = manager._term_cfgs[0][1]
    assert isinstance(outer_cfg.func, chained_terms_class)

    # Mid-level terms should also be instantiated class instances.
    inner_chain_cfg = outer_cfg.params["terms"]["inner_chain"]
    inner_list_cfg = outer_cfg.params["terms"]["inner_list"]
    assert isinstance(inner_chain_cfg.func, chained_terms_class)
    assert isinstance(inner_list_cfg.func, list_terms_class)

    # Leaf-level funcs inside the inner chain should be resolved callables.
    leaf_terms = inner_chain_cfg.params["terms"]
    assert leaf_terms["add_1"].func is increment_dummy1_by_one
    assert leaf_terms["add_2"].func is change_dummy1_by_value

    # Leaf-level funcs inside the inner list should be resolved callables.
    leaf_list = inner_list_cfg.params["term_list"]
    assert leaf_list[0].func is change_dummy1_by_value
    assert leaf_list[1].func is change_dummy1_by_value

    # Apply and verify: inner_chain adds (1 + 2) = 3, inner_list adds (10 + 20) = 30 -> total 33
    manager.apply(torch.arange(env.num_envs, device=env.device))
    torch.testing.assert_close(env.dummy1, 33 * torch.ones_like(env.dummy1))
