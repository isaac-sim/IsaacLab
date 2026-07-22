# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :meth:`EventManager.initialize_pre_physics_ready_terms`.

Lightweight (no Kit, no Sim) -- exercises the opt-in classmethod dispatch added for
pre-PHYSICS_READY USD-stage authoring. The hook is invoked from
:meth:`isaaclab.envs.ManagerBasedEnv._init_sim` before the renderer's ``prepare_stage`` fires,
to let opt-in event terms author USD-stage opinions that have to land before scene bake.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from isaaclab.managers.event_manager import EventManager


class _OptInTerm:
    """Stand-in for an event term class that opts into pre-PHYSICS_READY setup."""

    init_before_physics_ready: bool = True
    calls: list[tuple[object, object]] = []  # (cfg, env) for each invocation

    @classmethod
    def pre_physics_ready_setup(cls, cfg, env):
        cls.calls.append((cfg, env))


class _OptOutTerm:
    """Stand-in for an event term class that has NOT opted in. Should be skipped."""

    calls: list[tuple[object, object]] = []

    @classmethod
    def pre_physics_ready_setup(cls, cfg, env):
        cls.calls.append((cfg, env))  # would fail the test if invoked


class _OptInWithoutSetup:
    """Opted in but lacks the classmethod -- must be tolerated (silently skipped)."""

    init_before_physics_ready: bool = True


@pytest.fixture(autouse=True)
def _reset_recordings():
    _OptInTerm.calls.clear()
    _OptOutTerm.calls.clear()


def _make_cfg(**terms):
    """Build a configclass-shaped cfg whose attributes are EventTermCfg-shaped namespaces."""
    return SimpleNamespace(**{name: SimpleNamespace(func=cls, params={}) for name, cls in terms.items()})


def test_dispatches_only_opt_in_terms():
    cfg = _make_cfg(a=_OptInTerm, b=_OptOutTerm)
    env = object()
    EventManager.initialize_pre_physics_ready_terms(cfg, env)
    assert len(_OptInTerm.calls) == 1
    assert _OptInTerm.calls[0][1] is env
    assert _OptOutTerm.calls == []


def test_does_not_mutate_term_cfg_func():
    """The hook must leave ``term_cfg.func`` as a class so the subsequent
    ``EventManager.__init__`` ``copy.deepcopy(cfg)`` still works (instances would hold env refs
    pointing at unpickleable ``pxr.Usd.Stage``)."""
    cfg = _make_cfg(a=_OptInTerm)
    EventManager.initialize_pre_physics_ready_terms(cfg, object())
    assert cfg.a.func is _OptInTerm  # unchanged


def test_opt_in_without_classmethod_is_safe():
    """A class that opts in but defines no ``pre_physics_ready_setup`` is silently skipped --
    no AttributeError, no double-init, future-friendly for terms that only need the opt-in flag
    to participate in some other lifecycle phase later."""
    cfg = _make_cfg(a=_OptInWithoutSetup)
    EventManager.initialize_pre_physics_ready_terms(cfg, object())  # no exception


def test_accepts_dict_cfg():
    """Some envs pass the events cfg as a plain dict instead of a configclass."""
    cfg = {"a": SimpleNamespace(func=_OptInTerm, params={})}
    EventManager.initialize_pre_physics_ready_terms(cfg, object())
    assert len(_OptInTerm.calls) == 1


def test_empty_cfg_is_noop():
    EventManager.initialize_pre_physics_ready_terms(_make_cfg(), object())
    assert _OptInTerm.calls == []
