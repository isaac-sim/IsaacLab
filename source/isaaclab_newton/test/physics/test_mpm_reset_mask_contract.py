# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Isaac Lab's use of Newton's shared reset-mask contract."""

from types import SimpleNamespace

import numpy as np
import pytest
from isaaclab_newton.physics import NewtonManager, NewtonMPMManager


class _Mask:
    """Minimal host-readable stand-in for a Warp boolean array."""

    def __init__(self, values: list[bool]):
        self._values = np.asarray(values, dtype=np.bool_)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._values.shape

    def numpy(self) -> np.ndarray:
        return self._values.copy()


def _install_solver(monkeypatch, *, world_count: int = 2):
    """Install a fake initialized implicit-MPM manager and return its reset calls."""
    calls = []
    solver = SimpleNamespace(reset=lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(NewtonManager, "_solver", solver)
    monkeypatch.setattr(NewtonManager, "_model", SimpleNamespace(world_count=world_count))
    monkeypatch.setattr(NewtonMPMManager, "_implicit_mpm_solvers", classmethod(lambda cls: (solver,)))
    return calls


def test_mpm_manager_explicit_reset_accepts_canonical_mask(monkeypatch):
    """The explicit reset forwards Newton's canonical mask unchanged."""
    calls = _install_solver(monkeypatch)
    state = object()
    world_mask = _Mask([False, True, False])

    NewtonMPMManager.reset_solver_state(state=state, world_mask=world_mask, flags=0)

    assert calls == [((state,), {"world_mask": world_mask, "flags": 0})]


def test_mpm_manager_explicit_reset_rejects_local_only_mask(monkeypatch):
    """The explicit reset requires the final global-world entry."""
    _install_solver(monkeypatch)

    with pytest.raises(ValueError, match=r"world_mask must have shape \(3,\); got \(2,\)"):
        NewtonMPMManager.reset_solver_state(state=object(), world_mask=_Mask([False, True]), flags=0)


def test_mpm_manager_explicit_reset_promotes_selected_single_world(monkeypatch):
    """A selected one-world MPM grid resets all solver-owned history."""
    calls = _install_solver(monkeypatch, world_count=1)
    state = object()

    NewtonMPMManager.reset_solver_state(state=state, world_mask=_Mask([True, False]), flags=0)

    assert calls == [((state,), {"world_mask": None, "flags": 0})]


def test_mpm_manager_explicit_reset_deduplicates_manager_states(monkeypatch):
    """An implicit MPM reset visits each distinct manager state once."""
    calls = _install_solver(monkeypatch)
    state_0 = object()
    state_1 = object()
    monkeypatch.setattr(NewtonManager, "_state_0", state_0)
    monkeypatch.setattr(NewtonManager, "_state_1", state_1)
    world_mask = _Mask([True, False, False])

    NewtonMPMManager.reset_solver_state(world_mask=world_mask, flags=17)

    assert calls == [
        ((state_1,), {"world_mask": world_mask, "flags": 17}),
        ((state_0,), {"world_mask": world_mask, "flags": 17}),
    ]

    calls.clear()
    monkeypatch.setattr(NewtonManager, "_state_1", state_0)

    NewtonMPMManager.reset_solver_state(world_mask=world_mask, flags=17)

    assert calls == [((state_0,), {"world_mask": world_mask, "flags": 17})]
