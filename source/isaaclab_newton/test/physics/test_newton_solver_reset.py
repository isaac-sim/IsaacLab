# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for opt-in clearing of solver-owned state after authored writes."""

from __future__ import annotations

import pytest
from isaaclab_newton.physics import KaminoSolverCfg, NewtonCfg, NewtonKaminoManager, NewtonManager
from isaaclab_newton.physics import kamino_manager as kamino_manager_module

from isaaclab.physics import PhysicsManager


class _SolverRecorder:
    """Minimal solver that records model notifications and reset calls."""

    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events
        self.resets: list[tuple[object, object, int]] = []

    def notify_model_changed(self, change: int) -> None:
        self.events.append(f"notify:{self.name}:{change}")

    def reset(self, state: object, *, world_mask: object, flags: int) -> None:
        self.resets.append((state, world_mask, flags))
        self.events.append(f"reset:{self.name}")


class _Transaction:
    """Tiny transaction boundary that delegates to the manager policy."""

    def __init__(self, events: list[str], world_mask: object, fk_mask: object) -> None:
        self.events = events
        self.world_mask = world_mask
        self.fk_mask = fk_mask

    def flush(self) -> None:
        self.events.append("transaction")
        NewtonManager._apply_state_writes(self.world_mask, self.fk_mask)


def test_newton_cfg_solver_reset_is_strictly_opt_in():
    """The direct Newton configuration field defaults to disabled."""
    assert NewtonCfg().solver_reset is False
    assert NewtonCfg(solver_reset=True).solver_reset is True


@pytest.mark.parametrize("value", [None, 1])
def test_newton_cfg_rejects_non_boolean_solver_reset(value):
    """Invalid reset policy fails before solver initialization."""
    with pytest.raises(TypeError, match="solver_reset must be a bool"):
        NewtonCfg(solver_reset=value)


@pytest.mark.parametrize("enabled", [False, True])
def test_flush_orders_optional_reset_after_required_sync(monkeypatch, enabled):
    """Every owned solver sees model and authored state before optional clearing."""
    events: list[str] = []
    first = _SolverRecorder("first", events)
    second = _SolverRecorder("second", events)
    state = object()
    world_mask = object()
    fk_mask = object()

    monkeypatch.setattr(PhysicsManager, "_cfg", NewtonCfg(solver_reset=enabled), raising=False)
    monkeypatch.setattr(PhysicsManager, "_device", "cpu", raising=False)
    monkeypatch.setattr(NewtonManager, "_state_0", state, raising=False)
    monkeypatch.setattr(NewtonManager, "_model_changes", {4}, raising=False)
    monkeypatch.setattr(NewtonManager, "_state_writes", _Transaction(events, world_mask, fk_mask), raising=False)
    monkeypatch.setattr(NewtonManager, "_owned_solvers", classmethod(lambda cls: (first, second)))
    monkeypatch.setattr(
        NewtonManager,
        "_eval_fk_impl",
        classmethod(lambda cls, worlds, articulations: events.append("fk")),
    )
    monkeypatch.setattr(
        NewtonManager,
        "_synchronize_solver_state",
        classmethod(lambda cls, solver, worlds, articulations: events.append(f"sync:{solver.name}")),
    )

    NewtonManager._flush_pending_changes()

    expected = ["notify:first:4", "notify:second:4", "transaction", "fk", "sync:first"]
    if enabled:
        expected.append("reset:first")
    expected.append("sync:second")
    if enabled:
        expected.append("reset:second")
    assert events == expected
    assert NewtonManager._model_changes == set()

    for solver in (first, second):
        if enabled:
            assert solver.resets == [(state, world_mask, 0)]
        else:
            assert solver.resets == []


def test_kamino_mandatory_sync_is_not_reset_twice(monkeypatch):
    """Enabling the policy does not duplicate Kamino's required reset operation."""

    class _KaminoResetRecorder:
        class ResetConfig:
            @staticmethod
            def from_joints():
                return "from_joints"

            @staticmethod
            def preserve():
                return "preserve"

        def __init__(self) -> None:
            self.calls: list[tuple[object, object, object, object]] = []

        def reset(self, state, *, world_mask, flags=None, config=None):
            self.calls.append((state, world_mask, flags, config))

    solver = _KaminoResetRecorder()
    state = object()
    world_mask = object()
    monkeypatch.setattr(
        PhysicsManager,
        "_cfg",
        NewtonCfg(solver_cfg=KaminoSolverCfg(use_fk_solver=True), solver_reset=True),
        raising=False,
    )
    monkeypatch.setattr(kamino_manager_module, "SolverKamino", _KaminoResetRecorder)
    monkeypatch.setattr(NewtonKaminoManager, "_solver", solver, raising=False)
    monkeypatch.setattr(NewtonKaminoManager, "_state_0", state, raising=False)

    NewtonKaminoManager._apply_state_writes(world_mask, object())

    assert solver.calls == [(state, world_mask, None, "from_joints")]
