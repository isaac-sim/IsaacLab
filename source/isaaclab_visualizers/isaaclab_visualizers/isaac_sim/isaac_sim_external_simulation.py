# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Sim 7 host orchestration over a stage whose ordinals another owner controls.

The Isaac Sim 7 host normally runs its world under ``isaacsim.foundation.simulation.Simulation``, which takes exclusive
use of the stage, allocates ordinals and seals them. :class:`ExternalSimulation` offers the part of that interface the
host uses, without owning the stage: the stage's owner (Isaac Lab) runs the host's participants once per frame at the
ordinal it has open, seals it, and then acknowledges the publication. Native participants, such as the host's
renderers, are not supported; the owner renders instead.

This module only depends on Isaac Sim 7's Python API and is written so it could move into the host unchanged.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


class OrdinalWriter(Protocol):
    """Writes into the stage at the ordinal its owner has open."""

    def write(self, paths: Sequence[str], attribute: str, values: Any) -> None:
        """Write one value per path."""

    def clone(self, source: str, targets: Sequence[str]) -> None:
        """Clone a prim subtree to new paths."""

    def remove(self, paths: Sequence[str]) -> None:
        """Remove prim subtrees."""


def _query(paths: Any) -> Any:
    from isaacsim.common.interfaces import Query

    if isinstance(paths, Query):
        return paths
    return Query(paths if isinstance(paths, str) else list(paths))


class ActivationView:
    """Simulation view lent to the host's participants during one owner-driven activation.

    Reads go to the stage's sealed state, writes to the owner's open ordinal.

    Args:
        simulation: Simulation that runs the activation.
    """

    def __init__(self, simulation: ExternalSimulation):
        self._simulation = simulation

    @property
    def current_time(self) -> int:
        """Activation time [ns]."""
        return self._simulation.current_time

    @property
    def epoch(self) -> int:
        """Timeline epoch, constant for an owner-driven simulation."""
        return 0

    def read(self, paths: Any, attribute: str, time: int | None = None) -> Any:
        """Read sealed values of the selected prims."""
        return self._simulation.context.read(_query(paths), attribute)

    def write(self, paths: Any, attribute: str, values: Any, time: int | None = None) -> None:
        """Write values at the owner's open ordinal."""
        self._simulation.writer.write([paths] if isinstance(paths, str) else list(paths), attribute, values)

    def clone_prim(self, source: str, targets: Sequence[str]) -> None:
        """Clone ``source`` to ``targets`` at the owner's open ordinal."""
        self._simulation.writer.clone(source, list(targets))

    def remove_prim(self, paths: Any) -> None:
        """Remove prims at the owner's open ordinal."""
        self._simulation.writer.remove([paths] if isinstance(paths, str) else list(paths))

    def __getattr__(self, name: str) -> Any:
        # Other stage queries (is_a, metadata, variants) read the sealed state through the host's stage context.
        return getattr(self._simulation.context, name)


class _ClockView:
    """View for a participant that runs its own worker, such as the host's simulation clock.

    ``step()`` blocks until the owner publishes an activation and returns ``False`` once the simulation stops.
    """

    def __init__(self, simulation: ExternalSimulation):
        self._simulation = simulation
        self._published = 0
        self._consumed = 0
        self._condition = threading.Condition()
        self._stopped = False

    @property
    def current_time(self) -> int:
        return self._simulation.current_time

    @property
    def epoch(self) -> int:
        return 0

    def step(self) -> bool:
        with self._condition:
            self._condition.wait_for(lambda: self._stopped or self._published > self._consumed)
            if self._stopped:
                return False
            self._consumed = self._published
            return True

    def publish(self) -> None:
        with self._condition:
            self._published += 1
            self._condition.notify_all()

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()


@dataclass
class _Attached:
    identity: str
    position: int
    participant: Any = None
    suspended: bool = False
    initialized: bool = False
    view: _ClockView | None = None
    errors: list[BaseException] = field(default_factory=list)


class ExternalSimulation:
    """Host orchestration stand-in whose activations its stage's owner runs.

    Activations land on the host's frame period, starting at zero, as the host's participants require; the owner's
    own clock does not need to match it.

    Args:
        context: The host's stage context, used for sealed reads (an ``isaacsim.context.usd.UsdContext``).
        writer: Writes at the owner's open ordinal.
        rate: Host frame rate (an ``isaacsim.foundation.orchestration.RationalRate``).
    """

    def __init__(self, context: Any, writer: OrdinalWriter, rate: Any):
        from isaacsim.foundation.orchestration import Period

        self.context = context
        self.writer = writer
        self._period = Period(rate)
        self._index = 0
        self._lock = threading.RLock()
        self._attached: dict[str, _Attached] = {}
        self._time_ns = 0
        self._stopped = False
        self._closed = False
        self.failure: str | None = None
        self._view = ActivationView(self)

    # Lifecycle used by the host's world.

    def create(self) -> None:
        """Nothing to prepare: the owner created and populated the stage."""

    def attach_participant(self, identity: str, participant: Any = None, *, position: int = 0, **_: Any) -> None:
        """Register a host participant, run in position order on each owner activation."""
        with self._lock:
            if identity in self._attached:
                raise ValueError(f"participant already attached: {identity}")
            self._attached[identity] = _Attached(identity, position, participant)

    def make_simulation_view(self, context: Any) -> _ClockView:
        """Lend a self-driven participant a view whose ``step()`` follows the owner's publications."""
        with self._lock:
            attached = self._attached[context.identity]
            attached.view = _ClockView(self)
            return attached.view

    def suspend_participant(self, identity: str) -> None:
        with self._lock:
            self._attached[identity].suspended = True

    def resume_participant(self, identity: str) -> None:
        with self._lock:
            self._attached[identity].suspended = False

    @property
    def current_time(self) -> int:
        """Owner time of the latest activation [ns]."""
        return self._time_ns

    def step(self, duration: int) -> None:
        """Host-requested steps do nothing: the owner decides when the stage advances."""

    def stop(self) -> None:
        """Stop running activations and release self-driven participants."""
        with self._lock:
            self._stopped = True
            attached = list(self._attached.values())
        for item in attached:
            if item.view is not None:
                item.view.stop()
            elif item.initialized and hasattr(item.participant, "on_stopped"):
                item.participant.on_stopped()

    def close(self) -> None:
        """Close the participants; the stage stays with its owner."""
        if not self._stopped:
            self.stop()
        with self._lock:
            if self._closed:
                return
            self._closed = True
            attached = list(self._attached.values())
        for item in attached:
            if item.participant is not None and hasattr(item.participant, "close"):
                item.participant.close()

    def fail(self, message: str) -> None:
        """Record a host failure; later activations are skipped."""
        self.failure = self.failure or message

    # Data access used by host services outside activations.

    def read(self, paths: Any, attribute: str, time: int | None = None) -> Any:
        from isaacsim.common.exceptions import StatusCode, StatusError

        try:
            return self.context.read(_query(paths), attribute)
        except StatusError as error:
            # The stage context reports an unauthored attribute as "Attribute not found"; the host's orchestration,
            # whose wording the host's services recognize as an absent value, reports "No value for attribute".
            if error.code == StatusCode.NOT_FOUND and "Attribute not found" in str(error):
                raise StatusError(f"No value for attribute {attribute!r}", error.code, error.details) from error
            raise

    def read_with_timestamps(self, paths: Any, attribute: str, time: int | None = None) -> tuple:
        """Read sealed values with the time of the latest activation for each path.

        The stage does not record per-path publication times; the owner's latest activation bounds them.
        """
        values = self.read(paths, attribute)
        count = 1 if isinstance(paths, str) else len(list(paths))
        return values, [self._time_ns] * count

    def write(self, paths: Any, attribute: str, values: Any, time: int | None = None) -> None:
        self._view.write(paths, attribute, values)

    # Owner-driven activations.

    def activate(self) -> None:
        """Run every resumed participant once at the owner's open ordinal, in position order, on the next frame."""
        from isaacsim.foundation.orchestration import ParticipantActivation

        with self._lock:
            if self._stopped or self.failure is not None:
                return
            time_ns = self._period.time_at(self._index)
            elapsed_ns = time_ns - self._time_ns
            self._index += 1
            self._time_ns = time_ns
            ordered = sorted(self._attached.values(), key=lambda item: item.position)
            for item in ordered:
                if item.participant is None or item.suspended:
                    continue
                if not item.initialized:
                    item.participant.initialize(self._view)
                    item.initialized = True
                try:
                    item.participant.execute(ParticipantActivation(time_ns, elapsed_ns, 0))
                except Exception as error:  # noqa: BLE001 - reported like a failed participant
                    self.fail(f"{item.identity}: {type(error).__name__}: {error}")
                    raise

    def published(self) -> None:
        """Acknowledge that the owner sealed the activation's writes."""
        with self._lock:
            attached = list(self._attached.values())
        for item in attached:
            if item.view is not None:
                item.view.publish()
            elif item.initialized and hasattr(item.participant, "on_published"):
                item.participant.on_published()


class BorrowedContext:
    """Stage context whose stage outlives the host's world: ``close()`` leaves it open.

    Args:
        context: The owner's ``isaacsim.context.usd.UsdContext``.
    """

    def __init__(self, context: Any):
        self._context = context

    def close(self) -> None:
        """Keep the stage open for its owner."""

    def __getattr__(self, name: str) -> Any:
        return getattr(self._context, name)
