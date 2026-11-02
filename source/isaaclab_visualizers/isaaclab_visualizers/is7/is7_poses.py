# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Publish Isaac Lab body poses into the Isaac Sim 7 host's stage as a host physics participant.

The host orchestrates its stage: it allocates ordinals, seals them and recreates the stage when the user stops the
scene. An engine registered with ``isaacsim.physics.registration`` is the sanctioned writer: on each activation the
host lends it the stage with a reserved output ordinal. Isaac Lab's engine performs no physics. It writes the latest
poses that Isaac Lab handed to :class:`PoseSlot` as world matrices, plus the transform journal that lets the OVGL
renderer refresh only those bodies.
"""

from __future__ import annotations

import secrets
import threading
from collections.abc import Callable
from typing import Any

import numpy as np

_WORLD_MATRIX = "omni:fabric:worldMatrix"

# Transform journal read by the OVGL renderer (isaacsim/common/ovstage/TransformJournal.hpp in Isaac Sim 7).
_JOURNAL_PRIM = "/_IsaacSimTransformJournal"
_JOURNAL_ATTRIBUTE = "omni:isaac:transformJournal"
_JOURNAL_MAGIC = 0x4A4E525458465349
_JOURNAL_VERSION = 1
_JOURNAL_HEADER = np.dtype(
    [
        ("magic", "<u8"),
        ("version", "<u8"),
        ("generation", "<u8"),
        ("base_ordinal", "<u8"),
        ("ordinal", "<u8"),
        ("entry_count", "<u8"),
    ]
)
_JOURNAL_ENTRY = np.dtype([("prim_path", "<u8"), ("world_matrix", "<f8", (16,))])


class PoseSlot:
    """Latest body world matrices, handed from Isaac Lab's loop to the host's participant thread.

    Args:
        paths: Body prim paths, in the row order of every published array.
    """

    def __init__(self, paths: list[str]):
        self.paths = paths
        self._lock = threading.Lock()
        self._rows: np.ndarray | None = None
        self._version = 0
        self.wake: Callable[[], None] | None = None
        """Requests a host frame, which a stopped host renders only on request."""

    def publish(self, rows: np.ndarray) -> None:
        """Replace the latest poses.

        Args:
            rows: Row-major world matrices with shape ``[len(paths), 16]``; the slot keeps the array.
        """
        with self._lock:
            self._rows = rows
            self._version += 1
            wake = self.wake
        if wake is not None:
            wake()

    def latest(self) -> tuple[np.ndarray | None, int]:
        """Return the latest poses and their version."""
        with self._lock:
            return self._rows, self._version


class _PoseExecution:
    """Host engine execution that writes the slot's poses at the host's output ordinal without sealing.

    Args:
        slot: Poses handed over by Isaac Lab.
    """

    def __init__(self, slot: PoseSlot):
        self._slot = slot
        self._stage = None
        self._dictionary = None
        self._queries: list[tuple[Any, Any]] = []
        self._written_version = 0
        # The renderer tells journal chains apart by generation; use the high half like OVNewton.
        self._generation = (1 << 63) | secrets.randbits(63)
        self._base_ordinal = 0
        self._last_ordinal = 0

    def initialize(self, stage_pointer: int, stage_id: int, input_ordinal: int, output_ordinal: int) -> None:
        """Borrow the host's stage and publish the current poses."""
        import ovstage

        from isaacsim.usd.ovstage import borrow_stage

        self._stage = borrow_stage(stage_pointer)
        self._dictionary = ovstage.PathDictionary(self._stage)
        self._bodies = self._query(self._slot.paths)
        self._journal = self._query([_JOURNAL_PRIM])
        self._path_ids = np.array(self._dictionary.get_paths(self._bodies[0]), dtype=np.uint64)
        self._attribute = self._dictionary.intern_token(_WORLD_MATRIX)
        self._journal_attribute = self._dictionary.intern_token(_JOURNAL_ATTRIBUTE)
        self._publish(input_ordinal, output_ordinal, force=True)

    def _query(self, paths: list[str]) -> tuple[Any, Any]:
        path_list = self._dictionary.create_path_list_from_strings(paths)
        query = self._stage.query_from_path_list(path_list)
        self._queries.append((path_list, query))
        return path_list, query

    def execute(self, time_ns: int, elapsed_ns: int, input_ordinal: int, output_ordinal: int) -> bool:
        """Publish poses that Isaac Lab handed over since the last activation."""
        return self._publish(input_ordinal, output_ordinal)

    def _publish(self, input_ordinal: int, output_ordinal: int, force: bool = False) -> bool:
        import ovstage

        rows, version = self._slot.latest()
        if rows is None or (version == self._written_version and not force):
            return False
        # Consecutive publications extend one journal chain, so the renderer may skip frames; a gap starts a new one.
        continues = self._last_ordinal != 0 and output_ordinal == self._last_ordinal + 1
        base_ordinal = self._base_ordinal if continues else input_ordinal
        transforms = self._stage.write_attribute(
            self._bodies[1],
            self._attribute,
            output_ordinal,
            rows,
            is_array=False,
            semantic=ovstage.AttributeSemantic.MATRIX,
        )
        try:
            header = np.zeros(1, dtype=_JOURNAL_HEADER)
            header[0] = (_JOURNAL_MAGIC, _JOURNAL_VERSION, self._generation, base_ordinal, output_ordinal, len(rows))
            entries = np.zeros(len(rows), dtype=_JOURNAL_ENTRY)
            entries["prim_path"] = self._path_ids
            entries["world_matrix"] = rows
            payload = np.concatenate([header.view(np.uint8), entries.view(np.uint8)])
            self._stage.write_attribute(
                self._journal[1], self._journal_attribute, output_ordinal, [payload], is_array=True
            ).wait()
        finally:
            transforms.wait()
        self._written_version = version
        self._base_ordinal = base_ordinal
        self._last_ordinal = output_ordinal
        return True

    def close(self) -> None:
        """Release the queries before their path lists, then the borrowed stage."""
        try:
            for path_list, query in reversed(self._queries):
                self._stage.release_query(query).wait()
                self._dictionary.destroy_path_list(path_list)
            self._queries.clear()
        finally:
            if self._stage is not None:
                self._stage.destroy()
                self._stage = None


class PoseEngine:
    """Process registration of the Isaac Lab pose engine with the Isaac Sim 7 physics registry.

    Args:
        slot: Poses handed over by Isaac Lab.
    """

    def __init__(self, slot: PoseSlot):
        from isaacsim.physics.registration import Simulation, register_simulation

        self.name = f"isaaclab-poses-{secrets.token_hex(4)}"
        self._simulation = Simulation()
        self._simulation.execution_factory = lambda: _PoseExecution(slot)
        self._id = register_simulation(self._simulation, self.name)

    def participant(self, rate: Any) -> Any:
        """Create a host participant that runs this engine at ``rate``."""
        from isaacsim.physics.manager import PhysicsConfig, make_physics

        config = PhysicsConfig()
        config.engine = self.name
        config.rate = rate
        return make_physics(config)

    def close(self) -> None:
        """Unregister the engine."""
        from isaacsim.physics.registration import unregister_simulation

        unregister_simulation(self._id)
