# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal pieces the Isaac Sim 7 viewer host needs from a world that Isaac Lab drives.

The host's panes read and write the world through a simulation handle and show time from a clock. Here neither runs a
scheduler: :class:`HostData` reads the shared stage and writes at Isaac Lab's open ordinal, :class:`HostClock` reports
Isaac Lab's simulation time, and Isaac Lab drains the host's command queues itself once per frame through a
:class:`StageView`.
"""

from __future__ import annotations

from collections.abc import Sequence
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


def _as_list(paths: Any) -> list[str]:
    return [paths] if isinstance(paths, str) else list(paths)


class HostData:
    """The simulation handle the host's services use: reads of the shared stage, writes at the owner's ordinal.

    Args:
        context: The host's stage context, used for sealed reads (an ``isaacsim.context.usd.UsdContext``).
        writer: Writes at the owner's open ordinal.
    """

    def __init__(self, context: Any, writer: OrdinalWriter):
        self.context = context
        self.writer = writer
        self.time_ns = 0
        """Owner simulation time [ns], set by Isaac Lab each frame."""
        self.failure: str | None = None

    @property
    def current_time(self) -> int:
        return self.time_ns

    def read(self, paths: Any, attribute: str, time: int | None = None) -> Any:
        from isaacsim.common.exceptions import StatusCode, StatusError

        try:
            return self.context.read(_query(paths), attribute)
        except StatusError as error:
            # The stage context reports an unauthored attribute as "Attribute not found"; the host's services expect
            # the orchestration's wording for an absent value.
            if error.code == StatusCode.NOT_FOUND and "Attribute not found" in str(error):
                raise StatusError(f"No value for attribute {attribute!r}", error.code, error.details) from error
            raise

    def read_with_timestamps(self, paths: Any, attribute: str, time: int | None = None) -> tuple:
        """Read values with the owner's latest time for each path; the stage keeps no per-path publication times."""
        values = self.read(paths, attribute)
        return values, [self.time_ns] * len(_as_list(paths))

    def write(self, paths: Any, attribute: str, values: Any, time: int | None = None) -> None:
        self.writer.write(_as_list(paths), attribute, values)

    def fail(self, message: str) -> None:
        self.failure = self.failure or message

    # The host starts, steps and stops a scheduler through these; Isaac Lab drives this world instead.

    def step(self, duration: int) -> None:
        pass

    def suspend_participant(self, identity: str) -> None:
        pass

    def resume_participant(self, identity: str) -> None:
        pass

    def attach_participant(self, identity: str, participant: Any = None, **_: Any) -> None:
        pass

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


class HostClock:
    """Display clock for the host: reports Isaac Lab's simulation time and never schedules anything."""

    identity = "isaaclab-clock"
    epoch = 0
    failure = None

    def __init__(self, data: HostData):
        self._data = data

    @property
    def sim_time_ns(self) -> int:
        return self._data.time_ns

    @property
    def next_index(self) -> int:
        return 0

    def next_owner_time(self) -> int:
        return self._data.time_ns

    def prepare_resume(self, owner_time: int) -> None:
        pass

    def confirm_step(self, index: int, epoch: int) -> None:
        pass

    def join(self, timeout_s: float = 30.0) -> bool:
        return True


class StageView:
    """The view the host's queued commands run against when Isaac Lab drains them.

    Args:
        data: The world's data handle.
    """

    epoch = 0

    def __init__(self, data: HostData):
        self._data = data

    @property
    def current_time(self) -> int:
        return self._data.time_ns

    def read(self, paths: Any, attribute: str, time: int | None = None) -> Any:
        return self._data.read(paths, attribute)

    def write(self, paths: Any, attribute: str, values: Any, time: int | None = None) -> None:
        self._data.write(paths, attribute, values)

    def clone_prim(self, source: str, targets: Sequence[str]) -> None:
        self._data.writer.clone(source, list(targets))

    def remove_prim(self, paths: Any) -> None:
        self._data.writer.remove(_as_list(paths))

    def __getattr__(self, name: str) -> Any:
        # Other stage queries (is_a, metadata, variants) read the sealed state through the stage context.
        return getattr(self._data.context, name)


class CloneAuthoring:
    """Authoring session that describes cloned prims by their source prim.

    Isaac Lab clones environments in the shared stage only, so the host's authored stage holds just the source of each
    clone. Without this, a cloned prim's property fields have no declared types. Clones are described read-only: an
    authored edit would land on the source prim.

    Args:
        session: The host's ``isaacsim.viewports.authoring.AuthoringSession``.
        copies: Source subtrees and their clone destinations.
    """

    def __init__(self, session: Any, copies: Sequence[tuple[str, Sequence[str]]]):
        self._session = session
        self._sources = {target: source for source, targets in copies for target in targets}

    def source_of(self, path: str) -> str | None:
        """Return the authored prim a cloned path was copied from, or ``None`` if it is not a clone."""
        prefix, suffix = path, ""
        while prefix and prefix != "/":
            source = self._sources.get(prefix)
            if source is not None:
                return source + suffix
            prefix, name = prefix.rsplit("/", 1)
            suffix = f"/{name}{suffix}"
        return None

    def attributes(self, path: str) -> dict[str, dict[str, Any]]:
        """Describe a prim's declared attributes, using the clone's source prim for cloned paths."""
        # Clones are looked up first: the authored stage may keep a deactivated copy at a clone's path.
        source = self.source_of(path)
        if source is None:
            return self._session.attributes(path)
        reason = f"Cloned from {source}; edit the source prim"
        return {
            name: {**row, "writable": False, "write_reason": reason}
            for name, row in self._session.attributes(source).items()
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)


def live_inspection_type() -> type:
    """Return a property inspection that reads live world poses from the shared stage.

    Isaac Sim's inspection composes poses with native helpers that need a native simulation view, and reads velocities
    through Isaac Sim physics views. Neither exists when Isaac Lab drives the world, so this one reads the stage's world
    matrices, which Isaac Lab keeps current, through the Python view it is drained with. Velocities keep their authored
    values.
    """
    import numpy as np

    from isaacsim.application.host import PropertyInspection

    class LiveInspection(PropertyInspection):
        def _capture_selection(self, view: Any, demand: Any) -> dict[str, Any]:
            from pxr import Gf

            objects = {}
            for path in demand.paths:
                item: dict[str, Any] = {"values": {}, "pose": None, "transform": None}
                try:
                    matrix = np.asarray(view.read([path], "omni:fabric:worldMatrix"), dtype=float).reshape(4, 4)
                except Exception:  # noqa: BLE001 - prims without a transform have no pose
                    objects[path] = item
                    continue
                transform = Gf.Transform(Gf.Matrix4d(*matrix.reshape(-1).tolist()))
                rotation = transform.GetRotation()
                quat = rotation.GetQuat()
                angles = rotation.Decompose(Gf.Vec3d.ZAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.XAxis())
                item["pose"] = {
                    "position": matrix[3, :3].tolist(),
                    "orientation": [*quat.GetImaginary(), quat.GetReal()],
                }
                item["transform"] = {
                    "matrix": matrix.reshape(-1).tolist(),
                    "position": list(transform.GetTranslation()),
                    "rotation": list(reversed(angles)),
                    "scale": list(transform.GetScale()),
                }
                objects[path] = item
            return {"state": "ready", "time_ns": str(view.current_time), "objects": objects}

    return LiveInspection


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
