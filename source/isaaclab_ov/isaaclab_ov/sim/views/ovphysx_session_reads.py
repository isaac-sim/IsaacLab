# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Captured ovphysx session reads of the GPU simulation state, behind :meth:`OvPhysxView.read_into`.

On a CUDA scene, :class:`OvPhysxSessionReads` serves the tensor types in :data:`READS`: simulation state that ovphysx
keeps on the GPU. The first read of a (view, tensor type, destination) holds an ovphysx query over the binding's prims,
maps the session's rows onto the binding's layout, and captures "session read + placement kernels" into one CUDA
graph; every later read replays it on the caller's current Warp stream without waiting on the host (ovphysx
ADR-0055). The graph writes the destination in place, so a caller passes a buffer it keeps, as the asset data classes
do. Every other type, and every read of a CPU scene, stays on the binding: ovphysx cannot capture its host-resident
properties or contacts, and an uncaptured session read costs more per call than the binding read.

Values and row order are the binding's, except that joint values, which the session reports in degrees on angular
axes, may differ in the last bit, and deformable positions, which the session reports in the simulation mesh's frame,
carry float32 rounding from that transform.

The module is internal to :mod:`isaaclab_ov`: :class:`OvPhysxBackend` owns one instance per CUDA ``PhysX`` instance and
:class:`OvPhysxView` routes to it; nothing else calls it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import ovstage
import warp as wp
from ovphysx.types import SimObjectType, TensorType

_POSE = ("position", "orientation")
_VELOCITY = ("linearVelocity", "angularVelocity")
_ACCELERATION = ("linearAcceleration", "angularAcceleration")

READS: dict[TensorType, tuple[SimObjectType, tuple[str, ...]]] = {
    TensorType.ARTICULATION_ROOT_POSE: (SimObjectType.ARTICULATION, ("rootPosition", "rootOrientation")),
    TensorType.ARTICULATION_ROOT_VELOCITY: (SimObjectType.ARTICULATION, ("rootLinearVelocity", "rootAngularVelocity")),
    TensorType.ARTICULATION_JACOBIAN: (SimObjectType.ARTICULATION, ("jacobian",)),
    TensorType.ARTICULATION_MASS_MATRIX: (SimObjectType.ARTICULATION, ("massMatrix",)),
    TensorType.ARTICULATION_LINK_POSE: (SimObjectType.ARTICULATION_LINK, _POSE),
    TensorType.ARTICULATION_LINK_VELOCITY: (SimObjectType.ARTICULATION_LINK, _VELOCITY),
    TensorType.ARTICULATION_LINK_ACCELERATION: (SimObjectType.ARTICULATION_LINK, _ACCELERATION),
    TensorType.ARTICULATION_LINK_INCOMING_JOINT_FORCE: (SimObjectType.ARTICULATION_LINK, ("linkIncomingJointForce",)),
    TensorType.ARTICULATION_DOF_POSITION: (SimObjectType.ARTICULATION_JOINT, ("jointPosition",)),
    TensorType.ARTICULATION_DOF_VELOCITY: (SimObjectType.ARTICULATION_JOINT, ("jointVelocity",)),
    TensorType.RIGID_BODY_POSE: (SimObjectType.RIGID_BODY, _POSE),
    TensorType.RIGID_BODY_VELOCITY: (SimObjectType.RIGID_BODY, _VELOCITY),
    TensorType.RIGID_BODY_ACCELERATION: (SimObjectType.RIGID_BODY, _ACCELERATION),
    TensorType.DEFORMABLE_SIM_NODAL_POSITION: (SimObjectType.DEFORMABLE_VOLUME, ("points",)),
    TensorType.DEFORMABLE_SIM_NODAL_VELOCITY: (SimObjectType.DEFORMABLE_VOLUME, ("velocities",)),
    TensorType.SURFACE_DEFORMABLE_SIM_POSITION: (SimObjectType.DEFORMABLE_SURFACE, ("points",)),
    TensorType.SURFACE_DEFORMABLE_SIM_VELOCITY: (SimObjectType.DEFORMABLE_SURFACE, ("velocities",)),
}
"""Tensor types read through captured session reads: the session object type, and the attributes whose lanes,
concatenated, form one binding row. The object type sets how session rows map onto binding rows: one per prim (one per
link of each articulation for links), one per joint axis, or one per node of a deformable body."""

_KINEMATIC_READS = frozenset((TensorType.ARTICULATION_JACOBIAN, TensorType.ARTICULATION_MASS_MATRIX))
"""Types computed from the link poses, which a joint-state write leaves stale until a kinematic refresh. Their
binding reads refresh the kinematics themselves; the session reads do not, so they refresh first."""

_DEFORMABLES = (SimObjectType.DEFORMABLE_VOLUME, SimObjectType.DEFORMABLE_SURFACE)

_ANGULAR_JOINT_TYPES = (2, 3, 4)
"""``dofJointType`` values whose axis the session reports in degrees: revolute, unwrapped revolute, spherical."""


@wp.kernel
def _place(
    src: wp.array(dtype=wp.float32),
    dst_rows: wp.array(dtype=wp.int32),
    scales: wp.array(dtype=wp.float32),
    lanes: wp.int32,
    offset: wp.int32,
    width: wp.int32,
    dst: wp.array(dtype=wp.float32),
):
    """``dst[dst_rows[i], offset + k] = src[i, k]``, times ``scales[i]`` when scales are given."""
    i, k = wp.tid()
    value = src[i * lanes + k]
    if scales.shape[0] > 0:
        value = value * scales[i]
    dst[dst_rows[i] * width + offset + k] = value


@wp.kernel
def _place_points(
    src: wp.array(dtype=wp.float32),
    dst_rows: wp.array(dtype=wp.int32),
    bodies: wp.array(dtype=wp.int32),
    frames: wp.array(dtype=wp.mat44),
    dst: wp.array(dtype=wp.float32),
):
    """Place one deformable node per thread, taking its sim-mesh-local point to the world frame."""
    i = wp.tid()
    p = wp.transform_point(frames[bodies[i]], wp.vec3(src[i * 3], src[i * 3 + 1], src[i * 3 + 2]))
    d = dst_rows[i] * 3
    dst[d] = p[0]
    dst[d + 1] = p[1]
    dst[d + 2] = p[2]


@wp.kernel
def _clear(dst: wp.array(dtype=wp.float32)):
    """Zero the destination, so the binding rows the session does not hold read zero, as the binding pads them."""
    dst[wp.tid()] = 0.0


@dataclass(eq=False)
class _Source:
    """One held query of a plan, and where each of its session rows lands in the destination."""

    query: Any
    attributes: list[tuple[str, int]] = field(default_factory=list)
    """Session attribute and lanes of each column, in binding lane order."""
    dst_rows: wp.array | None = None
    """Binding row of each session row."""
    scales: wp.array | None = None
    """Factor per session row (degrees to radians on an angular joint axis); empty when none applies."""
    bodies: wp.array | None = None
    """Body of each session row, indexing ``frames``."""
    frames: wp.array | None = None
    """World transform (column-vector 4x4) of each body, for sim-mesh-local deformable points; ``None`` otherwise."""
    read: Any = None
    """The read the graph replays; ovphysx binds it to the graph, so both stay open together."""


@dataclass
class _Plan:
    """The graph that refreshes one destination, and the sources it replays."""

    sources: list[_Source] = field(default_factory=list)
    graph: Any = None
    padded: bool = False
    """Whether the binding holds rows no session row fills (shorter articulations); the graph zeroes them first."""

    def close(self) -> None:
        """Release the graph, then every read, then every query, as ovphysx requires; raise the first failure."""
        # 1. Drop the graph, then list the reads and queries it bound, in release order.
        self.graph = None
        releases = [source.read.close for source in self.sources if source.read is not None]
        releases += [source.query.close for source in self.sources]
        # 2. Release each, keeping on after a failure.
        failures = []
        for release in releases:
            try:
                release().wait()
            except Exception as failure:  # keep releasing the rest
                failures.append(failure)
        if failures:
            raise failures[0]


def _read_groups(result: Any) -> list[Any]:
    """Every group of an ovphysx read that carries data; an empty selection reads none."""
    groups = list(result.groups or [])
    if not groups:
        while (group := result.fetch_next().wait()) is not None:
            groups.append(group)
    return [group for group in groups if group.tensors]


def _row_paths(result: Any, groups: list[Any]) -> list[str]:
    """Prim path of each session row of a read."""
    first = groups[0]
    handles = result.dictionary.get_paths(first.prim_list)
    return [result.dictionary.path_to_string(handles[first.prim_offset + i]) for i in range(first.prim_count)]


def _columns(result: Any) -> dict[str, wp.array]:
    """Each attribute's column as one flat ``float32`` array; an array group's per-prim tensors joined into one."""
    columns = {}
    for group in _read_groups(result):
        # 1. The group's per-prim tensors must form one block; an empty prim carries no memory.
        tensors = [tensor for tensor in group.tensors if tensor.size] or group.tensors[:1]
        for before, after in zip(tensors, tensors[1:]):
            if after.ptr != before.ptr + before.size * wp.types.type_size_in_bytes(before.dtype):
                raise ValueError("The session read returns an array group whose tensors are not contiguous.")
        # 2. One flat column per attribute.
        name = result.dictionary.token_to_string(group.attribute)
        if name in columns:  # ovphysx splits articulations of different structure into one group each
            raise ValueError(f"The session read returns {name!r} in several groups; one layout is required.")
        size = sum(tensor.size for tensor in tensors)
        columns[name] = wp.array(ptr=tensors[0].ptr, dtype=wp.float32, shape=(size,), device=tensors[0].device)
    return columns


def _dst_rows(slot: dict[Any, int], keys: list[Any]) -> list[int]:
    """Binding row of each session row, by its key; a key no binding row holds is an error."""
    missing = [key for key in keys if key not in slot]
    if missing:
        raise ValueError(f"The session read has rows {missing[:3]} that no binding row holds ({len(missing)} missing).")
    return [slot[key] for key in keys]


class OvPhysxSessionReads:
    """Captured ovphysx session reads of one CUDA ``PhysX`` instance, owned by its :class:`OvPhysxBackend`.

    Each (view, tensor type, destination) holds one plan: the queries it reads and the graph that refreshes the
    destination from them, replayed on the caller's current Warp stream.
    """

    _by_physx: ClassVar[dict[int, OvPhysxSessionReads]] = {}

    def __init__(self, physx: Any, stage: Any, device: str):
        self._physx = physx
        self._stage = stage
        self._device = wp.get_device(device)
        self._stream = wp.Stream(self._device)
        """The stream plans are captured on, owned here; a replay runs on the caller's stream."""
        self._cuda_stream = int(self._stream.cuda_stream)
        """The capture stream's handle, for ovphysx's ``cuda_stream``."""
        self._plans: dict[tuple[int, TensorType, int], _Plan] = {}
        self._pending: dict[int, wp.Event] = {}
        """The last replay on each stream that replayed, by stream handle."""
        self.refresh_kinematics: Callable[[], None] | None = None
        """Refreshes stale articulation kinematics on :attr:`stream`; set by the physics manager."""
        self.world_transforms: Callable[[list[str]], np.ndarray] | None = None
        """World transform of each prim on the stage ovphysx loaded, as column-vector 4x4 matrices [m]; set by the
        physics manager. Deformable ``points`` reads take their sim meshes' frames from it."""
        OvPhysxSessionReads._by_physx[id(physx)] = self

    @classmethod
    def for_physx(cls, physx: Any) -> OvPhysxSessionReads | None:
        """Return the session reads of ``physx``, or ``None`` on a CPU scene, which reads its bindings."""
        return cls._by_physx.get(id(physx))

    @property
    def stream(self) -> wp.Stream:
        """The caller's current stream on the simulation device: replays and kinematic refreshes run on it.

        Its ``cuda_stream`` is ``None`` for Warp's null stream, the legacy default stream ovphysx takes as ``0``.
        """
        return wp.get_stream(self._device)

    def read_into(self, view: Any, tensor_type: TensorType, binding: Any, dst: wp.array) -> bool:
        """Replay the captured read of ``tensor_type`` into the flat binding-layout ``dst``.

        Returns:
            Whether the session served the read; ``False`` for a type outside :data:`READS`, which the binding reads.

        Raises:
            ValueError: If the session's rows cannot be laid out over the binding's.
            RuntimeError: If called while the device is capturing a CUDA graph.
        """
        # 1. Types and destinations the session does not replay.
        if tensor_type not in READS:
            return False
        if dst.size == 0:  # e.g. the joints of an articulation without DOFs: nothing to read
            return True
        if self._device.is_capturing:
            raise RuntimeError(f"Cannot read {tensor_type.name} while {self._device} is capturing a CUDA graph.")
        # 2. Capture the plan on the first read of this view, type and destination.
        key = (id(view), tensor_type, dst.ptr)
        if key not in self._plans:
            self._plans[key] = self._capture(tensor_type, binding, dst)
        plan = self._plans[key]
        # 3. Replay it on the caller's stream, after the kinematic refresh it depends on.
        stream = self.stream
        if tensor_type in _KINEMATIC_READS and self.refresh_kinematics is not None:
            self.refresh_kinematics()
        wp.capture_launch(plan.graph, stream=stream)
        self._pending[stream.cuda_stream] = stream.record_event(self._pending.get(stream.cuda_stream))
        return True

    def wait(self) -> None:
        """Block until every launched replay has finished: ovphysx forbids stepping while one runs."""
        for event in self._pending.values():
            wp.synchronize_event(event)
        self._pending.clear()

    def release(self, view: Any) -> None:
        """Release the graphs, reads and queries ``view`` captured."""
        self.wait()
        self._close([self._plans.pop(key) for key in [key for key in self._plans if key[0] == id(view)]])

    def invalidate(self) -> None:
        """Wait for every replay and release every capture, so each read captures again.

        Called before a structural edit ovphysx does not report, which a replay in flight or a later replay of a graph
        captured before it would read through stale tables.
        """
        plans = list(self._plans.values())
        self._plans.clear()
        try:
            self.wait()
        finally:
            self._close(plans)

    def close(self) -> None:
        """Release every capture before the stage detaches."""
        OvPhysxSessionReads._by_physx.pop(id(self._physx), None)
        self.invalidate()

    @staticmethod
    def _close(plans: list[_Plan]) -> None:
        """Close every plan, then raise the first failure."""
        failures = []
        for plan in plans:
            try:
                plan.close()
            except Exception as failure:
                failures.append(failure)
        if failures:
            raise failures[0]

    def _capture(self, tensor_type: TensorType, binding: Any, dst: wp.array) -> _Plan:
        """Map the session's rows onto ``dst``, then capture every source's read and placement kernels in one graph."""
        object_type, names = READS[tensor_type]
        plan = _Plan()
        try:
            # 1. Hold the sources' queries and map each session row onto its binding row.
            with ovstage.PathDictionary(self._stage) as paths:
                if object_type == SimObjectType.ARTICULATION_JOINT:
                    rows = self._dof_sources(binding, names, paths, plan)
                elif object_type in _DEFORMABLES:
                    rows = self._node_sources(object_type, binding, names, paths, plan)
                else:
                    rows = self._prim_sources(object_type, binding, names, paths, plan)
            # 2. The rows and lanes cover the destination exactly.
            width = sum(lanes for _, lanes in plan.sources[0].attributes)
            if rows * width != dst.size:
                raise ValueError(f"{tensor_type.name}: {rows} rows of {width} lanes miss {dst.size}.")
            # 3. Capture every source's read and placement kernels into one graph.
            flat = wp.array(ptr=dst.ptr, dtype=wp.float32, shape=(dst.size,), device=dst.device)
            self._physx.wait_all()  # no deferred release may run inside the capture
            with wp.ScopedCapture(device=self._device, stream=self._stream) as capture:
                if plan.padded:
                    wp.launch(_clear, dim=dst.size, outputs=[flat], device=self._device, stream=self._stream)
                for source in plan.sources:
                    names_read = [name for name, _ in source.attributes]
                    source.read = self._physx.read(source.query, names_read, cuda_stream=self._cuda_stream).wait()
                    self._launch(source, _columns(source.read), width, flat)
            plan.graph = capture.graph
            return plan
        except BaseException:
            plan.close()
            raise

    def _prim_sources(self, object_type: SimObjectType, binding: Any, names: tuple, paths: Any, plan: _Plan) -> int:
        """Sources of a read with one binding row per prim, or per link of each articulation for a link read.

        A rigid-body binding's articulation links read as links, through a second query. Returns the binding's rows.
        """
        # 1. The binding row of each prim: its own, or each articulation's links in link order, padded to the longest.
        if object_type == SimObjectType.ARTICULATION_LINK:
            table = self._articulation_table(binding, ("linkPath", "linkCount"), paths)
            links, counts = binding.shape[1], table["linkCount"][:, 0]
            if (counts > links).any():
                raise ValueError(f"The articulations hold {counts} links; the binding holds {links}.")
            slot = {
                paths.path_to_string(ovstage.PrimPath(int(table["linkPath"][index, link]))): index * links + link
                for index, count in enumerate(counts)
                for link in range(int(count))
            }
            total = len(counts) * links
        else:
            slot = {prim: row for row, prim in enumerate(binding.prim_paths)}
            total = len(slot)
        plan.padded = plan.padded or len(slot) < total
        rows = list(slot)
        # 2. Read each object type the rows span and place each session row on its binding row.
        parts = (SimObjectType.RIGID_BODY, SimObjectType.ARTICULATION_LINK)
        placed = 0
        for part_type in parts if object_type == SimObjectType.RIGID_BODY else (object_type,):
            opened = self._open(part_type, rows, names, paths, plan)
            if opened is not None:  # a selection keeps only the prims of its object type
                source, session_rows, _ = opened
                self._fill(source, _dst_rows(slot, session_rows))
                placed += len(session_rows)
        # 3. Every row the session holds is placed.
        if placed != len(rows):
            raise ValueError(f"The session reads {placed} of the binding's {len(rows)} rows.")
        return total

    def _dof_sources(self, binding: Any, names: tuple, paths: Any, plan: _Plan) -> int:
        """The source of a joint read: one session row per joint axis, placed in its DOF slot. Returns the slots."""
        # 1. The DOF slot of each (joint, axis) the binding holds, and whether the axis is angular.
        table = self._articulation_table(binding, ("dofJointPath", "dofAxisIndex", "dofCount", "dofJointType"), paths)
        width = binding.shape[1]
        slot, angular = {}, {}
        for index, used in enumerate(table["dofCount"][:, 0]):
            if used > width:
                raise ValueError(f"{binding.prim_paths[index]} has {used} DOFs; the binding holds {width}.")
            for dof in range(int(used)):
                key = (
                    paths.path_to_string(ovstage.PrimPath(int(table["dofJointPath"][index, dof]))),
                    int(table["dofAxisIndex"][index, dof]),
                )
                slot[key] = index * width + dof
                angular[key] = int(table["dofJointType"][index, dof]) in _ANGULAR_JOINT_TYPES
        plan.padded = plan.padded or len(slot) < math.prod(binding.shape[:2])
        # 2. Read the joints; a joint's k-th session row is its axis k.
        joints = list(dict.fromkeys(joint for joint, _ in slot))
        opened = self._open(SimObjectType.ARTICULATION_JOINT, joints, names, paths, plan)
        session_rows = opened[1] if opened is not None else []
        keys, ordinal = [], {}
        for joint in session_rows:  # the k-th row of a joint's run is its axis ordinal k
            keys.append((joint, ordinal.get(joint, 0)))
            ordinal[joint] = keys[-1][1] + 1
        if len(keys) != len(slot):
            raise ValueError(f"The session read has {len(keys)} joint rows; the binding holds {len(slot)}.")
        # 3. Place each row in its slot, angular axes from degrees to radians.
        dst_rows = _dst_rows(slot, keys)
        self._fill(opened[0], dst_rows, scales=[math.radians(1.0) if angular[key] else 1.0 for key in keys])
        return math.prod(binding.shape[:2])

    def _node_sources(
        self,
        object_type: SimObjectType,
        binding: Any,
        names: tuple,
        paths: Any,
        plan: _Plan,
    ) -> int:
        """The source of a deformable read: one session row per node, padded to the binding's widest body.

        ovphysx reports ``points`` in the sim-mesh frame and ``velocities`` in world; the points pass through each sim
        mesh's world transform on the stage ovphysx loaded (:attr:`world_transforms`). Returns the binding's node rows.

        Raises:
            RuntimeError: If no :attr:`world_transforms` is set for a ``points`` read.
        """
        # 1. Read the deformable prim of each binding body.
        prims = self._deformable_prims(object_type, names[0], list(binding.prim_paths))
        opened = self._open(object_type, prims, names, paths, plan)
        session_rows, counts = (opened[1], opened[2]) if opened is not None else ([], [])
        nodes = binding.shape[1]
        if sorted(session_rows) != sorted(prims) or max(counts) > nodes:
            raise ValueError(
                f"The session reads {list(zip(session_rows, counts))[:3]}; the binding holds {nodes} nodes."
            )
        # 2. Node i of body b lands on binding row b * nodes + i.
        body_of = {prim: body for body, prim in enumerate(prims)}
        bodies = np.repeat([body_of[prim] for prim in session_rows], counts)
        dst_rows = bodies * nodes + np.concatenate([np.arange(count) for count in counts])
        # 3. Points also take their body's sim-mesh frame to world.
        if names[0] == "points":
            if self.world_transforms is None:
                raise RuntimeError("Deformable points need the sim meshes' world transforms; none were set.")
            self._fill(opened[0], dst_rows, bodies=bodies, frames=self.world_transforms(prims))
        else:
            self._fill(opened[0], dst_rows)
        return math.prod(binding.shape[:2])

    def _open(
        self, object_type: SimObjectType, prims: list[str], names: tuple, paths: Any, plan: _Plan
    ) -> tuple[_Source, list[str], list[int]] | None:
        """Hold a query over ``prims`` and read it once uncaptured: the source, its session rows, and rows per prim.

        The read plans the query and allocates its device columns, which the captured read reuses (ADR-0055). Each
        attribute's lanes come from it. Returns ``None`` when the session reads none of ``prims`` as ``object_type``.
        """
        # 1. Hold the query; the plan owns it from here.
        source = _Source(self._physx.query(object_type, prim_paths=[paths.intern_path(p) for p in prims]).wait())
        plan.sources.append(source)
        # 2. Read it once: the groups give each attribute's lanes and the session rows.
        warm = self._physx.read(source.query, list(names), cuda_stream=self._cuda_stream).wait()
        try:
            groups = _read_groups(warm)
            if not groups:
                plan.sources.remove(source)
                return None
            if any(group.tensors[0].device != self._device for group in groups):
                raise ValueError(f"{object_type.name} columns are not on {self._device}; they cannot be captured.")
            lanes = {}
            for group in groups:
                tensor = next((t for t in group.tensors if t.size), group.tensors[0])
                lanes[warm.dictionary.token_to_string(group.attribute)] = math.prod(tensor.shape[1:])
            source.attributes = [(name, lanes[name]) for name in names]
            counts = [int(tensor.shape[0]) for tensor in groups[0].tensors]
            return source, _row_paths(warm, groups), counts
        finally:
            # 3. Release the warm read, and the query when the session read none of the prims.
            warm.close().wait()
            if source not in plan.sources:  # read none: release the query now
                source.query.close().wait()

    def _fill(self, source: _Source, dst_rows: Any, scales: Any = None, bodies: Any = None, frames: Any = None):
        """Upload where each of the source's session rows lands, on the simulation device."""
        device = self._device
        source.dst_rows = wp.array(np.asarray(dst_rows, dtype=np.int32), dtype=wp.int32, device=device)
        source.scales = wp.array(np.asarray(scales if scales is not None else [], dtype=np.float32), device=device)
        if frames is not None:
            source.bodies = wp.array(np.asarray(bodies, dtype=np.int32), dtype=wp.int32, device=device)
            source.frames = wp.array(frames, dtype=wp.mat44, device=device)

    def _launch(self, source: _Source, columns: dict[str, wp.array], width: int, dst: wp.array) -> None:
        """Place each of the source's columns in ``dst`` on the capture stream."""
        count, offset = source.dst_rows.shape[0], 0
        for name, lanes in source.attributes:
            # Points pass through their body's frame; other columns are copied lane by lane, scaled per row.
            if source.frames is not None:
                inputs = [columns[name], source.dst_rows, source.bodies, source.frames]
                dim = count
                kernel = _place_points
            else:
                inputs = [columns[name], source.dst_rows, source.scales, lanes, offset, width]
                dim = (count, lanes)
                kernel = _place
            wp.launch(kernel, dim=dim, inputs=inputs, outputs=[dst], device=self._device, stream=self._stream)
            offset += lanes

    def _deformable_prims(self, object_type: SimObjectType, attribute: str, prims: list[str]) -> list[str]:
        """The deformable prim the session reads for each binding prim: the prim itself, or the one below it.

        A mesh-spawned body binds its outer prim, while the session reads the deformable mesh prim under it.
        """
        # 1. Every prim the session reads as this deformable type.
        with self._physx.read_sync(object_type, [attribute]) as result:
            session = [path for group in _read_groups(result) for path in _row_paths(result, [group])]
        # 2. Match each binding prim to itself or to the one deformable prim below it.
        known = set(session)
        found = []
        for prim in prims:
            below = [prim] if prim in known else [path for path in session if path.startswith(prim + "/")]
            if len(below) != 1:
                raise ValueError(f"{prim} has {len(below)} {object_type.name} prims; one is required.")
            found.append(below[0])
        return found

    def _articulation_table(self, binding: Any, names: tuple, paths: Any) -> dict[str, np.ndarray]:
        """Each of the articulation metadata columns ``names``, one row per binding articulation, in binding order."""
        roots = list(binding.prim_paths)
        handles = [paths.intern_path(p) for p in roots]
        with self._physx.read_sync(SimObjectType.ARTICULATION, list(names), prim_paths=handles) as result:
            groups = _read_groups(result)
            row_of = {path: row for row, path in enumerate(_row_paths(result, groups))}
            order = [row_of[root] for root in roots]
            # Host columns are views of the read's memory: index (copy) them before the read closes.
            return {
                result.dictionary.token_to_string(group.attribute): group.tensors[0]
                .numpy()
                .reshape(len(row_of), -1)[order]
                for group in groups
            }
