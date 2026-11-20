# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One OVStage shared by Isaac Lab and the Isaac Sim 7 viewer, with Isaac Lab owning its ordinals.

The Isaac Sim 7 host creates and registers the stage, so its stage tree and property panes read it through the
host's own stage context. Isaac Lab borrows the same native stage: it writes simulation state and the host's edits at
the ordinal it has open, seals that ordinal, renders the viewer's render products with OVRTX at it, and hands the
frames to the host.
"""

from __future__ import annotations

import ctypes
import itertools
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple

import numpy as np


class Replication(NamedTuple):
    """Environment copies and placement that complete an exported scene in the stage."""

    copies: Sequence[tuple[str, Sequence[str]]]
    """Source subtrees and their destinations, parents before children."""

    env_paths: Sequence[str]
    """Environment roots, in placement order."""

    positions: np.ndarray | None
    """Environment origins [m] with shape ``[len(env_paths), 3]``, or ``None`` to keep authored placement."""


_MATRIX_ATTRIBUTES = frozenset({"omni:xform", "omni:fabric:worldMatrix", "omni:fabric:localMatrix", "worldMatrix"})


class SharedStage:
    """Isaac Lab's side of the shared stage: queries, writes at the open ordinal, and sealing.

    Args:
        usd: USD text that the Isaac Sim 7 stage context populates the stage from.
        replication: Environment copies and placement applied once populated.
    """

    def __init__(self, usd: str, replication: Replication | None = None):
        import ovstage
        from isaaclab_ov.stage import create_ovstage

        from isaacsim.context.usd import Backend, make_usd_context
        from isaacsim.usd.ovstage import borrow_stage

        # ovstage fixes its hierarchy model per process at the first stage; claim Isaac Lab's GPU-incremental model
        # before the host creates the shared stage, so OVRTX can read world transforms on the device.
        self._anchor = create_ovstage("isaaclab.isaac_sim.anchor")
        self.context = make_usd_context(Backend.OVSTAGE)
        started = time.perf_counter()
        self.context.create_from_string(usd)
        self.startup = {"populate": time.perf_counter() - started}
        """Wall time [s] of population and cloning, for benchmarking."""
        self.stage = borrow_stage(self.context.stage_pointer)
        self._dictionary = ovstage.PathDictionary(self.stage)
        self._queries: dict[tuple[str, ...], tuple[Any, Any]] = {}
        self.sealed = self.stage.fetch_ordinal(self.stage.get_attribute_write_floor())
        """Latest sealed ordinal; population sealed the first."""
        self.open = self.sealed + 1
        """Ordinal that writes go to until :meth:`seal`."""
        if replication is not None:
            from isaaclab_ov.stage import ovstage_replicate

            started = time.perf_counter()
            ovstage_replicate(
                self.stage,
                self._dictionary,
                replication.copies,
                replication.env_paths,
                replication.positions,
                ordinal=self.open,
            )
            self.seal()
            self.startup["clone"] = time.perf_counter() - started

    def _query(self, paths: Sequence[str]) -> Any:
        key = tuple(paths)
        entry = self._queries.get(key)
        if entry is None:
            path_list = self._dictionary.create_path_list_from_strings(list(key))
            entry = self._queries[key] = (path_list, self.stage.query_from_path_list(path_list))
        return entry[1]

    def write(self, paths: Sequence[str], attribute: str, values: Any) -> None:
        """Write one value per path at the open ordinal."""
        import ovstage

        array = np.asarray(values)
        if array.dtype.kind in "USO":
            # Text values, such as ``visibility``, are stored as interned tokens.
            tokens = [self._dictionary.intern_token(str(value)) for value in array.reshape(-1)]
            array = np.asarray(tokens, dtype=np.uint64)
            kwargs = {"semantic": ovstage.AttributeSemantic.TOKEN_ID}
            self.stage.write_attribute(self._query(paths), attribute, self.open, array, is_array=False, **kwargs).wait()
            return
        if attribute in _MATRIX_ATTRIBUTES:
            array = np.ascontiguousarray(array, dtype=np.float64).reshape(len(paths), 16)
            semantic = ovstage.AttributeSemantic.MATRIX
        else:
            array = np.ascontiguousarray(array).reshape(len(paths), *array.shape[1:]) if array.ndim else array
            semantic = None
        kwargs = {} if semantic is None else {"semantic": semantic}
        self.stage.write_attribute(self._query(paths), attribute, self.open, array, is_array=False, **kwargs).wait()

    def write_device(self, paths: Sequence[str], attribute: str, tensor: Any, cuda_stream: int) -> None:
        """Write a device-resident matrix tensor with one ``[16]`` row per path at the open ordinal."""
        import ovstage

        self.stage.write_attribute(
            self._query(paths),
            attribute,
            self.open,
            tensor,
            is_array=False,
            semantic=ovstage.AttributeSemantic.MATRIX,
            cuda_stream=cuda_stream,
        ).wait()

    def clone(self, source: str, targets: Sequence[str]) -> None:
        """Clone a subtree at the open ordinal."""
        self.stage.clone(source, list(targets), self.open)

    def remove(self, paths: Sequence[str]) -> None:
        """Delete prim subtrees at the open ordinal."""
        self.stage.delete_attributes(self._query(paths), [], self.open).wait()

    def seal(self) -> int:
        """Seal the open ordinal so readers and renderers see it; return it."""
        self.stage.advance_write_floor(self.open).wait()
        self.sealed = self.open
        self.open += 1
        return self.sealed

    def close(self) -> None:
        """Release the queries and the borrowed view, then the host's stage context."""
        try:
            for path_list, query in self._queries.values():
                self.stage.release_query(query).wait()
                self._dictionary.destroy_path_list(path_list)
            self._queries.clear()
            self.stage.destroy()
        finally:
            self.context = None


class FrameStore:
    """Latest rendered frame per render product, in the shape the host's renderer outputs read."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frames: dict[str, Any] = {}
        self._ids = itertools.count(1)

    def publish(self, product: str, pixels: np.ndarray, time_ns: int, ordinal: int) -> None:
        frame = SimpleNamespace(
            pixels=pixels,
            memory="cpu",
            frame_id=next(self._ids),
            time_ns=time_ns,
            source="isaaclab-ovrtx",
            generation=0,
            input_ordinal=ordinal,
            output_ordinal=ordinal,
            capture_id=None,
            timeline="participant",
            transforms=[],
            camera=None,
            depth=None,
        )
        with self._lock:
            self._frames[product] = frame

    def latest(self, product: str) -> Any:
        with self._lock:
            return self._frames.get(product)


class ViewerRenderer:
    """OVRTX renderer for the viewer's render products, reading the shared stage at Isaac Lab's sealed ordinals.

    Args:
        stage: The shared stage.
        device: CUDA device ordinal to render on.
    """

    def __init__(self, stage: SharedStage, device: int = 0):
        import ovrtx
        import ovstage

        # OVRTX resolves this native dependency of ovstage through the process, as Isaac Lab's renderer does.
        dependency = Path(ovstage.__file__).parent / "bin/plugins/libosdCPU.so.3.6.0"
        if dependency.exists():
            ctypes.CDLL(str(dependency))
        self._ovrtx = ovrtx
        self._renderer = ovrtx.Renderer(
            ovrtx.RendererConfig(
                active_cuda_gpus=str(device),
                keep_system_alive=True,
                read_gpu_transforms=True,
                suppress_deprecation_warnings=True,
                texture_streaming_mode=ovrtx.TextureStreamingMode.SYNCHRONOUS,
            )
        )
        self._renderer.attach_ovstage(stage.stage)
        self._lock = threading.Lock()
        self._products: dict[str, bool] = {}
        self._pending: tuple | None = None
        self.timings = {"step": 0.0, "readback": 0.0, "renders": 0}
        """Accumulated wall time [s] of OVRTX submission and pixel readback on Isaac Lab's thread, for benchmarking."""

    def add_product(self, product: str) -> None:
        with self._lock:
            self._products.setdefault(product, True)

    def remove_product(self, product: str) -> None:
        with self._lock:
            self._products.pop(product, None)

    def set_product_enabled(self, product: str, enabled: bool) -> None:
        with self._lock:
            if product in self._products:
                self._products[product] = enabled

    def poll(self, frames: FrameStore) -> tuple[int, bool]:
        """Publish the in-flight render if it finished, without waiting.

        OVRTX synchronizes with the stage when a queued render starts and refuses unsealed changes then, so nothing
        may be written to the stage while a render is in flight.

        Returns:
            How many products were published, and whether a render is still in flight.
        """
        published = self._collect(frames, block=False)
        return published, self._pending is not None

    def render(self, ordinal: int, time_ns: int) -> None:
        """Start rendering the enabled products at a sealed ordinal; :meth:`poll` publishes the frames."""
        if self._pending is not None:
            raise RuntimeError("a render is already in flight")
        with self._lock:
            products = {product for product, enabled in self._products.items() if enabled}
        if products:
            start = time.perf_counter()
            operation = self._renderer.step_async(render_products=products, delta_time=1.0 / 60.0, ordinal=ordinal)
            self._pending = (operation, products, time_ns, ordinal)
            self.timings["step"] += time.perf_counter() - start

    def _collect(self, frames: FrameStore, *, block: bool) -> int:
        """Publish the in-flight render if it finished (or wait for it with ``block``); return products published."""
        import warp as wp

        if self._pending is None:
            return 0
        operation, products, time_ns, ordinal = self._pending
        pending = operation.wait(timeout_ns=None if block else 0)
        if pending is None:
            return 0
        self._pending = None
        start = time.perf_counter()
        result = pending.fetch()
        published = 0
        for product in products:
            output = result[product]
            if not output.frames:
                continue
            var = next(iter(output.frames[0].render_vars.values()))
            mapping = var.map(device=self._ovrtx.Device.CUDA, sync_stream=0)
            try:
                mapping.wait()
                pixels = wp.from_dlpack(mapping).numpy()
            finally:
                mapping.unmap(stream=1)
            frames.publish(product, pixels, time_ns, ordinal)
            published += 1
        self.timings["readback"] += time.perf_counter() - start
        self.timings["renders"] += 1
        return published

    def close(self) -> None:
        """Finish the in-flight render, then detach from the stage."""
        try:
            if self._pending is not None:
                self._pending[0].wait()
                self._pending = None
        finally:
            self._renderer.detach_ovstage()
