# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU memory and events shared between the camera and a model service on one machine, through Warp's CUDA IPC.

The two processes use different PyTorch builds, so they share GPU memory through Warp, which works beside either:
each process allocates the buffer it writes and the interprocess event it records on its own GPU and exports their
handles; the other process opens the buffer on that GPU and the event on its own GPU, whose stream waits for it.
Buffers appear in both processes as zero-copy uint8 tensors, and events order the producer's and consumer's CUDA
streams without synchronizing the host. The processes can use different GPUs that access each other's memory
directly, over NVLink or PCIe; the reader then copies between the GPUs without host memory. Warp supports CUDA IPC
on Linux; elsewhere :func:`available` returns False and images go through the socket.
"""

from __future__ import annotations

import base64
import math
import sys
from contextlib import ExitStack
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

_HANDLE_BYTES = 64


def _warp():
    import warp as wp

    wp.init()
    return wp


def available() -> bool:
    """Return whether this process can share GPU memory with another process through CUDA IPC."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        wp = _warp()
    except Exception:
        # Warp is missing, or it found no CUDA driver.
        return False
    return any(device.is_ipc_supported is not False for device in wp.get_cuda_devices())


def gpu_uuid(device_index: int) -> str:
    """Return the UUID of a CUDA device of this process, which names the same GPU in every process on any machine."""
    return _warp().get_device(f"cuda:{device_index}").uuid


def visible_gpus() -> list[str]:
    """Return the UUIDs of the CUDA devices this process can see.

    A camera whose GPU is not among a service's visible GPUs runs on another machine, or on a GPU the service cannot
    open buffers on.
    """
    return [device.uuid for device in _warp().get_cuda_devices()]


def device_by_uuid(uuid: str | None) -> int | None:
    """Return this process's index of the CUDA device with UUID ``uuid``, or None if it cannot see it."""
    if not isinstance(uuid, str):
        return None
    return next((device.ordinal for device in _warp().get_cuda_devices() if device.uuid == uuid), None)


def can_access_peer(index: int, peer_index: int) -> bool:
    """Return whether two CUDA devices of this process can read each other's memory directly, over NVLink or PCIe.

    A device is its own peer. Between devices without peer access, CUDA would stage copies through host memory.
    """
    if index == peer_index:
        return True
    wp = _warp()
    first, second = f"cuda:{index}", f"cuda:{peer_index}"
    return wp.is_peer_access_supported(first, second) and wp.is_peer_access_supported(second, first)


def _encode(handle: bytes) -> str:
    return base64.b64encode(handle).decode("ascii")


def _decode(text: str) -> bytes:
    raw = base64.b64decode(text)
    if len(raw) != _HANDLE_BYTES:
        raise ValueError("A CUDA IPC handle must contain 64 bytes.")
    return raw


class SharedBuffer:
    """A uint8 GPU buffer allocated here and opened by another process, or opened from a handle."""

    def __init__(self, device: torch.device, shape: tuple[int, ...], handle: str | None = None):
        import torch

        wp = _warp()
        self.device = torch.device(device)
        self.shape = tuple(shape)
        warp_device = wp.get_device(f"cuda:{self.device.index or 0}")
        # Warp arrays have at most four dimensions, so the buffer is flat and the tensor gives it its shape.
        size = math.prod(self.shape)
        if handle is None:
            # CUDA IPC shares only memory from the default allocator, not from Warp's memory pool.
            with wp.ScopedMempool(warp_device, False):
                self._array = wp.zeros(size, dtype=wp.uint8, device=warp_device)
            self.handle = _encode(self._array.ipc_handle())
        else:
            self._array = wp.from_ipc_handle(_decode(handle), wp.uint8, (size,), device=warp_device)
            self.handle = handle
        self.tensor = wp.to_torch(self._array).view(self.shape)

    def close(self) -> None:
        """Free the memory, or close this process's mapping of it. Repeated calls are safe.

        Warp frees or unmaps the buffer when its last reference goes, so callers must not keep views of
        :attr:`tensor`.
        """
        self.tensor = None
        self._array = None


class SharedEvent:
    """An interprocess CUDA event created here and opened by another process, or opened from a handle."""

    def __init__(self, device: torch.device, handle: str | None = None):
        import torch

        wp = _warp()
        self.device = torch.device(device)
        warp_device = wp.get_device(f"cuda:{self.device.index or 0}")
        if handle is None:
            self._event = wp.Event(warp_device, interprocess=True)
            self.handle = _encode(self._event.ipc_handle())
        else:
            self._event = wp.event_from_ipc_handle(_decode(handle), warp_device)
            self.handle = handle

    def record(self) -> None:
        """Mark the current point of this device's current PyTorch stream."""
        self._stream().record_event(self._event)

    def wait(self) -> None:
        """Make this device's current PyTorch stream wait for the last recorded point, without blocking the host."""
        self._stream().wait_event(self._event)

    def close(self) -> None:
        """Destroy this process's event. Repeated calls are safe."""
        self._event = None

    def _stream(self):
        import torch

        return _warp().stream_from_torch(torch.cuda.current_stream(self.device))


_KEPT: list[ExitStack] = []
"""Buffers and events kept until this process exits because the other process may still have them open."""


class SharedChannel:
    """The buffers and events of one camera session: controls in, generated images out.

    Each process creates the buffer it writes and the event it records on its own GPU: the camera creates
    ``control`` and ``control_ready``, the service ``output`` and ``output_ready``. Neither process writes memory the
    other can free. CUDA leaves freeing memory that another process still has open undefined, so each process
    releases the other's buffer first and frees its own only after the other confirms it released it. CUDA records
    an event only on the device that created it, so this split also lets the two processes use different GPUs. Each
    process maps the other's buffer on the GPU that holds it and imports the other's event on its own GPU, whose
    stream waits for it.
    """

    def __init__(self, device: torch.device, shape: tuple[int, ...], writes: str):
        """Create the buffer and event this process writes; :meth:`open` adds the other process's.

        Args:
            device: CUDA device of this process that holds them.
            shape: Largest chunk of every view, ``(views, frames, height, width, 3)``.
            writes: ``"control"`` for the camera, ``"output"`` for the service.

        Raises:
            ValueError: If ``writes`` names neither buffer.
        """
        if writes not in ("control", "output"):
            raise ValueError("A CUDA IPC channel writes either control or output.")
        self._writes = writes
        self._device = device
        self._shape = tuple(shape)
        self.control = self.control_ready = self.output = self.output_ready = None
        self._opened = ExitStack()
        with ExitStack() as created:
            buffer = SharedBuffer(device, self._shape)
            created.callback(buffer.close)
            event = SharedEvent(device)
            created.callback(event.close)
            self._created = created.pop_all()
        self._own = (buffer, event)
        self._assign(writes, buffer, event)

    @property
    def handles(self) -> dict[str, str]:
        """Handles of the buffer and event this process created, for the other process to open."""
        buffer, event = self._own
        return {self._writes: buffer.handle, f"{self._writes}_ready": event.handle}

    def open(self, device: torch.device, handles: dict) -> None:
        """Map the other process's buffer on ``device`` and import its event on this channel's GPU.

        ``device`` is this process's index of the GPU that holds the buffer. The event is imported on the GPU whose
        stream waits for it: a stream waiting on an event imported on another GPU crashes the CUDA driver.

        Raises:
            ValueError: If the handles do not name the other process's buffer and event.
        """
        reads = "output" if self._writes == "control" else "control"
        if (
            not isinstance(handles, dict)
            or set(handles) != {reads, f"{reads}_ready"}
            or not all(isinstance(value, str) for value in handles.values())
        ):
            raise ValueError(f"CUDA IPC needs the other process's {reads} and {reads}_ready handles.")
        with ExitStack() as opened:
            buffer = SharedBuffer(device, self._shape, handles[reads])
            opened.callback(buffer.close)
            event = SharedEvent(self._device, handles[f"{reads}_ready"])
            opened.callback(event.close)
            self._opened = opened.pop_all()
        self._assign(reads, buffer, event)

    def close_opened(self) -> None:
        """Release the other process's buffer and event before that process frees them. Repeated calls are safe."""
        self._opened.close()
        reads = "output" if self._writes == "control" else "control"
        self._assign(reads, None, None)

    def close(self, *, peer_released: bool = True) -> None:
        """Release the other process's buffer and event, then free this process's own. Repeated calls are safe.

        Args:
            peer_released: Whether the other process confirmed it released this process's buffer and event, or
                never received their handles. Otherwise they stay allocated until this process exits.
        """
        try:
            self.close_opened()
        finally:
            if peer_released:
                self._created.close()
            else:
                _KEPT.append(self._created.pop_all())
            self._own = (None, None)
            self._assign(self._writes, None, None)

    def _assign(self, name: str, buffer: SharedBuffer | None, event: SharedEvent | None) -> None:
        if name == "control":
            self.control, self.control_ready = buffer, event
        else:
            self.output, self.output_ready = buffer, event
