# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resident optional Cosmos service, independent of simulation and model framework imports."""

from __future__ import annotations

import logging
import os
import socket
import stat
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from typing import Protocol

import numpy as np

from .._protocol import (
    DEFAULT_ENDPOINT,
    MAX_ARRAYS,
    MAX_CHUNK_FRAMES,
    ProtocolError,
    disable_nagle,
    parse_endpoint,
    receive_message,
    send_message,
)

_LOGGER = logging.getLogger(__name__)


def serve(model: _Model | Callable[[], _Model], endpoint: str = DEFAULT_ENDPOINT, *, warmup: bool = False) -> None:
    """Reserve the endpoint, then serve a resident model until shutdown or interruption.

    A connection owns at most one generation session. Only one session may use the resident model
    at a time; separate status and shutdown connections remain available during generation.
    Disconnects and errors close session state while keeping model weights resident.
    Model operations always run on one persistent inference thread, including reconnects, because
    compiled CUDA graph state is thread-local.

    Args:
        model: Loaded model resource, or a factory called only after the endpoint is reserved. The worker uses
            a factory so a busy or invalid endpoint fails before model loading and warmup.
        endpoint: ``unix:///path`` for a Unix socket only this user can open, or ``tcp://host:port``. Use a
            loopback host for local TCP operation.
        warmup: Warm the model on its inference thread before exposing the ready endpoint.

    The service owns ``model`` and closes it on exit, including failed startup. The protocol has no
    authentication: remote connections should use a trusted tunnel rather than a public interface.
    """
    socket_path = None
    state = None
    resource = None if callable(model) else model
    connections: set[socket.socket] = set()
    threads: list[threading.Thread] = []
    connections_lock = threading.Lock()
    try:
        family, address = parse_endpoint(endpoint)
        with socket.socket(family, socket.SOCK_STREAM) as listener:
            if family == socket.AF_INET:
                if os.name == "nt":
                    listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                else:
                    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind(address)
                endpoint = f"tcp://{address[0]}:{listener.getsockname()[1]}"
            else:
                _bind_private_unix_socket(listener, address)
                socket_path = address
            listener.listen(16)
            listener.settimeout(0.25)
            _LOGGER.info(
                "Cosmos reserved %s; loading the model. Wait for the ready message before connecting.", endpoint
            )
            if resource is None:
                resource = model()
            state = _ServiceState(resource)
            state.transports, state.pci_bus_id = state.executor.submit(_transports, resource).result()
            if warmup:
                _LOGGER.info("Cosmos warming up; the first compilation can take several minutes.")
                state.executor.submit(resource.warmup).result()
            _LOGGER.info("Cosmos ready at %s", endpoint)
            while not state.stopping.is_set():
                try:
                    connection, _ = listener.accept()
                except TimeoutError:
                    continue
                connection.settimeout(600.0)
                if family == socket.AF_INET:
                    disable_nagle(connection)
                with connections_lock:
                    connections.add(connection)
                thread = threading.Thread(
                    target=_handle_connection,
                    args=(connection, state, connections, connections_lock),
                    name="cosmos-client",
                    daemon=True,
                )
                # Keep only live handlers so repeated status requests do not accumulate threads.
                threads = [thread for thread in threads if thread.is_alive()]
                threads.append(thread)
                thread.start()
    finally:
        try:
            if state is not None:
                state.stopping.set()
                with connections_lock:
                    for connection in connections:
                        with suppress(OSError):
                            connection.shutdown(socket.SHUT_RDWR)
                # In-flight inference and session cleanup complete before the model is released.
                for thread in threads:
                    thread.join()
                state.executor.submit(resource.close).result()
            elif resource is not None:
                resource.close()
        finally:
            if state is not None:
                state.executor.shutdown(wait=True)
            if socket_path is not None:
                with suppress(OSError):
                    os.unlink(socket_path)


def _bind_private_unix_socket(listener: socket.socket, path: str) -> None:
    """Bind a Unix socket only this user can open, replacing the socket file of a stopped service."""
    with suppress(FileNotFoundError):
        if not stat.S_ISSOCK(os.lstat(path).st_mode):
            raise RuntimeError(f"{path} exists and is not a socket; choose another Cosmos endpoint.")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            probe.settimeout(1.0)
            try:
                probe.connect(path)
            except OSError:
                os.unlink(path)
            else:
                raise RuntimeError(f"A Cosmos service is already running at unix://{path}.")
    # The process umask applies to the socket file; owner-only from the start leaves no window for other users.
    previous = os.umask(0o177)
    try:
        listener.bind(path)
    finally:
        os.umask(previous)


class _Stream(Protocol):
    def step(
        self, controls: list[np.ndarray], reset_rows: tuple[int, ...], seeds: tuple[int, ...], **episode: object
    ) -> list[np.ndarray]: ...

    def close(self) -> None: ...


class _Model(Protocol):
    capabilities: dict

    def warmup(self) -> None: ...

    def open_stream(
        self,
        *,
        num_views: int,
        seeds: tuple[int, ...],
        prompt: str | None | list[str | None],
        modality: str,
        height: int,
        width: int,
        max_episode_frames: int,
    ) -> _Stream: ...

    def close(self) -> None: ...


class _ServiceState:
    def __init__(self, model: _Model):
        self.model = model
        self.stopping = threading.Event()
        self.ownership_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cosmos-inference")
        self.owner: object | None = None
        self.transports: list[str] = ["socket"]
        self.pci_bus_id: str | None = None


def _transports(model: _Model) -> tuple[list[str], str | None]:
    """Return the offered transports and, for CUDA IPC, the PCI bus ID of the model's GPU.

    The device is read from its ``cuda:N`` name, so a service without Torch starts with the socket transport.
    """
    from .. import _cuda_ipc

    kind, _, index = str(model.capabilities.get("device", "cpu")).partition(":")
    if kind != "cuda" or not _cuda_ipc.available():
        return ["socket"], None
    return ["socket", "cuda_ipc"], _cuda_ipc.pci_bus_id(int(index or 0))


def _handle_connection(
    connection: socket.socket,
    state: _ServiceState,
    connections: set[socket.socket],
    connections_lock: threading.Lock,
) -> None:
    token = object()
    stream: _Stream | None = None
    channel = None
    image_shape: tuple[int, int, int] | None = None
    num_views = 0
    try:
        while not state.stopping.is_set():
            metadata, arrays = receive_message(connection)
            operation = metadata.get("op")
            if operation == "status":
                _check_fields(metadata, {"op", "version"}, arrays)
                with state.ownership_lock:
                    active = state.owner is not None
                send_message(
                    connection,
                    {
                        "ok": True,
                        "ready": not state.stopping.is_set(),
                        "capabilities": {
                            **state.model.capabilities,
                            "transports": state.transports,
                            "pci_bus_id": state.pci_bus_id,
                        },
                        "session_active": active,
                    },
                )
            elif operation == "open":
                _check_fields(
                    metadata,
                    {
                        "op",
                        "version",
                        "num_views",
                        "seeds",
                        "prompt",
                        "modality",
                        "height",
                        "width",
                        "max_episode_frames",
                    }
                    | ({"transport", "ipc"} if "transport" in metadata else set()),
                    arrays,
                )
                if stream is not None:
                    raise RuntimeError("This connection already has a Cosmos generation session.")
                transport = metadata.pop("transport", "socket")
                handles = metadata.pop("ipc", None)
                if transport not in state.transports:
                    raise ValueError(f"This Cosmos service does not offer the {transport} transport.")
                arguments = _open_arguments(metadata)
                if transport == "socket" and arguments["num_views"] > MAX_ARRAYS:
                    raise ValueError(f"The socket transport carries at most {MAX_ARRAYS} views; use CUDA IPC.")
                with state.ownership_lock:
                    if state.owner is not None:
                        raise RuntimeError(
                            "Cosmos already has an active generation session; close it before connecting."
                        )
                    if state.stopping.is_set():
                        raise RuntimeError("Cosmos service is shutting down.")
                    state.owner = token
                stream = state.executor.submit(state.model.open_stream, **arguments).result()
                image_shape = (arguments["height"], arguments["width"], 3)
                num_views = arguments["num_views"]
                if transport == "cuda_ipc":
                    channel = state.executor.submit(
                        _open_channel, state.model, num_views, image_shape, handles
                    ).result()
                send_message(connection, {"ok": True})
            elif operation == "step":
                fields = {"op", "version", "reset_rows", "seeds"} | ({"prompt"} if "prompt" in metadata else set())
                fields |= {"frames"} if channel is not None else set()
                _check_fields(metadata, fields, arrays, allow_arrays=channel is None)
                if stream is None:
                    raise RuntimeError("Open a Cosmos generation session before sending controls.")
                resets, seeds = _reset_arguments(metadata, num_views)
                episode = _episode_arguments(metadata, resets)
                if channel is not None:
                    frames = _channel_frames(metadata["frames"], num_views)
                    state.executor.submit(_step_channel, stream, channel, frames, resets, seeds, episode).result()
                    send_message(connection, {"ok": True, "frames": metadata["frames"]})
                    continue
                if len(arrays) != num_views or any(array.shape[1:] != image_shape for array in arrays):
                    raise ValueError("Cosmos controls must hold one chunk per view at the configured image size.")
                generated = state.executor.submit(stream.step, arrays, resets, seeds, **episode).result()
                _check_generated(generated, arrays)
                send_message(connection, {"ok": True}, generated)
            elif operation == "close":
                _check_fields(metadata, {"op", "version"}, arrays)
                if stream is not None:
                    state.executor.submit(stream.close).result()
                    stream = None
                if channel is not None:
                    state.executor.submit(channel.close).result()
                    channel = None
                with state.ownership_lock:
                    if state.owner is token:
                        state.owner = None
                image_shape = None
                send_message(connection, {"ok": True})
            elif operation == "shutdown":
                _check_fields(metadata, {"op", "version"}, arrays)
                send_message(connection, {"ok": True})
                state.stopping.set()
                break
            else:
                raise ProtocolError("Unknown Cosmos operation; expected status, open, step, close, or shutdown.")
    except (ConnectionError, TimeoutError, OSError):
        pass
    except Exception as exc:
        # A failed generation may have advanced temporal state. End the session rather than retry.
        if stream is not None:
            _LOGGER.exception("Cosmos generation session failed.")
        with suppress(ConnectionError, TimeoutError, OSError):
            send_message(connection, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        try:
            if stream is not None:
                state.executor.submit(stream.close).result()
            if channel is not None:
                state.executor.submit(channel.close).result()
        except Exception:
            _LOGGER.exception("Failed to close Cosmos generation state after a client disconnected.")
        finally:
            with state.ownership_lock:
                if state.owner is token:
                    state.owner = None
            with connections_lock:
                connections.discard(connection)
            connection.close()


def _open_channel(model: _Model, num_views: int, image_shape: tuple[int, int, int], handles: object):
    """Open the camera's shared GPU buffers, one row per view, and events on the model's device (inference thread)."""
    import torch

    from .._cuda_ipc import SharedChannel

    keys = {"control", "output", "control_ready", "output_ready"}
    if not isinstance(handles, dict) or set(handles) != keys or not all(isinstance(v, str) for v in handles.values()):
        raise ProtocolError("A CUDA IPC session needs control, output, control_ready and output_ready handles.")
    shape = (num_views, MAX_CHUNK_FRAMES, *image_shape)
    return SharedChannel(torch.device(model.capabilities["device"]), shape, handles)


def _step_channel(stream: _Stream, channel, frames: list[int], resets, seeds, episode: dict) -> None:
    """Generate from shared GPU controls into the shared output, ordered by events (inference thread)."""
    import torch

    channel.control_ready.wait()
    controls = [channel.control.tensor[view, :count] for view, count in enumerate(frames)]
    generated = stream.step(controls, resets, seeds, **episode)
    if (
        not isinstance(generated, list)
        or len(generated) != len(frames)
        or any(
            not isinstance(images, torch.Tensor)
            or images.dtype != torch.uint8
            or tuple(images.shape) != tuple(control.shape)
            for images, control in zip(generated, controls)
        )
    ):
        raise ValueError("Cosmos must return uint8 THWC RGB matching each view's control chunk.")
    for view, (images, count) in enumerate(zip(generated, frames)):
        channel.output.tensor[view, :count].copy_(images)
    channel.output_ready.record()
    # A GPU fault must reach the client as an error reply; its stream would otherwise wait on output_ready forever.
    # This waits in the service process only; the camera's process does not synchronize.
    torch.cuda.current_stream(channel.output.device).synchronize()


def _episode_arguments(metadata: dict, resets: tuple[int, ...]) -> dict:
    """Return a step's episode settings: a reset may bring a new appearance prompt; the weights stay loaded."""
    if "prompt" not in metadata:
        return {}
    if not resets:
        raise ValueError("A Cosmos prompt can change only with an episode reset.")
    if metadata["prompt"] is not None and not isinstance(metadata["prompt"], str):
        raise ValueError("Cosmos prompt must be a string or None.")
    return {"prompt": metadata["prompt"]}


def _channel_frames(frames: object, num_views: int) -> list[int]:
    """Return one frame count per view of a CUDA IPC step; a single integer is the one-view form."""
    frames = [frames] if type(frames) is int else frames
    if (
        not isinstance(frames, list)
        or len(frames) != num_views
        or any(type(count) is not int or not 1 <= count <= MAX_CHUNK_FRAMES for count in frames)
    ):
        raise ValueError(f"Cosmos chunks hold 1 to {MAX_CHUNK_FRAMES} frames for each view.")
    return frames


def _check_generated(generated: object, arrays: list[np.ndarray]) -> None:
    if (
        not isinstance(generated, list)
        or len(generated) != len(arrays)
        or any(
            not isinstance(images, np.ndarray) or images.dtype != np.uint8 or images.shape != array.shape
            for images, array in zip(generated, arrays)
        )
    ):
        raise ValueError("Cosmos must return uint8 THWC RGB matching each view's control chunk.")


def _check_fields(metadata: dict, fields: set[str], arrays: list[np.ndarray], *, allow_arrays: bool = False) -> None:
    if set(metadata) != fields:
        raise ProtocolError("Cosmos request contains missing or unsupported metadata fields.")
    if arrays and not allow_arrays:
        raise ProtocolError("This Cosmos operation does not accept image arrays.")


def _open_arguments(metadata: dict) -> dict:
    num_views = metadata["num_views"]
    if type(num_views) is not int or num_views < 1:
        raise ValueError("Cosmos num_views must be a positive integer.")
    for name in ("height", "width", "max_episode_frames"):
        if type(metadata[name]) is not int or metadata[name] <= 0:
            raise ValueError(f"Cosmos {name} must be a positive integer.")
    prompt = metadata["prompt"]
    prompts = prompt if isinstance(prompt, list) else [prompt]
    if (isinstance(prompt, list) and len(prompt) != num_views) or any(
        text is not None and not isinstance(text, str) for text in prompts
    ):
        raise ValueError("Cosmos prompt must be a string or None, or one per view.")
    if metadata["modality"] not in ("edge", "blur", "depth", "seg"):
        raise ValueError("Cosmos modality must be edge, blur, depth, or seg.")
    seeds = _seeds(metadata["seeds"], num_views)
    return {
        name: (seeds if name == "seeds" else value) for name, value in metadata.items() if name not in ("op", "version")
    }


def _reset_arguments(metadata: dict, num_views: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    rows = metadata["reset_rows"]
    if (
        not isinstance(rows, list)
        or any(type(row) is not int or not 0 <= row < num_views for row in rows)
        or rows != sorted(set(rows))
    ):
        raise ValueError("Cosmos resets name distinct opened views in increasing order.")
    return tuple(rows), _seeds(metadata["seeds"], len(rows))


def _seeds(values: object, count: int) -> tuple[int, ...]:
    if (
        not isinstance(values, list)
        or len(values) != count
        or any(type(seed) is not int or not 0 <= seed < 2**31 for seed in values)
    ):
        raise ValueError("Cosmos requires one integer seed in [0, 2**31) for each opened or reset view.")
    return tuple(values)
