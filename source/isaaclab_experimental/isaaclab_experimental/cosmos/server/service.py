# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Endpoint lifecycle and inference-thread ownership for the standalone Cosmos service."""

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

from .._protocol import DEFAULT_ENDPOINT, ProtocolError, disable_nagle, parse_endpoint

_LOGGER = logging.getLogger(__name__)
_IDLE_TIMEOUT_S = 600.0
"""Maximum idle connection time [s]."""


def serve(
    model: _Model | Callable[[], _Model],
    endpoint: str = DEFAULT_ENDPOINT,
    *,
    warmup: bool = False,
    handler: Callable[[socket.socket, _ServiceState, set[socket.socket], threading.Lock], None],
) -> None:
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
        handler: Serves the world-model protocol in a thread per connection. The service manages the endpoint,
            inference thread and shutdown.

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
                socket_path, socket_inode = address, os.stat(address).st_ino
            listener.listen(16)
            listener.settimeout(0.25)
            _LOGGER.info(
                "Cosmos reserved %s; loading the model. Wait for the ready message before connecting.", endpoint
            )
            if resource is None:
                resource = model()
            state = _ServiceState(resource)
            transports = state.executor.submit(_transports, resource).result()
            state.transports, state.host_id, state.pci_bus_id, state.visible_gpus = transports
            if warmup:
                _LOGGER.info("Cosmos warming up; the first compilation can take several minutes.")
                state.executor.submit(resource.warmup).result()
            _LOGGER.info("Cosmos ready at %s", endpoint)
            while not state.stopping.is_set():
                try:
                    connection, _ = listener.accept()
                except TimeoutError:
                    continue
                connection.settimeout(_IDLE_TIMEOUT_S)
                if family == socket.AF_INET:
                    disable_nagle(connection)
                with connections_lock:
                    connections.add(connection)
                thread = threading.Thread(
                    target=handler,
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
                    # Another service may have replaced the file meanwhile; its socket stays.
                    if os.stat(socket_path).st_ino == socket_inode:
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
            except ConnectionRefusedError:
                # Nothing listens on the file: a stopped service left it behind.
                os.unlink(path)
            except FileNotFoundError:
                pass
            except OSError as error:
                # A busy or inaccessible service can still be running; replacing its file would strand it.
                raise RuntimeError(f"Cannot tell whether a Cosmos service is running at unix://{path}.") from error
            else:
                raise RuntimeError(f"A Cosmos service is already running at unix://{path}.")
    # The process umask applies to the socket file; owner-only from the start leaves no window for other users.
    previous = os.umask(0o177)
    try:
        listener.bind(path)
    finally:
        os.umask(previous)


class _Model(Protocol):
    capabilities: dict

    def warmup(self) -> None: ...

    def open_session(self, num_envs: int): ...

    def close(self) -> None: ...


class _ServiceState:
    def __init__(self, model: _Model):
        self.model = model
        self.stopping = threading.Event()
        self.ownership_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cosmos-inference")
        self.owner: object | None = None
        self.transports: list[str] = ["socket"]
        self.host_id: str | None = None
        self.pci_bus_id: str | None = None
        self.visible_gpus: list[str] = []


def _transports(model: _Model) -> tuple[list[str], str | None, str | None, list[str]]:
    """Return the offered transports and, for CUDA IPC, the machine's boot ID, the PCI bus ID of the model's GPU,
    and the PCI bus IDs of the GPUs this service can see, one of which must be the client's.

    The device is read from its ``cuda:N`` name, so a service without Torch starts with the socket transport.
    """
    from .. import _cuda_ipc

    kind, _, index = str(model.capabilities.get("device", "cpu")).partition(":")
    if kind != "cuda" or not _cuda_ipc.available():
        return ["socket"], None, None, []
    model_gpu = _cuda_ipc.pci_bus_id(int(index or 0))
    return ["socket", "cuda_ipc"], _cuda_ipc.host_id(), model_gpu, _cuda_ipc.visible_gpus()


def _run_inference(state: _ServiceState, function, *args, **kwargs):
    """Run a model call on the inference thread and return its result.

    The connection handler reads an ``OSError`` as the client disconnecting or idling. One raised by the model is a
    failed generation, so it is raised as a ``RuntimeError``, which the handler reports to the client.
    """
    try:
        return state.executor.submit(function, *args, **kwargs).result()
    except OSError as exc:
        raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc


def _await_disconnect(connection: socket.socket, stopping: threading.Event) -> bool:
    """Wait until the client closes the connection, discarding anything it still sends, up to the idle timeout.

    Returns:
        Whether the client closed the connection, which confirms that it released this service's buffers. An idle
        timeout or the service's own shutdown confirms nothing.
    """
    try:
        while connection.recv(65536):
            pass
    except ConnectionError:
        # The client's process reset the connection; its mappings end with it.
        pass
    except OSError:
        return False
    return not stopping.is_set()


def _check_fields(metadata: dict, fields: set[str], arrays: list[np.ndarray], *, allow_arrays: bool = False) -> None:
    if set(metadata) != fields:
        raise ProtocolError("Cosmos request contains missing or unsupported metadata fields.")
    if arrays and not allow_arrays:
        raise ProtocolError("This Cosmos operation does not accept arrays.")
