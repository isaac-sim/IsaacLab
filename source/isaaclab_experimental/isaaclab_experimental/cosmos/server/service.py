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
    SEED_LIMIT,
    ProtocolError,
    disable_nagle,
    parse_endpoint,
    receive_message,
    send_message,
)

_LOGGER = logging.getLogger(__name__)

_IDLE_TIMEOUT_S = 600.0
"""Seconds a connection may stay silent before the service closes it, so a camera that stopped without closing its
connection does not hold the model. The connection that owns a session whose weights are offloaded waits without
limit, since a learner can use the GPU for longer between rollouts."""


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
                socket_path, socket_inode = address, os.stat(address).st_ino
            listener.listen(16)
            listener.settimeout(0.25)
            _LOGGER.info(
                "Cosmos reserved %s; loading the model. Wait for the ready message before connecting.", endpoint
            )
            if resource is None:
                resource = model()
            state = _ServiceState(resource)
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


class _Stream(Protocol):
    """A generation session stepped with the host arrays a camera sends; each step returns host arrays."""

    def step(
        self,
        controls: list[np.ndarray],
        reset_rows: tuple[int, ...],
        seeds: tuple[int, ...],
        **episode: object,
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
        height: int,
        width: int,
        max_episode_frames: int,
        fps: float,
    ) -> _Stream: ...

    def close(self) -> None: ...


class _ServiceState:
    def __init__(self, model: _Model):
        self.model = model
        self.stopping = threading.Event()
        self.ownership_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cosmos-inference")
        self.owner: object | None = None


def _run_inference(state: _ServiceState, function, *args, **kwargs):
    """Run a model call on the inference thread and return its result.

    The connection handler reads an ``OSError`` as the camera disconnecting or idling. One raised by the model is a
    failed generation, so it is raised as a ``RuntimeError``, which the handler reports to the camera.
    """
    try:
        return state.executor.submit(function, *args, **kwargs).result()
    except OSError as exc:
        raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc


def _handle_connection(
    connection: socket.socket,
    state: _ServiceState,
    connections: set[socket.socket],
    connections_lock: threading.Lock,
) -> None:
    token = object()
    stream: _Stream | None = None
    image_shape: tuple[int, int, int] | None = None
    num_views = 0
    try:
        while not state.stopping.is_set():
            # A learner may keep its session's weights offloaded for longer than the idle timeout between rollouts.
            offloaded = stream is not None and not getattr(state.model, "active", True)
            connection.settimeout(None if offloaded else _IDLE_TIMEOUT_S)
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
                        "capabilities": state.model.capabilities,
                        "session_active": active,
                        "model_active": getattr(state.model, "active", True),
                    },
                )
            elif operation in ("offload", "activate"):
                _check_fields(metadata, {"op", "version"}, arrays)
                _change_residency(state, token, operation)
                send_message(connection, {"ok": True})
            elif operation == "open":
                _check_fields(
                    metadata,
                    {
                        "op",
                        "version",
                        "num_views",
                        "seeds",
                        "prompt",
                        "height",
                        "width",
                        "max_episode_frames",
                        "fps",
                    },
                    arrays,
                )
                if stream is not None:
                    raise RuntimeError("This connection already has a Cosmos generation session.")
                arguments = _open_arguments(metadata)
                if arguments["num_views"] > MAX_ARRAYS:
                    raise ValueError(f"A Cosmos step message carries at most {MAX_ARRAYS} views.")
                with state.ownership_lock:
                    if state.owner is not None:
                        raise RuntimeError(
                            "Cosmos already has an active generation session; close it before connecting."
                        )
                    if state.stopping.is_set():
                        raise RuntimeError("Cosmos service is shutting down.")
                    state.owner = token
                stream = _run_inference(state, state.model.open_stream, **arguments)
                image_shape = (arguments["height"], arguments["width"], 3)
                num_views = arguments["num_views"]
                send_message(connection, {"ok": True})
            elif operation == "step":
                fields = {"op", "version", "reset_rows", "seeds"} | ({"prompt"} if "prompt" in metadata else set())
                _check_fields(metadata, fields, arrays, allow_arrays=True)
                if stream is None:
                    raise RuntimeError("Open a Cosmos generation session before sending controls.")
                resets, seeds = _reset_arguments(metadata, num_views)
                episode = _episode_arguments(metadata, resets)
                if len(arrays) != num_views or any(array.shape[1:] != image_shape for array in arrays):
                    raise ValueError("Cosmos controls must hold one chunk per view at the configured image size.")
                generated = _run_inference(state, stream.step, arrays, resets, seeds, **episode)
                _check_generated(generated, arrays)
                send_message(connection, {"ok": True}, generated)
            elif operation == "close":
                _check_fields(metadata, {"op", "version"}, arrays)
                if stream is not None:
                    _run_inference(state, stream.close)
                    stream = None
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
                raise ProtocolError(
                    "Unknown Cosmos operation; expected status, open, step, close, offload, activate, or shutdown."
                )
    except ConnectionError:
        # The camera disconnected; its session closes below.
        pass
    except TimeoutError:
        # An idle camera learns at its next request that the session ended.
        with suppress(OSError):
            send_message(connection, {"ok": False, "error": "The Cosmos session was idle too long and closed."})
    except OSError:
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
        except Exception:
            _LOGGER.exception("Failed to close Cosmos generation state after a client disconnected.")
        finally:
            with state.ownership_lock:
                if state.owner is token:
                    state.owner = None
            with connections_lock:
                connections.discard(connection)
            connection.close()


def _change_residency(state: _ServiceState, token: object, operation: str) -> None:
    """Offload or activate the weights for the connection that owns the session, or for any connection while idle.

    An idle service is reserved for the change, so no session opens while the weights move.
    """
    method = getattr(state.model, operation, None)
    if method is None:
        raise ValueError(f"This Cosmos service cannot {operation} its model.")
    with state.ownership_lock:
        if state.owner is not None and state.owner is not token:
            raise RuntimeError("Another connection owns the Cosmos session; offload or activate through it.")
        idle = state.owner is None
        state.owner = token
    try:
        _run_inference(state, method)
    finally:
        if idle:
            with state.ownership_lock:
                if state.owner is token:
                    state.owner = None


def _episode_arguments(metadata: dict, resets: tuple[int, ...]) -> dict:
    """Return a step's episode settings: a reset may bring a new appearance prompt; the weights stay loaded."""
    if "prompt" not in metadata:
        return {}
    if not resets:
        raise ValueError("A Cosmos prompt can change only with an episode reset.")
    prompt = metadata["prompt"]
    prompts = prompt if isinstance(prompt, list) else [prompt]
    if (isinstance(prompt, list) and len(prompt) != len(resets)) or any(
        text is not None and not isinstance(text, str) for text in prompts
    ):
        raise ValueError("Cosmos prompt must be a string or None, or one per reset view.")
    return {"prompt": prompt}


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
        or any(type(seed) is not int or not 0 <= seed < SEED_LIMIT for seed in values)
    ):
        raise ValueError("Cosmos requires one integer seed in [0, 2**31) for each opened or reset view.")
    return tuple(values)
