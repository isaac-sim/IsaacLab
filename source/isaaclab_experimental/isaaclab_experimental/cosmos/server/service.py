# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resident optional Cosmos service, independent of simulation and model framework imports."""

from __future__ import annotations

import logging
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from typing import Protocol

import numpy as np

from .._protocol import MAX_ARRAYS, ProtocolError, receive_message, send_message

_LOGGER = logging.getLogger(__name__)


def serve(model: _Model, host: str = "127.0.0.1", port: int = 5555, *, warmup: bool = False) -> None:
    """Serve an already initialized model until shutdown or interruption.

    A connection owns at most one generation session. Only one session may use the resident model
    at a time; separate status and shutdown connections remain available during generation.
    Disconnects and errors close session state while keeping model weights resident.
    Model operations always run on one persistent inference thread, including reconnects, because
    compiled CUDA graph state is thread-local.

    Args:
        model: Loaded model resource implementing the NumPy stream interface.
        host: Interface to bind. Use loopback for local operation.
        port: TCP listening port.
        warmup: Warm the model on its inference thread before exposing the ready endpoint.

    The service owns ``model`` and closes it on exit, including failed startup. The protocol has no
    authentication: remote connections should use a trusted tunnel rather than a public interface.
    """
    state = _ServiceState(model)
    connections: set[socket.socket] = set()
    threads: list[threading.Thread] = []
    connections_lock = threading.Lock()
    try:
        if warmup:
            state.executor.submit(model.warmup).result()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((host, port))
            listener.listen(16)
            listener.settimeout(0.25)
            _LOGGER.info("Cosmos ready at tcp://%s:%d", host, listener.getsockname()[1])
            while not state.stopping.is_set():
                try:
                    connection, _ = listener.accept()
                except TimeoutError:
                    continue
                connection.settimeout(600.0)
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
        state.stopping.set()
        with connections_lock:
            for connection in connections:
                with suppress(OSError):
                    connection.shutdown(socket.SHUT_RDWR)
        try:
            # In-flight inference and session cleanup complete before the model is released.
            for thread in threads:
                thread.join()
            state.executor.submit(model.close).result()
        finally:
            state.executor.shutdown(wait=True)


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
        prompt: str | None,
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
                    },
                    arrays,
                )
                if stream is not None:
                    raise RuntimeError("This connection already has a Cosmos generation session.")
                arguments = _open_arguments(metadata)
                if arguments["num_views"] > MAX_ARRAYS:
                    raise ValueError(f"One Cosmos message carries at most {MAX_ARRAYS} views.")
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
                generated = state.executor.submit(stream.step, arrays, resets, seeds, **episode).result()
                _check_generated(generated, arrays)
                send_message(connection, {"ok": True}, generated)
            elif operation == "close":
                _check_fields(metadata, {"op", "version"}, arrays)
                if stream is not None:
                    state.executor.submit(stream.close).result()
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
        except Exception:
            _LOGGER.exception("Failed to close Cosmos generation state after a client disconnected.")
        finally:
            with state.ownership_lock:
                if state.owner is token:
                    state.owner = None
            with connections_lock:
                connections.discard(connection)
            connection.close()


def _episode_arguments(metadata: dict, resets: tuple[int, ...]) -> dict:
    """Return a step's episode settings: a reset may bring a new appearance prompt; the weights stay loaded."""
    if "prompt" not in metadata:
        return {}
    if not resets:
        raise ValueError("A Cosmos prompt can change only with an episode reset.")
    if metadata["prompt"] is not None and not isinstance(metadata["prompt"], str):
        raise ValueError("Cosmos prompt must be a string or None.")
    return {"prompt": metadata["prompt"]}


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
