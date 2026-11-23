# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resident model service, independent of simulation and model framework imports."""

from __future__ import annotations

import logging
import os
import socket
import stat
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from typing import TYPE_CHECKING, Protocol

import numpy as np

from .protocol import (
    MAX_ARRAYS,
    SEED_LIMIT,
    ProtocolError,
    disable_nagle,
    parse_endpoint,
    receive_message,
    send_message,
)

if TYPE_CHECKING:
    import torch

_LOGGER = logging.getLogger(__name__)

_IDLE_TIMEOUT_S = 600.0
"""Seconds a connection may stay silent before the service closes it, so a camera that stopped without closing its
connection does not hold the model. The connection that owns a session whose weights are offloaded waits without
limit, since a learner can use the GPU for longer between rollouts."""


def serve(
    model: _Model | Callable[[], _Model], endpoint: str, *, warmup: bool = False, name: str = "Model service"
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
        name: The model's name in log messages, for example ``"Cosmos"``.

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
                "%s reserved %s; loading the model. Wait for the ready message before connecting.", name, endpoint
            )
            if resource is None:
                resource = model()
            state = _ServiceState(resource)
            transports = state.executor.submit(_transports, resource).result()
            state.transports, state.gpu_uuid, state.visible_gpus = transports
            if warmup:
                _LOGGER.info("%s warming up; the first compilation can take several minutes.", name)
                state.executor.submit(resource.warmup).result()
            _LOGGER.info("%s ready at %s", name, endpoint)
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
                    name="model-client",
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
            raise RuntimeError(f"{path} exists and is not a socket; choose another endpoint.")
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
                raise RuntimeError(f"Cannot tell whether a model service is running at unix://{path}.") from error
            else:
                raise RuntimeError(f"A model service is already running at unix://{path}.")
    # The process umask applies to the socket file; owner-only from the start leaves no window for other users.
    previous = os.umask(0o177)
    try:
        listener.bind(path)
    finally:
        os.umask(previous)


class _Stream(Protocol):
    """A generation session stepped with host arrays (socket) or CUDA views of the shared buffer (CUDA IPC).

    Each step returns images of the same kind as its controls. ``episode`` holds the client's episode settings for
    the views that reset, which the model checks.
    """

    def step(
        self,
        controls: list[np.ndarray] | list[torch.Tensor],
        reset_rows: tuple[int, ...],
        seeds: tuple[int, ...],
        **episode: object,
    ) -> list[np.ndarray] | list[torch.Tensor]: ...

    def close(self) -> None: ...


class _Model(Protocol):
    """A model the service keeps resident.

    ``capabilities`` describes it to clients. The service reads ``device``, the model's ``cuda:N`` device for CUDA
    IPC, and ``max_chunk_frames``, the most frames a view sends in one step (1 when absent). ``settings`` holds the
    client's model settings, which the model checks.
    """

    capabilities: dict

    def warmup(self) -> None: ...

    def open_stream(
        self, *, num_views: int, seeds: tuple[int, ...], height: int, width: int, **settings
    ) -> _Stream: ...

    def close(self) -> None: ...


class _ServiceState:
    def __init__(self, model: _Model):
        self.model = model
        self.stopping = threading.Event()
        self.ownership_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-inference")
        self.owner: object | None = None
        self.transports: list[str] = ["socket"]
        self.gpu_uuid: str | None = None
        self.visible_gpus: list[str] = []


def _transports(model: _Model) -> tuple[list[str], str | None, list[str]]:
    """Return the offered transports and, for CUDA IPC, the UUID of the model's GPU and the UUIDs of the GPUs this
    service can see, one of which must be the camera's.

    The device is read from its ``cuda:N`` name, so a service without Torch starts with the socket transport.
    """
    from . import ipc

    kind, _, index = str(model.capabilities.get("device", "cpu")).partition(":")
    if kind != "cuda" or not ipc.available():
        return ["socket"], None, []
    return ["socket", "cuda_ipc"], ipc.gpu_uuid(int(index or 0)), ipc.visible_gpus()


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
    channel = None
    camera_released = False
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
                        "capabilities": {
                            **state.model.capabilities,
                            "transports": state.transports,
                            "gpu_uuid": state.gpu_uuid,
                            "visible_gpus": state.visible_gpus,
                        },
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
                    {"op", "version", "num_views", "seeds", "height", "width", "settings"}
                    | ({"transport", "ipc"} if "transport" in metadata else set()),
                    arrays,
                )
                if stream is not None:
                    raise RuntimeError("This connection already has a generation session.")
                transport = metadata.pop("transport", "socket")
                handles = metadata.pop("ipc", None)
                if transport not in state.transports:
                    raise ValueError(f"This service does not offer the {transport} transport.")
                arguments, settings = _open_arguments(metadata)
                if transport == "socket" and arguments["num_views"] > MAX_ARRAYS:
                    raise ValueError(f"The socket transport carries at most {MAX_ARRAYS} views; use CUDA IPC.")
                with state.ownership_lock:
                    if state.owner is not None:
                        raise RuntimeError(
                            "The model already has an active generation session; close it before connecting."
                        )
                    if state.stopping.is_set():
                        raise RuntimeError("The model service is shutting down.")
                    state.owner = token
                stream = _run_inference(state, state.model.open_stream, **arguments, **settings)
                image_shape = (arguments["height"], arguments["width"], 3)
                num_views = arguments["num_views"]
                reply = {"ok": True}
                if transport == "cuda_ipc":
                    channel = _run_inference(state, _open_channel, state.model, num_views, image_shape, handles)
                    # The camera opens the generated-image buffer and event this service writes on its own GPU.
                    reply["ipc"] = channel.handles
                send_message(connection, reply)
            elif operation == "step":
                fields = {"op", "version", "reset_rows", "seeds"} | ({"episode"} if "episode" in metadata else set())
                fields |= {"frames"} if channel is not None else set()
                _check_fields(metadata, fields, arrays, allow_arrays=channel is None)
                if stream is None:
                    raise RuntimeError("Open a generation session before sending controls.")
                resets, seeds = _reset_arguments(metadata, num_views)
                episode = _episode_arguments(metadata)
                if channel is not None:
                    frames = _channel_frames(metadata["frames"], num_views, _max_chunk_frames(state.model))
                    _run_inference(state, _step_channel, stream, channel, frames, resets, seeds, episode)
                    send_message(connection, {"ok": True})
                    continue
                if len(arrays) != num_views or any(array.shape[1:] != image_shape for array in arrays):
                    raise ValueError("Controls must hold one chunk per view at the configured image size.")
                generated = _run_inference(state, stream.step, arrays, resets, seeds, **episode)
                _check_generated(generated, arrays)
                send_message(connection, {"ok": True}, generated)
            elif operation == "close":
                _check_fields(metadata, {"op", "version"}, arrays)
                if stream is not None:
                    _run_inference(state, stream.close)
                    stream = None
                if channel is not None:
                    _run_inference(state, channel.close)
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
                raise ProtocolError(
                    "Unknown operation; expected status, open, step, close, offload, activate, or shutdown."
                )
    except ConnectionError:
        # The camera disconnected. It releases the output buffer first, and a process's mappings end with it. The
        # service's own shutdown also ends the connection, which confirms nothing.
        camera_released = not state.stopping.is_set()
    except TimeoutError:
        # An idle camera learns at its next request that the session ended, while it still has its buffers open.
        with suppress(OSError):
            send_message(connection, {"ok": False, "error": "The session was idle too long and closed."})
    except OSError:
        pass
    except Exception as exc:
        # A failed generation may have advanced temporal state. End the session rather than retry.
        if stream is not None:
            _LOGGER.exception("Generation session failed.")
        with suppress(ConnectionError, TimeoutError, OSError):
            send_message(connection, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        try:
            if stream is not None:
                state.executor.submit(stream.close).result()
            if channel is not None:
                state.executor.submit(channel.close_opened).result()
        except Exception:
            _LOGGER.exception("Failed to close generation state after a client disconnected.")
        finally:
            with state.ownership_lock:
                if state.owner is token:
                    state.owner = None
            try:
                if channel is not None:
                    # The camera may still have the output buffer open; it releases it before disconnecting. Without
                    # that confirmation, the buffer stays allocated until this service exits.
                    with suppress(OSError):
                        connection.settimeout(_IDLE_TIMEOUT_S)
                    released = camera_released or _await_disconnect(connection, state.stopping)
                    state.executor.submit(channel.close, peer_released=released).result()
            finally:
                with connections_lock:
                    connections.discard(connection)
                connection.close()


def _change_residency(state: _ServiceState, token: object, operation: str) -> None:
    """Offload or activate the weights for the connection that owns the session, or for any connection while idle.

    An idle service is reserved for the change, so no session opens while the weights move.
    """
    method = getattr(state.model, operation, None)
    if method is None:
        raise ValueError(f"This service cannot {operation} its model.")
    with state.ownership_lock:
        if state.owner is not None and state.owner is not token:
            raise RuntimeError("Another connection owns the session; offload or activate through it.")
        idle = state.owner is None
        state.owner = token
    try:
        _run_inference(state, method)
    finally:
        if idle:
            with state.ownership_lock:
                if state.owner is token:
                    state.owner = None


def _await_disconnect(connection: socket.socket, stopping: threading.Event) -> bool:
    """Wait until the camera closes the connection, discarding anything it still sends, up to the idle timeout.

    Returns:
        Whether the camera closed the connection, which confirms that it released this service's buffers. An idle
        timeout or the service's own shutdown confirms nothing.
    """
    try:
        while connection.recv(65536):
            pass
    except ConnectionError:
        # The camera's process reset the connection; its mappings end with it.
        pass
    except OSError:
        return False
    return not stopping.is_set()


def _open_channel(model: _Model, num_views: int, image_shape: tuple[int, int, int], handles: object):
    """Create the output buffer and event on the model's GPU and open the camera's (inference thread).

    Buffers hold one row per view. The camera's controls stay on its GPU, which can be another GPU with direct
    access to the model's.
    """
    import torch

    from . import ipc

    if (
        not isinstance(handles, dict)
        or set(handles) != {"gpu_uuid", "control", "control_ready"}
        or not all(isinstance(value, str) for value in handles.values())
    ):
        raise ProtocolError("A CUDA IPC session needs the camera's gpu_uuid, control and control_ready handles.")
    device = torch.device(model.capabilities["device"])
    camera = ipc.device_by_uuid(handles["gpu_uuid"])
    if camera is None:
        raise ValueError(
            f"This service cannot see the camera's GPU {handles['gpu_uuid']}; keep that GPU visible to the service "
            "and select the model's GPU with its device option."
        )
    if not ipc.can_access_peer(camera, device.index or 0):
        raise ValueError("The camera's GPU and the service's GPU cannot access each other's memory directly.")
    channel = ipc.SharedChannel(device, (num_views, _max_chunk_frames(model), *image_shape), writes="output")
    try:
        channel.open(torch.device("cuda", camera), {name: handles[name] for name in ("control", "control_ready")})
    except Exception:
        channel.close()
        raise
    return channel


def _step_channel(stream: _Stream, channel, frames: list[int], resets, seeds, episode: dict) -> None:
    """Generate from the camera's shared controls into the shared output, ordered by events (inference thread)."""
    import torch

    device = channel.output.device
    channel.control_ready.wait()
    # On the camera's GPU this reads the controls in place; from another GPU it copies them between the GPUs.
    controls = [channel.control.tensor[view, :count].to(device) for view, count in enumerate(frames)]
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
        raise ValueError("The model must return uint8 THWC RGB matching each view's control chunk.")
    for view, (images, count) in enumerate(zip(generated, frames)):
        channel.output.tensor[view, :count].copy_(images)
    channel.output_ready.record()
    # A GPU fault must reach the client as an error reply; its stream would otherwise wait on output_ready forever.
    # This waits in the service process only; the camera's process does not synchronize.
    torch.cuda.current_stream(channel.output.device).synchronize()


def _max_chunk_frames(model: _Model) -> int:
    """Return the most frames a view sends in one step, as the model declares it."""
    limit = model.capabilities.get("max_chunk_frames", 1)
    if type(limit) is not int or limit < 1:
        raise ValueError("A model's max_chunk_frames must be a positive integer.")
    return limit


def _episode_arguments(metadata: dict) -> dict:
    """Return a step's episode settings for the views that reset, such as a new prompt; the model checks them."""
    episode = metadata.get("episode", {})
    if not isinstance(episode, dict) or not all(isinstance(name, str) for name in episode):
        raise ValueError("A step's episode settings must be a mapping of names to values.")
    return episode


def _channel_frames(frames: object, num_views: int, limit: int) -> list[int]:
    """Return one frame count per view of a CUDA IPC step."""
    if (
        not isinstance(frames, list)
        or len(frames) != num_views
        or any(type(count) is not int or not 1 <= count <= limit for count in frames)
    ):
        raise ValueError(f"Chunks hold 1 to {limit} frames for each view.")
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
        raise ValueError("The model must return uint8 THWC RGB matching each view's control chunk.")


def _check_fields(metadata: dict, fields: set[str], arrays: list[np.ndarray], *, allow_arrays: bool = False) -> None:
    if set(metadata) != fields:
        raise ProtocolError("The request contains missing or unsupported metadata fields.")
    if arrays and not allow_arrays:
        raise ProtocolError("This operation does not accept image arrays.")


def _open_arguments(metadata: dict) -> tuple[dict, dict]:
    """Return a session's view and image arguments, and the client's model settings, which the model checks."""
    num_views = metadata["num_views"]
    if type(num_views) is not int or num_views < 1:
        raise ValueError("num_views must be a positive integer.")
    for name in ("height", "width"):
        if type(metadata[name]) is not int or metadata[name] <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    settings = metadata["settings"]
    if not isinstance(settings, dict) or not all(isinstance(name, str) for name in settings):
        raise ValueError("A session's model settings must be a mapping of names to values.")
    seeds = _seeds(metadata["seeds"], num_views)
    arguments = {"num_views": num_views, "seeds": seeds, "height": metadata["height"], "width": metadata["width"]}
    return arguments, settings


def _reset_arguments(metadata: dict, num_views: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    rows = metadata["reset_rows"]
    if (
        not isinstance(rows, list)
        or any(type(row) is not int or not 0 <= row < num_views for row in rows)
        or rows != sorted(set(rows))
    ):
        raise ValueError("Resets name distinct opened views in increasing order.")
    return tuple(rows), _seeds(metadata["seeds"], len(rows))


def _seeds(values: object, count: int) -> tuple[int, ...]:
    if (
        not isinstance(values, list)
        or len(values) != count
        or any(type(seed) is not int or not 0 <= seed < SEED_LIMIT for seed in values)
    ):
        raise ValueError("Each opened or reset view needs one integer seed in [0, 2**31).")
    return tuple(values)
