# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded messages for the optional Cosmos service.

Messages consist of a fixed network-order header, UTF-8 JSON metadata, and contiguous uint8
THWC arrays. Array descriptors carry their shape and byte count; no executable serialization
or compressed payloads are accepted. This protocol is intended for trusted local connections.
"""

from __future__ import annotations

import json
import math
import os
import socket
import struct
from collections.abc import Sequence
from urllib.parse import urlsplit

import numpy as np

PROTOCOL_VERSION = 2
"""Version 2 lets an episode reset carry the next appearance prompt."""
CANVAS_ASPECT_RATIOS = {
    (480, 832): "16,9",
    (544, 736): "4,3",
    (640, 640): "1,1",
    (736, 544): "3,4",
    (832, 480): "9,16",
}
"""Image sizes ``(height, width)`` the Cosmos service accepts, with their aspect-ratio names."""
DEFAULT_MAX_EPISODE_FRAMES = 201
"""Default episode cap of the service in frames, the model's trained horizon;
``isaaclab-cosmos-server --max-episode-frames`` changes it."""
MAX_CHUNK_FRAMES = 4
"""Most frames one control chunk carries: one initial frame, then four per update."""
DEFAULT_ENDPOINT = (
    f"unix:///tmp/isaaclab-cosmos-{os.getuid()}.sock" if hasattr(socket, "AF_UNIX") else "tcp://127.0.0.1:5555"
)
"""Default service endpoint: a Unix socket only this user can open on Linux, local TCP where Unix sockets are
unavailable (Windows)."""
_ENDPOINT_FORMS = "unix:///absolute/path or tcp://host:port"
_MAX_UNIX_PATH_BYTES = 107
MAX_METADATA_BYTES = 64 * 1024
MAX_ARRAY_BYTES = 256 * 1024 * 1024
MAX_ARRAYS = 16
_HEADER = struct.Struct("!4sIQ")
_MAGIC = b"ILCS"


def connect(endpoint: str, timeout: float = 600.0) -> socket.socket:
    """Connect to a ``unix:///path`` or ``tcp://host:port`` service with a finite timeout [s]."""
    family, address = parse_endpoint(endpoint)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Cosmos timeout must be finite and positive.")
    if family == socket.AF_INET:
        connection = socket.create_connection(address, timeout=timeout)
        disable_nagle(connection)
        return connection
    connection = socket.socket(family, socket.SOCK_STREAM)
    try:
        connection.settimeout(timeout)
        connection.connect(address)
    except BaseException:
        connection.close()
        raise
    return connection


def disable_nagle(connection: socket.socket) -> None:
    """Send small TCP messages at once; otherwise each step can wait for a delayed acknowledgment (40 ms on Linux)."""
    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


def parse_endpoint(endpoint: str) -> tuple[socket.AddressFamily, str | tuple[str, int]]:
    """Validate an endpoint without contacting the service.

    ``unix:///path`` names a Unix socket on this machine; ``tcp://host:port`` a TCP service, also on another machine.

    Returns:
        The socket family and its address: the socket path, or ``(host, port)``.

    Raises:
        ValueError: If the endpoint is malformed, or names a Unix socket where Unix sockets are unavailable.
    """
    if isinstance(endpoint, str) and endpoint.startswith("unix://"):
        path = endpoint[len("unix://") :]
        if not hasattr(socket, "AF_UNIX"):
            raise ValueError("Unix socket endpoints need Linux; use tcp://host:port on this platform.")
        if not path.startswith("/") or "\0" in path or len(path.encode()) > _MAX_UNIX_PATH_BYTES:
            raise ValueError(
                f"Cosmos endpoint must have the form {_ENDPOINT_FORMS}, with a socket path of at most "
                f"{_MAX_UNIX_PATH_BYTES} bytes."
            )
        return socket.AF_UNIX, path
    try:
        parts = urlsplit(endpoint)
        host, port = parts.hostname, parts.port
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cosmos endpoint must have the form {_ENDPOINT_FORMS}.") from exc
    if (
        parts.scheme != "tcp"
        or not host
        or port is None
        or not 0 < port < 65536
        or parts.username is not None
        or parts.password is not None
        or parts.path
        or parts.query
        or parts.fragment
    ):
        raise ValueError(f"Cosmos endpoint must have the form {_ENDPOINT_FORMS}.")
    return socket.AF_INET, (host, port)


def send_message(sock: socket.socket, metadata: dict, arrays: Sequence[np.ndarray] = ()) -> None:
    """Send validated metadata and uint8 THWC arrays, bounded to one message's size limit."""
    if not isinstance(metadata, dict) or "arrays" in metadata:
        raise ProtocolError("Message metadata must be a dictionary without an 'arrays' field.")
    if metadata.get("version", PROTOCOL_VERSION) != PROTOCOL_VERSION:
        raise ProtocolError("Unsupported Cosmos protocol version.")
    if len(arrays) > MAX_ARRAYS:
        raise ProtocolError("Too many arrays in one Cosmos message.")
    descriptors = []
    body_size = 0
    for array in arrays:
        if (
            not isinstance(array, np.ndarray)
            or array.dtype != np.uint8
            or array.ndim != 4
            or array.shape[-1] != 3
            or any(size <= 0 for size in array.shape)
        ):
            raise ProtocolError("Cosmos arrays must be nonempty uint8 THWC images with three channels.")
        body_size += array.nbytes
        if body_size > MAX_ARRAY_BYTES:
            raise ProtocolError("Cosmos array payload exceeds the message size limit.")
        descriptors.append({"shape": list(array.shape), "dtype": "uint8", "nbytes": array.nbytes})
    envelope = {**metadata, "version": PROTOCOL_VERSION, "arrays": descriptors}
    try:
        encoded = json.dumps(envelope, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProtocolError("Cosmos metadata must be JSON serializable and finite.") from exc
    if not 0 < len(encoded) <= MAX_METADATA_BYTES:
        raise ProtocolError("Cosmos metadata exceeds the message size limit.")
    # One write for the header and metadata, so a small message leaves as one piece.
    sock.sendall(_HEADER.pack(_MAGIC, len(encoded), body_size) + encoded)
    for array in arrays:
        sock.sendall(memoryview(np.ascontiguousarray(array)).cast("B"))


def receive_message(sock: socket.socket) -> tuple[dict, list[np.ndarray]]:
    """Receive a bounded message, validating all descriptors before allocating image buffers."""
    magic, metadata_size, body_size = _HEADER.unpack(_receive_exact(sock, _HEADER.size))
    if magic != _MAGIC:
        raise ProtocolError("Invalid Cosmos protocol header.")
    if not 0 < metadata_size <= MAX_METADATA_BYTES or body_size > MAX_ARRAY_BYTES:
        raise ProtocolError("Cosmos message exceeds the metadata or array size limit.")
    try:
        metadata = json.loads(_receive_exact(sock, metadata_size), parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise ProtocolError("Invalid JSON in Cosmos metadata.") from exc
    if not isinstance(metadata, dict) or type(metadata.get("version")) is not int:
        raise ProtocolError("Cosmos metadata must include an integer protocol version.")
    if metadata["version"] != PROTOCOL_VERSION:
        raise ProtocolError("Unsupported Cosmos protocol version.")
    descriptors = metadata.pop("arrays", None)
    if not isinstance(descriptors, list) or len(descriptors) > MAX_ARRAYS:
        raise ProtocolError("Invalid Cosmos array descriptor list.")
    sizes = []
    shapes = []
    for descriptor in descriptors:
        if not isinstance(descriptor, dict) or set(descriptor) != {"shape", "dtype", "nbytes"}:
            raise ProtocolError("Invalid Cosmos array descriptor.")
        shape = descriptor["shape"]
        size = descriptor["nbytes"]
        if (
            descriptor["dtype"] != "uint8"
            or not isinstance(shape, list)
            or len(shape) != 4
            or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
            or shape[-1] != 3
            or type(size) is not int
            or size != math.prod(shape)
            or size > MAX_ARRAY_BYTES
        ):
            raise ProtocolError("Cosmos arrays must be bounded nonempty uint8 THWC images.")
        sizes.append(size)
        shapes.append(tuple(shape))
    if sum(sizes) != body_size:
        raise ProtocolError("Cosmos array descriptors do not match the payload size.")
    arrays = [
        np.frombuffer(_receive_exact(sock, size), dtype=np.uint8).reshape(shape) for size, shape in zip(sizes, shapes)
    ]
    return metadata, arrays


def request(endpoint: str, metadata: dict, timeout: float = 600.0) -> tuple[dict, list[np.ndarray]]:
    """Perform a one-shot service request, raising on a failed service reply."""
    with connect(endpoint, timeout) as sock:
        send_message(sock, metadata)
        reply, arrays = receive_message(sock)
    check_reply(reply)
    return reply, arrays


def check_reply(reply: dict) -> None:
    """Validate a service result, preserving its diagnostic when an operation failed."""
    if type(reply.get("ok")) is not bool:
        raise ProtocolError("Cosmos reply must include a boolean 'ok' field.")
    if not reply["ok"]:
        error = reply.get("error")
        if not isinstance(error, str):
            raise ProtocolError("Failed Cosmos reply must include an error message.")
        raise RuntimeError(f"Cosmos service rejected the request: {error}")


class ProtocolError(ValueError):
    """Invalid or oversized Cosmos protocol data."""


def _receive_exact(sock: socket.socket, size: int) -> bytearray:
    buffer = bytearray(size)
    view = memoryview(buffer)
    offset = 0
    while offset < size:
        count = sock.recv_into(view[offset:])
        if count == 0:
            raise ConnectionError("Cosmos connection closed before a complete message was received.")
        offset += count
    return buffer


def _reject_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON value {value!r} is not supported.")
