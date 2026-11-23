# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GR00T N1.5 inference client that does not import GR00T or its model dependencies."""

from __future__ import annotations

import io
from types import TracebackType
from typing import cast

import msgpack
import numpy as np
import zmq


class PolicyClient:
    """Connect to ``serve_policy.py`` running in the separate GR00T environment.

    Observations and actions contain NumPy arrays. Each request uses a fresh socket,
    so a timeout cannot leave subsequent requests in ZeroMQ's awaiting-reply state.
    Use as a context manager to release the ZeroMQ context on exit.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5555, timeout_ms: int = 15000) -> None:
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        self.address = f"tcp://{host}:{port}"
        self.timeout_ms = timeout_ms
        self.context = zmq.Context()

    def __enter__(self) -> PolicyClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Release client resources without stopping the separately managed server."""
        self.context.term()

    def _request(self, endpoint: str, data: dict | None = None) -> dict:
        payload = msgpack.packb({"endpoint": endpoint, "data": data}, default=_encode_array, use_bin_type=True)
        with self.context.socket(zmq.REQ) as socket:
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
            socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            socket.connect(self.address)
            try:
                socket.send(payload)
                response = msgpack.unpackb(socket.recv(), object_hook=_decode_array, raw=False)
            except zmq.Again as exc:
                raise TimeoutError(
                    f"GR00T server at {self.address} did not respond within {self.timeout_ms} ms. "
                    "Start serve_policy.py in the GR00T environment and wait for its ready message."
                ) from exc
        if "error" in response:
            raise RuntimeError(f"GR00T server: {response['error']}")
        return response

    def ping(self) -> None:
        """Check that the server is ready; raise on a timeout or server error."""
        self._request("ping")

    def get_action(self, observations: dict) -> dict:
        """Return action arrays for the supplied observation arrays."""
        return self._request("get_action", observations)


def _encode_array(value: object) -> dict[str, bool | bytes]:
    """Encode numeric arrays using the N1.5 server's non-pickle wire format."""
    if not isinstance(value, np.ndarray):
        raise TypeError(f"Unsupported observation type: {type(value).__name__}")
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return {"__ndarray_class__": True, "as_npy": buffer.getvalue()}


def _decode_array(value: dict[str, object]) -> dict[str, object] | np.ndarray:
    if value.get("__ndarray_class__"):
        return np.load(io.BytesIO(cast(bytes, value["as_npy"])), allow_pickle=False)
    return value
