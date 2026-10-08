# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Socket framing and camera-client ownership of a resident Cosmos service.

The deterministic resource exercises transport and lifetime boundaries without loading model weights.
Image-transfer scheduling and Cosmos generation quality have separate owners.
"""

from __future__ import annotations

import json
import pickle
import socket
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from isaaclab_experimental.cosmos import CosmosModelCfg, CosmosTransferModifierCfg, _protocol
from isaaclab_experimental.cosmos.client import CosmosModel
from isaaclab_experimental.cosmos.server import serve
from isaaclab_experimental.image_transfer import depth_to_control
from isaaclab_experimental.image_transfer import modifier as modifier_module

from isaaclab.utils.modifiers import ModifierCfg, ModifierChain

pytestmark = pytest.mark.unit


class TransportResource:
    """Already-loaded resource whose ownership is observable across a real socket."""

    capabilities = {"max_views": 1, "modalities": ["edge", "depth", "seg"]}

    def __init__(self):
        self.streams = []
        self.closed = threading.Event()
        self.fail = False
        self.calls = []
        self.step_entered = threading.Event()
        self.resume_step = threading.Event()
        self.resume_step.set()

    def warmup(self):
        self.calls.append(("warmup", threading.get_ident()))

    def open_stream(self, **settings):
        self.calls.append(("open", threading.get_ident()))
        stream = TransportStream(self, settings)
        self.streams.append(stream)
        return stream

    def close(self):
        self.calls.append(("model.close", threading.get_ident()))
        self.closed.set()


class TransportStream:
    def __init__(self, resource, settings):
        self.resource = resource
        self.settings = settings
        self.steps = []
        self.episode_prompts = []
        self.closed = threading.Event()

    def step(self, controls, reset_rows, seeds, **episode):
        self.resource.calls.append(("step", threading.get_ident()))
        self.steps.append((controls, reset_rows, seeds))
        self.episode_prompts.append(episode.get("prompt", "<kept>"))
        self.resource.step_entered.set()
        if not self.resource.resume_step.wait(5):
            raise TimeoutError("Test did not release the blocked generation")
        if self.resource.fail:
            raise RuntimeError("generation rejected this chunk")
        return [np.full_like(control, 127) for control in controls]

    def close(self):
        self.resource.calls.append(("stream.close", threading.get_ident()))
        self.closed.set()


@pytest.fixture
def cosmos_service():
    """Start the production listener, then shut it down through its ordinary endpoint."""
    resource = TransportResource()
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    endpoint = f"tcp://127.0.0.1:{port}"
    failures = []

    def run():
        try:
            serve(resource, host="127.0.0.1", port=port, warmup=True)
        except Exception as error:
            failures.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while True:
        if failures:
            raise failures[0]
        try:
            reply, _ = _protocol.request(endpoint, {"op": "status"}, timeout=0.2)
            assert reply["ready"]
            break
        except OSError:
            if time.monotonic() >= deadline:
                pytest.fail("Cosmos service did not become ready")
            time.sleep(0.01)
    try:
        yield SimpleNamespace(resource=resource, endpoint=endpoint)
    finally:
        if thread.is_alive():
            _protocol.request(endpoint, {"op": "shutdown"}, timeout=2)
        thread.join(timeout=5)
        assert not thread.is_alive(), "Cosmos service did not stop"
        assert not failures, failures
        assert resource.closed.is_set()
        assert resource.calls[-1] == ("model.close", resource.calls[0][1])


def _client(endpoint):
    return CosmosModel(CosmosModelCfg(endpoint=endpoint, prompt="A camera view", modality="edge", timeout=2))


def _controls():
    # A noncontiguous input verifies that tensor strides do not corrupt the transmitted image.
    return torch.arange(18, dtype=torch.uint8).reshape(1, 3, 2, 3).transpose(1, 2)


def _receive_packet(metadata, body=b"", *, claimed_body_size=None):
    """Send independently framed bytes, including corrupt envelopes that the sender API rejects."""
    sender, receiver = socket.socketpair()
    encoded = json.dumps(metadata).encode("utf-8")
    body_size = len(body) if claimed_body_size is None else claimed_body_size
    with sender, receiver:
        receiver.settimeout(2)
        sender.sendall(struct.pack("!4sIQ", b"ILCS", len(encoded), body_size) + encoded + body)
        sender.shutdown(socket.SHUT_WR)
        return _protocol.receive_message(receiver)


@pytest.mark.parametrize("metadata_size,body_size", [(64 * 1024 + 1, 0), (1, 256 * 1024 * 1024 + 1)])
def test_wire_rejects_oversized_headers_before_receiving_the_body(metadata_size, body_size):
    sender, receiver = socket.socketpair()
    with sender, receiver:
        sender.sendall(struct.pack("!4sIQ", b"ILCS", metadata_size, body_size))
        sender.shutdown(socket.SHUT_WR)
        with pytest.raises(_protocol.ProtocolError, match="size limit"):
            _protocol.receive_message(receiver)


@pytest.mark.parametrize(
    "descriptor,body_size,error",
    [
        ({"shape": [1, 1, 1, 3], "dtype": "object", "nbytes": 3}, 3, "uint8"),
        ({"shape": [1, 1, 1, 3], "dtype": "uint8", "nbytes": 3}, 6, "payload size"),
    ],
)
def test_wire_rejects_unsafe_or_inconsistent_descriptors(descriptor, body_size, error):
    with pytest.raises(_protocol.ProtocolError, match=error):
        _receive_packet({"version": _protocol.PROTOCOL_VERSION, "arrays": [descriptor]}, claimed_body_size=body_size)


def test_wire_requires_a_complete_binary_image():
    metadata = {
        "version": _protocol.PROTOCOL_VERSION,
        "arrays": [{"shape": [1, 1, 2, 3], "dtype": "uint8", "nbytes": 6}],
    }
    with pytest.raises(ConnectionError, match="complete message"):
        _receive_packet(metadata, b"\x00\x01\x02", claimed_body_size=6)


class PickleTrap:
    """Harmless local marker proving that binary image bytes are never deserialized as pickle."""

    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return Path.write_text, (self.marker, "deserialized")


def test_wire_treats_pickle_bytes_as_pixels_without_execution(tmp_path):
    marker = tmp_path / "pickle-executed"
    payload = pickle.dumps(PickleTrap(marker))
    payload += bytes(-len(payload) % 3)
    metadata = {
        "version": _protocol.PROTOCOL_VERSION,
        "arrays": [{"shape": [1, 1, len(payload) // 3, 3], "dtype": "uint8", "nbytes": len(payload)}],
    }

    _, arrays = _receive_packet(metadata, payload)

    assert arrays[0].dtype == np.uint8
    assert arrays[0].reshape(-1).tolist() == list(payload)
    assert not marker.exists()


def test_camera_client_closes_sessions_and_keeps_the_model_resident(cosmos_service):
    """Warmup and reconnect use one inference thread, while status stays responsive during generation."""
    client = _client(cosmos_service.endpoint)
    controls = _controls()
    stream = client.open_stream(num_views=1, seeds=(7,))
    result = stream.step([controls], reset_rows=(), seeds=())
    resident_stream = cosmos_service.resource.streams[0]

    assert result[0].shape == controls.shape
    assert result[0].dtype == torch.uint8 and result[0].device == controls.device
    assert torch.equal(result[0], torch.full((1, 2, 3, 3), 127, dtype=torch.uint8))
    np.testing.assert_array_equal(resident_stream.steps[0][0][0][0, 0], [[0, 1, 2], [6, 7, 8], [12, 13, 14]])
    assert resident_stream.settings["prompt"] == "A camera view"
    assert resident_stream.settings["height"] == 2 and resident_stream.settings["width"] == 3
    assert resident_stream.settings["seeds"] == (7,)

    cosmos_service.resource.step_entered.clear()
    cosmos_service.resource.resume_step.clear()
    with ThreadPoolExecutor(max_workers=1) as requests:
        pending_step = requests.submit(stream.step, [controls], reset_rows=(0,), seeds=(19,))
        try:
            assert cosmos_service.resource.step_entered.wait(2)
            reply, _ = _protocol.request(cosmos_service.endpoint, {"op": "status"}, timeout=0.5)
            assert reply["ready"]
            assert not pending_step.done()
        finally:
            cosmos_service.resource.resume_step.set()
        pending_step.result(timeout=2)
    assert resident_stream.steps[-1][1:] == ((0,), (19,))

    client.close()
    client.close()
    assert resident_stream.closed.wait(2)
    assert not cosmos_service.resource.closed.is_set()

    replacement = _client(cosmos_service.endpoint)
    try:
        next_stream = replacement.open_stream(num_views=1, seeds=(23,))
        next_stream.step([controls], reset_rows=(), seeds=())
        assert len(cosmos_service.resource.streams) == 2
        assert not cosmos_service.resource.streams[-1].closed.is_set()
    finally:
        replacement.close()

    assert cosmos_service.resource.calls[0][0] == "warmup"
    assert len({thread_id for _, thread_id in cosmos_service.resource.calls}) == 1
    assert cosmos_service.resource.calls[0][1] != threading.get_ident()


def test_a_prompt_list_cycles_per_episode_over_one_session(cosmos_service):
    """Each finished episode moves to the next prompt; a reset before the first update keeps the prompt."""
    prompts = ["A bright warehouse.", "A wooden kitchen."]
    client = CosmosModel(CosmosModelCfg(endpoint=cosmos_service.endpoint, prompt=prompts, timeout=2))
    stream = client.open_stream(num_views=1, seeds=(7,))
    first, update = _controls(), _controls().expand(4, -1, -1, -1).contiguous()
    stream.step([first], reset_rows=(), seeds=())
    stream.step([first], reset_rows=(0,), seeds=(8,))  # e.g. the environment's initial reset
    stream.step([update], reset_rows=(), seeds=())
    stream.step([first], reset_rows=(0,), seeds=(9,))
    stream.step([update], reset_rows=(), seeds=())
    stream.step([first], reset_rows=(0,), seeds=(10,))
    resident_stream = cosmos_service.resource.streams[0]
    client.close()

    assert resident_stream.settings["prompt"] == "A bright warehouse."
    sent = [prompt for prompt in resident_stream.episode_prompts if prompt != "<kept>"]
    assert sent == ["A bright warehouse.", "A wooden kitchen.", "A bright warehouse."]
    assert len(cosmos_service.resource.streams) == 1


def test_prompt_lists_must_hold_text_and_the_service_changes_prompts_only_at_resets(cosmos_service):
    with pytest.raises(ValueError, match="nonempty"):
        CosmosModel(CosmosModelCfg(prompt=["", "Two"]))
    with pytest.raises(ValueError, match="nonempty"):
        CosmosModel(CosmosModelCfg(prompt=[]))
    with socket.create_connection(_protocol.parse_endpoint(cosmos_service.endpoint), timeout=2) as connection:
        open_request = {
            "op": "open",
            "num_views": 1,
            "seeds": [1],
            "prompt": "A lab.",
            "modality": "edge",
            "height": 2,
            "width": 3,
            "max_episode_frames": 9,
        }
        _protocol.send_message(connection, open_request)
        _protocol.check_reply(_protocol.receive_message(connection)[0])
        _protocol.send_message(
            connection, {"op": "step", "reset_rows": [], "seeds": [], "prompt": "A kitchen."}, [_controls().numpy()]
        )
        reply, _ = _protocol.receive_message(connection)
    assert not reply["ok"] and "only with an episode reset" in reply["error"]


def test_only_one_generation_session_is_owned_and_disconnect_releases_it(cosmos_service):
    """Abrupt disconnect releases the reservation; unrelated status connections remain usable."""
    connection = _protocol.connect(cosmos_service.endpoint, timeout=2)
    _protocol.send_message(
        connection,
        {
            "op": "open",
            "num_views": 1,
            "seeds": [7],
            "prompt": "A camera view",
            "modality": "edge",
            "height": 2,
            "width": 3,
            "max_episode_frames": 201,
        },
    )
    reply, _ = _protocol.receive_message(connection)
    assert reply["ok"]
    first_stream = cosmos_service.resource.streams[0]
    contender = _client(cosmos_service.endpoint)
    try:
        competing_stream = contender.open_stream(num_views=1, seeds=(11,))
        with pytest.raises(RuntimeError, match="(?i)(active|session|busy)"):
            competing_stream.step([_controls()], reset_rows=(), seeds=())
        assert len(cosmos_service.resource.streams) == 1
    finally:
        contender.close()
        connection.close()

    assert first_stream.closed.wait(2)
    reconnect = _client(cosmos_service.endpoint)
    try:
        stream = reconnect.open_stream(num_views=1, seeds=(13,))
        assert stream.step([_controls()], reset_rows=(), seeds=())[0].shape == (1, 2, 3, 3)
    finally:
        reconnect.close()


def test_failed_generation_is_not_retried_and_releases_the_session(cosmos_service):
    cosmos_service.resource.fail = True
    client = _client(cosmos_service.endpoint)
    stream = client.open_stream(num_views=1, seeds=(3,))
    try:
        with pytest.raises(RuntimeError, match="generation rejected this chunk"):
            stream.step([_controls()], reset_rows=(), seeds=())
        resident_stream = cosmos_service.resource.streams[0]
        assert resident_stream.closed.wait(2)
        with pytest.raises(RuntimeError, match="(?i)(failed|closed)"):
            stream.step([_controls()], reset_rows=(), seeds=())
        assert len(resident_stream.steps) == 1
        assert not cosmos_service.resource.closed.is_set()
    finally:
        client.close()


def test_camera_postprocessing_publishes_rgb_from_the_connected_service(cosmos_service, monkeypatch):
    """The public Cosmos modifier plugs into the camera chain and owns only its client session."""
    monkeypatch.setattr(modifier_module, "SimulationContext", SimpleNamespace(instance=lambda: None))
    chain = ModifierChain(
        [
            ModifierCfg(func=depth_to_control, params={"near": 1.0, "far": 9.0}),
            CosmosTransferModifierCfg(
                backend=CosmosModelCfg(endpoint=cosmos_service.endpoint, modality="depth", timeout=2)
            ),
        ],
        device="cpu",
    )
    try:
        generated = chain(torch.tensor([1.0, 9.0]).reshape(1, 1, 2, 1))
        assert generated.shape == (1, 1, 2, 3) and generated.dtype == torch.uint8
        assert generated[0, 0].tolist() == [[127, 127, 127], [127, 127, 127]]
        session = cosmos_service.resource.streams[0]
        assert session.steps[0][0][0][0, 0].tolist() == [[255, 255, 255], [0, 0, 0]]
    finally:
        chain.close()
    assert session.closed.wait(2)
    assert not cosmos_service.resource.closed.is_set()


def test_incompatible_cosmos_cadence_is_rejected_before_opening_a_session(cosmos_service, monkeypatch):
    """A generic one-frame update must not open a Cosmos session that cannot consume it."""
    monkeypatch.setattr(modifier_module, "SimulationContext", SimpleNamespace(instance=lambda: None))
    chain = ModifierChain(
        [
            CosmosTransferModifierCfg(
                backend=CosmosModelCfg(endpoint=cosmos_service.endpoint, timeout=2), update_frames=1
            )
        ],
        device="cpu",
    )
    try:
        with pytest.raises(ValueError, match="Cosmos requires initial_frames=1 and update_frames=4"):
            chain(torch.zeros((1, 1, 2, 3), dtype=torch.uint8))
        assert not cosmos_service.resource.streams
    finally:
        chain.close()
