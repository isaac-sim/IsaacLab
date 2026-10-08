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
import os
import pickle
import shutil
import socket
import struct
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
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

_UNIX = pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="Unix sockets need Linux")


def _unix_endpoint() -> tuple[str, str]:
    """Return a short-path Unix socket endpoint and its directory; pytest's tmp_path can exceed the path limit."""
    directory = tempfile.mkdtemp(prefix="cosmos-", dir="/tmp")
    return f"unix://{directory}/service.sock", directory


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


@pytest.fixture(params=["tcp", pytest.param("unix", marks=_UNIX)])
def cosmos_service(request):
    """Start the production listener on a TCP or Unix socket endpoint, then shut it down through that endpoint."""
    resource = TransportResource()
    directory = None
    if request.param == "unix":
        endpoint, directory = _unix_endpoint()
    else:
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            port = reserved.getsockname()[1]
        endpoint = f"tcp://127.0.0.1:{port}"
    failures = []

    def run():
        try:
            serve(resource, endpoint, warmup=True)
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
        if directory is not None:
            assert os.listdir(directory) == [], "the service left its socket file behind"
            shutil.rmtree(directory)


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
    with _protocol.connect(cosmos_service.endpoint, timeout=2) as connection:
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


@pytest.mark.parametrize("cap", [1, 200])
def test_the_service_episode_cap_is_a_setting_of_1_plus_4k_frames(cap):
    """The cap is checked before the model loads, so a bad --max-episode-frames fails fast."""
    from isaaclab_experimental.cosmos.server import CosmosInferenceModel

    with pytest.raises(ValueError, match=r"1 \+ 4\*k"):
        CosmosInferenceModel("unused-checkpoint", max_episode_frames=cap)


@pytest.mark.parametrize(
    "transport,device,capabilities,available,expected",
    [
        ("socket", "cuda:0", {"transports": ["socket", "cuda_ipc"], "pci_bus_id": "0000:01:00.0"}, True, "socket"),
        ("auto", "cpu", {"transports": ["socket", "cuda_ipc"], "pci_bus_id": "0000:01:00.0"}, True, "socket"),
        ("auto", "cuda:0", {"transports": ["socket"]}, True, "socket"),
        ("auto", "cuda:0", {"transports": ["socket", "cuda_ipc"], "pci_bus_id": "0000:02:00.0"}, True, "socket"),
        ("auto", "cuda:0", {"transports": ["socket", "cuda_ipc"], "pci_bus_id": "0000:01:00.0"}, False, "socket"),
        ("auto", "cuda:0", {"transports": ["socket", "cuda_ipc"], "pci_bus_id": "0000:01:00.0"}, True, "cuda_ipc"),
        ("cuda_ipc", "cuda:0", {"transports": ["socket", "cuda_ipc"], "pci_bus_id": "0000:01:00.0"}, True, "cuda_ipc"),
    ],
)
def test_transport_keeps_images_on_the_gpu_only_when_the_service_shares_this_gpu(
    transport, device, capabilities, available, expected, monkeypatch
):
    """Auto uses CUDA IPC only for a local service on the same GPU; the socket is the fallback."""
    from isaaclab_experimental.cosmos import _cuda_ipc

    monkeypatch.setattr(_cuda_ipc, "available", lambda: available)
    monkeypatch.setattr(_cuda_ipc, "pci_bus_id", lambda index: "0000:01:00.0")
    stream = CosmosModel(CosmosModelCfg(transport=transport)).open_stream(num_views=1, seeds=(1,))
    stream._exchange = lambda metadata, arrays=(): ({"ok": True, "capabilities": capabilities}, [])
    assert stream._select_transport(torch.device(device)) == expected


def test_cuda_ipc_handles_keep_all_64_bytes_including_zero_bytes():
    """Driver handles are raw bytes; zero bytes must survive the trip to the other process."""
    from isaaclab_experimental.cosmos import _cuda_ipc

    raw = bytes([7, 0]) + bytes(range(62))
    handle = _cuda_ipc._Handle.from_buffer_copy(raw)
    assert bytes(_cuda_ipc._decode(_cuda_ipc._encode(handle)).reserved) == raw


def test_explicit_cuda_ipc_fails_clearly_when_unavailable_and_remote_services_use_the_socket(monkeypatch):
    from isaaclab_experimental.cosmos import _cuda_ipc

    monkeypatch.setattr(_cuda_ipc, "available", lambda: False)
    stream = CosmosModel(CosmosModelCfg(transport="cuda_ipc")).open_stream(num_views=1, seeds=(1,))
    stream._exchange = lambda metadata, arrays=(): ({"ok": True, "capabilities": {"transports": ["socket"]}}, [])
    with pytest.raises(RuntimeError, match="cuda_ipc is unavailable"):
        stream._select_transport(torch.device("cuda:0"))
    remote = CosmosModel(CosmosModelCfg(endpoint="tcp://10.1.2.3:5555")).open_stream(num_views=1, seeds=(1,))
    assert remote._select_transport(torch.device("cuda:0")) == "socket"


def test_a_service_without_cuda_ipc_rejects_shared_gpu_sessions(cosmos_service):
    reply, _ = _protocol.request(cosmos_service.endpoint, {"op": "status"}, timeout=2)
    assert reply["capabilities"]["transports"] == ["socket"]
    with _protocol.connect(cosmos_service.endpoint, timeout=2) as connection:
        open_request = {
            "op": "open",
            "num_views": 1,
            "seeds": [1],
            "prompt": None,
            "modality": "edge",
            "height": 2,
            "width": 3,
            "max_episode_frames": 9,
            "transport": "cuda_ipc",
            "ipc": {},
        }
        _protocol.send_message(connection, open_request)
        reply, _ = _protocol.receive_message(connection)
    assert not reply["ok"] and "does not offer the cuda_ipc transport" in reply["error"]


_GPU_SERVICE = r"""
import sys
import torch
from isaaclab_experimental.cosmos.server.service import serve


class Stream:
    def step(self, controls, reset_rows, seeds, **episode):
        return [255 - controls[0]]

    def close(self):
        pass


class Model:
    capabilities = {"device": "cuda:0", "max_episode_frames": 201}

    def warmup(self):
        pass

    def open_stream(self, **settings):
        return Stream()

    def close(self):
        pass


serve(Model(), sys.argv[1])
"""


@pytest.mark.skipif(
    not (sys.platform.startswith("linux") and torch.cuda.is_available()), reason="CUDA IPC needs Linux and a GPU"
)
def test_cuda_ipc_round_trip_keeps_controls_and_images_on_the_gpu():
    """A service in another process reads controls and writes images in shared GPU memory; no TCP is involved."""
    import subprocess

    endpoint, directory = _unix_endpoint()
    service = subprocess.Popen([sys.executable, "-c", _GPU_SERVICE, endpoint], env={**os.environ})
    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                _protocol.request(endpoint, {"op": "status"}, timeout=1)
                break
            except OSError:
                if time.monotonic() > deadline or service.poll() is not None:
                    pytest.fail("GPU test service did not start")
                time.sleep(0.2)
        client = CosmosModel(CosmosModelCfg(endpoint=endpoint, transport="cuda_ipc", timeout=10))
        stream = client.open_stream(num_views=1, seeds=(7,))
        first = torch.full((1, 4, 6, 3), 10, dtype=torch.uint8, device="cuda:0")
        update = torch.arange(4 * 4 * 6 * 3, device="cuda:0").remainder(256).to(torch.uint8).view(4, 4, 6, 3)
        images = [stream.step([first], (), ())[0], stream.step([update], (), ())[0]]
        assert stream.transport == "cuda_ipc"
        assert images[0].device == first.device and torch.equal(images[0], 255 - first)
        assert torch.equal(images[1], 255 - update)
        client.close()
    finally:
        with suppress(OSError):
            _protocol.request(endpoint, {"op": "shutdown"}, timeout=2)
        service.wait(timeout=30)
        shutil.rmtree(directory, ignore_errors=True)


def test_endpoints_name_an_absolute_unix_socket_or_a_tcp_host_and_port():
    assert _protocol.parse_endpoint("tcp://127.0.0.1:5555") == (socket.AF_INET, ("127.0.0.1", 5555))
    for malformed in ("tcp://127.0.0.1", "http://127.0.0.1:5555", "127.0.0.1:5555"):
        with pytest.raises(ValueError, match="unix:///absolute/path or tcp://host:port"):
            _protocol.parse_endpoint(malformed)
    if hasattr(socket, "AF_UNIX"):
        assert _protocol.parse_endpoint("unix:///tmp/a.sock") == (socket.AF_UNIX, "/tmp/a.sock")
        assert f"unix:///tmp/isaaclab-cosmos-{os.getuid()}.sock" == _protocol.DEFAULT_ENDPOINT
        with pytest.raises(ValueError, match="unix:///absolute/path"):
            _protocol.parse_endpoint("unix://relative.sock")
        with pytest.raises(ValueError, match="at most 107 bytes"):
            _protocol.parse_endpoint("unix:///" + "a" * 120)
    else:
        assert _protocol.DEFAULT_ENDPOINT == "tcp://127.0.0.1:5555"
        with pytest.raises(ValueError, match="need Linux"):
            _protocol.parse_endpoint("unix:///tmp/a.sock")


@_UNIX
def test_a_unix_socket_service_is_private_replaces_a_stale_socket_and_refuses_a_second_service():
    """Only the owner can open the socket; a stopped service's file is reused; a running service is not replaced."""
    import stat

    endpoint, directory = _unix_endpoint()
    path = endpoint[len("unix://") :]
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stale:
        stale.bind(path)
    resource = TransportResource()
    thread = threading.Thread(target=serve, args=(resource, endpoint), daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while True:
            try:
                reply, _ = _protocol.request(endpoint, {"op": "status"}, timeout=0.2)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    pytest.fail("Cosmos service did not replace the stale socket")
                time.sleep(0.01)
        assert reply["ready"]
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
        with pytest.raises(RuntimeError, match="already running"):
            serve(TransportResource(), endpoint)
    finally:
        with suppress(OSError):
            _protocol.request(endpoint, {"op": "shutdown"}, timeout=2)
        thread.join(timeout=5)
        shutil.rmtree(directory, ignore_errors=True)
    assert not thread.is_alive() and not os.path.exists(path)
