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
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager, suppress
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_cosmos_server import protocol, serve
from isaaclab_cosmos_server import service as service_module

from isaaclab.utils.tensor_processors import TensorProcessorCfg
from isaaclab.utils.warp import ProxyArray

from isaaclab_contrib.sensors.cosmos import CosmosModel, CosmosModelCfg, CosmosTransferProcessorCfg
from isaaclab_contrib.sensors.cosmos import cosmos_model as cosmos_model_module
from isaaclab_contrib.sensors.cosmos import image_transfer as processor_module
from isaaclab_contrib.sensors.cosmos.cosmos_model import split_views
from isaaclab_contrib.sensors.cosmos.image_transfer_cfg import DepthControlProcessorCfg

pytestmark = pytest.mark.unit

_UNIX = pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="Unix sockets need Linux")


def _unix_endpoint() -> tuple[str, str]:
    """Return a short-path Unix socket endpoint and its directory; pytest's tmp_path can exceed the path limit."""
    directory = tempfile.mkdtemp(prefix="cosmos-", dir="/tmp")
    return f"unix://{directory}/service.sock", directory


def _free_tcp_endpoint() -> str:
    """Return a local TCP endpoint that nothing listens on."""
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        return f"tcp://127.0.0.1:{reserved.getsockname()[1]}"


def _wait_until_ready(endpoint: str, timeout: float = 5.0, poll: float = 0.2, stopped=lambda: None) -> dict:
    """Return the status of the service at ``endpoint`` once it answers.

    Fails when ``stopped`` returns a reason the service cannot start, or when no answer comes in time.
    """
    deadline = time.monotonic() + timeout
    while True:
        if reason := stopped():
            pytest.fail(f"Cosmos service at {endpoint} stopped before it was ready: {reason!r}")
        try:
            return protocol.request(endpoint, {"op": "status"}, timeout=poll)[0]
        except OSError:
            if time.monotonic() >= deadline:
                pytest.fail(f"Cosmos service at {endpoint} did not become ready")
            time.sleep(0.01)


class TransportResource:
    """Already-loaded resource whose ownership is observable across a real socket."""

    capabilities = {}

    def __init__(self):
        self.streams = []
        self.closed = threading.Event()
        self.fail = False
        self.error: type[Exception] = RuntimeError
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

    active = True

    def offload(self):
        self.calls.append(("offload", threading.get_ident()))
        self.active = False

    def activate(self):
        self.calls.append(("activate", threading.get_ident()))
        self.active = True


class TransportStream:
    """Record a session's steps and return gray images, or each control when :attr:`echo` is set."""

    echo = False

    def __init__(self, resource, settings):
        self.resource = resource
        self.settings = settings
        self.steps = []
        self.episode_prompts = []
        self.closed = threading.Event()

    def step(self, controls, reset_rows, seeds, **episode):
        self.resource.calls.append(("step", threading.get_ident()))
        # As the batch stream of the Cosmos server: the session's opening state starts every view.
        if reset_rows and not self.steps and self.settings["num_views"] > 1:
            raise ValueError("Cosmos views start their first episode fresh; reset them after the first step.")
        self.steps.append((controls, reset_rows, seeds))
        self.episode_prompts.append(episode.get("prompt", "<kept>"))
        self.resource.step_entered.set()
        if not self.resource.resume_step.wait(5):
            raise TimeoutError("Test did not release the blocked generation")
        if self.resource.fail:
            raise self.resource.error("generation rejected this chunk")
        return [control.copy() if self.echo else np.full_like(control, 127) for control in controls]

    def close(self):
        self.resource.calls.append(("stream.close", threading.get_ident()))
        self.closed.set()


@pytest.fixture
def cosmos_service(request):
    """Start the production listener on a TCP or Unix socket endpoint, then shut it down through that endpoint."""
    with _running_service(getattr(request, "param", "tcp")) as service:
        yield service


@pytest.fixture
def cosmos_services(request):
    """Independent services, as one service per GPU would be: two, or the parametrized number."""
    with ExitStack() as services:
        yield [services.enter_context(_running_service()) for _ in range(getattr(request, "param", 2))]


@pytest.fixture
def no_sim(monkeypatch):
    """Processors create and own their model, as without a simulation."""
    monkeypatch.setattr(processor_module, "SimulationContext", SimpleNamespace(instance=lambda: None))


@contextmanager
def _running_service(kind: str = "tcp"):
    resource = TransportResource()
    directory = None
    if kind == "unix":
        endpoint, directory = _unix_endpoint()
    else:
        endpoint = _free_tcp_endpoint()
    failures = []

    def run():
        try:
            serve(resource, endpoint, warmup=True)
        except Exception as error:
            failures.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert _wait_until_ready(endpoint, stopped=lambda: failures)["ready"]
    try:
        yield SimpleNamespace(resource=resource, endpoint=endpoint)
    finally:
        if thread.is_alive():
            protocol.request(endpoint, {"op": "shutdown"}, timeout=2)
        thread.join(timeout=5)
        assert not thread.is_alive(), "Cosmos service did not stop"
        assert not failures, failures
        assert resource.closed.is_set()
        assert resource.calls[-1] == ("model.close", resource.calls[0][1])
        if directory is not None:
            assert os.listdir(directory) == [], "the service left its socket file behind"
            shutil.rmtree(directory)


def _client(endpoint):
    return CosmosModel(CosmosModelCfg(max_episode_frames=13, endpoint=endpoint, prompt="A camera view", timeout=2))


def _controls():
    # A noncontiguous input verifies that tensor strides do not corrupt the transmitted image.
    return torch.arange(18, dtype=torch.uint8).reshape(1, 3, 2, 3).transpose(1, 2)


def _proxy(tensor: torch.Tensor) -> ProxyArray:
    return ProxyArray(wp.from_torch(tensor.contiguous()))


def _single_frame_processor(endpoint, num_views=1, *, seed=0, image_size=None, **backend):
    """Return a single-frame Cosmos processor and its camera buffers: controls of 10 and rendered rgb of 50.

    ``backend`` sets further :class:`CosmosModelCfg` fields.
    """
    model = CosmosModelCfg(endpoint=endpoint, generation_mode="single_frame", timeout=2, **backend)
    cfg = CosmosTransferProcessorCfg(backend=model, seed=seed, image_size=image_size)
    cfg.validate_config()
    buffers = {
        "control": _proxy(torch.full((num_views, 1, 2, 3), 10, dtype=torch.uint8)),
        "rgb": _proxy(torch.full((num_views, 1, 2, 3), 50, dtype=torch.uint8)),
    }
    return cfg.func(cfg, buffers), buffers


def _open_request(**fields) -> dict:
    """Return an ``open`` request for one 2 x 3 view, with ``fields`` replacing or adding to its settings."""
    request = {
        "op": "open",
        "num_views": 1,
        "seeds": [1],
        "prompt": None,
        "height": 2,
        "width": 3,
        "max_episode_frames": 9,
        "fps": 30.0,
    }
    return {**request, **fields}


def _receive_packet(metadata, body=b"", *, claimed_body_size=None):
    """Send independently framed bytes, including corrupt envelopes that the sender API rejects."""
    sender, receiver = socket.socketpair()
    encoded = json.dumps(metadata).encode("utf-8")
    body_size = len(body) if claimed_body_size is None else claimed_body_size
    with sender, receiver:
        receiver.settimeout(2)
        sender.sendall(struct.pack("!4sIQ", b"ILCS", len(encoded), body_size) + encoded + body)
        sender.shutdown(socket.SHUT_WR)
        return protocol.receive_message(receiver)


@pytest.mark.parametrize("metadata_size,body_size", [(64 * 1024 + 1, 0), (1, 256 * 1024 * 1024 + 1)])
def test_wire_rejects_oversized_headers_before_receiving_the_body(metadata_size, body_size):
    sender, receiver = socket.socketpair()
    with sender, receiver:
        sender.sendall(struct.pack("!4sIQ", b"ILCS", metadata_size, body_size))
        sender.shutdown(socket.SHUT_WR)
        with pytest.raises(protocol.ProtocolError, match="size limit"):
            protocol.receive_message(receiver)


@pytest.mark.parametrize(
    "descriptor,body_size,error",
    [
        ({"shape": [1, 1, 1, 3], "dtype": "object", "nbytes": 3}, 3, "uint8"),
        ({"shape": [1, 1, 1, 3], "dtype": "uint8", "nbytes": 3}, 6, "payload size"),
    ],
)
def test_wire_rejects_unsafe_or_inconsistent_descriptors(descriptor, body_size, error):
    with pytest.raises(protocol.ProtocolError, match=error):
        _receive_packet({"version": protocol.PROTOCOL_VERSION, "arrays": [descriptor]}, claimed_body_size=body_size)


def test_wire_requires_a_complete_binary_image():
    metadata = {
        "version": protocol.PROTOCOL_VERSION,
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
        "version": protocol.PROTOCOL_VERSION,
        "arrays": [{"shape": [1, 1, len(payload) // 3, 3], "dtype": "uint8", "nbytes": len(payload)}],
    }

    _, arrays = _receive_packet(metadata, payload)

    assert arrays[0].dtype == np.uint8
    assert arrays[0].reshape(-1).tolist() == list(payload)
    assert not marker.exists()


@pytest.mark.parametrize("cosmos_service", ["tcp", pytest.param("unix", marks=_UNIX)], indirect=True)
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
    assert resident_stream.settings["seeds"] == (7,) and resident_stream.settings["fps"] == 30.0
    reply, _ = protocol.request(cosmos_service.endpoint, {"op": "status"}, timeout=0.5)
    assert reply["session_active"]

    cosmos_service.resource.step_entered.clear()
    cosmos_service.resource.resume_step.clear()
    with ThreadPoolExecutor(max_workers=1) as requests:
        pending_step = requests.submit(stream.step, [controls], reset_rows=(0,), seeds=(19,))
        try:
            assert cosmos_service.resource.step_entered.wait(2)
            reply, _ = protocol.request(cosmos_service.endpoint, {"op": "status"}, timeout=0.5)
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
    reply, _ = protocol.request(cosmos_service.endpoint, {"op": "status"}, timeout=0.5)
    assert not reply["session_active"] and reply["capabilities"] == cosmos_service.resource.capabilities

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
    client = CosmosModel(
        CosmosModelCfg(max_episode_frames=13, endpoint=cosmos_service.endpoint, prompt=prompts, timeout=2)
    )
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
        CosmosModel(CosmosModelCfg(max_episode_frames=13, prompt=["", "Two"]))
    with pytest.raises(ValueError, match="nonempty"):
        CosmosModel(CosmosModelCfg(max_episode_frames=13, prompt=[]))
    with protocol.connect(cosmos_service.endpoint, timeout=2) as connection:
        protocol.send_message(connection, _open_request(prompt="A lab."))
        protocol.check_reply(protocol.receive_message(connection)[0])
        protocol.send_message(
            connection, {"op": "step", "reset_rows": [], "seeds": [], "prompt": "A kitchen."}, [_controls().numpy()]
        )
        reply, _ = protocol.receive_message(connection)
    assert not reply["ok"] and "only with an episode reset" in reply["error"]


def test_single_frame_sessions_restart_every_view_with_new_seeds_and_per_view_prompts(cosmos_service):
    """Each image is a one-frame episode; a view's new environment episode moves the view to its next prompt."""
    prompts = ["A bright warehouse.", "A wooden kitchen.", "A night street."]
    client = CosmosModel(
        CosmosModelCfg(endpoint=cosmos_service.endpoint, prompt=prompts, generation_mode="single_frame", timeout=2)
    )
    # Adjacent seeds, as a camera's processor gives them: view v starts from seed + v, each reset adds num_views.
    stream = client.open_stream(num_views=2, seeds=(0, 1))
    one = [_controls(), _controls()]
    stream.step(one, reset_rows=(), seeds=())
    stream.step(one, reset_rows=(0,), seeds=(2,))  # the environment's initial reset keeps view 0's prompt
    stream.step(one, reset_rows=(), seeds=())
    stream.step(one, reset_rows=(0,), seeds=(4,))
    session = cosmos_service.resource.streams[0]
    with pytest.raises(ValueError, match="one frame per view"):
        stream.step([_controls().expand(4, -1, -1, -1).contiguous()] * 2, reset_rows=(), seeds=())
    client.close()

    assert session.settings["max_episode_frames"] == 1
    assert session.settings["prompt"] == prompts[:2]
    image_seed = cosmos_model_module._image_seed
    # The session opens with the first images' seeds, so the first step restarts no view; later steps restart all.
    assert session.settings["seeds"] == (0, 1)
    assert [(rows, seeds) for _, rows, seeds in session.steps] == [
        ((), ()),
        ((0, 1), (2, image_seed(1, 1))),
        ((0, 1), (image_seed(2, 1), image_seed(1, 2))),
        ((0, 1), (4, image_seed(1, 3))),
    ]
    # No image shares another view's or episode's seed, which would give identical images for identical controls.
    used = [*session.settings["seeds"], *(seed for _, _, seeds in session.steps for seed in seeds)]
    assert len(set(used)) == len(used)
    assert session.episode_prompts == ["<kept>", "<kept>", "<kept>", [prompts[1], prompts[1]]]


def test_single_frame_view_alone_on_its_service_takes_its_next_prompt_as_text(cosmos_services):
    """A one-view service session takes one prompt, also when the camera's other views are on another service."""
    first, second = cosmos_services
    prompts = ["A lab.", "A kitchen.", "A field."]
    client = CosmosModel(
        CosmosModelCfg(
            endpoint=[first.endpoint, second.endpoint], prompt=prompts, generation_mode="single_frame", timeout=2
        )
    )
    stream = client.open_stream(num_views=2, seeds=(0, 1))
    one = [_controls(), _controls()]
    stream.step(one, (), ())
    stream.step(one, (1,), (3,))  # the environment's initial reset keeps view 1's prompt
    stream.step(one, (), ())
    stream.step(one, (1,), (5,))
    sessions = [first.resource.streams[0], second.resource.streams[0]]
    client.close()

    assert [session.settings["num_views"] for session in sessions] == [1, 1]
    assert sessions[0].episode_prompts[-1] == "<kept>"
    assert sessions[1].episode_prompts[-1] == prompts[2]


def test_single_frame_settings_are_validated():
    with pytest.raises(ValueError, match="need single_frame"):
        CosmosModel(CosmosModelCfg(max_episode_frames=13, decision_interval=4))
    with pytest.raises(ValueError, match="need autoregressive"):
        CosmosModel(CosmosModelCfg(generation_mode="single_frame", cameras=("left", "right")))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        CosmosModel(CosmosModelCfg(generation_mode="single_frame", augmentation_probability=1.5))
    # Single-frame images are one-frame episodes, so no episode budget is needed.
    CosmosModel(CosmosModelCfg(generation_mode="single_frame")).close()


def test_validation_makes_rgb_an_input_exactly_when_single_frame_can_skip_captures():
    """The inputs follow the generation settings at each validation; autoregressive generation never reads rgb."""
    cfg = CosmosTransferProcessorCfg(backend=CosmosModelCfg(generation_mode="single_frame", decision_interval=2))
    cfg.validate_config()
    assert cfg.inputs == ["control", "rgb"]
    cfg.backend.decision_interval = 1
    cfg.validate_config()
    assert cfg.inputs == ["control"]
    autoregressive = CosmosTransferProcessorCfg(backend=CosmosModelCfg(max_episode_frames=13))
    autoregressive.validate_config()
    assert autoregressive.inputs == ["control"]


@pytest.mark.parametrize("mode,budget", [("single_frame", None), ("autoregressive", 13)])
def test_views_reset_before_the_first_step_open_the_session_with_their_episode_seeds(cosmos_service, mode, budget):
    """The first step's resets become the session's opening seeds, and the step itself restarts no view."""
    cfg = CosmosModelCfg(endpoint=cosmos_service.endpoint, generation_mode=mode, max_episode_frames=budget, timeout=2)
    client = CosmosModel(cfg)
    stream = client.open_stream(num_views=2, seeds=(0, 1))
    stream.step([_controls(), _controls()], reset_rows=(1,), seeds=(7,))
    session = cosmos_service.resource.streams[0]
    client.close()

    assert session.settings["seeds"] == (0, 7)
    assert session.steps[0][1:] == ((), ())


def test_single_frame_processor_generates_at_decision_points_and_publishes_rendered_rgb_otherwise(
    cosmos_service, no_sim
):
    backend = CosmosModelCfg(
        endpoint=cosmos_service.endpoint, generation_mode="single_frame", decision_interval=2, timeout=2
    )
    cfg = CosmosTransferProcessorCfg(backend=backend)
    with pytest.raises(ValueError, match="add 'rgb'"):
        cfg.func(cfg, {"control": _proxy(torch.zeros((1, 1, 2, 3), dtype=torch.uint8))})
    processor, buffers = _single_frame_processor(cosmos_service.endpoint, decision_interval=2)
    mask = wp.ones(1, dtype=wp.bool, device="cpu")
    try:
        published = [int(processor(buffers, env_mask=mask)["rgb"].torch.float().mean()) for _ in range(3)]
        # A reset does not move the schedule: the next capture shows the rendered rgb, never black, and the one
        # after it generates the view's new episode with its new seed.
        processor.reset([0])
        published += [int(processor(buffers, env_mask=mask)["rgb"].torch.float().mean()) for _ in range(2)]
        session = cosmos_service.resource.streams[0]
        assert published == [127, 50, 127, 50, 127]
        second = cosmos_model_module._image_seed(0, 1)
        assert second != 1
        # The session opens with the first image's seed; each later image restarts the view with its own.
        assert session.settings["seeds"] == (0,)
        assert [(rows, seeds) for _, rows, seeds in session.steps] == [((), ()), ((0,), (second,)), ((0,), (1,))]
    finally:
        processor.close()

    processor, buffers = _single_frame_processor(
        cosmos_service.endpoint, decision_interval=2, augmentation_probability=0.0
    )
    try:
        assert int(processor(buffers, env_mask=mask)["rgb"].torch.float().mean()) == 50
    finally:
        processor.close()
    assert len(cosmos_service.resource.streams) == 1


@pytest.mark.parametrize("cosmos_services", [1, 2], indirect=True)
def test_single_frame_processor_publishes_only_the_captured_views(cosmos_services, no_sim):
    """Views a capture does not select keep their image, on one service or split over two.

    This holds at decision points and between them; a seed list gives each view its seed.
    """
    endpoints = [service.endpoint for service in cosmos_services]
    processor, buffers = _single_frame_processor(endpoints, num_views=2, seed=[101, 202], decision_interval=2)

    def capture(*views: bool) -> list[int]:
        mask = wp.array(list(views), dtype=wp.bool, device="cpu")
        return [int(image.float().mean()) for image in processor(buffers, env_mask=mask)["rgb"].torch]

    try:
        assert capture(True, False) == [127, 0]
        assert capture(False, True) == [127, 50]
        assert capture(False, False) == [127, 50]
        sessions = [service.resource.streams[0] for service in cosmos_services]
        assert all(len(session.steps) == 2 for session in sessions)
        assert [seed for session in sessions for seed in session.settings["seeds"]] == [101, 202]
    finally:
        processor.close()


def test_single_frame_processor_draws_the_augmentation_only_at_decision_points(cosmos_service, no_sim, monkeypatch):
    """With an interval and a probability, only a decision point draws; a lost draw publishes the rendered rgb."""
    processor, buffers = _single_frame_processor(
        cosmos_service.endpoint, decision_interval=2, augmentation_probability=0.5
    )
    # Two draws serve four captures: the first decision point loses its draw, the second generates.
    monkeypatch.setattr(processor, "_random", SimpleNamespace(random=iter([0.9, 0.0]).__next__))
    mask = wp.ones(1, dtype=wp.bool, device="cpu")
    try:
        published = [int(processor(buffers, env_mask=mask)["rgb"].torch.float().mean()) for _ in range(4)]
    finally:
        processor.close()
    assert published == [50, 50, 127, 50]
    assert len(cosmos_service.resource.streams[0].steps) == 1


def test_single_frame_processor_publishes_at_the_image_size_and_a_reset_blacks_out_only_its_view(
    cosmos_service, no_sim
):
    processor, buffers = _single_frame_processor(cosmos_service.endpoint, num_views=2, image_size=(1, 1))
    try:
        published = processor(buffers, env_mask=wp.ones(2, dtype=wp.bool, device="cpu"))["rgb"].torch
        assert published.shape == (2, 1, 1, 3) and published.unique().tolist() == [127]
        processor.reset([1])
        assert published[:, 0, 0, 0].tolist() == [127, 0]
    finally:
        processor.close()


@pytest.mark.parametrize(
    "settings,draws,schedule,generations,opening,last",
    [
        # Captures 0 and 2 generate in the first episode; three more episodes start before the next generation.
        ({"decision_interval": 2}, None, "ccc rc rrc", 3, ("A", 0), ("D", (0,), (3,))),
        # Every episode starts with a reset; the first two draw the rendered rgb, the third a generated image.
        ({"augmentation_probability": 0.5}, [0.9, 0.9, 0.0], "rc rc rc", 1, ("C", 3), ("<kept>", (), ())),
    ],
)
def test_single_frame_prompts_and_seeds_follow_episodes_that_generated_no_image(
    cosmos_service, no_sim, monkeypatch, settings, draws, schedule, generations, opening, last
):
    """A view moves one prompt and one seed per episode, also for episodes that showed only the rendered rgb.

    ``schedule`` lists the camera's captures (c) and the view's resets (r) in order. The first generation opens
    the session with its prompt and seed (``opening``); ``last`` is the last step's prompt, resets and seeds.
    """
    processor, buffers = _single_frame_processor(cosmos_service.endpoint, prompt=["A", "B", "C", "D"], **settings)
    if draws:
        monkeypatch.setattr(processor, "_random", SimpleNamespace(random=iter(draws).__next__))
    mask = wp.ones(1, dtype=wp.bool, device="cpu")
    try:
        for event in schedule.replace(" ", ""):
            if event == "r":
                processor.reset([0])
            else:
                processor(buffers, env_mask=mask)
        session = cosmos_service.resource.streams[0]
        assert len(session.steps) == generations
        assert (session.settings["prompt"], session.settings["seeds"]) == (opening[0], (opening[1],))
        assert session.steps[0][1:] == ((), ())
        assert (session.episode_prompts[-1], *session.steps[-1][1:]) == last
    finally:
        processor.close()


def test_offload_and_activate_through_the_session_owner_or_an_idle_service(cosmos_service, monkeypatch):
    from isaaclab_contrib.sensors.cosmos import cli

    endpoint = cosmos_service.endpoint
    assert cli.main(["offload", "--endpoint", endpoint]) == 0
    status, _ = protocol.request(endpoint, {"op": "status"}, timeout=2)
    assert not status["model_active"] and not status["session_active"]
    assert cli.main(["activate", "--endpoint", endpoint]) == 0

    # A learner can keep its session offloaded for longer than the service's idle timeout.
    monkeypatch.setattr(service_module, "_IDLE_TIMEOUT_S", 0.5)
    single = CosmosModel(CosmosModelCfg(endpoint=endpoint, generation_mode="single_frame", timeout=2))
    stream = single.open_stream(num_views=1, seeds=(1,))
    stream.step([_controls()], reset_rows=(), seeds=())
    # Another connection cannot move the weights of a session it does not own.
    assert cli.main(["offload", "--endpoint", endpoint]) == 1
    single.offload()
    time.sleep(1.0)
    single.activate()
    stream.step([_controls()], reset_rows=(), seeds=())
    single.close()

    autoregressive = _client(endpoint)
    stream = autoregressive.open_stream(num_views=1, seeds=(1,))
    stream.step([_controls()], reset_rows=(), seeds=())
    with pytest.raises(RuntimeError, match="Close autoregressive"):
        autoregressive.offload()
    autoregressive.close()
    moves = [name for name, _ in cosmos_service.resource.calls if name in ("offload", "activate")]
    assert moves == ["offload", "activate", "offload", "activate"]


def test_only_one_generation_session_is_owned_and_disconnect_releases_it(cosmos_service):
    """Abrupt disconnect releases the reservation; unrelated status connections remain usable."""
    connection = protocol.connect(cosmos_service.endpoint, timeout=2)
    protocol.send_message(connection, _open_request(seeds=[7], prompt="A camera view", max_episode_frames=13))
    reply, _ = protocol.receive_message(connection)
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


@pytest.mark.parametrize("error", [RuntimeError, ConnectionError])
def test_failed_generation_is_not_retried_and_releases_the_session(cosmos_service, error):
    """The camera receives the model's error, also when it is of a type that the connection itself can raise."""
    cosmos_service.resource.fail = True
    cosmos_service.resource.error = error
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


def test_camera_postprocessing_publishes_rgb_from_the_connected_service(cosmos_service, no_sim):
    """Cosmos consumes masked captures; missing masks cannot start generation or contaminate a later chunk."""
    model = CosmosModelCfg(max_episode_frames=13, endpoint=cosmos_service.endpoint, timeout=2)
    cfgs = [DepthControlProcessorCfg(near=1.0, far=9.0), CosmosTransferProcessorCfg(backend=model)]
    buffers = {"distance_to_image_plane": _proxy(torch.tensor([1.0, 9.0]).reshape(1, 1, 2, 1))}
    processors = []
    mask = wp.ones(1, dtype=wp.bool, device="cpu")
    try:
        # Construct and run in order, as a camera does: each processor reads the outputs of the earlier ones.
        for cfg in cfgs:
            processors.append(cfg.func(cfg, buffers))
            if isinstance(cfg, CosmosTransferProcessorCfg):
                with pytest.raises(ValueError, match="synchronous rendering"):
                    processors[-1](buffers, env_mask=None)
                assert not cosmos_service.resource.streams
            buffers.update(processors[-1](buffers, env_mask=mask))
        generated = buffers["rgb"].torch
        assert generated.shape == (1, 1, 2, 3) and generated.dtype == torch.uint8
        assert generated[0, 0].tolist() == [[127, 127, 127], [127, 127, 127]]
        session = cosmos_service.resource.streams[0]
        assert session.steps[0][0][0][0, 0].tolist() == [[255, 255, 255], [0, 0, 0]]
        assert len(session.steps) == 1
        # A rejected call between chunks must not queue its controls or advance generation.
        transfer = processors[-1]
        buffers["control"].torch.fill_(99)
        with pytest.raises(ValueError, match="synchronous rendering"):
            transfer(buffers, env_mask=None)
        transfer(buffers, env_mask=wp.zeros(1, dtype=wp.bool, device="cpu"))
        buffers["control"].torch.fill_(25)
        for _ in range(3):
            transfer(buffers, env_mask=mask)
        assert len(session.steps) == 1
        transfer(buffers, env_mask=mask)
        assert len(session.steps) == 2
        np.testing.assert_array_equal(session.steps[1][0][0], np.full((4, 1, 2, 3), 25, dtype=np.uint8))
    finally:
        for processor in processors:
            processor.close()
    assert session.closed.wait(2)
    assert not cosmos_service.resource.closed.is_set()


class LazyCamera:
    """A scene camera that captures when its data is read, and only once per simulation step.

    :meth:`render` and :meth:`process` split a capture as an eager scene update does: it renders every camera, then
    runs each camera's processors.
    """

    def __init__(self, cfg, name, value, image_size=None):
        self.processor_cfg = CosmosTransferProcessorCfg(backend=cfg, camera=name, image_size=image_size)
        self.cfg = SimpleNamespace(tensor_processors=[self.processor_cfg], update_period=0.0)
        self.processor = None
        self.value = value
        self.due = False
        self.output = None
        self.capture_mask = wp.ones(2, dtype=wp.bool, device="cpu")
        self.frame = _proxy(torch.zeros(2, dtype=torch.int64))
        self.renders = 0

    @property
    def data(self):
        if self.due:
            self.render()
            self.process()
        return self.output

    def render(self):
        self.frame.torch.add_(1)
        self.renders += 1

    def process(self):
        self.due = False
        buffers = {"control": _proxy(torch.full((2, 2, 3, 3), self.value, dtype=torch.uint8))}
        # A camera constructs its processors at its first capture.
        if self.processor is None:
            self.processor = self.processor_cfg.func(self.processor_cfg, buffers)
        self.output = self.processor(buffers, env_mask=self.capture_mask)["rgb"].torch


def _camera_group(cosmos_service, monkeypatch, image_size=None):
    """A front and a wrist camera of two environments, sharing one session through their simulation's model."""
    cfg = CosmosModelCfg(
        max_episode_frames=13,
        endpoint=cosmos_service.endpoint,
        prompt=["Front 0.", "Front 1.", "Wrist 0.", "Wrist 1."],
        timeout=2,
        cameras=("front", "wrist"),
    )
    model = CosmosModel(cfg)
    sim = SimpleNamespace(get_or_create_backend=lambda backend: model)
    monkeypatch.setattr(processor_module, "SimulationContext", SimpleNamespace(instance=lambda: sim))
    cameras = {name: LazyCamera(cfg, name, value, image_size) for name, value in (("front", 10), ("wrist", 20))}
    monkeypatch.setattr(cosmos_model_module, "_scene_cameras", lambda: cameras)

    # The service echoes each view's control, so every camera can be checked for its own views.
    monkeypatch.setattr(TransportStream, "echo", True)
    return model, cameras


def test_cameras_sharing_a_session_generate_all_their_views_in_one_batch(cosmos_service, monkeypatch):
    """Whichever camera completes its chunk first updates the other; one 4-view step serves both cameras."""
    model, cameras = _camera_group(cosmos_service, monkeypatch)
    front, wrist = cameras["front"], cameras["wrist"]

    def capture(*order):
        front.due = wrist.due = True
        for camera in order:
            camera.data
        front.value, wrist.value = front.value + 1, wrist.value + 1

    try:
        # The front camera completes the initial chunk first, the wrist camera the update.
        capture(front, wrist)
        for _ in range(4):
            capture(wrist, front)
        assert front.output[:, 0, 0, 0].tolist() == [14, 14] and wrist.output[:, 0, 0, 0].tolist() == [24, 24]
        # Environment 1 resets on both cameras and restarts with its newest control.
        for camera in (front, wrist):
            camera.processor.reset([1])
        for _ in range(4):
            capture(front, wrist)
        assert front.output[:, 0, 0, 0].tolist() == [18, 18] and wrist.output[:, 0, 0, 0].tolist() == [28, 28]
    finally:
        model.close()

    assert len(cosmos_service.resource.streams) == 1
    session = cosmos_service.resource.streams[0]
    assert session.settings["num_views"] == 4
    assert session.settings["prompt"] == ["Front 0.", "Front 1.", "Wrist 0.", "Wrist 1."]
    # A camera's view takes its environment's seed, and an environment's reset restarts it on every camera.
    seed = CosmosTransferProcessorCfg().seed
    assert session.settings["seeds"] == (seed, seed + 1, seed, seed + 1)
    assert [len(controls) for controls, _, _ in session.steps] == [4, 4, 4]
    assert [int(control[0, 0, 0, 0]) for control in session.steps[0][0]] == [10, 10, 20, 20]
    assert [control.shape[0] for control in session.steps[2][0]] == [4, 1, 4, 1]
    assert session.steps[2][1:] == ((1, 3), (seed + 3, seed + 3))
    assert session.closed.wait(2)


@pytest.mark.parametrize("image_size", [None, (1, 1)])
def test_cameras_rendered_in_one_eager_update_are_not_rendered_again(cosmos_service, monkeypatch, image_size):
    """An eager scene update renders every camera, then runs their processors; the last camera generates for both.

    Both cameras publish the generating capture's images, also at a size other than the control's.
    """
    model, cameras = _camera_group(cosmos_service, monkeypatch, image_size)
    front, wrist = cameras["front"], cameras["wrist"]

    def eager_update() -> list[int]:
        """Run one update and return what the front camera published at its own processor call."""
        front.due = wrist.due = True
        for camera in (front, wrist):
            camera.render()
        front.process()
        published = front.output[:, 0, 0, 0].tolist()
        wrist.process()
        front.value, wrist.value = front.value + 1, wrist.value + 1
        return published

    try:
        # The front camera's step waits for the wrist camera, so its own call publishes nothing new.
        assert eager_update() == [0, 0]
        # The wrist camera completes each capture and publishes the front camera's images too.
        assert front.output[:, 0, 0, 0].tolist() == [10, 10] and wrist.output[:, 0, 0, 0].tolist() == [20, 20]
        assert [eager_update() for _ in range(4)][-1] == [10, 10]
        assert front.renders == wrist.renders == 5
        assert front.output[:, 0, 0, 0].tolist() == [14, 14] and wrist.output[:, 0, 0, 0].tolist() == [24, 24]
        assert front.output.shape[1:3] == wrist.output.shape[1:3] == (image_size or (2, 3))
    finally:
        model.close()
    assert len(cosmos_service.resource.streams[0].steps) == 2


def test_an_eager_update_needs_the_cosmos_processor_last_on_a_camera_published_later(cosmos_service, monkeypatch):
    """A camera whose images arrive when a later camera generates cannot run a processor after Cosmos."""
    model, cameras = _camera_group(cosmos_service, monkeypatch)
    front, wrist = cameras["front"], cameras["wrist"]
    front.cfg.tensor_processors.append(TensorProcessorCfg(inputs=["rgb"], outputs=["rgb"]))
    try:
        for camera in (front, wrist):
            camera.render()
        with pytest.raises(ValueError, match="'front' runs tensor processors after its Cosmos processor"):
            front.process()
    finally:
        model.close()
    assert not cosmos_service.resource.streams


def test_cameras_sharing_a_session_need_the_same_update_period(cosmos_service, monkeypatch):
    """Different periods fail at the first capture, which the cameras still take together."""
    model, cameras = _camera_group(cosmos_service, monkeypatch)
    front, wrist = cameras["front"], cameras["wrist"]
    wrist.cfg.update_period = 0.1
    try:
        front.due = wrist.due = True
        with pytest.raises(ValueError, match=r"same update period, got \{'front': 0.0, 'wrist': 0.1\}"):
            front.data
    finally:
        model.close()
    assert not cosmos_service.resource.streams


def test_cameras_sharing_a_closed_session_can_open_a_new_one(cosmos_service, monkeypatch):
    """Cameras rebuilt after their shared session closed, for example after a restart, generate in a new session."""
    model, cameras = _camera_group(cosmos_service, monkeypatch)
    front, wrist = cameras["front"], cameras["wrist"]
    try:
        for _ in range(2):
            front.due = wrist.due = True
            front.data
            assert front.output[:, 0, 0, 0].tolist() == [10, 10] and wrist.output[:, 0, 0, 0].tolist() == [20, 20]
            for camera in (front, wrist):
                camera.processor.close()
                camera.processor = None
    finally:
        model.close()
    assert len(cosmos_service.resource.streams) == 2


def test_a_camera_that_misses_a_shared_capture_fails_the_session_for_every_camera(cosmos_service, monkeypatch):
    model, cameras = _camera_group(cosmos_service, monkeypatch)
    front, wrist = cameras["front"], cameras["wrist"]
    try:
        front.due = wrist.due = True
        front.data
        for _ in range(3):
            front.due = True
            front.data
        front.due = True
        with pytest.raises(RuntimeError, match="'wrist' did not capture with the other cameras"):
            front.data
        for _ in range(3):
            wrist.due = True
            wrist.data
        wrist.due = True
        with pytest.raises(RuntimeError, match="shared by these cameras is closed or failed"):
            wrist.data
    finally:
        model.close()
    assert cosmos_service.resource.streams[0].closed.wait(2)


def test_incompatible_cosmos_cadence_is_rejected_before_opening_a_session(cosmos_service, no_sim):
    """A generic one-frame update must not open a Cosmos session that cannot consume it."""
    cfg = CosmosTransferProcessorCfg(
        backend=CosmosModelCfg(max_episode_frames=13, endpoint=cosmos_service.endpoint, timeout=2),
        update_frames=1,
    )
    with pytest.raises(ValueError, match="Cosmos requires initial_frames=1 and update_frames=4"):
        cfg.func(cfg, {"control": _proxy(torch.zeros((1, 1, 2, 3), dtype=torch.uint8))})
    assert not cosmos_service.resource.streams


@pytest.mark.parametrize("window,sink", [(0, 0), (3, 3), (30, -1)])
def test_the_history_window_must_hold_its_attention_sink(window, sink):
    """The window and sink are checked before the model loads, so bad --kv-window settings fail fast."""
    from isaaclab_cosmos_server import CosmosInferenceModel

    with pytest.raises(ValueError, match="attention sink smaller"):
        CosmosInferenceModel("unused-checkpoint", kv_window=window, attention_sink=sink)


@pytest.mark.parametrize("num_views,num_services,shares", [(8, 3, [(0, 3), (3, 6), (6, 8)]), (2, 2, [(0, 1), (1, 2)])])
def test_views_split_into_contiguous_near_equal_shares_with_one_or_more_per_service(num_views, num_services, shares):
    assert [(share.start, share.stop) for share in split_views(num_views, num_services)] == shares
    with pytest.raises(ValueError, match="at least one environment per service"):
        split_views(num_services - 1, num_services)


@pytest.mark.parametrize(
    "cosmos_services,shares,resets",
    [(1, [[0, 1, 2]], [(2,)]), (2, [[0, 1], [2]], [(), (0,)])],
    indirect=["cosmos_services"],
)
def test_views_keep_their_prompts_seeds_frames_and_resets_on_one_service_or_split_over_two(
    cosmos_services, shares, resets
):
    """Each service generates its share of three views as one batch; view 2 restarts with one frame."""
    prompts = ["A lab.", "A kitchen.", "A field."]
    cfg = CosmosModelCfg(
        max_episode_frames=13, endpoint=[service.endpoint for service in cosmos_services], prompt=prompts, timeout=2
    )
    client = CosmosModel(cfg)
    stream = client.open_stream(num_views=3, seeds=(1, 2, 3))
    stream.step([_controls() + view for view in range(3)], (), ())
    update = [(_controls() + view).expand(4, -1, -1, -1).contiguous() for view in range(2)]
    images = stream.step([*update, _controls() + 2], (2,), (9,))
    sessions = [service.resource.streams[0] for service in cosmos_services]
    client.close()

    assert [tuple(view.shape) for view in images] == [(4, 2, 3, 3), (4, 2, 3, 3), (1, 2, 3, 3)]
    for session, views, rows in zip(sessions, shares, resets):
        assert session.settings["num_views"] == len(views)
        assert session.settings["seeds"] == tuple(view + 1 for view in views)
        assert session.settings["prompt"] == [prompts[view] for view in views]
        # Each service receives its own views, and the reset of view 2 names its row on its service.
        assert [int(array[0, 0, 0, 0]) for array in session.steps[0][0]] == views
        assert [array.shape[0] for array in session.steps[-1][0]] == [1 if view == 2 else 4 for view in views]
        assert session.steps[-1][1:] == (rows, (9,) if rows else ())
        # Batched views keep their prompts, also a view alone on its service, so a reset sends none.
        assert session.episode_prompts[-1] == "<kept>"
    assert all(session.closed.wait(5) for session in sessions)


@pytest.mark.parametrize("failing", [0, 1])
def test_a_failing_service_closes_every_session_of_the_camera(cosmos_services, failing):
    """The other service may already have advanced its views, so no session stays open for a retry."""
    cfg = CosmosModelCfg(max_episode_frames=13, endpoint=[service.endpoint for service in cosmos_services], timeout=2)
    client = CosmosModel(cfg)
    stream = client.open_stream(num_views=2, seeds=(1, 2))
    stream.step([_controls(), _controls()], (), ())
    cosmos_services[failing].resource.fail = True
    with pytest.raises(RuntimeError, match="generation rejected this chunk"):
        stream.step([_controls().expand(4, -1, -1, -1).contiguous()] * 2, (), ())
    assert all(service.resource.streams[0].closed.wait(5) for service in cosmos_services)
    with pytest.raises(RuntimeError, match="closed or failed"):
        stream.step([_controls(), _controls()], (), ())
    client.close()


def test_the_service_accepts_only_distinct_ordered_resets_of_opened_views(cosmos_service):
    with protocol.connect(cosmos_service.endpoint, timeout=2) as connection:
        protocol.send_message(connection, _open_request(num_views=2, seeds=[1, 2]))
        protocol.check_reply(protocol.receive_message(connection)[0])
        protocol.send_message(connection, {"op": "step", "reset_rows": [2], "seeds": [5]}, [_controls().numpy()] * 2)
        reply, _ = protocol.receive_message(connection)
    assert not reply["ok"] and "distinct opened views" in reply["error"]


def test_small_messages_leave_in_one_write_and_tcp_sends_them_without_delay(cosmos_service):
    """Header and metadata go out together, and TCP connections disable Nagle, so small steps do not stall."""

    class RecordingSocket:
        def __init__(self):
            self.writes = []

        def sendall(self, data):
            self.writes.append(bytes(data))

    recording = RecordingSocket()
    protocol.send_message(recording, {"op": "step", "frames": 4})
    assert len(recording.writes) == 1
    with protocol.connect(cosmos_service.endpoint, timeout=2) as connection:
        assert connection.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)


def test_a_missing_service_is_reported_with_its_endpoint():
    endpoint = _free_tcp_endpoint()
    with pytest.raises(ConnectionError, match=f"No Cosmos service at {endpoint}"):
        protocol.connect(endpoint, timeout=1)


def test_worker_rejects_an_occupied_endpoint_before_loading_the_model(monkeypatch):
    from isaaclab_cosmos_server import worker

    def load(*args, **kwargs):
        pytest.fail("An occupied endpoint must not load the model")

    monkeypatch.setattr(worker, "CosmosInferenceModel", load)
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        endpoint = f"tcp://127.0.0.1:{occupied.getsockname()[1]}"
        with pytest.raises(OSError):
            worker.main(["--checkpoint", "unused", "--endpoint", endpoint, "--warmup"])


def test_failed_model_loading_releases_the_reserved_endpoint():
    endpoint = _free_tcp_endpoint()

    def load():
        raise ValueError("checkpoint unavailable")

    with pytest.raises(ValueError, match="checkpoint unavailable"):
        serve(load, endpoint)
    with socket.socket() as replacement:
        replacement.bind(protocol.parse_endpoint(endpoint)[1])


def test_endpoints_name_an_absolute_unix_socket_or_a_tcp_host_and_port():
    assert protocol.parse_endpoint("tcp://127.0.0.1:5555") == (socket.AF_INET, ("127.0.0.1", 5555))
    for malformed in ("tcp://127.0.0.1", "http://127.0.0.1:5555", "127.0.0.1:5555"):
        with pytest.raises(ValueError, match="unix:///absolute/path or tcp://host:port"):
            protocol.parse_endpoint(malformed)
    if hasattr(socket, "AF_UNIX"):
        assert protocol.parse_endpoint("unix:///tmp/a.sock") == (socket.AF_UNIX, "/tmp/a.sock")
        assert f"unix:///tmp/isaaclab-cosmos-{os.getuid()}.sock" == protocol.DEFAULT_ENDPOINT
        with pytest.raises(ValueError, match="unix:///absolute/path"):
            protocol.parse_endpoint("unix://relative.sock")
        with pytest.raises(ValueError, match="at most 107 bytes"):
            protocol.parse_endpoint("unix:///" + "a" * 120)
    else:
        assert protocol.DEFAULT_ENDPOINT == "tcp://127.0.0.1:5555"
        with pytest.raises(ValueError, match="need Linux"):
            protocol.parse_endpoint("unix:///tmp/a.sock")


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
        assert _wait_until_ready(endpoint)["ready"]
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
        with pytest.raises(RuntimeError, match="already running"):
            serve(TransportResource(), endpoint)
    finally:
        with suppress(OSError):
            protocol.request(endpoint, {"op": "shutdown"}, timeout=2)
        thread.join(timeout=5)
        shutil.rmtree(directory, ignore_errors=True)
    assert not thread.is_alive() and not os.path.exists(path)


@_UNIX
def test_a_stopping_service_removes_its_socket_file_but_not_a_replacement():
    """A service that took over the endpoint's path keeps its socket file when the earlier service stops."""
    endpoint, directory = _unix_endpoint()
    path = endpoint[len("unix://") :]
    moved = f"unix://{directory}/moved.sock"
    thread = threading.Thread(target=serve, args=(TransportResource(), endpoint), daemon=True)
    thread.start()
    try:
        _wait_until_ready(endpoint)
        os.rename(path, moved[len("unix://") :])
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as replacement:
            replacement.bind(path)
            protocol.request(moved, {"op": "shutdown"}, timeout=2)
            thread.join(timeout=5)
            assert not thread.is_alive() and os.path.exists(path)
    finally:
        shutil.rmtree(directory, ignore_errors=True)
