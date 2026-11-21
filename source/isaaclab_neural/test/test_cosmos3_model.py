# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The Cosmos3 client against the world-model service, with a fake model in place of the Cosmos Framework."""

import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from isaaclab_experimental.cosmos import _cuda_ipc, _protocol
from isaaclab_experimental.cosmos.server.service import serve
from isaaclab_experimental.cosmos.server.world_model import NUM_CONTROLS, Cosmos3Session, handle_world_model_connection

from isaaclab.utils import configclass

from isaaclab_neural.models.cosmos3 import Cosmos3Cfg, Cosmos3Model, JointMappingCfg

T, H, W = 4, 6, 8


class Mapping:
    def __init__(self, cfg):
        pass

    def encode(self, joint_targets, start_joint_state):
        controls = np.zeros((len(joint_targets), NUM_CONTROLS), np.float32)
        controls[:, 0] = joint_targets[:, 0] + start_joint_state[0]
        return controls


@configclass
class MappingCfg(JointMappingCfg):
    class_type: type = Mapping


class FakeSession:
    def __init__(self, model, num_envs):
        self.model, self.num_envs = model, num_envs

    def reset(self, env_ids, images, prompts, seeds):
        self.model.resets.append((env_ids, images[:, 0, 0, 0].tolist(), prompts, seeds))

    def step(self, controls):
        if self.model.fail:
            raise RuntimeError("generation failed")
        self.model.controls.append(controls.copy())
        value = self.model.first_value + len(self.model.controls) - 1
        return torch.full((self.num_envs, T, H, W, 3), value, dtype=torch.uint8)

    def close(self):
        self.model.closed_sessions += 1


class FakeWorldModel:
    """The service side of a world model, as the Cosmos3 service exposes it."""

    capabilities = {"kind": "world_model", "device": "cpu", "chunk_length": T, "fps": 30, "image_size": [H, W]}

    def __init__(self, first_value=1):
        self.capabilities = {**self.capabilities, "max_envs": None}
        self.resets, self.controls, self.closed_sessions, self.fail = [], [], 0, False
        # The value of every pixel of the first chunk, one more for each later chunk.
        self.first_value = first_value

    def warmup(self):
        pass

    def open_session(self, num_envs):
        return FakeSession(self, num_envs)

    def close(self):
        pass


@contextmanager
def fake_service(first_value=1):
    """Serve a fake world model on a free local port; yield it and its endpoint."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        endpoint = f"tcp://127.0.0.1:{probe.getsockname()[1]}"
    model = FakeWorldModel(first_value)
    thread = threading.Thread(
        target=serve, args=(model, endpoint), kwargs={"handler": handle_world_model_connection}, daemon=True
    )
    thread.start()
    for _ in range(100):
        try:
            _protocol.request(endpoint, {"op": "status"}, timeout=1)
            break
        except ConnectionError:
            time.sleep(0.05)
    try:
        yield model, endpoint
    finally:
        _protocol.request(endpoint, {"op": "shutdown"}, timeout=2)
        thread.join(timeout=5)


@pytest.fixture
def service():
    with fake_service() as served:
        yield served


def wait_until_inactive(endpoint):
    """Wait until the service closed its session, which it does after a failure or disconnect."""
    for _ in range(100):
        if not _protocol.request(endpoint, {"op": "status"}, timeout=1)[0]["session_active"]:
            return
        time.sleep(0.05)
    raise AssertionError(f"The service at {endpoint} kept its session.")


def start_images(count):
    """The RGB of the model's one view, whose start image ``i`` is filled with ``100 + i``."""
    rgb = (100 + torch.arange(count, dtype=torch.uint8)).view(-1, 1, 1, 1).expand(count, H, W, 3)
    return {"camera": {"rgb": rgb}}


@pytest.fixture
def native_session(tmp_path, monkeypatch):
    """A service session in the Framework's native session mode, with a fake pipeline that records its start images."""
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))

    class Native:
        closed = False

        @property
        def observation(self):
            if pipe.fail == "observation":
                raise RuntimeError("initial observation failed")
            return torch.zeros(1, 3, 1, H, W)

        def close(self):
            self.closed = True

    class Pipe:
        fail = None

        def __init__(self):
            self.paths, self.sessions = [], []

        def open_bimanual_session(self, **kwargs):
            self.paths.append(Path(kwargs["vision_path"]))
            assert self.paths[-1].is_file()
            if self.fail == "open":
                raise RuntimeError("open failed")
            native = Native()
            self.sessions.append(native)
            return native

    pipe = Pipe()
    settings = SimpleNamespace(mode="session", domain_name="hand_pose", fps=30, resolution="480")
    session = Cosmos3Session(SimpleNamespace(settings=settings, pipe=pipe), 1)
    try:
        yield session, pipe
    finally:
        session.close()


def test_a_native_session_removes_its_start_image_on_reset_and_close(native_session):
    session, pipe = native_session
    images = np.zeros((1, H, W, 3), np.uint8)
    session.reset([0], images, ["first"], [0])
    assert pipe.paths[0].is_file()
    session.reset([0], images, ["second"], [1])
    assert pipe.sessions[0].closed and not pipe.paths[0].parent.exists()
    assert pipe.paths[1].is_file()
    session.close()
    assert pipe.sessions[1].closed and not pipe.paths[1].parent.exists()


def test_window_mode_generates_every_environment_in_one_batch_with_its_own_seed(monkeypatch):
    # A fake Framework: each environment's data batch names it, and its decoded frames hold 10 times its index.
    inference = ModuleType("cosmos_framework.inference.inference")
    inference._merge_data_batches = lambda batches: {"envs": [env for batch in batches for env in batch["envs"]]}
    inference._finalize_data_batch = (
        lambda batch, batch_size, model: batch if len(batch["envs"]) == batch_size else None
    )
    for name in ("cosmos_framework", "cosmos_framework.inference"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, inference.__name__, inference)
    monkeypatch.setattr(Cosmos3Session, "_batch", lambda self, env, controls: {"envs": [env]})
    calls = []

    class Model:
        def generate_samples_from_batch(self, batch, *, seed, n_sample, **kwargs):
            assert len(batch["envs"]) == len(seed) == n_sample
            calls.append((batch["envs"], seed))
            return {"vision": [torch.tensor(float(env)) for env in batch["envs"]]}

        def decode(self, latent):
            return torch.full((1, 3, T + 1, H, W), (float(latent) * 10 + 0.5) / 255 * 2 - 1)

    settings = SimpleNamespace(mode="window", chunk_size=T, guidance=1.0, shift=1.0, num_inference_steps=1)
    session = Cosmos3Session(SimpleNamespace(settings=settings, pipe=SimpleNamespace(model=Model())), 5)
    seeds = [10, 20, 30, 40, 2**32 - 1]
    session.reset(list(range(5)), np.zeros((5, H, W, 3), np.uint8), [""] * 5, seeds)
    for chunk in range(2):
        calls.clear()
        frames = session.step(np.zeros((5, T, NUM_CONTROLS), np.float32))
        # Each environment's seed advances with its own elapsed frames, as when it was generated alone.
        assert calls == [([0, 1, 2, 3, 4], [(seed + chunk * T) % 2**32 for seed in seeds])]
        assert frames.shape == (5, T, H, W, 3)
        assert [int(frame[0, 0, 0, 0]) for frame in frames] == [0, 10, 20, 30, 40]
        assert [int(condition[0, 0, 0, 0]) for condition in session.condition] == [0, 10, 20, 30, 40]


@pytest.mark.parametrize("failure", ["open", "observation"])
def test_a_failed_native_reset_removes_its_start_image_and_session(native_session, failure):
    session, pipe = native_session
    pipe.fail = failure
    with pytest.raises(RuntimeError, match="failed"):
        session.reset([0], np.zeros((1, H, W, 3), np.uint8), ["test"], [0])
    assert not pipe.paths[0].parent.exists()
    assert all(native.closed for native in pipe.sessions)


def test_the_client_reads_the_spec_resets_from_start_images_and_steps_with_mapped_controls(service):
    fake, endpoint = service
    model = Cosmos3Model(Cosmos3Cfg(endpoint=endpoint, timeout=5, joint_mapping=MappingCfg()), 2, "cpu")
    assert (model.spec().chunk_length, model.spec().image_size) == (T, (H, W))
    model.reset([0, 1], start_images(2), ["unfold", "fold"], [3, 4])
    assert fake.resets == [([0, 1], [100, 101], ["unfold", "fold"], [3, 4])]
    targets = torch.arange(2 * T * 3, dtype=torch.float32).reshape(2, T, 3)
    model.write("joint_targets", targets)
    model.write("start_joint_state", torch.tensor([[1.0, 0, 0], [5.0, 0, 0]]))
    frames = model.step()["camera"]["rgb"]
    assert frames.shape == (2, T, H, W, 3) and frames.dtype == torch.uint8 and (frames == 1).all()
    assert fake.controls[0].shape == (2, T, NUM_CONTROLS) and fake.controls[0][1, 0, 0] == targets[1, 0, 0] + 5
    model.close()
    assert fake.closed_sessions == 1


def test_a_failed_step_ends_the_session_and_frees_the_service(service):
    fake, endpoint = service
    model = Cosmos3Model(Cosmos3Cfg(endpoint=endpoint, timeout=5), 1, "cpu")
    model.reset([0], start_images(1), ["unfold"], [0])
    fake.fail = True
    with pytest.raises(RuntimeError, match="generation failed"):
        model.step_controls(np.zeros((1, T, NUM_CONTROLS), np.float32))
    with pytest.raises(RuntimeError, match="closed or failed"):
        model.step_controls(np.zeros((1, T, NUM_CONTROLS), np.float32))
    # The service closes the failed session, so a new client can open one.
    fake.fail = False
    wait_until_inactive(endpoint)
    other = Cosmos3Model(Cosmos3Cfg(endpoint=endpoint, timeout=5), 1, "cpu")
    other.reset([0], start_images(1), ["unfold"], [0])
    assert other.step_controls(np.zeros((1, T, NUM_CONTROLS), np.float32)).shape == (1, T, H, W, 3)
    other.close()


def test_several_services_each_generate_a_contiguous_share_and_a_failure_closes_all():
    with (
        fake_service(first_value=1) as (first, first_endpoint),
        fake_service(first_value=11) as (second, second_endpoint),
    ):
        endpoints = [first_endpoint, second_endpoint]
        with pytest.raises(ValueError, match="at least one environment per service"):
            Cosmos3Model(Cosmos3Cfg(endpoint=endpoints), 1, "cpu")
        model = Cosmos3Model(Cosmos3Cfg(endpoint=endpoints, timeout=5), 5, "cpu")
        model.reset([4, 0, 2], start_images(3), ["e4", "e0", "e2"], [4, 0, 2])
        # Environments 0-2 go to the first service and 3-4 to the second, which numbers them from 0.
        assert first.resets == [([0, 2], [101, 102], ["e0", "e2"], [0, 2])]
        assert second.resets == [([1], [100], ["e4"], [4])]
        controls = np.arange(5, dtype=np.float32)[:, None, None].repeat(T, 1).repeat(NUM_CONTROLS, 2)
        frames = model.step_controls(controls)
        assert first.controls[0][:, 0, 0].tolist() == [0, 1, 2] and second.controls[0][:, 0, 0].tolist() == [3, 4]
        assert [int(frame[0, 0, 0, 0]) for frame in frames] == [1, 1, 1, 11, 11]
        second.fail = True
        with pytest.raises(RuntimeError, match="generation failed"):
            model.step_controls(controls)
        # The first service's session is closed too, since its environments advanced without the others.
        wait_until_inactive(first_endpoint)
        wait_until_inactive(second_endpoint)
        assert first.closed_sessions == 1


def test_cuda_ipc_is_refused_with_its_reason_when_unavailable(service):
    _, endpoint = service
    model = Cosmos3Model(Cosmos3Cfg(endpoint=endpoint, timeout=5, transport="cuda_ipc"), 1, "cpu")
    with pytest.raises(RuntimeError, match="cuda_ipc is unavailable: the simulation does not run on a CUDA device"):
        model.reset([0], start_images(1), ["unfold"], [0])


_GPU_SERVICE = """
import sys

import torch

from isaaclab_experimental.cosmos.server.service import serve
from isaaclab_experimental.cosmos.server.world_model import handle_world_model_connection

T, H, W = 4, 6, 8


class Session:
    chunks = 0

    def reset(self, env_ids, images, prompts, seeds):
        pass

    def step(self, controls):
        Session.chunks += 1
        return torch.full((len(controls), T, H, W, 3), Session.chunks, dtype=torch.uint8, device="cuda:0")

    def close(self):
        pass


class Model:
    capabilities = {
        "kind": "world_model", "device": "cuda:0", "chunk_length": T, "fps": 30, "image_size": [H, W], "max_envs": None
    }

    def warmup(self):
        pass

    def open_session(self, num_envs):
        return Session()

    def close(self):
        pass


serve(Model(), sys.argv[1], handler=handle_world_model_connection)
"""


@pytest.mark.skipif(not (torch.cuda.is_available() and _cuda_ipc.available()), reason="needs CUDA IPC (Linux)")
def test_chunks_reach_the_simulation_device_through_cuda_ipc():
    # CUDA opens an IPC handle only in another process than the one that exported it, so the service runs in its own.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        endpoint = f"tcp://127.0.0.1:{probe.getsockname()[1]}"
    service = subprocess.Popen([sys.executable, "-c", _GPU_SERVICE, endpoint])
    try:
        for _ in range(600):
            try:
                _protocol.request(endpoint, {"op": "status"}, timeout=1)
                break
            except ConnectionError:
                assert service.poll() is None, "the world-model service exited"
                time.sleep(0.1)
        model = Cosmos3Model(Cosmos3Cfg(endpoint=endpoint, timeout=10, transport="cuda_ipc"), 2, "cuda:0")
        model.reset([0, 1], start_images(2), ["unfold", "fold"], [3, 4])
        assert model.transports == ["cuda_ipc"]
        controls = np.zeros((2, T, NUM_CONTROLS), np.float32)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            # Keep the first copy pending while the next request asks the service to overwrite its buffer.
            torch.cuda._sleep(1_000_000_000)
            first = model.step_controls(controls)
        second = model.step_controls(controls)
        stream.synchronize()
        assert first.device == second.device == torch.device("cuda:0")
        assert (first == 1).all(), "The next chunk overwrote the buffer before the client's copy finished."
        assert (second == 2).all()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(200_000_000)
            third = model.step_controls(controls)
        model.close()
        assert (third == 3).all(), "Closing must finish the copy before releasing the service's buffer."
        # The client released the buffer before closing, so the service freed it and serves the next session.
        assert not _protocol.request(endpoint, {"op": "status"}, timeout=2)[0]["session_active"]
    finally:
        with suppress(OSError):
            _protocol.request(endpoint, {"op": "shutdown"}, timeout=2)
        service.wait(timeout=30)
