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
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from isaaclab_experimental.cosmos import _cuda_ipc, _protocol
from isaaclab_experimental.cosmos.server.service import serve
from isaaclab_experimental.cosmos.server.world_model import NUM_CONTROLS, Cosmos3Session, handle_world_model_connection

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_neural.models.cosmos3 import Cosmos3Cfg, Cosmos3Model, JointMappingCfg
from isaaclab_neural.physics import WorldModelCfg

from .conftest import write_manifest

pytestmark = pytest.mark.kitless

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
        return torch.full((self.num_envs, T, H, W, 3), len(self.model.controls), dtype=torch.uint8)

    def close(self):
        self.model.closed_sessions += 1


class FakeWorldModel:
    """The service side of a world model, as the Cosmos3 service exposes it."""

    capabilities = {"kind": "world_model", "device": "cpu", "chunk_length": T, "fps": 30, "image_size": [H, W]}

    def __init__(self):
        self.capabilities = {**self.capabilities, "max_envs": None}
        self.resets, self.controls, self.closed_sessions, self.fail = [], [], 0, False

    def warmup(self):
        pass

    def open_session(self, num_envs):
        return FakeSession(self, num_envs)

    def close(self):
        pass


@pytest.fixture
def service():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        endpoint = f"tcp://127.0.0.1:{probe.getsockname()[1]}"
    model = FakeWorldModel()
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
    yield model, endpoint
    _protocol.request(endpoint, {"op": "shutdown"}, timeout=2)
    thread.join(timeout=5)


def start_images(count):
    """The model's one view, whose start image ``i`` is filled with ``100 + i``."""
    return {"camera": (100 + torch.arange(count, dtype=torch.uint8)).view(-1, 1, 1, 1).expand(count, H, W, 3)}


@pytest.fixture
def native_session(tmp_path, monkeypatch):
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


def test_native_reset_replaces_temporary_images_and_close_removes_them(native_session):
    session, pipe = native_session
    images = np.zeros((1, H, W, 3), np.uint8)
    session.reset([0], images, ["first"], [0])
    assert pipe.paths[0].is_file()
    session.reset([0], images, ["second"], [1])
    assert pipe.sessions[0].closed and not pipe.paths[0].parent.exists()
    assert pipe.paths[1].is_file()
    session.close()
    assert pipe.sessions[1].closed and not pipe.paths[1].parent.exists()


@pytest.mark.parametrize("failure", ["open", "observation"])
def test_failed_native_initialization_releases_its_image_and_session(native_session, failure):
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
    (frames,) = model.step(targets, torch.tensor([[1.0, 0, 0], [5.0, 0, 0]])).values()
    assert frames.shape == (2, T, H, W, 3) and frames.dtype == torch.uint8 and (frames == 1).all()
    assert fake.controls[0].shape == (2, T, NUM_CONTROLS) and fake.controls[0][1, 0, 0] == targets[1, 0, 0] + 5
    model.reset([1], start_images(1), ["restart"], [42])
    assert fake.resets[-1] == ([1], [100], ["restart"], [42])
    previous = frames.clone()
    (next_frames,) = model.step(targets, torch.zeros(2, 3)).values()
    assert (next_frames == 2).all() and torch.equal(frames, previous)
    model.close()
    assert fake.closed_sessions == 1


def test_only_one_client_owns_the_model_and_close_allows_the_next_client(service):
    fake, endpoint = service
    first = Cosmos3Model(Cosmos3Cfg(endpoint=endpoint, timeout=5, transport="socket"), 1, "cpu")
    other = Cosmos3Model(Cosmos3Cfg(endpoint=endpoint, timeout=5), 1, "cpu")
    try:
        first.reset([0], start_images(1), ["first"], [0])
        assert first.transport == "socket"
        assert other.spec().chunk_length == T
        with pytest.raises(RuntimeError, match="already serves a session"):
            other.reset([0], start_images(1), ["other"], [1])
        first.close()
        other.reset([0], start_images(1), ["other"], [1])
        assert other.step_controls(np.zeros((1, T, NUM_CONTROLS), np.float32)).shape == (1, T, H, W, 3)
    finally:
        first.close()
        other.close()
    assert fake.closed_sessions == 2


def test_backend_publishes_service_frames_after_collecting_one_action_chunk(service, tmp_path):
    fake, endpoint = service
    physics = WorldModelCfg(
        model=Cosmos3Cfg(endpoint=endpoint, timeout=5, joint_mapping=MappingCfg()),
        reset_manifest=write_manifest(tmp_path, split="validation"),
        split="validation",
    )
    with sim_utils.build_simulation_context(sim_cfg=SimulationCfg(dt=1 / 30, device="cpu", physics=physics)) as sim:
        manager = sim.backend_manager
        manager.bind(2, 3)
        manager.reset_envs(torch.ones(2, dtype=torch.bool))
        assert not fake.resets
        assert manager.frames().frames[:, 0, 0, 0, 0].tolist() == [200, 201]
        for index in range(T):
            manager.joint_target.fill_(index + 1)
            manager.step()
            assert manager.new_frames == (index == T - 1)
        assert len(fake.resets) == len(fake.controls) == 1
        assert fake.resets[0][:3] == ([0, 1], [200, 201], ["prompt 0", "prompt 1"])
        np.testing.assert_allclose(fake.controls[0][1, :, 0], [1.2, 2.2, 3.2, 4.2])
        assert (manager.frames().frames == 1).all()
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
    for _ in range(100):
        if not _protocol.request(endpoint, {"op": "status"}, timeout=1)[0]["session_active"]:
            break
        time.sleep(0.05)
    other = Cosmos3Model(Cosmos3Cfg(endpoint=endpoint, timeout=5), 1, "cpu")
    other.reset([0], start_images(1), ["unfold"], [0])
    assert other.step_controls(np.zeros((1, T, NUM_CONTROLS), np.float32)).shape == (1, T, H, W, 3)
    other.close()


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
        assert model.transport == "cuda_ipc"
        controls = np.zeros((2, T, NUM_CONTROLS), np.float32)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            # Keep the first copy pending while the next request attempts to reuse the service's buffer.
            torch.cuda._sleep(1_000_000_000)
            first = model.step_controls(controls)
        second = model.step_controls(controls)
        stream.synchronize()
        assert first.device == second.device == torch.device("cuda:0")
        assert (first == 1).all(), "The next chunk overwrote frames before the client's copy completed."
        assert (second == 2).all()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(200_000_000)
            third = model.step_controls(controls)
        model.close()
        assert (third == 3).all(), "Closing must finish the copy before unmapping the service's buffer."
        # The client released the buffer before closing, so the service freed it and serves the next session.
        assert not _protocol.request(endpoint, {"op": "status"}, timeout=2)[0]["session_active"]
    finally:
        with suppress(OSError):
            _protocol.request(endpoint, {"op": "shutdown"}, timeout=2)
        service.wait(timeout=30)
