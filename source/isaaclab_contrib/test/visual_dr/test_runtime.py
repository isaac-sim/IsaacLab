# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless contract tests; CPU tensors exercise control and masks, not CUDA transport."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from isaaclab_contrib.visual_dr import ActionChunkSchedule, DRFrame, DRObservation, VisualDRRuntime
from isaaclab_contrib.visual_dr.cosmos import CosmosDRCfg, CosmosInput, create_cosmos_backend
from isaaclab_contrib.visual_dr.observations import image_runtime_dr, preserve_mask


@pytest.fixture
def frame():
    return DRFrame(
        torch.tensor([[[[10, 20, 30], [40, 50, 60]]]], dtype=torch.uint8),
        torch.ones(1, 1, 2, 1),
        torch.tensor([[[[True], [False]]]]),
    )


def test_skipped_reads_and_lifecycle(frame):
    backend = Mock()
    runtime = VisualDRRuntime(backend)
    runtime.begin(DRObservation(0, 0, 42, consumed=False))
    assert torch.equal(runtime.process("table", frame), frame.rgb)
    backend.generate.assert_not_called()
    runtime.begin(DRObservation(1, 0, 42))
    with pytest.raises(RuntimeError, match="Activate"):
        runtime.process("table", frame)
    runtime.activate()
    runtime.offload()
    runtime.activate()
    with pytest.raises(RuntimeError, match="begin"):
        runtime.process("table", frame)
    runtime.close()
    runtime.close()
    backend.close.assert_called_once()
    with pytest.raises(RuntimeError, match="closed"):
        runtime.activate()


def test_cache_reset_identity_and_foreground(frame, monkeypatch):
    # Bypass ONLY the CUDA boundary to test scheduling/compositing on this CPU host.
    monkeypatch.setattr(DRFrame, "validate", lambda self: None)
    backend = Mock()

    def generate(owned, observation, camera):
        owned.rgb.zero_()
        owned.preserve.zero_()
        return torch.full_like(owned.rgb, 99)

    backend.generate.side_effect = generate
    runtime = VisualDRRuntime(backend)
    runtime.activate()
    context = DRObservation(0, 0, 42)
    runtime.begin(context)
    result = runtime.process("table", frame)
    assert result.tolist() == [[[[10, 20, 30], [99, 99, 99]]]]
    result.zero_()
    runtime.begin(context)
    assert runtime.process("table", frame)[0, 0, 1].tolist() == [99, 99, 99]
    make_frame = Mock(side_effect=AssertionError("Cached reads must not fetch camera signals"))
    assert runtime.read("table", frame.rgb, make_frame)[0, 0, 1].tolist() == [99, 99, 99]
    make_frame.assert_not_called()
    backend.generate.assert_called_once()
    assert frame.rgb[0, 0, 1].tolist() == [40, 50, 60]
    with pytest.raises(ValueError, match="sequence"):
        runtime.begin(DRObservation(0, 1, 42))
    runtime.begin(DRObservation(1, 1, 42))
    runtime.process("table", frame)
    assert backend.generate.call_count == 2
    runtime.process("wrist", frame)
    assert backend.generate.call_count == 3
    runtime.begin(DRObservation(2, 1, 42))
    backend.generate.return_value = None
    backend.generate.side_effect = lambda *args: torch.zeros(1)
    with pytest.raises(ValueError, match="Backend changed"):
        runtime.process("table", frame)


def test_no_cpu_payload_or_implicit_file_backend(frame):
    with pytest.raises(ValueError, match="CUDA"):
        frame.validate()
    with pytest.raises(NotImplementedError, match="tensor-native"):
        create_cosmos_backend(CosmosDRCfg("checkpoint", "background"))


def test_semantics_preserve_unknown_and_foreground():
    segmentation = torch.tensor([[[[8], [3], [99]]]], dtype=torch.int32)
    labels = {"8": {"class": "robot"}, "3": {"class": "ground"}}
    assert preserve_mask(segmentation, labels, ("ground",)).flatten().tolist() == [True, False, True]
    for invalid, mapping in ((segmentation, {}), (segmentation.to(torch.uint8), labels)):
        with pytest.raises(ValueError, match="semantic"):
            preserve_mask(invalid, mapping, ("ground",))


def test_observation_probe_and_attached_runtime(frame):
    output = {
        name: SimpleNamespace(torch=value)
        for name, value in {
            "rgb": frame.rgb,
            "distance_to_image_plane": frame.depth,
            "semantic_segmentation": torch.tensor([[[[8], [3]]]], dtype=torch.int32),
        }.items()
    }
    labels = {"8": {"class": "robot"}, "3": {"class": "ground"}}
    data = SimpleNamespace(output=output, info={"semantic_segmentation": {"idToLabels": labels}})
    env = SimpleNamespace(scene=SimpleNamespace(sensors={"table": SimpleNamespace(data=data)}))
    assert torch.equal(image_runtime_dr(env, "table"), frame.rgb)
    runtime = env.visual_dr_runtime = Mock()
    image_runtime_dr(env, "table")
    camera, rgb, make_frame = runtime.read.call_args.args
    request = make_frame()
    assert camera == "table"
    assert request.rgb is frame.rgb  # extraction uses the tensor view, not a CPU conversion
    assert torch.equal(request.preserve, frame.preserve)


def test_nixl_is_optional_and_linux_only(monkeypatch):
    from isaaclab_contrib.visual_dr import nixl

    monkeypatch.setattr(nixl.sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="Windows"):
        nixl.create_nixl_agent("test")
    monkeypatch.setattr(nixl.sys, "platform", "linux")
    monkeypatch.setitem(nixl.sys.modules, "nixl", None)
    with pytest.raises(ImportError, match=r"isaaclab_contrib\[nixl\]"):
        nixl.create_nixl_agent("test")


def test_chunk_schedule_short_chunks_bootstrap_and_reset():
    schedule = ActionChunkSchedule(4, seed=42)
    with pytest.raises(RuntimeError, match="reset"):
        schedule.after_action()
    initial = schedule.reset()
    assert initial == DRObservation(0, 0, 42)
    assert [schedule.after_action().consumed for _ in range(4)] == [False, False, False, True]
    assert not schedule.after_action().consumed
    assert schedule.after_action(end_chunk=True).consumed
    assert schedule.after_action(bootstrap=True).consumed
    assert [schedule.after_action().consumed for _ in range(3)] == [False, False, True]
    reset = schedule.reset()
    assert reset.episode == 1 and reset.sequence == 11 and reset.consumed
    with pytest.raises(ValueError):
        ActionChunkSchedule(0)


def test_skipped_signals_and_faulted_cleanup(frame):
    backend = Mock()
    runtime = VisualDRRuntime(backend)
    runtime.begin(DRObservation(0, 0, 42, consumed=False))
    data = SimpleNamespace(output={"rgb": SimpleNamespace(torch=frame.rgb)})
    env = SimpleNamespace(
        scene=SimpleNamespace(sensors={"table": SimpleNamespace(data=data)}), visual_dr_runtime=runtime
    )
    assert torch.equal(image_runtime_dr(env, "table"), frame.rgb)  # no auxiliary buffers or labels required
    backend.generate.assert_not_called()
    runtime.activate()
    backend.offload.side_effect = RuntimeError("drain failed")
    with pytest.raises(RuntimeError, match="drain"):
        runtime.offload()
    with pytest.raises(RuntimeError, match="faulted"):
        runtime.activate()
    backend.close.side_effect = [RuntimeError("retry cleanup"), None]
    with pytest.raises(RuntimeError, match="retry"):
        runtime.close()
    runtime.close()
    assert backend.close.call_count == 2


def test_cosmos_tensor_formats_seed_and_output_validation(frame):
    cfg = CosmosDRCfg("checkpoint", "background")
    frame.depth[0, 0, 0, 0] = float("nan")
    frame.depth[0, 0, 1, 0] = -1
    observation = DRObservation(3, 2, 42)
    request = CosmosInput.prepare(frame, observation, "table", cfg)
    assert request.rgb.shape == (1, 3, 1, 1, 2)
    assert request.depth.flatten().tolist() == pytest.approx([2.0, 0.1])
    assert request.preserve.flatten().tolist() == [True, False]
    assert torch.equal(request.decode(request.rgb), frame.rgb)
    assert request.seed == CosmosInput.prepare(frame, observation, "table", cfg).seed
    assert request.seed != CosmosInput.prepare(frame, observation, "wrist", cfg).seed
    for output in (request.rgb.to(torch.uint8), torch.zeros(1), torch.full_like(request.rgb, float("nan"))):
        with pytest.raises(ValueError):
            request.decode(output)


def test_cosmos_model_residency_and_compile_once(frame, monkeypatch):
    # Exercise lifecycle calls with a real CPU module; CUDA movement/synchronization are test doubles.
    model = torch.nn.Linear(1, 1)
    move = Mock(return_value=model)
    monkeypatch.setattr(model, "to", move)
    synchronize = Mock()
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "empty_cache", Mock())
    compile_model = Mock(return_value=model)
    monkeypatch.setattr(torch, "compile", compile_model)
    factory = Mock(return_value=model)
    backend = create_cosmos_backend(CosmosDRCfg("checkpoint", "background", compile=True), model_factory=factory)
    factory.assert_not_called()
    with pytest.raises(RuntimeError, match="Activate"):
        backend.generate(frame, DRObservation(0, 0, 42), "table")
    backend.activate()
    backend.activate()
    assert not model.training and not model.weight.requires_grad
    backend.offload()
    move.assert_called_with("cpu")
    backend.activate()
    factory.assert_called_once()
    compile_model.assert_called_once_with(model, dynamic=False)
    backend.close()
    backend.close()
    assert synchronize.call_count == 2
    with pytest.raises(RuntimeError, match="closed"):
        backend.activate()
    with pytest.raises(NotImplementedError, match="FP8"):
        create_cosmos_backend(CosmosDRCfg("checkpoint", "background", fp8=True), model_factory=factory)


def test_preserved_boundary_padding():
    segmentation = torch.zeros(1, 5, 5, 1, dtype=torch.int32)
    segmentation[0, 2, 2, 0] = 1
    labels = {"0": {"class": "ground"}, "1": {"class": "robot"}}
    mask = preserve_mask(segmentation, labels, ("ground",), boundary_px=1)
    assert mask[0, 1:4, 1:4].all() and mask.sum() == 9


@pytest.mark.parametrize(
    "kwargs",
    [
        {"device": "cpu"},
        {"device": "cuda"},
        {"num_inference_steps": 0},
        {"depth_range_m": (2.0, 0.1)},
        {"depth_range_m": (0.1, float("inf"))},
    ],
)
def test_cosmos_configuration_bounds(kwargs):
    with pytest.raises(ValueError):
        CosmosDRCfg("checkpoint", "background", **kwargs)
