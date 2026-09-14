# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless contract tests; CPU tensors exercise control and masks, not CUDA transport."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from isaaclab_contrib.visual_dr import DRFrame, DRObservation, VisualDRRuntime
from isaaclab_contrib.visual_dr.cosmos import CosmosDRCfg, create_cosmos_backend
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
    camera, request = runtime.process.call_args.args
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
