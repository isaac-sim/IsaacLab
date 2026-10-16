# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contracts for buffer routing, ownership and lifecycle independent of camera rendering."""

from builtins import ExceptionGroup

import pytest
import torch
import warp as wp

from isaaclab.utils.tensor_processors import DictTensorProcessor, DictTensorProcessorCfg
from isaaclab.utils.tensor_processors._pipeline import _DictTensorProcessorPipeline
from isaaclab.utils.warp import ProxyArray

pytestmark = pytest.mark.unit


class _Scale(DictTensorProcessor):
    def __init__(self, cfg, data):
        super().__init__(cfg, data)
        self.outputs[cfg.outputs[0]] = ProxyArray(wp.empty_like(data[cfg.inputs[0]].warp))

    def __call__(self, data):
        torch.mul(
            data[self.cfg.inputs[0]].torch,
            self.cfg.params.get("factor", 2.0),
            out=self.outputs[self.cfg.outputs[0]].torch,
        )
        return dict(self.outputs)


class _Composite(DictTensorProcessor):
    def __init__(self, cfg, data):
        super().__init__(cfg, data)
        color = data["color"]
        self.outputs["color"] = ProxyArray(wp.empty((color.shape[0], 1, 1, 3), dtype=color.dtype, device=color.device))

    def __call__(self, data):
        torch.sub(data["color"].torch[:, :1, :1], data["range"].torch[:, :1, :1], out=self.outputs["color"].torch)
        return dict(self.outputs)


@pytest.fixture
def composite_pipe():
    sources = {
        "raw_color": ProxyArray(wp.zeros((2, 2, 2, 3), dtype=wp.float32, device="cpu")),
        "range": ProxyArray(wp.zeros((2, 2, 2, 1), dtype=wp.float32, device="cpu")),
    }
    steps = [
        DictTensorProcessorCfg(class_type=_Scale, inputs=["raw_color"], outputs=["color"]),
        DictTensorProcessorCfg(
            class_type=_Composite,
            inputs=["color", "range"],
            outputs=["color"],
        ),
    ]
    assert _DictTensorProcessorPipeline.required_source_names(steps, ["color", "labels"]) == {
        "raw_color",
        "range",
        "labels",
    }
    pipe = _DictTensorProcessorPipeline(sources, steps)
    yield pipe
    pipe.close()


def test_ordered_multi_input_operations_preserve_sources_and_reuse_output_storage(composite_pipe):
    """Bindings consume earlier replacements; shape changes and strided inputs preserve ownership."""
    backing = torch.full((2, 2, 2, 4), 3.0)
    color = ProxyArray(wp.from_torch(backing[..., :3]))
    depth = ProxyArray(wp.full((2, 2, 2, 1), 2.0, dtype=wp.float32, device="cpu"))
    labels = ProxyArray(wp.ones((2, 2, 2, 1), dtype=wp.int32, device="cpu"))
    sources = {"raw_color": color, "range": depth, "labels": labels}
    output = composite_pipe(sources)
    snapshot = output["color"].torch.clone()
    destination = output["color"]

    torch.testing.assert_close(snapshot, torch.full((2, 1, 1, 3), 4.0))
    torch.testing.assert_close(backing, torch.full_like(backing, 3.0))
    torch.testing.assert_close(depth.torch, torch.full_like(depth.torch, 2.0))
    assert output["labels"] is labels
    assert output["raw_color"] is color
    assert set(sources) == {"raw_color", "range", "labels"}
    assert composite_pipe.outputs["color"].shape == (2, 1, 1, 3)

    backing.fill_(5.0)
    updated = composite_pipe(sources)
    assert updated["color"] is destination
    torch.testing.assert_close(destination.torch, torch.full_like(snapshot, 8.0))
    torch.testing.assert_close(snapshot, torch.full_like(snapshot, 4.0))
    updated.pop("color")
    assert composite_pipe.outputs["color"] is destination


def test_tensor_adapter_preserves_incoming_transformations_and_returns_snapshots(composite_pipe):
    """The current observation supplies color, auxiliary depth stays read-only, and outputs are independent."""
    original = torch.full((2, 2, 2, 3), 3.0)
    incoming = original * 2.0
    depth = ProxyArray(wp.full((2, 2, 2, 1), 2.0, dtype=wp.float32, device="cpu"))
    kwargs = {"input_name": "raw_color", "output_name": "color", "auxiliary_buffers": {"range": depth}}
    first = composite_pipe.apply_to_tensor(incoming, **kwargs)
    second = composite_pipe.apply_to_tensor(torch.full_like(incoming, 9.0), **kwargs)
    torch.testing.assert_close(first, torch.full((2, 1, 1, 3), 10.0))
    torch.testing.assert_close(second, torch.full_like(first, 16.0))
    torch.testing.assert_close(incoming, torch.full_like(incoming, 6.0))
    torch.testing.assert_close(depth.torch, torch.full_like(depth.torch, 2.0))
    assert first.data_ptr() != second.data_ptr()

    with pytest.raises(ValueError, match="must not replace"):
        composite_pipe.apply_to_tensor(
            incoming,
            input_name="raw_color",
            output_name="color",
            auxiliary_buffers={"raw_color": ProxyArray(wp.from_torch(original)), "range": depth},
        )


def test_tensor_snapshot_accepts_passthrough_buffer_omitted_from_previous_call():
    """Optional pass-through buffers can return after a call that did not supply them."""
    original = ProxyArray(wp.zeros((2,), dtype=wp.float32, device="cpu"))
    pipe = _DictTensorProcessorPipeline({"value": original}, [])
    try:
        pipe({})
        current = ProxyArray(wp.full((2,), 3.0, dtype=wp.float32, device="cpu"))
        snapshot = pipe.as_tensor({"value": current}, output_name="value")
        current.torch.fill_(5.0)
        torch.testing.assert_close(snapshot, torch.full((2,), 3.0))
    finally:
        pipe.close()


@pytest.mark.parametrize("invalid", ["missing", "shape", "dtype"])
def test_invalid_source_buffers_are_rejected_before_processing(invalid):
    """Missing or incompatible external storage must fail before a callback reads it."""
    calls = []
    source = ProxyArray(wp.zeros((2, 1, 1, 1), dtype=wp.float32, device="cpu"))

    class CountCalls(_Scale):
        def __call__(self, data):
            calls.append(True)
            return super().__call__(data)

    pipe = _DictTensorProcessorPipeline(
        {"source": source},
        [DictTensorProcessorCfg(class_type=CountCalls, inputs=["source"], outputs=["result"])],
    )
    try:
        shape = (1, 1, 1, 1) if invalid == "shape" else source.shape
        dtype = wp.uint8 if invalid == "dtype" else wp.float32
        inputs = {} if invalid == "missing" else {"source": ProxyArray(wp.zeros(shape, dtype=dtype, device="cpu"))}
        with pytest.raises(ValueError, match="Missing required|shape, dtype and device"):
            pipe(inputs)
        assert not calls
    finally:
        pipe.close()


@pytest.mark.parametrize("invalid", ["missing_dependency", "output_names", "duplicate_outputs", "output_type"])
def test_invalid_declarations_release_processors(invalid):
    """Construction failure closes each successfully constructed processor in reverse order."""
    closed = []
    source = ProxyArray(wp.zeros((2, 1, 1, 1), dtype=wp.float32, device="cpu"))

    class Tracked(_Scale):
        def __init__(self, cfg, data):
            super().__init__(cfg, data)
            if cfg.params["name"] == "second":
                if invalid == "output_names":
                    self.outputs = {"wrong": next(iter(self.outputs.values()))}
                elif invalid == "output_type":
                    self.outputs[cfg.outputs[0]] = self.outputs[cfg.outputs[0]].warp

        def close(self):
            closed.append(self.cfg.params["name"])

    first = DictTensorProcessorCfg(Tracked, ["source"], ["middle"], {"name": "first"})
    second = DictTensorProcessorCfg(Tracked, ["middle"], ["result"], {"name": "second"})
    if invalid == "missing_dependency":
        second.inputs = ["absent"]
        message = "Missing input buffers for preparation"
    elif invalid == "output_names":
        message = "declared output names"
    elif invalid == "duplicate_outputs":
        second.outputs = ["result", "result"]
        message = "distinct output names"
    else:
        message = "outputs must be ProxyArray"
    with pytest.raises(TypeError if invalid == "output_type" else ValueError, match=message):
        _DictTensorProcessorPipeline({"source": source}, [first, second])
    assert closed == (["second", "first"] if invalid in ("output_names", "output_type") else ["first"])


@pytest.mark.parametrize("close_fails", [False, True], ids=["normal_close", "failing_close"])
def test_reset_selects_batch_entries_and_close_releases_operations_in_reverse_order(close_fails):
    """State belongs to each processor; selected resets and idempotent teardown preserve its lifecycle."""
    closed = []
    source = ProxyArray(wp.zeros((2, 1, 1, 1), dtype=wp.float32, device="cpu"))

    class History(_Scale):
        def __init__(self, cfg, data):
            super().__init__(cfg, data)
            self.history = torch.zeros(data["value"].shape)

        def __call__(self, data):
            self.history.add_(data["value"].torch)
            self.outputs["value"].torch.copy_(self.history)
            return dict(self.outputs)

        def reset(self, env_ids=None):
            self.history[slice(None) if env_ids is None else env_ids] = 0

        def close(self):
            name = self.cfg.params["name"]
            closed.append(name)
            if close_fails and name == "second":
                raise RuntimeError("Processor cleanup failed")

    pipe = _DictTensorProcessorPipeline(
        {"value": source},
        [DictTensorProcessorCfg(History, ["value"], ["value"], {"name": name}) for name in ("first", "second")],
    )
    source = {"value": ProxyArray(wp.ones(source.shape, dtype=wp.float32, device="cpu"))}
    try:
        torch.testing.assert_close(pipe(source)["value"].torch.flatten(), torch.tensor([1.0, 1.0]))
        torch.testing.assert_close(pipe(source)["value"].torch.flatten(), torch.tensor([3.0, 3.0]))
        pipe.reset([0])
        torch.testing.assert_close(pipe(source)["value"].torch.flatten(), torch.tensor([1.0, 6.0]))
        pipe.reset()
        torch.testing.assert_close(pipe(source)["value"].torch.flatten(), torch.tensor([1.0, 1.0]))
    finally:
        if close_fails:
            with pytest.raises(ExceptionGroup, match="Failed to close"):
                pipe.close()
        else:
            pipe.close()
    pipe.close()
    assert closed == ["second", "first"]
    with pytest.raises(RuntimeError, match="closed"):
        pipe(source)
    with pytest.raises(RuntimeError, match="closed"):
        pipe.as_tensor(source, output_name="value")


@pytest.mark.parametrize("replacement", ["missing", "new_storage"])
def test_processor_cannot_replace_published_output_storage(replacement):
    """Camera consumers retain prepared tensors, so missing or replaced outputs must fail."""

    class ReplacesOutput(_Scale):
        def __call__(self, data):
            super().__call__(data)
            if replacement == "missing":
                return {}
            self.outputs["result"] = ProxyArray(wp.empty_like(data["source"].warp))
            return dict(self.outputs)

    source = ProxyArray(wp.ones((2,), dtype=wp.float32, device="cpu"))
    pipe = _DictTensorProcessorPipeline(
        {"source": source}, [DictTensorProcessorCfg(ReplacesOutput, ["source"], ["result"])]
    )
    try:
        with pytest.raises(ValueError, match="persistent named output buffers"):
            pipe({"source": source})
    finally:
        pipe.close()
