# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contracts for buffer routing, ownership and lifecycle independent of camera rendering."""

from builtins import ExceptionGroup
from dataclasses import replace

import pytest
import torch
import warp as wp

from isaaclab.utils.images import ImageBufferSpec, PreparedBufferOperation, RenderBufferPipe, RenderBufferStepCfg
from isaaclab.utils.warp import ProxyArray

pytestmark = pytest.mark.unit


def _prepare_scale(specs, *, factor=2.0):
    def apply(inputs, outputs):
        torch.mul(inputs["value"].torch, factor, out=outputs["value"].torch)

    return PreparedBufferOperation({"value": specs["value"]}, apply)


def _prepare_composite(specs):
    """Combine color and depth, retaining only one spatial sample per batch entry."""
    assert specs["color"].encoding == "scene_linear"
    assert specs["depth"].encoding == "meters"
    shape = (specs["color"].shape[0], 1, 1, 3)

    def apply(inputs, outputs):
        torch.sub(
            inputs["color"].torch[:, :1, :1],
            inputs["depth"].torch[:, :1, :1],
            out=outputs["value"].torch,
        )

    return PreparedBufferOperation({"value": replace(specs["color"], shape=shape)}, apply)


@pytest.fixture
def composite_pipe():
    specs = {
        "raw_color": ImageBufferSpec((2, 2, 2, 3), wp.float32, "cpu", "scene_linear"),
        "range": ImageBufferSpec((2, 2, 2, 1), wp.float32, "cpu", "meters"),
    }
    steps = [
        RenderBufferStepCfg(prepare=_prepare_scale, inputs={"value": "raw_color"}, outputs={"value": "color"}),
        RenderBufferStepCfg(
            prepare=_prepare_composite,
            inputs={"color": "color", "depth": "range"},
            outputs={"value": "color"},
        ),
    ]
    assert RenderBufferPipe.required_source_names(steps, ["color", "labels"]) == {"raw_color", "range", "labels"}
    pipe = RenderBufferPipe(specs, steps)
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
    assert composite_pipe.output_specs["color"].shape == (2, 1, 1, 3)

    backing.fill_(5.0)
    updated = composite_pipe(sources)
    assert updated["color"] is destination
    torch.testing.assert_close(destination.torch, torch.full_like(snapshot, 8.0))
    torch.testing.assert_close(snapshot, torch.full_like(snapshot, 4.0))


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


@pytest.mark.parametrize("invalid", ["missing", "shape", "dtype"])
def test_invalid_source_buffers_are_rejected_before_processing(invalid):
    """Missing or incompatible external storage must fail before a callback reads it."""
    calls = []
    spec = ImageBufferSpec((2, 1, 1, 1), wp.float32, "cpu")

    def prepare(specs):
        return PreparedBufferOperation({"value": specs["value"]}, lambda *_: calls.append(True))

    pipe = RenderBufferPipe(
        {"source": spec},
        [RenderBufferStepCfg(prepare=prepare, inputs={"value": "source"}, outputs={"value": "result"})],
    )
    try:
        shape = (1, 1, 1, 1) if invalid == "shape" else spec.shape
        dtype = wp.uint8 if invalid == "dtype" else wp.float32
        inputs = {} if invalid == "missing" else {"source": ProxyArray(wp.zeros(shape, dtype=dtype, device="cpu"))}
        with pytest.raises(ValueError, match="Missing required|shape, dtype and device"):
            pipe(inputs)
        assert not calls
    finally:
        pipe.close()


@pytest.mark.parametrize("invalid", ["missing_dependency", "output_roles", "duplicate_outputs"])
def test_invalid_bindings_release_prepared_operations(invalid):
    """Construction failure closes owned operations, including one whose outputs could not be bound."""
    closed = []
    spec = ImageBufferSpec((2, 1, 1, 1), wp.float32, "cpu")

    def prepare(specs, *, name):
        return PreparedBufferOperation({"value": specs["value"]}, lambda *_: None, close=lambda: closed.append(name))

    first = RenderBufferStepCfg(
        prepare=prepare, inputs={"value": "source"}, outputs={"value": "middle"}, params={"name": "first"}
    )
    second = RenderBufferStepCfg(
        prepare=prepare, inputs={"value": "middle"}, outputs={"value": "result"}, params={"name": "second"}
    )
    if invalid == "missing_dependency":
        second.inputs = {"value": "absent"}
        message = "Missing image buffer declarations"
    elif invalid == "output_roles":
        second.outputs = {"wrong": "result"}
        message = "output roles"
    else:
        second.outputs = {"value": "result", "other": "result"}
        message = "distinct output names"
    with pytest.raises(ValueError, match=message):
        RenderBufferPipe({"source": spec}, [first, second])
    assert closed == (["second", "first"] if invalid == "output_roles" else ["first"])


@pytest.mark.parametrize("close_fails", [False, True], ids=["normal_close", "failing_close"])
def test_reset_selects_batch_entries_and_close_releases_operations_in_reverse_order(close_fails):
    """State belongs to each operation; selected resets and idempotent teardown preserve its lifecycle."""
    closed = []
    spec = ImageBufferSpec((2, 1, 1, 1), wp.float32, "cpu")

    def prepare(specs, *, name):
        history = torch.zeros(specs["value"].shape)

        def apply(inputs, outputs):
            history.add_(inputs["value"].torch)
            outputs["value"].torch.copy_(history)

        def reset(env_ids):
            history[slice(None) if env_ids is None else env_ids] = 0

        def close():
            closed.append(name)
            if close_fails and name == "second":
                raise RuntimeError("Operation cleanup failed")

        return PreparedBufferOperation({"value": specs["value"]}, apply, reset=reset, close=close)

    pipe = RenderBufferPipe(
        {"value": spec},
        [
            RenderBufferStepCfg(
                prepare=prepare, inputs={"value": "value"}, outputs={"value": "value"}, params={"name": name}
            )
            for name in ("first", "second")
        ],
    )
    source = {"value": ProxyArray(wp.ones(spec.shape, dtype=wp.float32, device="cpu"))}
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
