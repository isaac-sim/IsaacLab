# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import subprocess
import sys
from dataclasses import MISSING

import pytest
import torch
import warp as wp

import isaaclab.utils.tensor_processors as tensor_processors
from isaaclab.test.utils import test_devices
from isaaclab.utils import configclass, to_dict
from isaaclab.utils.warp import ProxyArray

pytestmark = pytest.mark.unit


def test_legacy_modifier_imports_warn_and_preserve_aliases():
    """Legacy imports and serialized paths resolve to the canonical processor APIs."""
    script = """
import warnings
from importlib import import_module

from isaaclab.utils.string import string_to_callable

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    from isaaclab.utils import ModifierBase, ModifierCfg, modifiers, tensor_processors
    from isaaclab_experimental.utils import modifiers as warp_modifiers

assert ModifierBase is tensor_processors.TensorProcessor
assert ModifierCfg is tensor_processors.TensorProcessorCfg
for legacy in (modifiers, warp_modifiers):
    processors = import_module(legacy.__name__.replace("modifiers", "tensor_processors"))
    assert import_module(f"{legacy.__name__}.modifier_base").ModifierBase is processors.TensorProcessor
    assert import_module(f"{legacy.__name__}.modifier_cfg").ModifierCfg is processors.TensorProcessorCfg
    assert string_to_callable(f"{legacy.__name__}.modifier:bias") is processors.bias
    exported = {}
    exec(f"from {legacy.__name__} import *", exported)
    assert exported["ModifierBase"] is processors.TensorProcessor
    assert exported["ModifierCfg"] is processors.TensorProcessorCfg
    assert exported["scale"] is processors.scale
    assert any(
        warning.category is DeprecationWarning
        and legacy.__name__ in str(warning.message)
        and "tensor_processors" in str(warning.message)
        for warning in caught
    )
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


@configclass
class TensorProcessorTestCfg:
    """Configuration for testing tensor processors."""

    cfg: tensor_processors.TensorProcessorCfg = MISSING
    init_data: torch.Tensor = MISSING
    result: torch.Tensor = MISSING
    num_iter: int = 10


@pytest.mark.parametrize(
    "func, params, result",
    [
        (tensor_processors.scale, {"multiplier": 2.0}, [2.0, 4.0, 6.0]),
        (tensor_processors.bias, {"value": 1.0}, [2.0, 3.0, 4.0]),
        (tensor_processors.clip, {"bounds": (1.5, 2.5)}, [1.5, 2.0, 2.5]),
        (tensor_processors.clip, {"bounds": (1.5, None)}, [1.5, 2.0, 3.0]),
        (tensor_processors.clip, {"bounds": (None, 2.5)}, [1.0, 2.0, 2.5]),
    ],
    ids=["scale", "bias", "clip", "clip_no_upper_bound", "clip_no_lower_bound"],
)
def test_stateless_tensor_processors(func, params, result):
    """Test the stateless scale, bias, and clip tensor processors."""
    cfg = tensor_processors.TensorProcessorCfg(func=func, params=params)
    output = cfg.func(torch.tensor([1.0, 2.0, 3.0]), **cfg.params)
    assert torch.allclose(output, torch.tensor(result))


class ScaledSumProcessor(tensor_processors.TensorProcessor):
    """Combine named inputs into one persistent tensor without requiring dimensions."""

    def __init__(self, cfg: tensor_processors.TensorProcessorCfg, data: list[dict[str, ProxyArray]]) -> None:
        super().__init__(cfg)
        self.output = torch.empty_like(data[0][cfg.inputs[0]].torch)

    def __call__(self, data: list[dict[str, ProxyArray]]) -> torch.Tensor:
        key = self._cfg.inputs[0]
        torch.add(data[0][key].torch, data[1][key].torch, out=self.output)
        return self.output.mul_(self._cfg.params["multiplier"])


def test_structured_tensor_processor():
    """Structured inputs share the processor lifecycle, configuration, and callable interface."""
    first = torch.tensor([1.0, 2.0, 3.0])
    second = torch.tensor([4.0, 5.0, 6.0])
    data = [{"samples": ProxyArray(wp.from_torch(value))} for value in (first, second)]
    cfg = tensor_processors.TensorProcessorCfg(
        func=ScaledSumProcessor, inputs=["samples"], outputs=["sum"], params={"multiplier": 2.0}
    )
    processor = cfg.func(cfg, data)

    output = processor(data)
    torch.testing.assert_close(output, torch.tensor([10.0, 14.0, 18.0]))
    torch.testing.assert_close(first, torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(second, torch.tensor([4.0, 5.0, 6.0]))

    first.add_(1.0)
    assert processor(data) is output
    torch.testing.assert_close(output, torch.tensor([12.0, 16.0, 20.0]))
    torch.testing.assert_close(first, torch.tensor([2.0, 3.0, 4.0]))
    torch.testing.assert_close(second, torch.tensor([4.0, 5.0, 6.0]))

    processor.reset(env_ids=[0])
    processor.reset()
    processor.close()
    processor.close()
    torch.testing.assert_close(output, torch.tensor([12.0, 16.0, 20.0]))

    restored_cfg = tensor_processors.TensorProcessorCfg(**to_dict(cfg))
    assert isinstance(restored_cfg.func, str)
    assert restored_cfg.inputs == ["samples"]
    assert restored_cfg.outputs == ["sum"]
    restored_processor = restored_cfg.func(restored_cfg, data)
    torch.testing.assert_close(restored_processor(data), torch.tensor([12.0, 16.0, 20.0]))


@pytest.mark.parametrize("device", test_devices())
def test_digital_filter(device):
    """Test digital filter tensor processor."""
    # create test data
    init_data = torch.tensor([0.0, 0.0, 0.0], device=device)
    A = [0.0, 0.1]
    B = [0.5, 0.5]
    result = torch.tensor([-0.45661893, -0.45661893, -0.45661893], device=device)

    # create test config
    test_cfg = TensorProcessorTestCfg(
        cfg=tensor_processors.DigitalFilterCfg(A=A, B=B), init_data=init_data, result=result, num_iter=16
    )

    # create a tensor processor instance
    tensor_processor = test_cfg.cfg.func(test_cfg.cfg, test_cfg.init_data.shape, device=device)

    # test the tensor processor
    theta = torch.tensor([0.0], device=device)
    delta = torch.pi / torch.tensor([8.0, 8.0, 8.0], device=device)

    for _ in range(5):
        # reset the tensor processor
        tensor_processor.reset()

        # apply the tensor processor multiple times
        for i in range(test_cfg.num_iter):
            data = torch.sin(theta + i * delta)
            processed_data = tensor_processor(data)

            assert data.shape == processed_data.shape, "Modified data shape does not equal original"

        # check if the modified data is close to the expected result
        torch.testing.assert_close(processed_data, test_cfg.result)


@pytest.mark.parametrize("device", test_devices())
def test_integral(device):
    """Test integral tensor processor."""
    # create test data
    init_data = torch.tensor([0.0], device=device)
    dt = 1.0
    result = torch.tensor([12.5], device=device)

    # create test config
    test_cfg = TensorProcessorTestCfg(
        cfg=tensor_processors.IntegratorCfg(dt=dt),
        init_data=init_data,
        result=result,
        num_iter=6,
    )

    # create a tensor processor instance
    tensor_processor = test_cfg.cfg.func(test_cfg.cfg, test_cfg.init_data.shape, device=device)

    # test the tensor processor
    delta = torch.tensor(1.0, device=device)

    for _ in range(5):
        # reset the tensor processor
        tensor_processor.reset()

        # clone the data to avoid modifying the original
        data = test_cfg.init_data.clone()
        # apply the tensor processor multiple times
        for _ in range(test_cfg.num_iter):
            processed_data = tensor_processor(data)
            data = data + delta

            assert data.shape == processed_data.shape, "Modified data shape does not equal original"

        # check if the modified data is close to the expected result
        torch.testing.assert_close(processed_data, test_cfg.result)
