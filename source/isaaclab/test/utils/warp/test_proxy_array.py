# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the ProxyArray class."""

import operator
import warnings

import pytest
import torch
import warp as wp

from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils import warp as warp_utils
from isaaclab.utils.warp.proxy_array import ProxyArray

pytestmark = pytest.mark.unit


@pytest.fixture(params=test_devices())
def device(request):
    return request.param


@pytest.fixture
def warn_once(monkeypatch):
    """Arm the one-shot deprecation warning so the test observes it."""
    monkeypatch.setattr(ProxyArray, "_deprecation_warned", False)


@pytest.fixture
def silent(monkeypatch):
    """Mark the deprecation bridge as already warned so tests only check values."""
    monkeypatch.setattr(ProxyArray, "_deprecation_warned", True)


def _proxy(values, device="cpu") -> ProxyArray:
    return ProxyArray(wp.array(values, dtype=wp.float32, device=device))


def _recorded_warnings(category):
    return warnings.catch_warnings(record=True), category


def test_cached_tensor_shares_warp_storage(device):
    arr = wp.zeros((7, 3), dtype=wp.float32, device=device)
    ta = ProxyArray(arr)
    assert ta.warp is arr
    assert ta.torch is ta.torch
    ta.torch[0, 0] = 42.0
    assert arr.numpy()[0, 0] == 42.0
    # convenience accessors delegate to the warp array
    assert (ta.shape, ta.dtype, ta.device, len(ta)) == ((7, 3), wp.float32, arr.device, 7)
    assert repr(ta) == f"ProxyArray(shape=(7, 3), dtype={wp.float32}, device={arr.device})"


def test_immutable_and_rejects_non_warp_input(device):
    ta = ProxyArray(wp.zeros(2, dtype=wp.float32, device=device))
    with pytest.raises(AttributeError, match="immutable"):
        ta._warp = wp.ones(2, dtype=wp.float32, device=device)
    with pytest.raises(AttributeError, match="immutable"):
        ta.new_field = 42
    with pytest.raises(TypeError, match="expects a warp.array"):
        ProxyArray(ta)


@pytest.mark.parametrize(
    ("dtype", "shape", "components"),
    [(wp.vec3f, (8,), 3), (wp.quatf, (8,), 4), (wp.transformf, (8,), 7), (wp.spatial_vectorf, (4, 5), 6)],
)
def test_structured_dtypes_expand_trailing_dim(device, dtype, shape, components):
    assert ProxyArray(wp.zeros(shape, dtype=dtype, device=device)).torch.shape == (*shape, components)


@pytest.mark.parametrize("cuda_device", test_devices(DeviceScope.CUDA))
def test_array_interfaces_follow_device(cuda_device):
    """``wp.launch`` consumes a ProxyArray through the array interface of its device."""

    @wp.kernel
    def _add_one(src: wp.array(dtype=wp.float32), dst: wp.array(dtype=wp.float32)):
        i = wp.tid()
        dst[i] = src[i] + 1.0

    src = ProxyArray(wp.zeros(5, dtype=wp.float32, device=cuda_device))
    dst = ProxyArray(wp.zeros(5, dtype=wp.float32, device=cuda_device))
    assert src.__cuda_array_interface__["shape"] == (5,)
    wp.launch(_add_one, dim=5, inputs=[src], outputs=[dst], device=cuda_device)
    wp.synchronize_device(cuda_device)
    assert dst.torch.tolist() == [1.0] * 5

    cpu = ProxyArray(wp.zeros(3, dtype=wp.float32, device="cpu"))
    assert cpu.__array_interface__["shape"] == (3,)
    with pytest.raises(AttributeError):
        _ = cpu.__cuda_array_interface__


@pytest.mark.parametrize(
    ("env_value", "dtype", "expect_warning"),
    [(None, wp.quatf, False), ("0", wp.quatf, False), ("1", wp.vec3f, False), ("1", wp.quatf, True)],
)
def test_quatf_torch_access_warning_is_opt_in(device, monkeypatch, env_value, dtype, expect_warning):
    """Only ``WARN_ON_TORCH_QUATF_ACCESS=1`` on a quatf array reports the reading call site."""
    if env_value is None:
        monkeypatch.delenv("WARN_ON_TORCH_QUATF_ACCESS", raising=False)
    else:
        monkeypatch.setenv("WARN_ON_TORCH_QUATF_ACCESS", env_value)
    ta = ProxyArray(wp.zeros(4, dtype=dtype, device=device))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = ta.torch  # the reported call site
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == int(expect_warning)
    if expect_warning:
        assert "(x, y, z, w)" in str(user_warnings[0].message)
        assert user_warnings[0].filename == __file__


def test_implicit_torch_usage_warns_once(device, warn_once):
    ta = ProxyArray(wp.ones(5, dtype=wp.float32, device=device))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        total = torch.sum(ta)
        joined = torch.cat([ta, ta])
        mixed = torch.ones(5, device=device) + ta
        scaled = ta * 2.0
        first = ta[0]
    assert [w.category for w in caught] == [DeprecationWarning]
    assert total.item() == 5.0 and joined.shape == (10,) and first.item() == 1.0
    torch.testing.assert_close(mixed, torch.full((5,), 2.0, device=device))
    torch.testing.assert_close(scaled, torch.full((5,), 2.0, device=device))


@pytest.mark.parametrize(
    "op", [operator.add, operator.sub, operator.mul, operator.truediv, operator.pow], ids=lambda f: f.__name__
)
def test_binary_and_reflected_operators(silent, op):
    ta = _proxy([1.0, 2.0])
    reference = torch.tensor([1.0, 2.0])
    torch.testing.assert_close(op(ta, 2.0), op(reference, 2.0))
    torch.testing.assert_close(op(2.0, ta), op(2.0, reference))
    torch.testing.assert_close(op(ta, _proxy([3.0, 4.0])), op(reference, torch.tensor([3.0, 4.0])))


@pytest.mark.parametrize("op", [operator.neg, operator.pos, operator.abs], ids=lambda f: f.__name__)
def test_unary_operators(silent, op):
    torch.testing.assert_close(op(_proxy([1.0, -2.0])), op(torch.tensor([1.0, -2.0])))


@pytest.mark.parametrize(
    "op", [operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge], ids=lambda f: f.__name__
)
def test_comparison_operators(silent, op):
    assert op(_proxy([1.0, 2.0, 3.0]), 2.0).tolist() == op(torch.tensor([1.0, 2.0, 3.0]), 2.0).tolist()


def test_indexing_reads_and_writes_through(silent):
    wp_arr = wp.zeros((3, 4), dtype=wp.vec3f, device="cpu")
    ta = ProxyArray(wp_arr)
    assert ta[:, 0, :].shape == (3, 3) and ta[0, :, 2].shape == (4,)
    ta[1, 2] = 9.0
    assert wp_arr.numpy()[1, 2].tolist() == [9.0, 9.0, 9.0]
    assert ta[-1, -1].tolist() == [0.0, 0.0, 0.0]


def test_wp_to_torch_shim(monkeypatch):
    """``wp.to_torch`` keeps working for raw arrays and returns the cached view for proxies with one warning."""
    raw = wp.array([1.0, 2.0], dtype=wp.float32, device="cpu")
    view = wp.to_torch(raw, requires_grad=False)
    assert isinstance(view, torch.Tensor) and view.requires_grad is False

    monkeypatch.setattr(warp_utils, "_WP_TO_TORCH_WARNED", False)
    proxy = _proxy([7.0, 8.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = wp.to_torch(proxy)
        second = wp.to_torch(proxy)
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert first is proxy.torch and second is proxy.torch
    assert len(deprecations) == 1 and "ProxyArray" in str(deprecations[0].message)
