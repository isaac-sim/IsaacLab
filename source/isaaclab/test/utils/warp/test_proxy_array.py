# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the ProxyArray class."""

import warnings

import pytest
import torch
import warp as wp

from isaaclab.test.utils import DeviceScope, test_devices

pytestmark = pytest.mark.unit

wp.config.quiet = True
wp.init()


@pytest.fixture(params=test_devices())
def device(request):
    """Parametrize tests across CPU and CUDA devices."""
    return request.param


cpu_only = pytest.mark.parametrize("device", ["cpu"])
"""Run device-independent wrapper bookkeeping once on CPU instead of on every device."""


class TestProxyArrayBasic:
    """Tests for basic ProxyArray functionality."""

    @cpu_only
    def test_warp_returns_original(self, device):
        """Test that .warp returns the original warp array."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        arr = wp.zeros(10, dtype=wp.float32, device=device)
        ta = ProxyArray(arr)
        assert ta.warp is arr

    @cpu_only
    def test_torch_is_cached(self, device):
        """Test that .torch returns the same tensor object on repeated access."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        arr = wp.zeros(10, dtype=wp.float32, device=device)
        ta = ProxyArray(arr)
        t1 = ta.torch
        t2 = ta.torch
        assert t1 is t2

    def test_torch_shares_memory(self, device):
        """Test that .torch provides a zero-copy view (shares memory with warp)."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        arr = wp.zeros(10, dtype=wp.float32, device=device)
        ta = ProxyArray(arr)
        t = ta.torch
        # Modify the torch tensor
        t[0] = 42.0
        # The change should be visible in the warp array
        arr_np = arr.numpy()
        assert arr_np[0] == 42.0

    @cpu_only
    def test_immutable_warp_cannot_be_reassigned(self, device):
        """ProxyArray._warp cannot be reassigned; callers must construct a new wrapper."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        arr = wp.zeros(10, dtype=wp.float32, device=device)
        ta = ProxyArray(arr)

        with pytest.raises(AttributeError, match="immutable"):
            ta._warp = wp.ones(10, dtype=wp.float32, device=device)
        with pytest.raises(AttributeError, match="immutable"):
            ta.new_field = 42  # arbitrary attribute writes also blocked

    @pytest.mark.parametrize("cuda_device", test_devices(DeviceScope.CUDA))
    def test_cuda_array_interface(self, cuda_device):
        """Test that __cuda_array_interface__ delegates to the underlying warp array."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        arr = wp.zeros(10, dtype=wp.float32, device=cuda_device)
        ta = ProxyArray(arr)
        cai = ta.__cuda_array_interface__
        assert isinstance(cai, dict)
        assert "data" in cai
        assert "shape" in cai
        assert cai["shape"] == arr.__cuda_array_interface__["shape"]

    def test_cuda_array_interface_not_on_cpu(self):
        """Test that __cuda_array_interface__ raises AttributeError on CPU arrays."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        arr = wp.zeros(10, dtype=wp.float32, device="cpu")
        ta = ProxyArray(arr)
        with pytest.raises(AttributeError):
            _ = ta.__cuda_array_interface__

    @pytest.mark.parametrize("cuda_device", test_devices(DeviceScope.CUDA))
    def test_wp_launch_accepts_proxy_array(self, cuda_device):
        """Test that wp.launch() can consume a ProxyArray via __cuda_array_interface__."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        @wp.kernel
        def _add_one(src: wp.array(dtype=wp.float32), dst: wp.array(dtype=wp.float32)):
            i = wp.tid()
            dst[i] = src[i] + 1.0

        src = ProxyArray(wp.zeros(5, dtype=wp.float32, device=cuda_device))
        dst = ProxyArray(wp.zeros(5, dtype=wp.float32, device=cuda_device))
        wp.launch(_add_one, dim=5, inputs=[src], outputs=[dst], device=cuda_device)
        wp.synchronize_device(cuda_device)
        assert dst.torch[0].item() == 1.0
        assert dst.torch[4].item() == 1.0


@cpu_only
class TestProxyArrayStructuredTypes:
    """Tests for ProxyArray with structured warp types (vec3f, quatf, etc)."""

    @pytest.mark.parametrize(
        "dtype, shape, expected_shape",
        [
            (wp.vec3f, 8, (8, 3)),
            (wp.quatf, 8, (8, 4)),
            (wp.transformf, 8, (8, 7)),
            (wp.spatial_vectorf, 8, (8, 6)),
            (wp.vec3f, (4, 5), (4, 5, 3)),
        ],
        ids=["vec3f", "quatf", "transformf", "spatial_vectorf", "2d_vec3f"],
    )
    def test_structured_type_shape(self, device, dtype, shape, expected_shape):
        """Test that structured types expose their components as a trailing torch dimension."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        arr = wp.zeros(shape, dtype=dtype, device=device)
        ta = ProxyArray(arr)
        assert isinstance(ta.torch, torch.Tensor)
        assert ta.torch.shape == expected_shape


@cpu_only
class TestProxyArrayQuatfTorchAccessWarning:
    """Tests for the WARN_ON_TORCH_QUATF_ACCESS opt-in runtime detector."""

    @pytest.mark.parametrize("env_value", [None, "0"], ids=["unset", "zero"])
    def test_default_no_warning(self, device, monkeypatch, env_value):
        """No env var or ``"0"`` → quatf .torch access is silent (only ``"1"`` enables the detector)."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        if env_value is None:
            monkeypatch.delenv("WARN_ON_TORCH_QUATF_ACCESS", raising=False)
        else:
            monkeypatch.setenv("WARN_ON_TORCH_QUATF_ACCESS", env_value)
        ta = ProxyArray(wp.zeros(4, dtype=wp.quatf, device=device))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = ta.torch
            assert [x for x in w if issubclass(x.category, UserWarning)] == []

    def test_env_set_warns_on_quatf(self, device, monkeypatch):
        """WARN_ON_TORCH_QUATF_ACCESS=1 → quatf .torch read emits a UserWarning at the call site."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        monkeypatch.setenv("WARN_ON_TORCH_QUATF_ACCESS", "1")
        ta = ProxyArray(wp.zeros(4, dtype=wp.quatf, device=device))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = ta.torch  # the .torch read on this line is what should be reported
            user_warns = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warns) == 1
            assert "quatf" in str(user_warns[0].message)
            assert "(w, x, y, z)" in str(user_warns[0].message)
            assert "(x, y, z, w)" in str(user_warns[0].message)
            # stacklevel=2 → the warning's filename is this test file, not proxy_array.py
            assert user_warns[0].filename == __file__

    def test_env_set_does_not_warn_on_non_quatf(self, device, monkeypatch):
        """The detector only fires for wp.quatf — float32 / vec3f / transformf are silent."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        monkeypatch.setenv("WARN_ON_TORCH_QUATF_ACCESS", "1")
        for dtype in (wp.float32, wp.vec3f, wp.transformf, wp.spatial_vectorf):
            ta = ProxyArray(wp.zeros(4, dtype=dtype, device=device))
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _ = ta.torch
                assert [x for x in w if issubclass(x.category, UserWarning)] == []


class TestProxyArrayConvenienceProperties:
    """Tests for convenience properties: shape, dtype, device, len, repr."""

    @cpu_only
    def test_convenience_properties(self, device):
        """Test that shape, dtype, device, len(), and repr() forward the warp array's metadata."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        arr = wp.zeros((7, 3), dtype=wp.float32, device=device)
        ta = ProxyArray(arr)
        assert ta.shape == (7, 3)
        assert ta.dtype == wp.float32
        assert ta.device == arr.device
        assert len(ta) == 7
        r = repr(ta)
        assert "ProxyArray" in r
        assert "float32" in r


class TestProxyArrayDeprecationBridge:
    """Tests for the deprecation bridge: __torch_function__, operators."""

    def setup_method(self):
        """Reset the deprecation warning flag before each test."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        ProxyArray._deprecation_warned = False

    @cpu_only
    def test_torch_function_works_and_warns(self, device):
        """Test that __torch_function__ enables torch ops and emits a deprecation warning."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        arr = wp.ones(5, dtype=wp.float32, device=device)
        ta = ProxyArray(arr)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = torch.sum(ta)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert isinstance(result, torch.Tensor)
            assert result.item() == pytest.approx(5.0)

    @cpu_only
    def test_torch_cat_works_and_warns(self, device):
        """Test that torch.cat works with ProxyArray and emits a deprecation warning."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        a1 = wp.ones(3, dtype=wp.float32, device=device)
        a2 = wp.ones(4, dtype=wp.float32, device=device)
        ta1, ta2 = ProxyArray(a1), ProxyArray(a2)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = torch.cat([ta1, ta2])
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert result.shape == (7,)

    @cpu_only
    def test_warns_only_once(self, device):
        """Test that the deprecation warning is emitted only once per session."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        arr = wp.ones(5, dtype=wp.float32, device=device)
        ta = ProxyArray(arr)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = ta + 1.0
            _ = ta * 2.0
            _ = ta - 0.5
            # Only one warning despite three operations
            assert len(w) == 1

    @cpu_only
    def test_tensor_plus_proxy_array(self, device):
        """Test that torch.Tensor + ProxyArray works via __torch_function__."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        arr = wp.ones(5, dtype=wp.float32, device=device)
        ta = ProxyArray(arr)
        t = torch.ones(5, device=device) * 2.0

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = t + ta
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            expected = torch.full((5,), 3.0, device=device)
            torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize(
        "op, scalar, expected",
        [
            ("+", 1.0, [2.0, 3.0]),
            ("-", 1.0, [0.0, 1.0]),
            ("*", 2.0, [2.0, 4.0]),
            ("/", 2.0, [0.5, 1.0]),
            ("**", 2.0, [1.0, 4.0]),
        ],
    )
    def test_binary_operators(self, op, scalar, expected):
        """Test forward binary operators: +, -, *, /, **."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        ProxyArray._deprecation_warned = True
        ta = ProxyArray(wp.array([1.0, 2.0], dtype=wp.float32, device="cpu"))  # noqa: F841
        result = eval(f"ta {op} scalar")  # noqa: S307
        assert torch.allclose(result, torch.tensor(expected))

    @pytest.mark.parametrize(
        "op, scalar, expected",
        [
            ("+", 1.0, [2.0, 3.0]),
            ("-", 1.0, [0.0, -1.0]),
            ("*", 2.0, [2.0, 4.0]),
            ("/", 2.0, [2.0, 1.0]),
            ("**", 2.0, [2.0, 4.0]),
        ],
    )
    def test_reflected_operators(self, op, scalar, expected):
        """Test reflected binary operators: scalar op ProxyArray."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        ProxyArray._deprecation_warned = True
        ta = ProxyArray(wp.array([1.0, 2.0], dtype=wp.float32, device="cpu"))  # noqa: F841
        result = eval(f"scalar {op} ta")  # noqa: S307
        assert torch.allclose(result, torch.tensor(expected))

    def test_proxy_array_op_proxy_array(self):
        """Test binary operations between two ProxyArray instances."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        ProxyArray._deprecation_warned = True
        ta1 = ProxyArray(wp.array([1.0, 2.0], dtype=wp.float32, device="cpu"))
        ta2 = ProxyArray(wp.array([3.0, 4.0], dtype=wp.float32, device="cpu"))
        assert torch.allclose(ta1 + ta2, torch.tensor([4.0, 6.0]))
        assert torch.allclose(ta1 * ta2, torch.tensor([3.0, 8.0]))
        assert torch.allclose(ta2 - ta1, torch.tensor([2.0, 2.0]))

    @pytest.mark.parametrize(
        "op, values, expected",
        [
            ("-", [1.0, -2.0], [-1.0, 2.0]),
            ("+", [1.0, -2.0], [1.0, -2.0]),
            ("abs", [-1.0, 2.0], [1.0, 2.0]),
        ],
    )
    def test_unary_operators(self, op, values, expected):
        """Test unary operators: -, +, abs."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        ProxyArray._deprecation_warned = True
        ta = ProxyArray(wp.array(values, dtype=wp.float32, device="cpu"))  # noqa: F841
        result = eval(f"{op}(ta)" if op == "abs" else f"{op}ta")  # noqa: S307
        assert torch.allclose(result, torch.tensor(expected))

    @pytest.mark.parametrize(
        "op, expected",
        [
            ("==", [False, True, False]),
            ("!=", [True, False, True]),
            ("<", [True, False, False]),
            ("<=", [True, True, False]),
            (">", [False, False, True]),
            (">=", [False, True, True]),
        ],
    )
    def test_comparison_operators(self, op, expected):
        """Test comparison operators: ==, !=, <, <=, >, >=."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        ProxyArray._deprecation_warned = True
        ta = ProxyArray(wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device="cpu"))  # noqa: F841
        result = eval(f"ta {op} 2.0")  # noqa: S307
        assert result.tolist() == expected

    def test_getitem_1d(self):
        """Test 1D indexing via __getitem__."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        ProxyArray._deprecation_warned = True
        ta = ProxyArray(wp.array([10.0, 20.0, 30.0], dtype=wp.float32, device="cpu"))
        assert ta[0].item() == 10.0
        assert ta[-1].item() == 30.0
        assert ta[1:].tolist() == [20.0, 30.0]

    def test_getitem_nd(self):
        """Test multi-dimensional indexing via __getitem__ with structured types."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        ProxyArray._deprecation_warned = True
        wp_arr = wp.zeros((3, 4), dtype=wp.vec3f, device="cpu")
        ta = ProxyArray(wp_arr)
        # torch view is (3, 4, 3)
        result = ta[:, 0, :]
        assert result.shape == (3, 3)
        result = ta[0, :, 2]
        assert result.shape == (4,)

    def test_setitem_writes_through(self):
        """Test __setitem__ writes through to shared warp memory."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        ProxyArray._deprecation_warned = True
        wp_arr = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device="cpu")
        ta = ProxyArray(wp_arr)
        ta[0] = 99.0
        assert wp_arr.numpy()[0] == 99.0

    def test_getitem_warns(self):
        """Test __getitem__ emits deprecation warning."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        ProxyArray._deprecation_warned = False
        ta = ProxyArray(wp.array([1.0, 2.0], dtype=wp.float32, device="cpu"))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = ta[0]
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 1


class TestWpToTorchShim:
    """Tests for the ``wp.to_torch`` shim installed by ``isaaclab.utils.warp``.

    The shim makes legacy call sites like ``wp.to_torch(asset.data.joint_pos)``
    keep working after the ProxyArray migration, instead of raising
    ``AttributeError`` on ``requires_grad`` lookup.
    """

    def test_proxy_array_returns_torch_with_warning(self):
        """``wp.to_torch(ProxyArray)`` returns the cached .torch view and warns once."""
        import isaaclab.utils.warp as iw  # noqa: F401  # ensure shim is installed
        from isaaclab.utils.warp.proxy_array import ProxyArray

        # Reset the module-level one-shot flag so the warning fires in this test.
        iw._WP_TO_TORCH_WARNED = False

        arr = wp.array([7.0, 8.0], dtype=wp.float32, device="cpu")
        proxy = ProxyArray(arr)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t = wp.to_torch(proxy)
            deprecation = [w for w in caught if issubclass(w.category, DeprecationWarning)]

        assert isinstance(t, torch.Tensor)
        assert t is proxy.torch, "shim should return the cached .torch view"
        assert len(deprecation) == 1
        assert "ProxyArray" in str(deprecation[0].message)

    def test_proxy_array_warning_is_one_shot(self):
        """Repeated ``wp.to_torch(ProxyArray)`` calls must not spam warnings."""
        import isaaclab.utils.warp as iw
        from isaaclab.utils.warp.proxy_array import ProxyArray

        iw._WP_TO_TORCH_WARNED = True  # pretend the warning already fired

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            wp.to_torch(ProxyArray(wp.zeros(2, dtype=wp.float32, device="cpu")))
            deprecation = [w for w in caught if issubclass(w.category, DeprecationWarning)]

        assert not deprecation, "shim must not re-warn after the first call"

    def test_requires_grad_forwarded_to_raw_wp_array(self):
        """``wp.to_torch(wp.array)`` still produces a torch view and forwards ``requires_grad``."""
        import isaaclab.utils.warp  # noqa: F401  # ensure shim is installed

        arr = wp.array([1.0, 2.0], dtype=wp.float32, device="cpu")
        t = wp.to_torch(arr, requires_grad=False)
        assert isinstance(t, torch.Tensor)
        assert t.shape == (2,)
        assert t.requires_grad is False
