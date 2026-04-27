# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the ProxyArray class."""

import warnings

import pytest
import torch
import warp as wp

wp.config.quiet = True
wp.init()


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    """Parametrize tests across CPU and CUDA devices."""
    return request.param


class TestProxyArrayBasic:
    """Tests for basic ProxyArray functionality."""

    def test_warp_returns_original(self, device):
        """Test that .warp returns the original warp array."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        arr = wp.zeros(10, dtype=wp.float32, device=device)
        ta = ProxyArray(arr)
        assert ta.warp is arr

    def test_torch_returns_tensor(self, device):
        """Test that .torch returns a torch.Tensor."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        arr = wp.zeros(10, dtype=wp.float32, device=device)
        ta = ProxyArray(arr)
        assert isinstance(ta.torch, torch.Tensor)

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

    def test_immutable_warp_cannot_be_reassigned(self, device):
        """ProxyArray._warp cannot be reassigned; callers must construct a new wrapper."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        arr = wp.zeros(10, dtype=wp.float32, device=device)
        ta = ProxyArray(arr)

        with pytest.raises(AttributeError, match="immutable"):
            ta._warp = wp.ones(10, dtype=wp.float32, device=device)
        with pytest.raises(AttributeError, match="immutable"):
            ta.new_field = 42  # arbitrary attribute writes also blocked

    def test_immutable_allows_internal_torch_cache(self, device):
        """Lazy .torch caching still works — only _torch_cache is allowed as a post-init write."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        arr = wp.zeros(10, dtype=wp.float32, device=device)
        ta = ProxyArray(arr)
        # First access populates the cache; no exception.
        first = ta.torch
        # Subsequent accesses return the same cached tensor.
        second = ta.torch
        assert first is second

    def test_cuda_array_interface(self):
        """Test that __cuda_array_interface__ delegates to the underlying warp array."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        arr = wp.zeros(10, dtype=wp.float32, device="cuda:0")
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

    def test_wp_launch_accepts_proxy_array(self):
        """Test that wp.launch() can consume a ProxyArray via __cuda_array_interface__."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        @wp.kernel
        def _add_one(src: wp.array(dtype=wp.float32), dst: wp.array(dtype=wp.float32)):
            i = wp.tid()
            dst[i] = src[i] + 1.0

        src = ProxyArray(wp.zeros(5, dtype=wp.float32, device="cuda:0"))
        dst = ProxyArray(wp.zeros(5, dtype=wp.float32, device="cuda:0"))
        wp.launch(_add_one, dim=5, inputs=[src], outputs=[dst], device="cuda:0")
        wp.synchronize_device("cuda:0")
        assert dst.torch[0].item() == 1.0
        assert dst.torch[4].item() == 1.0


class TestProxyArrayStructuredTypes:
    """Tests for ProxyArray with structured warp types (vec3f, quatf, etc)."""

    def test_vec3f_shape(self, device):
        """Test that vec3f arrays produce (N, 3) torch tensors."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        arr = wp.zeros(8, dtype=wp.vec3f, device=device)
        ta = ProxyArray(arr)
        assert ta.torch.shape == (8, 3)

    def test_quatf_shape(self, device):
        """Test that quatf arrays produce (N, 4) torch tensors."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        arr = wp.zeros(8, dtype=wp.quatf, device=device)
        ta = ProxyArray(arr)
        assert ta.torch.shape == (8, 4)

    def test_transformf_shape(self, device):
        """Test that transformf arrays produce (N, 7) torch tensors."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        arr = wp.zeros(8, dtype=wp.transformf, device=device)
        ta = ProxyArray(arr)
        assert ta.torch.shape == (8, 7)

    def test_spatial_vectorf_shape(self, device):
        """Test that spatial_vectorf arrays produce (N, 6) torch tensors."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        arr = wp.zeros(8, dtype=wp.spatial_vectorf, device=device)
        ta = ProxyArray(arr)
        assert ta.torch.shape == (8, 6)

    def test_2d_vec3f_shape(self, device):
        """Test that 2D vec3f arrays produce (N, M, 3) torch tensors."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        arr = wp.zeros((4, 5), dtype=wp.vec3f, device=device)
        ta = ProxyArray(arr)
        assert ta.torch.shape == (4, 5, 3)


class TestProxyArrayQuatfTorchAccessWarning:
    """Tests for the WARN_ON_TORCH_QUATF_ACCESS opt-in runtime detector."""

    def test_default_no_warning(self, device, monkeypatch):
        """No env var → quatf .torch access is silent."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        monkeypatch.delenv("WARN_ON_TORCH_QUATF_ACCESS", raising=False)
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

    def test_env_zero_does_not_warn(self, device, monkeypatch):
        """WARN_ON_TORCH_QUATF_ACCESS=0 → silent (only ``"1"`` enables the detector)."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        monkeypatch.setenv("WARN_ON_TORCH_QUATF_ACCESS", "0")
        ta = ProxyArray(wp.zeros(4, dtype=wp.quatf, device=device))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = ta.torch
            assert [x for x in w if issubclass(x.category, UserWarning)] == []


class TestProxyArrayConvenienceProperties:
    """Tests for convenience properties: shape, dtype, device, len, repr."""

    def test_shape(self, device):
        """Test that .shape returns the warp array shape."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        arr = wp.zeros((3, 4), dtype=wp.float32, device=device)
        ta = ProxyArray(arr)
        assert ta.shape == (3, 4)

    def test_dtype(self, device):
        """Test that .dtype returns the warp dtype."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        arr = wp.zeros(10, dtype=wp.float32, device=device)
        ta = ProxyArray(arr)
        assert ta.dtype == wp.float32

    def test_device(self, device):
        """Test that .device returns the warp device string."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        arr = wp.zeros(10, dtype=wp.float32, device=device)
        ta = ProxyArray(arr)
        assert ta.device == arr.device

    def test_len(self, device):
        """Test that len() returns the first dimension size."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        arr = wp.zeros((7, 3), dtype=wp.float32, device=device)
        ta = ProxyArray(arr)
        assert len(ta) == 7

    def test_repr(self, device):
        """Test that repr() contains ProxyArray and key info."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        arr = wp.zeros(5, dtype=wp.float32, device=device)
        ta = ProxyArray(arr)
        r = repr(ta)
        assert "ProxyArray" in r
        assert "float32" in r


class TestProxyArrayDeprecationBridge:
    """Tests for the deprecation bridge: __torch_function__, operators."""

    def setup_method(self):
        """Reset the deprecation warning flag before each test."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        ProxyArray._deprecation_warned = False

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

    def test_arithmetic_operators_work_and_warn(self, device):
        """Test that arithmetic operators work and emit deprecation warnings."""
        from isaaclab.utils.warp.proxy_array import ProxyArray

        arr = wp.ones(5, dtype=wp.float32, device=device)
        ta = ProxyArray(arr)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = ta + 1.0
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert isinstance(result, torch.Tensor)
            expected = torch.full((5,), 2.0, device=device)
            torch.testing.assert_close(result, expected)

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

    def test_raw_wp_array_unchanged(self):
        """``wp.to_torch(wp.array)`` must still produce a zero-copy torch view."""
        import isaaclab.utils.warp  # noqa: F401  # ensure shim is installed

        arr = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device="cpu")
        t = wp.to_torch(arr)
        assert isinstance(t, torch.Tensor)
        assert t.shape == (3,)

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
        """The ``requires_grad`` kwarg still reaches the original ``wp.to_torch``."""
        import isaaclab.utils.warp  # noqa: F401

        arr = wp.array([1.0, 2.0], dtype=wp.float32, device="cpu")
        t = wp.to_torch(arr, requires_grad=False)
        assert isinstance(t, torch.Tensor)
        assert t.requires_grad is False
