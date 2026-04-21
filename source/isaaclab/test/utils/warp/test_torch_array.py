# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the TorchArray class."""

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


class TestTorchArrayBasic:
    """Tests for basic TorchArray functionality."""

    def test_init_rejects_non_warp(self):
        """Test that __init__ raises TypeError for non-warp.array arguments."""
        from isaaclab.utils.warp.torch_array import TorchArray

        with pytest.raises(TypeError, match="expects a warp.array"):
            TorchArray(torch.zeros(10))
        with pytest.raises(TypeError, match="use it directly"):
            TorchArray(TorchArray(wp.zeros(5, dtype=wp.float32, device="cpu")))

    def test_warp_returns_original(self, device):
        """Test that .warp returns the original warp array."""
        from isaaclab.utils.warp.torch_array import TorchArray

        arr = wp.zeros(10, dtype=wp.float32, device=device)
        ta = TorchArray(arr)
        assert ta.warp is arr

    def test_torch_returns_tensor(self, device):
        """Test that .torch returns a torch.Tensor."""
        from isaaclab.utils.warp.torch_array import TorchArray

        arr = wp.zeros(10, dtype=wp.float32, device=device)
        ta = TorchArray(arr)
        assert isinstance(ta.torch, torch.Tensor)

    def test_torch_is_cached(self, device):
        """Test that .torch returns the same tensor object on repeated access."""
        from isaaclab.utils.warp.torch_array import TorchArray

        arr = wp.zeros(10, dtype=wp.float32, device=device)
        ta = TorchArray(arr)
        t1 = ta.torch
        t2 = ta.torch
        assert t1 is t2

    def test_torch_shares_memory(self, device):
        """Test that .torch provides a zero-copy view (shares memory with warp)."""
        from isaaclab.utils.warp.torch_array import TorchArray

        arr = wp.zeros(10, dtype=wp.float32, device=device)
        ta = TorchArray(arr)
        t = ta.torch
        # Modify the torch tensor
        t[0] = 42.0
        # The change should be visible in the warp array
        arr_np = arr.numpy()
        assert arr_np[0] == 42.0

    def test_rebind_invalidates_cache(self, device):
        """Test that rebind() updates the warp array and invalidates the cached torch tensor."""
        from isaaclab.utils.warp.torch_array import TorchArray

        arr1 = wp.zeros(10, dtype=wp.float32, device=device)
        ta = TorchArray(arr1)
        t1 = ta.torch  # cache created

        arr2 = wp.ones(10, dtype=wp.float32, device=device)
        ta.rebind(arr2)  # invalidate cache

        t2 = ta.torch  # should be a new tensor from arr2
        assert t1 is not t2
        assert t2[0].item() == 1.0
        assert ta.warp is arr2

    def test_rebind_stale_reference(self, device):
        """Test that a held .torch reference becomes stale after rebind()."""
        from isaaclab.utils.warp.torch_array import TorchArray

        arr1 = wp.zeros(10, dtype=wp.float32, device=device)
        ta = TorchArray(arr1)
        stale_ref = ta.torch  # hold a reference

        arr2 = wp.ones(10, dtype=wp.float32, device=device)
        ta.rebind(arr2)

        fresh_ref = ta.torch
        # The stale reference should NOT be the same object as the fresh one
        assert stale_ref is not fresh_ref
        # The stale reference still points to the OLD data (zeros)
        assert stale_ref[0].item() == 0.0
        # The fresh reference points to the NEW data (ones)
        assert fresh_ref[0].item() == 1.0

    def test_cuda_array_interface(self):
        """Test that __cuda_array_interface__ delegates to the underlying warp array."""
        from isaaclab.utils.warp.torch_array import TorchArray

        arr = wp.zeros(10, dtype=wp.float32, device="cuda:0")
        ta = TorchArray(arr)
        cai = ta.__cuda_array_interface__
        assert isinstance(cai, dict)
        assert "data" in cai
        assert "shape" in cai
        assert cai["shape"] == arr.__cuda_array_interface__["shape"]

    def test_cuda_array_interface_not_on_cpu(self):
        """Test that __cuda_array_interface__ raises AttributeError on CPU arrays."""
        from isaaclab.utils.warp.torch_array import TorchArray

        arr = wp.zeros(10, dtype=wp.float32, device="cpu")
        ta = TorchArray(arr)
        with pytest.raises(AttributeError):
            _ = ta.__cuda_array_interface__

    def test_wp_launch_accepts_torch_array(self):
        """Test that wp.launch() can consume a TorchArray via __cuda_array_interface__."""
        from isaaclab.utils.warp.torch_array import TorchArray

        @wp.kernel
        def _add_one(src: wp.array(dtype=wp.float32), dst: wp.array(dtype=wp.float32)):
            i = wp.tid()
            dst[i] = src[i] + 1.0

        src = TorchArray(wp.zeros(5, dtype=wp.float32, device="cuda:0"))
        dst = TorchArray(wp.zeros(5, dtype=wp.float32, device="cuda:0"))
        wp.launch(_add_one, dim=5, inputs=[src], outputs=[dst], device="cuda:0")
        wp.synchronize_device("cuda:0")
        assert dst.torch[0].item() == 1.0
        assert dst.torch[4].item() == 1.0


class TestTorchArrayStructuredTypes:
    """Tests for TorchArray with structured warp types (vec3f, quatf, etc)."""

    def test_vec3f_shape(self, device):
        """Test that vec3f arrays produce (N, 3) torch tensors."""
        from isaaclab.utils.warp.torch_array import TorchArray

        arr = wp.zeros(8, dtype=wp.vec3f, device=device)
        ta = TorchArray(arr)
        assert ta.torch.shape == (8, 3)

    def test_quatf_shape(self, device):
        """Test that quatf arrays produce (N, 4) torch tensors."""
        from isaaclab.utils.warp.torch_array import TorchArray

        arr = wp.zeros(8, dtype=wp.quatf, device=device)
        ta = TorchArray(arr)
        assert ta.torch.shape == (8, 4)

    def test_transformf_shape(self, device):
        """Test that transformf arrays produce (N, 7) torch tensors."""
        from isaaclab.utils.warp.torch_array import TorchArray

        arr = wp.zeros(8, dtype=wp.transformf, device=device)
        ta = TorchArray(arr)
        assert ta.torch.shape == (8, 7)

    def test_spatial_vectorf_shape(self, device):
        """Test that spatial_vectorf arrays produce (N, 6) torch tensors."""
        from isaaclab.utils.warp.torch_array import TorchArray

        arr = wp.zeros(8, dtype=wp.spatial_vectorf, device=device)
        ta = TorchArray(arr)
        assert ta.torch.shape == (8, 6)

    def test_2d_vec3f_shape(self, device):
        """Test that 2D vec3f arrays produce (N, M, 3) torch tensors."""
        from isaaclab.utils.warp.torch_array import TorchArray

        arr = wp.zeros((4, 5), dtype=wp.vec3f, device=device)
        ta = TorchArray(arr)
        assert ta.torch.shape == (4, 5, 3)


class TestTorchArrayConvenienceProperties:
    """Tests for convenience properties: shape, dtype, device, len, repr."""

    def test_shape(self, device):
        """Test that .shape returns the warp array shape."""
        from isaaclab.utils.warp.torch_array import TorchArray

        arr = wp.zeros((3, 4), dtype=wp.float32, device=device)
        ta = TorchArray(arr)
        assert ta.shape == (3, 4)

    def test_dtype(self, device):
        """Test that .dtype returns the warp dtype."""
        from isaaclab.utils.warp.torch_array import TorchArray

        arr = wp.zeros(10, dtype=wp.float32, device=device)
        ta = TorchArray(arr)
        assert ta.dtype == wp.float32

    def test_device(self, device):
        """Test that .device returns the warp device string."""
        from isaaclab.utils.warp.torch_array import TorchArray

        arr = wp.zeros(10, dtype=wp.float32, device=device)
        ta = TorchArray(arr)
        assert ta.device == arr.device

    def test_len(self, device):
        """Test that len() returns the first dimension size."""
        from isaaclab.utils.warp.torch_array import TorchArray

        arr = wp.zeros((7, 3), dtype=wp.float32, device=device)
        ta = TorchArray(arr)
        assert len(ta) == 7

    def test_repr(self, device):
        """Test that repr() contains TorchArray and key info."""
        from isaaclab.utils.warp.torch_array import TorchArray

        arr = wp.zeros(5, dtype=wp.float32, device=device)
        ta = TorchArray(arr)
        r = repr(ta)
        assert "TorchArray" in r
        assert "float32" in r


class TestTorchArrayDeprecationBridge:
    """Tests for the deprecation bridge: __torch_function__, operators."""

    def setup_method(self):
        """Reset the deprecation warning flag before each test."""
        from isaaclab.utils.warp.torch_array import TorchArray

        TorchArray._deprecation_warned = False

    def test_torch_function_works_and_warns(self, device):
        """Test that __torch_function__ enables torch ops and emits a deprecation warning."""
        from isaaclab.utils.warp.torch_array import TorchArray

        arr = wp.ones(5, dtype=wp.float32, device=device)
        ta = TorchArray(arr)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = torch.sum(ta)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert isinstance(result, torch.Tensor)
            assert result.item() == pytest.approx(5.0)

    def test_torch_cat_works_and_warns(self, device):
        """Test that torch.cat works with TorchArray and emits a deprecation warning."""
        from isaaclab.utils.warp.torch_array import TorchArray

        a1 = wp.ones(3, dtype=wp.float32, device=device)
        a2 = wp.ones(4, dtype=wp.float32, device=device)
        ta1, ta2 = TorchArray(a1), TorchArray(a2)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = torch.cat([ta1, ta2])
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert result.shape == (7,)

    def test_arithmetic_operators_work_and_warn(self, device):
        """Test that arithmetic operators work and emit deprecation warnings."""
        from isaaclab.utils.warp.torch_array import TorchArray

        arr = wp.ones(5, dtype=wp.float32, device=device)
        ta = TorchArray(arr)

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
        from isaaclab.utils.warp.torch_array import TorchArray

        arr = wp.ones(5, dtype=wp.float32, device=device)
        ta = TorchArray(arr)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = ta + 1.0
            _ = ta * 2.0
            _ = ta - 0.5
            # Only one warning despite three operations
            assert len(w) == 1

    def test_tensor_plus_torch_array(self, device):
        """Test that torch.Tensor + TorchArray works via __torch_function__."""
        from isaaclab.utils.warp.torch_array import TorchArray

        arr = wp.ones(5, dtype=wp.float32, device=device)
        ta = TorchArray(arr)
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
        from isaaclab.utils.warp.torch_array import TorchArray

        TorchArray._deprecation_warned = True
        ta = TorchArray(wp.array([1.0, 2.0], dtype=wp.float32, device="cpu"))  # noqa: F841
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
        """Test reflected binary operators: scalar op TorchArray."""
        from isaaclab.utils.warp.torch_array import TorchArray

        TorchArray._deprecation_warned = True
        ta = TorchArray(wp.array([1.0, 2.0], dtype=wp.float32, device="cpu"))  # noqa: F841
        result = eval(f"scalar {op} ta")  # noqa: S307
        assert torch.allclose(result, torch.tensor(expected))

    def test_torch_array_op_torch_array(self):
        """Test binary operations between two TorchArray instances."""
        from isaaclab.utils.warp.torch_array import TorchArray

        TorchArray._deprecation_warned = True
        ta1 = TorchArray(wp.array([1.0, 2.0], dtype=wp.float32, device="cpu"))
        ta2 = TorchArray(wp.array([3.0, 4.0], dtype=wp.float32, device="cpu"))
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
        from isaaclab.utils.warp.torch_array import TorchArray

        TorchArray._deprecation_warned = True
        ta = TorchArray(wp.array(values, dtype=wp.float32, device="cpu"))  # noqa: F841
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
        from isaaclab.utils.warp.torch_array import TorchArray

        TorchArray._deprecation_warned = True
        ta = TorchArray(wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device="cpu"))  # noqa: F841
        result = eval(f"ta {op} 2.0")  # noqa: S307
        assert result.tolist() == expected

    def test_getitem_1d(self):
        """Test 1D indexing via __getitem__."""
        from isaaclab.utils.warp.torch_array import TorchArray

        TorchArray._deprecation_warned = True
        ta = TorchArray(wp.array([10.0, 20.0, 30.0], dtype=wp.float32, device="cpu"))
        assert ta[0].item() == 10.0
        assert ta[-1].item() == 30.0
        assert ta[1:].tolist() == [20.0, 30.0]

    def test_getitem_nd(self):
        """Test multi-dimensional indexing via __getitem__ with structured types."""
        from isaaclab.utils.warp.torch_array import TorchArray

        TorchArray._deprecation_warned = True
        wp_arr = wp.zeros((3, 4), dtype=wp.vec3f, device="cpu")
        ta = TorchArray(wp_arr)
        # torch view is (3, 4, 3)
        result = ta[:, 0, :]
        assert result.shape == (3, 3)
        result = ta[0, :, 2]
        assert result.shape == (4,)

    def test_setitem_writes_through(self):
        """Test __setitem__ writes through to shared warp memory."""
        from isaaclab.utils.warp.torch_array import TorchArray

        TorchArray._deprecation_warned = True
        wp_arr = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device="cpu")
        ta = TorchArray(wp_arr)
        ta[0] = 99.0
        assert wp_arr.numpy()[0] == 99.0

    def test_getitem_warns(self):
        """Test __getitem__ emits deprecation warning."""
        from isaaclab.utils.warp.torch_array import TorchArray

        TorchArray._deprecation_warned = False
        ta = TorchArray(wp.array([1.0, 2.0], dtype=wp.float32, device="cpu"))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = ta[0]
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 1
