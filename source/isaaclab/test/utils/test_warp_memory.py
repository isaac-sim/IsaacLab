# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent layout, stride, view-aliasing and masked-write contracts."""

import numpy as np
import pytest
import warp as wp

from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.warp import memory
from isaaclab.utils.warp.index_kernel import selector_dtype

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("first_type", "second_type", "result_type", "shape", "numpy_type"),
    [
        (wp.transformf, wp.spatial_vectorf, wp.types.vector(13, wp.float32), (4,), np.float32),
        (wp.vec2d, wp.vec3d, wp.types.vector(5, wp.float64), (2, 2), np.float64),
    ],
)
def test_concatenation_and_split_preserve_component_order(first_type, second_type, result_type, shape, numpy_type):
    with wp.ScopedDevice("cpu"):
        size, first_size = len(result_type()), len(first_type())
        expected = np.arange(1, 4 * size + 1, dtype=numpy_type).reshape(*shape, size)
        first = wp.array(expected[..., :first_size].copy(), dtype=first_type)
        second = wp.array(expected[..., first_size:].copy(), dtype=second_type)
        output = wp.empty(shape, dtype=result_type)
        kernel = memory.make_concat(first_type, second_type, result_type, len(shape))
        wp.launch(kernel, shape, [first, second, output])
        np.testing.assert_array_equal(output.numpy(), expected)
        # Independently supplied components, not a round trip of the concatenation result.
        first.zero_()
        second.zero_()
        wp.launch(
            memory.make_split(first_type, second_type, len(shape)),
            shape,
            [wp.array(expected, dtype=wp.float64 if numpy_type == np.float64 else wp.float32), None, first, second],
        )
        np.testing.assert_array_equal(first.numpy(), expected[..., :first_size])
        np.testing.assert_array_equal(second.numpy(), expected[..., first_size:])


def test_transform_packing_and_component_views_preserve_strides_and_aliasing():
    with wp.ScopedDevice("cpu"):
        expected = np.arange(1, 29, dtype=np.float32).reshape(4, 7)
        for shape in ((4,), (2, 2)):
            transforms = wp.array(expected.reshape((*shape, 7)), dtype=wp.transformf)
            pos, quat = memory.split_views(transforms, (wp.vec3f, wp.quatf))
            # Both inputs are strided component views; output skips alternating transforms.
            full = wp.full(tuple(2 * n if i == len(shape) - 1 else n for i, n in enumerate(shape)), wp.transformf(99.0))
            output = full[::2] if len(shape) == 1 else full[:, ::2]
            wp.launch(memory.make_concat(wp.vec3f, wp.quatf, wp.transformf, len(shape)), shape, [pos, quat, output])
            np.testing.assert_array_equal(output.numpy(), expected.reshape((*shape, 7)))
            untouched = full[1::2] if len(shape) == 1 else full[:, 1::2]
            np.testing.assert_array_equal(untouched.numpy(), 99.0)
            pos.fill_(wp.vec3f(-3.0))
            np.testing.assert_array_equal(transforms.numpy()[..., :3], -3.0)
            np.testing.assert_array_equal(transforms.numpy()[..., 3:], expected.reshape((*shape, 7))[..., 3:])
        transforms = wp.array(expected.reshape(1, 2, 2, 7), dtype=wp.transformf)
        pos, quat = memory.split_views(transforms, (wp.vec3f, wp.quatf))
        np.testing.assert_array_equal(pos.numpy(), expected.reshape(1, 2, 2, 7)[..., :3])
        np.testing.assert_array_equal(quat.numpy(), expected.reshape(1, 2, 2, 7)[..., 3:])
        with pytest.raises(ValueError, match="ndim=4"):
            memory.split_views(wp.zeros((1, 1, 1, 1), dtype=wp.transformf), (wp.vec3f, wp.quatf))
        with pytest.raises(TypeError):
            memory.split_views(wp.zeros(3, dtype=wp.vec3f), (wp.vec3d,))
        for dtype in (wp.float32, wp.mat33f):
            with pytest.raises(TypeError, match="one-dimensional composite"):
                memory.make_concat(dtype, wp.vec3f, wp.transformf, 1)
            with pytest.raises(TypeError, match="one-dimensional composite"):
                memory.make_split(dtype, wp.vec3f, 1)
            with pytest.raises(TypeError, match="one-dimensional composite"):
                memory.split_views(wp.empty(1, dtype=dtype), (wp.vec3f,))


def test_spatial_split_and_pair_clear_preserve_masked_rows():
    with wp.ScopedDevice("cpu"):
        mask = wp.array([True, False], dtype=wp.bool)
        spatial = wp.array([[[1, 2, 3, 4, 5, 6]], [[7, 8, 9, 10, 11, 12]]], dtype=wp.spatial_vectorf)
        first, second = [wp.full((2, 1), wp.vec3f(99.0)) for _ in range(2)]
        wp.launch(memory.make_split(wp.vec3f, wp.vec3f, 2), (2, 1), [spatial.view(wp.float32), mask, first, second])
        np.testing.assert_array_equal(first.numpy(), [[[1, 2, 3]], [[99, 99, 99]]])
        np.testing.assert_array_equal(second.numpy(), [[[4, 5, 6]], [[99, 99, 99]]])
        for shape in ((2, 1), (2,)):
            first.fill_(wp.vec3f(99.0))
            second.fill_(wp.vec3f(88.0))
            wp.launch(
                memory.make_clear_pair(wp.vec3f, len(shape)),
                shape,
                [mask, first.reshape(shape), second.reshape(shape)],
            )
            np.testing.assert_array_equal(first.numpy(), [[[0, 0, 0]], [[99, 99, 99]]])
            np.testing.assert_array_equal(second.numpy(), [[[0, 0, 0]], [[88, 88, 88]]])


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_generated_operations_replay_changed_strided_inputs_and_selectors(device):
    with wp.ScopedDevice(device):
        original = wp.array([[[1, 2, 3, 0, 0, 0, 1], [4, 5, 6, 0, 0, 1, 0]]], dtype=wp.transformf)
        pos, quat = memory.split_views(original, (wp.vec3f, wp.quatf))
        full = wp.full((1, 4), wp.transformf(99.0))
        output = full[:, ::2]
        kernel = memory.make_concat(wp.vec3f, wp.quatf, wp.transformf, 2)
        wp.launch(kernel, (1, 2), [pos, quat, output])
        rows = wp.array([0, 99], dtype=wp.int64)[::2]
        columns = wp.array([2, 99, 0, 99], dtype=wp.int32)[::2]
        values = wp.array([[11.0, 12.0]], dtype=wp.float32)
        target = wp.full((2, 3), -1.0)
        selected = wp.full(1, -1, dtype=wp.int32)

        row_mask = wp.array([False, True], dtype=wp.bool)
        column_mask = wp.array([False, True, False], dtype=wp.bool)

        wp.launch(
            memory.make_copy(wp.float32, 2, selector_dtype(rows), selector_dtype(columns)),
            (1, 2),
            [values, rows, columns, target, selected],
        )
        wp.launch(memory.make_fill(wp.float32, 2), (2, 3), [8.0, row_mask, column_mask, target, None])
        with wp.ScopedCapture(device=device) as capture:
            wp.launch(kernel, (1, 2), [pos, quat, output])
            wp.launch(
                memory.make_copy(wp.float32, 2, selector_dtype(rows), selector_dtype(columns)),
                (1, 2),
                [values, rows, columns, target, selected],
            )
            wp.launch(memory.make_fill(wp.float32, 2), (2, 3), [8.0, row_mask, column_mask, target, None])
        rows.fill_(1)
        values.fill_(7.0)
        row_mask.assign(np.array([True, False]))
        target.fill_(-1.0)
        pos.fill_(wp.vec3f(-7.0))
        # A later specialization must not invalidate a kernel already in a graph.
        before = memory.make_copy(wp.float32, 2, wp.int64, wp.int32).module.hash_module()
        memory.make_copy(wp.float64, 3, wp.int32, wp.int64).module.load(wp.get_device(device))
        assert memory.make_copy(wp.float32, 2, wp.int64, wp.int32).module.hash_module() == before
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(output.numpy(), [[[-7, -7, -7, 0, 0, 0, 1], [-7, -7, -7, 0, 0, 1, 0]]])
        np.testing.assert_array_equal(full[:, 1::2].numpy(), 99.0)
        np.testing.assert_array_equal(target.numpy(), [[-1, 8, -1], [7, -1, 7]])
        np.testing.assert_array_equal(selected.numpy(), [1])


def test_mask_resolution_preserves_selector_contracts_and_reuses_storage():
    import torch

    all_mask = wp.ones(5, dtype=wp.bool, device="cpu")
    scratch = wp.zeros_like(all_mask)
    kwargs = dict(all_mask=all_mask, scratch_mask=scratch)
    assert memory.resolve_1d_mask(**kwargs) is all_mask
    assert memory.resolve_1d_mask(ids=slice(None), **kwargs) is all_mask
    explicit = wp.array([False, True, False, False, True], dtype=wp.bool, device="cpu")
    assert memory.resolve_1d_mask(ids=[0], mask=explicit, **kwargs) is explicit
    for ids, expected in [
        (list(np.array([1, 3], dtype=np.int64)), [False, True, False, True, False]),
        ([2, 0, 2, -1], [True, False, True, False, True]),
        ([], [False] * 5),
        (slice(4, 0, -2), [False, False, True, False, True]),
        (slice(1, 1), [False] * 5),
        (torch.tensor([1, 99, 3, 99], dtype=torch.int64)[::2], [False, True, False, True, False]),
        (wp.array([-2], dtype=wp.int32, device="cpu"), [False, False, False, True, False]),
        (wp.array([2**40], dtype=wp.int64, device="cpu"), [False] * 5),
    ]:
        result = memory.resolve_1d_mask(ids=ids, **kwargs)
        assert result is scratch
        np.testing.assert_array_equal(result.numpy(), expected)
    for ids in (
        [False, True, False, False, True],
        list(np.array([False, True, False, False, True])),
        explicit,
        torch.tensor([False, True, False, False, True]),
    ):
        np.testing.assert_array_equal(memory.resolve_1d_mask(ids=ids, **kwargs).numpy(), explicit.numpy())
    with pytest.raises(ValueError, match="slice step"):
        memory.resolve_1d_mask(ids=slice(None, None, 0), **kwargs)
    with pytest.raises(ValueError, match="destination shape"):
        memory.resolve_1d_mask(mask=wp.ones(2, dtype=wp.bool, device="cpu"), **kwargs)
    with pytest.raises(TypeError, match="Selectors must"):
        memory.resolve_1d_mask(ids=torch.tensor([1.5]), **kwargs)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_mask_resolution_replays_device_indices_and_rejects_capture_uploads(device):
    import torch

    all_mask = wp.ones(5, dtype=wp.bool, device=device)
    scratch = wp.zeros_like(all_mask)
    storage = torch.tensor([1, 99, -1, 99], dtype=torch.int64, device=device)
    ids = storage[::2]
    kwargs = dict(all_mask=all_mask, scratch_mask=scratch)
    memory.resolve_1d_mask(ids=ids, **kwargs)
    host_ids = torch.tensor([1], dtype=torch.int32)
    # The target's capture state matters even if another device is the Python default.
    with wp.ScopedDevice("cpu"), wp.ScopedCapture(device=device) as capture:
        with pytest.raises(RuntimeError, match="upload host selectors"):
            memory.resolve_1d_mask(ids=[1], **kwargs)
        with pytest.raises(RuntimeError, match="transfer selectors"):
            memory.resolve_1d_mask(ids=host_ids, **kwargs)
        assert memory.resolve_1d_mask(ids=[1], mask=all_mask, **kwargs) is all_mask
        memory.resolve_1d_mask(ids=ids, **kwargs)
    storage[0] = 0
    storage[2] = 2
    wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(scratch.numpy(), [True, False, True, False, False])

    # Eager temporary uploads must remain alive on an imported nonblocking stream.
    torch_stream = torch.cuda.Stream(device=device)
    with wp.ScopedStream(wp.stream_from_torch(torch_stream)):
        result = memory.resolve_1d_mask(ids=[1, -1], **kwargs)
    np.testing.assert_array_equal(result.numpy(), [False, True, False, False, True])


@pytest.mark.parametrize(
    ("row_type", "column_type"),
    [(wp.int32, wp.int32), (wp.int64, wp.int64), (wp.int32, wp.int64), (wp.int64, wp.int32)],
)
def test_selected_write_preserves_order_strides_widths_and_untouched_values(row_type, column_type):
    with wp.ScopedDevice("cpu"):
        rows = wp.array([1, 99, 0, 99], dtype=row_type)[::2]
        columns = wp.array([2, 99, 0, 99], dtype=column_type)[::2]
        if row_type == wp.int64 and column_type == wp.int32:
            rows = wp.to_torch(rows)  # Callers may supply a strided Torch selector directly.
        source = wp.array([[11.0, 99, 12.0, 99], [21.0, 99, 22.0, 99]], dtype=wp.float32)[:, ::2]
        full = wp.full((2, 6), -1.0)
        target = full[:, ::2]
        selected = wp.full(2, -1, dtype=wp.int32)

        wp.launch(
            memory.make_copy(wp.float32, 2, selector_dtype(rows), selector_dtype(columns)),
            (2, 2),
            [source, rows, columns, target, selected],
        )
        np.testing.assert_array_equal(target.numpy(), [[22.0, -1.0, 21.0], [12.0, -1.0, 11.0]])
        np.testing.assert_array_equal(full[:, 1::2].numpy(), -1.0)
        np.testing.assert_array_equal(selected.numpy(), [1, 0])
        full_source = wp.array([[31, 32, 33], [41, 42, 43]], dtype=wp.float32)
        wp.launch(
            memory.make_copy(wp.float32, 2, selector_dtype(rows), selector_dtype(columns), full_source=True),
            (2, 2),
            [full_source, rows, columns, target, None],
        )
        np.testing.assert_array_equal(target.numpy(), [[31, -1, 33], [41, -1, 43]])

        wp.launch(
            memory.make_fill(wp.float32, 2, selector_dtype(rows), selector_dtype(columns)),
            (2, 2),
            [7.0, rows, columns, target, selected],
        )
        np.testing.assert_array_equal(target.numpy(), [[7, -1, 7], [7, -1, 7]])
        selected.fill_(-2)
        wp.launch(
            memory.make_fill(wp.float32, 2, selector_dtype(rows), selector_dtype(columns)),
            (2, 0),
            [8.0, rows, columns[:0], target, selected],
        )
        np.testing.assert_array_equal(selected.numpy(), [-2, -2])


def test_selected_write_handles_typed_elements_masks_and_trailing_components():
    with wp.ScopedDevice("cpu"):
        # A caller resolves its first-column frequency into a strided rank-1 view.
        elements = wp.array(np.arange(42, dtype=np.float32).reshape(2, 3, 7), dtype=wp.transformf)
        first_column = wp.full(2, wp.transformf(-1.0))
        wp.launch(
            memory.make_copy(wp.transformf, 1),
            2,
            [elements[:, 0], None, None, first_column, None],
        )
        np.testing.assert_array_equal(first_column.numpy(), [[0, 1, 2, 3, 4, 5, 6], [21, 22, 23, 24, 25, 26, 27]])
        selected_rows = wp.array([1, 0], dtype=wp.int64)
        reverse = wp.empty_like(first_column)
        wp.launch(
            memory.make_copy(wp.transformf, 1, selector_dtype(selected_rows)),
            2,
            [first_column, selected_rows, None, reverse, None],
        )
        np.testing.assert_array_equal(reverse.numpy(), [[21, 22, 23, 24, 25, 26, 27], [0, 1, 2, 3, 4, 5, 6]])
        rows = wp.array([False, True], dtype=wp.bool)
        columns = wp.array([True, False, True], dtype=wp.bool)
        for dtype, size in ((wp.vec3f, 3), (wp.transformf, 7)):
            source = wp.array(np.arange(6 * size, dtype=np.float32).reshape(2, 3, size), dtype=dtype)
            target = wp.full((2, 3), dtype(-1.0))
            wp.launch(
                memory.make_copy(dtype, 2),
                (2, 3),
                [source, rows, columns, target, None],
            )
            expected = np.full((2, 3, size), -1.0)
            expected[1, 0] = np.arange(3 * size, 4 * size)
            expected[1, 2] = np.arange(5 * size, 6 * size)
            np.testing.assert_array_equal(target.numpy(), expected)
        # A trailing dimension unrelated to any physical tensor layout.
        source = wp.array(np.arange(24, dtype=np.float32).reshape(2, 3, 4), dtype=wp.float32)
        target = wp.full((2, 3, 8), -1.0)[:, :, ::2]
        wp.launch(
            memory.make_copy(wp.float32, 3),
            (2, 3, 4),
            [source, rows, columns, target, None],
        )
        expected = np.full((2, 3, 4), -1.0)
        expected[1, 0], expected[1, 2] = [12, 13, 14, 15], [20, 21, 22, 23]
        np.testing.assert_array_equal(target.numpy(), expected)
        row_ids = wp.array([1, 1, 0], dtype=wp.int64)
        column_ids = wp.array([2, 0], dtype=wp.int32)
        selected = wp.full(3, -1, dtype=wp.int32)
        wp.launch(
            memory.make_copy(wp.float32, 3, selector_dtype(row_ids), selector_dtype(column_ids), full_source=True),
            (3, 2, 4),
            [source, row_ids, column_ids, target, selected],
        )
        np.testing.assert_array_equal(selected.numpy(), [1, 1, 0])
        expected[0, 0], expected[0, 2] = [0, 1, 2, 3], [8, 9, 10, 11]
        np.testing.assert_array_equal(target.numpy(), expected)


def test_memory_ownership_and_specialization(source_checkout_root):
    """Memory accepts caller layouts and selectors without application-domain ownership."""
    import ast
    import re

    first = memory.make_concat(wp.vec3f, wp.quatf, wp.transformf, 2)
    before = first.module.hash_module()
    second = memory.make_concat(wp.vec2d, wp.vec3d, wp.types.vector(5, wp.float64), 2)
    assert first is memory.make_concat(wp.vec3f, wp.quatf, wp.transformf, 2)
    assert first.module is not second.module
    assert first.module.hash_module() == before
    for factory in (memory.make_copy, memory.make_fill):
        first = factory(wp.float64, 1, wp.int32)
        before = first.module.hash_module()
        assert isinstance(first, wp.Kernel)
        assert factory(wp.float64, 1, wp.int32) is first
        second = factory(wp.float64, 1, wp.int64)
        assert first.module is not second.module
        assert first.module.hash_module() == before
    source = source_checkout_root / "source"
    core = source / "isaaclab/isaaclab"
    owner = core / "utils/warp/memory.py"
    assert not hasattr(memory, "make_write")
    assert not hasattr(memory, "IndexKernelDispatcher")
    for node in ast.walk(ast.parse(owner.read_text())):
        name = getattr(node, "id", getattr(node, "arg", getattr(node, "name", "")))
        assert not set(re.split(r"_+", name.lower())) & {
            "body",
            "joint",
            "env",
            "pose",
            "velocity",
            "spatial",
            "transform",
            "vec13f",
        }, name
        if isinstance(node, ast.FunctionDef) and node.name.lstrip("_").startswith("make_"):
            assert not node.name.endswith("_kernel"), node.name
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "wp":
            assert not node.attr.startswith(("vec", "quat", "transform", "spatial")), node.attr
        modules = [a.name for a in node.names] if isinstance(node, ast.Import) else []
        if isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
        assert not any(
            m.startswith(
                (
                    "isaaclab.assets",
                    "isaaclab.sensors",
                    "isaaclab_newton",
                    "isaaclab_physx",
                    "isaaclab_ov",
                    "newton",
                    "isaacsim",
                )
            )
            for m in modules
        ), owner
    for retired in ("assets/kernels.py", "utils/warp/math_ops.py"):
        assert not (core / retired).exists()
    retired_functions = {
        "concat_pose_and_vel_to_state_func",
        "concat_root_pose_and_vel_to_state",
        "concat_body_pose_and_vel_to_state",
        "split_state_to_root_pose_and_vel",
    }
    retired_prefixes = (
        "write_2d_data_to_buffer",
        "float_data_to_buffer",
        "write_body_com_pose_to_buffer",
        "write_body_com_position_to_buffer",
        "write_body_inertia_to_buffer",
        "write_single_body_inertia_to_buffer",
        "write_joint_state_to_buffer_with_indices",
        "scatter_rows_partial",
        "copy_first_body",
    )
    for backend in ("ov", "physx", "newton"):
        for relative in ("assets/kernels.py", "assets/articulation/kernels.py"):
            file = source / f"isaaclab_{backend}/isaaclab_{backend}" / relative
            definitions = {n.name for n in ast.parse(file.read_text()).body if isinstance(n, ast.FunctionDef)}
            assert not definitions & retired_functions, file
            assert not any(name.lstrip("_").startswith(retired_prefixes) for name in definitions), file
            if backend != "ov":
                assert not definitions & {"set_state_transforms_func", "set_state_velocities_func"}, file
    factories = {"make_concat", "make_split", "make_clear_pair", "make_copy", "make_fill"}
    for package in source.iterdir():
        for file in (package / package.name).rglob("*.py"):
            tree = ast.parse(file.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "vec13f" for t in node.targets
                ):
                    assert file == core / "assets/rigid_object/base_rigid_object_data.py", file
                if isinstance(node, ast.FunctionDef) and node.name == "resolve_1d_mask":
                    assert file == owner, file
                modules = [a.name for a in node.names] if isinstance(node, ast.Import) else []
                if isinstance(node, ast.ImportFrom):
                    modules.append(node.module or "")
                    if node.module == "isaaclab_experimental.utils.warp":
                        assert not any(a.name == "resolve_1d_mask" for a in node.names), file
                assert not {"isaaclab.assets.kernels", "isaaclab.utils.warp.math_ops"}.intersection(modules), file
            for node in tree.body:
                if isinstance(node, (ast.Assign, ast.AnnAssign)) and not file.name.endswith("kernels.py"):
                    assert not any(
                        isinstance(call, ast.Call)
                        and getattr(call.func, "id", getattr(call.func, "attr", "")) in factories
                        for call in ast.walk(node)
                    ), file
    ordering = ast.parse((core / "assets/articulation/ordering_kernels.py").read_text())
    for node in ordering.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, (ast.Dict, ast.DictComp)):
            assert not any(
                isinstance(call, ast.Call)
                and getattr(call.func, "id", getattr(call.func, "attr", "")) in {"IndexKernelDispatcher", "overload"}
                for call in ast.walk(node)
            ), "Ordering must request concrete kernels instead of enumerating dtype variants"
        if isinstance(node, ast.FunctionDef):
            assert node.name != "write_3d_user_to_backend_with_indices"
            assert not any(
                isinstance(call, ast.Call)
                and getattr(call.func, "id", getattr(call.func, "attr", "")) in {"IndexKernelDispatcher", "overload"}
                for call in ast.walk(node)
            )
