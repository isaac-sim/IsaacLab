# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Literal unit tests for :class:`isaaclab.utils.wrench_composer.WrenchComposer`.

Every case uses a two-environment, two-body composer on CPU with hand-computed expected values. A body rotated a
quarter turn about +Z maps a world vector ``(x, y, z)`` to the body vector ``(y, -x, z)``; a half turn maps it to
``(-x, -y, z)``. Real-solver delivery is covered by ``test_wrench_composer_integration.py``.
"""

import ast
import inspect
from types import SimpleNamespace
from typing import get_type_hints
from unittest.mock import patch

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.utils.warp import ProxyArray
from isaaclab.utils.wrench_composer import WrenchComposer

pytestmark = pytest.mark.unit

_IDENTITY = (0.0, 0.0, 0.0, 1.0)
_QUARTER_TURN_Z = (0.0, 0.0, 2.0**-0.5, 2.0**-0.5)
_HALF_TURN_Z = (0.0, 0.0, 1.0, 0.0)
_METHODS = [
    "add_forces_and_torques_index",
    "add_forces_and_torques_mask",
    "set_forces_and_torques_index",
    "set_forces_and_torques_mask",
]
# ``(env, body)`` cell -> literal 3-vector or quaternion
_Cells = dict[tuple[int, int], tuple[float, ...]]


def test_wrench_composer_uses_asset_frame_conventions():
    """Keep frame selection as an is_global boolean, with no enum or content-classification layer."""
    tree = ast.parse(inspect.getsource(WrenchComposer))
    assert [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)] == ["WrenchComposer"]
    assert not any(
        isinstance(node, ast.Attribute) and node.attr in {"_content", "_classify"} for node in ast.walk(tree)
    )
    module_tree = ast.parse(inspect.getsource(inspect.getmodule(WrenchComposer)))
    assert not any(isinstance(node, ast.ImportFrom) and node.module == "enum" for node in ast.walk(module_tree))
    assert not hasattr(WrenchComposer, "resolve_submission")
    assert get_type_hints(WrenchComposer.get_forces_and_torques)["return"] == tuple[wp.array, wp.array, bool]


# ---------------------------------------------------------------------------
# Literal fixture helpers
# ---------------------------------------------------------------------------


class _AssetData:
    """Body poses that :class:`WrenchComposer` reads when it composes into the body frame."""

    def __init__(self, com_pos_w: torch.Tensor, link_quat_w: torch.Tensor) -> None:
        self.set_pose(com_pos_w, link_quat_w)

    def set_pose(self, com_pos_w: torch.Tensor, link_quat_w: torch.Tensor) -> None:
        self.body_com_pos_w = ProxyArray(wp.from_torch(com_pos_w, dtype=wp.vec3f))
        self.body_link_quat_w = ProxyArray(wp.from_torch(link_quat_w, dtype=wp.quatf))


def _poses(
    com: _Cells | None = None, quat: _Cells | None = None, shape: tuple[int, int] = (2, 2)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CoM positions and link quaternions: zero and identity except for the given ``(env, body)`` cells."""
    com_pos_w = torch.zeros((*shape, 3))
    link_quat_w = torch.tensor(_IDENTITY).repeat(*shape, 1)
    for cell, value in (com or {}).items():
        com_pos_w[cell] = torch.tensor(value)
    for cell, value in (quat or {}).items():
        link_quat_w[cell] = torch.tensor(value)
    return com_pos_w, link_quat_w


def _make_composer(
    com: _Cells | None = None, quat: _Cells | None = None, *, supports_world_at_com: bool = False
) -> WrenchComposer:
    """Create a two-environment, two-body CPU composer with literal body poses."""
    asset = SimpleNamespace(num_instances=2, num_bodies=2, device="cpu", data=_AssetData(*_poses(com, quat)))
    return WrenchComposer(asset, supports_world_at_com=supports_world_at_com)


def _grid(cells: _Cells, shape: tuple[int, int] = (2, 2)) -> torch.Tensor:
    """Return a ``(num_envs, num_bodies, 3)`` tensor that is zero except for the given ``(env, body)`` cells."""
    values = torch.zeros((*shape, 3))
    for cell, value in cells.items():
        values[cell] = torch.tensor(value)
    return values


def _vectors(values: torch.Tensor) -> wp.array:
    """Convert a contiguous ``(..., 3)`` tensor to a Warp vector array."""
    return wp.from_torch(values, dtype=wp.vec3f)


def _mask(values: list[bool]) -> wp.array:
    """Convert literal booleans to a CPU Warp mask array."""
    return wp.array(values, dtype=wp.bool, device="cpu")


def _apply_to_cell(composer: WrenchComposer, method: str, cell: tuple[int, int], is_global: bool, **vectors) -> None:
    """Apply literal ``forces``/``torques``/``positions`` to one ``(env, body)`` cell through any entry point."""
    env, body = cell
    if method.endswith("_index"):
        inputs = {name: _vectors(torch.tensor(value).reshape(1, 1, 3)) for name, value in vectors.items()}
        getattr(composer, method)(**inputs, env_ids=[env], body_ids=[body], is_global=is_global)
    else:
        # Fill the masked-out cells with a sentinel so a mask that leaks shows up in the result.
        inputs = {}
        for name, value in vectors.items():
            full = torch.full((2, 2, 3), 999.0)
            full[cell] = torch.tensor(value)
            inputs[name] = _vectors(full)
        env_mask, body_mask = _mask([i == env for i in range(2)]), _mask([i == body for i in range(2)])
        getattr(composer, method)(**inputs, env_mask=env_mask, body_mask=body_mask, is_global=is_global)


def _get_wrench_without_pose_reads(composer: WrenchComposer) -> tuple[wp.array, wp.array, bool]:
    """Read a fast-path wrench while rejecting the body-pose queries that composition needs."""
    with patch.object(composer._asset, "data", None):
        return composer.get_forces_and_torques()


def _assert_vectors(actual: wp.array | ProxyArray, expected: torch.Tensor, atol: float = 1.0e-6) -> None:
    """Compare a Warp vector buffer (or output proxy) with an expected tensor."""
    actual = actual.torch if isinstance(actual, ProxyArray) else wp.to_torch(actual)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0.0)


# ---------------------------------------------------------------------------
# Frames and composition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", _METHODS)
def test_local_wrench_at_position_composes_without_pose_reads(method: str) -> None:
    """Local forces at local offsets add ``cross(p, F)``; torque-only input ignores positions; no pose is read."""
    composer = _make_composer(quat={(1, 0): _QUARTER_TURN_Z}, supports_world_at_com=True)

    _apply_to_cell(
        composer, method, (1, 0), False, forces=(2.0, 0.0, 0.0), torques=(0.0, 0.0, 5.0), positions=(0.0, 3.0, 0.0)
    )
    _apply_to_cell(composer, method, (0, 1), False, torques=(1.0, 0.0, 0.0), positions=(0.0, 3.0, 0.0))

    force, torque, is_global = _get_wrench_without_pose_reads(composer)
    assert is_global is False
    _assert_vectors(force, _grid({(1, 0): (2.0, 0.0, 0.0)}))
    # cross((0, 3, 0), (2, 0, 0)) = (0, 0, -6) is added to the local torque (0, 0, 5).
    _assert_vectors(torque, _grid({(1, 0): (0.0, 0.0, -1.0), (0, 1): (1.0, 0.0, 0.0)}))


def test_add_accumulates_local_wrenches() -> None:
    composer = _make_composer()

    for force, torque in (((1.0, 2.0, 3.0), (0.0, 1.0, 0.0)), ((4.0, -1.0, 0.0), (2.0, 0.0, -3.0))):
        _apply_to_cell(composer, "add_forces_and_torques_index", (0, 1), False, forces=force, torques=torque)

    _assert_vectors(composer.out_force_b, _grid({(0, 1): (5.0, 1.0, 3.0)}))
    _assert_vectors(composer.out_torque_b, _grid({(0, 1): (2.0, 1.0, -3.0)}))


@pytest.mark.parametrize("method", _METHODS)
def test_global_wrench_at_com_uses_world_fast_path_or_rotates(method: str) -> None:
    """A world-at-CoM consumer receives the world wrench as-is; a body-frame consumer receives it rotated."""
    composers = [_make_composer(quat={(0, 0): _QUARTER_TURN_Z}, supports_world_at_com=s) for s in (True, False)]
    for composer in composers:
        _apply_to_cell(composer, method, (0, 0), True, forces=(2.0, 0.0, 0.0))
        # Positions are unused when only torques are supplied, so the wrench stays a world-at-CoM wrench.
        _apply_to_cell(composer, method, (1, 1), True, torques=(0.0, 3.0, 0.0), positions=(5.0, 5.0, 5.0))
    world_consumer, body_consumer = composers

    force, torque, is_global = _get_wrench_without_pose_reads(world_consumer)
    assert is_global is True
    _assert_vectors(force, _grid({(0, 0): (2.0, 0.0, 0.0)}))
    _assert_vectors(torque, _grid({(1, 1): (0.0, 3.0, 0.0)}))

    force, torque, is_global = body_consumer.get_forces_and_torques()
    assert is_global is False
    _assert_vectors(force, _grid({(0, 0): (0.0, -2.0, 0.0)}))
    _assert_vectors(torque, _grid({(1, 1): (0.0, 3.0, 0.0)}))
    # The body-frame outputs of a world-at-CoM composer hold the same rotated wrench.
    _assert_vectors(world_consumer.out_force_b, _grid({(0, 0): (0.0, -2.0, 0.0)}))


@pytest.mark.parametrize("method", _METHODS)
def test_global_force_at_position_uses_live_com_and_rotation(method: str) -> None:
    """World positions add torque about the live CoM of each body, even far from the origin."""
    composer = _make_composer(
        com={(0, 0): (1.0, 2.0, 3.0), (1, 1): (2000.0, 0.0, 1.0)},
        quat={(0, 0): _QUARTER_TURN_Z},
        supports_world_at_com=True,
    )
    _apply_to_cell(composer, method, (0, 0), True, forces=(2.0, 0.0, 0.0), positions=(1.0, 4.0, 3.0))
    _apply_to_cell(composer, method, (1, 1), True, forces=(0.0, 10.0, 0.0), positions=(2001.0, 0.0, 1.0))

    _assert_vectors(composer.global_force_w, _grid({(0, 0): (2.0, 0.0, 0.0), (1, 1): (0.0, 10.0, 0.0)}))
    force, torque, is_global = composer.get_forces_and_torques()
    assert is_global is False
    _assert_vectors(force, _grid({(0, 0): (0.0, -2.0, 0.0), (1, 1): (0.0, 10.0, 0.0)}))
    # Lever arms (0, 2, 0) and (1, 0, 0): world torques (0, 0, -4) and (0, 0, 10).
    _assert_vectors(torque, _grid({(0, 0): (0.0, 0.0, -4.0), (1, 1): (0.0, 0.0, 10.0)}), atol=1.0e-3)

    # Without new input, the next read uses the live pose: the CoM moved onto the force line and the body half-turned.
    composer._asset.data.set_pose(
        *_poses(com={(0, 0): (1.0, 4.0, 3.0), (1, 1): (2000.0, 0.0, 1.0)}, quat={(0, 0): _HALF_TURN_Z})
    )
    force, torque, is_global = composer.get_forces_and_torques()
    assert is_global is False
    _assert_vectors(force, _grid({(0, 0): (-2.0, 0.0, 0.0), (1, 1): (0.0, 10.0, 0.0)}))
    _assert_vectors(torque, _grid({(1, 1): (0.0, 0.0, 10.0)}), atol=1.0e-3)


def test_mixed_local_and_global_wrenches_compose_in_body_frame() -> None:
    composer = _make_composer(quat={(0, 0): _QUARTER_TURN_Z}, supports_world_at_com=True)

    _apply_to_cell(composer, "add_forces_and_torques_index", (0, 0), False, forces=(1.0, 0.0, 0.0))
    _apply_to_cell(composer, "add_forces_and_torques_index", (0, 0), True, forces=(2.0, 0.0, 0.0))

    _assert_vectors(composer.local_force_b, _grid({(0, 0): (1.0, 0.0, 0.0)}))
    _assert_vectors(composer.global_force_at_com_w, _grid({(0, 0): (2.0, 0.0, 0.0)}))
    force, _, is_global = composer.get_forces_and_torques()
    assert is_global is False
    _assert_vectors(force, _grid({(0, 0): (1.0, -2.0, 0.0)}))


# ---------------------------------------------------------------------------
# Selection, set, reset, and merge
# ---------------------------------------------------------------------------


def test_index_and_mask_selection_change_only_selected_cells() -> None:
    composer = _make_composer()

    composer.add_forces_and_torques_index(
        forces=_vectors(torch.tensor([[[1.0, 0.0, 0.0]]])), body_ids=torch.tensor([1]), env_ids=torch.tensor([0])
    )
    composer.add_forces_and_torques_mask(
        forces=_vectors(torch.tensor([[[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]], [[30.0, 0.0, 0.0], [40.0, 0.0, 0.0]]])),
        env_mask=_mask([False, True]),
        body_mask=_mask([True, False]),
    )

    _assert_vectors(composer.out_force_b, _grid({(0, 1): (1.0, 0.0, 0.0), (1, 0): (30.0, 0.0, 0.0)}))


@pytest.mark.parametrize("env_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("body_dtype", [torch.int32, torch.int64])
def test_index_dtype_combinations_preserve_selected_wrench_cells(
    env_dtype: torch.dtype, body_dtype: torch.dtype
) -> None:
    """Set, add, and reset selected cells with either index width."""
    asset = SimpleNamespace(num_instances=3, num_bodies=3, device="cpu", data=_AssetData(*_poses(shape=(3, 3))))
    composer = WrenchComposer(asset)
    env_ids = torch.tensor([2, 0], dtype=env_dtype)
    body_ids = torch.tensor([1, 2], dtype=body_dtype)
    reset_env_ids = env_ids[:1]
    set_forces_np = np.arange(1, 13, dtype=np.float32).reshape(2, 2, 3)
    set_torques_np = set_forces_np + 20.0
    add_forces_np = np.full((2, 2, 3), 100.0, dtype=np.float32)
    add_torques_np = np.full((2, 2, 3), 200.0, dtype=np.float32)

    composer.set_forces_and_torques_index(
        forces=wp.from_numpy(set_forces_np, dtype=wp.vec3f, device="cpu"),
        torques=wp.from_numpy(set_torques_np, dtype=wp.vec3f, device="cpu"),
        env_ids=env_ids,
        body_ids=body_ids,
    )
    composer.add_forces_and_torques_index(
        forces=wp.from_numpy(add_forces_np, dtype=wp.vec3f, device="cpu"),
        torques=wp.from_numpy(add_torques_np, dtype=wp.vec3f, device="cpu"),
        env_ids=env_ids,
        body_ids=body_ids,
    )

    expected_forces = np.zeros((3, 3, 3), dtype=np.float32)
    expected_torques = np.zeros_like(expected_forces)
    expected_forces[np.ix_([2, 0], [1, 2])] = set_forces_np + add_forces_np
    expected_torques[np.ix_([2, 0], [1, 2])] = set_torques_np + add_torques_np
    np.testing.assert_array_equal(composer.local_force_b.numpy(), expected_forces)
    np.testing.assert_array_equal(composer.local_torque_b.numpy(), expected_torques)

    composer.reset(env_ids=reset_env_ids)
    expected_forces[2] = 0.0
    expected_torques[2] = 0.0
    np.testing.assert_array_equal(composer.local_force_b.numpy(), expected_forces)
    np.testing.assert_array_equal(composer.local_torque_b.numpy(), expected_torques)


def _fill_all_input_buffers(composer: WrenchComposer) -> None:
    """Give every cell a distinct contribution in each of the five input buffers."""

    def every_cell(value: tuple[float, float, float]) -> wp.array:
        return _vectors(torch.tensor(value).repeat(2, 2, 1))

    composer.add_forces_and_torques_index(
        forces=every_cell((0.0, 0.0, 1.0)), positions=every_cell((0.0, 1.0, 0.0)), is_global=True
    )
    composer.add_forces_and_torques_index(forces=every_cell((1.0, 0.0, 0.0)), is_global=True)
    composer.add_forces_and_torques_index(forces=every_cell((0.0, 1.0, 0.0)), torques=every_cell((0.0, 0.0, 1.0)))


# buffer -> per-cell value written by ``_fill_all_input_buffers``
_FILLED_BUFFERS = {
    "global_force_w": (0.0, 0.0, 1.0),
    "global_torque_w": (1.0, 0.0, 0.0),  # cross((0, 1, 0), (0, 0, 1))
    "global_force_at_com_w": (1.0, 0.0, 0.0),
    "local_force_b": (0.0, 1.0, 0.0),
    "local_torque_b": (0.0, 0.0, 1.0),
}


@pytest.mark.parametrize("method", ["set_forces_and_torques_index", "set_forces_and_torques_mask"])
def test_set_clears_every_buffer_of_targeted_environments_only(method: str) -> None:
    composer = _make_composer()
    _fill_all_input_buffers(composer)

    _apply_to_cell(composer, method, (0, 1), False, forces=(5.0, 0.0, 0.0))

    for name, value in _FILLED_BUFFERS.items():
        expected = _grid({(1, 0): value, (1, 1): value})
        if name == "local_force_b":
            expected[0, 1] = torch.tensor([5.0, 0.0, 0.0])
        _assert_vectors(getattr(composer, name), expected)


@pytest.mark.parametrize("source_is_global", [False, True], ids=["local_source", "global_source"])
@pytest.mark.parametrize("selector", ["list", "slice", "mask"])
def test_partial_reset_zeros_only_selected_environments(selector: str, source_is_global: bool) -> None:
    """Partial resets keep the remaining merged wrench and its required body-frame composition."""
    composer = _make_composer(supports_world_at_com=True)
    composer.add_forces_and_torques_index(
        forces=_vectors(_grid({(0, 0): (1.0, 0.0, 0.0), (1, 0): (1.0, 0.0, 0.0)})), is_global=True
    )
    assert _get_wrench_without_pose_reads(composer)[2] is True
    source = _make_composer()
    source.add_forces_and_torques_index(
        forces=_vectors(_grid({(0, 0): (0.0, 2.0, 0.0), (1, 0): (0.0, 2.0, 0.0)})),
        positions=_vectors(_grid({(0, 0): (1.0, 0.0, 0.0), (1, 0): (1.0, 0.0, 0.0)})),
        is_global=source_is_global,
    )
    composer.add_raw_buffers_from(source)

    reset_selection = {
        "list": {"env_ids": [1]},
        "slice": {"env_ids": slice(1, None)},
        "mask": {"env_mask": _mask([False, True])},
    }
    composer.reset(**reset_selection[selector])

    assert composer.active
    assert composer._dirty
    force, torque, is_global = composer.get_forces_and_torques()
    assert is_global is False
    _assert_vectors(force, _grid({(0, 0): (1.0, 2.0, 0.0)}))
    # cross((1, 0, 0), (0, 2, 0)) = (0, 0, 2) about a CoM at the origin, in either frame.
    _assert_vectors(torque, _grid({(0, 0): (0.0, 0.0, 2.0)}))


@pytest.mark.parametrize("env_ids", [None, slice(None)], ids=["none", "full_slice"])
def test_full_reset_clears_every_buffer_and_frame_flag(env_ids: slice | None) -> None:
    composer = _make_composer(supports_world_at_com=True)
    _fill_all_input_buffers(composer)
    composer.get_forces_and_torques()
    assert composer.active

    composer.reset(env_ids=env_ids)

    assert not composer.active
    assert not composer._dirty
    for name in (*_FILLED_BUFFERS, "out_force_b", "out_torque_b"):
        _assert_vectors(getattr(composer, name), torch.zeros((2, 2, 3)))
    force, torque, is_global = _get_wrench_without_pose_reads(composer)
    assert is_global is False
    _assert_vectors(force, torch.zeros((2, 2, 3)))
    _assert_vectors(torque, torch.zeros((2, 2, 3)))
    # The frame flags were cleared too: a world-at-CoM wrench takes the world fast path again.
    _apply_to_cell(composer, "add_forces_and_torques_index", (0, 0), True, forces=(1.0, 0.0, 0.0))
    assert _get_wrench_without_pose_reads(composer)[2] is True


def test_raw_buffer_merge_accumulates_all_five_buffers_and_ignores_inactive_sources() -> None:
    # Alone, the world-at-CoM destination would take the world fast path.
    destination = _make_composer(supports_world_at_com=True)
    source = _make_composer()
    inactive_source = _make_composer()
    _apply_to_cell(destination, "add_forces_and_torques_index", (0, 0), True, forces=(2.0, 0.0, 0.0))
    _apply_to_cell(
        source, "add_forces_and_torques_index", (0, 0), False, forces=(1.0, 0.0, 0.0), torques=(0.0, 2.0, 0.0)
    )
    _apply_to_cell(source, "add_forces_and_torques_index", (0, 0), True, forces=(0.0, 0.0, 3.0))
    _apply_to_cell(
        source,
        "add_forces_and_torques_index",
        (0, 0),
        True,
        forces=(0.0, 4.0, 0.0),
        torques=(0.0, 0.0, 5.0),
        positions=(0.0, 0.0, 0.0),
    )
    # A raw write that never activated its composer must not be merged.
    wp.copy(inactive_source.local_force_b, _vectors(_grid({(1, 1): (8.0, 0.0, 0.0)})))
    assert not inactive_source.active

    destination.add_raw_buffers_from(source)
    destination.add_raw_buffers_from(inactive_source)

    _assert_vectors(destination.local_force_b, _grid({(0, 0): (1.0, 0.0, 0.0)}))
    _assert_vectors(destination.local_torque_b, _grid({(0, 0): (0.0, 2.0, 0.0)}))
    _assert_vectors(destination.global_force_w, _grid({(0, 0): (0.0, 4.0, 0.0)}))
    _assert_vectors(destination.global_force_at_com_w, _grid({(0, 0): (2.0, 0.0, 3.0)}))
    _assert_vectors(destination.global_torque_w, _grid({(0, 0): (0.0, 0.0, 5.0)}))
    # The merged frame flags (local input and world positions) require body-frame composition.
    force, torque, is_global = destination.get_forces_and_torques()
    assert is_global is False
    _assert_vectors(force, _grid({(0, 0): (3.0, 4.0, 3.0)}))
    _assert_vectors(torque, _grid({(0, 0): (0.0, 2.0, 5.0)}))

    # Merging an inactive source leaves an inactive destination inactive.
    empty_destination = _make_composer()
    empty_destination.add_raw_buffers_from(inactive_source)
    assert not empty_destination.active


# ---------------------------------------------------------------------------
# Lazy composition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("input_name, output_name", [("forces", "out_force_b"), ("torques", "out_torque_b")])
def test_output_property_triggers_lazy_composition(input_name: str, output_name: str):
    """Test that reading out_force_b/out_torque_b without explicit compose_to_body_frame returns correct results."""
    composer = _make_composer(quat={(0, 1): _QUARTER_TURN_Z})
    _apply_to_cell(composer, "add_forces_and_torques_index", (0, 1), True, **{input_name: (2.0, 0.0, 0.0)})

    # Do NOT call compose_to_body_frame -- rely on lazy composition
    assert composer._dirty
    _assert_vectors(getattr(composer, output_name), _grid({(0, 1): (0.0, -2.0, 0.0)}))
    assert not composer._dirty


def test_lazy_composition_reflects_later_adds():
    """Test that an add after a lazy composition is reflected by the next output read."""
    composer = _make_composer()
    ones = torch.ones((2, 2, 3))
    composer.add_forces_and_torques_index(forces=_vectors(ones))
    # Reading out_force_b composes lazily
    _assert_vectors(composer.out_force_b, ones)

    # Another add must be reflected by the next read of either output
    composer.add_forces_and_torques_index(forces=_vectors(ones), torques=_vectors(ones))
    _assert_vectors(composer.out_torque_b, ones)
    _assert_vectors(composer.out_force_b, 2.0 * ones)
    # Composing again without new input is idempotent.
    composer.compose_to_body_frame()
    _assert_vectors(composer.out_force_b, 2.0 * ones)
    _assert_vectors(composer.out_torque_b, ones)


# ---------------------------------------------------------------------------
# Deprecated API and input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "deprecated_name, name, input_name",
    [("composed_force", "out_force_b", "forces"), ("composed_torque", "out_torque_b", "torques")],
)
def test_composed_wrench_emits_deprecation_warning(deprecated_name: str, name: str, input_name: str):
    """Test that accessing composed_force/composed_torque emits a DeprecationWarning and aliases the output."""
    composer = _make_composer()
    _apply_to_cell(composer, "add_forces_and_torques_index", (1, 0), False, **{input_name: (1.0, 2.0, 3.0)})

    with pytest.warns(DeprecationWarning, match=f"{deprecated_name}.*is deprecated"):
        result = getattr(composer, deprecated_name)

    # The deprecated property aliases the new output property.
    assert result is getattr(composer, name)
    _assert_vectors(result, _grid({(1, 0): (1.0, 2.0, 3.0)}))


def test_deprecated_writers_warn_and_keep_add_and_set_behavior() -> None:
    composer = _make_composer()

    with pytest.warns(DeprecationWarning, match="add_forces_and_torques.*is deprecated"):
        composer.add_forces_and_torques(
            forces=_vectors(torch.tensor([[[3.0, 0.0, 0.0]]])),
            torques=_vectors(torch.tensor([[[0.0, 0.0, 2.0]]])),
            body_ids=[1],
            env_ids=[1],
        )
    _assert_vectors(composer.out_force_b, _grid({(1, 1): (3.0, 0.0, 0.0)}))
    _assert_vectors(composer.out_torque_b, _grid({(1, 1): (0.0, 0.0, 2.0)}))

    with pytest.warns(DeprecationWarning, match="set_forces_and_torques.*is deprecated"):
        composer.set_forces_and_torques(forces=_vectors(torch.tensor([[[4.0, 0.0, 0.0]]])), body_ids=[1], env_ids=[1])

    # The deprecated set replaces the earlier wrench of the targeted environment, torque included.
    _assert_vectors(composer.out_force_b, _grid({(1, 1): (4.0, 0.0, 0.0)}))
    _assert_vectors(composer.out_torque_b, torch.zeros((2, 2, 3)))


def test_invalid_selection_and_empty_wrench_input_are_reported() -> None:
    composer = _make_composer()

    with pytest.raises(TypeError, match="env_ids must be"):
        composer.add_forces_and_torques_index(forces=_vectors(torch.ones((1, 2, 3))), env_ids=(0,))
    for method in ("add_forces_and_torques_index", "set_forces_and_torques_mask"):
        with pytest.warns(UserWarning, match="No forces or torques"):
            getattr(composer, method)()
    assert not composer.active
