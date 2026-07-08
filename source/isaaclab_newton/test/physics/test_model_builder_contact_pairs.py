# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exact output-equivalence tests for the vectorized contact-pair finalize step.

Runs Newton's reference ``ModelBuilder.find_shape_contact_pairs`` and
:func:`~isaaclab_newton.physics.model_builder.find_shape_contact_pairs_vectorized`
on identical builders and asserts the resulting pair arrays are bit-identical
(same pairs, same order).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics.model_builder import NewtonModelBuilder, find_shape_contact_pairs_vectorized
from newton import ModelBuilder, ShapeFlags

wp.init()

pytestmark = pytest.mark.unit

_COLLIDE = int(ShapeFlags.COLLIDE_SHAPES)


def _make_fake_model(builder: ModelBuilder) -> SimpleNamespace:
    """Minimal stand-in for the finalize-time model the pair search writes into."""
    return SimpleNamespace(
        device="cpu",
        shape_collision_filter_pairs={(min(a, b), max(a, b)) for a, b in builder.shape_collision_filter_pairs},
        shape_contact_pairs=None,
        shape_contact_pair_count=None,
    )


def _populate_shapes(builder: ModelBuilder, worlds: list[int], groups: list[int], flags: list[int]) -> None:
    """Directly populate the shape fields the pair search reads."""
    builder.shape_world.extend(worlds)
    builder.shape_collision_group.extend(groups)
    builder.shape_flags.extend(flags)
    builder.shape_type.extend([0] * len(worlds))


def _assert_same_pairs(builder: ModelBuilder) -> None:
    reference = _make_fake_model(builder)
    ModelBuilder.find_shape_contact_pairs(builder, reference)
    vectorized = _make_fake_model(builder)
    find_shape_contact_pairs_vectorized(builder, vectorized)

    assert vectorized.shape_contact_pair_count == reference.shape_contact_pair_count
    ref_pairs = reference.shape_contact_pairs.numpy()
    vec_pairs = vectorized.shape_contact_pairs.numpy()
    assert ref_pairs.shape == vec_pairs.shape
    np.testing.assert_array_equal(ref_pairs, vec_pairs)


def test_multi_world_pairs_match_reference():
    builder = ModelBuilder()
    builder.world_count = 3
    # Two globals (one negative group), three worlds with mixed groups.
    worlds = [-1, -1] + [0] * 4 + [1] * 4 + [2] * 4
    groups = [1, -1] + [1, 1, 2, -3] * 3
    flags = [_COLLIDE] * len(worlds)
    _populate_shapes(builder, worlds, groups, flags)
    _assert_same_pairs(builder)


def test_filtered_pairs_match_reference():
    builder = ModelBuilder()
    builder.world_count = 2
    worlds = [-1] + [0] * 3 + [1] * 3
    groups = [1] * len(worlds)
    flags = [_COLLIDE] * len(worlds)
    _populate_shapes(builder, worlds, groups, flags)
    builder.shape_collision_filter_pairs.extend([(1, 2), (0, 4), (5, 6)])
    _assert_same_pairs(builder)


def test_zero_groups_and_noncolliding_flags_match_reference():
    builder = ModelBuilder()
    builder.world_count = 2
    worlds = [-1, 0, 0, 0, 1, 1, 1]
    groups = [0, 1, 0, -1, 1, 0, -2]
    flags = [_COLLIDE, _COLLIDE, 0, _COLLIDE, 0, _COLLIDE, _COLLIDE]
    _populate_shapes(builder, worlds, groups, flags)
    _assert_same_pairs(builder)


def test_heterogeneous_world_sizes_match_reference():
    builder = ModelBuilder()
    builder.world_count = 3
    worlds = [0] * 2 + [1] * 5 + [2] * 1
    groups = [1, -2, 1, 2, -1, -2, 1, 3]
    flags = [_COLLIDE] * len(worlds)
    _populate_shapes(builder, worlds, groups, flags)
    _assert_same_pairs(builder)


def test_empty_and_single_shape_match_reference():
    builder = ModelBuilder()
    _assert_same_pairs(builder)

    builder = ModelBuilder()
    builder.world_count = 1
    _populate_shapes(builder, [0], [1], [_COLLIDE])
    _assert_same_pairs(builder)


def test_globals_only_match_reference():
    builder = ModelBuilder()
    worlds = [-1] * 5
    groups = [1, 1, -1, 2, 0]
    flags = [_COLLIDE] * 5
    _populate_shapes(builder, worlds, groups, flags)
    builder.shape_collision_filter_pairs.append((0, 1))
    _assert_same_pairs(builder)


def test_fuzzed_builders_match_reference():
    rng = np.random.default_rng(42)
    for _ in range(20):
        builder = ModelBuilder()
        num_worlds = int(rng.integers(1, 6))
        builder.world_count = num_worlds
        num_shapes = int(rng.integers(0, 40))
        worlds = np.sort(rng.integers(-1, num_worlds, size=num_shapes)).tolist()
        groups = rng.integers(-3, 4, size=num_shapes).tolist()
        flags = rng.choice([0, _COLLIDE], size=num_shapes, p=[0.2, 0.8]).tolist()
        _populate_shapes(builder, worlds, groups, flags)
        if num_shapes >= 2:
            for _ in range(int(rng.integers(0, 5))):
                a, b = rng.integers(0, num_shapes, size=2)
                if a != b:
                    builder.shape_collision_filter_pairs.append((min(int(a), int(b)), max(int(a), int(b))))
        _assert_same_pairs(builder)


def _make_source_builder(root: str = "/World/envs/env_0", num_links: int = 4) -> ModelBuilder:
    builder = ModelBuilder(up_axis="Z")
    builder.add_shape_plane(body=-1, label=f"{root}/ground", width=1.0, length=1.0)
    for i in range(num_links):
        body = builder.add_link(xform=wp.transform((0.1 * i, 0.0, 0.3), wp.quat_identity()), label=f"{root}/b{i}")
        builder.add_shape_box(body=body, hx=0.05, hy=0.05, hz=0.05, label=f"{root}/b{i}/geom")
        builder.add_shape_sphere(body=body, radius=0.02, label=f"{root}/b{i}/geom2")
        joint = builder.add_joint_free(child=body, label=f"{root}/b{i}/joint")
        builder.add_articulation([joint], label=f"{root}/b{i}/art")
    builder.add_shape_collision_filter_pair(1, 3)
    builder.add_shape_collision_filter_pair(2, 4)
    return builder


def _batched_replicate(num_worlds: int, *, with_globals: bool, hetero: bool = False) -> ModelBuilder:
    import torch

    from isaaclab_newton.cloner.batched_model_builder import replicate_builder_mapping_batched

    builder = ModelBuilder(up_axis="Z")
    if with_globals:
        builder.add_ground_plane()
        global_body = builder.add_body(xform=wp.transform((0.0, 0.0, 2.0), wp.quat_identity()), label="/World/g")
        builder.add_shape_sphere(body=global_body, radius=0.5, label="/World/g/geom")
        builder.add_shape_collision_filter_pair(0, 1)

    if hetero:
        sources = ["/World/envs/env_0/Robot", "/World/envs/env_1/Robot"]
        source_builders = {s: _make_source_builder(s, num_links=2 + i) for i, s in enumerate(sources)}
        destinations = ["/World/envs/env_{}/Robot"] * 2
        mapping = torch.zeros((2, num_worlds), dtype=torch.bool)
        mapping[0, 0::2] = True
        mapping[1, 1::2] = True
    else:
        sources = ["/World/envs/env_0"]
        source_builders = {sources[0]: _make_source_builder()}
        destinations = ["/World/envs/env_{}"]
        mapping = torch.ones((1, num_worlds), dtype=torch.bool)

    positions = torch.zeros((num_worlds, 3))
    positions[:, 0] = torch.arange(num_worlds, dtype=torch.float32)
    quaternions = torch.zeros((num_worlds, 4))
    quaternions[:, 3] = 1.0
    replicate_builder_mapping_batched(
        builder,
        sources,
        destinations,
        torch.arange(num_worlds),
        mapping,
        positions,
        quaternions,
        source_builders,
    )
    return builder


def test_homogeneous_tiled_path_matches_reference():
    from isaaclab_newton.physics.model_builder import _HOMOGENEOUS_META_ATTR

    for with_globals in (False, True):
        builder = _batched_replicate(16, with_globals=with_globals)
        assert getattr(builder, _HOMOGENEOUS_META_ATTR, None) is not None
        _assert_same_pairs(builder)


def test_stale_homogeneous_metadata_falls_back():
    builder = _batched_replicate(8, with_globals=True)
    # Appending a shape after replication invalidates the recorded totals.
    builder.add_shape_sphere(body=-1, radius=0.1, label="/World/late")
    _assert_same_pairs(builder)


def test_heterogeneous_mapping_is_not_marked_homogeneous():
    from isaaclab_newton.physics.model_builder import _HOMOGENEOUS_META_ATTR

    builder = _batched_replicate(6, with_globals=True, hetero=True)
    assert getattr(builder, _HOMOGENEOUS_META_ATTR, None) is None
    _assert_same_pairs(builder)


def test_real_builder_via_subclass_matches_reference():
    """End-to-end: build real shapes on the subclass and compare full finalize output."""
    def build(cls):
        builder = cls(up_axis="Z")
        builder.add_ground_plane()
        for world in range(4):
            builder.begin_world()
            for i in range(3):
                body = builder.add_body(
                    xform=wp.transform((0.2 * i, 0.0, 0.5), wp.quat_identity()), label=f"w{world}_b{i}"
                )
                builder.add_shape_box(body=body, hx=0.05, hy=0.05, hz=0.05)
                builder.add_shape_sphere(body=body, radius=0.03)
            builder.end_world()
        return builder.finalize(device="cpu")

    reference = build(ModelBuilder)
    fast = build(NewtonModelBuilder)
    assert fast.shape_contact_pair_count == reference.shape_contact_pair_count
    np.testing.assert_array_equal(fast.shape_contact_pairs.numpy(), reference.shape_contact_pairs.numpy())
