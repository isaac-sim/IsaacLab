# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exact output-equivalence tests for the batched Newton model builder.

Each test replicates the same source builders through both paths:

* legacy: :func:`~isaaclab_newton.cloner.newton_clone_utils.replicate_builder_mapping`
  followed by :func:`~isaaclab_newton.cloner.newton_clone_utils.rename_builder_labels`;
* batched: :func:`~isaaclab_newton.cloner.batched_model_builder.replicate_builder_mapping_batched`;

and asserts that the resulting builder states, finalized models, site maps, world
transforms, and Fabric body bindings are identical.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_newton.cloner.batched_model_builder import replicate_builder_mapping_batched
from isaaclab_newton.cloner.builder_diff import compare_builder_states, compare_finalized_models
from isaaclab_newton.cloner.newton_clone_utils import rename_builder_labels, replicate_builder_mapping
from newton import ModelBuilder

wp.init()

pytestmark = pytest.mark.unit

_ENV_TEMPLATE = "/World/envs/env_{}"


def _make_robot_builder(root: str, num_links: int = 3) -> ModelBuilder:
    """Build a synthetic source covering the replication-relevant entity types.

    Contains a static shape, an articulated chain with a free root joint and revolute
    joints, a world-root fixed joint, a mimic constraint, an equality-constraint row
    with body/joint/world references, and a collision filter pair.
    """
    builder = ModelBuilder(up_axis="Z", gravity=-9.81)
    builder.add_shape_plane(body=-1, label=f"{root}/ground", width=2.0, length=2.0)
    prev = -1
    joints = []
    for i in range(num_links):
        body = builder.add_link(
            xform=wp.transform((0.1 * i, 0.2, 0.3 + 0.1 * i), wp.quat_identity()),
            mass=1.0 + i,
            label=f"{root}/Robot/link{i}",
        )
        builder.add_shape_box(
            body=body,
            xform=wp.transform((0.0, 0.0, 0.05), wp.quat_identity()),
            hx=0.05,
            hy=0.04,
            hz=0.03,
            label=f"{root}/Robot/link{i}/geom",
        )
        if i == 0:
            joint = builder.add_joint_free(child=body, label=f"{root}/Robot/root_joint")
            q_start = builder.joint_q_start[joint]
            builder.joint_q[q_start : q_start + 7] = [0.05, 0.02, 0.4, 0.0, 0.0, 0.0, 1.0]
        else:
            joint = builder.add_joint_revolute(
                parent=prev, child=body, axis=(0.0, 0.0, 1.0), label=f"{root}/Robot/joint{i}"
            )
        joints.append(joint)
        prev = body
    builder.add_articulation(joints, label=f"{root}/Robot")

    table = builder.add_link(xform=wp.transform((0.5, 0.5, 0.1), wp.quat_identity()), label=f"{root}/Table")
    builder.add_shape_sphere(body=table, radius=0.1, label=f"{root}/Table/geom")
    table_joint = builder.add_joint_fixed(parent=-1, child=table, label=f"{root}/Table/joint")
    builder.add_articulation([table_joint], label=f"{root}/TableArt")

    if num_links >= 3:
        builder.add_constraint_mimic(joint0=joints[1], joint1=joints[2], coef0=1.0, coef1=-0.5, label=f"{root}/mimic0")

    reference_joint = joints[1] if len(joints) > 1 else joints[0]
    builder.add_custom_values(
        **{
            "mujoco:equality_constraint_type": 1,
            "mujoco:equality_constraint_body1": 0,
            "mujoco:equality_constraint_body2": min(1, num_links - 1),
            "mujoco:equality_constraint_joint1": reference_joint,
            "mujoco:equality_constraint_label": f"{root}/eq0",
            "mujoco:equality_constraint_world": 0,
            "mujoco:equality_constraint_target_kind": 1,
            "mujoco:equality_constraint_target": reference_joint,
        }
    )
    builder.add_shape_collision_filter_pair(0, 1)
    return builder


def _world_transforms(num_worlds: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Grid positions plus a non-identity rotation on every other world."""
    positions = torch.zeros((num_worlds, 3))
    positions[:, 0] = torch.arange(num_worlds, dtype=torch.float32) * 2.0
    positions[:, 1] = torch.arange(num_worlds, dtype=torch.float32) * -1.0
    quaternions = torch.zeros((num_worlds, 4))
    quaternions[:, 3] = 1.0
    quaternions[1::2, 2] = 0.3826834  # 45 degrees about Z
    quaternions[1::2, 3] = 0.9238795
    return positions, quaternions


def _replicate_both(
    sources: list[str],
    source_builders: dict[str, ModelBuilder],
    destinations: list[str],
    mapping: torch.Tensor,
    *,
    env_root_sites: dict[str, wp.transform] | None = None,
    source_site_indices: dict[int, dict[str, list[int]]] | None = None,
):
    """Run legacy and batched replication on identical inputs."""
    num_worlds = mapping.size(1)
    env_ids = torch.arange(num_worlds)
    positions, quaternions = _world_transforms(num_worlds)

    legacy = ModelBuilder(up_axis="Z")
    legacy_site_map, legacy_world_xforms = replicate_builder_mapping(
        legacy,
        sources,
        mapping,
        positions,
        quaternions,
        source_builders,
        source_site_indices=source_site_indices,
        env_root_sites=env_root_sites,
    )
    legacy_bindings = rename_builder_labels(legacy, sources, destinations, env_ids, mapping)

    batched = ModelBuilder(up_axis="Z")
    batched_site_map, batched_world_xforms, batched_bindings = replicate_builder_mapping_batched(
        batched,
        sources,
        destinations,
        env_ids,
        mapping,
        positions,
        quaternions,
        source_builders,
        source_site_indices=source_site_indices,
        env_root_sites=env_root_sites,
    )
    return (
        (legacy, legacy_site_map, legacy_world_xforms, legacy_bindings),
        (batched, batched_site_map, batched_world_xforms, batched_bindings),
    )


def _assert_equivalent(legacy_result, batched_result, *, finalize: bool = True):
    legacy, legacy_site_map, legacy_world_xforms, legacy_bindings = legacy_result
    batched, batched_site_map, batched_world_xforms, batched_bindings = batched_result

    errors = compare_builder_states(legacy, batched)
    assert not errors, "builder state mismatch:\n" + "\n".join(errors)

    assert legacy_site_map == batched_site_map
    assert [tuple(x) for x in legacy_world_xforms] == [tuple(x) for x in batched_world_xforms]
    assert legacy_bindings == batched_bindings

    if finalize:
        model_errors = compare_finalized_models(
            legacy.finalize(device="cpu", skip_all_validations=False),
            batched.finalize(device="cpu", skip_all_validations=False),
        )
        assert not model_errors, "finalized model mismatch:\n" + "\n".join(model_errors)


@pytest.fixture(scope="module")
def robot_source() -> ModelBuilder:
    return _make_robot_builder(_ENV_TEMPLATE.format(0))


@pytest.fixture()
def env_root_sites() -> dict[str, wp.transform]:
    return {
        "ft_0": wp.transform((0.0, 0.0, 0.5), wp.quat_identity()),
        "ft_1": wp.transform((0.1, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.3)),
    }


def test_single_world_matches_legacy(robot_source):
    source = _ENV_TEMPLATE.format(0)
    mapping = torch.ones((1, 1), dtype=torch.bool)
    legacy_result, batched_result = _replicate_both([source], {source: robot_source}, [_ENV_TEMPLATE], mapping)
    _assert_equivalent(legacy_result, batched_result)


def test_many_homogeneous_worlds_match_legacy(robot_source):
    source = _ENV_TEMPLATE.format(0)
    mapping = torch.ones((1, 64), dtype=torch.bool)
    legacy_result, batched_result = _replicate_both([source], {source: robot_source}, [_ENV_TEMPLATE], mapping)
    _assert_equivalent(legacy_result, batched_result)


def test_env_root_sites_match_legacy(robot_source, env_root_sites):
    source = _ENV_TEMPLATE.format(0)
    mapping = torch.ones((1, 16), dtype=torch.bool)
    legacy_result, batched_result = _replicate_both(
        [source], {source: robot_source}, [_ENV_TEMPLATE], mapping, env_root_sites=env_root_sites
    )
    _assert_equivalent(legacy_result, batched_result)


def test_injected_source_sites_match_legacy(robot_source, env_root_sites):
    source = _ENV_TEMPLATE.format(0)
    site_source = _make_robot_builder(source)
    site_xform = wp.transform((0.0, 0.0, 0.2), wp.quat_identity())
    site_indices = [
        site_source.add_site(body=0, xform=site_xform, label=f"{source}/Robot/link0/ft_body"),
        site_source.add_site(body=1, xform=site_xform, label=f"{source}/Robot/link1/ft_body"),
    ]
    mapping = torch.ones((1, 8), dtype=torch.bool)
    legacy_result, batched_result = _replicate_both(
        [source],
        {source: site_source},
        [_ENV_TEMPLATE],
        mapping,
        env_root_sites=env_root_sites,
        source_site_indices={id(site_source): {"ft_body": site_indices}},
    )
    _assert_equivalent(legacy_result, batched_result)


def test_heterogeneous_mapping_matches_legacy(env_root_sites):
    """Two robot variants alternating across worlds, prototypes in different envs."""
    source_a = "/World/envs/env_0/Robot"
    source_b = "/World/envs/env_1/Robot"
    variant_a = _make_robot_builder(source_a, num_links=2)
    variant_b = _make_robot_builder(source_b, num_links=4)
    num_worlds = 6
    mapping = torch.zeros((2, num_worlds), dtype=torch.bool)
    mapping[0, 0::2] = True
    mapping[1, 1::2] = True
    legacy_result, batched_result = _replicate_both(
        [source_a, source_b],
        {source_a: variant_a, source_b: variant_b},
        ["/World/envs/env_{}/Robot", "/World/envs/env_{}/Robot"],
        mapping,
        env_root_sites=env_root_sites,
    )
    _assert_equivalent(legacy_result, batched_result)


def test_multi_row_worlds_match_legacy():
    """Every world combines two rows (robot + object), as with multiple asset cfgs."""
    robot_src = "/World/envs/env_0/Robot"
    object_src = "/World/envs/env_0/Object"
    robot = _make_robot_builder(robot_src, num_links=3)
    obj = _make_robot_builder(object_src, num_links=1)
    mapping = torch.ones((2, 5), dtype=torch.bool)
    legacy_result, batched_result = _replicate_both(
        [robot_src, object_src],
        {robot_src: robot, object_src: obj},
        ["/World/envs/env_{}/Robot", "/World/envs/env_{}/Object"],
        mapping,
    )
    _assert_equivalent(legacy_result, batched_result)


def test_worlds_without_rows_match_legacy(env_root_sites):
    """Worlds not covered by any row still get sites and default gravity."""
    source = _ENV_TEMPLATE.format(0)
    robot = _make_robot_builder(source)
    mapping = torch.zeros((1, 4), dtype=torch.bool)
    mapping[0, 0] = True
    mapping[0, 2] = True
    legacy_result, batched_result = _replicate_both(
        [source], {source: robot}, [_ENV_TEMPLATE], mapping, env_root_sites=env_root_sites
    )
    _assert_equivalent(legacy_result, batched_result)


def test_destination_builder_with_global_entities_matches_legacy(robot_source):
    """Replication appends after pre-existing global (world -1) entities."""
    source = _ENV_TEMPLATE.format(0)
    mapping = torch.ones((1, 4), dtype=torch.bool)
    num_worlds = mapping.size(1)
    env_ids = torch.arange(num_worlds)
    positions, quaternions = _world_transforms(num_worlds)

    def make_dest() -> ModelBuilder:
        dest = ModelBuilder(up_axis="Z")
        dest.add_ground_plane()
        body = dest.add_body(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()), label="/World/global_body")
        dest.add_shape_sphere(body=body, radius=0.2, label="/World/global_body/geom")
        return dest

    legacy = make_dest()
    legacy_site_map, legacy_world_xforms = replicate_builder_mapping(
        legacy, [source], mapping, positions, quaternions, {source: robot_source}
    )
    legacy_bindings = rename_builder_labels(legacy, [source], [_ENV_TEMPLATE], env_ids, mapping)

    batched = make_dest()
    batched_site_map, batched_world_xforms, batched_bindings = replicate_builder_mapping_batched(
        batched, [source], [_ENV_TEMPLATE], env_ids, mapping, positions, quaternions, {source: robot_source}
    )
    _assert_equivalent(
        (legacy, legacy_site_map, legacy_world_xforms, legacy_bindings),
        (batched, batched_site_map, batched_world_xforms, batched_bindings),
    )


def test_transform_math_matches_warp():
    """The vectorized float32 transform composition matches Warp's builtins exactly."""
    from isaaclab_newton.cloner.batched_model_builder import _transform_mul

    rng = np.random.default_rng(7)
    a = rng.standard_normal((256, 7)).astype(np.float32)
    b = rng.standard_normal((256, 7)).astype(np.float32)
    # normalize quaternions
    for arr in (a, b):
        arr[:, 3:] /= np.linalg.norm(arr[:, 3:], axis=1, keepdims=True)
    out = _transform_mul(a, b)
    for i in range(a.shape[0]):
        expected = wp.transform_multiply(wp.transform(*a[i]), wp.transform(*b[i]))
        np.testing.assert_array_equal(out[i], np.array(expected), err_msg=f"row {i}")
