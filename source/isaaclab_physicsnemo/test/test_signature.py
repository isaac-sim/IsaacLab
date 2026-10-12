# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Solver-independent model identity of Isaac Lab Newton scenes."""

import pytest

pytest.importorskip("physicsnemo_robotics")

import newton
import warp as wp
from isaaclab_physicsnemo import isaaclab_model_signature
from physicsnemo_robotics.newton import NewtonModelSignature


def _replicated_model(label_prefix: str, *, register_mujoco_attributes: bool = False) -> newton.Model:
    """Build two worlds of a one-joint articulation whose labels start with ``label_prefix.format(world)``."""
    scene = newton.ModelBuilder()
    if register_mujoco_attributes:
        newton.solvers.SolverMuJoCo.register_custom_attributes(scene)
    for world in range(2):
        robot = newton.ModelBuilder()
        link = robot.add_link(mass=1.0, inertia=wp.mat33(0.3, 0.01, 0.0, 0.01, 0.2, 0.0, 0.0, 0.0, 0.1), label="link")
        hinge = robot.add_joint_revolute(-1, link, label="hinge")
        robot.add_articulation([hinge], label="robot")
        scene.begin_world()
        scene.add_builder(robot, label_prefix=label_prefix.format(world))
        scene.end_world()
    return scene.finalize(device="cpu")


def test_signature_accepts_isaaclab_world_prim_paths():
    """Isaac Lab prim-path labels identify the same system as Newton's ``env_<i>/`` world labels."""
    isaaclab_model = _replicated_model("/World/envs/env_{}/Robot")
    with pytest.raises(ValueError, match="differs across worlds"):
        NewtonModelSignature.from_model(isaaclab_model)

    newton_model = _replicated_model("env_{}/Robot")
    assert isaaclab_model_signature(isaaclab_model) == isaaclab_model_signature(newton_model)
    assert isaaclab_model_signature(isaaclab_model) != isaaclab_model_signature(_replicated_model("env_{}/Other"))


def test_signature_ignores_solver_private_attributes():
    """Recording with a MuJoCo Warp teacher and deploying without it yield the same identity."""
    teacher_model = _replicated_model("/World/envs/env_{}/Robot", register_mujoco_attributes=True)
    learned_model = _replicated_model("/World/envs/env_{}/Robot")
    assert any(name.startswith("mujoco:") for name in teacher_model.attribute_frequency)
    assert isaaclab_model_signature(teacher_model) == isaaclab_model_signature(learned_model)


@pytest.mark.parametrize("offset", [1.0e-3, 1.0e-7], ids=["millimeter", "sub-micrometer"])
def test_signature_ignores_rounding_noise_but_not_physical_changes(offset):
    """Transforms that differ only in their last bits, as across environment counts, share an identity.

    Small but physical values, such as a sub-micrometer offset, still change the identity.
    """
    reference = _replicated_model("/World/envs/env_{}/Robot")
    noisy = _replicated_model("/World/envs/env_{}/Robot")
    wp.to_torch(noisy.joint_X_p)[:, 0] += 1.0e-12
    wp.to_torch(noisy.joint_X_p)[:, 1] = -0.0
    # PhysicsNeMo Robotics itself rounds world-root placements in joint_X_p, so move the child anchor.
    moved = _replicated_model("/World/envs/env_{}/Robot")
    wp.to_torch(moved.joint_X_c)[:, 0] += offset

    assert isaaclab_model_signature(noisy) == isaaclab_model_signature(reference)
    assert isaaclab_model_signature(moved) != isaaclab_model_signature(reference)


def test_signature_ignores_device_rounding_of_derived_inertials():
    """Inverse inertias that differ in their last bit, as between GPUs, share an identity; inertias do not."""
    import numpy as np

    reference = _replicated_model("/World/envs/env_{}/Robot")
    rounded = _replicated_model("/World/envs/env_{}/Robot")
    inverse = rounded.body_inv_inertia.numpy().copy()
    rounded.body_inv_inertia.assign(np.nextafter(inverse, np.inf, dtype=np.float32))
    heavier = _replicated_model("/World/envs/env_{}/Robot")
    inertia = heavier.body_inertia.numpy()
    inertia[:, 0, 0] *= 1.001
    heavier.body_inertia.assign(inertia)

    assert isaaclab_model_signature(rounded) == isaaclab_model_signature(reference)
    assert isaaclab_model_signature(heavier) != isaaclab_model_signature(reference)
    assert np.array_equal(rounded.body_inv_inertia.numpy(), np.nextafter(inverse, np.inf, dtype=np.float32))


def test_signature_leaves_the_live_model_unchanged():
    """The live model keeps its labels, solver attributes, and simulation arrays."""
    model = _replicated_model("/World/envs/env_{}/Robot", register_mujoco_attributes=True)
    labels = list(model.articulation_label)
    frequencies = model.attribute_frequency
    joint_parent_transforms = model.joint_X_p

    isaaclab_model_signature(model)

    assert list(model.articulation_label) == labels
    assert model.attribute_frequency is frequencies
    assert model.joint_X_p is joint_parent_transforms


def test_world_canonicalization_matches_physicsnemo_robotics():
    """The linear-time canonicalizers return what PhysicsNeMo Robotics returns, errors included."""
    import types

    import numpy as np
    from isaaclab_physicsnemo import signature
    from physicsnemo_robotics.newton import spec

    def outcome(function, model, name, value):
        try:
            return function(model, name, value)
        except ValueError as error:
            return f"ValueError: {error}"

    rng = np.random.default_rng(0)
    for case in range(200):
        worlds, per_world, shared = int(rng.integers(1, 5)), int(rng.integers(0, 4)), int(rng.integers(0, 3))
        owners = np.array([-1] * shared + [w for w in range(worlds) for _ in range(per_world)])
        owners = owners[rng.permutation(owners.size)]
        model = types.SimpleNamespace(shape_count=owners.size, world_count=worlds, shape_world=owners)
        if owners.size == 0:
            continue
        # Replicate one world's pairs, then occasionally break the replication or add cross-world pairs.
        pairs = []
        local = [np.flatnonzero((owners == -1) | (owners == w)) for w in range(worlds)]
        template = [tuple(rng.integers(0, local[0].size, 2)) for _ in range(int(rng.integers(0, 5)))]
        for w in range(worlds):
            pairs += [(local[w][i], local[w][j]) for i, j in template]
        if case % 5 == 0:
            pairs.append(tuple(rng.integers(0, owners.size, 2)))
        rows = np.array(pairs, dtype=np.int64).reshape(-1, 2)
        # Newton stores contact pairs as an array and collision filter pairs as a set of tuples.
        for value in (types.SimpleNamespace(numpy=lambda rows=rows: rows), {tuple(map(int, row)) for row in rows}):
            assert outcome(signature._canonical_shape_pairs, model, "shape_contact_pairs", value) == outcome(
                signature._ORIGINAL_CANONICAL_SHAPE_PAIRS, model, "shape_contact_pairs", value
            )
        assert signature._world_ids(model, "shape").tolist() == spec._world_ids(model, "shape").tolist()
        labels = [f"env_{max(owner, 0)}/shape_{index % max(per_world, 1)}" for index, owner in enumerate(owners)]
        if case % 7 == 0:
            labels[-1] = "renamed"
        assert outcome(signature._canonical_labels, model, "shape_label", labels) == outcome(
            spec._canonical_labels, model, "shape_label", labels
        )
