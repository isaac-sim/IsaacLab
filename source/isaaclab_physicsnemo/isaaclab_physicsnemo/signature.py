# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Solver-independent Newton model identity for Isaac Lab scenes."""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from numbers import Integral
from typing import Any

import newton
import numpy as np
import warp as wp
from physicsnemo_robotics.newton import NewtonModelSignature
from physicsnemo_robotics.newton import spec as _spec

SOLVER_ATTRIBUTE_NAMESPACES: tuple[str, ...] = ("mujoco:",)
"""Custom attribute namespaces that Isaac Lab solver managers register for their own solver.

The teacher solver used for recording registers these attributes on the builder, while the learned
solver manager does not. They parameterize the numerical solver rather than the modeled system, so
they are excluded from the identity shared by recorded datasets and deployed artifacts.
"""

SIGNATURE_SIGNIFICANT_DIGITS = 6
"""Significant digits to which transform model attributes are rounded before fingerprinting.

Isaac Lab derives local shape and joint transforms from USD world transforms, so their last bits depend
on where the first environment sits in the environment grid, and therefore on the number of
environments. Rounding removes that noise, so recording and deployment may use different environment
counts, while relative differences above about one part per million still change the identity. Other
floating attributes are authored directly and stay exact, because PhysicsNeMo Robotics derives
semantics such as continuous revolute joints from exact thresholds on them.
"""

DERIVED_INERTIAL_FIELDS: tuple[str, ...] = ("body_inv_mass", "body_inv_inertia")
"""Model attributes that Newton derives from the body masses and inertias when it finalizes the model.

Newton computes them in a Warp kernel on the model device, and the last bit of a float32 inverse inertia depends
on the GPU and the driver that compile it: an ANYmal-C model finalized on an L40S and on an RTX PRO 5000 differs in
one entry of one body by one unit in the last place. The signature recomputes them on the host from the masses and
inertias, which it hashes exactly, so the identity does not depend on the machine that built the model.
"""

SIGNATURE_ZERO_TOLERANCE = 1.0e-8
"""Magnitude, relative to the largest magnitude of the same attribute, below which a value counts as zero.

The same USD round-off leaves residues such as ``1e-15`` or ``7e-9`` where the authored value is zero.
Their last digits are noise, so they are flushed to zero before significant-digit rounding.
"""

# Newton model label fields that PhysicsNeMo Robotics includes in a model signature.
_LABEL_FIELDS = (
    "particle_label",
    "body_label",
    "shape_label",
    "joint_label",
    "articulation_label",
    "constraint_mimic_label",
)
_ENV_PRIM_PREFIX = re.compile(r"^/World/envs/(env_\d+)/")


def isaaclab_model_signature(model: newton.Model) -> NewtonModelSignature:
    """Return the identity of an Isaac Lab Newton model, independent of the active solver manager.

    Recording, training, and deployment use this identity, so an artifact trained on data recorded with
    a numerical teacher solver matches the same scene stepped by the learned solver. It is the
    PhysicsNeMo Robotics :meth:`~physicsnemo_robotics.newton.NewtonModelSignature.from_model` signature of
    a canonical view of the model:

    * Isaac Lab per-world prim prefixes (``/World/envs/env_3/...``) are rewritten to the ``env_3/...``
      world prefix that PhysicsNeMo Robotics removes;
    * solver-private custom attributes listed in :data:`SOLVER_ATTRIBUTE_NAMESPACES` are ignored;
    * static planes are placed at their point closest to the world origin, so a ground plane replicated
      at every environment origin is the same shape in every world;
    * transform attributes are rounded to :data:`SIGNATURE_SIGNIFICANT_DIGITS` significant digits, and
      values below :data:`SIGNATURE_ZERO_TOLERANCE` times the largest magnitude of their attribute count
      as zero;
    * the inverse masses and inertias in :data:`DERIVED_INERTIAL_FIELDS` are recomputed in double precision
      on the host from the body masses and inertias and rounded the same way.

    The model is restored before the function returns.

    Args:
        model: Finalized Newton model built by Isaac Lab.

    Returns:
        The model signature of the canonical view.
    """
    with isaaclab_canonical_view(model):
        return NewtonModelSignature.from_model(model)


@contextmanager
def isaaclab_canonical_view(model: newton.Model) -> Iterator[None]:
    """Temporarily present an Isaac Lab model in the canonical view of :func:`isaaclab_model_signature`.

    Build PhysicsNeMo Robotics objects that fingerprint model geometry, such as fixed contact probes,
    inside this context, so that their fingerprints do not depend on the number of environments. Objects
    built inside keep the canonical copies of the static arrays they cache, which describe the same
    geometry. The model's labels, attribute registry, and arrays are restored on exit. Do not step the
    simulation inside the context.

    Args:
        model: Finalized Newton model built by Isaac Lab.
    """
    saved_labels = {}
    for name in _LABEL_FIELDS:
        labels = getattr(model, name, None)
        if labels is not None:
            saved_labels[name] = labels
            setattr(model, name, [_ENV_PRIM_PREFIX.sub(r"\1/", str(label), count=1) for label in labels])
    frequencies = model.attribute_frequency
    model.attribute_frequency = {
        key: value for key, value in frequencies.items() if not str(key).startswith(SOLVER_ATTRIBUTE_NAMESPACES)
    }
    saved_canonicalizers = (_spec._canonical_labels, _spec._canonical_shape_pairs, _spec._world_ids)
    _spec._canonical_labels, _spec._canonical_shape_pairs, _spec._world_ids = (
        _canonical_labels,
        _canonical_shape_pairs,
        _world_ids,
    )
    saved_arrays = {}
    specs = getattr(model, "attribute_specs", {})
    for name in model.attribute_frequency:
        # Reading a deprecated attribute warns, and none of them is a transform.
        deprecated = getattr(specs.get(name), "deprecated", False)
        array = getattr(model, name, None) if ":" not in name and not deprecated else None
        if isinstance(array, wp.array) and array.size and wp.types.type_is_transformation(array.dtype):
            saved_arrays[name] = array
            values = _canonical_plane_placements(model, array.numpy()) if name == "shape_transform" else array.numpy()
            setattr(model, name, wp.array(_round_significant(values), dtype=array.dtype, device=array.device))
    if model.body_count:
        for name, values in zip(DERIVED_INERTIAL_FIELDS, _host_inverse_inertials(model)):
            array = getattr(model, name)
            saved_arrays[name] = array
            setattr(model, name, wp.array(_round_significant(values), dtype=array.dtype, device=array.device))
    try:
        yield
    finally:
        _spec._canonical_labels, _spec._canonical_shape_pairs, _spec._world_ids = saved_canonicalizers
        for name, array in saved_arrays.items():
            setattr(model, name, array)
        model.attribute_frequency = frequencies
        for name, labels in saved_labels.items():
            setattr(model, name, labels)


def _canonical_plane_placements(model: newton.Model, transforms: np.ndarray) -> np.ndarray:
    """Move every static plane origin to the point of the plane closest to the world origin.

    A plane is unchanged by sliding its origin within the plane, so a ground plane replicated into every
    environment at the environment origin describes the same geometry in every world.
    """
    planes = (model.shape_body.numpy() == -1) & (model.shape_type.numpy() == int(newton.GeoType.PLANE))
    if not planes.any():
        return transforms
    transforms = transforms.astype(np.float64)
    x, y, z, w = transforms[planes, 3:7].T
    # plane normal: the local z axis rotated by the (x, y, z, w) orientation
    normal = np.stack((2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)), axis=-1)
    offset = (transforms[planes, :3] * normal).sum(axis=-1, keepdims=True)
    transforms[planes, :3] = offset * normal
    return transforms


def _host_inverse_inertials(model: newton.Model) -> tuple[np.ndarray, np.ndarray]:
    """Inverse body masses and inertias in double precision, zero for massless bodies as in Newton."""
    mass = model.body_mass.numpy().astype(np.float64)
    inertia = model.body_inertia.numpy().astype(np.float64).reshape(-1, 3, 3)
    massive = mass > 0.0
    inverse_mass = np.divide(1.0, mass, out=np.zeros_like(mass), where=massive)
    invertible = massive & (np.linalg.det(inertia) != 0.0)
    inverse_inertia = np.zeros_like(inertia)
    inverse_inertia[invertible] = np.linalg.inv(inertia[invertible])
    return inverse_mass, inverse_inertia


def _round_significant(values: np.ndarray) -> np.ndarray:
    """Round to :data:`SIGNATURE_SIGNIFICANT_DIGITS` digits and flush relative residues to positive zero."""
    values = values.astype(np.float64)
    magnitude = np.abs(values)
    zero = magnitude <= SIGNATURE_ZERO_TOLERANCE * magnitude.max(initial=0.0)
    exponent = np.floor(np.log10(np.where(zero, 1.0, magnitude)))
    scale = 10.0 ** (exponent - (SIGNATURE_SIGNIFICANT_DIGITS - 1))
    # Adding zero turns -0.0 into +0.0, which hash differently.
    return np.where(zero, 0.0, np.round(values / scale) * scale) + 0.0


# PhysicsNeMo Robotics canonicalizes replicated worlds by scanning every entity once per world, which takes
# hours for the tens of thousands of environments of locomotion training. The canonical view swaps in these
# equivalents, which group entities by world in one pass and return the same values and errors.


def _canonical_labels(model: Any, name: str, value: Any) -> tuple[str, ...]:
    """Linear-time :func:`physicsnemo_robotics.newton.spec._canonical_labels`."""
    labels = tuple(str(item) for item in value)
    world_ids = _spec._world_ids(model, name.removesuffix("_label"))
    normalized = tuple(_spec._WORLD_LABEL_PREFIX.sub("", label, count=1) for label in labels)
    if world_ids is None or len(labels) != world_ids.shape[0]:
        return normalized
    owners = world_ids.tolist()
    local: list[list[str]] = [[] for _ in range(int(getattr(model, "world_count", 0) or 1))]
    for label, owner in zip(normalized, owners):
        if owner >= 0:
            local[owner].append(label)
    if any(candidate != local[0] for candidate in local[1:]):
        raise ValueError(f"Newton model field {name!r} differs across worlds")
    return tuple(label for label, owner in zip(normalized, owners) if owner in (-1, 0))


def _canonical_shape_pairs(model: Any, name: str, value: Any) -> tuple[tuple[int, int], ...]:
    """Linear-time :func:`physicsnemo_robotics.newton.spec._canonical_shape_pairs`."""
    if hasattr(value, "numpy"):
        rows = _spec._numpy_value(value)
        if rows.size == 0:
            rows = np.zeros((0, 2), dtype=np.int64)
        elif rows.ndim == 1 and rows.shape[0] == 2:
            rows = rows.reshape(1, 2)
        elif not (rows.ndim == 2 and rows.shape[1] == 2):
            raise ValueError(f"Newton model field {name!r} is not a shape-pair array")
        if rows.size and (rows.dtype == np.bool_ or not np.issubdtype(rows.dtype, np.integer)):
            raise ValueError(f"Newton model field {name!r} must contain integer shape pairs")
    else:
        try:
            items = tuple(value)
        except TypeError as error:
            raise ValueError(f"Newton model field {name!r} is not an iterable of shape pairs") from error
        try:
            pairs = [tuple(pair) for pair in items]
        except TypeError as error:
            raise ValueError(f"Newton model field {name!r} contains a non-pair value") from error
        if any(
            len(pair) != 2 or any(isinstance(end, bool) or not isinstance(end, Integral) for end in pair)
            for pair in pairs
        ):
            raise ValueError(f"Newton model field {name!r} must contain integer shape pairs")
        rows = np.array(pairs, dtype=np.int64).reshape(-1, 2)
    rows = rows.astype(np.int64)
    shape_count = int(getattr(model, "shape_count", 0) or 0)
    world_ids = _spec._world_ids(model, "shape")
    if world_ids is not None and world_ids.shape[0] != shape_count:
        raise ValueError("model.shape_world length differs from model.shape_count")
    if rows.size and (rows.min() < 0 or rows.max() >= shape_count):
        raise ValueError(f"Newton model field {name!r} contains an invalid shape index")
    shape_a, shape_b = rows[:, 0], rows[:, 1]
    if world_ids is None:
        return tuple(sorted(zip(np.minimum(shape_a, shape_b).tolist(), np.maximum(shape_a, shape_b).tolist())))

    worlds = int(getattr(model, "world_count", 0) or 1)
    is_global = world_ids == -1
    globals_before = np.concatenate(([0], np.cumsum(is_global)[:-1]))
    world_keys = np.sort(world_ids[~is_global] * shape_count + np.flatnonzero(~is_global))

    def local_index(shapes: np.ndarray, world: np.ndarray) -> np.ndarray:
        # Position of each shape among the global shapes and the shapes of ``world``, in model order.
        own_before = np.searchsorted(world_keys, world * shape_count + shapes) - np.searchsorted(
            world_keys, world * shape_count
        )
        return globals_before[shapes] + own_before

    owner_a, owner_b = world_ids[shape_a], world_ids[shape_b]
    shared = (owner_a == -1) & (owner_b == -1)
    local = ~shared & ((owner_a == -1) | (owner_b == -1) | (owner_a == owner_b))
    world = np.where(owner_a == -1, owner_b, owner_a)[local]
    pair_worlds = [world, np.repeat(np.arange(worlds), np.count_nonzero(shared))]
    pair_a = [shape_a[local], np.tile(shape_a[shared], worlds)]
    pair_b = [shape_b[local], np.tile(shape_b[shared], worlds)]
    world, first, second = (np.concatenate(parts) for parts in (pair_worlds, pair_a, pair_b))
    first, second = local_index(first, world), local_index(second, world)
    canonical = np.unique(np.stack((world, np.minimum(first, second), np.maximum(first, second)), axis=1), axis=0)
    counts = np.bincount(canonical[:, 0], minlength=worlds)
    if np.any(counts != counts[0]):
        raise ValueError(f"Newton model field {name!r} differs across worlds")
    per_world = canonical[:, 1:].reshape(worlds, int(counts[0]), 2)
    if np.any(per_world != per_world[:1]):
        raise ValueError(f"Newton model field {name!r} differs across worlds")
    return tuple(map(tuple, per_world[0].tolist()))


def _world_ids(model: Any, entity: str) -> np.ndarray | None:
    """:func:`physicsnemo_robotics.newton.spec._world_ids` with one count over all worlds."""
    value = getattr(model, f"{entity}_world", None)
    if value is None:
        return None
    result = _spec._numpy_value(value).reshape(-1).astype(np.int64, copy=False)
    world_count = int(getattr(model, "world_count", 0) or 1)
    if np.any((result < -1) | (result >= world_count)):
        raise ValueError(f"model.{entity}_world contains an invalid world id")
    local_counts = np.bincount(result[result >= 0], minlength=world_count).tolist()
    if world_count > 1 and len(set(local_counts)) != 1:
        raise ValueError(f"Newton model has non-uniform {entity} counts per world: {local_counts}")
    return result


_ORIGINAL_CANONICAL_SHAPE_PAIRS = _spec._canonical_shape_pairs
