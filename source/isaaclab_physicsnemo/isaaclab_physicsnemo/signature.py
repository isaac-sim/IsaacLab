# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Solver-independent Newton model identity for Isaac Lab scenes."""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager

import newton
import numpy as np
import warp as wp
from physicsnemo_robotics.newton import NewtonModelSignature

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
      as zero.

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
    saved_arrays = {}
    for name in model.attribute_frequency:
        array = getattr(model, name, None) if ":" not in name else None
        if isinstance(array, wp.array) and array.size and wp.types.type_is_transformation(array.dtype):
            saved_arrays[name] = array
            values = _canonical_plane_placements(model, array.numpy()) if name == "shape_transform" else array.numpy()
            setattr(model, name, wp.array(_round_significant(values), dtype=array.dtype, device=array.device))
    try:
        yield
    finally:
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


def _round_significant(values: np.ndarray) -> np.ndarray:
    """Round to :data:`SIGNATURE_SIGNIFICANT_DIGITS` digits and flush relative residues to positive zero."""
    values = values.astype(np.float64)
    magnitude = np.abs(values)
    zero = magnitude <= SIGNATURE_ZERO_TOLERANCE * magnitude.max(initial=0.0)
    exponent = np.floor(np.log10(np.where(zero, 1.0, magnitude)))
    scale = 10.0 ** (exponent - (SIGNATURE_SIGNIFICANT_DIGITS - 1))
    # Adding zero turns -0.0 into +0.0, which hash differently.
    return np.where(zero, 0.0, np.round(values / scale) * scale) + 0.0
