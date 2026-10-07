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
"""Significant digits to which floating model attributes are rounded before fingerprinting.

Isaac Lab derives local shape and joint transforms from USD world transforms, so their last bits depend
on where the first environment sits in the environment grid, and therefore on the number of
environments. Rounding removes that noise, so recording and deployment may use different environment
counts, while relative differences above about one part per million still change the identity.
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
_FLOAT_SCALARS = (wp.float16, wp.float32, wp.float64)


def isaaclab_model_signature(model: newton.Model) -> NewtonModelSignature:
    """Return the identity of an Isaac Lab Newton model, independent of the active solver manager.

    Recording, training, and deployment use this identity, so an artifact trained on data recorded with
    a numerical teacher solver matches the same scene stepped by the learned solver. It is the
    PhysicsNeMo Robotics :meth:`~physicsnemo_robotics.newton.NewtonModelSignature.from_model` signature of
    a canonical view of the model:

    * Isaac Lab per-world prim prefixes (``/World/envs/env_3/...``) are rewritten to the ``env_3/...``
      world prefix that PhysicsNeMo Robotics removes;
    * solver-private custom attributes listed in :data:`SOLVER_ATTRIBUTE_NAMESPACES` are ignored;
    * floating model attributes are rounded to :data:`SIGNATURE_SIGNIFICANT_DIGITS` significant digits,
      and values below :data:`SIGNATURE_ZERO_TOLERANCE` times the largest magnitude of their attribute
      count as zero.

    The model is restored before the function returns.

    Args:
        model: Finalized Newton model built by Isaac Lab.

    Returns:
        The model signature of the canonical view.
    """
    with _canonical_view(model):
        return NewtonModelSignature.from_model(model)


@contextmanager
def _canonical_view(model: newton.Model) -> Iterator[None]:
    """Temporarily present the canonical view described in :func:`isaaclab_model_signature`."""
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
        if isinstance(array, wp.array) and array.size and _scalar_type(array.dtype) in _FLOAT_SCALARS:
            saved_arrays[name] = array
            setattr(model, name, wp.array(_round_significant(array.numpy()), dtype=array.dtype, device=array.device))
    try:
        yield
    finally:
        for name, array in saved_arrays.items():
            setattr(model, name, array)
        model.attribute_frequency = frequencies
        for name, labels in saved_labels.items():
            setattr(model, name, labels)


def _round_significant(values: np.ndarray) -> np.ndarray:
    """Round to :data:`SIGNATURE_SIGNIFICANT_DIGITS` digits and flush relative residues to positive zero."""
    values = values.astype(np.float64)
    magnitude = np.abs(values)
    zero = magnitude <= SIGNATURE_ZERO_TOLERANCE * magnitude.max(initial=0.0)
    exponent = np.floor(np.log10(np.where(zero, 1.0, magnitude)))
    scale = 10.0 ** (exponent - (SIGNATURE_SIGNIFICANT_DIGITS - 1))
    # Adding zero turns -0.0 into +0.0, which hash differently.
    return np.where(zero, 0.0, np.round(values / scale) * scale) + 0.0


def _scalar_type(dtype: type) -> type:
    """Return the scalar type of a Warp scalar, vector, matrix, or transform dtype."""
    # Composite Warp types carry their scalar type; scalar types are their own scalar type.
    return getattr(dtype, "_wp_scalar_type_", dtype)
