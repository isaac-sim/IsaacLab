# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests: the batched clone transform math matches the Warp transform builtins."""

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.cloner.newton_clone_utils import (
    _compose_world_xforms,
    _invert_xform,
    _quat_multiply,
    _quat_rotate,
)

_TOL = 1e-5

# Hand-picked quaternions (xyzw) that cover identity and the 180-degree cases where a sign
# slip is invisible, followed by seeded random unit quaternions.
_FIXED_QUATS = [
    (0.0, 0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, -1.0),
]


def _sample_quats(num_samples: int) -> np.ndarray:
    """Return unit xyzw quaternions: the fixed cases followed by seeded random ones."""
    rng = np.random.default_rng(0)
    random_quats = rng.normal(size=(num_samples, 4))
    random_quats /= np.linalg.norm(random_quats, axis=1, keepdims=True)
    return np.concatenate([np.array(_FIXED_QUATS), random_quats]).astype(np.float32)


def _sample_positions(num_samples: int) -> np.ndarray:
    """Return translations paired with :func:`_sample_quats`, including the origin."""
    rng = np.random.default_rng(1)
    return np.concatenate([np.zeros((len(_FIXED_QUATS), 3)), rng.uniform(-5.0, 5.0, size=(num_samples, 3))]).astype(
        np.float32
    )


def test_quat_multiply_matches_warp():
    quats = _sample_quats(16)
    for quat_a in quats:
        expected = np.array([wp.mul(wp.quat(*quat_a), wp.quat(*quat_b)) for quat_b in quats])
        np.testing.assert_allclose(_quat_multiply(np.broadcast_to(quat_a, quats.shape), quats), expected, atol=_TOL)


def test_quat_rotate_matches_warp():
    quats = _sample_quats(16)
    vectors = _sample_positions(16)
    expected = np.array([wp.quat_rotate(wp.quat(*q), wp.vec3(*v)) for q, v in zip(quats, vectors, strict=True)])
    np.testing.assert_allclose(_quat_rotate(quats, vectors), expected, atol=_TOL)


def test_compose_world_xforms_matches_transform_multiply():
    world_p = _sample_positions(16)
    world_q = _sample_quats(16)
    local_p = _sample_positions(16)
    local_q = _sample_quats(16)

    for local in np.concatenate([local_p, local_q], axis=1):
        expected = np.array(
            [
                wp.transform_multiply(
                    wp.transform(wp.vec3(*p), wp.quat(*q)), wp.transform(wp.vec3(*local[:3]), wp.quat(*local[3:]))
                )
                for p, q in zip(world_p, world_q, strict=True)
            ]
        )
        np.testing.assert_allclose(_compose_world_xforms(world_p, world_q, local), expected, atol=_TOL)


def test_invert_xform_matches_transform_inverse():
    positions = _sample_positions(16)
    quats = _sample_quats(16)
    for position, quat in zip(positions, quats, strict=True):
        expected = np.array(wp.transform_inverse(wp.transform(wp.vec3(*position), wp.quat(*quat))))
        np.testing.assert_allclose(_invert_xform(np.concatenate([position, quat])), expected, atol=_TOL)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
