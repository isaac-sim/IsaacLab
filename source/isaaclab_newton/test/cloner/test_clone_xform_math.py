# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests: the batched clone transform math matches the Warp transform builtins."""

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.cloner.newton_clone_utils import _quat_rotate

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


def test_quat_rotate_matches_warp():
    quats = _sample_quats(16)
    vectors = _sample_positions(16)
    expected = np.array([wp.quat_rotate(wp.quat(*q), wp.vec3(*v)) for q, v in zip(quats, vectors, strict=True)])
    np.testing.assert_allclose(_quat_rotate(quats, vectors), expected, atol=_TOL)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
