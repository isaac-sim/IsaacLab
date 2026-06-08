# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVPhysX-only unit tests for rigid-object helpers.

These tests cover OVPhysX-specific scaffolding (mock binding-set shape
contracts for ``asset_kind="rigid_object"``) that has no PhysX equivalent
and therefore does not appear in the PhysX-mirrored ``test_rigid_object.py``.
"""

from __future__ import annotations

import pytest
import warp as wp

# The CI isaaclab_ov* pattern unintentionally collects isaaclab_ovphysx tests,
# but the ovphysx wheel is not installed in that environment. Skip gracefully
# so the isaaclab_ov CI pipeline is not blocked by an unrelated dependency.
pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

from isaaclab_ovphysx import tensor_types as TT  # noqa: E402
from isaaclab_ovphysx.test.mock_interfaces.views import MockOvPhysxBindingSet  # noqa: E402

wp.init()


def test_mock_binding_set_rigid_object_shapes():
    pytest.importorskip("isaaclab_ovphysx.tensor_types").RIGID_BODY_POSE  # gates on wheel

    bindings = MockOvPhysxBindingSet(
        num_instances=4,
        num_joints=0,
        num_bodies=1,
        asset_kind="rigid_object",
    )
    assert bindings.bindings[TT.RIGID_BODY_POSE].shape == (4, 7)
    assert bindings.bindings[TT.RIGID_BODY_VELOCITY].shape == (4, 6)
    assert bindings.bindings[TT.RIGID_BODY_WRENCH].shape == (4, 9)
    assert bindings.bindings[TT.RIGID_BODY_MASS].shape == (4,)
    assert bindings.bindings[TT.RIGID_BODY_INERTIA].shape == (4, 9)
    # Articulation-only bindings must be absent
    assert TT.DOF_POSITION not in bindings.bindings
    assert TT.LINK_WRENCH not in bindings.bindings
