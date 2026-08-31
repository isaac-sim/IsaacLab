# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Isaac Lab's pin of Newton's joint position target layout."""

import isaaclab_newton  # noqa: F401  # applies the pin on import
import newton

_UNIT_INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def test_floating_base_position_targets_use_dof_layout():
    """Joint position targets stay DOF-shaped downstream of a free joint."""
    builder = newton.ModelBuilder()
    base = builder.add_link(mass=1.0, inertia=_UNIT_INERTIA)
    link = builder.add_link(mass=1.0, inertia=_UNIT_INERTIA)
    free_joint = builder.add_joint_free(child=base)
    revolute = builder.add_joint_revolute(parent=base, child=link, axis=newton.Axis.Z)
    builder.add_articulation([free_joint, revolute])
    model = builder.finalize(device="cpu")

    # The free joint stores more coordinates than DOFs, so the two layouts disagree past it.
    assert model.joint_coord_count != model.joint_dof_count
    assert model.joint_target_q.shape == (model.joint_dof_count,)
