# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from newton import ModelBuilder
from newton.solvers import SolverVBD

from isaaclab_contrib.deformable.vbd_manager import NewtonVBDManager


def test_prepare_builder_colors_rigid_bodies():
    builder = ModelBuilder()
    builder.add_rod(
        positions=[(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0)],
        radius=0.01,
        label="/World/Cable",
        body_frame_origin="com",
    )

    assert builder.body_count == 2
    assert not builder.body_color_groups

    NewtonVBDManager._prepare_builder_for_finalize(builder)

    assert builder.body_color_groups
    model = builder.finalize()
    SolverVBD(model)
