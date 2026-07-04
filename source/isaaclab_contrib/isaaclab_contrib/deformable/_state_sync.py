# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""State synchronization helpers for coupled Newton solvers."""

import warp as wp
from newton import Model, State


@wp.kernel(enable_backward=False)
def _rebase_rigid_body_history(
    fk_mask: wp.array(dtype=wp.bool),
    joint_articulation: wp.array(dtype=wp.int32),
    joint_child: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transformf),
    body_qd: wp.array(dtype=wp.spatial_vectorf),
    body_q_prev: wp.array(dtype=wp.transformf),
    body_qd_prev: wp.array(dtype=wp.spatial_vectorf),
):
    """Copy current rigid state into history for authored articulations."""
    joint = wp.tid()
    articulation = joint_articulation[joint]
    if articulation >= 0 and fk_mask[articulation]:
        body = joint_child[joint]
        body_q_prev[body] = body_q[body]
        body_qd_prev[body] = body_qd[body]


def rebase_rigid_body_history(model: Model, state: State, state_prev: State, fk_mask: wp.array) -> None:
    """Rebase coupled finite-difference history after authored rigid state writes."""
    if model.joint_count:
        wp.launch(
            _rebase_rigid_body_history,
            dim=model.joint_count,
            inputs=[
                fk_mask,
                model.joint_articulation,
                model.joint_child,
                state.body_q,
                state.body_qd,
            ],
            outputs=[state_prev.body_q, state_prev.body_qd],
            device=state.body_q.device,
        )
