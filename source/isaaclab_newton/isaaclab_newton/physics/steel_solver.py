# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton Steel solver integration."""

from __future__ import annotations

import numpy as np
import warp as wp
from newton import Model
from newton.solvers import SolverBase, SolverSteel

from .newton_solver import NewtonSolver
from .steel_solver_cfg import SteelSolverCfg


class SteelSolver(NewtonSolver):
    """Build and drive Newton's reduced-coordinate Steel solver."""

    solver_class = SolverSteel
    deterministic = True

    def build(self, model: Model, mode: wp.DeterministicMode, env_template: str | None = None) -> SolverBase:
        """Construct Steel after applying the model edits its configuration selects."""
        cfg: SteelSolverCfg = self.cfg
        if not cfg.contact_torsion:
            model.steel.min_torsional_patch_radius.zero_()
        if cfg.filter_fixed_root_contacts:
            _drop_fixed_root_static_pairs(model)
        return super().build(model, mode, env_template)


def _drop_fixed_root_static_pairs(model: Model) -> None:
    """Remove contact pairs between a static shape and a body welded to the world by a zero-DOF root joint."""
    pairs = model.shape_contact_pairs
    if pairs is None:
        return
    qd_start = model.joint_qd_start.numpy()
    welded_roots = (model.joint_parent.numpy() < 0) & (qd_start[1:] == qd_start[:-1])
    # The extra trailing entry stays False and answers the static body index -1.
    welded_body = np.zeros(model.body_count + 1, dtype=bool)
    welded_body[model.joint_child.numpy()[welded_roots]] = True
    shape_body = model.shape_body.numpy()
    static = shape_body < 0
    on_welded_root = welded_body[shape_body]
    shapes = pairs.numpy()
    a, b = shapes[:, 0], shapes[:, 1]
    drop = (on_welded_root[a] & static[b]) | (static[a] & on_welded_root[b])
    model.shape_contact_pairs = wp.array(shapes[~drop], dtype=wp.vec2i, device=model.device)
