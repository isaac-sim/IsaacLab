# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""XPBD Newton manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

from newton import Model
from newton.solvers import SolverXPBD

from .newton_manager import NewtonManager
from .xpbd_manager_cfg import XPBDSolverCfg

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim.usd_export import UsdWriter


class NewtonXPBDManager(NewtonManager):
    """:class:`NewtonManager` specialization for the XPBD solver.

    Always uses Newton's :class:`CollisionPipeline` for contact handling.
    """

    @classmethod
    def author_fixed_configuration(cls, writer: UsdWriter, scene: InteractiveScene) -> None:
        """Export the XPBD driver's effective constructor settings."""
        import inspect
        import json

        super().author_fixed_configuration(writer, scene)
        if not writer.include_solver_settings:
            return
        solver = cls._solver
        options = {}
        for name, parameter in inspect.signature(SolverXPBD).parameters.items():
            if name in {"model", "deterministic"}:
                continue
            value = getattr(solver, name, parameter.default)
            if value is None:
                continue
            if not isinstance(value, (bool, int, float, str)):
                raise NotImplementedError(f"No XPBD export representation for {name}: {value!r}.")
            options[name] = value
        from newton.usd import PrimType, SchemaResolverNewton

        from isaaclab.assets.physics_properties import UsdAttribute

        target = SchemaResolverNewton.mapping[PrimType.SCENE]["max_solver_iterations"].name
        writer.write_attribute(
            scene.physics_scene_path, UsdAttribute(target, type_name="int"), int(options.pop("iterations"))
        )
        writer.stage.GetRootLayer().customLayerData = {
            **writer.stage.GetRootLayer().customLayerData,
            "isaaclab:newtonDriver": {"solver": "xpbd", "options": json.dumps(options)},
        }

    @classmethod
    def _create_solver(cls, model: Model, solver_cfg: XPBDSolverCfg) -> SolverXPBD:
        """Construct the configured XPBD solver."""
        return SolverXPBD(model, **cls._filter_solver_kwargs(SolverXPBD, solver_cfg))

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: XPBDSolverCfg) -> None:
        """Construct :class:`SolverXPBD` and populate the base-class slots.

        XPBD always uses Newton's :class:`CollisionPipeline` and steps with
        separate input/output states, so the flags are fixed.
        """
        NewtonManager._solver = cls._create_solver(model, solver_cfg)
        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = True
        NewtonManager._supports_rigid_body_force_input = True
