# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physics interface for SimulationContext."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from pxr import Gf, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.physics.physics_manager import PhysicsManager
from .interface import Interface

if TYPE_CHECKING:
    from .simulation_context import SimulationContext

logger = logging.getLogger(__name__)


class PhysicsInterface(Interface):
    """Manages USD physics scene and delegates to PhysicsManager.

    This interface handles:
    - USD physics scene creation and configuration
    - Gravity, timestep, and unit settings
    - Delegating lifecycle operations to the physics manager
    """

    def __init__(self, sim_context: "SimulationContext"):
        """Initialize physics scene and physics manager.

        Args:
            sim_context: Parent simulation context.
        """
        super().__init__(sim_context)

        # Get config reference
        self._cfg = self._sim.cfg.physics_manager_cfg
        self.physics_prim_path = self._cfg.physics_prim_path
        self.backend = "torch"

        # Get the physics manager class from config
        self.physics_manager: type[PhysicsManager] = self._cfg.create_manager()

        # Initialize USD physics scene
        self._init_usd_physics_scene()
        # Initialize the physics manager
        self.physics_manager.initialize(sim_context)

    @property
    def physics_dt(self) -> float:
        """Physics timestep."""
        return self.physics_manager.get_physics_dt()

    @property
    def rendering_dt(self) -> float:
        """Rendering timestep."""
        return self._cfg.dt * self._cfg.render_interval

    @property
    def device(self) -> str:
        """Device used for physics simulation."""
        return self.physics_manager.get_device()

    @property
    def physics_sim_view(self):
        """Physics simulation view with torch backend."""
        return self.physics_manager.get_physics_sim_view()

    def is_fabric_enabled(self) -> bool:
        """Returns whether the fabric interface is enabled."""
        return self.physics_manager.is_fabric_enabled()

    def _init_usd_physics_scene(self) -> None:
        """Create and configure the USD physics scene."""
        stage = self._sim.stage
        with sim_utils.use_stage(stage):
            # Set stage conventions for metric units
            UsdGeom.SetStageUpAxis(stage, "Z")
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)

            # Find and delete any existing physics scene
            for prim in stage.Traverse():
                if prim.GetTypeName() == "PhysicsScene":
                    sim_utils.delete_prim(prim.GetPath().pathString, stage=stage)

            # Create a new physics scene
            if stage.GetPrimAtPath(self._cfg.physics_prim_path).IsValid():
                raise RuntimeError(f"A prim already exists at path '{self._cfg.physics_prim_path}'.")

            physics_scene = UsdPhysics.Scene.Define(stage, self._cfg.physics_prim_path)

            # Pre-create gravity tensor to avoid torch heap corruption issues (torch 2.1+)
            gravity = torch.tensor(self._cfg.gravity, dtype=torch.float32, device=self._cfg.device)
            gravity_magnitude = torch.norm(gravity).item()

            # Avoid division by zero
            if gravity_magnitude == 0.0:
                gravity_direction = [0.0, 0.0, -1.0]
            else:
                gravity_direction = (gravity / gravity_magnitude).tolist()

            physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(*gravity_direction))
            physics_scene.CreateGravityMagnitudeAttr(gravity_magnitude)

    def reset(self, soft: bool = False) -> None:
        """Reset physics simulation.

        Args:
            soft: If True, skip full reinitialization.
        """
        self.physics_manager.reset(soft)

    def forward(self) -> None:
        """Update articulation kinematics for rendering."""
        self.physics_manager.forward()

    def step(self, render: bool = True) -> None:
        """Step physics simulation (physics only).

        Args:
            render: Unused, kept for interface compatibility.
        """
        self.physics_manager.step()

    def close(self) -> None:
        """Clean up physics resources."""
        self.physics_manager.close()
