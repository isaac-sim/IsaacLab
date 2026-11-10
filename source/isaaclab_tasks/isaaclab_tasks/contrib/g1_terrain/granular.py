# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime GPU/readback setup for the pre-authored PhysX particle course."""

import torch

from isaaclab.sim import SimulationContext
from isaaclab.terrains import TerrainImporter


class ParticleTerrainImporter(TerrainImporter):
    """Load the particle USD and enable GPU simulation with host state buffers."""

    def import_usd(self, name: str, usd_path: str):
        from pxr import PhysxSchema  # noqa: PLC0415

        super().import_usd(name, usd_path)
        sim = SimulationContext.instance()
        sim.set_setting("/physics/cudaDevice", torch.cuda.current_device())
        sim.set_setting("/physics/suppressReadback", False)
        scene = PhysxSchema.PhysxSceneAPI(sim.stage.GetPrimAtPath(sim.cfg.physics_prim_path))
        scene.CreateEnableGPUDynamicsAttr(True)
        scene.CreateBroadphaseTypeAttr("GPU")
