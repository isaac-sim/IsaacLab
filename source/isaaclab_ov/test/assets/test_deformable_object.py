# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal real-OVPhysX integration coverage for deformable objects."""

from __future__ import annotations

import pytest
import torch
import warp as wp

pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

from isaaclab_ov import tensor_types as TT  # noqa: E402
from isaaclab_ov.physics import OvPhysxCfg, OvPhysxManager  # noqa: E402
from isaaclab_physx.sim.spawners.materials import PhysxDeformableBodyMaterialCfg  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import DeformableObject, DeformableObjectCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, build_simulation_context  # noqa: E402

from ..deformable_utils import (  # noqa: E402
    pre_tetrahedralized_deformable_spawn_cfg,
    pretriangulated_surface_deformable_spawn_cfg,
)

pytestmark = pytest.mark.integration


def _sim_context():
    """Build the CUDA OVPhysX context required by deformable bindings."""
    return build_simulation_context(
        sim_cfg=SimulationCfg(physics=OvPhysxCfg(), device="cuda:0", dt=0.01),
        auto_add_lighting=False,
    )


def _spawn_deformables(spawn: sim_utils.SpawnerCfg) -> DeformableObject:
    """Author two local deformables in an in-memory USD stage."""
    for index in range(2):
        sim_utils.create_prim(f"/World/Env_{index}", "Xform", translation=(index * 1.0, 0.0, 1.0))
    return DeformableObject(
        DeformableObjectCfg(
            prim_path="/World/Env_[^/]*/Object",
            spawn=spawn,
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        )
    )


def _canonical_connectivity(connectivity: torch.Tensor) -> list[set[tuple[int, ...]]]:
    """Convert each body's topology into order-independent literal elements."""
    return [{tuple(sorted(element)) for element in body} for body in connectivity.cpu().tolist()]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OVPhysX deformables require CUDA")
@pytest.mark.isaacsim_ci
def test_volume_deformable_real_ovphysx_seams() -> None:
    """Prove volume topology, partial state/target/material writes, and stepping."""
    with _sim_context() as sim:
        material_path = "/World/Env_0/ObjectSiblingMaterial"
        deformable = _spawn_deformables(pre_tetrahedralized_deformable_spawn_cfg(material_path=material_path))
        distractor_cfg = PhysxDeformableBodyMaterialCfg()
        distractor_cfg.func("/World/Env_1/ObjectSiblingMaterial", distractor_cfg)
        sim.reset()

        assert deformable.is_initialized
        assert deformable.num_instances == 2
        assert deformable.data.nodal_state_w.torch.shape == (2, 5, 6)
        topology = wp.to_torch(deformable.root_view.get_attribute(TT.DEFORMABLE_SIM_ELEMENT_INDICES))
        assert _canonical_connectivity(topology) == [
            {(0, 1, 2, 3), (1, 2, 3, 4)},
            {(0, 1, 2, 3), (1, 2, 3, 4)},
        ]

        initial_pos = deformable.data.nodal_pos_w.torch.clone()
        updated_pos = initial_pos[1:2].clone()
        updated_pos[..., 0] += 0.025
        deformable.write_nodal_pos_to_sim_index(updated_pos, env_ids=torch.tensor([1], device="cuda:0"))
        readback_pos = wp.to_torch(deformable.root_view.get_attribute(TT.DEFORMABLE_SIM_NODAL_POSITION))
        torch.testing.assert_close(readback_pos[0], initial_pos[0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(readback_pos[1], updated_pos[0], rtol=1e-5, atol=1e-5)

        targets = deformable.data.nodal_kinematic_target
        assert targets is not None
        updated_targets = targets.torch[1:2].clone()
        updated_targets[..., :3] = readback_pos[1:2] + torch.tensor([0.0, 0.0, 0.03], device="cuda:0")
        updated_targets[..., 3] = 0.0
        deformable.write_nodal_kinematic_target_to_sim_index(
            updated_targets, env_ids=torch.tensor([1], device="cuda:0")
        )
        readback_targets = wp.to_torch(deformable.root_view.get_attribute(TT.DEFORMABLE_SIM_KINEMATIC_TARGET))
        torch.testing.assert_close(readback_targets[0, :, 3], torch.ones_like(readback_targets[0, :, 3]))
        torch.testing.assert_close(readback_targets[1], updated_targets[0], rtol=1e-5, atol=1e-5)

        material_view = deformable.material_physx_view
        assert material_view is not None
        assert material_view.count == 1
        youngs = torch.tensor([1500.0])
        material_view.set_attribute(
            TT.DEFORMABLE_MATERIAL_YOUNGS_MODULUS,
            wp.from_torch(youngs),
            indices=wp.array([0], dtype=wp.int32),
        )
        torch.testing.assert_close(
            wp.to_torch(material_view.get_attribute(TT.DEFORMABLE_MATERIAL_YOUNGS_MODULUS)), youngs
        )

        original_view = deformable.root_view
        original_position_binding = original_view.binding_for(TT.DEFORMABLE_SIM_NODAL_POSITION)
        OvPhysxManager._warmup_done = False
        sim.reset()
        assert deformable.is_initialized
        assert deformable.root_view is not original_view
        assert deformable.root_view.binding_for(TT.DEFORMABLE_SIM_NODAL_POSITION) is not original_position_binding
        assert torch.isfinite(deformable.data.nodal_state_w.torch).all()

        sim.step()
        deformable.update(sim.cfg.dt)
        assert torch.isfinite(deformable.data.nodal_state_w.torch).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OVPhysX deformables require CUDA")
@pytest.mark.isaacsim_ci
def test_surface_deformable_real_ovphysx_seams() -> None:
    """Prove surface topology/state/material, rejected targets, and stepping."""
    with _sim_context() as sim:
        deformable = _spawn_deformables(pretriangulated_surface_deformable_spawn_cfg())
        sim.reset()

        assert deformable.is_initialized
        assert deformable.data.nodal_state_w.torch.shape == (2, 4, 6)
        topology = wp.to_torch(deformable.root_view.get_attribute(TT.SURFACE_DEFORMABLE_SIM_ELEMENT_INDICES))
        assert _canonical_connectivity(topology) == [{(0, 1, 2), (0, 2, 3)}, {(0, 1, 2), (0, 2, 3)}]
        assert deformable.data.nodal_kinematic_target is None
        with pytest.raises(ValueError, match="Kinematic targets can only be set for volume deformable bodies"):
            deformable.write_nodal_kinematic_target_to_sim_index(torch.zeros((2, 4, 4), device="cuda:0"))

        initial_pos = deformable.data.nodal_pos_w.torch.clone()
        updated_pos = initial_pos[1:2].clone()
        updated_pos[..., 1] += 0.025
        deformable.write_nodal_pos_to_sim_index(updated_pos, env_ids=torch.tensor([1], device="cuda:0"))
        readback_pos = wp.to_torch(deformable.root_view.get_attribute(TT.SURFACE_DEFORMABLE_SIM_POSITION))
        torch.testing.assert_close(readback_pos[0], initial_pos[0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(readback_pos[1], updated_pos[0], rtol=1e-5, atol=1e-5)

        material_view = deformable.material_physx_view
        assert material_view is not None
        damping = torch.tensor([0.08, 0.04])
        material_view.set_attribute(
            TT.DEFORMABLE_MATERIAL_BENDING_DAMPING,
            wp.from_torch(damping),
            indices=wp.array([0], dtype=wp.int32),
        )
        torch.testing.assert_close(
            wp.to_torch(material_view.get_attribute(TT.DEFORMABLE_MATERIAL_BENDING_DAMPING)), damping
        )

        sim.step()
        deformable.update(sim.cfg.dt)
        assert torch.isfinite(deformable.data.nodal_state_w.torch).all()
