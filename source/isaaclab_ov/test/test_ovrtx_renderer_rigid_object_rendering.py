# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVRTX adapter for the shared rigid-object rendering contract."""

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from isaaclab.sim import SimulationCfg, build_simulation_context

_CONTRACT_DIR = Path(__file__).resolve().parents[2] / "isaaclab" / "test" / "renderers"
if str(_CONTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(_CONTRACT_DIR))

from rigid_object_rendering_contract import (  # noqa: E402
    RigidObjectRenderingBackend,
    run_rigid_object_scale_and_pose_rendering_contract,
)

_REQUIRED_MODULES = ("isaaclab_ov", "ovrtx", "ovphysx")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]
_OVSTAGE_AVAILABLE = importlib.util.find_spec("ovstage") is not None

pytestmark = [
    pytest.mark.integration,
    pytest.mark.rendering,
    pytest.mark.isaacsim_ci,
    pytest.mark.skipif(bool(_MISSING_MODULES), reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}"),
]

if not _MISSING_MODULES:
    from isaaclab_ov.physics import OvPhysxCfg  # noqa: E402
    from isaaclab_ov.renderers import OVRTXRendererCfg  # noqa: E402
else:
    OVRTXRendererCfg = None
    OvPhysxCfg = None


def test_kinematic_rigid_object_scale_and_pose_are_rendered(monkeypatch: pytest.MonkeyPatch) -> None:
    """An isolated rendering OVStage preserves native physics cloning and rendered poses and scale."""
    import ovphysx
    import ovrtx
    import ovstage

    monkeypatch.delenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", raising=False)
    copies, physics_copies, attached = [], [], []
    clone_physics = ovphysx.PhysX.clone
    clone = ovstage.Stage.clone
    attach_physics = ovphysx.PhysX.attach_ovstage
    attach_renderer = ovrtx.Renderer.attach_ovstage

    def record_clone(stage, source, targets, **kwargs):
        copies.extend((source, target) for target in targets)
        return clone(stage, source, targets, **kwargs)

    def attach_physx(physics, stage, **kwargs):
        attached.append(stage)
        return attach_physics(physics, stage, **kwargs)

    def attach_rtx(renderer, stage):
        assert len(attached) == 1 and attached[0] is not stage
        return attach_renderer(renderer, stage)

    def record_physics_clone(*args, **kwargs):
        physics_copies.append((args, kwargs))
        return clone_physics(*args, **kwargs)

    def reject_native_clone(*args, **kwargs):
        pytest.fail("OVRTX must use its isolated OVStage cloning path")

    monkeypatch.setattr(ovstage.Stage, "clone", record_clone)
    monkeypatch.setattr(ovphysx.PhysX, "clone", record_physics_clone)
    monkeypatch.setattr(ovrtx.Renderer, "clone_usd", reject_native_clone)
    monkeypatch.setattr(ovphysx.PhysX, "attach_ovstage", attach_physx)
    monkeypatch.setattr(ovrtx.Renderer, "attach_ovstage", attach_rtx)
    sim_cfg = SimulationCfg(device="cuda:0", gravity=(0.0, 0.0, 0.0), physics=OvPhysxCfg())

    @contextmanager
    def simulation():
        with build_simulation_context(sim_cfg=sim_cfg) as sim:
            yield sim
            from isaaclab_ov.cloner import OvPhysxReplicateContext, OvrtxReplicateContext, OvstageReplicateContext

            assert OvstageReplicateContext in sim.clone_contexts
            assert OvPhysxReplicateContext in sim.clone_contexts
            assert OvrtxReplicateContext not in sim.clone_contexts

    run_rigid_object_scale_and_pose_rendering_contract(
        RigidObjectRenderingBackend(
            name="OVPhysX + isolated OVRTX stage",
            simulation_context_factory=simulation,
            renderer_cfg=OVRTXRendererCfg(),
            with_articulation=True,
        )
    )
    assert physics_copies
    assert copies and len(copies) == len(set(copies))
