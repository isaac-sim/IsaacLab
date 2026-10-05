# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVRTX adapter for the shared rigid-object rendering contract."""

import importlib.util
import sys
from pathlib import Path

import pytest

from isaaclab.sim import SimulationCfg, build_simulation_context

_CONTRACT_DIR = Path(__file__).resolve().parents[2] / "isaaclab" / "test" / "renderers"
if str(_CONTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(_CONTRACT_DIR))

from rigid_object_rendering_contract import (  # noqa: E402
    RigidObjectRenderingBackend,
    run_rigid_object_long_motion_rendering_contract,
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


def _ovrtx_backend(stage_mode: str) -> RigidObjectRenderingBackend:
    """Return the OVRTX renderer with OVPhysX physics as a contract backend."""
    assert OVRTXRendererCfg is not None
    assert OvPhysxCfg is not None
    sim_cfg = SimulationCfg(device="cuda:0", gravity=(0.0, 0.0, 0.0), physics=OvPhysxCfg())
    return RigidObjectRenderingBackend(
        name=f"ovrtx (OVPhysX, {stage_mode})",
        simulation_context_factory=lambda: build_simulation_context(sim_cfg=sim_cfg),
        renderer_cfg=OVRTXRendererCfg(),
    )


@pytest.mark.parametrize(
    "use_ovstage",
    [
        pytest.param(False, id="legacy"),
        pytest.param(
            True,
            id="ovstage",
            marks=pytest.mark.skipif(not _OVSTAGE_AVAILABLE, reason="requires optional module: ovstage"),
        ),
    ],
)
def test_kinematic_rigid_object_scale_and_pose_are_rendered(monkeypatch: pytest.MonkeyPatch, use_ovstage: bool) -> None:
    """Kinematic OVPhysX transforms and root scale must reach OVRTX."""
    monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", str(int(use_ovstage)))
    run_rigid_object_scale_and_pose_rendering_contract(_ovrtx_backend("ovstage" if use_ovstage else "legacy"))


def test_kinematic_rigid_object_motion_is_rendered_beyond_500_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    """Object-only motion must keep reaching OVRTX color output past the RTX eco-mode idle limit."""
    monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", "0")
    run_rigid_object_long_motion_rendering_contract(_ovrtx_backend("legacy"))
