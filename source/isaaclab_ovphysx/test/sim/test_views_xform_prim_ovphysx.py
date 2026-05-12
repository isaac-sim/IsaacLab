# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Real-backend tests for the OVPhysX FrameView.

Run via ``./scripts/run_ovphysx.sh -m pytest`` (kitless, no ``AppLauncher``).
"""

from __future__ import annotations

import pytest
from isaaclab_ovphysx.physics import OvPhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.sim.views import FrameView

OVPHYSX_SIM_CFG = SimulationCfg(physics=OvPhysxCfg())


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_factory_dispatches_to_ovphysx_frame_view(device):
    """``FrameView(...)`` under an OVPhysX ``SimulationContext`` returns an ``OvPhysxFrameView``."""
    OVPHYSX_SIM_CFG.device = device
    with build_simulation_context(device=device, sim_cfg=OVPHYSX_SIM_CFG, add_ground_plane=True):
        # Define a plain Xform prim so the pattern matches at least one prim.
        stage = sim_utils.get_current_stage()
        prim = stage.DefinePrim("/World/marker", "Xform")
        sim_utils.standardize_xform_ops(prim)

        from isaaclab_ovphysx.sim.views import OvPhysxFrameView

        view = FrameView("/World/marker", device=device)
        assert isinstance(view, OvPhysxFrameView), f"Expected OvPhysxFrameView, got {type(view).__name__}"


def test_view_raises_before_physics_ready():
    """A view constructed before PHYSICS_READY raises a clear error on pose-method calls."""
    device = "cpu"
    OVPHYSX_SIM_CFG.device = device
    with build_simulation_context(device=device, sim_cfg=OVPHYSX_SIM_CFG, add_ground_plane=False):
        stage = sim_utils.get_current_stage()
        prim = stage.DefinePrim("/World/marker_pre", "Xform")
        sim_utils.standardize_xform_ops(prim)
        view = FrameView("/World/marker_pre", device=device)
        if hasattr(view, "_site_body"):
            pytest.skip("PHYSICS_READY already fired; cannot exercise the deferred-init path here.")
        with pytest.raises(RuntimeError, match="used before initialization"):
            view.get_world_poses()


def test_view_errors_when_newton_model_not_required():
    """If the scene declares no Newton-model requirement, the view raises a hint.

    ``build_simulation_context`` does not accept ``scene_data_requirements`` directly;
    instead we call ``sim.update_scene_data_requirements`` after context creation to
    reset the requirement flag to ``False``.  The actual class is
    ``SceneDataRequirement`` (singular) from
    ``isaaclab.physics.scene_data_requirements``.

    Two error paths are possible depending on whether ``PHYSICS_READY`` fires:

    * If the event fires synchronously (full OVPhysX context), the
      ``_on_physics_ready`` callback sees ``get_newton_state() is None`` and raises
      a ``RuntimeError`` containing ``"requires_newton_model=True"``.
    * If the event does not fire (SDP not yet ready at construction time and no
      step is issued), ``_require_initialized`` raises
      ``"OvPhysxFrameView used before initialization"``.

    The ``match`` regex covers both paths.
    """
    from isaaclab.physics.scene_data_requirements import SceneDataRequirement

    device = "cpu"
    OVPHYSX_SIM_CFG.device = device
    with build_simulation_context(device=device, sim_cfg=OVPHYSX_SIM_CFG, add_ground_plane=False) as sim:
        # Override the scene-data requirement so that requires_newton_model=False.
        # This causes get_newton_state() to return None even after PHYSICS_READY.
        sim.update_scene_data_requirements(SceneDataRequirement(requires_newton_model=False))

        stage = sim_utils.get_current_stage()
        prim = stage.DefinePrim("/World/marker_noreq", "Xform")
        sim_utils.standardize_xform_ops(prim)
        with pytest.raises(RuntimeError, match="used before initialization|requires_newton_model"):
            FrameView("/World/marker_noreq", device=device).get_world_poses()


# ==================================================================
# Shared FrameView contract suite
# ==================================================================

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "isaaclab" / "test" / "sim"))

import torch  # noqa: E402
import warp as wp  # noqa: E402
from frame_view_contract_utils import *  # noqa: F401, F403, E402 -- import all contract tests
from frame_view_contract_utils import CHILD_OFFSET, ViewBundle  # noqa: E402

from pxr import Gf  # noqa: E402

from isaaclab.assets import RigidObjectCfg  # noqa: E402
from isaaclab.physics.scene_data_requirements import SceneDataRequirement  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402


@configclass
class _OvPhysxFrameViewSceneCfg(InteractiveSceneCfg):
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )


@pytest.fixture
def view_factory():
    """OVPhysX factory: CameraMount child Xform at CHILD_OFFSET under each Cube body."""
    from isaaclab_ovphysx.sim.views import OvPhysxFrameView  # noqa: PLC0415

    contexts: list = []

    def _build(num_envs: int, device: str) -> ViewBundle:
        OVPHYSX_SIM_CFG.device = device
        ctx = build_simulation_context(device=device, sim_cfg=OVPHYSX_SIM_CFG, add_ground_plane=True)
        sim = ctx.__enter__()
        sim._app_control_on_stop_handle = None
        contexts.append(ctx)

        InteractiveScene(_OvPhysxFrameViewSceneCfg(num_envs=num_envs, env_spacing=2.0))

        stage = sim_utils.get_current_stage()
        for i in range(num_envs):
            prim = stage.DefinePrim(f"/World/envs/env_{i}/Cube/CameraMount", "Xform")
            sim_utils.standardize_xform_ops(prim)
            prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*CHILD_OFFSET))
            prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))

        # Activate Newton model sync so get_newton_state() returns body_q.
        sim.update_scene_data_requirements(SceneDataRequirement(requires_newton_model=True))
        sim.reset()
        view = OvPhysxFrameView("/World/envs/env_.*/Cube/CameraMount", device=device)

        sdp = SimulationContext.instance().initialize_scene_data_provider()
        body_labels = list(sdp._rigid_body_paths)
        cube_indices = [body_labels.index(f"/World/envs/env_{i}/Cube") for i in range(num_envs)]

        def _get_parent_pos(n: int, dev: str) -> torch.Tensor:
            body_q = sdp.get_newton_state().body_q
            torch_q = wp.to_torch(body_q).to(dev)
            return torch_q[cube_indices, :3].clone()

        def _set_parent_pos(positions: torch.Tensor, n: int) -> None:
            body_q = sdp.get_newton_state().body_q
            torch_q = wp.to_torch(body_q)
            torch_q[cube_indices, :3] = positions.to(torch_q.device, torch_q.dtype)

        return ViewBundle(
            view=view,
            get_parent_pos=_get_parent_pos,
            set_parent_pos=_set_parent_pos,
            teardown=lambda: None,
        )

    yield _build

    for cm in contexts:
        cm.__exit__(None, None, None)
