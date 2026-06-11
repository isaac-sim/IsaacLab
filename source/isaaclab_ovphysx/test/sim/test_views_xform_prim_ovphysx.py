# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Real-backend tests for the OVPhysX FrameView.

Run via ``./scripts/run_ovphysx.sh -m pytest`` (kitless, no ``AppLauncher``).
"""

from __future__ import annotations

import pytest

# The OVPhysX runtime wheel is optional. Skip gracefully when it is not installed;
# CI jobs that need OVPhysX coverage install it explicitly.
pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

from isaaclab_ovphysx.physics import OvPhysxCfg  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim import SimulationCfg, build_simulation_context  # noqa: E402
from isaaclab.sim.views import FrameView  # noqa: E402

OVPHYSX_SIM_CFG = SimulationCfg(physics=OvPhysxCfg())

pytestmark = pytest.mark.device_split


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


# Note: an earlier test ``test_view_errors_when_newton_model_not_required`` was
# removed when ``OvPhysxFrameView`` was reworked to read poses from a direct
# OVPhysX ``RIGID_BODY_POSE`` tensor binding instead of the SDP's Newton state.
# The view no longer depends on ``requires_newton_model``.


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
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils.configclass import configclass  # noqa: E402


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
    """OVPhysX factory: CameraMount child Xform at CHILD_OFFSET under each Cube body.

    Test scaffolding note: ``OvPhysxFrameView`` reads body poses from a live
    OVPhysX ``RIGID_BODY_POSE`` tensor binding each frame. The shared contract
    tests inject synthetic parent poses via ``set_parent_pos`` and expect the
    very next ``get_world_poses`` call to reflect them -- without stepping the
    sim. To make that work, the fixture detaches the binding after one
    initial read so subsequent reads return the contents of ``_pose_buf``
    directly, and the get/set callbacks drive ``_pose_buf`` in place.
    """
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

        sim.reset()
        view = OvPhysxFrameView("/World/envs/env_.*/Cube/CameraMount", device=device)

        # Capture binding row order, populate _pose_buf once with the live spawn poses,
        # then detach the binding so subsequent reads do not overwrite the buffer.
        assert view._pose_binding is not None, "Fixture expects a non-empty pose binding."
        view._pose_binding.read(view._pose_buf)
        path_to_row = {p: i for i, p in enumerate(view._pose_binding.prim_paths)}
        view._pose_binding = None

        cube_rows = [path_to_row[f"/World/envs/env_{i}/Cube"] for i in range(num_envs)]
        pose_buf_torch = wp.to_torch(view._pose_buf)  # shape [num_bodies, 7] float32

        def _get_parent_pos(n: int, dev: str) -> torch.Tensor:
            return pose_buf_torch[cube_rows, :3].to(dev).clone()

        def _set_parent_pos(positions: torch.Tensor, n: int) -> None:
            pose_buf_torch[cube_rows, :3] = positions.to(pose_buf_torch.device, pose_buf_torch.dtype)

        return ViewBundle(
            view=view,
            get_parent_pos=_get_parent_pos,
            set_parent_pos=_set_parent_pos,
            teardown=lambda: None,
        )

    yield _build

    for cm in contexts:
        cm.__exit__(None, None, None)
