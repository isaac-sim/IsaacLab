# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Real-backend tests for the OVPhysX FrameView.

Run via ``./scripts/run_ovphysx.sh -m pytest`` (kitless, no ``AppLauncher``).
"""

from __future__ import annotations

import numpy as np
import pytest

# The OVPhysX runtime wheel is optional. Skip gracefully when it is not installed;
# CI jobs that need OVPhysX coverage install it explicitly.
pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

from isaaclab_ov.physics import OvPhysxCfg  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab import cloner  # noqa: E402
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim import SimulationCfg, build_simulation_context  # noqa: E402
from isaaclab.sim.views import FrameView  # noqa: E402

OVPHYSX_SIM_CFG = SimulationCfg(physics=OvPhysxCfg())

pytestmark = pytest.mark.device_split


def test_view_raises_before_physics_ready():
    """A view constructed before PHYSICS_READY raises a clear error on pose-method calls."""
    device = "cpu"
    OVPHYSX_SIM_CFG.device = device
    with build_simulation_context(device=device, sim_cfg=OVPHYSX_SIM_CFG, add_ground_plane=False):
        stage = sim_utils.get_current_stage()
        prim = stage.DefinePrim("/World/marker_pre", "Xform")
        sim_utils.standardize_xform_ops(prim)
        view = FrameView("/World/marker_pre", device=device)
        # Nothing has played the simulation yet, so the view must still be uninitialized.
        assert not hasattr(view, "_site_body")
        with pytest.raises(RuntimeError, match="used before initialization"):
            view.get_world_poses()


@pytest.mark.parametrize("repeated", [False, True])
def test_world_attached_source_prim_expands_from_clone_plan(repeated):
    """Nested frames resolve every populated variant and repeated native name without cloned USD."""
    from isaaclab_ov.sim.views import OvPhysxFrameView

    device = "cpu"
    OVPHYSX_SIM_CFG.device = device
    with build_simulation_context(
        device=device, sim_cfg=OVPHYSX_SIM_CFG, auto_add_lighting=False, add_ground_plane=False
    ) as sim:
        sim._app_control_on_stop_handle = None
        scene = InteractiveScene(InteractiveSceneCfg(num_envs=12, env_spacing=2.0))
        target_env_ids = tuple(range(scene.num_envs))
        path = "/World/envs/env_[^/]+/WorldCamera"
        positions = np.zeros((scene.num_envs, 3), dtype=np.float32)
        positions[:, 0] = np.arange(scene.num_envs) * 3 + 2
        plan = cloner.make_clone_plan(
            (
                AssetBaseCfg(prim_path=path),
                AssetBaseCfg(prim_path=path),
                AssetBaseCfg(prim_path=path + "/Frame", spawn=sim_utils.SpawnerCfg()),
            ),
            ((0, 0), (1, 1)) if repeated else ((0,), (1,)),
            scene.num_envs,
            weights=(10, 2),
            positions=positions,
        )
        sim.clone_contexts[cloner.UsdReplicateContext] = cloner.UsdReplicateContext(sim.stage, plan)
        sim.set_clone_plan(plan)
        stage = sim_utils.get_current_stage()
        for env_id, offset in ((0, 0.25), (10, 0.5)):
            root = stage.DefinePrim(f"/World/envs/env_{env_id}", "Xform")
            sim_utils.standardize_xform_ops(root, translation=tuple(map(float, positions[env_id])))
            stage.DefinePrim(f"/World/envs/env_{env_id}/WorldCamera", "Xform")
            prim = stage.DefinePrim(f"/World/envs/env_{env_id}/WorldCamera/Frame", "Xform")
            sim_utils.standardize_xform_ops(prim, translation=(offset, -0.5, 1.0))
        sim.reset()

        name = "WorldCamera_1/Frame" if repeated else "WorldCamera/Frame"
        view = FrameView(f"/World/envs/env_[^/]+/{name}", device=device)

        assert isinstance(view, OvPhysxFrameView)
        assert not stage.GetPrimAtPath(f"/World/envs/env_{target_env_ids[1]}").IsValid()
        assert not stage.GetPrimAtPath(f"/World/envs/env_{target_env_ids[1]}/WorldCamera").IsValid()
        assert view.count == scene.num_envs
        assert len(view.prims) == scene.num_envs
        assert {prim.GetPath().pathString for prim in view.prims} == {
            f"/World/envs/env_{index}/WorldCamera/Frame" for index in (0, 10)
        }
        assert view.prim_paths == [f"/World/envs/env_{i}/{name}" for i in target_env_ids]
        positions, _ = view.get_world_poses()
        expected_positions = scene.env_origins + torch.tensor([0.25, -0.5, 1.0], device=device)
        expected_positions[10:, 0] += 0.25
        for world_expr, indices in (("11", [11]), ("[19]", [1, 9]), ("1[01]", [10, 11])):
            selected = FrameView(f"/World/envs/env_{world_expr}/{name}", device=device)
            assert selected.count == len(indices)
            assert selected.prim_paths == [f"/World/envs/env_{index}/{name}" for index in indices]
            torch.testing.assert_close(selected.get_world_poses()[0].torch, expected_positions[indices])
            selected.close()
    torch.testing.assert_close(positions.torch, expected_positions)


def test_reinitialization_closes_previous_root_view(monkeypatch):
    """A repeated PHYSICS_READY event closes the FrameView's previous root binding."""
    from isaaclab_ov.sim.views import OvPhysxFrameView

    events = []
    frame_view = object.__new__(OvPhysxFrameView)
    physx = object()
    replacement = object()

    class PreviousRootView:
        def close(self):
            events.append("close")

    frame_view._root_view = PreviousRootView()
    frame_view._pose_binding = object()
    monkeypatch.setattr(frame_view, "_try_get_physx", lambda: physx)

    def initialize(value):
        assert frame_view._root_view is None
        assert frame_view._pose_binding is None
        events.append(("initialize", value))
        frame_view._root_view = replacement

    monkeypatch.setattr(frame_view, "_initialize_impl", initialize)

    frame_view._on_physics_ready(None)

    assert events == ["close", ("initialize", physx)]
    assert frame_view._root_view is replacement


# Note: an earlier test ``test_view_errors_when_newton_model_not_required`` was
# removed when ``OvPhysxFrameView`` was reworked to read poses from a direct
# OVPhysX ``RIGID_BODY_POSE`` tensor binding instead of the SDP's Newton state.
# The view no longer depends on the ``NEWTON_MODEL`` scene-data requirement.


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
from isaaclab.utils import configclass  # noqa: E402


@configclass
class _OvPhysxFrameViewSceneCfg(InteractiveSceneCfg):
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=1.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
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
    from isaaclab_ov.sim.views import OvPhysxFrameView  # noqa: PLC0415

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
        view = OvPhysxFrameView("/World/envs/env_[^/]+/Cube/CameraMount", device=device)

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
