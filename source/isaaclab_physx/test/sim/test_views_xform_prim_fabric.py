# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX Fabric backend tests for FrameView.

Imports the shared contract tests and provides the Fabric-specific
``view_factory`` fixture (SimulationContext with use_fabric=True,
Camera prim type for Fabric SelectPrims compatibility).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "isaaclab" / "test" / "sim"))

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import pytest  # noqa: E402
import torch  # noqa: E402
from frame_view_contract_utils import *  # noqa: F401, F403, E402
from frame_view_contract_utils import CHILD_OFFSET, ViewBundle  # noqa: E402
from isaaclab_physx.sim.views import FabricFrameView as FrameView  # noqa: E402

from pxr import Gf, UsdGeom  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402

pytestmark = pytest.mark.isaacsim_ci
PARENT_POS = (0.0, 0.0, 1.0)


@pytest.fixture(autouse=True)
def test_setup_teardown():
    sim_utils.create_new_stage()
    sim_utils.update_stage()
    yield
    sim_utils.clear_stage()
    sim_utils.SimulationContext.clear_instance()


def _skip_if_unavailable(device: str):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if device == "cpu":
        pytest.skip("Warp fabricarray operations on CPU have known issues")


# ------------------------------------------------------------------
# Parent position helpers (via USD xformOps)
# ------------------------------------------------------------------


def _get_parent_positions(num_envs, device="cpu"):
    stage = sim_utils.get_current_stage()
    xform_cache = UsdGeom.XformCache()
    positions = []
    for i in range(num_envs):
        prim = stage.GetPrimAtPath(f"/World/Parent_{i}")
        tf = xform_cache.GetLocalToWorldTransform(prim)
        t = tf.ExtractTranslation()
        positions.append([float(t[0]), float(t[1]), float(t[2])])
    return torch.tensor(positions, dtype=torch.float32, device=device)


def _set_parent_positions(positions, num_envs):
    from pxr import Sdf  # noqa: PLC0415

    stage = sim_utils.get_current_stage()
    with Sdf.ChangeBlock():
        for i in range(num_envs):
            prim = stage.GetPrimAtPath(f"/World/Parent_{i}")
            pos = positions[i]
            prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))


# ------------------------------------------------------------------
# Contract fixture
# ------------------------------------------------------------------


@pytest.fixture
def view_factory():
    """Fabric factory: Camera child at CHILD_OFFSET under parent Xforms, with Fabric enabled."""

    def factory(num_envs: int, device: str) -> ViewBundle:
        _skip_if_unavailable(device)

        stage = sim_utils.get_current_stage()
        for i in range(num_envs):
            sim_utils.create_prim(f"/World/Parent_{i}", "Xform", translation=PARENT_POS, stage=stage)
            sim_utils.create_prim(f"/World/Parent_{i}/Child", "Camera", translation=CHILD_OFFSET, stage=stage)

        sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=device, use_fabric=True))
        view = FrameView("/World/Parent_.*/Child", device=device, sync_usd_on_fabric_write=True)
        return ViewBundle(
            view=view,
            get_parent_pos=_get_parent_positions,
            set_parent_pos=_set_parent_positions,
            teardown=lambda: None,
        )

    return factory


# ------------------------------------------------------------------
# Backend-specific contract: world poses must follow physics integration
# ------------------------------------------------------------------
#
# The shared contract uses static USD writes to move the parent. Real callers
# (RayCaster, Camera, IMU spawned under articulation/rigid bodies) rely on
# PhysX → Fabric per-step writes propagating through ``IFabricHierarchy`` to
# child Xforms. None of the existing static tests exercise that path, which
# allowed a fabric-write regression to ship undetected: ``RayCaster`` parented
# under an articulation body returns its spawn-time pose forever even as the
# body moves meters under gravity.


@pytest.mark.parametrize("device", ["cuda:0"])
def test_world_pose_tracks_physics_body_parent(device):
    """Child Xform world pose must follow a RigidBody parent through physics integration.

    Spawns a child Xform under a ``RigidBody`` + ``ArticulationRoot`` parent
    elevated at z=5, lets gravity drop it for 1 s, then asserts the
    :class:`FabricFrameView` returns a fresh world pose. With the working
    PhysX → Fabric write path, the child should drop several meters; with a
    broken write path, ``get_world_poses`` returns the spawn pose forever.
    """
    _skip_if_unavailable(device)

    from pxr import UsdPhysics

    initial_z = 5.0
    parent_path = "/World/PhysicsParent"
    child_path = f"{parent_path}/Child"

    stage = sim_utils.get_current_stage()
    sim_utils.create_prim(parent_path, "Xform", translation=(0.0, 0.0, initial_z), stage=stage)
    parent_prim = stage.GetPrimAtPath(parent_path)
    UsdPhysics.RigidBodyAPI.Apply(parent_prim)
    UsdPhysics.ArticulationRootAPI.Apply(parent_prim)
    UsdPhysics.MassAPI.Apply(parent_prim).CreateMassAttr().Set(1.0)
    cube_path = f"{parent_path}/CollisionCube"
    cube = UsdGeom.Cube.Define(stage, cube_path)
    cube.CreateSizeAttr().Set(0.1)
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(cube_path))
    sim_utils.create_prim(child_path, "Camera", translation=CHILD_OFFSET, stage=stage)
    sim_utils.update_stage()

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=device, use_fabric=True))
    view = FrameView(child_path, device=device, sync_usd_on_fabric_write=True)
    sim.reset()

    pos_before = view.get_world_poses()[0].torch[0].clone()

    # ~1 s of free fall: child should drop several meters in z.
    for _ in range(100):
        sim.step(render=False)

    pos_after = view.get_world_poses()[0].torch[0]
    drift_z = (pos_before[2] - pos_after[2]).item()

    # Free-fall over 1 s under g≈9.81 should drop the body well past any
    # spawn-time noise. A drift below 0.5 m means the FrameView is reading
    # a stale fabric matrix that PhysX never updated.
    assert drift_z > 0.5, (
        f"FabricFrameView returned stale pose after physics integration. "
        f"z before={pos_before[2].item():.4f} z after={pos_after[2].item():.4f} "
        f"drift={drift_z:.4f}m. PhysX → Fabric write path is broken or the "
        f"hierarchy isn't propagating parent body movement to the child Xform."
    )
