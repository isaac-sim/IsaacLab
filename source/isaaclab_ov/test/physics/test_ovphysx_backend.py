# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native contracts for explicitly owned OVPhysX backends without a simulation context."""

import sys

import pytest

from pxr import Usd

from isaaclab.test.utils import test_devices

pytestmark = [pytest.mark.ov, pytest.mark.kitless, pytest.mark.integration]

_DEVICES = test_devices()

pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")


@pytest.mark.parametrize(
    ("device", "other_device"),
    [(device, device) for device in _DEVICES]
    + [("cpu", device) for device in _DEVICES if "cpu" in _DEVICES and device.startswith("cuda:")],
)
def test_explicit_backends_preserve_native_stage_ownership_and_live_gravity(device, other_device):
    """Simultaneous owners isolate motion, gravity, publications, reload, and close."""
    from isaaclab_ov.physics import OvPhysxBackend, OvPhysxBackendCfg

    stage_usda = """#usda 1.0
(defaultPrim = "World"
 metersPerUnit = 1
 upAxis = "Z")
def Xform "World" {
    def PhysicsScene "physicsScene" (prepend apiSchemas = ["PhysxSceneAPI"]) {
        vector3f physics:gravityDirection = (0, 0, -1)
        float physics:gravityMagnitude = 10
        bool physxScene:enableGPUDynamics = false
        token physxScene:broadphaseType = "MBP"
    }
    def Cube "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "PhysicsMassAPI"]) {
        double size = 0.1
        double3 xformOp:translate = (0, 0, 10)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        float physics:mass = 1
    }
}
"""

    def scene(device, *, height=10, gravity=10):
        gpu = device.startswith("cuda:")
        return (
            stage_usda.replace("enableGPUDynamics = false", f"enableGPUDynamics = {str(gpu).lower()}")
            .replace('broadphaseType = "MBP"', f'broadphaseType = "{"GPU" if gpu else "MBP"}"')
            .replace("(0, 0, 10)", f"(0, 0, {height})")
            .replace("gravityMagnitude = 10", f"gravityMagnitude = {gravity}")
        )

    def height(backend):
        # Refresh even an idle instance so cached values cannot hide cross-instance writes.
        backend.forward()
        return float(backend.scene_data.transforms.transforms.numpy()[0, 2])

    host_pxr = sys.modules["pxr"]
    first = OvPhysxBackend(OvPhysxBackendCfg(device=device))
    first_publication = first.scene_data
    second = None
    try:
        assert sys.modules["pxr"] is host_pxr
        assert sys.modules["pxr.Usd"] is Usd
        with pytest.raises(RuntimeError, match="attached stage|stage attached"):
            first.step(0.01)
        first.load(scene(device))
        with pytest.raises(RuntimeError, match="Warm up or step"):
            _ = first_publication.transform_count
        first.warmup()
        second = OvPhysxBackend(OvPhysxBackendCfg(device=other_device))
        second.load(scene(other_device, height=20, gravity=0))
        # DirectGPU session discovery requires initialized native physics buffers.
        second.warmup()
        for backend, target_device in ((first, device), (second, other_device)):
            assert backend.scene_data.transform_count == 1
            assert backend.scene_data.transform_paths == ["/World/Body"]
            assert backend.rigid_body_view.native_device.device_type.value == (
                2 if target_device.startswith("cuda:") else 1
            )

        assert height(second) == pytest.approx(20.0, abs=1.0e-4)
        for _ in range(10):
            first.step(0.01)
        before_gravity_change = height(first)
        assert before_gravity_change < 9.96
        assert height(second) == pytest.approx(20.0, abs=1.0e-4)

        for _ in range(10):
            second.step(0.01)
        assert height(second) == pytest.approx(20.0, abs=1.0e-4)
        assert height(first) == before_gravity_change

        first.set_gravity((0.0, 0.0, 30.0), physics_prim_path="/World/physicsScene")
        assert first.gravity["/World/physicsScene"] == (0.0, 0.0, 30.0)
        for _ in range(30):
            first.step(0.01)
        after_gravity_change = height(first)
        assert after_gravity_change > before_gravity_change + 0.5
        assert height(second) == pytest.approx(20.0, abs=1.0e-4)

        second.set_gravity((0.0, 0.0, -5.0), physics_prim_path="/World/physicsScene")
        assert second.gravity["/World/physicsScene"] == (0.0, 0.0, -5.0)
        assert first.gravity["/World/physicsScene"] == (0.0, 0.0, 30.0)
        for _ in range(10):
            second.step(0.01)
        second_falling_height = height(second)
        assert second_falling_height < 19.99
        assert height(first) == after_gravity_change

        first.load(scene(device, height=30))
        first.warmup()
        assert first_publication.transform_count == 1
        assert first_publication.transform_paths == ["/World/Body"]
        assert height(first) == pytest.approx(30.0, abs=1.0e-4)
        assert height(second) == second_falling_height
        first.close()
        assert first_publication.transform_count == 0
        with pytest.raises(RuntimeError, match="closed"):
            first.load(scene(device))

        for _ in range(10):
            second.step(0.01)
        assert height(second) < second_falling_height - 0.05
        second.close()
        assert second.scene_data.transform_count == 0
        assert not second.gravity
    finally:
        if second is not None:
            second.close()
        first.close()
