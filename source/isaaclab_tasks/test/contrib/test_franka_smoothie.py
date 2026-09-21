# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contracts for the smoothie task's scene, authored assets, controllers and ordered milestones."""

import math
import unittest
from types import SimpleNamespace

import pytest
import torch

from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab_tasks.contrib.franka_smoothie.basket_geometry import BASKET_BOTTOM_RADIUS
from isaaclab_tasks.contrib.franka_smoothie.scene_cfg import ASSETS, FRUIT_GROUPS, FRUIT_LAYOUT
from isaaclab_tasks.contrib.franka_smoothie.smoothie_env import SmoothieBlenderEnv
from isaaclab_tasks.contrib.franka_smoothie.smoothie_env_cfg import FrankaSmoothieEnvCfg
from isaaclab_tasks.contrib.franka_smoothie.smoothie_task import (
    TARGET_VOLUME_M3,
    SmoothiePhase,
    SmoothieTaskState,
    all_fruit_delivered,
)


def test_basket_expert_uses_independent_clocks_and_reset_poses(monkeypatch):
    from isaaclab_tasks.contrib.franka_smoothie import pose_control

    class Controller:
        def compute(self, position, rotation, close):
            return position.clone(), rotation.clone(), close.clone()

    monkeypatch.setattr(pose_control, "PoseController", lambda _: Controller())
    grasp = torch.tensor([[0.1, 0.0, 0.2], [0.2, 0.0, 0.2]])
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)
    tcp = torch.tensor([[0.1, 0.0, 0.22], [0.2, 0.0, 0.24]])
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        step_dt=1 / 60,
        grasp_pose=lambda: (grasp.clone(), rotations.clone()),
        tcp=lambda: tcp.clone(),
    )
    expert = pose_control.BasketExpert(env)
    position, _, close = expert.compute(torch.tensor([0, 450]))
    torch.testing.assert_close(position[:, 2], torch.tensor([0.35, 0.24]))
    assert close.tolist() == [False, True]
    position, _, close = expert.compute(torch.tensor([450, 720]))
    torch.testing.assert_close(position[:, 2], torch.tensor([0.22, 0.24 + 0.15 / 3.5]))
    assert close.all()
    grasp[0, 0] += 0.1
    rotations[0] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    position, rotation, close = expert.compute(torch.tensor([0, 840]))
    torch.testing.assert_close(position[0], grasp[0] + torch.tensor([0.0, 0.0, 0.15]))
    torch.testing.assert_close(position[1], tcp[1] + torch.tensor([0.0, 0.0, 0.15 * 3.0 / 3.5]))
    torch.testing.assert_close(rotation, rotations)
    assert close.tolist() == [False, True]


def test_scattered_fruits_are_separate_scene_objects():
    cfg = FrankaSmoothieEnvCfg()
    fruits = [getattr(cfg.scene, name) for name in FRUIT_LAYOUT]
    assert len(fruits) == 16
    assert [len(names) for names in FRUIT_GROUPS.values()] == [4, 4, 4, 4]
    assert len({fruit.prim_path for fruit in fruits}) == 16
    assert len({fruit.init_state.pos for fruit in fruits}) == 16
    assert all(
        fruit.spawn.rigid_props is None or fruit.spawn.rigid_props.kinematic_enabled is not True for fruit in fruits
    )
    # Retained fruits keep the rotations used by the original packed basket.
    for kind, original_start in (("strawberry", 0), ("blueberry", 5), ("blackberry", 11), ("mango", 16)):
        for instance, name in enumerate(FRUIT_GROUPS[kind]):
            yaw = (original_start + instance) * 2.399963229728653
            assert getattr(cfg.scene, name).init_state.rot == pytest.approx(
                (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
            )


def test_any_fruit_instance_can_satisfy_the_cup_receipt_check():
    poses = {name: torch.tensor([[0.4, 0.15, 0.03, 0.0, 0.0, 0.0, 1.0]]) for name in FRUIT_LAYOUT}
    poses["cup"] = torch.tensor([[0.57, -0.07, 0.013, 0.0, 0.0, 0.0, 1.0]])
    env = SimpleNamespace(device="cpu", num_envs=1, pose=poses.__getitem__)
    env.local = lambda name, points: SmoothieBlenderEnv.local(env, name, points)
    assert not SmoothieBlenderEnv.fruit_inside(env).any()
    assert SmoothieBlenderEnv.fruit_fraction(env).item() == 0.0
    for i, names in enumerate(FRUIT_GROUPS.values()):
        # The original single fruit stays in the basket; a duplicate is loaded instead.
        poses[names[-1]][:, :3] = poses["cup"][:, :3] + torch.tensor([[0.0, 0.0, 0.05]])
        expected = torch.arange(len(FRUIT_GROUPS)) <= i
        assert torch.equal(SmoothieBlenderEnv.fruit_inside(env)[0], expected)
    assert SmoothieBlenderEnv.fruit_fraction(env).item() == pytest.approx(4 / len(FRUIT_LAYOUT))


def test_basket_is_one_rigid_body_with_an_open_lattice():
    cfg = FrankaSmoothieEnvCfg()
    stage = Usd.Stage.Open(cfg.scene.basket.spawn.usd_path)
    bodies = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    assert len(bodies) == 1
    assert UsdPhysics.MassAPI(bodies[0]).GetMassAttr().Get() > 0
    assert not hasattr(cfg.scene, "bag")
    floor = UsdGeom.Cylinder(stage.GetPrimAtPath("/Asset/Basket/Floor"))
    assert floor.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
    assert floor.GetHeightAttr().Get() > 0
    # The floor must reach past the radius the lattice bars are held outside of.
    assert floor.GetRadiusAttr().Get() >= BASKET_BOTTOM_RADIUS
    bars = [UsdGeom.Mesh(prim) for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    assert bars
    for mesh in bars:
        assert mesh.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
        assert UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get() == "convexHull"
        # Individual bars stay outside the inner base radius, leaving the central mouth open.
        points = torch.tensor(mesh.GetPointsAttr().Get(), dtype=torch.float64)
        assert (points[:, :2].norm(dim=-1) >= BASKET_BOTTOM_RADIUS - 1.0e-6).all()
    # The lattice uses separate uprights and hoops, not a convexified closed container.
    assert stage.GetPrimAtPath("/Asset/Basket/Uprights")
    assert stage.GetPrimAtPath("/Asset/Basket/Hoops")
    assert not stage.GetPrimAtPath("/Asset/Film")


@pytest.mark.parametrize("file", ["cup.usda", "blade_cap.usda"])
def test_threads_have_separate_convex_contact_surfaces(file):
    stage = Usd.Stage.Open(str(ASSETS / file))
    threads = [p for p in stage.Traverse() if "/Thread/T" in str(p.GetPath())]
    assert len(threads) > 32
    for prim in threads:
        mesh = UsdGeom.Mesh(prim)
        points = torch.tensor(mesh.GetPointsAttr().Get(), dtype=torch.float64)
        faces = torch.tensor(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 4)
        volume = 0.0
        for face in faces:
            a, b, c, d = points[face]
            volume += (a.dot(torch.linalg.cross(b, c)) + a.dot(torch.linalg.cross(c, d))) / 6.0
        assert volume > 0.0, "Inward faces corrupt the collider's mass and signed distance."
        assert prim.HasAPI(UsdPhysics.CollisionAPI)
        assert UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == "convexHull"
    # There is no joint or drive imposing the screw's pitch.
    assert not any(p.IsA(UsdPhysics.PrismaticJoint) for p in stage.Traverse())


def test_cup_handle_supports_leave_cavity_open():
    stage = Usd.Stage.Open(str(ASSETS / "cup.usda"))
    wall = UsdGeom.Mesh(stage.GetPrimAtPath("/Asset/Cup/Wall/S000"))
    radii = [math.hypot(point[0], point[1]) for point in wall.GetPointsAttr().Get()]
    inner_radius, outer_radius = min(radii), max(radii)
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    grip = bounds.ComputeWorldBound(stage.GetPrimAtPath("/Asset/Cup/Handle/Grip")).ComputeAlignedRange()
    supports = [p for p in stage.GetPrimAtPath("/Asset/Cup/Handle").GetChildren() if p.GetName().startswith("Bridge")]
    assert supports
    for support in supports:
        assert support.HasAPI(UsdPhysics.CollisionAPI)
        bridge = bounds.ComputeWorldBound(support).ComputeAlignedRange()
        # Supports approach from negative Y and must end within the wall, outside the cavity.
        near_radius = -bridge.GetMax()[1]
        assert inner_radius <= near_radius <= outer_radius
        assert all(
            min(bridge.GetMax()[axis], grip.GetMax()[axis]) > max(bridge.GetMin()[axis], grip.GetMin()[axis])
            for axis in range(3)
        ), "Each support must remain connected to the exterior grip."


class TapTaskChecks(unittest.TestCase):
    def setUp(self):
        self.task = SmoothieTaskState(2, "cpu", 0.1)
        self.inputs = {
            name: torch.zeros(2, dtype=torch.bool)
            for name in (
                "fruit_delivered",
                "cup_under_tap",
                "cup_upright",
                "cup_open",
                "lid_fastened",
                "cup_docked",
                "blender_button_pressed",
                "failed",
            )
        }
        self.inputs["valve_depression_m"] = torch.zeros(2)

    def step(self, count=1, **values):
        for name, value in values.items():
            self.inputs[name][:] = torch.as_tensor(value)
        for _ in range(count):
            self.task.update(**self.inputs)

    def start_tap(self):
        self.step(fruit_delivered=True, cup_under_tap=True, cup_upright=True, cup_open=True)
        self.assertTrue((self.task.phase == SmoothiePhase.TAP).all())
        self.step(valve_depression_m=0.002)

    def finish_tap(self):
        self.start_tap()
        self.step(19)
        torch.testing.assert_close(self.task.fill_volume_m3, torch.full((2,), TARGET_VOLUME_M3))
        self.step(valve_depression_m=0.001)
        self.step(valve_depression_m=0.002)
        self.assertTrue((self.task.phase == SmoothiePhase.LID).all())

    def reach_button(self):
        self.finish_tap()
        self.step(lid_fastened=True, cup_open=False)
        self.step(cup_docked=True)
        self.assertTrue((self.task.phase == SmoothiePhase.BUTTON).all())

    def test_all_sixteen_whole_fruits_required(self):
        hulls = torch.ones(2, 16, dtype=torch.bool)
        hulls[0, -1] = False
        self.step(fruit_delivered=all_fruit_delivered(hulls))
        self.assertEqual(self.task.phase.tolist(), [SmoothiePhase.FRUIT, SmoothiePhase.TAP])
        for invalid in (torch.ones(2, 15, dtype=torch.bool), torch.ones(2, 16), torch.ones(16, dtype=torch.bool)):
            with self.assertRaises(ValueError):
                all_fruit_delivered(invalid)

    def test_complete_ordered_sequence_and_press_edge(self):
        self.reach_button()
        self.assertFalse(self.task.completed.any())
        self.step(blender_button_pressed=False)
        self.step(blender_button_pressed=True)
        self.assertTrue(self.task.completed.all())
        self.assertTrue(self.task.milestones.all())
        self.assertFalse(self.task.tap_on.any())
        self.assertTrue((self.task.fill_fraction == 1).all())

    def test_held_valve_toggles_once_and_hysteresis_requires_release(self):
        self.start_tap()
        self.step(3, valve_depression_m=0.003)
        self.assertTrue(self.task.tap_on.all())
        self.step(valve_depression_m=0.0015)
        self.step(valve_depression_m=0.002)
        self.assertTrue(self.task.tap_on.all())
        self.step(valve_depression_m=0.001)
        self.step(valve_depression_m=0.002)
        self.assertFalse(self.task.tap_on.any())
        self.assertFalse(self.task.tap_off_seen.any())
        self.assertTrue((self.task.phase == SmoothiePhase.TAP).all())

    def test_each_nozzle_gate_pauses_filling(self):
        for gate in ("cup_under_tap", "cup_upright", "cup_open"):
            with self.subTest(gate=gate):
                self.setUp()
                self.start_tap()
                before = self.task.fill_volume_m3.clone()
                self.step(30, **{gate: False})
                torch.testing.assert_close(self.task.fill_volume_m3, before)
                self.step(19, **{gate: True})
                self.assertTrue((self.task.fill_fraction == 1).all())
                self.assertTrue((self.task.phase == SmoothiePhase.TAP).all())

    def test_fill_requires_two_seconds_and_valve_off_after_filling(self):
        self.start_tap()
        self.step(18)
        torch.testing.assert_close(self.task.fill_fraction, torch.full((2,), 0.95))
        self.step(valve_depression_m=0.001)
        self.assertTrue((self.task.fill_fraction == 1).all())
        self.assertTrue((self.task.phase == SmoothiePhase.TAP).all())
        self.step(valve_depression_m=0.002, cup_under_tap=[False, True])
        self.assertEqual(self.task.phase.tolist(), [SmoothiePhase.TAP, SmoothiePhase.LID])
        self.step(cup_under_tap=True)
        self.assertTrue((self.task.phase == SmoothiePhase.LID).all())

    def test_early_held_valve_cannot_supply_ordered_tap_evidence(self):
        self.step()
        self.step(valve_depression_m=0.002)
        self.assertTrue(self.task.tap_on.all())
        self.step(fruit_delivered=True, cup_under_tap=True, cup_upright=True, cup_open=True)
        self.step(30)
        self.assertFalse(self.task.tap_on_seen.any())
        self.assertFalse(self.task.fill_volume_m3.any())
        self.assertTrue((self.task.phase == SmoothiePhase.TAP).all())

    def test_lid_and_dock_cannot_skip_a_phase(self):
        self.step(lid_fastened=True, cup_docked=True, blender_button_pressed=True)
        self.assertTrue((self.task.phase == SmoothiePhase.FRUIT).all())
        self.finish_tap()
        self.assertTrue((self.task.phase == SmoothiePhase.LID).all())
        self.step(lid_fastened=False)
        self.assertTrue((self.task.phase == SmoothiePhase.LID).all())
        self.step(lid_fastened=True, cup_docked=False)
        self.step(2)
        self.assertTrue((self.task.phase == SmoothiePhase.DOCK).all())

    def test_early_held_blender_press_needs_release_during_button_phase(self):
        self.inputs["blender_button_pressed"][:] = True
        self.reach_button()
        self.step(5)
        self.assertFalse(self.task.completed.any())
        self.step(blender_button_pressed=False)
        self.step(blender_button_pressed=True)
        self.assertTrue(self.task.completed.all())

    def test_final_press_rechecks_fruit_lid_and_dock(self):
        for gate in ("fruit_delivered", "lid_fastened", "cup_docked"):
            with self.subTest(gate=gate):
                self.setUp()
                self.reach_button()
                self.step(blender_button_pressed=False)
                self.step(blender_button_pressed=True, **{gate: False})
                self.assertFalse(self.task.completed.any())
                self.step(**{gate: True})
                self.assertFalse(self.task.completed.any())
                self.step(blender_button_pressed=False)
                self.step(blender_button_pressed=True)
                self.assertTrue(self.task.completed.all())

    def test_late_tap_on_prevents_completion_without_further_filling(self):
        self.reach_button()
        self.step(valve_depression_m=0.001, blender_button_pressed=False)
        self.step(valve_depression_m=0.002, blender_button_pressed=True)
        self.assertTrue(self.task.tap_on.all())
        self.assertFalse(self.task.completed.any())
        before = self.task.fill_volume_m3.clone()
        self.step(5)
        torch.testing.assert_close(self.task.fill_volume_m3, before)

    def test_failure_latches_and_partial_reset_preserves_other_world(self):
        self.start_tap()
        before = self.task.fill_volume_m3.clone()
        self.step(failed=[True, False], valve_depression_m=0.001)
        self.step(30, failed=False, valve_depression_m=0.002)
        self.assertTrue(self.task.failed[0])
        self.assertEqual(self.task.fill_volume_m3[0], before[0])
        self.assertTrue(self.task.tap_on[0])
        other = {name: value[1].clone() for name, value in vars(self.task).items() if isinstance(value, torch.Tensor)}
        self.task.reset(torch.tensor([0]))
        self.assertEqual(self.task.phase[0], SmoothiePhase.FRUIT)
        self.assertFalse(self.task.failed[0])
        self.assertFalse(self.task.fill_volume_m3[0])
        self.assertFalse(self.task.tap_on[0])
        self.assertFalse(self.task.milestones[0].any())
        for name, expected in other.items():
            torch.testing.assert_close(getattr(self.task, name)[1], expected)
        self.task.reset()
        self.step(fruit_delivered=True, valve_depression_m=0.003)
        self.step(5)
        self.assertFalse(self.task.tap_on.any())

    def test_invalid_inputs_and_reset_indices_leave_state_unchanged(self):
        self.start_tap()
        before = {name: value.clone() for name, value in vars(self.task).items() if isinstance(value, torch.Tensor)}
        for name, value in (
            ("fruit_delivered", torch.ones(2)),
            ("failed", torch.zeros(3, dtype=torch.bool)),
            ("valve_depression_m", torch.tensor([float("nan"), 0.0])),
            ("valve_depression_m", torch.zeros(2, dtype=torch.long)),
        ):
            with self.assertRaises(ValueError):
                self.task.update(**{**self.inputs, name: value})
        for indices in (torch.tensor([-1]), torch.tensor([2]), torch.tensor([0, 0]), torch.tensor([0.0])):
            with self.assertRaises(ValueError):
                self.task.reset(indices)
        for name, expected in before.items():
            torch.testing.assert_close(getattr(self.task, name), expected)

    def test_fill_duration_at_thirty_hertz(self):
        self.task = SmoothieTaskState(2, "cpu", 1 / 30)
        self.start_tap()
        self.step(58)
        self.assertTrue((self.task.fill_fraction < 1).all())
        self.step()
        self.assertTrue((self.task.fill_fraction == 1).all())


def test_smoothie_uses_rigid_physics_at_fifty_hertz():
    cfg = FrankaSmoothieEnvCfg()
    cfg.validate()
    assert cfg.sim.dt * cfg.decimation == pytest.approx(1 / 50)
    assert not any(hasattr(cfg.scene, name) for name in ("media", "milk", "milk_cap"))
    assert cfg.sim.physics.solver_cfg.solver_type == "mujoco_warp"


def test_tap_workstation_keeps_cup_support_and_removes_carton_slot():
    cfg = FrankaSmoothieEnvCfg()
    stage = Usd.Stage.Open(cfg.scene.workstation.spawn.usd_path)
    assert not stage.GetPrimAtPath("/Asset/MilkHolder").IsActive()
    assert stage.GetPrimAtPath("/Asset/CupHolder").IsActive()
    assert stage.GetPrimAtPath("/Asset/CapStand").IsActive()
    tap = Usd.Stage.Open(cfg.scene.tap.spawn.usd_path)
    joint = UsdPhysics.PrismaticJoint(tap.GetPrimAtPath("/Asset/TapButton"))
    assert joint.GetLowerLimitAttr().Get() == pytest.approx(-0.004)
    assert joint.GetUpperLimitAttr().Get() == 0


def test_tap_bundled_robot_has_required_arm_collision_proxies(monkeypatch):
    monkeypatch.delenv("ISAACLAB_FRANKA_POUR_ROBOT_USD_PATH", raising=False)
    cfg = FrankaSmoothieEnvCfg()
    stage = Usd.Stage.Open(cfg.scene.robot.spawn.usd_path)
    assert not any(stage.GetRootLayer().GetExternalReferences())
    names = {prim.GetName() for prim in stage.Traverse()}
    assert {
        "link0_c",
        "link1_c",
        "link2_c",
        "link3_c",
        "link4_c",
        "link5_c0",
        "link5_c1",
        "link5_c2",
        "link6_c",
        "link7_c",
    } <= names


@pytest.mark.parametrize("succeeded", [False, True])
def test_runner_only_records_successful_episodes(monkeypatch, tmp_path, succeeded):
    import contextlib
    import runpy
    import sys
    from pathlib import Path

    import isaaclab.app

    from isaaclab_tasks.contrib.franka_smoothie import smoothie_controller, smoothie_env

    env = SimpleNamespace(
        reset=lambda **_: None,
        close=lambda: None,
        max_episode_length=1,
        step_dt=0.02,
        step=lambda _: (None, None, torch.tensor([True]), torch.tensor([False]), {}),
        termination_manager=SimpleNamespace(get_term=lambda _: torch.tensor([succeeded])),
    )
    controller = SimpleNamespace(compute=lambda _: torch.zeros((1, 8)), stage="test", close_trace=lambda: None)
    monkeypatch.setattr(isaaclab.app, "launch_simulation", lambda _: contextlib.nullcontext())
    monkeypatch.setattr(smoothie_env, "SmoothieBlenderEnv", lambda _: env)
    monkeypatch.setattr(smoothie_controller, "SmoothieSequenceController", lambda _: controller)
    recording = tmp_path / "trajectory.npz"
    runner = Path(__file__).resolve().parents[4] / "scripts/environments/run_franka_smoothie.py"
    monkeypatch.setattr(sys, "argv", [str(runner), "--headless", "--record", str(recording)])
    if succeeded:
        runpy.run_path(str(runner), run_name="__main__")
        assert recording.is_file()
    else:
        with pytest.raises(SystemExit, match="2"):
            runpy.run_path(str(runner), run_name="__main__")
        assert not recording.exists()
