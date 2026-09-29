# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Real PhysX rigid-object coverage.

Most checks run against one module-scoped scene of locally authored cubes. Each rigid object holds two
environments so that partial writes can target environment 1 and prove that environment 0 is preserved in the
real PhysX state. Tests that need their own simulation context are defined first: pytest runs them before the
composite scene is created, and a new simulation context would replace the composite stage.
"""

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from isaaclab.test.utils import DeviceScope, launch_test_simulation, test_devices

launch_test_simulation()

import math
import sys
from dataclasses import dataclass
from typing import Literal

import pytest
import torch
import warp as wp
from isaaclab_physx.assets import RigidObject
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim import SimulationContext, build_simulation_context
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_apply_inverse, quat_mul

_NUM_ENVS = 2


def _cube_cfg(prim_path: str, *, kinematic: bool = False, disable_gravity: bool = False) -> RigidObjectCfg:
    """Create a local 1 kg collision cube one meter above its environment origin."""
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=[
                sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=kinematic),
                PhysxRigidBodyCfg(disable_gravity=disable_gravity),
            ],
            mass_props=sim_utils.MassCfg(mass=1.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )


def _spawn_envs(root: str, y_offset: float = 0.0) -> torch.Tensor:
    """Create two environment prims 2 m apart and return their origins."""
    origins = torch.tensor([(2.0 * index, y_offset, 0.0) for index in range(_NUM_ENVS)])
    for index, origin in enumerate(origins.tolist()):
        sim_utils.create_prim(f"{root}/Env_{index}", "Xform", translation=origin)
    return origins


def _yaw_quat(angle: float) -> tuple[float, float, float, float]:
    """Return a unit quaternion ``(x, y, z, w)`` for a rotation of ``angle`` [rad] about the world z axis."""
    return (0.0, 0.0, math.sin(0.5 * angle), math.cos(0.5 * angle))


##
# Tests that own their simulation context. Keep them above the composite scene.
##


@pytest.mark.parametrize(
    "api",
    [
        "none",
        pytest.param(
            "articulation_root",
            marks=pytest.mark.xfail(
                strict=True, reason="The PhysX rigid object no longer rejects an enabled articulation root."
            ),
        ),
    ],
)
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.isaacsim_ci
def test_initialization_rejects_invalid_rigid_body(device, api: Literal["none", "articulation_root"]):
    """Initialization fails without a rigid body and when the rigid body is an articulation root."""
    with build_simulation_context(device=device, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        _spawn_envs("/World")
        if api == "none":
            # Without rigid body properties the cube is a static collider.
            cfg = RigidObjectCfg(
                prim_path="/World/Env_[^/]*/Object",
                spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1), collision_props=sim_utils.UsdPhysicsCollisionCfg()),
            )
            cube_object = RigidObject(cfg=cfg)
        else:
            cube_object = RigidObject(cfg=_cube_cfg("/World/Env_[^/]*/Object"))
            # The articulation root above the rigid body turns the body into an articulation link.
            for index in range(_NUM_ENVS):
                UsdPhysics.ArticulationRootAPI.Apply(sim.stage.GetPrimAtPath(f"/World/Env_{index}"))

        # Check that the framework doesn't hold excessive strong references.
        assert sys.getrefcount(cube_object) < 10

        with pytest.raises(RuntimeError):
            sim.reset()


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("gravity_enabled", [True, False])
@pytest.mark.isaacsim_ci
def test_gravity_vec_w(device, gravity_enabled):
    """Test that gravity vector direction is set correctly for the rigid object."""
    with build_simulation_context(device=device, gravity_enabled=gravity_enabled) as sim:
        sim._app_control_on_stop_handle = None
        _spawn_envs("/World")
        cube_object = RigidObject(cfg=_cube_cfg("/World/Env_[^/]*/Object"))

        # Obtain gravity direction
        if gravity_enabled:
            gravity_dir = (0.0, 0.0, -1.0)
        else:
            gravity_dir = (0.0, 0.0, 0.0)

        # Play sim
        sim.reset()

        # Check that gravity is set correctly
        assert cube_object.data.GRAVITY_VEC_W.torch[0, 0] == gravity_dir[0]
        assert cube_object.data.GRAVITY_VEC_W.torch[0, 1] == gravity_dir[1]
        assert cube_object.data.GRAVITY_VEC_W.torch[0, 2] == gravity_dir[2]

        # Simulate physics
        for _ in range(2):
            # perform rendering
            sim.step()
            # update object
            cube_object.update(sim.cfg.dt)

            # Expected gravity value is the acceleration of the body
            gravity = torch.zeros(_NUM_ENVS, 1, 6, device=device)
            if gravity_enabled:
                gravity[:, :, 2] = -9.81
            # Check the body accelerations are correct
            torch.testing.assert_close(cube_object.data.body_acc_w.torch, gravity)


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("device", test_devices())
def test_warmup_loads_physics_once(device):
    """Attach on GPU or force-load on CPU, without destroying and rebuilding native objects."""
    from unittest.mock import MagicMock, patch

    import omni.kit.app
    import omni.physx

    with build_simulation_context(device=device, add_ground_plane=True, dt=0.01, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        sim_utils.create_prim("/World/Table_0", "Xform", translation=(0.0, 0.0, 1.0))
        cube = RigidObject(cfg=_cube_cfg("/World/Table_[^/]*/Object"))

        # Wrap read-only native interfaces through their accessors; real calls still execute.
        physx_spy = MagicMock(wraps=omni.physx.get_physx_interface())
        physx_sim_spy = MagicMock(wraps=omni.physx.get_physx_simulation_interface())
        with (
            patch("omni.physx.get_physx_interface", return_value=physx_spy),
            patch("omni.physx.get_physx_simulation_interface", return_value=physx_sim_spy),
        ):
            sim.reset()

        extension_manager = omni.kit.app.get_app().get_extension_manager()
        assert extension_manager.is_extension_enabled("omni.physics.physx"), (
            "The omni.physics.physx bridge must register PhysX with the unified physics API."
        )
        assert physx_sim_spy.attach_stage.call_count == int(device.startswith("cuda"))
        assert physx_spy.force_load_physics_from_usd.call_count == int(device == "cpu")
        assert cube.is_initialized and cube.num_instances == 1
        initial = cube.data.root_pos_w.torch.clone()
        for _ in range(10):
            sim.step(render=False)
            cube.update(sim.get_physics_dt())
        assert torch.all(cube.data.root_pos_w.torch[:, 2] < initial[:, 2])


##
# Composite scene shared by the remaining tests.
##


@dataclass
class _RigidObjectScene:
    """Rigid objects that share one real PhysX lifecycle under gravity."""

    sim: SimulationContext
    device: str
    cubes: RigidObject
    """Dynamic cubes that ignore gravity."""
    kinematic: RigidObject
    """Kinematic cubes."""
    origins: dict[str, torch.Tensor]
    """Environment origins keyed by object name."""
    refcounts: dict[str, int]
    """Reference count of each object right after construction."""

    def step(self, num_steps: int = 1) -> None:
        """Write, step, and update every object."""
        for _ in range(num_steps):
            for rigid_object in (self.cubes, self.kinematic):
                rigid_object.write_data_to_sim()
            self.sim.step()
            for rigid_object in (self.cubes, self.kinematic):
                rigid_object.update(self.sim.cfg.dt)

    def place_cubes_at_rest(self, root_pose: torch.Tensor) -> None:
        """Teleport the dynamic cubes to ``root_pose`` at rest and clear their external wrenches."""
        self.cubes.write_root_link_pose_to_sim_index(root_pose=root_pose)
        self.cubes.write_root_com_velocity_to_sim_index(root_velocity=torch.zeros((_NUM_ENVS, 6), device=self.device))
        self.cubes.reset()


@pytest.fixture(scope="module", params=test_devices())
def rigid_object_scene(request) -> _RigidObjectScene:
    """Initialize the composite rigid-object scene once per device."""
    device = request.param
    with build_simulation_context(device=device, gravity_enabled=True) as sim:
        sim._app_control_on_stop_handle = None
        origins = {"cubes": _spawn_envs("/World/Cubes"), "kinematic": _spawn_envs("/World/Kinematic", 3.0)}
        cubes = RigidObject(cfg=_cube_cfg("/World/Cubes/Env_[^/]*/Object", disable_gravity=True))
        kinematic = RigidObject(cfg=_cube_cfg("/World/Kinematic/Env_[^/]*/Object", kinematic=True))
        refcounts = {"cubes": sys.getrefcount(cubes), "kinematic": sys.getrefcount(kinematic)}
        sim.reset()
        yield _RigidObjectScene(
            sim=sim,
            device=device,
            cubes=cubes,
            kinematic=kinematic,
            origins={name: value.to(device) for name, value in origins.items()},
            refcounts=refcounts,
        )


def test_rigid_object_initialization(rigid_object_scene: _RigidObjectScene):
    """Initialize local cubes with the expected buffers; kinematic cubes hold their default pose under gravity."""
    scene = rigid_object_scene
    for name, rigid_object in (("cubes", scene.cubes), ("kinematic", scene.kinematic)):
        # Check that the framework doesn't hold excessive strong references.
        assert scene.refcounts[name] < 10
        assert rigid_object.is_initialized
        assert rigid_object.num_instances == _NUM_ENVS
        assert len(rigid_object.body_names) == 1
        assert rigid_object.data.root_pos_w.torch.shape == (_NUM_ENVS, 3)
        assert rigid_object.data.root_quat_w.torch.shape == (_NUM_ENVS, 4)
        assert rigid_object.data.body_mass.torch.shape == (_NUM_ENVS, 1)
        assert rigid_object.data.body_inertia.torch.shape == (_NUM_ENVS, 1, 9)

    kinematic = scene.kinematic
    for _ in range(2):
        scene.step()
        default_root_pose = kinematic.data.default_root_pose.torch.clone()
        default_root_pose[:, :3] += scene.origins["kinematic"]
        torch.testing.assert_close(kinematic.data.root_link_pose_w.torch, default_root_pose)
        torch.testing.assert_close(kinematic.data.root_com_vel_w.torch, kinematic.data.default_root_vel.torch)


def test_rigid_object_inertial_properties(rigid_object_scene: _RigidObjectScene):
    """Mass, center-of-mass, and inertia writes reach only the selected PhysX entries and survive a step."""
    scene = rigid_object_scene
    device = scene.device
    cube_object = scene.cubes

    # Full-data mass writes reach the view and survive a simulation step.
    original_masses = wp.to_torch(cube_object.root_view.get_masses()).clone()
    assert original_masses.shape == (_NUM_ENVS, 1)
    masses = original_masses + torch.tensor([[4.5], [6.5]])
    cube_object.set_masses_index(masses=masses.to(device))
    torch.testing.assert_close(cube_object.data.body_mass.torch, masses.to(device))
    torch.testing.assert_close(wp.to_torch(cube_object.root_view.get_masses()), masses)
    scene.step()
    torch.testing.assert_close(cube_object.data.body_mass.torch, masses.to(device))
    torch.testing.assert_close(wp.to_torch(cube_object.root_view.get_masses()), masses)

    # Partial writes with an int64 selector change environment 1 only.
    env_ids = torch.tensor([1], dtype=torch.int64, device=device)
    body_ids = torch.tensor([0], dtype=torch.int32, device=device)
    expected = {
        "body_mass": cube_object.data.body_mass.torch.clone(),
        "body_com_pose_b": cube_object.data.body_com_pose_b.torch.clone(),
        "body_inertia": cube_object.data.body_inertia.torch.clone(),
    }
    expected["body_mass"][1, 0] = 3.0
    expected["body_com_pose_b"][1, 0, :3] = torch.tensor([0.03, -0.02, 0.01], device=device)
    expected["body_inertia"][1, 0, [0, 4, 8]] *= torch.tensor([1.2, 1.3, 1.4], device=device)
    cube_object.set_masses_index(masses=expected["body_mass"][1:], env_ids=env_ids, body_ids=body_ids)
    cube_object.set_coms_index(coms=expected["body_com_pose_b"][1:], env_ids=env_ids, body_ids=body_ids)
    cube_object.set_inertias_index(inertias=expected["body_inertia"][1:], env_ids=env_ids, body_ids=body_ids)
    for name, raw in (
        ("body_mass", cube_object.root_view.get_masses()),
        ("body_com_pose_b", cube_object.root_view.get_coms().view(wp.float32)),
        ("body_inertia", cube_object.root_view.get_inertias()),
    ):
        torch.testing.assert_close(getattr(cube_object.data, name).torch, expected[name])
        torch.testing.assert_close(wp.to_torch(raw).to(device).reshape(expected[name].shape), expected[name])


def test_rigid_object_root_state_writes(rigid_object_scene: _RigidObjectScene):
    """Root writes round-trip through the written frame and keep the other frame consistent with the offset."""
    scene = rigid_object_scene
    device = scene.device
    cube_object = scene.cubes
    rest_pose = torch.cat((scene.origins["cubes"], torch.tensor([[0.0, 0.0, 0.0, 1.0]] * _NUM_ENVS, device=device)), -1)
    rest_pose[:, 2] += 1.0
    scene.place_cubes_at_rest(rest_pose)

    # A partial pose and velocity write reaches only the selected root.
    initial_pose = cube_object.data.root_link_pose_w.torch.clone()
    initial_velocity = cube_object.data.root_link_vel_w.torch.clone()
    target_pose = initial_pose[1:].clone()
    target_pose[:, :3] += torch.tensor([0.25, -0.1, 0.3], device=device)
    target_pose[:, 3:] = torch.tensor(_yaw_quat(0.4), device=device)
    target_velocity = torch.tensor([[0.0, 0.2, 0.0, 0.0, 0.0, 0.1]], device=device)
    cube_object.write_root_link_pose_to_sim_index(root_pose=target_pose, env_ids=[1])
    cube_object.write_root_link_velocity_to_sim_index(root_velocity=target_velocity, env_ids=[1])
    torch.testing.assert_close(cube_object.data.root_link_pose_w.torch[1:], target_pose)
    torch.testing.assert_close(cube_object.data.root_link_vel_w.torch[1:], target_velocity)
    torch.testing.assert_close(cube_object.data.root_link_pose_w.torch[:1], initial_pose[:1])
    torch.testing.assert_close(cube_object.data.root_link_vel_w.torch[:1], initial_velocity[:1])
    raw_pose = wp.to_torch(cube_object.root_view.get_transforms()).to(device)
    torch.testing.assert_close(raw_pose, torch.cat((initial_pose[:1], target_pose)))

    # With a center-of-mass offset, every writer round-trips through its own frame.
    offset = torch.tensor([0.1, 0.0, 0.0], device=device).repeat(_NUM_ENVS, 1)
    coms = cube_object.data.body_com_pose_b.torch.clone()
    coms[:, 0, :3] = offset
    coms[:, 0, 3:] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
    cube_object.set_coms_index(coms=coms)
    # random pose and velocity so frame conversions see a non-trivial rotation
    rand_state = torch.rand(_NUM_ENVS, 13, device=device)
    rand_state[..., :3] += rest_pose[:, :3]
    rand_state[..., 3:7] = torch.nn.functional.normalize(rand_state[..., 3:7], dim=-1)
    env_idx = torch.arange(_NUM_ENVS, dtype=torch.int32, device=device)
    for state_location in ("com", "link", "root"):
        for env_ids in (None, env_idx):
            if state_location == "com":
                cube_object.write_root_com_pose_to_sim_index(root_pose=rand_state[..., :7], env_ids=env_ids)
                cube_object.write_root_com_velocity_to_sim_index(root_velocity=rand_state[..., 7:], env_ids=env_ids)
                torch.testing.assert_close(rand_state[..., :7], cube_object.data.root_com_pose_w.torch)
                torch.testing.assert_close(rand_state[..., 7:], cube_object.data.root_com_vel_w.torch)
            elif state_location == "link":
                cube_object.write_root_link_pose_to_sim_index(root_pose=rand_state[..., :7], env_ids=env_ids)
                cube_object.write_root_link_velocity_to_sim_index(root_velocity=rand_state[..., 7:], env_ids=env_ids)
                torch.testing.assert_close(rand_state[..., :7], cube_object.data.root_link_pose_w.torch)
                torch.testing.assert_close(rand_state[..., 7:], cube_object.data.root_link_vel_w.torch)
            else:
                cube_object.write_root_pose_to_sim_index(root_pose=rand_state[..., :7], env_ids=env_ids)
                cube_object.write_root_velocity_to_sim_index(root_velocity=rand_state[..., 7:], env_ids=env_ids)
                torch.testing.assert_close(rand_state[..., :7], cube_object.data.root_link_pose_w.torch)
                torch.testing.assert_close(rand_state[..., 7:], cube_object.data.root_com_vel_w.torch)

            # the frame that was not written must follow through the center-of-mass offset
            root_link_pose_w = cube_object.data.root_link_pose_w.torch
            body_com_pose_b = cube_object.data.body_com_pose_b.torch
            expected_com_pos, expected_com_quat = combine_frame_transforms(
                root_link_pose_w[:, :3],
                root_link_pose_w[:, 3:],
                body_com_pose_b[:, 0, :3],
                body_com_pose_b[:, 0, 3:7],
            )
            torch.testing.assert_close(
                torch.cat((expected_com_pos, expected_com_quat), dim=1), cube_object.data.root_com_pose_w.torch
            )
            torch.testing.assert_close(
                cube_object.data.root_com_vel_w.torch[:, 3:], cube_object.data.root_link_vel_w.torch[:, 3:]
            )

    # Spinning about the center of mass keeps it in place while the link frame orbits it.
    scene.place_cubes_at_rest(rest_pose)
    spin_twist = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.3], device=device).repeat(_NUM_ENVS, 1)
    initial_com_pos = cube_object.data.root_com_pose_w.torch[:, :3].clone()
    for _ in range(10):
        cube_object.write_root_com_velocity_to_sim_index(root_velocity=spin_twist)
        scene.step()
        root_link_pose_w = cube_object.data.root_link_pose_w.torch
        root_link_vel_w = cube_object.data.root_link_vel_w.torch
        root_com_pose_w = cube_object.data.root_com_pose_w.torch
        root_com_vel_w = cube_object.data.root_com_vel_w.torch
        body_link_pose_w = cube_object.data.body_link_pose_w.torch
        body_com_pose_w = cube_object.data.body_com_pose_w.torch
        # center of mass position will be constant (i.e. spinning around com)
        torch.testing.assert_close(root_com_pose_w[:, :3], initial_com_pos)
        torch.testing.assert_close(body_com_pose_w[:, 0, :3], initial_com_pos)
        # link position will be moving but should stay constant away from center of mass
        torch.testing.assert_close(
            -offset, quat_apply_inverse(root_link_pose_w[:, 3:], root_link_pose_w[:, :3] - root_com_pose_w[:, :3])
        )
        # orientation of com will be a constant rotation from link orientation
        com_quat_w = quat_mul(body_link_pose_w[..., 3:], cube_object.data.body_com_quat_b.torch)
        torch.testing.assert_close(com_quat_w, body_com_pose_w[..., 3:])
        torch.testing.assert_close(com_quat_w[:, 0], root_com_pose_w[:, 3:])
        # center of mass is at rest while the link frame moves with angular velocity cross offset
        torch.testing.assert_close(torch.zeros_like(root_com_vel_w[:, :3]), root_com_vel_w[:, :3])
        lin_vel_rel_root = quat_apply_inverse(root_link_pose_w[:, 3:], root_link_vel_w[:, :3])
        lin_vel_rel_gt = torch.linalg.cross(spin_twist[:, 3:], -offset)
        torch.testing.assert_close(lin_vel_rel_gt, lin_vel_rel_root, atol=1e-4, rtol=1e-4)
        # ang_vel will always match
        torch.testing.assert_close(root_com_vel_w[:, 3:], root_link_vel_w[:, 3:])


def test_rigid_object_wrench_delivery_and_reset(rigid_object_scene: _RigidObjectScene):
    """External wrenches act in the frame they are given in on the selected root; reset clears them."""
    scene = rigid_object_scene
    device = scene.device
    cube_object = scene.cubes
    rest_pose = torch.cat(
        (scene.origins["cubes"], torch.tensor(_yaw_quat(0.5 * math.pi), device=device).expand(_NUM_ENVS, 4)), -1
    )
    rest_pose[:, 2] += 1.0
    # Give both environments the inertial properties of environment 0 so that their responses are comparable.
    for setter, kwarg, name in (
        (cube_object.set_masses_index, "masses", "body_mass"),
        (cube_object.set_coms_index, "coms", "body_com_pose_b"),
        (cube_object.set_inertias_index, "inertias", "body_inertia"),
    ):
        value = getattr(cube_object.data, name).torch[:1]
        setter(**{kwarg: value.expand(_NUM_ENVS, *value.shape[1:]).contiguous()})

    # A body-frame force on environment 1 accelerates only that cube, along the rotated force direction.
    scene.place_cubes_at_rest(rest_pose)
    cube_object.permanent_wrench_composer.set_forces_and_torques_index(
        forces=torch.tensor([[[6.0, 0.0, 0.0]]], device=device), env_ids=[1]
    )
    scene.step()
    velocity = cube_object.data.root_com_lin_vel_w.torch
    assert velocity[1, 1] > 1e-2, velocity
    torch.testing.assert_close(velocity[0], torch.zeros(3, device=device), atol=1e-5, rtol=0.0)

    # A world-frame force, and a force at a position, match their body-frame counterparts.
    for local_wrench, global_wrench, response in (
        ({"forces": [[[6.0, 0.0, 0.0]]]}, {"forces": [[[0.0, 6.0, 0.0]]]}, "root_com_lin_vel_w"),
        (
            {"forces": [[[0.0, 0.0, 6.0]]], "positions": [[[0.0, 0.1, 0.0]]]},
            {"forces": [[[0.0, 0.0, 6.0]]], "positions": [[[-0.1, 0.0, 0.0]]]},
            "root_com_ang_vel_b",
        ),
    ):
        scene.place_cubes_at_rest(rest_pose)
        local_wrench = {key: torch.tensor(value, device=device) for key, value in local_wrench.items()}
        global_wrench = {key: torch.tensor(value, device=device) for key, value in global_wrench.items()}
        if "positions" in global_wrench:
            # World-frame positions add the rotated body-frame lever arm to the center of mass.
            global_wrench["positions"] = global_wrench["positions"] + cube_object.data.body_com_pos_w.torch[:1]
        cube_object.permanent_wrench_composer.set_forces_and_torques_index(env_ids=[1], **local_wrench)
        cube_object.permanent_wrench_composer.set_forces_and_torques_index(env_ids=[0], is_global=True, **global_wrench)
        scene.step()
        response_value = getattr(cube_object.data, response).torch
        torch.testing.assert_close(response_value[0], response_value[1], atol=1e-4, rtol=1e-3)
    # An upward force 0.1 m along the body y-axis rolls the cube about its x-axis.
    assert torch.all(cube_object.data.root_com_ang_vel_b.torch[:, 0] > 0.1)
    assert torch.all(
        quat_apply(cube_object.data.root_link_quat_w.torch, torch.tensor([[1.0, 0.0, 0.0]] * 2, device=device))[:, 1]
        > 0.99
    )

    # A partial reset clears only the selected environment; a full reset deactivates the composers.
    composers = (cube_object.instantaneous_wrench_composer, cube_object.permanent_wrench_composer)
    ones = torch.ones((_NUM_ENVS, 1, 3), device=device)
    cube_object.permanent_wrench_composer.set_forces_and_torques_index(forces=ones, torques=ones)
    cube_object.instantaneous_wrench_composer.add_forces_and_torques_index(forces=ones, torques=ones)
    cube_object.reset(env_ids=torch.tensor([0], device=device))
    for composer in composers:
        assert composer.active
        for buffer in (composer.out_force_b.torch, composer.out_torque_b.torch):
            assert torch.count_nonzero(buffer[0]) == 0
            assert torch.count_nonzero(buffer[1:]) == buffer[1:].numel()
    cube_object.reset()
    for composer in composers:
        assert not composer.active
        assert torch.count_nonzero(composer.out_force_b.torch) == 0
        assert torch.count_nonzero(composer.out_torque_b.torch) == 0
