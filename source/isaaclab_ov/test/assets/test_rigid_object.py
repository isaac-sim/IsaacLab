# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none


"""Real-backend tests for the OVPhysX RigidObject.

Run via ``./scripts/run_ovphysx.sh -m pytest`` (kitless, no the Kit launcher).

The OVPhysX runtime fixes device mode (CPU vs GPU) when the process creates
its first ``ovphysx.PhysX`` instance and cannot switch it without a process
restart. Full coverage therefore requires two separate pytest
invocations -- once with ``-k 'cpu'`` and once with ``-k 'cuda:0'``.  The
``_ovphysx_skip_other_device`` autouse fixture below preempts the manager's
:exc:`RuntimeError` by ``pytest.skip``-ing on the unlocked device so
single-device runs finish cleanly.
"""

from __future__ import annotations

import logging
import sys
from typing import Literal

import pytest
import torch
import warp as wp
from flaky import flaky

from pxr import UsdPhysics

from isaaclab.test.utils import test_devices

# The OVPhysX runtime wheel is optional. Skip gracefully when it is not installed;
# CI jobs that need OVPhysX coverage install it explicitly.
pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

from isaaclab_ov import tensor_types as TT  # noqa: E402
from isaaclab_ov.assets import RigidObject  # noqa: E402
from isaaclab_ov.physics import OvPhysxCfg, OvPhysxManager  # noqa: E402
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg  # noqa: E402
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab import cloner  # noqa: E402
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, build_simulation_context  # noqa: E402
from isaaclab.sim.spawners import materials  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    combine_frame_transforms,
    default_orientation,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
    quat_rotate,
    random_orientation,
)

wp.init()

pytestmark = pytest.mark.device_split

_logger = logging.getLogger(__name__)


_LOCKED_DEVICE: list[str | None] = [None]
"""Device the session pins to on the first parametrized test that runs."""


@pytest.fixture(autouse=True)
def _ovphysx_skip_other_device(request):
    """Skip parametrized tests on the device the session is not pinned to.

    See the module docstring for the wheel's process-global device-mode lock.
    """
    callspec = getattr(request.node, "callspec", None)
    device = callspec.params.get("device") if callspec is not None else None
    if device is None:
        return
    locked = _LOCKED_DEVICE[0]
    if locked is None:
        _LOCKED_DEVICE[0] = device
        return
    if device != locked:
        pytest.skip(
            f"ovphysx process-global device lock is held by '{locked}'; cannot run '{device}' "
            "tests in the same session.  Run pytest twice (once per device) for full coverage."
        )


def _ovphysx_sim_context(device: str, **kwargs):
    """Wrapper around :func:`build_simulation_context` that injects OVPhysX cfg.

    PhysX tests pass ``device=device`` directly and let
    :func:`build_simulation_context` build a default :class:`SimulationCfg`.
    OVPhysX needs ``physics=OvPhysxCfg()`` set on the cfg so the manager
    dispatches to OVPhysX rather than PhysX, so we build the cfg here and
    pass it through.  ``gravity_enabled`` is consumed locally (it is ignored
    by ``build_simulation_context`` once a ``sim_cfg`` is provided).
    ``add_ground_plane``, ``auto_add_lighting``, and other kwargs continue
    to flow through ``build_simulation_context`` as before.
    """
    dt = kwargs.pop("dt", 1.0 / 60.0)
    gravity_enabled = kwargs.pop("gravity_enabled", True)
    gravity = (0.0, 0.0, -9.81) if gravity_enabled else (0.0, 0.0, 0.0)
    sim_cfg = SimulationCfg(physics=OvPhysxCfg(), device=device, dt=dt, gravity=gravity)
    return build_simulation_context(device=device, sim_cfg=sim_cfg, **kwargs)


@configclass
class HeterogeneousRigidSceneCfg(InteractiveSceneCfg):
    """Two object variants dropping onto an independently cloned support."""

    support = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Support",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 0.2),
            rigid_props=sim_utils.RigidBodyBaseCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionBaseCfg(),
        ),
    )
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[sim_utils.CuboidCfg(size=(0.2, 0.2, 0.2)), sim_utils.SphereCfg(radius=0.1)],
            rigid_props=sim_utils.RigidBodyBaseCfg(),
            collision_props=sim_utils.CollisionBaseCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize(
    ("scene_variant", "filter_collisions"),
    [
        (variant, filter_collisions)
        for variant in ("native", "usd_and_native", "usd", "usd_nested")
        for filter_collisions in (False, True)
    ]
    + [("compliant", False)],
)
def test_heterogeneous_clone_contacts(device, scene_variant, filter_collisions):
    """Scene variants contact their own support through cloning paths and compliant contacts."""
    if scene_variant == "compliant" and not device.startswith("cuda"):
        pytest.skip("Compliant-contact path is CUDA-only")
    with _ovphysx_sim_context(device=device, dt=1.0 / 120.0) as sim:
        cfg = HeterogeneousRigidSceneCfg(num_envs=6, env_spacing=2.0, filter_collisions=filter_collisions)
        compliant = scene_variant == "compliant"
        if compliant:
            material = PhysxRigidBodyMaterialCfg(
                compliant_contact_stiffness=15700.0,
                compliant_contact_damping=1120.0,
                compliant_contact_acceleration_spring=True,
            )
            cfg.support.spawn.physics_material = cfg.object.spawn.physics_material = material
            cfg.object.spawn.mass_props.mass = 0.02
            cfg.object.spawn.rigid_props = [
                sim_utils.UsdPhysicsRigidBodyCfg(),
                PhysxRigidBodyCfg(
                    sleep_threshold=0.0, stabilization_threshold=0.0, solver_position_iteration_count=128
                ),
            ]
        if scene_variant not in ("native", "compliant"):
            cfg.clone_cfg.clone_template = "/Scenes/World_{}"
            cfg.support.cloning_contexts = ("isaaclab.cloner:UsdReplicateContext",)
            if scene_variant == "usd_and_native":
                cfg.support.cloning_contexts += ("isaaclab_ov.cloner:OvPhysxReplicateContext",)
        if scene_variant == "usd_nested":
            cfg.support.prim_path = "{ENV_REGEX_NS}/Assembly/Support"
            cfg.object.spawn.assets_cfg = cfg.object.spawn.assets_cfg[:1]
            spawn = sim_utils.SpawnerCfg(func=lambda path, cfg, **kwargs: sim.stage.DefinePrim(path, "Xform"))
            cfg.assembly = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Assembly", spawn=spawn, cloning_contexts=None)
            cfg.clone_cfg.clone_combinations = [cloner.InclusionSet(assets=names) for names in (["assembly"], [])]
        scene = InteractiveScene(cfg)
        sim.reset()
        for _ in range(360 if compliant else 240):
            sim.step()
            scene.update(sim.get_physics_dt())
        positions = scene["object"].data.root_pos_w.torch
        torch.testing.assert_close(positions[:, 2], torch.full_like(positions[:, 2], 0.2), atol=0.02, rtol=0.0)
        torch.testing.assert_close(positions[:, :2], scene.env_origins[:, :2], atol=0.02, rtol=0.0)
        if compliant:
            heights = positions[:, 2].reshape(2, 3)
            torch.testing.assert_close(heights[:, 1:], heights[:, :1].expand(-1, 2), atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("device", test_devices())
def test_heterogeneous_clone_collision_isolation(device):
    """Collision groups isolate overlapping environments, including both retained sources."""
    with _ovphysx_sim_context(device=device, dt=1.0 / 120.0) as sim:
        # Without GPU environment-ID filtering, 2048 overlapping worlds exhaust broadphase pairs.
        num_envs = 2048 if device.startswith("cuda") else 6
        scene = InteractiveScene(HeterogeneousRigidSceneCfg(num_envs=num_envs, env_spacing=0.0))
        sim.reset()
        for _ in range(240):
            sim.step()
            scene.update(sim.get_physics_dt())
        positions = scene["object"].data.root_pos_w.torch
        expected = scene.env_origins.clone()
        expected[:, 2] = 0.2
        torch.testing.assert_close(positions, expected, atol=0.02, rtol=0.0)


@pytest.mark.parametrize("device", test_devices())
def test_heterogeneous_clone_indexed_state(device):
    """Indexed reads and writes use environment order across six interleaved variants."""
    with _ovphysx_sim_context(device=device, gravity_enabled=False) as sim:
        scene = InteractiveScene(HeterogeneousRigidSceneCfg(num_envs=6, env_spacing=2.0))
        sim.reset()
        obj = scene["object"]
        expected_paths = [f"/World/envs/env_{i}/Object" for i in range(scene.num_envs)]
        assert obj.root_view.prim_paths == expected_paths
        torch.testing.assert_close(obj.data.root_pos_w.torch[:, :2], scene.env_origins[:, :2])
        selected = torch.tensor([3], device=device)
        pose = obj.data.root_pose_w.torch[selected].clone()
        pose[:, 2] = 3.0
        obj.write_root_pose_to_sim(pose, env_ids=selected)
        sim.step()
        scene.update(sim.get_physics_dt())
        # An independent exact-path binding detects a write to the wrong physical object.
        binding = OvPhysxManager.get_physx_instance().create_tensor_binding(
            prim_paths=expected_paths, tensor_type=TT.RIGID_BODY_POSE
        )
        try:
            actual = torch.empty(binding.shape, device=device)
            binding.read(actual)
            expected_height = torch.ones(scene.num_envs, device=device)
            expected_height[selected] = 3.0
            torch.testing.assert_close(actual[:, 2], expected_height)
            torch.testing.assert_close(obj.data.root_pos_w.torch, actual[:, :3])
        finally:
            binding.destroy()


def generate_cubes_scene(
    num_cubes: int = 1,
    height=1.0,
    api: Literal["none", "rigid_body"] = "rigid_body",
    kinematic_enabled: bool = False,
    device: str = "cuda:0",
) -> tuple[RigidObject, torch.Tensor]:
    """Generate a scene with the provided number of cubes.

    Args:
        num_cubes: Number of cubes to generate.
        height: Height of the cubes.
        api: The type of API that the cubes should have.
        kinematic_enabled: Whether the cubes are kinematic.
        device: Device to use for the simulation.

    Returns:
        A tuple containing the rigid object representing the cubes and the origins of the cubes.

    """
    origins = torch.tensor([(i * 1.0, 0, height) for i in range(num_cubes)]).to(device)
    # Create Top-level Xforms, one for each cube
    for i, origin in enumerate(origins):
        sim_utils.create_prim(f"/World/Table_{i}", "Xform", translation=origin)

    # Resolve spawn configuration
    if api == "none":
        # since no rigid body properties defined, this is just a static collider
        spawn_cfg = sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        )
    elif api == "rigid_body":
        spawn_cfg = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=kinematic_enabled),
        )
    else:
        raise ValueError(f"Unknown api: {api}")

    # Create rigid object.  OVPhysX matches prim paths via fnmatch globs (not regex),
    # so use ``Table_*`` rather than the PhysX ``Table_.*`` form.
    cube_object_cfg = RigidObjectCfg(
        prim_path="/World/Table_[^/]+/Object",
        spawn=spawn_cfg,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, height)),
    )
    cube_object = RigidObject(cfg=cube_object_cfg)

    return cube_object, origins


# ---------------------------------------------------------------------------
# Per-shape material helpers (friction / restitution)
#
# OVPhysX exposes per-collision-shape material as the
# ``rigid_body_shape_friction_and_restitution`` tensor binding (shape ``[N, S, 3]`` =
# static friction, dynamic friction, restitution), addressed through the OvPhysxView.
# (The PhysX backend instead uses ``root_view.get/set_material_properties`` / ``max_shapes``.)
# ---------------------------------------------------------------------------


def _num_shapes(cube_object) -> int:
    """Per-body collision-shape count, from the material binding's ``[N, S, 3]`` shape."""
    return cube_object.root_view.binding_for(TT.RIGID_BODY_SHAPE_FRICTION_AND_RESTITUTION).shape[1]


def _write_shape_material(cube_object, materials_nS3: torch.Tensor) -> None:
    """Write a full ``[N, S, 3]`` (static friction, dynamic friction, restitution) tensor via the view."""
    cube_object.root_view.set_attribute(
        TT.RIGID_BODY_SHAPE_FRICTION_AND_RESTITUTION, wp.from_torch(materials_nS3.contiguous(), dtype=wp.float32)
    )


@pytest.mark.parametrize("num_cubes", [2])
@pytest.mark.parametrize("device", test_devices())
def test_initialization_with_kinematic_enabled(num_cubes, device):
    """Test that initialization for prim with kinematic flag enabled."""
    with _ovphysx_sim_context(device=device, auto_add_lighting=True) as sim:
        cube_object, origins = generate_cubes_scene(num_cubes=num_cubes, kinematic_enabled=True, device=device)
        assert sys.getrefcount(cube_object) < 10
        sim.reset()
        provider = sim.get_scene_data_provider()
        assert provider.transform_count == num_cubes
        assert set(provider.backend.transform_paths) == {f"/World/Table_{i}/Object" for i in range(num_cubes)}
        assert cube_object.is_initialized
        assert len(cube_object.body_names) == 1
        assert cube_object.data.root_pos_w.torch.shape == (num_cubes, 3)
        assert cube_object.data.root_quat_w.torch.shape == (num_cubes, 4)
        for _ in range(2):
            sim.step()
            cube_object.update(sim.cfg.dt)
            default_root_pose = cube_object.data.default_root_pose.torch.clone()
            default_root_vel = cube_object.data.default_root_vel.torch.clone()
            default_root_pose[:, :3] += origins
            torch.testing.assert_close(cube_object.data.root_link_pose_w.torch, default_root_pose)
            torch.testing.assert_close(cube_object.data.root_com_vel_w.torch, default_root_vel)


@pytest.mark.parametrize("num_cubes", [2])
@pytest.mark.parametrize("device", test_devices())
def test_initialization_with_no_rigid_body(num_cubes, device):
    """Test that initialization fails when no rigid body is found at the provided prim path."""
    with _ovphysx_sim_context(device=device, auto_add_lighting=True) as sim:
        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, api="none", device=device)
        assert sys.getrefcount(cube_object) < 10
        with pytest.raises(RuntimeError, match="Expected 1 prims"):
            sim.reset()


@pytest.mark.parametrize("num_cubes", [2])
@pytest.mark.parametrize("device", test_devices())
@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason="OVPhysX RigidObject has no ArticulationRootAPI guard; the rigid body initializes normally.",
)
def test_initialization_with_articulation_root(num_cubes, device):
    """Test that initialization fails when an articulation root is found at the provided prim path."""
    with _ovphysx_sim_context(device=device, auto_add_lighting=True) as sim:
        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, device=device)
        stage = sim_utils.get_current_stage()
        for i in range(num_cubes):
            UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(f"/World/Table_{i}/Object"))
        assert sys.getrefcount(cube_object) < 10
        with pytest.raises(RuntimeError):
            sim.reset()


@pytest.mark.parametrize("num_cubes", [2])
@pytest.mark.parametrize("device", test_devices())
def test_external_force_on_single_body(num_cubes, device):
    """Test application of external force on the base of the object."""
    with _ovphysx_sim_context(device=device, add_ground_plane=True, auto_add_lighting=True) as sim:
        cube_object, origins = generate_cubes_scene(num_cubes=num_cubes, device=device)
        sim.reset()
        body_ids, _ = cube_object.find_bodies(".*")
        external_wrench_b = torch.zeros(cube_object.num_instances, len(body_ids), 6, device=sim.device)
        external_wrench_b[0::2, :, 2] = 9.81 * cube_object.data.body_mass.torch[0]
        for i in range(5):
            root_pose = cube_object.data.default_root_pose.torch.clone()
            root_vel = cube_object.data.default_root_vel.torch.clone()
            root_pose[:, :3] = origins
            cube_object.write_root_pose_to_sim_index(root_pose=root_pose)
            cube_object.write_root_velocity_to_sim_index(root_velocity=root_vel)
            cube_object.reset()
            is_global = i % 2 == 0
            positions = cube_object.data.body_com_pos_w.torch[:, body_ids, :3] if is_global else None
            cube_object.permanent_wrench_composer.set_forces_and_torques_index(
                forces=external_wrench_b[..., :3],
                torques=external_wrench_b[..., 3:],
                positions=positions,
                body_ids=body_ids,
                is_global=is_global,
            )
            for _ in range(5):
                cube_object.write_data_to_sim()
                sim.step()
                cube_object.update(sim.cfg.dt)
            torch.testing.assert_close(
                cube_object.data.root_pos_w.torch[0::2, 2], torch.ones(num_cubes // 2, device=sim.device)
            )
            assert torch.all(cube_object.data.root_pos_w.torch[1::2, 2] < 1.0)


@pytest.mark.parametrize("num_cubes", [2])
@pytest.mark.parametrize("device", test_devices())
def test_external_force_on_single_body_at_position(num_cubes, device):
    """Test application of external force on the base of the object at a specific position."""
    with _ovphysx_sim_context(device=device, add_ground_plane=True, auto_add_lighting=True) as sim:
        cube_object, origins = generate_cubes_scene(num_cubes=num_cubes, device=device)
        sim.reset()
        body_ids, _ = cube_object.find_bodies(".*")
        external_wrench_b = torch.zeros(cube_object.num_instances, len(body_ids), 6, device=sim.device)
        external_wrench_positions_b = torch.zeros(cube_object.num_instances, len(body_ids), 3, device=sim.device)
        external_wrench_b[0::2, :, 2] = 500.0
        external_wrench_positions_b[0::2, :, 1] = 1.0
        desired_force = torch.zeros(cube_object.num_instances, len(body_ids), 3, device=sim.device)
        desired_force[0::2, :, 2] = 1000.0
        desired_torque = torch.zeros(cube_object.num_instances, len(body_ids), 3, device=sim.device)
        desired_torque[0::2, :, 0] = 1000.0
        for i in range(5):
            root_pose = cube_object.data.default_root_pose.torch.clone()
            root_vel = cube_object.data.default_root_vel.torch.clone()
            root_pose[:, :3] = origins
            cube_object.write_root_pose_to_sim_index(root_pose=root_pose)
            cube_object.write_root_velocity_to_sim_index(root_velocity=root_vel)
            cube_object.reset()
            is_global = i % 2 == 0
            external_wrench_positions_b.zero_()
            external_wrench_positions_b[..., 1] = 1.0
            if is_global:
                external_wrench_positions_b += cube_object.data.body_com_pos_w.torch[:, body_ids, :3]
            cube_object.permanent_wrench_composer.set_forces_and_torques_index(
                forces=external_wrench_b[..., :3],
                torques=external_wrench_b[..., 3:],
                positions=external_wrench_positions_b,
                body_ids=body_ids,
                is_global=is_global,
            )
            cube_object.permanent_wrench_composer.add_forces_and_torques_index(
                forces=external_wrench_b[..., :3],
                torques=external_wrench_b[..., 3:],
                positions=external_wrench_positions_b,
                body_ids=body_ids,
                is_global=is_global,
            )
            torch.testing.assert_close(
                cube_object._permanent_wrench_composer.composed_force.torch[:, 0, :],
                desired_force[:, 0, :],
                rtol=1e-6,
                atol=1e-7,
            )
            torch.testing.assert_close(
                cube_object._permanent_wrench_composer.composed_torque.torch[:, 0, :],
                desired_torque[:, 0, :],
                rtol=1e-6,
                atol=1e-7,
            )
            for _ in range(5):
                cube_object.write_data_to_sim()
                sim.step()
                cube_object.update(sim.cfg.dt)
            assert torch.all(torch.abs(cube_object.data.root_ang_vel_b.torch[0::2, 0]) > 0.1)
            assert torch.all(cube_object.data.root_pos_w.torch[1::2, 2] < 1.0)


@pytest.mark.parametrize("num_cubes", [2])
@pytest.mark.parametrize("device", test_devices())
def test_set_rigid_object_state(num_cubes, device):
    """Test setting the state of the rigid object."""
    with _ovphysx_sim_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, device=device)
        sim.reset()
        state_types = ["root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w"]
        for state_type_to_randomize in state_types:
            state_dict = {
                "root_pos_w": torch.zeros_like(cube_object.data.root_pos_w.torch, device=sim.device),
                "root_quat_w": default_orientation(num=num_cubes, device=sim.device),
                "root_lin_vel_w": torch.zeros_like(cube_object.data.root_lin_vel_w.torch, device=sim.device),
                "root_ang_vel_w": torch.zeros_like(cube_object.data.root_ang_vel_w.torch, device=sim.device),
            }
            for _ in range(5):
                cube_object.reset()
                if state_type_to_randomize == "root_quat_w":
                    state_dict[state_type_to_randomize] = random_orientation(num=num_cubes, device=sim.device)
                else:
                    state_dict[state_type_to_randomize] = torch.randn(num_cubes, 3, device=sim.device)
                for _ in range(5):
                    root_pose = torch.cat([state_dict["root_pos_w"], state_dict["root_quat_w"]], dim=-1)
                    root_vel = torch.cat([state_dict["root_lin_vel_w"], state_dict["root_ang_vel_w"]], dim=-1)
                    cube_object.write_root_pose_to_sim_index(root_pose=root_pose)
                    cube_object.write_root_velocity_to_sim_index(root_velocity=root_vel)
                    sim.step()
                    for key, expected_value in state_dict.items():
                        torch.testing.assert_close(
                            getattr(cube_object.data, key).torch, expected_value, rtol=1e-3, atol=1e-3
                        )
                    cube_object.update(sim.cfg.dt)


@pytest.mark.parametrize("num_cubes", [2])
@pytest.mark.parametrize("device", test_devices())
def test_reset_rigid_object(num_cubes, device):
    """Test resetting the state of the rigid object."""
    with _ovphysx_sim_context(device=device, gravity_enabled=True, auto_add_lighting=True) as sim:
        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, device=device)
        sim.reset()
        for i in range(5):
            sim.step()
            cube_object.update(sim.cfg.dt)
            root_pose = cube_object.data.default_root_pose.torch.clone()
            root_pose[:, :3] = torch.randn(num_cubes, 3, device=sim.device)
            root_pose[:, 3:7] = random_orientation(num=num_cubes, device=sim.device)
            cube_object.write_root_pose_to_sim_index(root_pose=root_pose)
            root_vel = cube_object.data.default_root_vel.torch.clone()
            cube_object.write_root_velocity_to_sim_index(root_velocity=root_vel)
            if i % 2 == 0:
                cube_object.permanent_wrench_composer.set_forces_and_torques_index(
                    forces=torch.ones((num_cubes, 1, 3), device=device),
                    torques=torch.ones((num_cubes, 1, 3), device=device),
                )
                cube_object.instantaneous_wrench_composer.add_forces_and_torques_index(
                    forces=torch.ones((num_cubes, 1, 3), device=device),
                    torques=torch.ones((num_cubes, 1, 3), device=device),
                )
                assert cube_object._instantaneous_wrench_composer.active
                assert cube_object._permanent_wrench_composer.active
                cube_object.reset()
                assert not cube_object._instantaneous_wrench_composer.active
                assert not cube_object._permanent_wrench_composer.active
                assert torch.count_nonzero(cube_object._instantaneous_wrench_composer.composed_force.torch) == 0
                assert torch.count_nonzero(cube_object._instantaneous_wrench_composer.composed_torque.torch) == 0
                assert torch.count_nonzero(cube_object._permanent_wrench_composer.composed_force.torch) == 0
                assert torch.count_nonzero(cube_object._permanent_wrench_composer.composed_torque.torch) == 0


@pytest.mark.parametrize("num_cubes", [2])
@pytest.mark.parametrize("device", test_devices())
def test_rigid_body_no_friction(num_cubes, device):
    """Test that a rigid object with no friction maintains its tangential velocity on a plane."""
    with _ovphysx_sim_context(device=device, auto_add_lighting=True) as sim:
        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, height=0.0, device=device)
        cfg = sim_utils.GroundPlaneCfg(
            physics_material=materials.RigidBodyMaterialBaseCfg(
                static_friction=0.0, dynamic_friction=0.0, restitution=0.0
            )
        )
        cfg.func("/World/GroundPlane", cfg)
        sim.reset()
        num_shapes = _num_shapes(cube_object)
        cube_materials = torch.zeros(num_cubes, num_shapes, 3, device="cpu")
        _write_shape_material(cube_object, cube_materials)
        for _ in range(30):
            sim.step()
            cube_object.update(sim.cfg.dt)
        initial_velocity = torch.zeros((num_cubes, 6), device=device)
        initial_velocity[:, 0] = 0.1
        cube_object.write_root_velocity_to_sim_index(root_velocity=initial_velocity)
        tolerance = 1e-2 if device == "cuda:0" else 1e-5
        for _ in range(5):
            sim.step()
            cube_object.update(sim.cfg.dt)
            torch.testing.assert_close(
                cube_object.data.root_lin_vel_w.torch[:, :2], initial_velocity[:, :2], rtol=1e-5, atol=tolerance
            )


@pytest.mark.parametrize("num_cubes", [2])
@pytest.mark.parametrize("device", test_devices())
def test_rigid_body_with_static_friction(num_cubes, device):
    """Test that static friction applied to rigid object works as expected."""
    with _ovphysx_sim_context(device=device, dt=0.01, auto_add_lighting=True) as sim:
        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, height=0.03125, device=device)
        mu = 0.5
        cfg = sim_utils.GroundPlaneCfg(
            physics_material=materials.RigidBodyMaterialBaseCfg(static_friction=mu, dynamic_friction=mu)
        )
        cfg.func("/World/GroundPlane", cfg)
        sim.reset()
        num_shapes = _num_shapes(cube_object)
        cube_materials = torch.zeros(num_cubes, num_shapes, 3, device="cpu")
        cube_materials[..., 0] = mu
        cube_materials[..., 1] = mu
        _write_shape_material(cube_object, cube_materials)
        for _ in range(100):
            sim.step()
            cube_object.update(sim.cfg.dt)
        cube_object.write_root_velocity_to_sim_index(root_velocity=torch.zeros((num_cubes, 6), device=device))
        cube_mass = cube_object.data.body_mass.torch[:, 0]
        gravity_magnitude = abs(sim.cfg.gravity[2])
        for force in "below_mu", "above_mu":
            cube_object.write_root_velocity_to_sim_index(root_velocity=torch.zeros((num_cubes, 6), device=device))
            external_wrench_b = torch.zeros((num_cubes, 1, 6), device=device)
            factor = 0.99 if force == "below_mu" else 1.01
            external_wrench_b[:, 0, 0] = mu * cube_mass * gravity_magnitude * factor
            cube_object.permanent_wrench_composer.set_forces_and_torques_index(
                forces=external_wrench_b[..., :3], torques=external_wrench_b[..., 3:]
            )
            initial_root_pos = cube_object.data.root_pos_w.torch.clone()
            for _ in range(200):
                cube_object.write_data_to_sim()
                sim.step()
                cube_object.update(sim.cfg.dt)
                if force == "below_mu":
                    torch.testing.assert_close(
                        cube_object.data.root_pos_w.torch, initial_root_pos, rtol=2e-3, atol=2e-3
                    )
            if force == "above_mu":
                assert (cube_object.data.root_pos_w.torch[..., 0] - initial_root_pos[..., 0] > 0.02).all()


@pytest.mark.parametrize("num_cubes", [2])
@pytest.mark.parametrize("device", test_devices())
def test_rigid_body_with_restitution(num_cubes, device):
    """Test that restitution when applied to rigid object works as expected."""
    for expected_collision_type in "partially_elastic", "inelastic":
        with _ovphysx_sim_context(device=device, auto_add_lighting=True) as sim:
            cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, height=1.0, device=device)
            restitution_coefficient = 0.0 if expected_collision_type == "inelastic" else 0.5
            cfg = sim_utils.GroundPlaneCfg(
                physics_material=materials.RigidBodyMaterialBaseCfg(restitution=restitution_coefficient)
            )
            cfg.func("/World/GroundPlane", cfg)
            sim.reset()
            root_pose = torch.zeros(num_cubes, 7, device=device)
            root_pose[:, 3] = 1.0
            for i in range(num_cubes):
                root_pose[i, 1] = float(i)
            root_pose[:, 2] = 1.0
            root_vel = torch.zeros(num_cubes, 6, device=device)
            root_vel[:, 2] = -1.0
            cube_object.write_root_pose_to_sim_index(root_pose=root_pose)
            cube_object.write_root_velocity_to_sim_index(root_velocity=root_vel)
            num_shapes = _num_shapes(cube_object)
            cube_materials = torch.zeros(num_cubes, num_shapes, 3, device="cpu")
            cube_materials[..., 2] = restitution_coefficient
            _write_shape_material(cube_object, cube_materials)
            curr_z_velocity = cube_object.data.root_lin_vel_w.torch[:, 2].clone()
            prev_z_velocity = curr_z_velocity
            for _ in range(100):
                sim.step()
                cube_object.update(sim.cfg.dt)
                curr_z_velocity = cube_object.data.root_lin_vel_w.torch[:, 2].clone()
                if expected_collision_type == "inelastic":
                    assert (curr_z_velocity <= 0.0).all()
                if torch.all(curr_z_velocity <= 0.0):
                    prev_z_velocity = curr_z_velocity
                else:
                    break
            if expected_collision_type == "partially_elastic":
                assert torch.all(torch.le(abs(curr_z_velocity), abs(prev_z_velocity)))
                assert (curr_z_velocity > 0.0).all()


@pytest.mark.parametrize("num_cubes", [2])
@pytest.mark.parametrize("device", test_devices())
def test_rigid_body_set_mass(num_cubes, device):
    """Test getting and setting mass of rigid object."""
    with _ovphysx_sim_context(
        device=device, gravity_enabled=False, add_ground_plane=True, auto_add_lighting=True
    ) as sim:
        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, height=1.0, device=device)
        sim.reset()
        original_masses = cube_object.data.body_mass.torch.clone()
        assert original_masses.shape == (num_cubes, 1)
        masses = original_masses + torch.FloatTensor(num_cubes, 1).uniform_(4, 8).to(sim.device)
        indices = torch.tensor(range(num_cubes), dtype=torch.int32)
        cube_object.set_masses_index(
            masses=wp.from_torch(masses.contiguous(), dtype=wp.float32),
            env_ids=wp.from_torch(indices, dtype=wp.int32),
        )
        torch.testing.assert_close(cube_object.data.body_mass.torch, masses)
        sim.step()
        cube_object.update(sim.cfg.dt)
        masses_to_check = wp.to_torch(cube_object.root_view.get_attribute(TT.RIGID_BODY_MASS))
        torch.testing.assert_close(masses_to_check.reshape(num_cubes, 1), masses.cpu())


@pytest.mark.parametrize("num_cubes", [2])
@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("gravity_enabled", [True, False])
def test_gravity_vec_w(num_cubes, device, gravity_enabled):
    """Test initialization and that gravity vector direction is set correctly for the rigid object."""
    with _ovphysx_sim_context(device=device, gravity_enabled=gravity_enabled) as sim:
        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, device=device)
        assert sys.getrefcount(cube_object) < 10
        gravity_dir = (0.0, 0.0, -1.0) if gravity_enabled else (0.0, 0.0, 0.0)
        sim.reset()
        assert cube_object.is_initialized
        assert len(cube_object.body_names) == 1
        assert cube_object.data.root_pos_w.torch.shape == (num_cubes, 3)
        assert cube_object.data.root_quat_w.torch.shape == (num_cubes, 4)
        assert cube_object.data.body_mass.torch.shape == (num_cubes, 1)
        assert cube_object.data.body_inertia.torch.shape == (num_cubes, 1, 9)
        assert cube_object.data.GRAVITY_VEC_W.torch[0, 0] == gravity_dir[0]
        assert cube_object.data.GRAVITY_VEC_W.torch[0, 1] == gravity_dir[1]
        assert cube_object.data.GRAVITY_VEC_W.torch[0, 2] == gravity_dir[2]
        for _ in range(2):
            sim.step()
            cube_object.update(sim.cfg.dt)
            gravity = torch.zeros(num_cubes, 1, 6, device=device)
            if gravity_enabled:
                gravity[:, :, 2] = -9.81
            torch.testing.assert_close(cube_object.data.body_acc_w.torch, gravity)


@pytest.mark.parametrize("num_cubes", [2])
@pytest.mark.parametrize("device", test_devices())
@flaky(max_runs=3, min_passes=1)
def test_body_root_state_properties(num_cubes, device):
    """Test the root and body link and COM state properties."""
    with _ovphysx_sim_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        cube_object, env_pos = generate_cubes_scene(num_cubes=num_cubes, height=0.0, device=device)
        env_idx = torch.arange(num_cubes, dtype=torch.int32)
        sim.reset()
        assert cube_object.is_initialized
        offset = torch.tensor([0.1, 0.0, 0.0], device=device).repeat(num_cubes, 1)
        com = cube_object.data.body_com_pose_b.torch.clone()
        com[..., :3] = offset.unsqueeze(1)
        cube_object.set_coms_index(
            coms=wp.from_torch(com.contiguous(), dtype=wp.transformf),
            env_ids=wp.from_torch(env_idx, dtype=wp.int32),
        )
        torch.testing.assert_close(cube_object.data.body_com_pose_b.torch, com)
        spin_twist = torch.zeros(6, device=device)
        spin_twist[5] = torch.randn(1, device=device)
        for _ in range(100):
            cube_object.write_root_velocity_to_sim_index(root_velocity=spin_twist.repeat(num_cubes, 1))
            sim.step()
            cube_object.update(sim.cfg.dt)
            root_link_pose_w = cube_object.data.root_link_pose_w.torch
            root_link_vel_w = cube_object.data.root_link_vel_w.torch
            root_com_pose_w = cube_object.data.root_com_pose_w.torch
            root_com_vel_w = cube_object.data.root_com_vel_w.torch
            body_link_pose_w = cube_object.data.body_link_pose_w.torch
            body_link_vel_w = cube_object.data.body_link_vel_w.torch
            body_com_pose_w = cube_object.data.body_com_pose_w.torch
            body_com_vel_w = cube_object.data.body_com_vel_w.torch
            torch.testing.assert_close(env_pos + offset, root_com_pose_w[..., :3])
            torch.testing.assert_close(env_pos + offset, body_com_pose_w[..., :3].squeeze(-2))
            root_link_state_pos_rel_com = quat_apply_inverse(
                root_link_pose_w[..., 3:], root_link_pose_w[..., :3] - root_com_pose_w[..., :3]
            )
            torch.testing.assert_close(-offset, root_link_state_pos_rel_com)
            body_link_state_pos_rel_com = quat_apply_inverse(
                body_link_pose_w[..., 3:], body_link_pose_w[..., :3] - body_com_pose_w[..., :3]
            )
            torch.testing.assert_close(-offset, body_link_state_pos_rel_com.squeeze(-2))
            com_quat_b = cube_object.data.body_com_quat_b.torch
            com_quat_w = quat_mul(body_link_pose_w[..., 3:], com_quat_b)
            torch.testing.assert_close(com_quat_w, body_com_pose_w[..., 3:])
            torch.testing.assert_close(com_quat_w.squeeze(-2), root_com_pose_w[..., 3:])
            torch.testing.assert_close(torch.zeros_like(root_com_vel_w[..., :3]), root_com_vel_w[..., :3])
            torch.testing.assert_close(torch.zeros_like(body_com_vel_w[..., :3]), body_com_vel_w[..., :3])
            lin_vel_rel_root_gt = quat_apply_inverse(root_link_pose_w[..., 3:], root_link_vel_w[..., :3])
            lin_vel_rel_body_gt = quat_apply_inverse(body_link_pose_w[..., 3:], body_link_vel_w[..., :3])
            com_lin_vel_rel_gt = quat_apply_inverse(root_link_pose_w[..., 3:], root_com_vel_w[..., :3])
            com_ang_vel_rel_gt = quat_apply_inverse(root_link_pose_w[..., 3:], root_com_vel_w[..., 3:])
            lin_vel_rel_gt = com_lin_vel_rel_gt + torch.linalg.cross(com_ang_vel_rel_gt, -offset)
            torch.testing.assert_close(lin_vel_rel_gt, lin_vel_rel_root_gt, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(lin_vel_rel_gt, lin_vel_rel_body_gt.squeeze(-2), atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(root_com_vel_w[..., 3:], root_link_vel_w[..., 3:])
            torch.testing.assert_close(body_com_vel_w[..., 3:], body_link_vel_w[..., 3:])


@pytest.mark.parametrize("num_cubes", [2])
@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("state_location", ["com", "link", "root"])
def test_write_root_state(num_cubes, device, state_location):
    """Test the root-state setters in the link, center-of-mass, and root reference frames."""
    with _ovphysx_sim_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        cube_object, env_pos = generate_cubes_scene(num_cubes=num_cubes, height=0.0, device=device)
        env_idx = torch.arange(num_cubes, dtype=torch.int32)
        sim.reset()
        assert cube_object.is_initialized
        offset = torch.tensor([0.1, 0.0, 0.0], device=device).repeat(num_cubes, 1)
        com = cube_object.data.body_com_pose_b.torch.clone()
        com[..., :3] = offset.unsqueeze(1)
        cube_object.set_coms_index(
            coms=wp.from_torch(com.contiguous(), dtype=wp.transformf),
            env_ids=wp.from_torch(env_idx, dtype=wp.int32),
        )
        torch.testing.assert_close(cube_object.data.body_com_pose_b.torch, com)
        rand_state = torch.rand(num_cubes, 13, device=device)
        rand_state[..., :3] += env_pos
        rand_state[..., 3:7] = torch.nn.functional.normalize(rand_state[..., 3:7], dim=-1)
        env_idx = env_idx.to(device)
        for i in range(10):
            sim.step()
            cube_object.update(sim.cfg.dt)
            write_env_ids = None if i % 2 == 0 else env_idx
            if state_location == "com":
                cube_object.write_root_com_pose_to_sim_index(root_pose=rand_state[..., :7], env_ids=write_env_ids)
                cube_object.write_root_com_velocity_to_sim_index(
                    root_velocity=rand_state[..., 7:], env_ids=write_env_ids
                )
            elif state_location == "link":
                cube_object.write_root_link_pose_to_sim_index(root_pose=rand_state[..., :7], env_ids=write_env_ids)
                cube_object.write_root_link_velocity_to_sim_index(
                    root_velocity=rand_state[..., 7:], env_ids=write_env_ids
                )
            else:
                cube_object.write_root_pose_to_sim_index(root_pose=rand_state[..., :7], env_ids=write_env_ids)
                cube_object.write_root_velocity_to_sim_index(root_velocity=rand_state[..., 7:], env_ids=write_env_ids)
            root_link_pose_w = cube_object.data.root_link_pose_w.torch
            root_link_vel_w = cube_object.data.root_link_vel_w.torch
            root_com_pose_w = cube_object.data.root_com_pose_w.torch
            root_com_vel_w = cube_object.data.root_com_vel_w.torch
            body_com_pose_b = cube_object.data.body_com_pose_b.torch
            if state_location == "com":
                torch.testing.assert_close(rand_state[..., :7], root_com_pose_w)
                torch.testing.assert_close(rand_state[..., 7:], root_com_vel_w)
                expected_root_link_pos, expected_root_link_quat = combine_frame_transforms(
                    root_com_pose_w[:, :3],
                    root_com_pose_w[:, 3:],
                    quat_rotate(quat_inv(body_com_pose_b[:, 0, 3:7]), -body_com_pose_b[:, 0, :3]),
                    quat_inv(body_com_pose_b[:, 0, 3:7]),
                )
                expected_root_link_pose = torch.cat((expected_root_link_pos, expected_root_link_quat), dim=1)
                torch.testing.assert_close(expected_root_link_pose, root_link_pose_w)
            else:
                torch.testing.assert_close(rand_state[..., :7], root_link_pose_w)
                written_vel = root_link_vel_w if state_location == "link" else root_com_vel_w
                torch.testing.assert_close(rand_state[..., 7:], written_vel)
                expected_com_pos, expected_com_quat = combine_frame_transforms(
                    root_link_pose_w[:, :3],
                    root_link_pose_w[:, 3:],
                    body_com_pose_b[:, 0, :3],
                    body_com_pose_b[:, 0, 3:7],
                )
                expected_com_pose = torch.cat((expected_com_pos, expected_com_quat), dim=1)
                torch.testing.assert_close(expected_com_pose, root_com_pose_w)
            torch.testing.assert_close(root_com_vel_w[:, 3:], root_link_vel_w[:, 3:])
