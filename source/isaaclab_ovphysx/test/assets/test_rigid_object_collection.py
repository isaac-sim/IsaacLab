# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none


"""Real-backend tests for the OVPhysX RigidObjectCollection.

Run via ``./scripts/run_ovphysx.sh -m pytest`` (kitless, no ``AppLauncher``).

The OVPhysX runtime binds device mode (CPU vs GPU) at the C++ layer on the
first ``ovphysx.PhysX(device=...)`` construction and cannot swap it without a
process restart.  Full coverage therefore requires two separate pytest
invocations -- once with ``-k 'cpu'`` and once with ``-k 'cuda:0'``.  The
``_ovphysx_skip_other_device`` autouse fixture below preempts the manager's
:exc:`RuntimeError` by ``pytest.skip``-ing on the unlocked device so
single-device runs finish cleanly.
"""

from __future__ import annotations

import sys

import pytest
import torch
import warp as wp

# The OVPhysX runtime wheel is optional. Skip gracefully when it is not installed;
# CI jobs that need OVPhysX coverage install it explicitly.
pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

from isaaclab_ovphysx.assets import RigidObjectCollection  # noqa: E402
from isaaclab_ovphysx.physics import OvPhysxCfg  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, build_simulation_context  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    combine_frame_transforms,
    default_orientation,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
    quat_rotate,
    random_orientation,
    subtract_frame_transforms,
)

wp.init()

pytestmark = pytest.mark.device_split


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
        # Test does not parametrize on device.
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


# ---------------------------------------------------------------------------
# Material-property gap (xfail reason shared by the test below)
# ---------------------------------------------------------------------------

_MATERIAL_GAP_REASON = (
    "Requires RIGID_BODY_MATERIAL TensorType (or a view-helper) on the ovphysx "
    "wheel side.  RigidObjectCollection.root_view is a per-tensor-type bindings dict on "
    "OVPhysX, so root_view.get_material_properties() / set_material_properties() "
    "are not available.  See "
    "docs/superpowers/specs/2026-04-28-ovphysx-wheel-gaps-for-marco.md."
)


def generate_cubes_scene(
    num_envs: int = 1,
    num_cubes: int = 1,
    height=1.0,
    has_api: bool = True,
    kinematic_enabled: bool = False,
    device: str = "cuda:0",
) -> tuple[RigidObjectCollection, torch.Tensor]:
    """Generate a scene with the provided number of cubes.

    Args:
        num_envs: Number of envs to generate.
        num_cubes: Number of cubes to generate.
        height: Height of the cubes.
        has_api: Whether the cubes have a rigid body API on them.
        kinematic_enabled: Whether the cubes are kinematic.
        device: Device to use for the simulation.

    Returns:
        A tuple containing the rigid object representing the cubes and the origins of the cubes.

    """
    origins = torch.tensor([(i * 3.0, 0, height) for i in range(num_envs)]).to(device)
    # Create Top-level Xforms, one for each cube
    for i, origin in enumerate(origins):
        sim_utils.create_prim(f"/World/Table_{i}", "Xform", translation=origin)

    # Resolve spawn configuration
    if has_api:
        spawn_cfg = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=kinematic_enabled),
        )
    else:
        # since no rigid body properties defined, this is just a static collider
        spawn_cfg = sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )

    # create the rigid object configs.  OVPhysX matches prim paths via fnmatch globs (not regex),
    # so use ``Table_*`` rather than the PhysX ``Table_.*`` form.
    cube_config_dict = {}
    for i in range(num_cubes):
        cube_object_cfg = RigidObjectCfg(
            prim_path=f"/World/Table_*/Object_{i}",
            spawn=spawn_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 3 * i, height)),
        )
        cube_config_dict[f"cube_{i}"] = cube_object_cfg
    # create the rigid object collection
    cube_object_collection_cfg = RigidObjectCollectionCfg(rigid_objects=cube_config_dict)
    cube_object_colection = RigidObjectCollection(cfg=cube_object_collection_cfg)

    return cube_object_colection, origins


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("num_cubes", [1, 3])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_initialization(num_envs, num_cubes, device):
    """Test initialization for prim with rigid body API at the provided prim path."""
    with _ovphysx_sim_context(device=device, auto_add_lighting=True) as sim:
        object_collection, _ = generate_cubes_scene(num_envs=num_envs, num_cubes=num_cubes, device=device)

        # Check that the framework doesn't hold excessive strong references.
        assert sys.getrefcount(object_collection) < 10

        # Play sim
        sim.reset()

        # Check if object is initialized
        assert object_collection.is_initialized
        assert len(object_collection.body_names) == num_cubes

        # Check buffers that exist and have correct shapes
        assert object_collection.data.body_link_pos_w.torch.shape == (num_envs, num_cubes, 3)
        assert object_collection.data.body_link_quat_w.torch.shape == (num_envs, num_cubes, 4)
        assert object_collection.data.body_mass.torch.shape == (num_envs, num_cubes)
        assert object_collection.data.body_inertia.torch.shape == (num_envs, num_cubes, 9)

        # Simulate physics
        for _ in range(2):
            sim.step()
            object_collection.update(sim.cfg.dt)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_id_conversion(device):
    """Test environment and object index conversion to physics view indices."""
    with _ovphysx_sim_context(device=device, auto_add_lighting=True) as sim:
        object_collection, _ = generate_cubes_scene(num_envs=2, num_cubes=3, device=device)

        # Play sim
        sim.reset()

        expected = [
            torch.tensor([4, 5], device=device, dtype=torch.int32),
            torch.tensor([4], device=device, dtype=torch.int32),
            torch.tensor([0, 2, 4], device=device, dtype=torch.int32),
            torch.tensor([1, 3, 5], device=device, dtype=torch.int32),
        ]

        torch_all_env_indices = wp.to_torch(object_collection._ALL_ENV_INDICES)
        torch_all_body_indices = wp.to_torch(object_collection._ALL_BODY_INDICES)

        view_ids = object_collection._env_body_ids_to_view_ids(
            torch_all_env_indices, torch_all_body_indices[None, 2], device=device
        )
        assert (wp.to_torch(view_ids) == expected[0]).all()
        view_ids = object_collection._env_body_ids_to_view_ids(
            torch_all_env_indices[None, 0], torch_all_body_indices[None, 2], device=device
        )
        assert (wp.to_torch(view_ids) == expected[1]).all()
        view_ids = object_collection._env_body_ids_to_view_ids(
            torch_all_env_indices[None, 0], torch_all_body_indices, device=device
        )
        assert (wp.to_torch(view_ids) == expected[2]).all()
        view_ids = object_collection._env_body_ids_to_view_ids(
            torch_all_env_indices[None, 1], torch_all_body_indices, device=device
        )
        assert (wp.to_torch(view_ids) == expected[3]).all()


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("num_cubes", [1, 3])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_initialization_with_kinematic_enabled(num_envs, num_cubes, device):
    """Test that initialization for prim with kinematic flag enabled."""
    with _ovphysx_sim_context(device=device, auto_add_lighting=True) as sim:
        object_collection, origins = generate_cubes_scene(
            num_envs=num_envs, num_cubes=num_cubes, kinematic_enabled=True, device=device
        )

        # Check that the framework doesn't hold excessive strong references.
        assert sys.getrefcount(object_collection) < 10

        # Play sim
        sim.reset()

        # Check if object is initialized
        assert object_collection.is_initialized
        assert len(object_collection.body_names) == num_cubes

        # Check buffers that exist and have correct shapes
        assert object_collection.data.body_link_pos_w.torch.shape == (num_envs, num_cubes, 3)
        assert object_collection.data.body_link_quat_w.torch.shape == (num_envs, num_cubes, 4)

        # Simulate physics
        for _ in range(2):
            sim.step()
            object_collection.update(sim.cfg.dt)
            # check that the object is kinematic
            default_body_pose = object_collection.data.default_body_pose.torch.clone()
            default_body_vel = object_collection.data.default_body_vel.torch.clone()
            default_body_pose[..., :3] += origins.unsqueeze(1)
            torch.testing.assert_close(object_collection.data.body_link_pose_w.torch, default_body_pose)
            torch.testing.assert_close(object_collection.data.body_link_vel_w.torch, default_body_vel)


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_initialization_with_no_rigid_body(num_cubes, device):
    """Test that initialization fails when no rigid body is found at the provided prim path."""
    with _ovphysx_sim_context(device=device, auto_add_lighting=True) as sim:
        object_collection, _ = generate_cubes_scene(num_cubes=num_cubes, has_api=False, device=device)

        # Check that the framework doesn't hold excessive strong references.
        assert sys.getrefcount(object_collection) < 10

        # Play sim
        with pytest.raises(RuntimeError):
            sim.reset()


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_external_force_buffer(device):
    """Test if external force buffer correctly updates in the force value is zero case."""
    num_envs = 2
    num_cubes = 1
    with _ovphysx_sim_context(device=device, auto_add_lighting=True) as sim:
        object_collection, origins = generate_cubes_scene(num_envs=num_envs, num_cubes=num_cubes, device=device)
        sim.reset()

        # find objects to apply the force
        object_ids, object_names = object_collection.find_bodies(".*")
        # reset object
        object_collection.reset()

        # perform simulation
        for step in range(5):
            # initiate force tensor
            external_wrench_b = torch.zeros(object_collection.num_instances, len(object_ids), 6, device=sim.device)

            # decide if zero or non-zero force
            if step == 0 or step == 3:
                force = 1.0
            else:
                force = 0.0

            # apply force to the object
            external_wrench_b[:, :, 0] = force
            external_wrench_b[:, :, 3] = force

            object_collection.permanent_wrench_composer.set_forces_and_torques_index(
                forces=external_wrench_b[..., :3],
                torques=external_wrench_b[..., 3:],
                body_ids=object_ids,
                env_ids=None,
            )

            # check if the object collection's force and torque buffers are correctly updated
            for i in range(num_envs):
                assert object_collection._permanent_wrench_composer.composed_force.torch[i, 0, 0].item() == force
                assert object_collection._permanent_wrench_composer.composed_torque.torch[i, 0, 0].item() == force

            object_collection.instantaneous_wrench_composer.add_forces_and_torques_index(
                body_ids=object_ids,
                forces=external_wrench_b[..., :3],
                torques=external_wrench_b[..., 3:],
            )

            # apply action to the object collection
            object_collection.write_data_to_sim()
            sim.step()
            object_collection.update(sim.cfg.dt)


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("num_cubes", [1, 4])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_external_force_on_single_body(num_envs, num_cubes, device):
    """Test application of external force on the base of the object."""
    with _ovphysx_sim_context(device=device, auto_add_lighting=True) as sim:
        object_collection, origins = generate_cubes_scene(num_envs=num_envs, num_cubes=num_cubes, device=device)
        sim.reset()

        # find objects to apply the force
        object_ids, object_names = object_collection.find_bodies(".*")

        # Sample a force equal to the weight of the object
        external_wrench_b = torch.zeros(object_collection.num_instances, len(object_ids), 6, device=sim.device)
        # Every 2nd cube should have a force applied to it
        external_wrench_b[:, 0::2, 2] = 9.81 * object_collection.data.body_mass.torch[:, 0::2]

        for i in range(5):
            # reset object state
            body_pose = object_collection.data.default_body_pose.torch.clone()
            body_vel = object_collection.data.default_body_vel.torch.clone()
            # need to shift the position of the cubes otherwise they will be on top of each other
            body_pose[..., :2] += origins.unsqueeze(1)[..., :2]
            object_collection.write_body_link_pose_to_sim_index(body_poses=body_pose)
            object_collection.write_body_com_velocity_to_sim_index(body_velocities=body_vel)
            # reset object
            object_collection.reset()

            is_global = False
            if i % 2 == 0:
                positions = object_collection.data.body_link_pos_w.torch[:, object_ids, :3]
                is_global = True
            else:
                positions = None

            # apply force
            object_collection.permanent_wrench_composer.set_forces_and_torques_index(
                forces=external_wrench_b[..., :3],
                torques=external_wrench_b[..., 3:],
                positions=positions,
                body_ids=object_ids,
                env_ids=None,
                is_global=is_global,
            )
            for _ in range(10):
                # write data to sim
                object_collection.write_data_to_sim()
                # step sim
                sim.step()
                # update object collection
                object_collection.update(sim.cfg.dt)

            # First object should still be at the same Z position (1.0)
            torch.testing.assert_close(
                object_collection.data.body_link_pos_w.torch[:, 0::2, 2],
                torch.ones_like(object_collection.data.body_link_pos_w.torch[:, 0::2, 2]),
            )
            # Second object should have fallen, so it's Z height should be less than initial height of 1.0
            assert torch.all(object_collection.data.body_link_pos_w.torch[:, 1::2, 2] < 1.0)


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("num_cubes", [1, 4])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_external_force_on_single_body_at_position(num_envs, num_cubes, device):
    """Test application of external force on the base of the object at a specific position.

    In this test, we apply a force equal to the weight of an object on the base of
    one of the objects at 1m in the Y direction, we check that the object rotates around it's X axis.
    For the other object, we do not apply any force and check that it falls down.
    """
    with _ovphysx_sim_context(device=device, auto_add_lighting=True) as sim:
        object_collection, origins = generate_cubes_scene(num_envs=num_envs, num_cubes=num_cubes, device=device)
        sim.reset()

        # find objects to apply the force
        object_ids, object_names = object_collection.find_bodies(".*")

        # Sample a force equal to the weight of the object
        external_wrench_b = torch.zeros(object_collection.num_instances, len(object_ids), 6, device=sim.device)
        external_wrench_positions_b = torch.zeros(
            object_collection.num_instances, len(object_ids), 3, device=sim.device
        )
        # Every 2nd cube should have a force applied to it
        external_wrench_b[:, 0::2, 2] = 500.0
        external_wrench_positions_b[:, 0::2, 1] = 1.0

        # Desired force and torque
        for i in range(5):
            # reset object state
            body_pose = object_collection.data.default_body_pose.torch.clone()
            body_vel = object_collection.data.default_body_vel.torch.clone()
            # need to shift the position of the cubes otherwise they will be on top of each other
            body_pose[..., :2] += origins.unsqueeze(1)[..., :2]
            object_collection.write_body_link_pose_to_sim_index(body_poses=body_pose)
            object_collection.write_body_com_velocity_to_sim_index(body_velocities=body_vel)
            # reset object
            object_collection.reset()

            is_global = False
            if i % 2 == 0:
                body_com_pos_w = object_collection.data.body_link_pos_w.torch[:, object_ids, :3]
                external_wrench_positions_b[..., 0] = 0.0
                external_wrench_positions_b[..., 1] = 1.0
                external_wrench_positions_b[..., 2] = 0.0
                external_wrench_positions_b += body_com_pos_w
                is_global = True
            else:
                external_wrench_positions_b[..., 0] = 0.0
                external_wrench_positions_b[..., 1] = 1.0
                external_wrench_positions_b[..., 2] = 0.0

            # apply force
            object_collection.permanent_wrench_composer.set_forces_and_torques_index(
                forces=external_wrench_b[..., :3],
                torques=external_wrench_b[..., 3:],
                positions=external_wrench_positions_b,
                body_ids=object_ids,
                env_ids=None,
                is_global=is_global,
            )
            object_collection.permanent_wrench_composer.add_forces_and_torques_index(
                forces=external_wrench_b[..., :3],
                torques=external_wrench_b[..., 3:],
                positions=external_wrench_positions_b,
                body_ids=object_ids,
                is_global=is_global,
            )

            for _ in range(10):
                # write data to sim
                object_collection.write_data_to_sim()
                # step sim
                sim.step()
                # update object collection
                object_collection.update(sim.cfg.dt)

            # First object should be rotating around it's X axis
            assert torch.all(object_collection.data.body_com_ang_vel_b.torch[:, 0::2, 0] > 0.1)
            # Second object should have fallen, so it's Z height should be less than initial height of 1.0
            assert torch.all(object_collection.data.body_link_pos_w.torch[:, 1::2, 2] < 1.0)


@pytest.mark.parametrize("num_envs", [1, 3])
@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("gravity_enabled", [False])
@pytest.mark.isaacsim_ci
def test_set_object_state(num_envs, num_cubes, device, gravity_enabled):
    """Test setting the state of the object.

    .. note::
        Turn off gravity for this test as we don't want any external forces acting on the object
        to ensure state remains static
    """
    with _ovphysx_sim_context(device=device, gravity_enabled=gravity_enabled, auto_add_lighting=True) as sim:
        object_collection, origins = generate_cubes_scene(num_envs=num_envs, num_cubes=num_cubes, device=device)
        sim.reset()

        state_types = ["body_link_pos_w", "body_link_quat_w", "body_com_lin_vel_w", "body_com_ang_vel_w"]

        # Set each state type individually as they are dependent on each other
        for state_type_to_randomize in state_types:
            state_dict = {
                "body_link_pos_w": torch.zeros_like(object_collection.data.body_link_pos_w.torch, device=sim.device),
                "body_link_quat_w": default_orientation(num=num_cubes * num_envs, device=sim.device).view(
                    num_envs, num_cubes, 4
                ),
                "body_com_lin_vel_w": torch.zeros_like(
                    object_collection.data.body_com_lin_vel_w.torch, device=sim.device
                ),
                "body_com_ang_vel_w": torch.zeros_like(
                    object_collection.data.body_com_ang_vel_w.torch, device=sim.device
                ),
            }

            for _ in range(5):
                # reset object
                object_collection.reset()

                # Set random state
                if state_type_to_randomize == "body_link_quat_w":
                    state_dict[state_type_to_randomize] = random_orientation(
                        num=num_cubes * num_envs, device=sim.device
                    ).view(num_envs, num_cubes, 4)
                else:
                    state_dict[state_type_to_randomize] = torch.randn(num_envs, num_cubes, 3, device=sim.device)
                    # make sure objects do not overlap
                    if state_type_to_randomize == "body_link_pos_w":
                        state_dict[state_type_to_randomize][..., :2] += origins.unsqueeze(1)[..., :2]

                # perform simulation
                for _ in range(5):
                    body_pose = torch.cat(
                        [state_dict["body_link_pos_w"], state_dict["body_link_quat_w"]],
                        dim=-1,
                    )
                    body_vel = torch.cat(
                        [state_dict["body_com_lin_vel_w"], state_dict["body_com_ang_vel_w"]],
                        dim=-1,
                    )
                    # reset object state
                    object_collection.write_body_link_pose_to_sim_index(body_poses=body_pose)
                    object_collection.write_body_com_velocity_to_sim_index(body_velocities=body_vel)
                    sim.step()

                    # assert that set object quantities are equal to the ones set in the state_dict
                    for key, expected_value in state_dict.items():
                        value = getattr(object_collection.data, key).torch
                        torch.testing.assert_close(value, expected_value, rtol=1e-5, atol=1e-5)

                    object_collection.update(sim.cfg.dt)


@pytest.mark.parametrize("num_envs", [1, 4])
@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("with_offset", [True, False])
@pytest.mark.parametrize("gravity_enabled", [False])
@pytest.mark.isaacsim_ci
def test_object_state_properties(num_envs, num_cubes, device, with_offset, gravity_enabled):
    """Test the object_com_state_w and object_link_state_w properties."""
    with _ovphysx_sim_context(device=device, gravity_enabled=gravity_enabled, auto_add_lighting=True) as sim:
        cube_object, env_pos = generate_cubes_scene(num_envs=num_envs, num_cubes=num_cubes, height=0.0, device=device)
        env_ids = torch.tensor([x for x in range(num_envs)], dtype=torch.int32)

        sim.reset()

        # check if cube_object is initialized
        assert cube_object.is_initialized

        # change center of mass offset from link frame
        offset = (
            torch.tensor([0.1, 0.0, 0.0], device=device).repeat(num_envs, num_cubes, 1)
            if with_offset
            else torch.tensor([0.0, 0.0, 0.0], device=device).repeat(num_envs, num_cubes, 1)
        )

        # Read current COMs, mutate the translation, write back via the OVPhysX
        # ``set_coms_index`` setter (PhysX uses ``root_view.set_coms`` + reshape helpers
        # for the same operation; OVPhysX wraps the wheel RIGID_BODY_COM_POSE write in
        # :meth:`set_coms_index`).
        com = cube_object.data.body_com_pose_b.torch.clone()  # shape (num_envs, num_cubes, 7)
        com[..., :3] = offset.to(com.device)
        cube_object.set_coms_index(
            coms=wp.from_torch(com.contiguous(), dtype=wp.transformf),
            env_ids=wp.from_torch(env_ids, dtype=wp.int32),
        )

        # check center of mass has been set
        torch.testing.assert_close(cube_object.data.body_com_pose_b.torch, com)

        # random z spin velocity
        spin_twist = torch.zeros(6, device=device)
        spin_twist[5] = torch.randn(1, device=device)

        # initial spawn point
        init_com = cube_object.data.body_com_pose_w.torch[..., :3]

        for i in range(10):
            # spin the object around Z axis (com)
            cube_object.write_body_com_velocity_to_sim_index(body_velocities=spin_twist.repeat(num_envs, num_cubes, 1))
            sim.step()
            cube_object.update(sim.cfg.dt)

            # get state properties
            object_link_pose_w = cube_object.data.body_link_pose_w.torch
            object_link_vel_w = cube_object.data.body_link_vel_w.torch
            object_com_pose_w = cube_object.data.body_com_pose_w.torch
            object_com_vel_w = cube_object.data.body_com_vel_w.torch

            # if offset is [0,0,0] all object_state_%_w will match and all body_%_w will match
            if not with_offset:
                torch.testing.assert_close(object_link_pose_w, object_com_pose_w)
                torch.testing.assert_close(object_com_vel_w, object_link_vel_w)
            else:
                # cubes are spinning around center of mass
                # position will not match
                # center of mass position will be constant (i.e. spinning around com)
                torch.testing.assert_close(init_com, object_com_pose_w[..., :3])

                # link position will be moving but should stay constant away from center of mass
                object_link_state_pos_rel_com = quat_apply_inverse(
                    object_link_pose_w[..., 3:],
                    object_link_pose_w[..., :3] - object_com_pose_w[..., :3],
                )

                torch.testing.assert_close(-offset, object_link_state_pos_rel_com)

                # orientation of com will be a constant rotation from link orientation
                com_quat_b = cube_object.data.body_com_quat_b.torch
                com_quat_w = quat_mul(object_link_pose_w[..., 3:], com_quat_b)
                torch.testing.assert_close(com_quat_w, object_com_pose_w[..., 3:])

                # orientation of link will match object state will always match
                torch.testing.assert_close(object_link_pose_w[..., 3:], object_link_pose_w[..., 3:])

                # lin_vel will not match
                # center of mass vel will be constant (i.e. spinning around com)
                torch.testing.assert_close(
                    torch.zeros_like(object_com_vel_w[..., :3]),
                    object_com_vel_w[..., :3],
                )

                # link frame will be moving, and should be equal to input angular velocity cross offset
                lin_vel_rel_object_gt = quat_apply_inverse(object_link_pose_w[..., 3:], object_link_vel_w[..., :3])
                lin_vel_rel_gt = torch.linalg.cross(spin_twist.repeat(num_envs, num_cubes, 1)[..., 3:], -offset)
                torch.testing.assert_close(lin_vel_rel_gt, lin_vel_rel_object_gt, atol=1e-4, rtol=1e-3)

                # ang_vel will always match
                torch.testing.assert_close(object_com_vel_w[..., 3:], object_com_vel_w[..., 3:])
                torch.testing.assert_close(object_com_vel_w[..., 3:], object_link_vel_w[..., 3:])


@pytest.mark.parametrize("num_envs", [1, 3])
@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("with_offset", [True, False])
@pytest.mark.parametrize("state_location", ["com", "link"])
@pytest.mark.parametrize("gravity_enabled", [False])
@pytest.mark.isaacsim_ci
def test_write_object_state(num_envs, num_cubes, device, with_offset, state_location, gravity_enabled):
    """Test the setters for object_state using both the link frame and center of mass as reference frame."""
    with _ovphysx_sim_context(device=device, gravity_enabled=gravity_enabled, auto_add_lighting=True) as sim:
        # Create a scene with random cubes
        cube_object, env_pos = generate_cubes_scene(num_envs=num_envs, num_cubes=num_cubes, height=0.0, device=device)
        env_ids = torch.tensor([x for x in range(num_envs)], dtype=torch.int32)
        object_ids = torch.tensor([x for x in range(num_cubes)], dtype=torch.int32)

        sim.reset()

        # Check if cube_object is initialized
        assert cube_object.is_initialized

        # change center of mass offset from link frame
        offset = (
            torch.tensor([0.1, 0.0, 0.0], device=device).repeat(num_envs, num_cubes, 1)
            if with_offset
            else torch.tensor([0.0, 0.0, 0.0], device=device).repeat(num_envs, num_cubes, 1)
        )

        com = cube_object.data.body_com_pose_b.torch.clone()  # shape (num_envs, num_cubes, 7)
        com[..., :3] = offset.to(com.device)
        cube_object.set_coms_index(
            coms=wp.from_torch(com.contiguous(), dtype=wp.transformf),
            env_ids=wp.from_torch(env_ids, dtype=wp.int32),
        )
        # check center of mass has been set
        torch.testing.assert_close(cube_object.data.body_com_pose_b.torch, com)

        rand_state = torch.zeros(num_envs, num_cubes, 13, device=device)
        rand_state[..., :7] = cube_object.data.default_body_pose.torch
        rand_state[..., :3] += cube_object.data.body_link_pos_w.torch
        # make quaternion a unit vector
        rand_state[..., 3:7] = torch.nn.functional.normalize(rand_state[..., 3:7], dim=-1)

        env_ids = env_ids.to(device)
        object_ids = object_ids.to(device)
        for i in range(10):
            sim.step()
            cube_object.update(sim.cfg.dt)

            if state_location == "com":
                if i % 2 == 0:
                    cube_object.write_body_com_pose_to_sim_index(body_poses=rand_state[..., :7])
                    cube_object.write_body_com_velocity_to_sim_index(body_velocities=rand_state[..., 7:])
                else:
                    cube_object.write_body_com_pose_to_sim_index(
                        body_poses=rand_state[..., :7], env_ids=env_ids, body_ids=object_ids
                    )
                    cube_object.write_body_com_velocity_to_sim_index(
                        body_velocities=rand_state[..., 7:], env_ids=env_ids, body_ids=object_ids
                    )
            elif state_location == "link":
                if i % 2 == 0:
                    cube_object.write_body_link_pose_to_sim_index(body_poses=rand_state[..., :7])
                    cube_object.write_body_link_velocity_to_sim_index(body_velocities=rand_state[..., 7:])
                else:
                    cube_object.write_body_link_pose_to_sim_index(
                        body_poses=rand_state[..., :7], env_ids=env_ids, body_ids=object_ids
                    )
                    cube_object.write_body_link_velocity_to_sim_index(
                        body_velocities=rand_state[..., 7:], env_ids=env_ids, body_ids=object_ids
                    )

            if state_location == "com":
                torch.testing.assert_close(rand_state[..., :7], cube_object.data.body_com_pose_w.torch)
                torch.testing.assert_close(rand_state[..., 7:], cube_object.data.body_com_vel_w.torch)
            elif state_location == "link":
                torch.testing.assert_close(rand_state[..., :7], cube_object.data.body_link_pose_w.torch)
                torch.testing.assert_close(rand_state[..., 7:], cube_object.data.body_link_vel_w.torch)


@pytest.mark.parametrize("num_envs", [1, 3])
@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_reset_object_collection(num_envs, num_cubes, device):
    """Test resetting the state of the rigid object."""
    with _ovphysx_sim_context(device=device, auto_add_lighting=True) as sim:
        object_collection, _ = generate_cubes_scene(num_envs=num_envs, num_cubes=num_cubes, device=device)
        sim.reset()

        for i in range(5):
            sim.step()
            object_collection.update(sim.cfg.dt)

            # Move the object to a random position
            body_pose = object_collection.data.default_body_pose.torch.clone()
            body_pose[..., :3] = torch.randn(num_envs, num_cubes, 3, device=sim.device)
            # Random orientation
            body_pose[..., 3:7] = random_orientation(num=num_cubes, device=sim.device)
            object_collection.write_body_link_pose_to_sim_index(body_poses=body_pose)
            body_vel = object_collection.data.default_body_vel.torch.clone()
            object_collection.write_body_com_velocity_to_sim_index(body_velocities=body_vel)

            if i % 2 == 0:
                object_collection.reset()

                # Reset should zero external forces and torques
                assert not object_collection._instantaneous_wrench_composer.active
                assert not object_collection._permanent_wrench_composer.active
                assert torch.count_nonzero(object_collection._instantaneous_wrench_composer.composed_force.torch) == 0
                assert torch.count_nonzero(object_collection._instantaneous_wrench_composer.composed_torque.torch) == 0
                assert torch.count_nonzero(object_collection._permanent_wrench_composer.composed_force.torch) == 0
                assert torch.count_nonzero(object_collection._permanent_wrench_composer.composed_torque.torch) == 0


@pytest.mark.xfail(reason=_MATERIAL_GAP_REASON, strict=False)
@pytest.mark.parametrize("num_envs", [1, 3])
@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_set_material_properties(num_envs, num_cubes, device):
    """Test getting and setting material properties of rigid object."""
    raise NotImplementedError(_MATERIAL_GAP_REASON)


@pytest.mark.parametrize("num_envs", [1, 3])
@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("gravity_enabled", [True, False])
@pytest.mark.isaacsim_ci
def test_gravity_vec_w(num_envs, num_cubes, device, gravity_enabled):
    """Test that gravity vector direction is set correctly for the rigid object."""
    with _ovphysx_sim_context(device=device, gravity_enabled=gravity_enabled, auto_add_lighting=True) as sim:
        object_collection, _ = generate_cubes_scene(num_envs=num_envs, num_cubes=num_cubes, device=device)

        # Obtain gravity direction
        gravity_dir = (0.0, 0.0, -1.0) if gravity_enabled else (0.0, 0.0, 0.0)

        sim.reset()

        # Check if gravity vector is set correctly
        gravity_vec = object_collection.data.GRAVITY_VEC_W.torch
        assert gravity_vec[0, 0, 0] == gravity_dir[0]
        assert gravity_vec[0, 0, 1] == gravity_dir[1]
        assert gravity_vec[0, 0, 2] == gravity_dir[2]

        # Perform simulation
        for _ in range(2):
            sim.step()
            object_collection.update(sim.cfg.dt)

            # Expected gravity value is the acceleration of the body
            gravity = torch.zeros(num_envs, num_cubes, 6, device=device)
            if gravity_enabled:
                gravity[..., 2] = -9.81

            # Check the body accelerations are correct
            torch.testing.assert_close(object_collection.data.body_com_acc_w.torch, gravity)


@pytest.mark.parametrize("num_envs", [1, 3])
@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("with_offset", [True])
@pytest.mark.parametrize("state_location", ["com", "link", "root"])
@pytest.mark.parametrize("gravity_enabled", [False])
@pytest.mark.isaacsim_ci
def test_write_object_state_functions_data_consistency(
    num_envs, num_cubes, device, with_offset, state_location, gravity_enabled
):
    """Test the setters for object_state using both the link frame and center of mass as reference frame."""
    with _ovphysx_sim_context(device=device, gravity_enabled=gravity_enabled, auto_add_lighting=True) as sim:
        # Create a scene with random cubes
        cube_object, env_pos = generate_cubes_scene(num_envs=num_envs, num_cubes=num_cubes, height=0.0, device=device)
        env_ids = torch.tensor([x for x in range(num_envs)], dtype=torch.int32)
        object_ids = torch.tensor([x for x in range(num_cubes)], dtype=torch.int32)

        sim.reset()

        # Check if cube_object is initialized
        assert cube_object.is_initialized

        # change center of mass offset from link frame
        offset = (
            torch.tensor([0.1, 0.0, 0.0], device=device).repeat(num_envs, num_cubes, 1)
            if with_offset
            else torch.tensor([0.0, 0.0, 0.0], device=device).repeat(num_envs, num_cubes, 1)
        )

        com = cube_object.data.body_com_pose_b.torch.clone()  # shape (num_envs, num_cubes, 7)
        com[..., :3] = offset.to(com.device)
        cube_object.set_coms_index(
            coms=wp.from_torch(com.contiguous(), dtype=wp.transformf),
            env_ids=wp.from_torch(env_ids, dtype=wp.int32),
        )

        # check center of mass has been set
        torch.testing.assert_close(cube_object.data.body_com_pose_b.torch, com)

        rand_state = torch.rand(num_envs, num_cubes, 13, device=device)
        rand_state[..., :3] += cube_object.data.body_link_pos_w.torch
        # make quaternion a unit vector
        rand_state[..., 3:7] = torch.nn.functional.normalize(rand_state[..., 3:7], dim=-1)

        env_ids = env_ids.to(device)
        object_ids = object_ids.to(device)
        sim.step()
        cube_object.update(sim.cfg.dt)

        body_link_pose_w = cube_object.data.body_link_pose_w.torch
        body_com_pose_w = cube_object.data.body_com_pose_w.torch
        object_link_to_com_pos, object_link_to_com_quat = subtract_frame_transforms(
            body_link_pose_w[..., :3].view(-1, 3),
            body_link_pose_w[..., 3:7].view(-1, 4),
            body_com_pose_w[..., :3].view(-1, 3),
            body_com_pose_w[..., 3:7].view(-1, 4),
        )

        if state_location == "com":
            cube_object.write_body_com_pose_to_sim_index(
                body_poses=rand_state[..., :7], env_ids=env_ids, body_ids=object_ids
            )
            cube_object.write_body_com_velocity_to_sim_index(
                body_velocities=rand_state[..., 7:], env_ids=env_ids, body_ids=object_ids
            )
        elif state_location == "link":
            cube_object.write_body_link_pose_to_sim_index(
                body_poses=rand_state[..., :7], env_ids=env_ids, body_ids=object_ids
            )
            cube_object.write_body_link_velocity_to_sim_index(
                body_velocities=rand_state[..., 7:], env_ids=env_ids, body_ids=object_ids
            )
        elif state_location == "root":
            cube_object.write_body_link_pose_to_sim_index(
                body_poses=rand_state[..., :7], env_ids=env_ids, body_ids=object_ids
            )
            cube_object.write_body_com_velocity_to_sim_index(
                body_velocities=rand_state[..., 7:], env_ids=env_ids, body_ids=object_ids
            )

        if state_location == "com":
            com_pose_w = cube_object.data.body_com_pose_w.torch
            com_vel_w = cube_object.data.body_com_vel_w.torch
            expected_root_link_pos, expected_root_link_quat = combine_frame_transforms(
                com_pose_w[..., :3].view(-1, 3),
                com_pose_w[..., 3:].view(-1, 4),
                quat_rotate(quat_inv(object_link_to_com_quat), -object_link_to_com_pos),
                quat_inv(object_link_to_com_quat),
            )
            expected_object_link_pose = torch.cat((expected_root_link_pos, expected_root_link_quat), dim=1).view(
                num_envs, -1, 7
            )
            link_pose_w = cube_object.data.body_link_pose_w.torch
            link_vel_w = cube_object.data.body_link_vel_w.torch
            # test both root_pose and root_link successfully updated when root_com updates
            torch.testing.assert_close(expected_object_link_pose, link_pose_w)
            # skip lin_vel because it differs from link frame, this should be fine because we are only checking
            # if velocity update is triggered, which can be determined by comparing angular velocity
            torch.testing.assert_close(com_vel_w[..., 3:], link_vel_w[..., 3:])
            torch.testing.assert_close(expected_object_link_pose, link_pose_w)
            torch.testing.assert_close(com_vel_w[..., 3:], cube_object.data.body_com_vel_w.torch[..., 3:])
        elif state_location == "link":
            link_pose_w = cube_object.data.body_link_pose_w.torch
            link_vel_w = cube_object.data.body_link_vel_w.torch
            expected_com_pos, expected_com_quat = combine_frame_transforms(
                link_pose_w[..., :3].view(-1, 3),
                link_pose_w[..., 3:].view(-1, 4),
                object_link_to_com_pos,
                object_link_to_com_quat,
            )
            expected_object_com_pose = torch.cat((expected_com_pos, expected_com_quat), dim=1).view(num_envs, -1, 7)
            com_pose_w = cube_object.data.body_com_pose_w.torch
            com_vel_w = cube_object.data.body_com_vel_w.torch
            # test both root_pose and root_com successfully updated when root_link updates
            torch.testing.assert_close(expected_object_com_pose, com_pose_w)
            # skip lin_vel because it differs from link frame, this should be fine because we are only checking
            # if velocity update is triggered, which can be determined by comparing angular velocity
            torch.testing.assert_close(link_vel_w[..., 3:], com_vel_w[..., 3:])
            torch.testing.assert_close(link_pose_w, cube_object.data.body_link_pose_w.torch)
            torch.testing.assert_close(link_vel_w[..., 3:], cube_object.data.body_com_vel_w.torch[..., 3:])
        elif state_location == "root":
            body_link_pose_w = cube_object.data.body_link_pose_w.torch
            body_com_vel_w = cube_object.data.body_com_vel_w.torch
            expected_object_com_pos, expected_object_com_quat = combine_frame_transforms(
                body_link_pose_w[..., :3].view(-1, 3),
                body_link_pose_w[..., 3:].view(-1, 4),
                object_link_to_com_pos,
                object_link_to_com_quat,
            )
            expected_object_com_pose = torch.cat((expected_object_com_pos, expected_object_com_quat), dim=1).view(
                num_envs, -1, 7
            )
            com_pose_w = cube_object.data.body_com_pose_w.torch
            com_vel_w = cube_object.data.body_com_vel_w.torch
            link_pose_w = cube_object.data.body_link_pose_w.torch
            link_vel_w = cube_object.data.body_link_vel_w.torch
            # test both root_com and root_link successfully updated when root_pose updates
            torch.testing.assert_close(expected_object_com_pose, com_pose_w)
            torch.testing.assert_close(body_com_vel_w, com_vel_w)
            torch.testing.assert_close(body_link_pose_w, link_pose_w)
            torch.testing.assert_close(body_com_vel_w[..., 3:], link_vel_w[..., 3:])
