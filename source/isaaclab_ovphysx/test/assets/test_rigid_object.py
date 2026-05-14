# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none


"""Real-backend tests for the OVPhysX RigidObject.

Run via ``./scripts/run_ovphysx.sh -m pytest`` (kitless, no ``AppLauncher``).

``ovphysx<=0.3.7`` binds device mode (CPU vs GPU) at the C++ layer on the
first ``ovphysx.PhysX(device=...)`` construction and cannot swap it without a
process restart.  Full coverage therefore requires two separate pytest
invocations -- once with ``-k 'cpu'`` and once with ``-k 'cuda:0'``.  The
``_ovphysx_skip_other_device`` autouse fixture below preempts the manager's
:exc:`RuntimeError` by ``pytest.skip``-ing on the unlocked device so
single-device runs finish cleanly.
"""

from __future__ import annotations

import logging
import sys
from typing import Literal
from unittest.mock import MagicMock

import pytest
import torch
import warp as wp
from flaky import flaky

# The CI isaaclab_ov* pattern unintentionally collects isaaclab_ovphysx tests,
# but the ovphysx wheel is not installed in that environment. Skip gracefully
# so the isaaclab_ov CI pipeline is not blocked by an unrelated dependency.
pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

from isaaclab_ovphysx.assets import RigidObject  # noqa: E402
from isaaclab_ovphysx.physics import OvPhysxCfg, OvPhysxManager  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObjectCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, build_simulation_context  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR  # noqa: E402
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
        # Test does not parametrize on device (e.g. test_warmup_attach_stage_not_called_for_cpu).
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


def generate_cubes_scene(
    num_cubes: int = 1,
    height=1.0,
    api: Literal["none", "rigid_body", "articulation_root"] = "rigid_body",
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
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
    elif api == "rigid_body":
        spawn_cfg = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=kinematic_enabled),
        )
    elif api == "articulation_root":
        spawn_cfg = sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Tests/RigidObject/Cube/dex_cube_instanceable_with_articulation_root.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=kinematic_enabled),
        )
    else:
        raise ValueError(f"Unknown api: {api}")

    # Create rigid object.  OVPhysX matches prim paths via fnmatch globs (not regex),
    # so use ``Table_*`` rather than the PhysX ``Table_.*`` form.
    cube_object_cfg = RigidObjectCfg(
        prim_path="/World/Table_*/Object",
        spawn=spawn_cfg,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, height)),
    )
    cube_object = RigidObject(cfg=cube_object_cfg)

    return cube_object, origins


# ---------------------------------------------------------------------------
# Material-property gap (xfail reason shared by 5 tests below)
# ---------------------------------------------------------------------------

_MATERIAL_GAP_REASON = (
    "Requires RIGID_BODY_MATERIAL TensorType (or a view-helper) on the ovphysx "
    "wheel side.  RigidObject.root_view is a per-tensor-type bindings dict on "
    "OVPhysX, so root_view.get_material_properties() / set_material_properties() "
    "are not available.  See "
    "docs/superpowers/specs/2026-04-28-ovphysx-wheel-gaps-for-marco.md."
)


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_initialization(num_cubes, device):
    """Test initialization for prim with rigid body API at the provided prim path."""
    with _ovphysx_sim_context(device=device, auto_add_lighting=True) as sim:
        # Generate cubes scene
        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, device=device)

        # Check that the framework doesn't hold excessive strong references.
        assert sys.getrefcount(cube_object) < 10

        # Play sim
        sim.reset()

        # Check if object is initialized
        assert cube_object.is_initialized
        assert len(cube_object.body_names) == 1

        # Check buffers that exists and have correct shapes
        assert cube_object.data.root_pos_w.torch.shape == (num_cubes, 3)
        assert cube_object.data.root_quat_w.torch.shape == (num_cubes, 4)
        assert cube_object.data.body_mass.torch.shape == (num_cubes, 1)
        assert cube_object.data.body_inertia.torch.shape == (num_cubes, 1, 9)

        # Simulate physics
        for _ in range(2):
            # perform rendering
            sim.step()
            # update object
            cube_object.update(sim.cfg.dt)


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_initialization_with_kinematic_enabled(num_cubes, device):
    """Test that initialization for prim with kinematic flag enabled."""
    with _ovphysx_sim_context(device=device, auto_add_lighting=True) as sim:
        # Generate cubes scene
        cube_object, origins = generate_cubes_scene(num_cubes=num_cubes, kinematic_enabled=True, device=device)

        # Check that the framework doesn't hold excessive strong references.
        assert sys.getrefcount(cube_object) < 10

        # Play sim
        sim.reset()

        # Check if object is initialized
        assert cube_object.is_initialized
        assert len(cube_object.body_names) == 1

        # Check buffers that exists and have correct shapes
        assert cube_object.data.root_pos_w.torch.shape == (num_cubes, 3)
        assert cube_object.data.root_quat_w.torch.shape == (num_cubes, 4)

        # Simulate physics
        for _ in range(2):
            # perform rendering
            sim.step()
            # update object
            cube_object.update(sim.cfg.dt)
            # check that the object is kinematic
            default_root_pose = cube_object.data.default_root_pose.torch.clone()
            default_root_vel = cube_object.data.default_root_vel.torch.clone()
            default_root_pose[:, :3] += origins
            torch.testing.assert_close(cube_object.data.root_link_pose_w.torch, default_root_pose)
            torch.testing.assert_close(cube_object.data.root_com_vel_w.torch, default_root_vel)


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_initialization_with_no_rigid_body(num_cubes, device):
    """Test that initialization fails when no rigid body is found at the provided prim path."""
    with _ovphysx_sim_context(device=device, auto_add_lighting=True) as sim:
        # Generate cubes scene
        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, api="none", device=device)

        # Check that the framework doesn't hold excessive strong references.
        assert sys.getrefcount(cube_object) < 10

        # Play sim
        with pytest.raises(RuntimeError):
            sim.reset()


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_initialization_with_articulation_root(num_cubes, device):
    """Test that initialization fails when an articulation root is found at the provided prim path."""
    with _ovphysx_sim_context(device=device, auto_add_lighting=True) as sim:
        # Generate cubes scene
        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, api="articulation_root", device=device)

        # Check that the framework doesn't hold excessive strong references.
        assert sys.getrefcount(cube_object) < 10

        # Play sim
        with pytest.raises(RuntimeError):
            sim.reset()


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_external_force_buffer(device):
    """Test if external force buffer correctly updates in the force value is zero case.

    In this test, we apply a non-zero force, then a zero force, then finally a non-zero force
    to an object. We check if the force buffer is properly updated at each step.
    """

    # Generate cubes scene
    with _ovphysx_sim_context(device=device, add_ground_plane=True, auto_add_lighting=True) as sim:
        cube_object, origins = generate_cubes_scene(num_cubes=1, device=device)

        # play the simulator
        sim.reset()

        # find bodies to apply the force
        body_ids, body_names = cube_object.find_bodies(".*")

        # reset object
        cube_object.reset()

        # perform simulation
        for step in range(5):
            # initiate force tensor
            external_wrench_b = torch.zeros(cube_object.num_instances, len(body_ids), 6, device=sim.device)

            if step == 0 or step == 3:
                # set a non-zero force
                force = 1
            else:
                # set a zero force
                force = 0

            # set force value
            external_wrench_b[:, :, 0] = force
            external_wrench_b[:, :, 3] = force

            # apply force
            cube_object.permanent_wrench_composer.set_forces_and_torques_index(
                forces=external_wrench_b[..., :3],
                torques=external_wrench_b[..., 3:],
                body_ids=body_ids,
            )

            # check if the cube's force and torque buffers are correctly updated
            for i in range(cube_object.num_instances):
                assert cube_object._permanent_wrench_composer.composed_force.torch[i, 0, 0].item() == force
                assert cube_object._permanent_wrench_composer.composed_torque.torch[i, 0, 0].item() == force

            # Check if the instantaneous wrench is correctly added to the permanent wrench
            cube_object.permanent_wrench_composer.add_forces_and_torques_index(
                forces=external_wrench_b[..., :3],
                torques=external_wrench_b[..., 3:],
                body_ids=body_ids,
            )

            # apply action to the object
            cube_object.write_data_to_sim()

            # perform step
            sim.step()

            # update buffers
            cube_object.update(sim.cfg.dt)


@pytest.mark.parametrize("num_cubes", [2, 4])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_external_force_on_single_body(num_cubes, device):
    """Test application of external force on the base of the object.

    In this test, we apply a force equal to the weight of an object on the base of
    one of the objects. We check that the object does not move. For the other object,
    we do not apply any force and check that it falls down.

    We validate that this works when we apply the force in the global frame and in the local frame.
    """
    # Generate cubes scene
    with _ovphysx_sim_context(device=device, add_ground_plane=True, auto_add_lighting=True) as sim:
        cube_object, origins = generate_cubes_scene(num_cubes=num_cubes, device=device)

        # Play the simulator
        sim.reset()

        # Find bodies to apply the force
        body_ids, body_names = cube_object.find_bodies(".*")

        # Sample a force equal to the weight of the object.  PhysX reads the mass
        # from ``root_view.get_masses()``; OVPhysX exposes the same value via
        # ``cube_object.data.body_mass`` (shape ``(N, 1)``).
        external_wrench_b = torch.zeros(cube_object.num_instances, len(body_ids), 6, device=sim.device)
        # Every 2nd cube should have a force applied to it
        external_wrench_b[0::2, :, 2] = 9.81 * cube_object.data.body_mass.torch[0]

        # Now we are ready!
        for i in range(5):
            # reset root state
            root_pose = cube_object.data.default_root_pose.torch.clone()
            root_vel = cube_object.data.default_root_vel.torch.clone()

            # need to shift the position of the cubes otherwise they will be on top of each other
            root_pose[:, :3] = origins
            cube_object.write_root_pose_to_sim_index(root_pose=root_pose)
            cube_object.write_root_velocity_to_sim_index(root_velocity=root_vel)

            # reset object
            cube_object.reset()

            is_global = False
            if i % 2 == 0:
                is_global = True
                positions = cube_object.data.body_com_pos_w.torch[:, body_ids, :3]
            else:
                positions = None

            # apply force
            cube_object.permanent_wrench_composer.set_forces_and_torques_index(
                forces=external_wrench_b[..., :3],
                torques=external_wrench_b[..., 3:],
                positions=positions,
                body_ids=body_ids,
                is_global=is_global,
            )
            # perform simulation
            for _ in range(5):
                # apply action to the object
                cube_object.write_data_to_sim()

                # perform step
                sim.step()

                # update buffers
                cube_object.update(sim.cfg.dt)

            # First object should still be at the same Z position (1.0)
            torch.testing.assert_close(
                cube_object.data.root_pos_w.torch[0::2, 2], torch.ones(num_cubes // 2, device=sim.device)
            )
            # Second object should have fallen, so it's Z height should be less than initial height of 1.0
            assert torch.all(cube_object.data.root_pos_w.torch[1::2, 2] < 1.0)


@pytest.mark.parametrize("num_cubes", [2, 4])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_external_force_on_single_body_at_position(num_cubes, device):
    """Test application of external force on the base of the object at a specific position.

    In this test, we apply a force equal to the weight of an object on the base of
    one of the objects at 1m in the Y direction, we check that the object rotates around it's X axis.
    For the other object, we do not apply any force and check that it falls down.

    We validate that this works when we apply the force in the global frame and in the local frame.
    """
    # Generate cubes scene
    with _ovphysx_sim_context(device=device, add_ground_plane=True, auto_add_lighting=True) as sim:
        cube_object, origins = generate_cubes_scene(num_cubes=num_cubes, device=device)

        # Play the simulator
        sim.reset()

        # Find bodies to apply the force
        body_ids, body_names = cube_object.find_bodies(".*")

        # Sample a force equal to the weight of the object
        external_wrench_b = torch.zeros(cube_object.num_instances, len(body_ids), 6, device=sim.device)
        external_wrench_positions_b = torch.zeros(cube_object.num_instances, len(body_ids), 3, device=sim.device)
        # Every 2nd cube should have a force applied to it
        external_wrench_b[0::2, :, 2] = 500.0
        external_wrench_positions_b[0::2, :, 1] = 1.0

        # Desired force and torque
        desired_force = torch.zeros(cube_object.num_instances, len(body_ids), 3, device=sim.device)
        desired_force[0::2, :, 2] = 1000.0
        desired_torque = torch.zeros(cube_object.num_instances, len(body_ids), 3, device=sim.device)
        desired_torque[0::2, :, 0] = 1000.0
        # Now we are ready!
        for i in range(5):
            # reset root state
            root_pose = cube_object.data.default_root_pose.torch.clone()
            root_vel = cube_object.data.default_root_vel.torch.clone()

            # need to shift the position of the cubes otherwise they will be on top of each other
            root_pose[:, :3] = origins
            cube_object.write_root_pose_to_sim_index(root_pose=root_pose)
            cube_object.write_root_velocity_to_sim_index(root_velocity=root_vel)

            # reset object
            cube_object.reset()

            is_global = False
            if i % 2 == 0:
                is_global = True
                body_com_pos_w = cube_object.data.body_com_pos_w.torch[:, body_ids, :3]
                external_wrench_positions_b[..., 0] = 0.0
                external_wrench_positions_b[..., 1] = 1.0
                external_wrench_positions_b[..., 2] = 0.0
                external_wrench_positions_b += body_com_pos_w
            else:
                external_wrench_positions_b[..., 0] = 0.0
                external_wrench_positions_b[..., 1] = 1.0
                external_wrench_positions_b[..., 2] = 0.0

            # apply force
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
            # perform simulation
            for _ in range(5):
                # apply action to the object
                cube_object.write_data_to_sim()

                # perform step
                sim.step()

                # update buffers
                cube_object.update(sim.cfg.dt)

            # The first object should be rotating around it's X axis
            assert torch.all(torch.abs(cube_object.data.root_ang_vel_b.torch[0::2, 0]) > 0.1)
            # Second object should have fallen, so it's Z height should be less than initial height of 1.0
            assert torch.all(cube_object.data.root_pos_w.torch[1::2, 2] < 1.0)


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_set_rigid_object_state(num_cubes, device):
    """Test setting the state of the rigid object.

    In this test, we set the state of the rigid object to a random state and check
    that the object is in that state after simulation. We set gravity to zero as
    we don't want any external forces acting on the object to ensure state remains static.
    """
    # Turn off gravity for this test as we don't want any external forces acting on the object
    # to ensure state remains static
    with _ovphysx_sim_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        # Generate cubes scene
        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, device=device)

        # Play the simulator
        sim.reset()

        state_types = ["root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w"]

        # Set each state type individually as they are dependent on each other
        for state_type_to_randomize in state_types:
            state_dict = {
                "root_pos_w": torch.zeros_like(cube_object.data.root_pos_w.torch, device=sim.device),
                "root_quat_w": default_orientation(num=num_cubes, device=sim.device),
                "root_lin_vel_w": torch.zeros_like(cube_object.data.root_lin_vel_w.torch, device=sim.device),
                "root_ang_vel_w": torch.zeros_like(cube_object.data.root_ang_vel_w.torch, device=sim.device),
            }

            # Now we are ready!
            for _ in range(5):
                # reset object
                cube_object.reset()

                # Set random state
                if state_type_to_randomize == "root_quat_w":
                    state_dict[state_type_to_randomize] = random_orientation(num=num_cubes, device=sim.device)
                else:
                    state_dict[state_type_to_randomize] = torch.randn(num_cubes, 3, device=sim.device)

                # perform simulation
                for _ in range(5):
                    root_pose = torch.cat(
                        [state_dict["root_pos_w"], state_dict["root_quat_w"]],
                        dim=-1,
                    )
                    root_vel = torch.cat(
                        [state_dict["root_lin_vel_w"], state_dict["root_ang_vel_w"]],
                        dim=-1,
                    )
                    # reset root state
                    cube_object.write_root_pose_to_sim_index(root_pose=root_pose)
                    cube_object.write_root_velocity_to_sim_index(root_velocity=root_vel)

                    sim.step()

                    # assert that set root quantities are equal to the ones set in the state_dict
                    for key, expected_value in state_dict.items():
                        value = getattr(cube_object.data, key).torch
                        torch.testing.assert_close(value, expected_value, rtol=1e-3, atol=1e-3)

                    cube_object.update(sim.cfg.dt)


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_reset_rigid_object(num_cubes, device):
    """Test resetting the state of the rigid object."""
    with _ovphysx_sim_context(device=device, gravity_enabled=True, auto_add_lighting=True) as sim:
        # Generate cubes scene
        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, device=device)

        # Play the simulator
        sim.reset()

        for i in range(5):
            # perform rendering
            sim.step()

            # update object
            cube_object.update(sim.cfg.dt)

            # Move the object to a random position
            root_pose = cube_object.data.default_root_pose.torch.clone()
            root_pose[:, :3] = torch.randn(num_cubes, 3, device=sim.device)

            # Random orientation
            root_pose[:, 3:7] = random_orientation(num=num_cubes, device=sim.device)
            cube_object.write_root_pose_to_sim_index(root_pose=root_pose)
            root_vel = cube_object.data.default_root_vel.torch.clone()
            cube_object.write_root_velocity_to_sim_index(root_velocity=root_vel)

            if i % 2 == 0:
                # reset object
                cube_object.reset()

                # Reset should zero external forces and torques
                assert not cube_object._instantaneous_wrench_composer.active
                assert not cube_object._permanent_wrench_composer.active
                assert torch.count_nonzero(cube_object._instantaneous_wrench_composer.composed_force.torch) == 0
                assert torch.count_nonzero(cube_object._instantaneous_wrench_composer.composed_torque.torch) == 0
                assert torch.count_nonzero(cube_object._permanent_wrench_composer.composed_force.torch) == 0
                assert torch.count_nonzero(cube_object._permanent_wrench_composer.composed_torque.torch) == 0


@pytest.mark.xfail(reason=_MATERIAL_GAP_REASON, strict=False)
@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_rigid_body_set_material_properties(num_cubes, device):
    """Test getting and setting material properties of rigid object."""
    raise NotImplementedError(_MATERIAL_GAP_REASON)


@pytest.mark.xfail(reason=_MATERIAL_GAP_REASON, strict=False)
@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_set_material_properties_via_view(num_cubes, device):
    """Test setting material properties via the PhysX view-level API."""
    raise NotImplementedError(_MATERIAL_GAP_REASON)


@pytest.mark.xfail(reason=_MATERIAL_GAP_REASON, strict=False)
@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_rigid_body_no_friction(num_cubes, device):
    """Test that a rigid object with no friction will maintain it's velocity when sliding across a plane."""
    raise NotImplementedError(_MATERIAL_GAP_REASON)


@pytest.mark.xfail(reason=_MATERIAL_GAP_REASON, strict=False)
@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.isaacsim_ci
def test_rigid_body_with_static_friction(num_cubes, device):
    """Test that static friction applied to rigid object works as expected.

    This test works by applying a force to the object and checking if the object moves or not based on the
    mu (coefficient of static friction) value set for the object. We set the static friction to be non-zero and
    apply a force to the object. When the force applied is below mu, the object should not move. When the force
    applied is above mu, the object should move.
    """
    raise NotImplementedError(_MATERIAL_GAP_REASON)


@pytest.mark.xfail(reason=_MATERIAL_GAP_REASON, strict=False)
@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_rigid_body_with_restitution(num_cubes, device):
    """Test that restitution when applied to rigid object works as expected.

    This test works by dropping a block from a height and checking if the block bounces or not based on the
    restitution value set for the object. We set the restitution to be non-zero and drop the block from a height.
    When the restitution is 0, the block should not bounce. When the restitution is between 0 and 1, the block
    should bounce with less energy.
    """
    raise NotImplementedError(_MATERIAL_GAP_REASON)


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_rigid_body_set_mass(num_cubes, device):
    """Test getting and setting mass of rigid object."""
    with _ovphysx_sim_context(
        device=device, gravity_enabled=False, add_ground_plane=True, auto_add_lighting=True
    ) as sim:
        # Create a scene with random cubes
        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, height=1.0, device=device)

        # Play sim
        sim.reset()

        # Get masses before increasing
        original_masses = cube_object.data.body_mass.torch.clone()

        assert original_masses.shape == (num_cubes, 1)

        # Randomize mass of the object
        masses = original_masses + torch.FloatTensor(num_cubes, 1).uniform_(4, 8).to(sim.device)

        indices = torch.tensor(range(num_cubes), dtype=torch.int32)

        # Set the new masses via the OVPhysX writer (matches PhysX/Newton).
        cube_object.set_masses_index(
            masses=wp.from_torch(masses.contiguous(), dtype=wp.float32),
            env_ids=wp.from_torch(indices, dtype=wp.int32),
        )

        torch.testing.assert_close(cube_object.data.body_mass.torch, masses)

        # Simulate physics
        # perform rendering
        sim.step()
        # update object
        cube_object.update(sim.cfg.dt)

        masses_to_check = cube_object.data.body_mass.torch

        # Check if mass is set correctly
        torch.testing.assert_close(masses, masses_to_check)


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("gravity_enabled", [True, False])
@pytest.mark.isaacsim_ci
def test_gravity_vec_w(num_cubes, device, gravity_enabled):
    """Test that gravity vector direction is set correctly for the rigid object."""
    with _ovphysx_sim_context(device=device, gravity_enabled=gravity_enabled) as sim:
        # Create a scene with random cubes
        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, device=device)

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
            gravity = torch.zeros(num_cubes, 1, 6, device=device)
            if gravity_enabled:
                gravity[:, :, 2] = -9.81
            # Check the body accelerations are correct
            torch.testing.assert_close(cube_object.data.body_acc_w.torch, gravity)


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("with_offset", [True, False])
@pytest.mark.isaacsim_ci
@flaky(max_runs=3, min_passes=1)
def test_body_root_state_properties(num_cubes, device, with_offset):
    """Test the root_com_state_w, root_link_state_w, body_com_state_w, and body_link_state_w properties."""
    with _ovphysx_sim_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        # Create a scene with random cubes
        cube_object, env_pos = generate_cubes_scene(num_cubes=num_cubes, height=0.0, device=device)
        env_idx = torch.tensor([x for x in range(num_cubes)], dtype=torch.int32)

        # Play sim
        sim.reset()

        # Check if cube_object is initialized
        assert cube_object.is_initialized

        # change center of mass offset from link frame
        if with_offset:
            offset = torch.tensor([0.1, 0.0, 0.0], device=device).repeat(num_cubes, 1)
        else:
            offset = torch.tensor([0.0, 0.0, 0.0], device=device).repeat(num_cubes, 1)

        # Read current COMs, mutate the translation, write back via the OVPhysX
        # ``set_coms_index`` setter (PhysX uses ``root_view.set_coms`` for the same
        # operation; OVPhysX wraps the wheel ``RIGID_BODY_COM_POSE`` write in
        # :meth:`set_coms_index`, which follows the PhysX ``wp.transformf`` contract).
        com = cube_object.data.body_com_pose_b.torch.clone()  # shape (N, 1, 7)
        com[..., :3] = offset.to(com.device).unsqueeze(1)
        cube_object.set_coms_index(
            coms=wp.from_torch(com.contiguous(), dtype=wp.transformf),
            env_ids=wp.from_torch(env_idx, dtype=wp.int32),
        )

        # check ceter of mass has been set
        torch.testing.assert_close(cube_object.data.body_com_pose_b.torch, com)

        # random z spin velocity
        spin_twist = torch.zeros(6, device=device)
        spin_twist[5] = torch.randn(1, device=device)

        # Simulate physics
        for _ in range(100):
            # spin the object around Z axis (com)
            cube_object.write_root_velocity_to_sim_index(root_velocity=spin_twist.repeat(num_cubes, 1))
            # perform rendering
            sim.step()
            # update object
            cube_object.update(sim.cfg.dt)

            # get state properties
            root_link_pose_w = cube_object.data.root_link_pose_w.torch
            root_link_vel_w = cube_object.data.root_link_vel_w.torch
            root_com_pose_w = cube_object.data.root_com_pose_w.torch
            root_com_vel_w = cube_object.data.root_com_vel_w.torch
            body_link_pose_w = cube_object.data.body_link_pose_w.torch
            body_link_vel_w = cube_object.data.body_link_vel_w.torch
            body_com_pose_w = cube_object.data.body_com_pose_w.torch
            body_com_vel_w = cube_object.data.body_com_vel_w.torch

            # if offset is [0,0,0] all root_state_%_w will match and all body_%_w will match
            if not with_offset:
                torch.testing.assert_close(root_link_pose_w, root_com_pose_w)
                torch.testing.assert_close(root_com_vel_w, root_link_vel_w)
                torch.testing.assert_close(root_link_pose_w, root_link_pose_w)
                torch.testing.assert_close(root_com_vel_w, root_link_vel_w)
                torch.testing.assert_close(body_link_pose_w, body_com_pose_w)
                torch.testing.assert_close(body_com_vel_w, body_link_vel_w)
                torch.testing.assert_close(body_link_pose_w, body_link_pose_w)
                torch.testing.assert_close(body_com_vel_w, body_link_vel_w)
            else:
                # cubes are spinning around center of mass
                # position will not match
                # center of mass position will be constant (i.e. spinning around com)
                torch.testing.assert_close(env_pos + offset, root_com_pose_w[..., :3])
                torch.testing.assert_close(env_pos + offset, body_com_pose_w[..., :3].squeeze(-2))
                # link position will be moving but should stay constant away from center of mass
                root_link_state_pos_rel_com = quat_apply_inverse(
                    root_link_pose_w[..., 3:],
                    root_link_pose_w[..., :3] - root_com_pose_w[..., :3],
                )
                torch.testing.assert_close(-offset, root_link_state_pos_rel_com)
                body_link_state_pos_rel_com = quat_apply_inverse(
                    body_link_pose_w[..., 3:],
                    body_link_pose_w[..., :3] - body_com_pose_w[..., :3],
                )
                torch.testing.assert_close(-offset, body_link_state_pos_rel_com.squeeze(-2))

                # orientation of com will be a constant rotation from link orientation
                com_quat_b = cube_object.data.body_com_quat_b.torch
                com_quat_w = quat_mul(body_link_pose_w[..., 3:], com_quat_b)
                torch.testing.assert_close(com_quat_w, body_com_pose_w[..., 3:])
                torch.testing.assert_close(com_quat_w.squeeze(-2), root_com_pose_w[..., 3:])

                # orientation of link will match root state will always match
                torch.testing.assert_close(root_link_pose_w[..., 3:], root_link_pose_w[..., 3:])
                torch.testing.assert_close(body_link_pose_w[..., 3:], body_link_pose_w[..., 3:])

                # lin_vel will not match
                # center of mass vel will be constant (i.e. spinning around com)
                torch.testing.assert_close(torch.zeros_like(root_com_vel_w[..., :3]), root_com_vel_w[..., :3])
                torch.testing.assert_close(torch.zeros_like(body_com_vel_w[..., :3]), body_com_vel_w[..., :3])
                # link frame will be moving, and should be equal to input angular velocity cross offset
                lin_vel_rel_root_gt = quat_apply_inverse(root_link_pose_w[..., 3:], root_link_vel_w[..., :3])
                lin_vel_rel_body_gt = quat_apply_inverse(body_link_pose_w[..., 3:], body_link_vel_w[..., :3])
                lin_vel_rel_gt = torch.linalg.cross(spin_twist.repeat(num_cubes, 1)[..., 3:], -offset)
                torch.testing.assert_close(lin_vel_rel_gt, lin_vel_rel_root_gt, atol=1e-4, rtol=1e-4)
                torch.testing.assert_close(lin_vel_rel_gt, lin_vel_rel_body_gt.squeeze(-2), atol=1e-4, rtol=1e-4)

                # ang_vel will always match
                torch.testing.assert_close(root_com_vel_w[..., 3:], root_com_vel_w[..., 3:])
                torch.testing.assert_close(root_com_vel_w[..., 3:], root_link_vel_w[..., 3:])
                torch.testing.assert_close(body_com_vel_w[..., 3:], body_com_vel_w[..., 3:])
                torch.testing.assert_close(body_com_vel_w[..., 3:], body_link_vel_w[..., 3:])


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("with_offset", [True, False])
@pytest.mark.parametrize("state_location", ["com", "link"])
@pytest.mark.isaacsim_ci
def test_write_root_state(num_cubes, device, with_offset, state_location):
    """Test the setters for root_state using both the link frame and center of mass as reference frame."""
    with _ovphysx_sim_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        # Create a scene with random cubes
        cube_object, env_pos = generate_cubes_scene(num_cubes=num_cubes, height=0.0, device=device)
        env_idx = torch.tensor([x for x in range(num_cubes)], dtype=torch.int32)

        # Play sim
        sim.reset()

        # Check if cube_object is initialized
        assert cube_object.is_initialized

        # change center of mass offset from link frame
        if with_offset:
            offset = torch.tensor([0.1, 0.0, 0.0], device=device).repeat(num_cubes, 1)
        else:
            offset = torch.tensor([0.0, 0.0, 0.0], device=device).repeat(num_cubes, 1)

        com = cube_object.data.body_com_pose_b.torch.clone()  # shape (N, 1, 7)
        com[..., :3] = offset.to(com.device).unsqueeze(1)
        cube_object.set_coms_index(
            coms=wp.from_torch(com.contiguous(), dtype=wp.transformf),
            env_ids=wp.from_torch(env_idx, dtype=wp.int32),
        )

        # check center of mass has been set
        torch.testing.assert_close(cube_object.data.body_com_pose_b.torch, com)

        rand_state = torch.zeros(num_cubes, 13, device=device)
        rand_state[..., :7] = cube_object.data.default_root_pose.torch
        rand_state[..., :3] += env_pos
        # make quaternion a unit vector
        rand_state[..., 3:7] = torch.nn.functional.normalize(rand_state[..., 3:7], dim=-1)

        env_idx = env_idx.to(device)
        for i in range(10):
            # perform step
            sim.step()
            # update buffers
            cube_object.update(sim.cfg.dt)

            if state_location == "com":
                if i % 2 == 0:
                    cube_object.write_root_com_pose_to_sim_index(root_pose=rand_state[..., :7])
                    cube_object.write_root_com_velocity_to_sim_index(root_velocity=rand_state[..., 7:])
                else:
                    cube_object.write_root_com_pose_to_sim_index(root_pose=rand_state[..., :7], env_ids=env_idx)
                    cube_object.write_root_com_velocity_to_sim_index(root_velocity=rand_state[..., 7:], env_ids=env_idx)
            elif state_location == "link":
                if i % 2 == 0:
                    cube_object.write_root_link_pose_to_sim_index(root_pose=rand_state[..., :7])
                    cube_object.write_root_link_velocity_to_sim_index(root_velocity=rand_state[..., 7:])
                else:
                    cube_object.write_root_link_pose_to_sim_index(root_pose=rand_state[..., :7], env_ids=env_idx)
                    cube_object.write_root_link_velocity_to_sim_index(
                        root_velocity=rand_state[..., 7:], env_ids=env_idx
                    )

            if state_location == "com":
                torch.testing.assert_close(rand_state[..., :7], cube_object.data.root_com_pose_w.torch)
                torch.testing.assert_close(rand_state[..., 7:], cube_object.data.root_com_vel_w.torch)
            elif state_location == "link":
                torch.testing.assert_close(rand_state[..., :7], cube_object.data.root_link_pose_w.torch)
                torch.testing.assert_close(rand_state[..., 7:], cube_object.data.root_link_vel_w.torch)


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("with_offset", [True])
@pytest.mark.parametrize("state_location", ["com", "link", "root"])
@pytest.mark.isaacsim_ci
def test_write_state_functions_data_consistency(num_cubes, device, with_offset, state_location):
    """Test the setters for root_state using both the link frame and center of mass as reference frame."""
    with _ovphysx_sim_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        # Create a scene with random cubes
        cube_object, env_pos = generate_cubes_scene(num_cubes=num_cubes, height=0.0, device=device)
        env_idx = torch.tensor([x for x in range(num_cubes)], dtype=torch.int32)

        # Play sim
        sim.reset()

        # Check if cube_object is initialized
        assert cube_object.is_initialized

        # change center of mass offset from link frame
        if with_offset:
            offset = torch.tensor([0.1, 0.0, 0.0], device=device).repeat(num_cubes, 1)
        else:
            offset = torch.tensor([0.0, 0.0, 0.0], device=device).repeat(num_cubes, 1)

        com = cube_object.data.body_com_pose_b.torch.clone()  # shape (N, 1, 7)
        com[..., :3] = offset.to(com.device).unsqueeze(1)
        cube_object.set_coms_index(
            coms=wp.from_torch(com.contiguous(), dtype=wp.transformf),
            env_ids=wp.from_torch(env_idx, dtype=wp.int32),
        )

        # check ceter of mass has been set
        torch.testing.assert_close(cube_object.data.body_com_pose_b.torch, com)

        rand_state = torch.rand(num_cubes, 13, device=device)
        rand_state[..., :3] += env_pos
        # make quaternion a unit vector
        rand_state[..., 3:7] = torch.nn.functional.normalize(rand_state[..., 3:7], dim=-1)

        env_idx = env_idx.to(device)

        # perform step
        sim.step()
        # update buffers
        cube_object.update(sim.cfg.dt)

        if state_location == "com":
            cube_object.write_root_com_pose_to_sim_index(root_pose=rand_state[..., :7])
            cube_object.write_root_com_velocity_to_sim_index(root_velocity=rand_state[..., 7:])
        elif state_location == "link":
            cube_object.write_root_link_pose_to_sim_index(root_pose=rand_state[..., :7])
            cube_object.write_root_link_velocity_to_sim_index(root_velocity=rand_state[..., 7:])
        elif state_location == "root":
            cube_object.write_root_pose_to_sim_index(root_pose=rand_state[..., :7])
            cube_object.write_root_velocity_to_sim_index(root_velocity=rand_state[..., 7:])

        if state_location == "com":
            root_com_pose_w = cube_object.data.root_com_pose_w.torch
            root_com_vel_w = cube_object.data.root_com_vel_w.torch
            body_com_pose_b = cube_object.data.body_com_pose_b.torch
            expected_root_link_pos, expected_root_link_quat = combine_frame_transforms(
                root_com_pose_w[:, :3],
                root_com_pose_w[:, 3:],
                quat_rotate(quat_inv(body_com_pose_b[:, 0, 3:7]), -body_com_pose_b[:, 0, :3]),
                quat_inv(body_com_pose_b[:, 0, 3:7]),
            )
            expected_root_link_pose = torch.cat((expected_root_link_pos, expected_root_link_quat), dim=1)
            root_link_pose_w = cube_object.data.root_link_pose_w.torch
            root_link_vel_w = cube_object.data.root_link_vel_w.torch
            # test both root_pose and root_link successfully updated when root_com updates
            torch.testing.assert_close(expected_root_link_pose, root_link_pose_w)
            # skip lin_vel because it differs from link frame, this should be fine because we are only checking
            # if velocity update is triggered, which can be determined by comparing angular velocity
            torch.testing.assert_close(root_com_vel_w[:, 3:], root_link_vel_w[:, 3:])
            torch.testing.assert_close(expected_root_link_pose, root_link_pose_w)
            torch.testing.assert_close(root_com_vel_w[:, 3:], cube_object.data.root_com_vel_w.torch[:, 3:])
        elif state_location == "link":
            root_link_pose_w = cube_object.data.root_link_pose_w.torch
            root_link_vel_w = cube_object.data.root_link_vel_w.torch
            body_com_pose_b = cube_object.data.body_com_pose_b.torch
            expected_com_pos, expected_com_quat = combine_frame_transforms(
                root_link_pose_w[:, :3],
                root_link_pose_w[:, 3:],
                body_com_pose_b[:, 0, :3],
                body_com_pose_b[:, 0, 3:7],
            )
            expected_com_pose = torch.cat((expected_com_pos, expected_com_quat), dim=1)
            root_com_pose_w = cube_object.data.root_com_pose_w.torch
            root_com_vel_w = cube_object.data.root_com_vel_w.torch
            # test both root_pose and root_com successfully updated when root_link updates
            torch.testing.assert_close(expected_com_pose, root_com_pose_w)
            # skip lin_vel because it differs from link frame, this should be fine because we are only checking
            # if velocity update is triggered, which can be determined by comparing angular velocity
            torch.testing.assert_close(root_link_vel_w[:, 3:], root_com_vel_w[:, 3:])
            torch.testing.assert_close(root_link_pose_w, cube_object.data.root_link_pose_w.torch)
            torch.testing.assert_close(root_link_vel_w[:, 3:], cube_object.data.root_com_vel_w.torch[:, 3:])
        elif state_location == "root":
            root_link_pose_w = cube_object.data.root_link_pose_w.torch
            root_com_vel_w = cube_object.data.root_com_vel_w.torch
            body_com_pose_b = cube_object.data.body_com_pose_b.torch
            expected_com_pos, expected_com_quat = combine_frame_transforms(
                root_link_pose_w[:, :3],
                root_link_pose_w[:, 3:],
                body_com_pose_b[:, 0, :3],
                body_com_pose_b[:, 0, 3:7],
            )
            expected_com_pose = torch.cat((expected_com_pos, expected_com_quat), dim=1)
            root_com_pose_w = cube_object.data.root_com_pose_w.torch
            root_link_vel_w = cube_object.data.root_link_vel_w.torch
            # test both root_com and root_link successfully updated when root_pose updates
            torch.testing.assert_close(expected_com_pose, root_com_pose_w)
            torch.testing.assert_close(root_com_vel_w, cube_object.data.root_com_vel_w.torch)
            torch.testing.assert_close(root_link_pose_w, cube_object.data.root_link_pose_w.torch)
            torch.testing.assert_close(root_com_vel_w[:, 3:], root_link_vel_w[:, 3:])


@pytest.mark.isaacsim_ci
def test_warmup_attach_stage_not_called_for_cpu():
    """Regression test: ``physx.warmup_gpu()`` must not be called for CPU.

    OVPhysX-equivalent of PhysX's ``test_warmup_attach_stage_not_called_for_cpu``:
    PhysX guards :meth:`attach_stage` with ``if is_gpu:`` so the CPU MBP
    broadphase is not double-initialised.  The OVPhysX manager has the same
    structural guard around :meth:`OvPhysxManager._physx.warmup_gpu`: it is
    only invoked when ``ovphysx_device == "gpu"``.

    We monkey-patch ``OvPhysxManager._physx`` with a :class:`MagicMock`
    wrapping the live PhysX object so that ``warmup_gpu`` becomes a spy while
    other calls continue to forward, then assert ``warmup_gpu.call_count == 0``
    after a CPU-mode :meth:`sim.reset`.

    The test always runs CPU regardless of session parametrization, so it is
    skipped when the session-locked device is anything other than CPU.  The
    skip is enforced inline (rather than in the autouse fixture) so the rest
    of the suite can still pin to GPU when invoked together.
    """
    if _LOCKED_DEVICE[0] not in (None, "cpu"):
        pytest.skip(
            f"ovphysx process-global device lock is held by '{_LOCKED_DEVICE[0]}'; cannot run "
            "CPU-only regression test in the same session."
        )
    _LOCKED_DEVICE[0] = "cpu"

    with _ovphysx_sim_context(device="cpu", add_ground_plane=True, dt=0.01, auto_add_lighting=True) as sim:
        # Allocate a single rigid body so the manager has something to load.
        generate_cubes_scene(num_cubes=1, height=1.0, device="cpu")

        # First reset constructs (or reuses) the real ovphysx.PhysX so we have
        # a live instance to wrap.  The PhysX object is a C++ binding, so we
        # cannot patch attributes directly — replace the class-level reference
        # with a MagicMock(wraps=...) that forwards every call.
        sim.reset()
        original_physx = OvPhysxManager._physx
        assert original_physx is not None, "PhysX should be constructed after sim.reset()"
        spy = MagicMock(wraps=original_physx)
        OvPhysxManager._physx = spy
        # Force _warmup_and_load to run again on the next reset so the spy
        # observes the warmup_gpu (or non-call) decision; close() resets
        # _warmup_done back to False but we just called sim.reset() above.
        OvPhysxManager._warmup_done = False
        try:
            sim.reset()
        finally:
            OvPhysxManager._physx = original_physx

        assert spy.warmup_gpu.call_count == 0, (
            f"warmup_gpu() was called {spy.warmup_gpu.call_count} time(s) during CPU warmup. "
            "OvPhysxManager._warmup_and_load() must guard warmup_gpu() with "
            "ovphysx_device == 'gpu' so the CPU pipeline is not mis-initialised."
        )
