# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Launch Isaac Sim Simulator first."""

from omni.isaac.lab.app import AppLauncher, run_tests

HEADLESS = True

# launch omniverse app
app_launcher = AppLauncher(headless=HEADLESS)
simulation_app = app_launcher.app

"""Rest everything follows."""

import ctypes
import torch
import unittest
from typing import Literal

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.string as string_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.sim import build_simulation_context
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from omni.isaac.lab_assets import ANYMAL_C_CFG, FRANKA_PANDA_CFG, SHADOW_HAND_CFG  # isort:skip


def generate_articulation_cfg(
    articulation_type: Literal["humanoid", "panda", "anymal", "shadow_hand", "single_joint"],
    stiffness: float | None = 10.0,
    damping: float | None = 2.0,
) -> ArticulationCfg:
    """Generate an articulation configuration.

    Args:
        articulation_type: Type of articulation to generate.
        stiffness: Stiffness value for the articulation's actuators. Only currently used for humanoid.
        damping: Damping value for the articulation's actuators. Only currently used for humanoid.

    Returns:
        The articulation configuration for the requested articulation type.

    """
    if articulation_type == "humanoid":
        articulation_cfg = ArticulationCfg(
            spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Humanoid/humanoid_instanceable.usd"),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.34)),
            actuators={"body": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=stiffness, damping=damping)},
        )
    elif articulation_type == "panda":
        articulation_cfg = FRANKA_PANDA_CFG
    elif articulation_type == "anymal":
        articulation_cfg = ANYMAL_C_CFG
    elif articulation_type == "shadow_hand":
        articulation_cfg = SHADOW_HAND_CFG
    elif articulation_type == "single_joint":
        articulation_cfg = ArticulationCfg(
            spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Simple/revolute_articulation.usd"),
            actuators={
                "joint": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    effort_limit=400.0,
                    velocity_limit=100.0,
                    stiffness=0.0,
                    damping=10.0,
                ),
            },
        )
    else:
        raise ValueError(
            f"Invalid articulation type: {articulation_type}, valid options are 'humanoid', 'panda', 'anymal',"
            " 'shadow_hand' and 'single_joint'."
        )

    return articulation_cfg


def generate_articulation(
    articulation_cfg: ArticulationCfg, num_articulations: int, device: str
) -> tuple[Articulation, torch.tensor]:
    """Generate an articulation from a configuration.

    Handles the creation of the articulation, the environment prims and the articulation's environment
    translations

    Args:
        articulation_cfg: Articulation configuration.
        num_articulations: Number of articulations to generate.
        device: Device to use for the tensors.

    Returns:
        The articulation and environment translations.

    """
    # Generate translations of 1.5m in x for each articulation
    translations = torch.zeros(num_articulations, 3, device=device)
    translations[:, 0] = torch.arange(num_articulations) * 1.5

    # Create Top-level Xforms, one for each articulation
    for i in range(num_articulations):
        prim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=translations[i][:3])
    articulation = Articulation(articulation_cfg.replace(prim_path="/World/Env_.*/Robot"))

    return articulation, translations


class TestArticulation(unittest.TestCase):
    """Test for articulation class."""

    """
    Tests
    """

    def test_initialization_floating_base_non_root(self):
        """Test initialization for a floating-base with articulation root on a rigid body.
        under the provided prim path."""
        for num_articulations in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_articulations=num_articulations, device=device):
                    with build_simulation_context(device=device, add_ground_plane=True, auto_add_lighting=True) as sim:
                        articulation_cfg = generate_articulation_cfg(
                            articulation_type="humanoid", stiffness=0.0, damping=0.0
                        )
                        articulation, _ = generate_articulation(articulation_cfg, num_articulations, device)

                        # Check that boundedness of articulation is correct
                        self.assertEqual(ctypes.c_long.from_address(id(articulation)).value, 1)

                        # Play sim
                        sim.reset()

                        # # Check if articulation is initialized
                        self.assertTrue(articulation.is_initialized)
                        # Check that is fixed base
                        self.assertFalse(articulation.is_fixed_base)
                        # Check buffers that exists and have correct shapes
                        self.assertEqual(articulation.data.root_pos_w.shape, (num_articulations, 3))
                        self.assertEqual(articulation.data.root_quat_w.shape, (num_articulations, 4))
                        self.assertEqual(articulation.data.joint_pos.shape, (num_articulations, 21))

                        # Check some internal physx data for debugging
                        # -- joint related
                        self.assertEqual(
                            articulation.root_physx_view.max_dofs,
                            articulation.root_physx_view.shared_metatype.dof_count,
                        )
                        # -- link related
                        self.assertEqual(
                            articulation.root_physx_view.max_links,
                            articulation.root_physx_view.shared_metatype.link_count,
                        )
                        # -- link names (check within articulation ordering is correct)
                        prim_path_body_names = [
                            path.split("/")[-1] for path in articulation.root_physx_view.link_paths[0]
                        ]
                        self.assertListEqual(prim_path_body_names, articulation.body_names)

                        # Simulate physics
                        for _ in range(10):
                            # perform rendering
                            sim.step()
                            # update articulation
                            articulation.update(sim.cfg.dt)

    def test_initialization_floating_base(self):
        """Test initialization for a floating-base with articulation root on provided prim path."""
        for num_articulations in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_articulations=num_articulations, device=device):
                    with build_simulation_context(device=device, add_ground_plane=True, auto_add_lighting=True) as sim:
                        articulation_cfg = generate_articulation_cfg(
                            articulation_type="anymal", stiffness=0.0, damping=0.0
                        )
                        articulation, _ = generate_articulation(articulation_cfg, num_articulations, device)

                        # Check that boundedness of articulation is correct
                        self.assertEqual(ctypes.c_long.from_address(id(articulation)).value, 1)

                        # Play sim
                        sim.reset()
                        # Check if articulation is initialized
                        self.assertTrue(articulation.is_initialized)
                        # Check that floating base
                        self.assertFalse(articulation.is_fixed_base)
                        # Check buffers that exists and have correct shapes
                        self.assertEqual(articulation.data.root_pos_w.shape, (num_articulations, 3))
                        self.assertEqual(articulation.data.root_quat_w.shape, (num_articulations, 4))
                        self.assertEqual(articulation.data.joint_pos.shape, (num_articulations, 12))
                        self.assertEqual(
                            articulation.data.default_mass.shape, (num_articulations, articulation.num_bodies)
                        )
                        self.assertEqual(
                            articulation.data.default_inertia.shape, (num_articulations, articulation.num_bodies, 9)
                        )

                        # Check some internal physx data for debugging
                        # -- joint related
                        self.assertEqual(
                            articulation.root_physx_view.max_dofs,
                            articulation.root_physx_view.shared_metatype.dof_count,
                        )
                        # -- link related
                        self.assertEqual(
                            articulation.root_physx_view.max_links,
                            articulation.root_physx_view.shared_metatype.link_count,
                        )
                        # -- link names (check within articulation ordering is correct)
                        prim_path_body_names = [
                            path.split("/")[-1] for path in articulation.root_physx_view.link_paths[0]
                        ]
                        self.assertListEqual(prim_path_body_names, articulation.body_names)

                        # Simulate physics
                        for _ in range(10):
                            # perform rendering
                            sim.step()
                            # update articulation
                            articulation.update(sim.cfg.dt)

    def test_initialization_fixed_base(self):
        """Test initialization for fixed base."""
        for num_articulations in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_articulations=num_articulations, device=device):
                    with build_simulation_context(device=device, add_ground_plane=False, auto_add_lighting=True) as sim:
                        articulation_cfg = generate_articulation_cfg(articulation_type="panda")
                        articulation, translations = generate_articulation(articulation_cfg, num_articulations, device)

                        # Check that boundedness of articulation is correct
                        self.assertEqual(ctypes.c_long.from_address(id(articulation)).value, 1)

                        # Play sim
                        sim.reset()
                        # Check if articulation is initialized
                        self.assertTrue(articulation.is_initialized)
                        # Check that fixed base
                        self.assertTrue(articulation.is_fixed_base)
                        # Check buffers that exists and have correct shapes
                        self.assertEqual(articulation.data.root_pos_w.shape, (num_articulations, 3))
                        self.assertEqual(articulation.data.root_quat_w.shape, (num_articulations, 4))
                        self.assertEqual(articulation.data.joint_pos.shape, (num_articulations, 9))
                        self.assertEqual(
                            articulation.data.default_mass.shape, (num_articulations, articulation.num_bodies)
                        )
                        self.assertEqual(
                            articulation.data.default_inertia.shape, (num_articulations, articulation.num_bodies, 9)
                        )

                        # Check some internal physx data for debugging
                        # -- joint related
                        self.assertEqual(
                            articulation.root_physx_view.max_dofs,
                            articulation.root_physx_view.shared_metatype.dof_count,
                        )
                        # -- link related
                        self.assertEqual(
                            articulation.root_physx_view.max_links,
                            articulation.root_physx_view.shared_metatype.link_count,
                        )
                        # -- link names (check within articulation ordering is correct)
                        prim_path_body_names = [
                            path.split("/")[-1] for path in articulation.root_physx_view.link_paths[0]
                        ]
                        self.assertListEqual(prim_path_body_names, articulation.body_names)

                        # Simulate physics
                        for _ in range(10):
                            # perform rendering
                            sim.step()
                            # update articulation
                            articulation.update(sim.cfg.dt)

                            # check that the root is at the correct state - its default state as it is fixed base
                            default_root_state = articulation.data.default_root_state.clone()
                            default_root_state[:, :3] = default_root_state[:, :3] + translations

                            torch.testing.assert_close(articulation.data.root_state_w, default_root_state)

    def test_initialization_fixed_base_single_joint(self):
        """Test initialization for fixed base articulation with a single joint."""
        for num_articulations in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_articulations=num_articulations, device=device):
                    with build_simulation_context(device=device, add_ground_plane=True, auto_add_lighting=True) as sim:
                        articulation_cfg = generate_articulation_cfg(articulation_type="single_joint")
                        articulation, translations = generate_articulation(articulation_cfg, num_articulations, device)

                        # Check that boundedness of articulation is correct
                        self.assertEqual(ctypes.c_long.from_address(id(articulation)).value, 1)

                        # Play sim
                        sim.reset()
                        # Check if articulation is initialized
                        self.assertTrue(articulation.is_initialized)
                        # Check that fixed base
                        self.assertTrue(articulation.is_fixed_base)
                        # Check buffers that exists and have correct shapes
                        self.assertEqual(articulation.data.root_pos_w.shape, (num_articulations, 3))
                        self.assertEqual(articulation.data.root_quat_w.shape, (num_articulations, 4))
                        self.assertEqual(articulation.data.joint_pos.shape, (num_articulations, 1))
                        self.assertEqual(
                            articulation.data.default_mass.shape, (num_articulations, articulation.num_bodies)
                        )
                        self.assertEqual(
                            articulation.data.default_inertia.shape, (num_articulations, articulation.num_bodies, 9)
                        )

                        # Check some internal physx data for debugging
                        # -- joint related
                        self.assertEqual(
                            articulation.root_physx_view.max_dofs,
                            articulation.root_physx_view.shared_metatype.dof_count,
                        )
                        # -- link related
                        self.assertEqual(
                            articulation.root_physx_view.max_links,
                            articulation.root_physx_view.shared_metatype.link_count,
                        )
                        # -- link names (check within articulation ordering is correct)
                        prim_path_body_names = [
                            path.split("/")[-1] for path in articulation.root_physx_view.link_paths[0]
                        ]
                        self.assertListEqual(prim_path_body_names, articulation.body_names)

                        # Simulate physics
                        for _ in range(10):
                            # perform rendering
                            sim.step()
                            # update articulation
                            articulation.update(sim.cfg.dt)

                            # check that the root is at the correct state - its default state as it is fixed base
                            default_root_state = articulation.data.default_root_state.clone()
                            default_root_state[:, :3] = default_root_state[:, :3] + translations

                            torch.testing.assert_close(articulation.data.root_state_w, default_root_state)

    def test_initialization_hand_with_tendons(self):
        """Test initialization for fixed base articulated hand with tendons."""
        for num_articulations in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_articulations=num_articulations, device=device):
                    with build_simulation_context(device=device, add_ground_plane=False, auto_add_lighting=True) as sim:
                        articulation_cfg = generate_articulation_cfg(articulation_type="shadow_hand")
                        articulation, _ = generate_articulation(articulation_cfg, num_articulations, device)

                        # Check that boundedness of articulation is correct
                        self.assertEqual(ctypes.c_long.from_address(id(articulation)).value, 1)

                        # Play sim
                        sim.reset()
                        # Check if articulation is initialized
                        self.assertTrue(articulation.is_initialized)
                        # Check that fixed base
                        self.assertTrue(articulation.is_fixed_base)
                        # Check buffers that exists and have correct shapes
                        self.assertTrue(articulation.data.root_pos_w.shape == (num_articulations, 3))
                        self.assertTrue(articulation.data.root_quat_w.shape == (num_articulations, 4))
                        self.assertTrue(articulation.data.joint_pos.shape == (num_articulations, 24))
                        self.assertEqual(
                            articulation.data.default_mass.shape, (num_articulations, articulation.num_bodies)
                        )
                        self.assertEqual(
                            articulation.data.default_inertia.shape, (num_articulations, articulation.num_bodies, 9)
                        )

                        # Check some internal physx data for debugging
                        # -- joint related
                        self.assertEqual(
                            articulation.root_physx_view.max_dofs,
                            articulation.root_physx_view.shared_metatype.dof_count,
                        )
                        # -- link related
                        self.assertEqual(
                            articulation.root_physx_view.max_links,
                            articulation.root_physx_view.shared_metatype.link_count,
                        )

                        # Simulate physics
                        for _ in range(10):
                            # perform rendering
                            sim.step()
                            # update articulation
                            articulation.update(sim.cfg.dt)

    def test_initialization_floating_base_made_fixed_base(self):
        """Test initialization for a floating-base articulation made fixed-base using schema properties."""
        for num_articulations in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_articulations=num_articulations, device=device):
                    with build_simulation_context(device=device, add_ground_plane=True, auto_add_lighting=True) as sim:
                        articulation_cfg = generate_articulation_cfg(articulation_type="anymal")
                        # Fix root link
                        articulation_cfg.spawn.articulation_props.fix_root_link = True
                        articulation, translations = generate_articulation(articulation_cfg, num_articulations, device)

                        # Check that boundedness of articulation is correct
                        self.assertEqual(ctypes.c_long.from_address(id(articulation)).value, 1)

                        # Play sim
                        sim.reset()
                        # Check if articulation is initialized
                        self.assertTrue(articulation.is_initialized)
                        # Check that is fixed base
                        self.assertTrue(articulation.is_fixed_base)
                        # Check buffers that exists and have correct shapes
                        self.assertEqual(articulation.data.root_pos_w.shape, (num_articulations, 3))
                        self.assertEqual(articulation.data.root_quat_w.shape, (num_articulations, 4))
                        self.assertEqual(articulation.data.joint_pos.shape, (num_articulations, 12))

                        # Check some internal physx data for debugging
                        # -- joint related
                        self.assertEqual(
                            articulation.root_physx_view.max_dofs,
                            articulation.root_physx_view.shared_metatype.dof_count,
                        )
                        # -- link related
                        self.assertEqual(
                            articulation.root_physx_view.max_links,
                            articulation.root_physx_view.shared_metatype.link_count,
                        )
                        # -- link names (check within articulation ordering is correct)
                        prim_path_body_names = [
                            path.split("/")[-1] for path in articulation.root_physx_view.link_paths[0]
                        ]
                        self.assertListEqual(prim_path_body_names, articulation.body_names)

                        # Simulate physics
                        for _ in range(10):
                            # perform rendering
                            sim.step()
                            # update articulation
                            articulation.update(sim.cfg.dt)

                            # check that the root is at the correct state - its default state as it is fixed base
                            default_root_state = articulation.data.default_root_state.clone()
                            default_root_state[:, :3] = default_root_state[:, :3] + translations

                            torch.testing.assert_close(articulation.data.root_state_w, default_root_state)

    def test_initialization_fixed_base_made_floating_base(self):
        """Test initialization for fixed base made floating-base using schema properties."""
        for num_articulations in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_articulations=num_articulations, device=device):
                    with build_simulation_context(device=device, add_ground_plane=True, auto_add_lighting=True) as sim:
                        articulation_cfg = generate_articulation_cfg(articulation_type="panda")
                        # Unfix root link
                        articulation_cfg.spawn.articulation_props.fix_root_link = False
                        articulation, _ = generate_articulation(articulation_cfg, num_articulations, device)

                        # Check that boundedness of articulation is correct
                        self.assertEqual(ctypes.c_long.from_address(id(articulation)).value, 1)

                        # Play sim
                        sim.reset()
                        # Check if articulation is initialized
                        self.assertTrue(articulation.is_initialized)
                        # Check that is floating base
                        self.assertFalse(articulation.is_fixed_base)
                        # Check buffers that exists and have correct shapes
                        self.assertEqual(articulation.data.root_pos_w.shape, (num_articulations, 3))
                        self.assertEqual(articulation.data.root_quat_w.shape, (num_articulations, 4))
                        self.assertEqual(articulation.data.joint_pos.shape, (num_articulations, 9))

                        # Check some internal physx data for debugging
                        # -- joint related
                        self.assertEqual(
                            articulation.root_physx_view.max_dofs,
                            articulation.root_physx_view.shared_metatype.dof_count,
                        )
                        # -- link related
                        self.assertEqual(
                            articulation.root_physx_view.max_links,
                            articulation.root_physx_view.shared_metatype.link_count,
                        )
                        # -- link names (check within articulation ordering is correct)
                        prim_path_body_names = [
                            path.split("/")[-1] for path in articulation.root_physx_view.link_paths[0]
                        ]
                        self.assertListEqual(prim_path_body_names, articulation.body_names)

                        # Simulate physics
                        for _ in range(10):
                            # perform rendering
                            sim.step()
                            # update articulation
                            articulation.update(sim.cfg.dt)

    def test_out_of_range_default_joint_pos(self):
        """Test that the default joint position from configuration is out of range."""
        for num_articulations in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_articulations=num_articulations, device=device):
                    with build_simulation_context(device=device, add_ground_plane=True, auto_add_lighting=True) as sim:
                        # Create articulation
                        articulation_cfg = generate_articulation_cfg(articulation_type="panda")
                        articulation_cfg.init_state.joint_pos = {
                            "panda_joint1": 10.0,
                            "panda_joint[2, 4]": -20.0,
                        }

                        articulation, _ = generate_articulation(articulation_cfg, num_articulations, device)

                        # Check that boundedness of articulation is correct
                        self.assertEqual(ctypes.c_long.from_address(id(articulation)).value, 1)

                        # Play sim
                        sim.reset()
                        # Check if articulation is initialized
                        self.assertFalse(articulation._is_initialized)

    def test_out_of_range_default_joint_vel(self):
        """Test that the default joint velocity from configuration is out of range."""
        with build_simulation_context(device="cuda:0", add_ground_plane=False, auto_add_lighting=True) as sim:
            # Create articulation
            articulation_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/Robot")
            articulation_cfg.init_state.joint_vel = {
                "panda_joint1": 100.0,
                "panda_joint[2, 4]": -60.0,
            }
            articulation = Articulation(articulation_cfg)

            # Check that boundedness of articulation is correct
            self.assertEqual(ctypes.c_long.from_address(id(articulation)).value, 1)

            # Play sim
            sim.reset()
            # Check if articulation is initialized
            self.assertFalse(articulation._is_initialized)

    def test_joint_limits(self):
        """Test write_joint_limits_to_sim API and when default pos falls outside of the new limits."""
        for num_articulations in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_articulations=num_articulations, device=device):
                    with build_simulation_context(device=device, add_ground_plane=True, auto_add_lighting=True) as sim:
                        # Create articulation
                        articulation_cfg = generate_articulation_cfg(articulation_type="panda")
                        articulation, _ = generate_articulation(articulation_cfg, num_articulations, device)

                        # Play sim
                        sim.reset()
                        # Check if articulation is initialized
                        self.assertTrue(articulation._is_initialized)

                        # Get current default joint pos
                        default_joint_pos = articulation._data.default_joint_pos.clone()

                        # Set new joint limits
                        limits = torch.zeros(num_articulations, articulation.num_joints, 2, device=device)
                        limits[..., 0] = (
                            torch.rand(num_articulations, articulation.num_joints, device=device) + 5.0
                        ) * -1.0
                        limits[..., 1] = torch.rand(num_articulations, articulation.num_joints, device=device) + 5.0
                        articulation.write_joint_limits_to_sim(limits)

                        # Check new limits are in place
                        torch.testing.assert_close(articulation._data.joint_limits, limits)
                        torch.testing.assert_close(articulation._data.default_joint_pos, default_joint_pos)

                        # Set new joint limits with indexing
                        env_ids = torch.arange(1, device=device)
                        joint_ids = torch.arange(2, device=device)
                        limits = torch.zeros(env_ids.shape[0], joint_ids.shape[0], 2, device=device)
                        limits[..., 0] = (torch.rand(env_ids.shape[0], joint_ids.shape[0], device=device) + 5.0) * -1.0
                        limits[..., 1] = torch.rand(env_ids.shape[0], joint_ids.shape[0], device=device) + 5.0
                        articulation.write_joint_limits_to_sim(limits, env_ids=env_ids, joint_ids=joint_ids)

                        # Check new limits are in place
                        torch.testing.assert_close(articulation._data.joint_limits[env_ids][:, joint_ids], limits)
                        torch.testing.assert_close(articulation._data.default_joint_pos, default_joint_pos)

                        # Set new joint limits that invalidate default joint pos
                        limits = torch.zeros(num_articulations, articulation.num_joints, 2, device=device)
                        limits[..., 0] = torch.rand(num_articulations, articulation.num_joints, device=device) * -0.1
                        limits[..., 1] = torch.rand(num_articulations, articulation.num_joints, device=device) * 0.1
                        articulation.write_joint_limits_to_sim(limits)

                        # Check if all values are within the bounds
                        within_bounds = (articulation._data.default_joint_pos >= limits[..., 0]) & (
                            articulation._data.default_joint_pos <= limits[..., 1]
                        )
                        self.assertTrue(torch.all(within_bounds))

                        # Set new joint limits that invalidate default joint pos with indexing
                        limits = torch.zeros(env_ids.shape[0], joint_ids.shape[0], 2, device=device)
                        limits[..., 0] = torch.rand(env_ids.shape[0], joint_ids.shape[0], device=device) * -0.1
                        limits[..., 1] = torch.rand(env_ids.shape[0], joint_ids.shape[0], device=device) * 0.1
                        articulation.write_joint_limits_to_sim(limits, env_ids=env_ids, joint_ids=joint_ids)

                        # Check if all values are within the bounds
                        within_bounds = (
                            articulation._data.default_joint_pos[env_ids][:, joint_ids] >= limits[..., 0]
                        ) & (articulation._data.default_joint_pos[env_ids][:, joint_ids] <= limits[..., 1])
                        self.assertTrue(torch.all(within_bounds))

    def test_external_force_on_single_body(self):
        """Test application of external force on the base of the articulation."""
        for num_articulations in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_articulations=num_articulations, device=device):
                    with build_simulation_context(device=device, add_ground_plane=False, auto_add_lighting=True) as sim:
                        articulation_cfg = generate_articulation_cfg(articulation_type="anymal")
                        articulation, _ = generate_articulation(articulation_cfg, num_articulations, device)
                        # Play the simulator
                        sim.reset()

                        # Find bodies to apply the force
                        body_ids, _ = articulation.find_bodies("base")
                        # Sample a large force
                        external_wrench_b = torch.zeros(articulation.num_instances, len(body_ids), 6, device=sim.device)
                        external_wrench_b[..., 1] = 1000.0

                        # Now we are ready!
                        for _ in range(5):
                            # reset root state
                            root_state = articulation.data.default_root_state.clone()

                            articulation.write_root_state_to_sim(root_state)
                            # reset dof state
                            joint_pos, joint_vel = (
                                articulation.data.default_joint_pos,
                                articulation.data.default_joint_vel,
                            )
                            articulation.write_joint_state_to_sim(joint_pos, joint_vel)
                            # reset articulation
                            articulation.reset()
                            # apply force
                            articulation.set_external_force_and_torque(
                                external_wrench_b[..., :3], external_wrench_b[..., 3:], body_ids=body_ids
                            )
                            # perform simulation
                            for _ in range(100):
                                # apply action to the articulation
                                articulation.set_joint_position_target(articulation.data.default_joint_pos.clone())
                                articulation.write_data_to_sim()
                                # perform step
                                sim.step()
                                # update buffers
                                articulation.update(sim.cfg.dt)
                            # check condition that the articulations have fallen down
                            for i in range(num_articulations):
                                self.assertLess(articulation.data.root_pos_w[i, 2].item(), 0.2)

    def test_external_force_on_multiple_bodies(self):
        """Test application of external force on the legs of the articulation."""
        for num_articulations in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_articulations=num_articulations, device=device):
                    with build_simulation_context(device=device, add_ground_plane=False, auto_add_lighting=True) as sim:
                        articulation_cfg = generate_articulation_cfg(articulation_type="anymal")
                        articulation, _ = generate_articulation(articulation_cfg, num_articulations, device)

                        # Play the simulator
                        sim.reset()

                        # Find bodies to apply the force
                        body_ids, _ = articulation.find_bodies(".*_SHANK")
                        # Sample a large force
                        external_wrench_b = torch.zeros(articulation.num_instances, len(body_ids), 6, device=sim.device)
                        external_wrench_b[..., 1] = 100.0

                        # Now we are ready!
                        for _ in range(5):
                            # reset root state
                            articulation.write_root_state_to_sim(articulation.data.default_root_state.clone())
                            # reset dof state
                            joint_pos, joint_vel = (
                                articulation.data.default_joint_pos,
                                articulation.data.default_joint_vel,
                            )
                            articulation.write_joint_state_to_sim(joint_pos, joint_vel)
                            # reset articulation
                            articulation.reset()
                            # apply force
                            articulation.set_external_force_and_torque(
                                external_wrench_b[..., :3], external_wrench_b[..., 3:], body_ids=body_ids
                            )
                            # perform simulation
                            for _ in range(100):
                                # apply action to the articulation
                                articulation.set_joint_position_target(articulation.data.default_joint_pos.clone())
                                articulation.write_data_to_sim()
                                # perform step
                                sim.step()
                                # update buffers
                                articulation.update(sim.cfg.dt)
                            # check condition
                            for i in range(num_articulations):
                                # since there is a moment applied on the articulation, the articulation should rotate
                                self.assertTrue(articulation.data.root_ang_vel_w[i, 2].item() > 0.1)

    def test_loading_gains_from_usd(self):
        """Test that gains are loaded from USD file if actuator model has them as None."""
        for num_articulations in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_articulations=num_articulations, device=device):
                    with build_simulation_context(device=device, add_ground_plane=False, auto_add_lighting=True) as sim:
                        articulation_cfg = generate_articulation_cfg(
                            articulation_type="humanoid", stiffness=None, damping=None
                        )
                        articulation, _ = generate_articulation(articulation_cfg, num_articulations, device)

                        # Play sim
                        sim.reset()

                        # Expected gains
                        # -- Stiffness values
                        expected_stiffness = {
                            ".*_waist.*": 20.0,
                            ".*_upper_arm.*": 10.0,
                            "pelvis": 10.0,
                            ".*_lower_arm": 2.0,
                            ".*_thigh:0": 10.0,
                            ".*_thigh:1": 20.0,
                            ".*_thigh:2": 10.0,
                            ".*_shin": 5.0,
                            ".*_foot.*": 2.0,
                        }
                        indices_list, _, values_list = string_utils.resolve_matching_names_values(
                            expected_stiffness, articulation.joint_names
                        )
                        expected_stiffness = torch.zeros(
                            articulation.num_instances, articulation.num_joints, device=articulation.device
                        )
                        expected_stiffness[:, indices_list] = torch.tensor(values_list, device=articulation.device)
                        # -- Damping values
                        expected_damping = {
                            ".*_waist.*": 5.0,
                            ".*_upper_arm.*": 5.0,
                            "pelvis": 5.0,
                            ".*_lower_arm": 1.0,
                            ".*_thigh:0": 5.0,
                            ".*_thigh:1": 5.0,
                            ".*_thigh:2": 5.0,
                            ".*_shin": 0.1,
                            ".*_foot.*": 1.0,
                        }
                        indices_list, _, values_list = string_utils.resolve_matching_names_values(
                            expected_damping, articulation.joint_names
                        )
                        expected_damping = torch.zeros_like(expected_stiffness)
                        expected_damping[:, indices_list] = torch.tensor(values_list, device=articulation.device)

                        # Check that gains are loaded from USD file
                        torch.testing.assert_close(articulation.actuators["body"].stiffness, expected_stiffness)
                        torch.testing.assert_close(articulation.actuators["body"].damping, expected_damping)

    def test_setting_gains_from_cfg(self):
        """Test that gains are loaded from the configuration correctly.

        Note: We purposefully give one argument as int and other as float to check that it is handled correctly.
        """
        for num_articulations in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_articulations=num_articulations, device=device):
                    with build_simulation_context(device=device, add_ground_plane=True, auto_add_lighting=True) as sim:
                        articulation_cfg = generate_articulation_cfg(articulation_type="humanoid")
                        articulation, _ = generate_articulation(
                            articulation_cfg=articulation_cfg, num_articulations=num_articulations, device=device
                        )

                        # Play sim
                        sim.reset()

                        # Expected gains
                        expected_stiffness = torch.full(
                            (articulation.num_instances, articulation.num_joints), 10.0, device=articulation.device
                        )
                        expected_damping = torch.full_like(expected_stiffness, 2.0)

                        # Check that gains are loaded from USD file
                        torch.testing.assert_close(articulation.actuators["body"].stiffness, expected_stiffness)
                        torch.testing.assert_close(articulation.actuators["body"].damping, expected_damping)

    def test_setting_gains_from_cfg_dict(self):
        """Test that gains are loaded from the configuration dictionary correctly.

        Note: We purposefully give one argument as int and other as float to check that it is handled correctly.
        """
        for num_articulations in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_articulations=num_articulations, device=device):
                    with build_simulation_context(device=device, add_ground_plane=False, auto_add_lighting=True) as sim:
                        articulation_cfg = generate_articulation_cfg(articulation_type="humanoid")
                        articulation, _ = generate_articulation(
                            articulation_cfg=articulation_cfg, num_articulations=num_articulations, device=device
                        )
                        # Play sim
                        sim.reset()

                        # Expected gains
                        expected_stiffness = torch.full(
                            (articulation.num_instances, articulation.num_joints), 10.0, device=articulation.device
                        )
                        expected_damping = torch.full_like(expected_stiffness, 2.0)

                        # Check that gains are loaded from USD file
                        torch.testing.assert_close(articulation.actuators["body"].stiffness, expected_stiffness)
                        torch.testing.assert_close(articulation.actuators["body"].damping, expected_damping)

    def test_reset(self):
        """Test that reset method works properly.

        Need to check that all actuators are reset and that forces, torques and last body velocities are reset to 0.0.

        NOTE: Currently no way to determine actuators have been reset, can leave this to actuator tests that
        implement reset method.

        """
        for num_articulations in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_articulations=num_articulations, device=device):
                    with build_simulation_context(device=device, add_ground_plane=False, auto_add_lighting=True) as sim:
                        articulation_cfg = generate_articulation_cfg(articulation_type="humanoid")
                        articulation, _ = generate_articulation(
                            articulation_cfg=articulation_cfg, num_articulations=num_articulations, device=device
                        )

                        # Play the simulator
                        sim.reset()

                        # Now we are ready!
                        # reset articulation
                        articulation.reset()

                        # Reset should zero external forces and torques
                        self.assertFalse(articulation.has_external_wrench)
                        self.assertEqual(torch.count_nonzero(articulation._external_force_b), 0)
                        self.assertEqual(torch.count_nonzero(articulation._external_torque_b), 0)

    def test_apply_joint_command(self):
        for num_articulations in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_articulations=num_articulations, device=device):
                    with build_simulation_context(
                        gravity_enabled=True, device=device, add_ground_plane=True, auto_add_lighting=True
                    ) as sim:
                        articulation_cfg = generate_articulation_cfg(articulation_type="panda")
                        articulation, _ = generate_articulation(
                            articulation_cfg=articulation_cfg, num_articulations=num_articulations, device=device
                        )

                        # Play the simulator
                        sim.reset()

                        for _ in range(100):
                            # perform step
                            sim.step()
                            # update buffers
                            articulation.update(sim.cfg.dt)

                        # reset dof state
                        joint_pos = articulation.data.default_joint_pos
                        joint_pos[:, 3] = 0.0

                        # apply action to the articulation
                        articulation.set_joint_position_target(joint_pos)
                        articulation.write_data_to_sim()

                        for _ in range(100):
                            # perform step
                            sim.step()
                            # update buffers
                            articulation.update(sim.cfg.dt)

                        # Check that current joint position is not the same as default joint position, meaning
                        # the articulation moved. We can't check that it reached it's desired joint position as the gains
                        # are not properly tuned
                        assert not torch.allclose(articulation.data.joint_pos, joint_pos)


if __name__ == "__main__":
    run_tests()
