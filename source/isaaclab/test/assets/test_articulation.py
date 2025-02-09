# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

HEADLESS = True

# launch omniverse app
app_launcher = AppLauncher(headless=HEADLESS)
simulation_app = app_launcher.app

"""Rest everything follows."""

import ctypes
import torch
import unittest
from typing import Literal

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import build_simulation_context
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from isaaclab_assets import ANYMAL_C_CFG, FRANKA_PANDA_CFG, SHADOW_HAND_CFG  # isort:skip


def generate_articulation_cfg(
    articulation_type: Literal["humanoid", "panda", "anymal", "shadow_hand", "single_joint"],
    stiffness: float | None = 10.0,
    damping: float | None = 2.0,
    vel_limit: float | None = 100.0,
    effort_limit: float | None = 400.0,
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
                    effort_limit=effort_limit,
                    velocity_limit=vel_limit,
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
    # Generate translations of 2.5 m in x for each articulation
    translations = torch.zeros(num_articulations, 3, device=device)
    translations[:, 0] = torch.arange(num_articulations) * 2.5

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
                        sim._app_control_on_stop_handle = None
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
                        sim._app_control_on_stop_handle = None
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
                        sim._app_control_on_stop_handle = None
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
                        sim._app_control_on_stop_handle = None
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
                        sim._app_control_on_stop_handle = None
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
                        sim._app_control_on_stop_handle = None
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
                        sim._app_control_on_stop_handle = None
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
                        sim._app_control_on_stop_handle = None
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
            sim._app_control_on_stop_handle = None
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
                        sim._app_control_on_stop_handle = None
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

    def test_external_force_buffer(self):
        """Test if external force buffer correctly updates in the force value is zero case."""

        num_articulations = 2
        for device in ("cuda:0", "cpu"):
            with self.subTest(num_articulations=num_articulations, device=device):
                with build_simulation_context(device=device, add_ground_plane=False, auto_add_lighting=True) as sim:
                    sim._app_control_on_stop_handle = None
                    articulation_cfg = generate_articulation_cfg(articulation_type="anymal")
                    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device)

                    # play the simulator
                    sim.reset()

                    # find bodies to apply the force
                    body_ids, _ = articulation.find_bodies("base")

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

                    # perform simulation
                    for step in range(5):
                        # initiate force tensor
                        external_wrench_b = torch.zeros(articulation.num_instances, len(body_ids), 6, device=sim.device)

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
                        articulation.set_external_force_and_torque(
                            external_wrench_b[..., :3], external_wrench_b[..., 3:], body_ids=body_ids
                        )

                        # check if the articulation's force and torque buffers are correctly updated
                        for i in range(num_articulations):
                            self.assertTrue(articulation._external_force_b[i, 0, 0].item() == force)
                            self.assertTrue(articulation._external_torque_b[i, 0, 0].item() == force)

                        # apply action to the articulation
                        articulation.set_joint_position_target(articulation.data.default_joint_pos.clone())
                        articulation.write_data_to_sim()

                        # perform step
                        sim.step()

                        # update buffers
                        articulation.update(sim.cfg.dt)

    def test_external_force_on_single_body(self):
        """Test application of external force on the base of the articulation."""
        for num_articulations in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_articulations=num_articulations, device=device):
                    with build_simulation_context(device=device, add_ground_plane=False, auto_add_lighting=True) as sim:
                        sim._app_control_on_stop_handle = None
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

                            articulation.write_root_pose_to_sim(root_state[:, :7])
                            articulation.write_root_velocity_to_sim(root_state[:, 7:])
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
                        sim._app_control_on_stop_handle = None
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
                            articulation.write_root_pose_to_sim(articulation.data.default_root_state.clone()[:, :7])
                            articulation.write_root_velocity_to_sim(articulation.data.default_root_state.clone()[:, 7:])
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
                        sim._app_control_on_stop_handle = None
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
                        sim._app_control_on_stop_handle = None
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
                        sim._app_control_on_stop_handle = None
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

    def test_setting_velocity_limits(self):
        """Test that velocity limits are loaded form the configuration correctly."""
        for num_articulations in (1, 2):
            for device in ("cuda:0", "cpu"):
                for limit in (5.0, None):
                    with self.subTest(num_articulations=num_articulations, device=device, limit=limit):
                        with build_simulation_context(
                            device=device, add_ground_plane=False, auto_add_lighting=True
                        ) as sim:
                            sim._app_control_on_stop_handle = None
                            articulation_cfg = generate_articulation_cfg(
                                articulation_type="single_joint", vel_limit=limit, effort_limit=limit
                            )
                            articulation, _ = generate_articulation(
                                articulation_cfg=articulation_cfg, num_articulations=num_articulations, device=device
                            )
                            # Play sim
                            sim.reset()

                            if limit is not None:
                                # Expected gains
                                expected_velocity_limit = torch.full(
                                    (articulation.num_instances, articulation.num_joints),
                                    limit,
                                    device=articulation.device,
                                )
                                # Check that gains are loaded from USD file
                                torch.testing.assert_close(
                                    articulation.actuators["joint"].velocity_limit, expected_velocity_limit
                                )
                                torch.testing.assert_close(
                                    articulation.data.joint_velocity_limits, expected_velocity_limit
                                )
                                torch.testing.assert_close(
                                    articulation.root_physx_view.get_dof_max_velocities().to(device),
                                    expected_velocity_limit,
                                )

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
                        sim._app_control_on_stop_handle = None
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
                        sim._app_control_on_stop_handle = None
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

    def test_body_root_state(self):
        """Test for the root_state_w property"""
        for num_articulations in (1, 2):
            # for num_articulations in ( 2,):
            for device in ("cuda:0", "cpu"):
                # for device in ("cuda:0",):
                for with_offset in [True, False]:
                    # for with_offset in [True,]:
                    with self.subTest(num_articulations=num_articulations, device=device, with_offset=with_offset):
                        with build_simulation_context(
                            device=device, add_ground_plane=False, auto_add_lighting=True
                        ) as sim:
                            sim._app_control_on_stop_handle = None
                            articulation_cfg = generate_articulation_cfg(articulation_type="single_joint")
                            articulation, env_pos = generate_articulation(articulation_cfg, num_articulations, device)
                            env_idx = torch.tensor([x for x in range(num_articulations)])
                            # Check that boundedness of articulation is correct
                            self.assertEqual(ctypes.c_long.from_address(id(articulation)).value, 1)
                            # Play sim
                            sim.reset()
                            # Check if articulation is initialized
                            self.assertTrue(articulation.is_initialized)
                            # Check that fixed base
                            self.assertTrue(articulation.is_fixed_base)

                            # change center of mass offset from link frame
                            if with_offset:
                                offset = [0.5, 0.0, 0.0]
                            else:
                                offset = [0.0, 0.0, 0.0]

                            # create com offsets
                            num_bodies = articulation.num_bodies
                            com = articulation.root_physx_view.get_coms()
                            link_offset = [1.0, 0.0, 0.0]  # the offset from CenterPivot to Arm frames
                            new_com = torch.tensor(offset, device=device).repeat(num_articulations, 1, 1)
                            com[:, 1, :3] = new_com.squeeze(-2)
                            articulation.root_physx_view.set_coms(com, env_idx)

                            # check they are set
                            torch.testing.assert_close(articulation.root_physx_view.get_coms(), com)

                            for i in range(50):
                                # perform step
                                sim.step()
                                # update buffers
                                articulation.update(sim.cfg.dt)

                                # get state properties
                                root_state_w = articulation.data.root_state_w
                                root_link_state_w = articulation.data.root_link_state_w
                                root_com_state_w = articulation.data.root_com_state_w
                                body_state_w = articulation.data.body_state_w
                                body_link_state_w = articulation.data.body_link_state_w
                                body_com_state_w = articulation.data.body_com_state_w

                                if with_offset:
                                    # get joint state
                                    joint_pos = articulation.data.joint_pos.unsqueeze(-1)
                                    joint_vel = articulation.data.joint_vel.unsqueeze(-1)

                                    # LINK state
                                    # pose
                                    torch.testing.assert_close(root_state_w[..., :7], root_link_state_w[..., :7])
                                    torch.testing.assert_close(body_state_w[..., :7], body_link_state_w[..., :7])

                                    # lin_vel arm
                                    lin_vel_gt = torch.zeros(num_articulations, num_bodies, 3, device=device)
                                    vx = -(link_offset[0]) * joint_vel * torch.sin(joint_pos)
                                    vy = torch.zeros(num_articulations, 1, 1, device=device)
                                    vz = (link_offset[0]) * joint_vel * torch.cos(joint_pos)
                                    lin_vel_gt[:, 1, :] = torch.cat([vx, vy, vz], dim=-1).squeeze(-2)

                                    # linear velocity of root link should be zero
                                    torch.testing.assert_close(
                                        lin_vel_gt[:, 0, :], root_link_state_w[..., 7:10], atol=1e-3, rtol=1e-1
                                    )
                                    # linear velocity of pendulum link should be
                                    torch.testing.assert_close(
                                        lin_vel_gt, body_link_state_w[..., 7:10], atol=1e-3, rtol=1e-1
                                    )

                                    # ang_vel
                                    torch.testing.assert_close(root_state_w[..., 10:], root_link_state_w[..., 10:])
                                    torch.testing.assert_close(body_state_w[..., 10:], body_link_state_w[..., 10:])

                                    # COM state
                                    # position and orientation shouldn't match for the _state_com_w but everything else will
                                    pos_gt = torch.zeros(num_articulations, num_bodies, 3, device=device)
                                    px = (link_offset[0] + offset[0]) * torch.cos(joint_pos)
                                    py = torch.zeros(num_articulations, 1, 1, device=device)
                                    pz = (link_offset[0] + offset[0]) * torch.sin(joint_pos)
                                    pos_gt[:, 1, :] = torch.cat([px, py, pz], dim=-1).squeeze(-2)
                                    pos_gt += env_pos.unsqueeze(-2).repeat(1, num_bodies, 1)
                                    torch.testing.assert_close(
                                        pos_gt[:, 0, :], root_com_state_w[..., :3], atol=1e-3, rtol=1e-1
                                    )
                                    torch.testing.assert_close(pos_gt, body_com_state_w[..., :3], atol=1e-3, rtol=1e-1)

                                    # orientation
                                    com_quat_b = articulation.data.com_quat_b
                                    com_quat_w = math_utils.quat_mul(body_link_state_w[..., 3:7], com_quat_b)
                                    torch.testing.assert_close(com_quat_w, body_com_state_w[..., 3:7])
                                    torch.testing.assert_close(com_quat_w[:, 0, :], root_com_state_w[..., 3:7])

                                    # linear vel, and angular vel
                                    torch.testing.assert_close(root_state_w[..., 7:], root_com_state_w[..., 7:])
                                    torch.testing.assert_close(body_state_w[..., 7:], body_com_state_w[..., 7:])
                                else:
                                    # single joint center of masses are at link frames so they will be the same
                                    torch.testing.assert_close(root_state_w, root_link_state_w)
                                    torch.testing.assert_close(root_state_w, root_com_state_w)
                                    torch.testing.assert_close(body_state_w, body_link_state_w)
                                    torch.testing.assert_close(body_state_w, body_com_state_w)

    def test_write_root_state(self):
        """Test the setters for root_state using both the link frame and center of mass as reference frame."""
        for num_articulations in (1, 2):
            for device in ("cuda:0", "cpu"):
                for with_offset in [True, False]:
                    for state_location in ("com", "link"):
                        with self.subTest(
                            num_articulations=num_articulations,
                            device=device,
                            with_offset=with_offset,
                            state_location=state_location,
                        ):
                            with build_simulation_context(
                                device=device, add_ground_plane=False, auto_add_lighting=True, gravity_enabled=False
                            ) as sim:
                                sim._app_control_on_stop_handle = None
                                articulation_cfg = generate_articulation_cfg(articulation_type="anymal")
                                articulation, env_pos = generate_articulation(
                                    articulation_cfg, num_articulations, device
                                )
                                env_idx = torch.tensor([x for x in range(num_articulations)])

                                # Play sim
                                sim.reset()

                                # change center of mass offset from link frame
                                if with_offset:
                                    offset = torch.tensor([1.0, 0.0, 0.0]).repeat(num_articulations, 1, 1)
                                else:
                                    offset = torch.tensor([0.0, 0.0, 0.0]).repeat(num_articulations, 1, 1)

                                # create com offsets
                                com = articulation.root_physx_view.get_coms()
                                new_com = offset
                                com[:, 0, :3] = new_com.squeeze(-2)
                                articulation.root_physx_view.set_coms(com, env_idx)

                                # check they are set
                                torch.testing.assert_close(articulation.root_physx_view.get_coms(), com)

                                rand_state = torch.zeros_like(articulation.data.root_state_w)
                                rand_state[..., :7] = articulation.data.default_root_state[..., :7]
                                rand_state[..., :3] += env_pos
                                # make quaternion a unit vector
                                rand_state[..., 3:7] = torch.nn.functional.normalize(rand_state[..., 3:7], dim=-1)

                                env_idx = env_idx.to(device)
                                for i in range(10):

                                    # perform step
                                    sim.step()
                                    # update buffers
                                    articulation.update(sim.cfg.dt)

                                    if state_location == "com":
                                        if i % 2 == 0:
                                            articulation.write_root_com_state_to_sim(rand_state)
                                        else:
                                            articulation.write_root_com_state_to_sim(rand_state, env_ids=env_idx)
                                    elif state_location == "link":
                                        if i % 2 == 0:
                                            articulation.write_root_link_state_to_sim(rand_state)
                                        else:
                                            articulation.write_root_link_state_to_sim(rand_state, env_ids=env_idx)

                                    if state_location == "com":
                                        torch.testing.assert_close(rand_state, articulation.data.root_com_state_w)
                                    elif state_location == "link":
                                        torch.testing.assert_close(rand_state, articulation.data.root_link_state_w)

    def test_transform_inverses(self):
        """Test if math utilities are true inverses of each other."""

        pose01 = torch.rand(1, 7)
        pose01[:, 3:7] = torch.nn.functional.normalize(pose01[..., 3:7], dim=-1)

        pose12 = torch.rand(1, 7)
        pose12[:, 3:7] = torch.nn.functional.normalize(pose12[..., 3:7], dim=-1)

        pos02, quat02 = math_utils.combine_frame_transforms(
            pose01[..., :3], pose01[..., 3:7], pose12[:, :3], pose12[:, 3:7]
        )

        pos01, quat01 = math_utils.combine_frame_transforms(
            pos02,
            quat02,
            math_utils.quat_rotate(math_utils.quat_inv(pose12[:, 3:7]), -pose12[:, :3]),
            math_utils.quat_inv(pose12[:, 3:7]),
        )
        print("")
        print(pose01)
        print(pos01, quat01)
        torch.testing.assert_close(pose01, torch.cat((pos01, quat01), dim=-1))


if __name__ == "__main__":
    run_tests()
