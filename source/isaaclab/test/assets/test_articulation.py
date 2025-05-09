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

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.actuators import ActuatorBase, IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import build_simulation_context
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from isaaclab_assets import ANYMAL_C_CFG, FRANKA_PANDA_CFG, SHADOW_HAND_CFG  # isort:skip


def generate_articulation_cfg(
    articulation_type: str,
    stiffness: float | None = 10.0,
    damping: float | None = 2.0,
    velocity_limit: float | None = None,
    effort_limit: float | None = None,
    velocity_limit_sim: float | None = None,
    effort_limit_sim: float | None = None,
) -> ArticulationCfg:
    """Generate an articulation configuration.

    Args:
        articulation_type: Type of articulation to generate.
            It should be one of: "humanoid", "panda", "anymal", "shadow_hand", "single_joint_implicit",
            "single_joint_explicit".
        stiffness: Stiffness value for the articulation's actuators. Only currently used for "humanoid".
            Defaults to 10.0.
        damping: Damping value for the articulation's actuators. Only currently used for "humanoid".
            Defaults to 2.0.
        velocity_limit: Velocity limit for the actuators. Only currently used for "single_joint_implicit"
            and "single_joint_explicit".
        effort_limit: Effort limit for the actuators. Only currently used for "single_joint_implicit"
            and "single_joint_explicit".
        velocity_limit_sim: Velocity limit for the actuators (set into the simulation).
            Only currently used for "single_joint_implicit" and "single_joint_explicit".
        effort_limit_sim: Effort limit for the actuators (set into the simulation).
            Only currently used for "single_joint_implicit" and "single_joint_explicit".

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
    elif articulation_type == "single_joint_implicit":
        articulation_cfg = ArticulationCfg(
            # we set 80.0 default for max force because default in USD is 10e10 which makes testing annoying.
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Simple/revolute_articulation.usd",
                joint_drive_props=sim_utils.JointDrivePropertiesCfg(max_effort=80.0, max_velocity=5.0),
            ),
            actuators={
                "joint": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    effort_limit_sim=effort_limit_sim,
                    velocity_limit_sim=velocity_limit_sim,
                    effort_limit=effort_limit,
                    velocity_limit=velocity_limit,
                    stiffness=2000.0,
                    damping=100.0,
                ),
            },
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                joint_pos=({"RevoluteJoint": 1.5708}),
                rot=(0.7071055, 0.7071081, 0, 0),
            ),
        )
    elif articulation_type == "single_joint_explicit":
        # we set 80.0 default for max force because default in USD is 10e10 which makes testing annoying.
        articulation_cfg = ArticulationCfg(
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Simple/revolute_articulation.usd",
                joint_drive_props=sim_utils.JointDrivePropertiesCfg(max_effort=80.0, max_velocity=5.0),
            ),
            actuators={
                "joint": IdealPDActuatorCfg(
                    joint_names_expr=[".*"],
                    effort_limit_sim=effort_limit_sim,
                    velocity_limit_sim=velocity_limit_sim,
                    effort_limit=effort_limit,
                    velocity_limit=velocity_limit,
                    stiffness=0.0,
                    damping=10.0,
                ),
            },
        )
    else:
        raise ValueError(
            f"Invalid articulation type: {articulation_type}, valid options are 'humanoid', 'panda', 'anymal',"
            " 'shadow_hand', 'single_joint_implicit' or 'single_joint_explicit'."
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
                        # -- actuator type
                        for actuator_name, actuator in articulation.actuators.items():
                            is_implicit_model_cfg = isinstance(
                                articulation_cfg.actuators[actuator_name], ImplicitActuatorCfg
                            )
                            self.assertEqual(actuator.is_implicit_model, is_implicit_model_cfg)

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
                        # -- actuator type
                        for actuator_name, actuator in articulation.actuators.items():
                            is_implicit_model_cfg = isinstance(
                                articulation_cfg.actuators[actuator_name], ImplicitActuatorCfg
                            )
                            self.assertEqual(actuator.is_implicit_model, is_implicit_model_cfg)

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
                        # -- actuator type
                        for actuator_name, actuator in articulation.actuators.items():
                            is_implicit_model_cfg = isinstance(
                                articulation_cfg.actuators[actuator_name], ImplicitActuatorCfg
                            )
                            self.assertEqual(actuator.is_implicit_model, is_implicit_model_cfg)

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
                        articulation_cfg = generate_articulation_cfg(articulation_type="single_joint_implicit")
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
                        # -- actuator type
                        for actuator_name, actuator in articulation.actuators.items():
                            is_implicit_model_cfg = isinstance(
                                articulation_cfg.actuators[actuator_name], ImplicitActuatorCfg
                            )
                            self.assertEqual(actuator.is_implicit_model, is_implicit_model_cfg)

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

                        # check that the max force is what we set
                        physx_effort_limit = articulation.root_physx_view.get_dof_max_forces().to(device)
                        expected_joint_effort_limit = torch.full_like(
                            physx_effort_limit, articulation_cfg.spawn.joint_drive_props.max_effort
                        )
                        torch.testing.assert_close(physx_effort_limit, expected_joint_effort_limit)
                        # check that the max velocity is what we set
                        physx_vel_limit = articulation.root_physx_view.get_dof_max_velocities().to(device)
                        expected_joint_vel_limit = torch.full_like(
                            physx_vel_limit, articulation_cfg.spawn.joint_drive_props.max_velocity
                        )
                        torch.testing.assert_close(physx_vel_limit, expected_joint_vel_limit)

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
                        # -- actuator type
                        for actuator_name, actuator in articulation.actuators.items():
                            is_implicit_model_cfg = isinstance(
                                articulation_cfg.actuators[actuator_name], ImplicitActuatorCfg
                            )
                            self.assertEqual(actuator.is_implicit_model, is_implicit_model_cfg)

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
                        self.assertFalse(articulation.is_initialized)

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
            self.assertFalse(articulation.is_initialized)

    def test_joint_pos_limits(self):
        """Test write_joint_position_limit_to_sim API and when default position falls outside of the new limits."""
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
                        self.assertTrue(articulation.is_initialized)

                        # Get current default joint pos
                        default_joint_pos = articulation._data.default_joint_pos.clone()

                        # Set new joint limits
                        limits = torch.zeros(num_articulations, articulation.num_joints, 2, device=device)
                        limits[..., 0] = (
                            torch.rand(num_articulations, articulation.num_joints, device=device) + 5.0
                        ) * -1.0
                        limits[..., 1] = torch.rand(num_articulations, articulation.num_joints, device=device) + 5.0
                        articulation.write_joint_position_limit_to_sim(limits)

                        # Check new limits are in place
                        torch.testing.assert_close(articulation._data.joint_pos_limits, limits)
                        torch.testing.assert_close(articulation._data.default_joint_pos, default_joint_pos)

                        # Set new joint limits with indexing
                        env_ids = torch.arange(1, device=device)
                        joint_ids = torch.arange(2, device=device)
                        limits = torch.zeros(env_ids.shape[0], joint_ids.shape[0], 2, device=device)
                        limits[..., 0] = (torch.rand(env_ids.shape[0], joint_ids.shape[0], device=device) + 5.0) * -1.0
                        limits[..., 1] = torch.rand(env_ids.shape[0], joint_ids.shape[0], device=device) + 5.0
                        articulation.write_joint_position_limit_to_sim(limits, env_ids=env_ids, joint_ids=joint_ids)

                        # Check new limits are in place
                        torch.testing.assert_close(articulation._data.joint_pos_limits[env_ids][:, joint_ids], limits)
                        torch.testing.assert_close(articulation._data.default_joint_pos, default_joint_pos)

                        # Set new joint limits that invalidate default joint pos
                        limits = torch.zeros(num_articulations, articulation.num_joints, 2, device=device)
                        limits[..., 0] = torch.rand(num_articulations, articulation.num_joints, device=device) * -0.1
                        limits[..., 1] = torch.rand(num_articulations, articulation.num_joints, device=device) * 0.1
                        articulation.write_joint_position_limit_to_sim(limits)

                        # Check if all values are within the bounds
                        within_bounds = (articulation._data.default_joint_pos >= limits[..., 0]) & (
                            articulation._data.default_joint_pos <= limits[..., 1]
                        )
                        self.assertTrue(torch.all(within_bounds))

                        # Set new joint limits that invalidate default joint pos with indexing
                        limits = torch.zeros(env_ids.shape[0], joint_ids.shape[0], 2, device=device)
                        limits[..., 0] = torch.rand(env_ids.shape[0], joint_ids.shape[0], device=device) * -0.1
                        limits[..., 1] = torch.rand(env_ids.shape[0], joint_ids.shape[0], device=device) * 0.1
                        articulation.write_joint_position_limit_to_sim(limits, env_ids=env_ids, joint_ids=joint_ids)

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

    def test_setting_velocity_limit_implicit(self):
        """Test setting of velocity limit for implicit actuators.

        This test checks that the behavior of setting velocity limits are consistent for implicit actuators
        with previously defined behaviors.

        We do not set velocity limit to simulation when `velocity_limit` is specified. This is mainly for backwards
        compatibility. To set the velocity limit to simulation, users should set `velocity_limit_sim`.
        """
        for num_articulations in (1, 2):
            for device in ("cuda:0", "cpu"):
                for vel_limit_sim in (1e5, None):
                    for vel_limit in (1e2, None):
                        with self.subTest(
                            num_articulations=num_articulations,
                            device=device,
                            vel_limit_sim=vel_limit_sim,
                            vel_limit=vel_limit,
                        ):
                            with build_simulation_context(
                                device=device, add_ground_plane=False, auto_add_lighting=True
                            ) as sim:
                                # create simulation
                                sim._app_control_on_stop_handle = None
                                articulation_cfg = generate_articulation_cfg(
                                    articulation_type="single_joint_implicit",
                                    velocity_limit_sim=vel_limit_sim,
                                    velocity_limit=vel_limit,
                                )
                                articulation, _ = generate_articulation(
                                    articulation_cfg=articulation_cfg,
                                    num_articulations=num_articulations,
                                    device=device,
                                )
                                # Play sim
                                sim.reset()

                                if vel_limit_sim is not None and vel_limit is not None:
                                    # Case 1: during initialization, the actuator will raise a ValueError and fail to
                                    #  initialize when both these attributes are set.
                                    # note: The Exception is not caught with self.assertRaises or try-except
                                    self.assertTrue(len(articulation.actuators) == 0)
                                    continue

                                # read the values set into the simulation
                                physx_vel_limit = articulation.root_physx_view.get_dof_max_velocities().to(device)
                                # check data buffer
                                torch.testing.assert_close(articulation.data.joint_velocity_limits, physx_vel_limit)
                                # check actuator has simulation velocity limit
                                torch.testing.assert_close(
                                    articulation.actuators["joint"].velocity_limit_sim, physx_vel_limit
                                )
                                # check that both values match for velocity limit
                                torch.testing.assert_close(
                                    articulation.actuators["joint"].velocity_limit_sim,
                                    articulation.actuators["joint"].velocity_limit,
                                )

                                if vel_limit_sim is None:
                                    # Case 2: both velocity limit and velocity limit sim are not set
                                    #  This is the case where the velocity limit keeps its USD default value
                                    # Case 3: velocity limit sim is not set but velocity limit is set
                                    #   For backwards compatibility, we do not set velocity limit to simulation
                                    #   Thus, both default to USD default value.
                                    limit = articulation_cfg.spawn.joint_drive_props.max_velocity
                                else:
                                    # Case 4: only velocity limit sim is set
                                    #   In this case, the velocity limit is set to the USD value
                                    limit = vel_limit_sim

                                # check max velocity is what we set
                                expected_velocity_limit = torch.full_like(physx_vel_limit, limit)
                                torch.testing.assert_close(physx_vel_limit, expected_velocity_limit)

    def test_setting_velocity_limit_explicit(self):
        """Test setting of velocity limit for explicit actuators.

        This test checks that the behavior of setting velocity limits are consistent for explicit actuators
        with previously defined behaviors.

        Velocity limits to simulation for explicit actuators are only configured through `velocity_limit_sim`.
        """
        for num_articulations in (1, 2):
            for device in ("cuda:0", "cpu"):
                for vel_limit_sim in (1e5, None):
                    for vel_limit in (1e2, None):
                        with self.subTest(
                            num_articulations=num_articulations,
                            device=device,
                            vel_limit_sim=vel_limit_sim,
                            vel_limit=vel_limit,
                        ):
                            with build_simulation_context(
                                device=device, add_ground_plane=False, auto_add_lighting=True
                            ) as sim:
                                # create simulation
                                sim._app_control_on_stop_handle = None
                                articulation_cfg = generate_articulation_cfg(
                                    articulation_type="single_joint_explicit",
                                    velocity_limit_sim=vel_limit_sim,
                                    velocity_limit=vel_limit,
                                )
                                articulation, _ = generate_articulation(
                                    articulation_cfg=articulation_cfg,
                                    num_articulations=num_articulations,
                                    device=device,
                                )
                                # Play sim
                                sim.reset()

                                # collect limit init values
                                physx_vel_limit = articulation.root_physx_view.get_dof_max_velocities().to(device)
                                actuator_vel_limit = articulation.actuators["joint"].velocity_limit
                                actuator_vel_limit_sim = articulation.actuators["joint"].velocity_limit_sim

                                # check data buffer for joint_velocity_limits_sim
                                torch.testing.assert_close(articulation.data.joint_velocity_limits, physx_vel_limit)
                                # check actuator velocity_limit_sim is set to physx
                                torch.testing.assert_close(actuator_vel_limit_sim, physx_vel_limit)

                                if vel_limit is not None:
                                    expected_actuator_vel_limit = torch.full_like(actuator_vel_limit, vel_limit)
                                    # check actuator is set
                                    torch.testing.assert_close(actuator_vel_limit, expected_actuator_vel_limit)

                                    # check physx is not velocity_limit
                                    self.assertFalse(torch.allclose(actuator_vel_limit, physx_vel_limit))
                                else:
                                    # check actuator velocity_limit is the same as the PhysX default
                                    torch.testing.assert_close(actuator_vel_limit, physx_vel_limit)

                                # simulation velocity limit is set to USD value unless user overrides
                                if vel_limit_sim is not None:
                                    limit = vel_limit_sim
                                else:
                                    limit = articulation_cfg.spawn.joint_drive_props.max_velocity
                                # check physx is set to expected value
                                expected_vel_limit = torch.full_like(physx_vel_limit, limit)
                                torch.testing.assert_close(physx_vel_limit, expected_vel_limit)

    def test_setting_effort_limit_implicit(self):
        """Test setting of the effort limit for implicit actuators.

        In this case, the `effort_limit` and `effort_limit_sim` are treated as equivalent parameters.
        """
        for num_articulations in (1, 2):
            for device in ("cuda:0", "cpu"):
                for effort_limit_sim in (1e5, None):
                    for effort_limit in (1e2, None):
                        with self.subTest(
                            num_articulations=num_articulations,
                            device=device,
                            effort_limit_sim=effort_limit_sim,
                            effort_limit=effort_limit,
                        ):
                            with build_simulation_context(
                                device=device, add_ground_plane=False, auto_add_lighting=True
                            ) as sim:
                                # create simulation
                                sim._app_control_on_stop_handle = None
                                articulation_cfg = generate_articulation_cfg(
                                    articulation_type="single_joint_implicit",
                                    effort_limit_sim=effort_limit_sim,
                                    effort_limit=effort_limit,
                                )
                                articulation, _ = generate_articulation(
                                    articulation_cfg=articulation_cfg,
                                    num_articulations=num_articulations,
                                    device=device,
                                )
                                # Play sim
                                sim.reset()

                                if effort_limit_sim is not None and effort_limit is not None:
                                    # during initialization, the actuator will raise a ValueError and fail to
                                    # initialize. The Exception is not caught with self.assertRaises or try-except
                                    self.assertTrue(len(articulation.actuators) == 0)
                                    continue

                                # obtain the physx effort limits
                                physx_effort_limit = articulation.root_physx_view.get_dof_max_forces()
                                physx_effort_limit = physx_effort_limit.to(device=device)

                                # check that the two are equivalent
                                torch.testing.assert_close(
                                    articulation.actuators["joint"].effort_limit_sim,
                                    articulation.actuators["joint"].effort_limit,
                                )
                                torch.testing.assert_close(
                                    articulation.actuators["joint"].effort_limit_sim, physx_effort_limit
                                )

                                # decide the limit based on what is set
                                if effort_limit_sim is None and effort_limit is None:
                                    limit = articulation_cfg.spawn.joint_drive_props.max_effort
                                elif effort_limit_sim is not None and effort_limit is None:
                                    limit = effort_limit_sim
                                elif effort_limit_sim is None and effort_limit is not None:
                                    limit = effort_limit

                                # check that the max force is what we set
                                expected_effort_limit = torch.full_like(physx_effort_limit, limit)
                                torch.testing.assert_close(physx_effort_limit, expected_effort_limit)

    def test_setting_effort_limit_explicit(self):
        """Test setting of effort limit for explicit actuators.

        This test checks that the behavior of setting effort limits are consistent for explicit actuators
        with previously defined behaviors.

        Effort limits to simulation for explicit actuators are only configured through `effort_limit_sim`.
        """
        for num_articulations in (1, 2):
            for device in ("cuda:0", "cpu"):
                for effort_limit_sim in (1e5, None):
                    for effort_limit in (1e2, None):
                        with self.subTest(
                            num_articulations=num_articulations,
                            device=device,
                            effort_limit_sim=effort_limit_sim,
                            effort_limit=effort_limit,
                        ):
                            with build_simulation_context(
                                device=device, add_ground_plane=False, auto_add_lighting=True
                            ) as sim:
                                # create simulation
                                sim._app_control_on_stop_handle = None
                                articulation_cfg = generate_articulation_cfg(
                                    articulation_type="single_joint_explicit",
                                    effort_limit_sim=effort_limit_sim,
                                    effort_limit=effort_limit,
                                )
                                articulation, _ = generate_articulation(
                                    articulation_cfg=articulation_cfg,
                                    num_articulations=num_articulations,
                                    device=device,
                                )
                                # Play sim
                                sim.reset()

                                # collect limit init values
                                physx_effort_limit = articulation.root_physx_view.get_dof_max_forces().to(device)
                                actuator_effort_limit = articulation.actuators["joint"].effort_limit
                                actuator_effort_limit_sim = articulation.actuators["joint"].effort_limit_sim

                                # check actuator effort_limit_sim is set to physx
                                torch.testing.assert_close(actuator_effort_limit_sim, physx_effort_limit)

                                if effort_limit is not None:
                                    expected_actuator_effort_limit = torch.full_like(
                                        actuator_effort_limit, effort_limit
                                    )
                                    # check actuator is set
                                    torch.testing.assert_close(actuator_effort_limit, expected_actuator_effort_limit)

                                    # check physx effort limit does not match the one explicit actuator has
                                    self.assertFalse(torch.allclose(actuator_effort_limit, physx_effort_limit))
                                else:
                                    # check actuator effort_limit is the same as the PhysX default
                                    torch.testing.assert_close(actuator_effort_limit, physx_effort_limit)

                                # when using explicit actuators, the limits are set to high unless user overrides
                                if effort_limit_sim is not None:
                                    limit = effort_limit_sim
                                else:
                                    limit = ActuatorBase._DEFAULT_MAX_EFFORT_SIM  # type: ignore
                                # check physx internal value matches the expected sim value
                                expected_effort_limit = torch.full_like(physx_effort_limit, limit)
                                torch.testing.assert_close(physx_effort_limit, expected_effort_limit)

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
        """Test applying of joint position target functions correctly for a robotic arm."""
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

                        # reset joint state
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
        """Test for reading the `body_state_w` property"""
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
                            articulation_cfg = generate_articulation_cfg(articulation_type="single_joint_implicit")
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

    def test_body_incoming_joint_wrench_b_single_joint(self):
        """Test the data.body_incoming_joint_wrench_b buffer is populated correctly and statically correct for single joint."""
        for num_articulations in (2, 1):
            for device in ("cpu", "cuda:0"):
                print(num_articulations, device)
                with self.subTest(num_articulations=num_articulations, device=device):
                    with build_simulation_context(
                        gravity_enabled=True, device=device, add_ground_plane=False, auto_add_lighting=True
                    ) as sim:
                        sim._app_control_on_stop_handle = None
                        articulation_cfg = generate_articulation_cfg(articulation_type="single_joint_implicit")
                        articulation, _ = generate_articulation(
                            articulation_cfg=articulation_cfg, num_articulations=num_articulations, device=device
                        )

                        # Play the simulator
                        sim.reset()
                        # apply external force
                        external_force_vector_b = torch.zeros(
                            (num_articulations, articulation.num_bodies, 3), device=device
                        )
                        external_force_vector_b[:, 1, 1] = 10.0  # 10 N in Y direction
                        external_torque_vector_b = torch.zeros(
                            (num_articulations, articulation.num_bodies, 3), device=device
                        )
                        external_torque_vector_b[:, 1, 2] = 10.0  # 10 Nm in z direction

                        # apply action to the articulation
                        joint_pos = torch.ones_like(articulation.data.joint_pos) * 1.5708 / 2.0
                        articulation.write_joint_state_to_sim(
                            torch.ones_like(articulation.data.joint_pos), torch.zeros_like(articulation.data.joint_vel)
                        )
                        articulation.set_joint_position_target(joint_pos)
                        articulation.write_data_to_sim()
                        for _ in range(50):
                            articulation.set_external_force_and_torque(
                                forces=external_force_vector_b, torques=external_torque_vector_b
                            )
                            articulation.write_data_to_sim()
                            # perform step
                            sim.step()
                            # update buffers
                            articulation.update(sim.cfg.dt)

                            # check shape
                            self.assertEqual(
                                articulation.data.body_incoming_joint_wrench_b.shape,
                                (num_articulations, articulation.num_bodies, 6),
                            )

                        # calculate expected static
                        mass = articulation.data.default_mass
                        pos_w = articulation.data.body_pos_w
                        quat_w = articulation.data.body_quat_w

                        mass_link2 = mass[:, 1].view(num_articulations, -1)
                        gravity = (
                            torch.tensor(sim.cfg.gravity, device="cpu")
                            .repeat(num_articulations, 1)
                            .view((num_articulations, 3))
                        )

                        # NOTE: the com and link pose for single joint are colocated
                        weight_vector_w = mass_link2 * gravity
                        # expected wrench from link mass and external wrench
                        expected_wrench = torch.zeros((num_articulations, 6), device=device)
                        expected_wrench[:, :3] = math_utils.quat_apply(
                            math_utils.quat_conjugate(quat_w[:, 0, :]),
                            weight_vector_w.to(device)
                            + math_utils.quat_apply(quat_w[:, 1, :], external_force_vector_b[:, 1, :]),
                        )
                        expected_wrench[:, 3:] = math_utils.quat_apply(
                            math_utils.quat_conjugate(quat_w[:, 0, :]),
                            torch.cross(
                                pos_w[:, 1, :].to(device) - pos_w[:, 0, :].to(device),
                                weight_vector_w.to(device)
                                + math_utils.quat_apply(quat_w[:, 1, :], external_force_vector_b[:, 1, :]),
                                dim=-1,
                            )
                            + math_utils.quat_apply(quat_w[:, 1, :], external_torque_vector_b[:, 1, :]),
                        )

                        # check value of last joint wrench
                        torch.testing.assert_close(
                            expected_wrench,
                            articulation.data.body_incoming_joint_wrench_b[:, 1, :].squeeze(1),
                            atol=1e-2,
                            rtol=1e-3,
                        )


if __name__ == "__main__":
    run_tests()
