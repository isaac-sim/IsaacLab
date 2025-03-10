# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import unittest

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.api.simulation_context import SimulationContext
from pxr import UsdPhysics

import isaaclab.sim.schemas as schemas
from isaaclab.sim.utils import find_global_fixed_joint_prim
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.string import to_camel_case


class TestPhysicsSchema(unittest.TestCase):
    """Test fixture for checking schemas modifications through Isaac Lab."""

    def setUp(self) -> None:
        """Create a blank new stage for each test."""
        # Create a new stage
        stage_utils.create_new_stage()
        # Simulation time-step
        self.dt = 0.1
        # Load kit helper
        self.sim = SimulationContext(physics_dt=self.dt, rendering_dt=self.dt, backend="numpy")
        # Set some default values for test
        self.arti_cfg = schemas.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            articulation_enabled=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
            sleep_threshold=1.0,
            stabilization_threshold=5.0,
            fix_root_link=False,
        )
        self.rigid_cfg = schemas.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=False,
            disable_gravity=False,
            linear_damping=0.1,
            angular_damping=0.5,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
            max_contact_impulse=10.0,
            enable_gyroscopic_forces=True,
            retain_accelerations=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
            sleep_threshold=1.0,
            stabilization_threshold=6.0,
        )
        self.collision_cfg = schemas.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.05,
            rest_offset=0.001,
            min_torsional_patch_radius=0.1,
            torsional_patch_radius=1.0,
        )
        self.mass_cfg = schemas.MassPropertiesCfg(mass=1.0, density=100.0)
        self.joint_cfg = schemas.JointDrivePropertiesCfg(drive_type="acceleration")

    def tearDown(self) -> None:
        """Stops simulator after each test."""
        # stop simulation
        self.sim.stop()
        self.sim.clear()
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()

    def test_valid_properties_cfg(self):
        """Test that all the config instances have non-None values.

        This is to ensure that we check that all the properties of the schema are set.
        """
        for cfg in [self.arti_cfg, self.rigid_cfg, self.collision_cfg, self.mass_cfg, self.joint_cfg]:
            # check nothing is none
            for k, v in cfg.__dict__.items():
                self.assertIsNotNone(v, f"{cfg.__class__.__name__}:{k} is None. Please make sure schemas are valid.")

    def test_modify_properties_on_invalid_prim(self):
        """Test modifying properties on a prim that does not exist."""
        # set properties
        with self.assertRaises(ValueError):
            schemas.modify_rigid_body_properties("/World/asset_xyz", self.rigid_cfg)

    def test_modify_properties_on_articulation_instanced_usd(self):
        """Test modifying properties on articulation instanced usd.

        In this case, modifying collision properties on the articulation instanced usd will fail.
        """
        # spawn asset to the stage
        asset_usd_file = f"{ISAAC_NUCLEUS_DIR}/Robots/ANYbotics/anymal_instanceable.usd"
        prim_utils.create_prim("/World/asset_instanced", usd_path=asset_usd_file, translation=(0.0, 0.0, 0.62))

        # set properties on the asset and check all properties are set
        schemas.modify_articulation_root_properties("/World/asset_instanced", self.arti_cfg)
        schemas.modify_rigid_body_properties("/World/asset_instanced", self.rigid_cfg)
        schemas.modify_mass_properties("/World/asset_instanced", self.mass_cfg)
        schemas.modify_joint_drive_properties("/World/asset_instanced", self.joint_cfg)
        # validate the properties
        self._validate_articulation_properties_on_prim("/World/asset_instanced", has_default_fixed_root=False)
        self._validate_rigid_body_properties_on_prim("/World/asset_instanced")
        self._validate_mass_properties_on_prim("/World/asset_instanced")
        self._validate_joint_drive_properties_on_prim("/World/asset_instanced")

        # make a fixed joint
        # note: for this asset, it doesn't work because the root is not a rigid body
        self.arti_cfg.fix_root_link = True
        with self.assertRaises(NotImplementedError):
            schemas.modify_articulation_root_properties("/World/asset_instanced", self.arti_cfg)

    def test_modify_properties_on_articulation_usd(self):
        """Test setting properties on articulation usd."""
        # spawn asset to the stage
        asset_usd_file = f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka.usd"
        prim_utils.create_prim("/World/asset", usd_path=asset_usd_file, translation=(0.0, 0.0, 0.62))

        # set properties on the asset and check all properties are set
        schemas.modify_articulation_root_properties("/World/asset", self.arti_cfg)
        schemas.modify_rigid_body_properties("/World/asset", self.rigid_cfg)
        schemas.modify_collision_properties("/World/asset", self.collision_cfg)
        schemas.modify_mass_properties("/World/asset", self.mass_cfg)
        schemas.modify_joint_drive_properties("/World/asset", self.joint_cfg)
        # validate the properties
        self._validate_articulation_properties_on_prim("/World/asset", has_default_fixed_root=True)
        self._validate_rigid_body_properties_on_prim("/World/asset")
        self._validate_collision_properties_on_prim("/World/asset")
        self._validate_mass_properties_on_prim("/World/asset")
        self._validate_joint_drive_properties_on_prim("/World/asset")

        # make a fixed joint
        self.arti_cfg.fix_root_link = True
        schemas.modify_articulation_root_properties("/World/asset", self.arti_cfg)
        # validate the properties
        self._validate_articulation_properties_on_prim("/World/asset", has_default_fixed_root=True)

    def test_defining_rigid_body_properties_on_prim(self):
        """Test defining rigid body properties on a prim."""
        # create a prim
        prim_utils.create_prim("/World/parent", prim_type="XForm")
        # spawn a prim
        prim_utils.create_prim("/World/cube1", prim_type="Cube", translation=(0.0, 0.0, 0.62))
        # set properties on the asset and check all properties are set
        schemas.define_rigid_body_properties("/World/cube1", self.rigid_cfg)
        schemas.define_collision_properties("/World/cube1", self.collision_cfg)
        schemas.define_mass_properties("/World/cube1", self.mass_cfg)
        # validate the properties
        self._validate_rigid_body_properties_on_prim("/World/cube1")
        self._validate_collision_properties_on_prim("/World/cube1")
        self._validate_mass_properties_on_prim("/World/cube1")

        # spawn another prim
        prim_utils.create_prim("/World/cube2", prim_type="Cube", translation=(1.0, 1.0, 0.62))
        # set properties on the asset and check all properties are set
        schemas.define_rigid_body_properties("/World/cube2", self.rigid_cfg)
        schemas.define_collision_properties("/World/cube2", self.collision_cfg)
        # validate the properties
        self._validate_rigid_body_properties_on_prim("/World/cube2")
        self._validate_collision_properties_on_prim("/World/cube2")

        # check if we can play
        self.sim.reset()
        for _ in range(100):
            self.sim.step()

    def test_defining_articulation_properties_on_prim(self):
        """Test defining articulation properties on a prim."""
        # create a parent articulation
        prim_utils.create_prim("/World/parent", prim_type="Xform")
        schemas.define_articulation_root_properties("/World/parent", self.arti_cfg)
        # validate the properties
        self._validate_articulation_properties_on_prim("/World/parent", has_default_fixed_root=False)

        # create a child articulation
        prim_utils.create_prim("/World/parent/child", prim_type="Cube", translation=(0.0, 0.0, 0.62))
        schemas.define_rigid_body_properties("/World/parent/child", self.rigid_cfg)
        schemas.define_mass_properties("/World/parent/child", self.mass_cfg)

        # check if we can play
        self.sim.reset()
        for _ in range(100):
            self.sim.step()

    """
    Helper functions.
    """

    def _validate_articulation_properties_on_prim(
        self, prim_path: str, has_default_fixed_root: False, verbose: bool = False
    ):
        """Validate the articulation properties on the prim.

        If :attr:`has_default_fixed_root` is True, then the asset already has a fixed root link. This is used to check the
        expected behavior of the fixed root link configuration.
        """
        # the root prim
        root_prim = prim_utils.get_prim_at_path(prim_path)
        # check articulation properties are set correctly
        for attr_name, attr_value in self.arti_cfg.__dict__.items():
            # skip names we know are not present
            if attr_name == "func":
                continue
            # handle fixed root link
            if attr_name == "fix_root_link" and attr_value is not None:
                # obtain the fixed joint prim
                fixed_joint_prim = find_global_fixed_joint_prim(prim_path)
                # if asset does not have a fixed root link then check if the joint is created
                if not has_default_fixed_root:
                    if attr_value:
                        self.assertIsNotNone(fixed_joint_prim)
                    else:
                        self.assertIsNone(fixed_joint_prim)
                else:
                    # check a joint exists
                    self.assertIsNotNone(fixed_joint_prim)
                    # check if the joint is enabled or disabled
                    is_enabled = fixed_joint_prim.GetJointEnabledAttr().Get()
                    self.assertEqual(is_enabled, attr_value)
                # skip the rest of the checks
                continue
            # convert attribute name in prim to cfg name
            prim_prop_name = f"physxArticulation:{to_camel_case(attr_name, to='cC')}"
            # validate the values
            self.assertAlmostEqual(
                root_prim.GetAttribute(prim_prop_name).Get(),
                attr_value,
                places=5,
                msg=f"Failed setting for {prim_prop_name}",
            )

    def _validate_rigid_body_properties_on_prim(self, prim_path: str, verbose: bool = False):
        """Validate the rigid body properties on the prim.

        Note:
            Right now this function exploits the hierarchy in the asset to check the properties. This is not a
            fool-proof way of checking the properties.
        """
        # the root prim
        root_prim = prim_utils.get_prim_at_path(prim_path)
        # check rigid body properties are set correctly
        for link_prim in root_prim.GetChildren():
            if UsdPhysics.RigidBodyAPI(link_prim):
                for attr_name, attr_value in self.rigid_cfg.__dict__.items():
                    # skip names we know are not present
                    if attr_name in ["func", "rigid_body_enabled", "kinematic_enabled"]:
                        continue
                    # convert attribute name in prim to cfg name
                    prim_prop_name = f"physxRigidBody:{to_camel_case(attr_name, to='cC')}"
                    # validate the values
                    self.assertAlmostEqual(
                        link_prim.GetAttribute(prim_prop_name).Get(),
                        attr_value,
                        places=5,
                        msg=f"Failed setting for {prim_prop_name}",
                    )
            elif verbose:
                print(f"Skipping prim {link_prim.GetPrimPath()} as it is not a rigid body.")

    def _validate_collision_properties_on_prim(self, prim_path: str, verbose: bool = False):
        """Validate the collision properties on the prim.

        Note:
            Right now this function exploits the hierarchy in the asset to check the properties. This is not a
            fool-proof way of checking the properties.
        """
        # the root prim
        root_prim = prim_utils.get_prim_at_path(prim_path)
        # check collision properties are set correctly
        for link_prim in root_prim.GetChildren():
            for mesh_prim in link_prim.GetChildren():
                if UsdPhysics.CollisionAPI(mesh_prim):
                    for attr_name, attr_value in self.collision_cfg.__dict__.items():
                        # skip names we know are not present
                        if attr_name in ["func", "collision_enabled"]:
                            continue
                        # convert attribute name in prim to cfg name
                        prim_prop_name = f"physxCollision:{to_camel_case(attr_name, to='cC')}"
                        # validate the values
                        self.assertAlmostEqual(
                            mesh_prim.GetAttribute(prim_prop_name).Get(),
                            attr_value,
                            places=5,
                            msg=f"Failed setting for {prim_prop_name}",
                        )
                elif verbose:
                    print(f"Skipping prim {mesh_prim.GetPrimPath()} as it is not a collision mesh.")

    def _validate_mass_properties_on_prim(self, prim_path: str, verbose: bool = False):
        """Validate the mass properties on the prim.

        Note:
            Right now this function exploits the hierarchy in the asset to check the properties. This is not a
            fool-proof way of checking the properties.
        """
        # the root prim
        root_prim = prim_utils.get_prim_at_path(prim_path)
        # check rigid body mass properties are set correctly
        for link_prim in root_prim.GetChildren():
            if UsdPhysics.MassAPI(link_prim):
                for attr_name, attr_value in self.mass_cfg.__dict__.items():
                    # skip names we know are not present
                    if attr_name in ["func"]:
                        continue
                    # print(link_prim.GetProperties())
                    prim_prop_name = f"physics:{to_camel_case(attr_name, to='cC')}"
                    # validate the values
                    self.assertAlmostEqual(
                        link_prim.GetAttribute(prim_prop_name).Get(),
                        attr_value,
                        places=5,
                        msg=f"Failed setting for {prim_prop_name}",
                    )
            elif verbose:
                print(f"Skipping prim {link_prim.GetPrimPath()} as it is not a mass api.")

    def _validate_joint_drive_properties_on_prim(self, prim_path: str, verbose: bool = False):
        """Validate the mass properties on the prim.

        Note:
            Right now this function exploits the hierarchy in the asset to check the properties. This is not a
            fool-proof way of checking the properties.
        """
        # the root prim
        root_prim = prim_utils.get_prim_at_path(prim_path)
        # check joint drive properties are set correctly
        for link_prim in root_prim.GetAllChildren():
            for joint_prim in link_prim.GetChildren():
                if joint_prim.IsA(UsdPhysics.PrismaticJoint) or joint_prim.IsA(UsdPhysics.RevoluteJoint):
                    # check it has drive API
                    self.assertTrue(joint_prim.HasAPI(UsdPhysics.DriveAPI))
                    # iterate over the joint properties
                    for attr_name, attr_value in self.joint_cfg.__dict__.items():
                        # skip names we know are not present
                        if attr_name == "func":
                            continue
                        # manually check joint type
                        if attr_name == "drive_type":
                            if joint_prim.IsA(UsdPhysics.PrismaticJoint):
                                prim_attr_name = "drive:linear:physics:type"
                            elif joint_prim.IsA(UsdPhysics.RevoluteJoint):
                                prim_attr_name = "drive:angular:physics:type"
                            else:
                                raise ValueError(f"Unknown joint type for prim {joint_prim.GetPrimPath()}")
                            # check the value
                            self.assertEqual(attr_value, joint_prim.GetAttribute(prim_attr_name).Get())
                            continue
                elif verbose:
                    print(f"Skipping prim {joint_prim.GetPrimPath()} as it is not a joint drive api.")


if __name__ == "__main__":
    run_tests()
