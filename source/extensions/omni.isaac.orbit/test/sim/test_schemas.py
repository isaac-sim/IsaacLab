# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from omni.isaac.orbit.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import traceback
import unittest

import carb
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core.simulation_context import SimulationContext
from pxr import UsdPhysics

import omni.isaac.orbit.sim.schemas as schemas
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.orbit.utils.string import to_camel_case


class TestPhysicsSchema(unittest.TestCase):
    """Test fixture for checking schemas modifications through Orbit."""

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
        for cfg in [self.arti_cfg, self.rigid_cfg, self.collision_cfg, self.mass_cfg]:
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
        # validate the properties
        self._validate_properties_on_prim(
            "/World/asset_instanced", ["PhysxArticulationRootAPI", "PhysxRigidBodyAPI", "PhysicsMassAPI"]
        )

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
        # validate the properties
        self._validate_properties_on_prim(
            "/World/asset", ["PhysxArticulationAPI", "PhysxRigidBodyAPI", "PhysxCollisionAPI", "PhysicsMassAPI"]
        )

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
        self._validate_properties_on_prim("/World/cube1", ["PhysxRigidBodyAPI", "PhysxCollisionAPI", "PhysicsMassAPI"])

        # spawn another prim
        prim_utils.create_prim("/World/cube2", prim_type="Cube", translation=(1.0, 1.0, 0.62))
        # set properties on the asset and check all properties are set
        schemas.define_rigid_body_properties("/World/cube2", self.rigid_cfg)
        schemas.define_collision_properties("/World/cube2", self.collision_cfg)
        # validate the properties
        self._validate_properties_on_prim("/World/cube2", ["PhysxRigidBodyAPI", "PhysxCollisionAPI"])

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
        self._validate_properties_on_prim("/World/parent", ["PhysxArticulationRootAPI"])
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

    def _validate_properties_on_prim(self, prim_path: str, schema_names: list[str], verbose: bool = False):
        """Validate the properties on the prim.

        Note:
            Right now this function exploits the hierarchy in the asset to check the properties. This is not a
            fool-proof way of checking the properties. We should ideally check the properties on the prim itself
            and all its children.

        Args:
            prim_path: The prim name.
            schema_names: The list of schema names to validate.
            verbose: Whether to print verbose logs. Defaults to False.
        """
        # the root prim
        root_prim = prim_utils.get_prim_at_path(prim_path)
        # check articulation properties are set correctly
        if "PhysxArticulationRootAPI" in schema_names:
            for attr_name, attr_value in self.arti_cfg.__dict__.items():
                # skip names we know are not present
                if attr_name in ["func"]:
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
        # check rigid body properties are set correctly
        if "PhysxRigidBodyAPI" in schema_names:
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
        # check collision properties are set correctly
        # note: we exploit the hierarchy in the asset to check
        if "PhysxCollisionAPI" in schema_names:
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
        # check rigid body mass properties are set correctly
        # note: we exploit the hierarchy in the asset to check
        if "PhysicsMassAPI" in schema_names:
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


if __name__ == "__main__":
    try:
        unittest.main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
