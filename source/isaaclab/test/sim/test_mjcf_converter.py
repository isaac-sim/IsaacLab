# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import os

import pytest

import omni.kit.app

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

pytestmark = pytest.mark.isaacsim_ci

_MJCF_IMPORTER_EXTENSION = "isaacsim.asset.importer.mjcf"


def _get_extension_path_without_enabling(extension_name: str) -> str:
    """Get the path for an extension without enabling it."""
    manager = omni.kit.app.get_app().get_extension_manager()
    for extension in manager.get_extensions():
        if extension["name"] == extension_name:
            return extension["path"]
    raise RuntimeError(f"Extension not found: {extension_name}")


@pytest.fixture(autouse=True)
def test_setup_teardown():
    """Setup and teardown for each test."""
    # Setup: Create a new stage
    sim_utils.create_new_stage()

    # Setup: Create simulation context
    dt = 0.01
    sim = SimulationContext(SimulationCfg(dt=dt))

    # Setup: Create MJCF config
    extension_path = _get_extension_path_without_enabling(_MJCF_IMPORTER_EXTENSION)
    config = MjcfConverterCfg(
        asset_path=f"{extension_path}/data/mjcf/nv_ant.xml",
        self_collision=False,
    )

    # Yield the resources for the test
    yield sim, config

    # Teardown: Cleanup simulation
    sim._disable_app_control_on_stop_handle = True  # prevent timeout
    sim.stop()
    sim.clear_instance()


def test_converter_enables_importer_extension(test_setup_teardown):
    """Call conversion with the MJCF importer disabled. This should enable the importer extension."""
    _, mjcf_config = test_setup_teardown

    manager = omni.kit.app.get_app().get_extension_manager()
    if manager.is_extension_enabled(_MJCF_IMPORTER_EXTENSION):
        pytest.skip("MJCF importer extension was already enabled before constructing MjcfConverter.")

    MjcfConverter(mjcf_config)

    assert manager.is_extension_enabled(_MJCF_IMPORTER_EXTENSION)


def test_no_change(test_setup_teardown):
    """Call conversion twice. This should not generate a new USD file."""
    sim, mjcf_config = test_setup_teardown

    mjcf_converter = MjcfConverter(mjcf_config)
    time_usd_file_created = os.stat(mjcf_converter.usd_path).st_mtime_ns

    # no change to config only define the usd directory
    new_config = mjcf_config
    new_config.usd_dir = mjcf_converter.usd_dir
    # convert to usd but this time in the same directory as previous step
    new_mjcf_converter = MjcfConverter(new_config)
    new_time_usd_file_created = os.stat(new_mjcf_converter.usd_path).st_mtime_ns

    assert time_usd_file_created == new_time_usd_file_created


def test_config_change(test_setup_teardown):
    """Call conversion twice but change the config in the second call. This should generate a new USD file."""
    sim, mjcf_config = test_setup_teardown

    mjcf_converter = MjcfConverter(mjcf_config)
    time_usd_file_created = os.stat(mjcf_converter.usd_path).st_mtime_ns

    # change the config
    new_config = mjcf_config
    new_config.self_collision = not mjcf_config.self_collision
    # define the usd directory
    new_config.usd_dir = mjcf_converter.usd_dir
    # convert to usd but this time in the same directory as previous step
    new_mjcf_converter = MjcfConverter(new_config)
    new_time_usd_file_created = os.stat(new_mjcf_converter.usd_path).st_mtime_ns

    assert time_usd_file_created != new_time_usd_file_created


def test_create_prim_from_usd(test_setup_teardown):
    """Call conversion and create a prim from it."""
    sim, mjcf_config = test_setup_teardown

    mjcf_converter = MjcfConverter(mjcf_config)

    prim_path = "/World/Robot"
    sim_utils.create_prim(prim_path, usd_path=mjcf_converter.usd_path)

    assert sim.stage.GetPrimAtPath(prim_path).IsValid()


@pytest.mark.isaacsim_ci
def test_self_collision(test_setup_teardown):
    """Verify that ``self_collision=True`` enables self-collisions on the Newton articulation root.

    The Isaac Sim importer's ``enable_self_collision`` writes the ``newton:selfCollisionEnabled``
    attribute on prims tagged as articulation roots (``UsdPhysics.ArticulationRootAPI``,
    ``PhysicsArticulationRootAPI``, or ``NewtonArticulationRootAPI``).
    """
    sim, config = test_setup_teardown
    test_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(test_dir, "output", "mjcf_self_collision")
    os.makedirs(output_dir, exist_ok=True)

    config.self_collision = True
    config.force_usd_conversion = True
    config.usd_dir = output_dir
    mjcf_converter = MjcfConverter(config)

    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(mjcf_converter.usd_path)

    articulation_roots = [
        prim
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        or prim.HasAPI("PhysicsArticulationRootAPI")
        or prim.HasAPI("NewtonArticulationRootAPI")
    ]
    assert articulation_roots, "Expected at least one articulation root in the converted USD"

    found_self_collision = False
    for prim in articulation_roots:
        sc_attr = prim.GetAttribute("newton:selfCollisionEnabled")
        if sc_attr and sc_attr.HasValue() and sc_attr.Get():
            found_self_collision = True
            break

    assert found_self_collision, "Expected ``newton:selfCollisionEnabled`` to be True on a Newton articulation root"


@pytest.mark.isaacsim_ci
def test_collision_from_visuals(test_setup_teardown):
    """Verify that ``collision_from_visuals=True`` runs successfully and produces a spawnable USD."""
    sim, config = test_setup_teardown
    test_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(test_dir, "output", "mjcf_collision_visuals")
    os.makedirs(output_dir, exist_ok=True)

    config.collision_from_visuals = True
    config.force_usd_conversion = True
    config.usd_dir = output_dir
    mjcf_converter = MjcfConverter(config)

    assert os.path.exists(mjcf_converter.usd_path), "USD file should exist after conversion"

    prim_path = "/World/Robot"
    sim_utils.create_prim(prim_path, usd_path=mjcf_converter.usd_path)
    assert sim.stage.GetPrimAtPath(prim_path).IsValid()


@pytest.mark.isaacsim_ci
def test_collision_type_convex_decomposition(test_setup_teardown):
    """Verify that ``collision_type='Convex Decomposition'`` runs without error."""
    sim, config = test_setup_teardown
    test_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(test_dir, "output", "mjcf_convex_decomp")
    os.makedirs(output_dir, exist_ok=True)

    config.collision_from_visuals = True
    config.collision_type = "Convex Decomposition"
    config.force_usd_conversion = True
    config.usd_dir = output_dir
    mjcf_converter = MjcfConverter(config)

    assert os.path.exists(mjcf_converter.usd_path), "USD file should exist after conversion"

    prim_path = "/World/Robot"
    sim_utils.create_prim(prim_path, usd_path=mjcf_converter.usd_path)
    assert sim.stage.GetPrimAtPath(prim_path).IsValid()


@pytest.mark.isaacsim_ci
def test_link_density(test_setup_teardown):
    """Verify that ``link_density`` applies density without errors.

    ``nv_ant.xml`` has explicit inertial data on most bodies, so density is only applied where
    mass is unspecified. This test ensures the pipeline runs and the output is spawnable.
    """
    sim, config = test_setup_teardown
    test_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(test_dir, "output", "mjcf_link_density")
    os.makedirs(output_dir, exist_ok=True)

    config.link_density = 500.0
    config.force_usd_conversion = True
    config.usd_dir = output_dir
    mjcf_converter = MjcfConverter(config)

    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(mjcf_converter.usd_path)
    mass_prims = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.MassAPI)]
    assert len(mass_prims) > 0, "Expected prims with MassAPI"

    prim_path = "/World/Robot"
    sim_utils.create_prim(prim_path, usd_path=mjcf_converter.usd_path)
    assert sim.stage.GetPrimAtPath(prim_path).IsValid()


@pytest.mark.isaacsim_ci
def test_merge_mesh(test_setup_teardown):
    """Verify that ``merge_mesh=True`` runs successfully and still produces a spawnable USD."""
    sim, config = test_setup_teardown
    test_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(test_dir, "output", "mjcf_merge_mesh")
    os.makedirs(output_dir, exist_ok=True)

    config.merge_mesh = True
    config.force_usd_conversion = True
    config.usd_dir = output_dir
    mjcf_converter = MjcfConverter(config)

    assert os.path.exists(mjcf_converter.usd_path), "USD file should exist after conversion"

    prim_path = "/World/Robot"
    sim_utils.create_prim(prim_path, usd_path=mjcf_converter.usd_path)
    assert sim.stage.GetPrimAtPath(prim_path).IsValid()


@pytest.mark.isaacsim_ci
def test_import_physics_scene(test_setup_teardown):
    """Verify that ``import_physics_scene=True`` still produces a spawnable USD."""
    sim, config = test_setup_teardown
    test_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(test_dir, "output", "mjcf_physics_scene")
    os.makedirs(output_dir, exist_ok=True)

    config.import_physics_scene = True
    config.force_usd_conversion = True
    config.usd_dir = output_dir
    mjcf_converter = MjcfConverter(config)

    assert os.path.exists(mjcf_converter.usd_path), "USD file should exist after conversion"


@pytest.mark.isaacsim_ci
def test_run_asset_transformer_disabled(test_setup_teardown):
    """Verify that ``run_asset_transformer=False`` produces a flat USD that is still spawnable."""
    sim, config = test_setup_teardown
    test_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(test_dir, "output", "mjcf_no_transformer")
    os.makedirs(output_dir, exist_ok=True)

    config.run_asset_transformer = False
    config.force_usd_conversion = True
    config.usd_dir = output_dir
    mjcf_converter = MjcfConverter(config)

    assert os.path.exists(mjcf_converter.usd_path), "USD file should exist after conversion"

    prim_path = "/World/Robot"
    sim_utils.create_prim(prim_path, usd_path=mjcf_converter.usd_path)
    assert sim.stage.GetPrimAtPath(prim_path).IsValid()


@pytest.mark.isaacsim_ci
def test_override_actuator_gains(test_setup_teardown):
    """Verify that actuator gain overrides are written to ``MjcActuator`` prims.

    ``nv_ant.xml`` defines ``MjcActuator`` prims, so setting ``override_gain_type``,
    ``override_bias_type``, ``override_gain_prm``, and ``override_bias_prm`` should update the
    corresponding ``mjc:*`` attributes on every actuator.
    """
    sim, config = test_setup_teardown
    test_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(test_dir, "output", "mjcf_actuator_gains")
    os.makedirs(output_dir, exist_ok=True)

    kp = 50.0
    kd = 5.0
    # canonical position-control encoding from the importer's ``apply_mjc_actuator_gains``
    config.override_gain_type = "fixed"
    config.override_bias_type = "affine"
    config.override_gain_prm = [kp, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    config.override_bias_prm = [0.0, -kp, -kd, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    config.force_usd_conversion = True
    config.usd_dir = output_dir
    mjcf_converter = MjcfConverter(config)

    from pxr import Usd

    stage = Usd.Stage.Open(mjcf_converter.usd_path)

    ant = stage.GetPrimAtPath("/ant")
    ant.GetVariantSet("Physics").SetVariantSelection("mujoco")
    actuator_prims = [p for p in stage.Traverse() if p.GetTypeName() == "MjcActuator"]
    assert len(actuator_prims) > 0, "Expected MjcActuator prims in nv_ant.xml output"

    for prim in actuator_prims:
        gain_type_attr = prim.GetAttribute("mjc:gainType")
        bias_type_attr = prim.GetAttribute("mjc:biasType")
        gain_prm_attr = prim.GetAttribute("mjc:gainPrm")
        bias_prm_attr = prim.GetAttribute("mjc:biasPrm")

        assert gain_type_attr and gain_type_attr.HasValue()
        assert bias_type_attr and bias_type_attr.HasValue()
        assert gain_prm_attr and gain_prm_attr.HasValue()
        assert bias_prm_attr and bias_prm_attr.HasValue()

        assert gain_type_attr.Get() == "fixed"
        assert bias_type_attr.Get() == "affine"
        assert abs(gain_prm_attr.Get()[0] - kp) < 1e-6
        assert abs(bias_prm_attr.Get()[1] - (-kp)) < 1e-6
        assert abs(bias_prm_attr.Get()[2] - (-kd)) < 1e-6
