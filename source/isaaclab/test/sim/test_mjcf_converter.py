# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the converter from the standalone importer wheel when installed; otherwise launch Isaac Sim."""

from isaaclab.app import AppLauncher
from isaaclab.utils.version import standalone_importers_available

# Prefer kit-less; fall back to Kit when the standalone importers are not usable.
_USE_KIT = not standalone_importers_available() and AppLauncher.is_available()
simulation_app = AppLauncher(headless=True).app if _USE_KIT else None

"""Rest everything follows."""

import os
import sys
from types import SimpleNamespace

import pytest

if _USE_KIT:
    import omni.kit.app

import newton

from pxr import Usd, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

# The Kit-less container mounts the checkout read-only, so ``usd_dir`` goes under ``tmp_path``.
pytestmark = [pytest.mark.integration, pytest.mark.isaacsim_ci, pytest.mark.kitless]

_MJCF_IMPORTER_EXTENSION = "isaacsim.asset.importer.mjcf"

# Portable MJCF for the kitless path (the Kit path uses the importer extension's bundled
# ``nv_ant.xml``). ``newton`` ships the same NVIDIA Ant model and is a base dependency of both
# environments.
_PORTABLE_MJCF = os.path.join(os.path.dirname(newton.__file__), "examples", "assets", "nv_ant.xml")


def _get_extension_path_without_enabling(extension_name: str) -> str:
    manager = omni.kit.app.get_app().get_extension_manager()
    for extension in manager.get_extensions():
        if extension["name"] == extension_name:
            return extension["path"]
    raise RuntimeError(f"Extension not found: {extension_name}")


@pytest.fixture
def sim_config(tmp_path):
    """A stage to spawn into and an Ant converter config writing under ``tmp_path``."""
    stage = sim_utils.create_new_stage()
    if _USE_KIT:
        # Kit path: create a simulation context and use the importer extension's asset.
        sim = SimulationContext(SimulationCfg(dt=0.01))
        asset_path = f"{_get_extension_path_without_enabling(_MJCF_IMPORTER_EXTENSION)}/data/mjcf/nv_ant.xml"
    else:
        # Kitless path: the converter loads the importer from the standalone wheel. Spawning and
        # inspecting prims needs a USD stage but neither physics nor Kit, so the plain stage above
        # stands in for the simulation context; use newton's bundled ``nv_ant.xml``.
        sim = SimpleNamespace(stage=stage)
        asset_path = _PORTABLE_MJCF
    config = MjcfConverterCfg(
        asset_path=asset_path, self_collision=False, force_usd_conversion=True, usd_dir=str(tmp_path / "usd")
    )
    yield sim, config
    if _USE_KIT:
        sim._disable_app_control_on_stop_handle = True  # prevent timeout
        sim.stop()
        sim.clear_instance()


def _assert_spawnable(sim, usd_path: str, prim_path: str = "/World/Robot") -> None:
    assert os.path.exists(usd_path)
    sim_utils.create_prim(prim_path, usd_path=usd_path)
    assert sim.stage.GetPrimAtPath(prim_path).IsValid()


def test_converter_enables_importer_extension(sim_config):
    """Constructing the converter makes the importer API available on the active backend.

    Under Kit that means the owning extension is enabled; kitlessly the importer module is
    resolved from the standalone wheel instead, with no extension manager involved.
    """
    _, config = sim_config
    if not _USE_KIT:
        MjcfConverter(config)
        # the importer must come from the installed wheel, not a Kit extension tree
        module_path = sys.modules[_MJCF_IMPORTER_EXTENSION].__file__
        assert module_path is not None
        assert "site-packages" in module_path, f"expected a wheel-provided importer, got {module_path}"
        return

    manager = omni.kit.app.get_app().get_extension_manager()
    if manager.is_extension_enabled(_MJCF_IMPORTER_EXTENSION):
        pytest.skip("MJCF importer extension was already enabled before constructing MjcfConverter.")

    MjcfConverter(config)

    assert manager.is_extension_enabled(_MJCF_IMPORTER_EXTENSION)


def test_lazy_conversion_cache(sim_config):
    """Conversion is skipped for an unchanged asset and config, and re-run when the config changes."""
    sim, config = sim_config
    config.force_usd_conversion = False

    converter = MjcfConverter(config)
    created = os.stat(converter.usd_path).st_mtime_ns
    _assert_spawnable(sim, converter.usd_path)

    assert os.stat(MjcfConverter(config).usd_path).st_mtime_ns == created

    config.self_collision = not config.self_collision
    assert os.stat(MjcfConverter(config).usd_path).st_mtime_ns != created


def test_self_collision(sim_config):
    """``self_collision=True`` enables self-collisions on the Newton articulation root.

    The Isaac Sim importer's ``enable_self_collision`` writes the ``newton:selfCollisionEnabled``
    attribute on prims tagged as articulation roots.
    """
    _, config = sim_config
    config.self_collision = True

    stage = Usd.Stage.Open(MjcfConverter(config).usd_path)

    articulation_roots = [
        prim
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        or prim.HasAPI("PhysicsArticulationRootAPI")
        or prim.HasAPI("NewtonArticulationRootAPI")
    ]
    assert articulation_roots, "Expected at least one articulation root in the converted USD"
    assert any(bool(prim.GetAttribute("newton:selfCollisionEnabled").Get()) for prim in articulation_roots)


def test_importer_options_produce_spawnable_usd(sim_config):
    """Pass-through importer options run end to end and still yield a spawnable articulation.

    ``nv_ant.xml`` has explicit inertial data on most bodies, so ``link_density`` only fills in
    where mass is unspecified; the observable contract is a valid, mass-carrying output.
    """
    sim, config = sim_config
    config.collision_from_visuals = True
    config.collision_type = "Convex Decomposition"
    config.merge_mesh = True
    config.import_physics_scene = True
    config.run_asset_transformer = False
    config.link_density = 500.0

    converter = MjcfConverter(config)

    stage = Usd.Stage.Open(converter.usd_path)
    assert any(prim.HasAPI(UsdPhysics.MassAPI) for prim in stage.Traverse())
    _assert_spawnable(sim, converter.usd_path)


def test_override_actuator_gains(sim_config):
    """Actuator gain overrides are written to every ``MjcActuator`` prim.

    Uses the canonical position-control encoding from the importer's ``apply_mjc_actuator_gains``.
    """
    _, config = sim_config
    kp, kd = 50.0, 5.0
    config.override_gain_type = "fixed"
    config.override_bias_type = "affine"
    config.override_gain_prm = [kp] + [0.0] * 9
    config.override_bias_prm = [0.0, -kp, -kd] + [0.0] * 7

    stage = Usd.Stage.Open(MjcfConverter(config).usd_path)

    stage.GetPrimAtPath("/ant").GetVariantSet("Physics").SetVariantSelection("mujoco")
    actuators = [prim for prim in stage.Traverse() if prim.GetTypeName() == "MjcActuator"]
    assert actuators, "Expected MjcActuator prims in nv_ant.xml output"
    for prim in actuators:
        assert prim.GetAttribute("mjc:gainType").Get() == "fixed"
        assert prim.GetAttribute("mjc:biasType").Get() == "affine"
        assert prim.GetAttribute("mjc:gainPrm").Get()[0] == pytest.approx(kp)
        assert tuple(prim.GetAttribute("mjc:biasPrm").Get()[1:3]) == pytest.approx((-kp, -kd))
