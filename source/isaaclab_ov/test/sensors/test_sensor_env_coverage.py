# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVPhysX sensors on assets that exist in only some environments."""

import pytest

# OVRTX-only CI jobs collect the consolidated isaaclab_ov test suite without
# the optional ovphysx wheel. Skip the OVPhysX tests gracefully in that case.
pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

from isaaclab_ov.physics import OvPhysxCfg  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab import cloner  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import ArticulationCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import (  # noqa: E402
    ContactSensorCfg,
    FrameTransformerCfg,
    JointWrenchSensorCfg,
)
from isaaclab.sim import SimulationCfg, build_simulation_context  # noqa: E402
from isaaclab.terrains import TerrainImporterCfg  # noqa: E402
from isaaclab.test.utils import DeviceScope, test_devices  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402

SENSORS = ("contact", "frame_transformer", "joint_wrench")


@configclass
class _PartialAssetSceneCfg(InteractiveSceneCfg):
    """Scene with a crate and a robot; each test enables one sensor and limits the assets to some environments."""

    env_spacing = 2.0
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
    crate = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Crate",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=1.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    )
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/SimpleArticulation/revolute_articulation.usd"
        ),
        actuators={"joint": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)},
        init_state=ArticulationCfg.InitialStateCfg(pos=(1.0, 0.0, 1.0)),
    )
    contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Crate")
    frame_transformer = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Crate",
        target_frames=[FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Crate", name="crate")],
    )
    joint_wrench = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
@pytest.mark.parametrize("sensor_name", SENSORS)
def test_sensor_rejects_asset_missing_from_some_envs(device, sensor_name):
    """A sensor fails to initialize when its asset exists in only some environments.

    Its binding would cover only the environments holding the asset, so resets by environment
    index would clear other environments' rows or index past the sensor buffers.
    """
    cfg = _PartialAssetSceneCfg(num_envs=4)
    # sequential cloning places the first combination, with the crate and the robot, in environments 0 and 1
    cfg.clone_cfg.clone_combinations = [cloner.InclusionSet(assets=["crate", "robot"]), cloner.InclusionSet(assets=[])]
    for name in SENSORS:
        if name != sensor_name:
            setattr(cfg, name, None)
    asset_name = getattr(cfg, sensor_name).prim_path.rsplit("/", 1)[-1]

    sim_cfg = SimulationCfg(physics=OvPhysxCfg(), device=device)
    with build_simulation_context(device=device, sim_cfg=sim_cfg) as sim:
        # keep the scene alive: only live sensors initialize on reset
        _scene = InteractiveScene(cfg)
        with pytest.raises(RuntimeError, match=rf"/{asset_name}'.* found 2 instances.* in 4 environments"):
            sim.reset()
