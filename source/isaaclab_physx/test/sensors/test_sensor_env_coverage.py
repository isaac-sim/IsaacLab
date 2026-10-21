# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX sensors on assets that exist in only some environments."""

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation()

import pytest
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import (
    FrameTransformerCfg,
    ImuCfg,
    JointWrenchSensorCfg,
    MultiMeshRayCasterCfg,
    PvaCfg,
    RayCasterCfg,
    patterns,
)
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

SENSORS = ("frame_transformer", "imu", "joint_wrench", "pva", "ray_caster")


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
    frame_transformer = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Crate",
        target_frames=[FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Crate", name="crate")],
    )
    imu = ImuCfg(prim_path="{ENV_REGEX_NS}/Crate")
    joint_wrench = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")
    pva = PvaCfg(prim_path="{ENV_REGEX_NS}/Crate")
    ray_caster = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Crate",
        mesh_prim_paths=["/World/ground"],
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(0.0, 0.0)),
    )


@pytest.mark.parametrize("sensor_name", SENSORS)
def test_sensor_rejects_asset_missing_from_some_envs(sensor_name):
    """A sensor fails to initialize when its asset exists in only some environments.

    Its view would cover only the environments holding the asset, while its kernels and indices cover
    every environment: they would read and write past the view and sensor buffers.
    """
    cfg = _PartialAssetSceneCfg(num_envs=4)
    # sequential cloning places the first combination, with the crate and the robot, in environments 0 and 1
    cfg.clone_cfg.clone_combinations = [cloner.InclusionSet(assets=["crate", "robot"]), cloner.InclusionSet(assets=[])]
    for name in SENSORS:
        if name != sensor_name:
            setattr(cfg, name, None)
    asset_name = getattr(cfg, sensor_name).prim_path.rsplit("/", 1)[-1]

    with build_simulation_context(sim_cfg=SimulationCfg(physics=PhysxCfg())) as sim:
        # keep the scene alive: only live sensors initialize on reset
        _scene = InteractiveScene(cfg)
        with pytest.raises(RuntimeError, match=rf"/{asset_name}'.* found 2 instances.* in 4 environments"):
            sim.reset()


def test_multi_mesh_ray_caster_rejects_tracked_target_missing_from_some_envs():
    """A multi-mesh ray caster fails to initialize when a tracked target exists in only some environments.

    The target's body view would cover only the environments holding the target, while the mesh table holds a
    row for every environment: the target would be missing from its own environments or appear in others.
    """
    cfg = _PartialAssetSceneCfg(num_envs=4)
    # the robot carrying the sensor exists in every environment, the crate in environments 0 and 1
    cfg.clone_cfg.clone_combinations = [cloner.InclusionSet(assets=["crate"]), cloner.InclusionSet(assets=[])]
    for name in SENSORS:
        setattr(cfg, name, None)
    cfg.scanner = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Arm",
        mesh_prim_paths=[MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Crate")],
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(0.0, 0.0)),
    )

    with build_simulation_context(sim_cfg=SimulationCfg(physics=PhysxCfg())) as sim:
        # keep the scene alive: only live sensors initialize on reset
        _scene = InteractiveScene(cfg)
        with pytest.raises(RuntimeError, match=r"found 2 instances of '[^']*/Crate' in 4 environments"):
            sim.reset()


def test_ray_caster_rejects_frames_from_too_few_envs():
    """A ray caster fails to initialize when its frames fill the frame count from too few environments.

    Two frames in each of environments 0 and 1 make four frames for four environments: environments 1 to 3
    would read the frames of other environments.
    """
    cfg = _PartialAssetSceneCfg(num_envs=4)
    for name in SENSORS:
        setattr(cfg, name, None)
    # two static frames that exist in environments 0 and 1 only
    cfg.frame_a = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/FrameA", spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)))
    cfg.frame_b = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/FrameB", spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)))
    cfg.clone_cfg.clone_combinations = [
        cloner.InclusionSet(assets=["frame_a", "frame_b"]),
        cloner.InclusionSet(assets=[]),
    ]
    cfg.scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Frame[AB]",
        mesh_prim_paths=["/World/ground"],
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(0.0, 0.0)),
    )

    with build_simulation_context(sim_cfg=SimulationCfg(physics=PhysxCfg())) as sim:
        # keep the scene alive: only live sensors initialize on reset
        _scene = InteractiveScene(cfg)
        with pytest.raises(RuntimeError, match=r"found 2 instances in 4 environments"):
            sim.reset()
