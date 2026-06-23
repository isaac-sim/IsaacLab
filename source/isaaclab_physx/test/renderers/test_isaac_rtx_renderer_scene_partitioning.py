# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RTX scene-partitioning regression tests.

These tests live in their own module on purpose. RTX scene partitioning keys per-env
geometry by a ``primvars:omni:scenePartition`` token, and the renderer retains
partition state for the lifetime of the Kit app. Building one articulation scene,
tearing it down, and building another articulation scene in the *same* app poisons that
state, so a later instanced (articulation) scene stops isolating per env even though the
USD primvars are authored correctly. Co-locating these checks with other
scene-building tests (e.g. in ``test_interactive_scene.py``) therefore makes
:func:`test_partitioning_isolates_articulation` fail, while it passes when it is the
first articulation scene the app builds. The CI harness launches every
``isaacsim_ci`` test file in its own app, so keeping these tests isolated here gives them
a clean renderer and exercises the real single-scene use case.

Per-env scene partitioning is gated behind the
``ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION`` environment variable and is off
by default. Tests that verify partitioning is *active* use the ``enable_scene_partition``
fixture, which sets the variable for the duration of the test and restores the previous
state afterwards.

Launch Isaac Sim Simulator first.
"""

from isaaclab.app import AppLauncher

# launch omniverse app — cameras are required to read back per-env RGB tiles.
simulation_app = AppLauncher(headless=True, enable_cameras=True).app

"""Rest everything follows."""

import os

import pytest
import torch
import warp as wp
from isaaclab_physx.renderers.isaac_rtx_renderer import IsaacRtxRenderer, IsaacRtxRendererCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sim import build_simulation_context
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.kuka_allegro import KUKA_ALLEGRO_CFG

_ENV_VAR = "ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION"


@pytest.fixture()
def enable_scene_partition(monkeypatch):
    """Set ``ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION=1`` for the duration of one test."""
    monkeypatch.setenv(_ENV_VAR, "1")


@pytest.mark.isaacsim_ci
def test_partitioning_disabled_by_default(monkeypatch):
    """``primvars:omni:scenePartition`` must NOT be authored when the env var is absent.

    The feature is off by default; this test confirms that :meth:`IsaacRtxRenderer.prepare_stage`
    is a no-op without ``ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION=1``.
    """
    from pxr import Usd

    monkeypatch.delenv(_ENV_VAR, raising=False)

    stage = Usd.Stage.CreateInMemory()
    world = stage.DefinePrim("/World", "Xform")  # noqa: F841
    env0 = stage.DefinePrim("/World/envs/env_0", "Xform")  # noqa: F841

    renderer = object.__new__(IsaacRtxRenderer)
    renderer.cfg = IsaacRtxRendererCfg()
    renderer.prepare_stage(stage, num_envs=1)

    prim = stage.GetPrimAtPath("/World/envs/env_0")
    assert not prim.HasAttribute("primvars:omni:scenePartition"), (
        "primvars:omni:scenePartition must not be authored when partitioning is disabled."
    )


@pytest.mark.isaacsim_ci
def test_partitioning_isolates_rigid_object(enable_scene_partition):
    """Per-env :class:`~isaaclab.assets.RigidObject` instances at unique world positions render
    as visibly different per-env tiles when RTX honors ``primvars:omni:scenePartition``."""

    @configclass
    class _Scene(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
        light = AssetBaseCfg(
            prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9))
        )
        cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube",
            spawn=sim_utils.CuboidCfg(
                size=(0.25, 0.25, 0.25),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2)),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, 1.0)),
        )
        camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            update_period=0.0,
            height=128,
            width=192,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.0, clipping_range=(0.05, 100.0)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 1.0), rot=(0.0, 0.0, 0.0, 1.0), convention="world"),
        )

    with build_simulation_context(device="cuda:0", dt=1.0 / 60.0) as sim:
        sim._app_control_on_stop_handle = None
        scene = InteractiveScene(_Scene(num_envs=4, env_spacing=0.0, replicate_physics=False))
        sim.reset()
        # one settle step so RigidObject data buffers are populated before we write into them
        sim.step()
        scene.update(sim.cfg.dt)

        cube = scene["cube"]
        device = cube.device
        rng = torch.Generator(device=device).manual_seed(1234)
        offsets = (torch.rand((scene.num_envs, 2), generator=rng, device=device) - 0.5) * 1.0
        root_pose = torch.zeros((scene.num_envs, 7), device=device)
        root_pose[:, 0] = 2.0
        root_pose[:, 1] = offsets[:, 0]
        root_pose[:, 2] = 1.0 + offsets[:, 1]
        root_pose[:, 6] = 1.0
        cube.write_root_pose_to_sim_index(
            root_pose=wp.from_torch(root_pose.contiguous(), dtype=wp.transformf),
            env_ids=wp.from_torch(torch.arange(scene.num_envs, device=device, dtype=torch.int32)),
        )
        for _ in range(4):
            sim.step()
            scene.update(sim.cfg.dt)

        rgb = scene["camera"].data.output["rgb"].torch.float()
        max_diff = max(float((rgb[0] - rgb[i]).abs().mean()) for i in range(1, rgb.shape[0]))
        assert max_diff > 3.0, (
            f"RigidObject tiles render near-identical content (max tile diff {max_diff:.3f}). "
            "Top-level partitioning may have regressed."
        )


@pytest.mark.isaacsim_ci
def test_partitioning_isolates_articulation(enable_scene_partition):
    """Per-env :class:`~isaaclab.assets.Articulation` instances driven to wildly different joint
    poses render as visibly different per-env tiles when RTX honors top-level scene partitions."""

    @configclass
    class _Scene(InteractiveSceneCfg):
        light = AssetBaseCfg(
            prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9))
        )
        robot: ArticulationCfg = KUKA_ALLEGRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            update_period=0.0,
            height=128,
            width=192,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0, focus_distance=400.0, horizontal_aperture=24.0, clipping_range=(0.05, 100.0)
            ),
            offset=CameraCfg.OffsetCfg(pos=(-1.5, 0.0, 0.7), rot=(0.0, 0.0, 0.0, 1.0), convention="world"),
        )

    with build_simulation_context(device="cuda:0", dt=1.0 / 60.0) as sim:
        sim._app_control_on_stop_handle = None
        scene = InteractiveScene(_Scene(num_envs=4, env_spacing=0.0, replicate_physics=False))
        sim.reset()

        robot = scene["robot"]
        device = robot.device
        rng = torch.Generator(device=device).manual_seed(1234)
        # Spread joint positions across the full soft joint range so each env's robot is in a
        # visibly distinct pose. When partitioning is broken, the four poses overlay in every
        # tile and the resulting ghosted robots are obvious to the eye.
        limits = robot.data.soft_joint_pos_limits.torch
        target_pos = limits[..., 0] + (limits[..., 1] - limits[..., 0]) * torch.rand(
            (scene.num_envs, robot.num_joints), generator=rng, device=device
        )
        robot.write_joint_position_to_sim_index(position=target_pos)
        robot.set_joint_position_target_index(target=target_pos)
        for _ in range(4):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.cfg.dt)

        rgb = scene["camera"].data.output["rgb"].torch.float()
        if os.environ.get("ISAACLAB_DUMP_ARTICULATION_PARTITION_IMAGES"):
            _dump_articulation_partition_images(rgb)
        max_diff = max(float((rgb[0] - rgb[i]).abs().mean()) for i in range(1, rgb.shape[0]))
        assert max_diff > 5.0, (
            f"Articulation tiles render near-identical content (max tile diff {max_diff:.3f}). "
            "Top-level partitioning may have regressed."
        )


def _dump_articulation_partition_images(rgb: torch.Tensor) -> None:
    """Save the per-environment articulation camera tiles for visual inspection."""
    import matplotlib.pyplot as plt

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "test_partitioning_isolates_articulation.png")

    rgb_np = rgb.detach().cpu().clamp(0, 255).to(torch.uint8).numpy()
    fig, axs = plt.subplots(1, rgb_np.shape[0], figsize=(3.0 * rgb_np.shape[0], 3.0), squeeze=False)
    for env_id in range(rgb_np.shape[0]):
        ax = axs[0, env_id]
        ax.imshow(rgb_np[env_id])
        ax.set_title(f"env {env_id}")
        ax.axis("off")

    fig.suptitle("Articulation partitioning camera tiles")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
