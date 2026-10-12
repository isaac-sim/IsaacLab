# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(enable_cameras=True)

import pytest
import torch
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg, mdp
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.test.env_cfgs import make_empty_manager_based_env_cfg
from isaaclab.test.integration_scene_cfgs import CartpoleTestSceneCfg
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

pytestmark = [pytest.mark.integration, pytest.mark.rendering]

# resolution
HEIGHT = 256
WIDTH = 256

# grey default-material detection: channels within this tolerance and mean below threshold
GREY_CHANNEL_TOLERANCE = 3.0
GREY_MEAN_THRESHOLD = 85.0

# number of extra sim steps before capturing the stabilised reference frame
STABILISATION_STEPS = 5

# max allowed per-channel mean difference between first and stabilised frames
FRAME_CONSISTENCY_THRESHOLD = 15.0

# scene: dome light
DOME_LIGHT_INTENSITY = 3000.0

# scene: textured cube pose
CUBE_TRANSLATION = (0.0, 0.0, 0.6)
CUBE_ORIENTATION = (0.7071, 0.0, 0.7071, 0.0)  # rotate DexCube with its yellow "E" face texture up
CUBE_SCALE = (0.9, 0.9, 0.9)

# reset scenes: far camera clipping distance [m]
DEPTH_FAR = 10.0

# reset scenes: cube reset pose relative to its spawn pose along x [m]
CUBE_RESET_X_OFFSET = 0.15


def _is_grey(mean_rgb: torch.Tensor) -> bool:
    """Return True if mean_rgb looks like the grey default material."""
    channels_equal = (mean_rgb[1] - mean_rgb[0]).abs() < GREY_CHANNEL_TOLERANCE and (
        mean_rgb[2] - mean_rgb[0]
    ).abs() < GREY_CHANNEL_TOLERANCE
    all_low = mean_rgb.mean() < GREY_MEAN_THRESHOLD
    return bool(channels_equal and all_low)


@pytest.fixture(scope="function")
def setup_sim(device):
    """Fixture to set up and tear down the textured rendering test environment."""
    # Create a new stage
    sim_utils.create_new_stage()
    # Simulation time-step
    dt = 0.01
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=dt, device=device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # populate scene
    _populate_scene()
    yield sim, dt
    # Teardown
    sim.stop()
    sim.clear_instance()


def _assert_first_frame_textured(first_frame: torch.Tensor, stable_frame: torch.Tensor):
    """Verify that first_frame shows loaded textures and is consistent with stable_frame."""
    mean_first = first_frame.mean(dim=(0, 1))
    mean_stable = stable_frame.mean(dim=(0, 1))
    # Guard 1: not the grey default material
    assert not _is_grey(mean_first), (
        f"First frame looks like the grey default material "
        f"(mean RGB: {mean_first[0]:.1f}, {mean_first[1]:.1f}, {mean_first[2]:.1f}). "
        "The renderer's streaming wait (ensure_isaac_rtx_render_update) "
        "may not have completed texture loading before the first capture."
    )

    # Guard 2: first frame and stabilised frame are broadly consistent
    per_channel_diff = (mean_first - mean_stable).abs()
    assert per_channel_diff.max().item() < FRAME_CONSISTENCY_THRESHOLD, (
        f"First and stabilised frames differ too much per-channel "
        f"(max delta {per_channel_diff.max():.1f}, means: "
        f"first=({mean_first[0]:.1f}, {mean_first[1]:.1f}, {mean_first[2]:.1f}), "
        f"stable=({mean_stable[0]:.1f}, {mean_stable[1]:.1f}, {mean_stable[2]:.1f})). "
        "The first frame may not be fully textured."
    )


@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.isaacsim_ci
def test_first_frame_is_textured_camera(setup_sim, device):
    """First RTX frame from a USD Camera must show loaded textures, not a grey placeholder."""
    sim, dt = setup_sim
    camera_cfg = CameraCfg(
        height=HEIGHT,
        width=WIDTH,
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.75), rot=(0.0, 1.0, 0.0, 0.0), convention="ros"),
        prim_path="/World/Camera",
        update_period=0,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
    )
    # Create camera
    camera = Camera(camera_cfg)

    sim.reset()

    # The first sim step + camera update should produce textured output
    sim.step()
    camera.update(dt)
    first_frame = camera.data.output["rgb"][0].clone().to(dtype=torch.float32)

    # Let the renderer stabilise, then capture the reference frame
    for _ in range(STABILISATION_STEPS):
        sim.step()
    camera.update(dt)
    stable_frame = camera.data.output["rgb"][0].clone().to(dtype=torch.float32)

    del camera

    _assert_first_frame_textured(first_frame, stable_frame)


@configclass
class _DepthObservationsCfg(ObservationGroupCfg):
    """Camera geometry returned directly by environment reset."""

    concatenate_terms = False
    depth = ObservationTermCfg(
        func=mdp.image_depth,
        params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "distance_to_image_plane", "normalize": False},
    )


class _StaleCameraFrameError(Exception):
    """Raised when a reset camera observation does not show the reset pose."""


def _make_reset_camera_env_cfg(device: str, camera_height: float = 2.0) -> ManagerBasedEnvCfg:
    """Return a gravity-free Fabric scene whose top-down depth camera is observed on reset."""
    cfg = make_empty_manager_based_env_cfg(device=device)
    cfg.seed = 0
    cfg.decimation = 1
    cfg.sim.render_interval = 1
    cfg.sim.physics = PhysxCfg()
    cfg.sim.use_fabric = True
    cfg.sim.gravity = (0.0, 0.0, 0.0)
    cfg.sim.visualizer_cfgs = []
    cfg.num_rerenders_on_reset = 2
    cfg.observations = {"camera": _DepthObservationsCfg()}
    cfg.scene.camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        height=HEIGHT,
        width=WIDTH,
        data_types=["distance_to_image_plane"],
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, camera_height), convention="opengl"),
        spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.1, DEPTH_FAR)),
    )
    return cfg


def _add_reset_cube(cfg: ManagerBasedEnvCfg) -> None:
    """Add a cube whose reset event moves it :data:`CUBE_RESET_X_OFFSET` away from its spawn pose."""
    cfg.events.reset_cube_pose = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "pose_range": {"x": (CUBE_RESET_X_OFFSET, CUBE_RESET_X_OFFSET)},
            "velocity_range": {},
        },
    )
    cfg.scene.cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.15, 0.2),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=1.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.35, 0.0, 0.2)),
    )


def _cube_reset_pose(env: ManagerBasedEnv) -> torch.Tensor:
    """Return the world pose that the reset event writes for the cube."""
    pose = env.scene["cube"].data.default_root_pose.torch.clone()
    pose[:, :3] += env.scene.env_origins
    pose[:, 0] += CUBE_RESET_X_OFFSET
    return pose


def _moved_cube_pose(reset_pose: torch.Tensor) -> torch.Tensor:
    """Return a cube pose whose silhouette does not overlap the reset silhouette."""
    pose = reset_pose.clone()
    pose[:, 0] += 0.7
    pose[:, 3:] = torch.tensor((0.0, 0.0, 2**-0.5, 2**-0.5), device=pose.device)
    return pose


def _step_until_stable(env: ManagerBasedEnv) -> torch.Tensor:
    """Step with zero actions and return a copy of the last camera depth observation."""
    action = torch.zeros_like(env.action_manager.action)
    for _ in range(STABILISATION_STEPS):
        obs, _ = env.step(action)
    return obs["camera"]["depth"].clone()


def _silhouette(depth: torch.Tensor) -> torch.Tensor:
    """Return the pixels hit inside the camera clipping range."""
    return torch.isfinite(depth) & (depth > 0.0) & (depth < DEPTH_FAR)


def _silhouette_iou(mask: torch.Tensor, reference_mask: torch.Tensor) -> float:
    """Return the intersection over union of two silhouettes."""
    return ((mask & reference_mask).sum() / (mask | reference_mask).sum()).item()


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
@pytest.mark.isaacsim_ci
def test_env_reset_shows_reset_pose_in_camera_observation(device: str):
    """The first and subsequent resets must return the reset pose without stepping physics."""
    sim_utils.create_new_stage()
    cfg = _make_reset_camera_env_cfg(device)
    _add_reset_cube(cfg)
    env = ManagerBasedEnv(cfg)
    try:
        cube = env.scene["cube"]
        reset_pose = _cube_reset_pose(env)
        first_reset_obs, _ = env.reset()
        first_reset_depth = first_reset_obs["camera"]["depth"].clone()
        assert env.sim.get_physics_step_count() == 0
        torch.testing.assert_close(cube.data.root_link_pose_w.torch, reset_pose)

        # Establish rendered references at both poses before testing the reset boundary.
        reference_depth = _step_until_stable(env)
        # The reference frame is valid only if the cube held the reset pose while stabilizing.
        torch.testing.assert_close(cube.data.root_link_pose_w.torch, reset_pose)
        moved_pose = _moved_cube_pose(reset_pose)
        cube.write_root_pose_to_sim_index(root_pose=moved_pose)
        moved_depth = _step_until_stable(env)
        torch.testing.assert_close(cube.data.root_link_pose_w.torch, moved_pose)

        step_before_reset = env.sim.get_physics_step_count()
        reset_obs, _ = env.reset()
        reset_depth = reset_obs["camera"]["depth"]

        assert env.sim.get_physics_step_count() == step_before_reset
        torch.testing.assert_close(cube.data.root_link_pose_w.torch, reset_pose)

        # Depth silhouettes check geometry, independent of RGB exposure and temporal denoising.
        reference_mask, moved_mask = _silhouette(reference_depth), _silhouette(moved_depth)
        assert reference_mask.any() and moved_mask.any(), "The cube must be visible at both reference poses."
        assert not (reference_mask & moved_mask).any(), "Reference silhouettes must be spatially separated."
        for phase, depth in (("First", first_reset_depth), ("Later", reset_depth)):
            overlap = _silhouette_iou(_silhouette(depth), reference_mask)
            assert overlap > 0.9, f"{phase} reset camera missed the reset silhouette: IoU={overlap:.3f}"
    finally:
        env.close()


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
@pytest.mark.isaacsim_ci
def test_env_first_reset_shows_articulation_links_in_camera_observation(device: str):
    """The first reset must show articulation links at their reset joint positions without stepping physics."""
    sim_utils.create_new_stage()
    # The higher camera keeps the cart and the pole inside the image at the reset joint positions.
    cfg = _make_reset_camera_env_cfg(device, camera_height=6.0)
    cfg.scene.robot = CartpoleTestSceneCfg().robot
    # Link poses follow from joint positions through forward kinematics, not from root pose writes.
    joint_offset = 0.8
    cfg.events.reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (joint_offset, joint_offset),
            "velocity_range": (0.0, 0.0),
        },
    )
    env = ManagerBasedEnv(cfg)
    try:
        robot = env.scene["robot"]
        first_reset_obs, _ = env.reset()
        first_reset_depth = first_reset_obs["camera"]["depth"].clone()
        assert env.sim.get_physics_step_count() == 0
        torch.testing.assert_close(robot.data.joint_pos.torch, robot.data.default_joint_pos.torch + joint_offset)
        reset_link_poses = robot.data.body_link_pose_w.torch.clone()

        reference_depth = _step_until_stable(env)
        # Without gravity and drive stiffness the links hold still, so the later frame shows the reset pose.
        torch.testing.assert_close(robot.data.body_link_pose_w.torch, reset_link_poses)

        reference_mask = _silhouette(reference_depth)
        assert reference_mask.any(), "The cartpole must be visible in the reference frame."
        overlap = _silhouette_iou(_silhouette(first_reset_depth), reference_mask)
        assert overlap > 0.9, f"First reset camera missed the reset link silhouette: IoU={overlap:.3f}"
    finally:
        env.close()


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
@pytest.mark.isaacsim_ci
@pytest.mark.xfail(
    strict=True,
    raises=_StaleCameraFrameError,
    reason=(
        "PhysX Fabric limitation (#8428): after the Fabric re-attach on resume, force_update() ignores GPU tensor"
        " pose writes until the next simulate() call."
    ),
)
def test_env_reset_after_resume_shows_reset_pose_in_camera_observation(device: str):
    """A reset directly after pause and resume must return the reset pose without stepping physics."""
    sim_utils.create_new_stage()
    cfg = _make_reset_camera_env_cfg(device)
    _add_reset_cube(cfg)
    env = ManagerBasedEnv(cfg)
    try:
        cube = env.scene["cube"]
        reset_pose = _cube_reset_pose(env)
        env.reset()
        # Fabric shows the moved pose before the pause, so a stale frame cannot match the reset pose.
        cube.write_root_pose_to_sim_index(root_pose=_moved_cube_pose(reset_pose))
        _step_until_stable(env)

        env.sim.pause()
        env.sim.play()
        step_before_reset = env.sim.get_physics_step_count()
        reset_obs, _ = env.reset()
        reset_depth = reset_obs["camera"]["depth"].clone()
        assert env.sim.get_physics_step_count() == step_before_reset
        torch.testing.assert_close(cube.data.root_link_pose_w.torch, reset_pose)

        reference_depth = _step_until_stable(env)
        torch.testing.assert_close(cube.data.root_link_pose_w.torch, reset_pose)

        overlap = _silhouette_iou(_silhouette(reset_depth), _silhouette(reference_depth))
        if overlap <= 0.9:
            raise _StaleCameraFrameError(f"Reset after resume missed the reset silhouette: IoU={overlap:.3f}")
    finally:
        env.close()


"""
Helper functions.
"""


def _populate_scene():
    """Add prims to the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=DOME_LIGHT_INTENSITY, color=(1.0, 1.0, 1.0))
    cfg.func("/World/Light/Dome", cfg)
    # Textured cube rotated so yellow "E" face is visible
    sim_utils.create_prim(
        "/World/Objects/ReferenceCube",
        "Xform",
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        translation=CUBE_TRANSLATION,
        orientation=CUBE_ORIENTATION,
        scale=CUBE_SCALE,
    )
