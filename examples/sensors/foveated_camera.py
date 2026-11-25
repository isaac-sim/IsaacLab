# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Explore a cluttered scene through the experimental foveated camera.

The viewer shows the camera's image, the foveated camera's cortical view, covered at launch by a window that shows the
gaze on the viewing sphere beside that view; move or resize that window to see the viewer's own.
Use WASD or the arrow keys to move the camera, and Q/E to lower or raise it.
Drag with the left mouse button, or press I/K and J/L, to steer the gaze.
Below the side panel's Camera Source selector, gaze sliders follow the gaze and set it, and on Newton a foveation
slider steps the CMF constant (cmf_a) through --cmf_a_values. Press H to show or hide the viewer controls.
Use --target to choose where the retina looks at launch.

.. code-block:: bash

    uv run --extra fovi isaaclab example foveated-camera
    uv run --extra fovi --extra ovrtx isaaclab example foveated-camera renderer=ovrtx --viz newton
    # A large CMF constant gives the matched non-foveated retina, for comparison:
    uv run --extra fovi isaaclab example foveated-camera env.scene.eye.cmf_a_deg=1000
    # Other stops for the foveation slider, with the camera's own cmf_a added:
    uv run --extra fovi isaaclab example foveated-camera --cmf_a_values 0.2,2,20
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import gymnasium as gym
import torch
import warp as wp
from fovi.sensing.projection import gaze_rotation
from isaaclab_fovi.sensors import FoveatedCameraCfg
from isaaclab_fovi.sensors.native_grid import GRID_STYLE, native_grid_exit_uv
from isaaclab_isaacsim.physics import PhysxCfg
from isaaclab_isaacsim.renderers import IsaacRtxRendererCfg
from isaaclab_newton.physics import NewtonCfg, XPBDSolverCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_ov.renderers import OVRTXRendererCfg

import isaaclab.sim as sim_utils
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass, instantiate
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR, retrieve_file_path
from isaaclab.utils.math import (
    convert_camera_frame_orientation_convention,
    create_rotation_matrix_from_view,
    quat_apply,
    quat_apply_inverse,
    quat_from_matrix,
)
from isaaclab.visualizers import SceneCameraCfg, VisualizerCfg

from isaaclab_tasks.utils import preset, setup_preset_cli
from isaaclab_tasks.utils.hydra import resolve_task_config
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

if TYPE_CHECKING:
    # Importing the runtime classes loads USD, which must come from Kit when the selected backends start it.
    from isaaclab_fovi.sensors import FoveatedCamera

    from isaaclab.visualizers import BaseVisualizer

logger = logging.getLogger("isaaclab.examples.foveated_camera")

OFFSET = (0.5, 0.5)
"""Shift of the whole scene [m], which keeps the ground plane's origin marker behind the camera."""
EYE = (OFFSET[0], OFFSET[1], 0.35)
LOOK_AT = (OFFSET[0], OFFSET[1] + 1.0, 0.0)
YCB = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned"
# Docs overview layout: name, USD, azimuth left of centre [deg], distance [m].
OBJECTS = {
    "power_drill": (f"{YCB}/035_power_drill.usd", 80.0, 1.0),
    "foam_brick": (f"{YCB}/061_foam_brick.usd", 74.0, 1.1),
    "cracker_box": (f"{YCB}/003_cracker_box.usd", 64.0, 0.8),
    "banana": (f"{YCB}/011_banana.usd", 52.0, 0.38),
    "pitcher": (f"{YCB}/019_pitcher_base.usd", 40.0, 1.0),
    "tuna_can": (f"{YCB}/007_tuna_fish_can.usd", 30.0, 0.32),
    "sugar_box": (f"{YCB}/004_sugar_box.usd", 24.0, 0.6),
    "bowl": (f"{YCB}/024_bowl.usd", 13.0, 0.85),
    "mustard_bottle": (f"{YCB}/006_mustard_bottle.usd", 2.0, 0.5),
    "master_chef_can": (f"{YCB}/002_master_chef_can.usd", -10.0, 0.95),
    "soup_can": (f"{YCB}/005_tomato_soup_can.usd", -18.0, 0.38),
    "bleach": (f"{YCB}/021_bleach_cleanser.usd", -30.0, 0.85),
    "mug": (f"{YCB}/025_mug.usd", -42.0, 0.5),
    "toy_truck": (f"{ISAACLAB_NUCLEUS_DIR}/Objects/ToyTruck/toy_truck.usd", -54.0, 1.0),
    "marker": (f"{YCB}/040_large_marker.usd", -64.0, 0.4),
    "pudding_box": (f"{YCB}/008_pudding_box.usd", -74.0, 0.7),
    "clamp": (f"{YCB}/051_large_clamp.usd", -86.0, 0.45),
}

# An ideal equidistant fisheye (an OpenCV fisheye without distortion) whose 1024 px image spans 180 degrees.
FOCAL = 512.0 / (math.pi / 2)
FISHEYE = sim_utils.PinholeCameraCfg(
    clipping_range=(0.02, 100.0),
    distortion=sim_utils.OpenCvFisheyeDistortionCfg(fx=FOCAL, fy=FOCAL, cx=511.5, cy=511.5, image_size=(1024, 1024)),
)
PREVIEW_SIZE = 256
COMPARISON_WINDOW = "Foveated camera"
"""Title of the window that shows the viewing sphere beside the foveated camera's cortical view."""


def viewing_sphere_rays(screen: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Trace the docs' oblique perspective view from behind the camera mount."""
    yaw, pitch = -0.55, 0.5
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    inverse = screen.new_tensor(((cy, sy * sp, -sy * cp), (0, cp, sp), (sy, -cy * sp, cy * cp)))
    eye = inverse @ screen.new_tensor((0, 0, -3)) + screen.new_tensor((0, 0, 0.4))
    rays = torch.cat((screen / 2.16, torch.ones_like(screen[..., :1])), dim=-1) @ inverse.T
    return eye, rays


def outline(mask: torch.Tensor) -> torch.Tensor:
    """A one-pixel inner edge of a projected region."""
    return mask & torch.nn.functional.max_pool2d((~mask)[None, None].float(), 3, 1, 1)[0, 0].bool()


def _cmf_a_values(text: str) -> tuple[float, ...]:
    """Parse comma-separated CMF constants [deg], each positive and finite."""
    try:
        values = tuple(float(value) for value in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated numbers, got {text!r}") from None
    if not all(0 < value < math.inf for value in values):
        raise argparse.ArgumentTypeError(f"expected positive, finite CMF constants, got {text!r}")
    return values


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument(
    "--max_steps", type=int, default=-1, help="Number of render frames; -1 runs until the viewer closes."
)
parser.add_argument("--target", choices=OBJECTS, help="Set the initial gaze toward this object.")
parser.add_argument(
    "--cmf_a_values",
    type=_cmf_a_values,
    metavar="DEG,DEG,...",
    help="Comma-separated CMF constants [deg] the foveation slider below the Camera Source selector steps through,"
    " with the camera's own added.",
)
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton"])
args_cli, overrides = setup_preset_cli(parser)
if args_cli.max_steps == 0 or args_cli.max_steps < -1:
    parser.error("--max_steps must be positive or -1.")


def _eye_rotation() -> torch.Tensor:
    """Return the camera mount's OpenGL orientation in the scene frame."""
    rotation = create_rotation_matrix_from_view(torch.tensor([EYE]), torch.tensor([LOOK_AT]), up_axis="Z")
    return quat_from_matrix(rotation)


def _add_clutter(scene_cfg: InteractiveSceneCfg) -> dict[str, tuple[float, float, float]]:
    """Place the docs assets on the ground and return their centres in the scene frame."""
    # USD comes from Kit when the selected backends start it, so it is imported once the simulation has launched.
    from pxr import Usd, UsdGeom

    centres = {}
    for name, (usd, azimuth, distance) in OBJECTS.items():
        stage = Usd.Stage.Open(retrieve_file_path(usd))
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        bounds = cache.ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
        yaw = math.radians(azimuth + 180.0)
        angle = math.radians(azimuth)
        x = OFFSET[0] - distance * math.sin(angle)
        y = OFFSET[1] + distance * math.cos(angle)
        height = -bounds.GetMin()[2]
        asset = AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            spawn=sim_utils.UsdFileCfg(usd_path=usd),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(x, y, height),
                rot=(0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)),
            ),
        )
        setattr(scene_cfg, name, asset)
        centres[name] = (x, y, height + bounds.GetMidpoint()[2])
    return centres


@configclass
class FoveatedSceneCfg(InteractiveSceneCfg):
    """The cluttered fisheye scene from the foveated camera docs."""

    ground = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=900.0))
    # Keep one body below the ground for Newton's model.
    anchor = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Anchor",
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 0.01, 0.01),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=0.001),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -100.0)),
    )
    eye = FoveatedCameraCfg(
        prim_path="{ENV_REGEX_NS}/Eye",
        data_types=["rgb"],
        renderer_cfg=MultiBackendRendererCfg(
            default=NewtonWarpRendererCfg(render_order="pixel_priority"),
            newton_warp=NewtonWarpRendererCfg(render_order="pixel_priority"),
            ovrtx=OVRTXRendererCfg(),
            isaacsim_rtx=IsaacRtxRendererCfg(),
        ),
        spawn=FISHEYE,
        foveated_res=128,
        mask_to_sensor_fov=True,
        offset=FoveatedCameraCfg.OffsetCfg(pos=EYE, rot=tuple(_eye_rotation()[0].tolist()), convention="opengl"),
    )


@configclass
class FoveatedExampleCfg(ManagerBasedEnvCfg):
    """A static scene; physics initializes it but never advances."""

    decimation = 1
    actions = {}
    observations = {}
    events = None
    ui_window_class_type = None
    scene: FoveatedSceneCfg = FoveatedSceneCfg(num_envs=1, env_spacing=0.0)
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        physics=preset(
            default=NewtonCfg(solver_cfg=XPBDSolverCfg(), use_cuda_graph=False, load_visual_shapes=True),
            isaacsim_rtx=PhysxCfg(),
        ),
        visualizer_cfgs=VisualizerCfg(cameras=[SceneCameraCfg(prim_path="{ENV_REGEX_NS}/Eye")]),
    )


def gaze_toward(centre: tuple[float, float, float], camera_cfg: FoveatedCameraCfg) -> tuple[float, float]:
    """Return tilt and pan in degrees in the camera's optical frame."""
    offset = camera_cfg.offset
    mount = convert_camera_frame_orientation_convention(
        torch.tensor([offset.rot]), origin=offset.convention, target="ros"
    )
    delta = torch.tensor([centre]) - torch.tensor([offset.pos])
    x, y, z = quat_apply_inverse(mount, delta)[0].tolist()
    return -math.degrees(math.atan2(y, math.hypot(x, z))), -math.degrees(math.atan2(x, z))


def camera_comparison(
    camera: FoveatedCamera, overview: FoveatedCamera
) -> tuple[Callable[[], None], Callable[[torch.Tensor], wp.array]]:
    """Draw the lens and gaze on the viewing sphere used in the docs, beside the foveated camera's cortical view."""
    size, gap = PREVIEW_SIZE, 6
    divider = torch.tensor((25, 28, 32, 255), dtype=torch.uint8, device=camera.device).expand(size, gap, -1)

    coords = overview.sampling_coords
    pixels = torch.arange(size, device=camera.device)
    center = (pixels + 0.5) * (2 / size) - 1
    screen = torch.stack(torch.meshgrid(center, center, indexing="xy"), dim=-1)
    retina_disc = screen.square().sum(-1) <= 1
    gl_to_ros = screen.new_tensor((1, -1, -1))
    roll = math.radians(overview.cfg.image_rotation_deg)
    c, s = math.cos(roll), math.sin(roll)
    image_roll = screen.new_tensor(((c, -s, 0), (s, c, 0), (0, 0, 1)))
    eye, observer_rays = viewing_sphere_rays(screen)
    reference_pos, reference_q = (value[:1].clone() for value in camera.get_world_poses(convention="opengl"))
    world_axes = quat_apply(reference_q.expand(3, -1), torch.diag(gl_to_ros))
    lens_color = screen.new_tensor((20, 112, 102))
    gaze_color = screen.new_tensor((36, 84, 157))

    def prepare() -> None:
        overview.set_world_poses(*camera.get_world_poses(convention="opengl"), convention="opengl")
        if overview.supports_batch_update:
            overview.update_batch((overview,), 0.0)

    def compose(image: torch.Tensor) -> wp.array:
        """The view for one RGB cortical-view image of the foveated camera (H, W, 3): the viewing sphere, then that
        image."""
        # The retina may change its foveation at runtime, so its coordinates are read for every frame.
        retina_coords = camera.sampling_coords
        detail = torch.cat((image, torch.full_like(image[..., :1], 255)), dim=-1)

        # Transform the fixed observer into the live camera frame, independently of active gaze.
        position, orientation = camera.get_world_poses(convention="opengl")
        axes = quat_apply_inverse(orientation[:1].expand(3, -1), world_axes) * gl_to_ros
        offset = quat_apply_inverse(reference_q, position[:1] - reference_pos)[0] * gl_to_ros
        origin = (eye - 0.22 * torch.tanh(offset / 0.4)) @ axes
        rays = observer_rays @ axes
        a = rays.square().sum(-1)
        b = (rays * origin).sum(-1)
        discriminant = b.square() - a * (origin.square().sum() - 1)
        root = discriminant.clamp_min(0).sqrt()
        near_t, far_t = (-b - root) / a, (-b + root) / a
        near = origin + near_t[..., None] * rays
        far = origin + far_t[..., None] * rays
        _, near_valid = overview.camera_model.project(((near * gl_to_ros) @ image_roll.T) * gl_to_ros)
        _, far_valid = overview.camera_model.project(((far * gl_to_ros) @ image_roll.T) * gl_to_ros)
        near_hit = (discriminant >= 0) & (near_t > 0) & near_valid
        inside = (discriminant >= 0) & (far_t > 0) & far_valid & ~near_hit
        hit = near_hit | inside

        # The overview texture uses native x/y ordering. Its published image is upright.
        uv, texture_valid = native_grid_exit_uv(far * gl_to_ros, coords)
        texture = overview.data.output["rgb"].torch[0, ..., :3]
        uv = torch.stack((uv[..., 1], 1 - uv[..., 0]), dim=-1)
        ij = (uv * size).nan_to_num().long().clamp(0, size - 1)
        rgb = texture[ij[..., 1], ij[..., 0]].float()
        sampled = inside & texture_valid

        tilt, pan = camera.gaze_angles.torch[0].deg2rad().unbind()
        direction = torch.stack((-tilt.cos() * pan.sin(), -tilt.sin(), tilt.cos() * pan.cos()))
        alignment = (far * direction).sum(-1)
        if retina_coords.style == GRID_STYLE:
            rotation = gaze_rotation(direction, convention="pan_tilt")
            grid, _ = native_grid_exit_uv((far @ rotation) * gl_to_ros, retina_coords)
            radius = torch.linalg.vector_norm(2 * grid - 1, dim=-1)
        else:
            radius = torch.acos(alignment.clamp(-1, 1)) / math.radians(retina_coords.fov / 2)
        in_retina = radius <= 1.0

        # Match the docs: an untextured near shell, a faded peripheral texture, and independent lens/gaze edges.
        texel = 0.72 * rgb + 0.28 * 255
        texel = torch.where(in_retina[..., None], texel, 0.45 * texel + 0.55 * 255)
        shell = torch.where(near_hit, 218.0, 247.0)
        full = torch.where(sampled[..., None], texel, shell[..., None])
        if retina_coords.style == GRID_STYLE:
            cells = torch.floor(grid * 8)
            lines = torch.zeros_like(hit)
            lines[:, 1:] |= (cells[:, 1:] != cells[:, :-1]).any(-1)
            lines[1:, :] |= (cells[1:, :] != cells[:-1, :]).any(-1)
            lattice = sampled & in_retina & lines
            full = torch.where(lattice[..., None], torch.lerp(full, gaze_color, 0.42), full)
        fixation = alignment.masked_fill(~sampled, -torch.inf).argmax()
        row, col = fixation // size, fixation % size
        dot = (pixels[None, :] - col).square() + (pixels[:, None] - row).square() <= 9
        gaze_edge = outline(in_retina) | (dot & (alignment > 0.999))
        full = torch.where((sampled & gaze_edge)[..., None], gaze_color, full)
        full = torch.where(outline(hit)[..., None], lens_color, full)
        full = torch.cat((full.to(torch.uint8), torch.full_like(texture[..., :1], 255)), dim=-1)

        detail = detail.permute(2, 0, 1).unsqueeze(0)
        detail = torch.nn.functional.interpolate(detail, size=(size, size), mode="nearest")[0].permute(1, 2, 0)
        detail[..., :3].masked_fill_(~retina_disc[..., None], 0)
        return wp.from_torch(torch.cat((full, divider, detail), dim=1).contiguous())

    return prepare, compose


def newton_window(visualizer):
    """The window of a Newton visualizer, which shows the example's view. Isaac Lab's Newton raycast example reaches it
    the same way."""
    return visualizer._viewer


def comparison_image_hovered(imgui, edge: float = 8.0) -> bool:
    """Whether the mouse is inside the comparison window's image: below its title bar and away from its resize edges."""
    window = imgui.get_current_context().hovered_window
    if window is None or str(window.root_window.name) != COMPARISON_WINDOW:
        return False
    root = window.root_window
    mouse = imgui.get_io().mouse_pos
    left, top = root.pos.x + edge, root.title_bar_rect().max.y
    right, bottom = root.pos.x + root.size.x - edge, root.pos.y + root.size.y - edge
    return left <= mouse.x <= right and top <= mouse.y <= bottom


def cover_view_with_comparison():
    """A side-panel callback that stretches the comparison window over the whole view beside the panel.

    It places the window once it exists, and again only when that view changes size, as while the viewer opens or when
    its window or side panel is resized; otherwise the window's size and position are the user's.
    """
    covered = [None]

    def cover(imgui) -> None:
        # The callback draws inside the side panel, whose right edge is where the view begins.
        viewport = imgui.get_main_viewport()
        left = imgui.get_window_pos().x + imgui.get_window_width()
        width = viewport.work_pos.x + viewport.work_size.x - left
        view = (left, viewport.work_pos.y, width, viewport.work_size.y)
        # Placing a window by name does nothing before its first frame.
        if view == covered[0] or imgui.internal.find_window_by_name(COMPARISON_WINDOW) is None:
            return
        covered[0] = view
        imgui.set_window_pos(COMPARISON_WINDOW, imgui.ImVec2(left, viewport.work_pos.y))
        imgui.set_window_size(COMPARISON_WINDOW, imgui.ImVec2(width, viewport.work_size.y))

    return cover


def main() -> None:
    """Render the docs scene and let the viewer drive the camera."""
    # Config-only registration reuses the standard preset and Hydra override resolver.
    gym.register(
        "FoveatedCameraExample",
        entry_point="isaaclab.envs:ManagerBasedEnv",
        kwargs={"env_cfg_entry_point": FoveatedExampleCfg},
    )
    cfg, _ = resolve_task_config("FoveatedCameraExample", None, overrides=overrides)
    if cfg.scene.eye.renderer_cfg.renderer_type == "ovrtx":
        print("Starting OVRTX. The first frame can take several minutes with a cold shader cache.", flush=True)
    with launch_simulation(cfg, args_cli):
        if args_cli.cmf_a_values is not None:
            # The panel's controls load USD, so they are imported once the simulation has launched.
            from isaaclab_fovi.sensors import ViewerGazeControls

            ViewerGazeControls.cmf_a_values = args_cli.cmf_a_values
        visualizer_types = {v.backend.visualizer_type for v in cfg.sim.visualizer_cfgs}
        if visualizer_types - {"newton"}:
            logger.warning(
                "The foveated-camera example's gaze keys and viewing-sphere window need the Newton visualizer"
                " (--viz newton). With %s, the gaze stays where it starts (see --target).",
                ", ".join(sorted(visualizer_types - {"newton"})),
            )
        sim_utils.create_new_stage()
        sim = sim_utils.SimulationContext(cfg.sim)
        centres = _add_clutter(cfg.scene)
        compare = "newton" in visualizer_types
        if compare:
            cfg.scene.eye.mask_to_sensor_fov = True
            cfg.scene.overview = cfg.scene.eye.replace(
                prim_path="{ENV_REGEX_NS}/Overview",
                foveated_res=PREVIEW_SIZE,
                foveated_style="warped_cartesian_as_grid",
                cmf_a_deg=1.0e5,
                retinal_window_fraction=1.0,
                fov_type="circular",
                radius_norm=2.0,
                pool_enabled=False,
                mask_to_sensor_fov=True,
                data_types=["rgb"],
            )
        scene = instantiate(cfg.scene)
        sim.register_interactive_scene(scene)
        sim.reset()
        camera = scene["eye"]
        angles = torch.zeros(scene.num_envs, 2, device=sim.device)
        if args_cli.target is not None:
            angles[:] = angles.new_tensor(gaze_toward(centres[args_cli.target], cfg.scene.eye))
            camera.set_gaze_angles(angles)
        viewers = [v for v in sim.visualizers if v.cfg.backend.visualizer_type == "newton"]
        if viewers:
            prepare, compose = camera_comparison(camera, scene["overview"])
            gaze = GazeSteering(camera, viewers)
            for viewer in viewers:
                viewer.register_ui_callback(cover_view_with_comparison(), position="side")

        last_gaze_update = time.monotonic()
        shown_frame = None  # The viewer's frame of the eye that the comparison window shows.
        shown_state = None  # The gaze and CMF constant it was drawn with.
        frame = 0
        while sim.is_running() and (args_cli.max_steps < 0 or frame < args_cli.max_steps):
            now = time.monotonic()
            if viewers and now - last_gaze_update >= 1 / 30:
                elapsed = min(now - last_gaze_update, 0.05)
                last_gaze_update = now
                viewer = viewers[0]
                tilt = int(viewer.is_key_down("i")) - int(viewer.is_key_down("k"))
                pan = int(viewer.is_key_down("j")) - int(viewer.is_key_down("l"))
                gaze.steer(tilt * 50 * elapsed, pan * 50 * elapsed)
            scene.update(sim.get_physics_dt())
            sim.render()
            if viewers:
                # The comparison is redrawn from the viewer's frame of the eye, only when that frame changes or the
                # gaze or foveation it overlays does. The viewer keeps showing one frame until the eye moves, its gaze
                # or foveation changes; re-rendering in between would show OVRTX's frame-to-frame sampling noise as
                # jitter.
                latest = viewers[0].current_image_view.frame.data
                state = (tuple(camera.gaze_angles.torch[0].tolist()), camera.cfg.cmf_a_deg)
                if latest is not None and (
                    shown_frame is None or state != shown_state or not torch.equal(wp.to_torch(latest), shown_frame)
                ):
                    shown_frame = wp.to_torch(latest).clone()
                    shown_state = state
                    prepare()
                    view = compose(shown_frame[..., :3])
                    for viewer in viewers:
                        newton_window(viewer).log_image(COMPARISON_WINDOW, view)
            if not sim.visualizers:
                # Without a viewer, explicitly request sensor data to render the frame.
                _ = camera.data
            frame += 1


class GazeSteering:
    """Steer the eye's gaze with the mouse and I/J/K/L.

    Left-dragging the viewer steers only the gaze, as I/J/K/L and the gaze sliders do; a drag that starts over
    the viewer's panels stays with them. The viewer itself moves the eye's mount with WASD/QE.
    """

    DRAG_DEG_PER_PIXEL = 0.15
    GAZE_LIMIT_DEG = 85.0

    def __init__(self, camera: FoveatedCamera, visualizers: list[BaseVisualizer]) -> None:
        self.camera = camera
        self.visualizers = visualizers
        self.drag = [0.0, 0.0]
        self.press_over_panel = False
        for visualizer in visualizers:
            window = getattr(newton_window(visualizer), "renderer", None)
            native = getattr(window, "window", None)
            if native is not None:
                native.push_handlers(on_mouse_press=self._on_mouse_press, on_mouse_drag=self._on_mouse_drag)

    def steer(self, tilt_deg: float, pan_deg: float) -> None:
        """Turn the gaze by (tilt, pan) [deg], starting from wherever the sliders or a drag left it."""
        # Dragging right looks right (negative pan); dragging up looks up.
        pan_deg -= self.drag[0] * self.DRAG_DEG_PER_PIXEL
        tilt_deg += self.drag[1] * self.DRAG_DEG_PER_PIXEL
        self.drag = [0.0, 0.0]
        if not tilt_deg and not pan_deg:
            return
        # The camera's stored angles are writable; set_gaze_angles then publishes them.
        gaze = self.camera.gaze_angles.torch
        gaze.add_(gaze.new_tensor((tilt_deg, pan_deg))).clamp_(-self.GAZE_LIMIT_DEG, self.GAZE_LIMIT_DEG)
        self.camera.set_gaze_angles(gaze)
        for visualizer in self.visualizers:
            # The viewer shows the eye; a new gaze redraws it even while paused.
            visualizer.current_image_view.invalidate()

    def _on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> bool | None:
        """Decide where a left press's drag goes: inside the comparison image or on the bare view it steers the gaze;
        on any other panel, or on the comparison window's title bar or edges, it stays with the panel. Other buttons
        keep their usual handling."""
        from imgui_bundle import imgui
        from pyglet import event
        from pyglet.window import mouse

        if button != mouse.LEFT:
            return None
        self.press_over_panel = imgui.get_io().want_capture_mouse
        if self.press_over_panel and comparison_image_hovered(imgui):
            self.press_over_panel = False
            # Keep ImGui from taking the press, which would drag the window instead of the gaze.
            return event.EVENT_HANDLED
        return None

    def _on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> bool | None:
        """Collect left drags that started on the view."""
        from pyglet import event
        from pyglet.window import mouse

        if not buttons & mouse.LEFT or self.press_over_panel:
            return None
        self.drag[0] += dx
        self.drag[1] += dy
        return event.EVENT_HANDLED


if __name__ == "__main__":
    main()
