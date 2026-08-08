# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct scene lifecycle shared by renderer and visualizer tests."""

from __future__ import annotations

import copy
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

from isaaclab.physics import PhysicsCfg
from isaaclab.renderers import RendererCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import PinholeCameraCfg, SimulationCfg, SimulationContext, build_simulation_context
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix

CAMERA_EYE = (3.8, -4.2, 3.4)
CAMERA_TARGET = (0.05, 0.05, 1.25)
SEMANTIC_COLORS = {
    "class:ground": (94, 112, 131, 255),
    "class:robot": (246, 184, 59, 255),
    "class:moving_cube": (38, 141, 242, 255),
    "class:table": (153, 74, 33, 255),
    "class:cylinder": (239, 62, 54, 255),
    "class:sphere": (65, 201, 87, 255),
    "class:cube": (149, 76, 233, 255),
    "class:capsule": (20, 184, 166, 255),
    "class:cloth": (250, 204, 21, 255),
    "class:soft": (244, 114, 182, 255),
    "class:cone": (255, 145, 48, 255),
}


@dataclass
class RenderingScene:
    """A direct simulation/scene pair with one deterministic reset and step policy."""

    sim: SimulationContext
    scene: InteractiveScene
    camera_eye: tuple[float, float, float]
    camera_target: tuple[float, float, float]
    position_camera_from_view: bool

    preserve_fixed_articulation_roots: frozenset[str]

    @property
    def dt(self) -> float:
        return self.sim.get_physics_dt()

    @property
    def camera(self) -> Camera:
        camera = self.scene.sensors.get("camera")
        if not isinstance(camera, Camera):
            raise RuntimeError("This rendering scene was created without a camera.")
        return camera

    def reset(self) -> None:
        """Initialize backends, restore configured defaults, and synchronize without stepping."""
        self.sim.reset()
        self.scene.reset()
        self.scene.reset_to_default(
            reset_joint_targets=True,
            preserve_fixed_articulation_roots=self.preserve_fixed_articulation_roots,
        )
        self.scene.write_data_to_sim()
        self.sim.forward()
        if "camera" in self.scene.sensors and self.position_camera_from_view:
            origins = self.scene.env_origins
            eye = origins + torch.tensor(self.camera_eye, device=origins.device)
            target = origins + torch.tensor(self.camera_target, device=origins.device)
            self.camera.set_world_poses_from_view(eye, target)
        self.sim.set_camera_view(self.camera_eye, self.camera_target)
        self.scene.update(0.0)

    def step(self, *, render: bool = True) -> None:
        """Advance physics exactly once and update the scene buffers."""
        self.scene.write_data_to_sim()
        self.sim.step(render=render)
        self.scene.update(self.dt)

    def camera_outputs(self) -> tuple[dict[str, torch.Tensor], dict[str, Any] | None]:
        """Snapshot camera output tensors and metadata before a later render overwrites them."""
        data = self.camera.data
        outputs = {
            name: (value if torch.is_tensor(value) else value.torch).clone() for name, value in data.output.items()
        }
        return outputs, copy.deepcopy(data.info)

    def render_camera(self) -> None:
        """Render and refresh the camera without advancing physics."""
        self.sim.render()
        self.camera.update(0.0, force_recompute=True)

    def stabilize_camera(self, render_updates: int = 5) -> None:
        """Prime renderer history and asynchronous assets without advancing physics."""
        if getattr(self.camera.cfg.renderer_cfg, "renderer_type", None) == "isaac_rtx":
            import omni.usd

            omni.usd.get_context().reset_renderer_accumulation()
        for _ in range(render_updates):
            self.render_camera()


@contextmanager
def build_rendering_scene(
    scene_cfg: InteractiveSceneCfg,
    physics_backend: str,
    *,
    renderer: str | None = None,
    data_types: Sequence[str] = (),
    visualizer_cfgs: Any = None,
    background_color: tuple[float, float, float] | None = None,
    camera_eye: tuple[float, float, float] = CAMERA_EYE,
    camera_target: tuple[float, float, float] = CAMERA_TARGET,
    physics_cfg: PhysicsCfg | None = None,
    preserve_fixed_articulation_roots: Sequence[str] = (),
    device: str = "cuda:0",
) -> Iterator[RenderingScene]:
    """Build a caller-owned rendering scene without an RL/task environment."""
    scene_cfg = scene_cfg.copy()
    position_camera_from_view = scene_cfg.camera is None
    if renderer is not None:
        if position_camera_from_view:
            scene_cfg.camera = make_camera_cfg(
                renderer,
                data_types,
                background_color=background_color,
                camera_eye=camera_eye,
                camera_target=camera_target,
            )
        else:
            scene_cfg.camera = scene_cfg.camera.replace(
                data_types=list(data_types),
                background_color=background_color,
                renderer_cfg=make_renderer_cfg(renderer),
            )
    sim_cfg = SimulationCfg(
        dt=1.0 / 60.0,
        device=device,
        physics=make_physics_cfg(physics_backend) if physics_cfg is None else copy.deepcopy(physics_cfg),
        visualizer_cfgs=[] if visualizer_cfgs is None else visualizer_cfgs,
    )
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        if renderer is not None:
            sim.set_setting("/isaaclab/render/rtx_sensors", True)
        runtime = RenderingScene(
            sim,
            InteractiveScene(scene_cfg),
            camera_eye,
            camera_target,
            position_camera_from_view,
            frozenset(preserve_fixed_articulation_roots),
        )
        sim.register_interactive_scene(runtime.scene)
        try:
            runtime.reset()
            yield runtime
        finally:
            runtime.scene.close()


def make_camera_cfg(
    renderer: str,
    data_types: Sequence[str],
    *,
    background_color: tuple[float, float, float] | None = None,
    camera_eye: tuple[float, float, float] = CAMERA_EYE,
    camera_target: tuple[float, float, float] = CAMERA_TARGET,
) -> CameraCfg:
    """Create the one camera layout used by all golden AOV cases."""
    eyes = torch.tensor([camera_eye])
    targets = torch.tensor([camera_target])
    orientation = quat_from_matrix(create_rotation_matrix_from_view(eyes, targets))[0]
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        width=128,
        height=128,
        offset=CameraCfg.OffsetCfg(pos=camera_eye, rot=tuple(orientation.tolist()), convention="opengl"),
        update_period=0.0,
        update_latest_camera_pose=True,
        data_types=list(data_types),
        background_color=background_color,
        spawn=PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=5.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 50.0),
        ),
        renderer_cfg=make_renderer_cfg(renderer),
    )


def make_renderer_cfg(renderer: str) -> RendererCfg:
    """Resolve a renderer label without importing task preset infrastructure."""
    if renderer == "isaac_rtx":
        from isaaclab_physx.renderers import IsaacRtxRendererCfg

        cfg = IsaacRtxRendererCfg()
    elif renderer == "newton_warp":
        from isaaclab_newton.renderers import NewtonWarpRendererCfg

        cfg = NewtonWarpRendererCfg(render_order="pixel_priority")
    elif renderer == "ovrtx":
        from isaaclab_ov.renderers import OVRTXRendererCfg

        cfg = OVRTXRendererCfg(log_file_path="CON" if os.name == "nt" else "/dev/stdout")
    else:
        raise ValueError(f"Unknown renderer: {renderer!r}")
    if hasattr(cfg, "semantic_filter"):
        cfg.semantic_filter = ["class"]
    if hasattr(cfg, "semantic_segmentation_mapping"):
        cfg.semantic_segmentation_mapping = SEMANTIC_COLORS.copy()
    return cfg


def make_physics_cfg(physics_backend: str) -> PhysicsCfg:
    """Resolve a physics backend label without task or Hydra presets."""
    if physics_backend == "physx":
        from isaaclab_physx.physics import PhysxCfg

        return PhysxCfg(enable_enhanced_determinism=True, enable_external_forces_every_iteration=True)
    if physics_backend == "ovphysx":
        from isaaclab_ovphysx.physics import OvPhysxCfg

        return OvPhysxCfg(enable_enhanced_determinism=True, enable_external_forces_every_iteration=True)
    if physics_backend == "newton":
        from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

        return NewtonCfg(solver_cfg=MJWarpSolverCfg(), num_substeps=1, debug_mode=False, use_cuda_graph=True)
    raise ValueError(f"Unknown physics backend: {physics_backend!r}")
