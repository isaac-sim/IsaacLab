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

from isaaclab.renderers import RendererCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import PinholeCameraCfg, SimulationCfg, SimulationContext, build_simulation_context
from isaaclab.test.integration_scene_cfgs import RenderingTestSceneCfg

CAMERA_EYE = (3.8, -4.2, 3.4)
CAMERA_TARGET = (0.05, 0.05, 1.25)
SEMANTIC_COLORS = {
    "class:ground": (94, 112, 131, 255),
    "class:robot": (246, 184, 59, 255),
    "class:moving_cube": (38, 141, 242, 255),
    "class:table": (153, 74, 33, 255),
    "class:cylinder": (239, 62, 54, 255),
    "class:sphere": (65, 201, 87, 255),
}


@dataclass
class RenderingScene:
    """A direct simulation/scene pair with one deterministic reset and step policy."""

    sim: SimulationContext
    scene: InteractiveScene

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
        self.scene.reset_to_default(reset_joint_targets=True)
        self.scene.write_data_to_sim()
        self.sim.forward()
        if "camera" in self.scene.sensors:
            origins = self.scene.env_origins
            eye = origins + torch.tensor(CAMERA_EYE, device=origins.device)
            target = origins + torch.tensor(CAMERA_TARGET, device=origins.device)
            self.camera.set_world_poses_from_view(eye, target)
        self.sim.set_camera_view(CAMERA_EYE, CAMERA_TARGET)
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
        for _ in range(render_updates):
            self.render_camera()


@contextmanager
def build_rendering_scene(
    physics_backend: str,
    *,
    renderer: str | None = None,
    data_types: Sequence[str] = (),
    num_envs: int = 1,
    visualizer_cfgs: Any = None,
    background_color: tuple[float, float, float] | None = None,
    device: str = "cuda:0",
) -> Iterator[RenderingScene]:
    """Build the canonical rendering scene without an RL/task environment."""
    scene_cfg = RenderingTestSceneCfg(num_envs=num_envs, env_spacing=5.0, lazy_sensor_update=True)
    if renderer is not None:
        scene_cfg.camera = make_camera_cfg(renderer, data_types, background_color=background_color)
    sim_cfg = SimulationCfg(
        dt=1.0 / 60.0,
        device=device,
        physics=make_physics_cfg(physics_backend),
        visualizer_cfgs=[] if visualizer_cfgs is None else visualizer_cfgs,
    )
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        if renderer is not None:
            sim.set_setting("/isaaclab/render/rtx_sensors", True)
        runtime = RenderingScene(sim, InteractiveScene(scene_cfg))
        sim.register_interactive_scene(runtime.scene)
        runtime.reset()
        yield runtime


def make_camera_cfg(
    renderer: str,
    data_types: Sequence[str],
    *,
    background_color: tuple[float, float, float] | None = None,
) -> CameraCfg:
    """Create the one camera layout used by all golden AOV cases."""
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        width=128,
        height=128,
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


def make_physics_cfg(physics_backend: str) -> Any:
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
