# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct scene lifecycle for renderer correctness tests."""

from __future__ import annotations

import copy
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_ov.renderers import OVRTXRendererCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg

from isaaclab.app import launch_simulation
from isaaclab.physics import PhysicsCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import Camera
from isaaclab.sim import SimulationCfg, SimulationContext, build_simulation_context
from isaaclab.utils import instantiate
from isaaclab.visualizers import VisualizerCfg

SEMANTIC_COLORS = {
    "class:ground": (94, 112, 131, 255),
    "class:robot": (246, 184, 59, 255),
    "class:cartpole": (33, 243, 3, 255),
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

    @property
    def camera(self) -> Camera:
        return self.scene.sensors["camera"]

    def step(self, *, render: bool = True) -> None:
        """Advance physics exactly once and update the scene buffers."""
        self.scene.write_data_to_sim()
        self.sim.step(render=render)
        self.scene.update(self.sim.get_physics_dt())

    def camera_outputs(self) -> tuple[dict[str, torch.Tensor], dict[str, Any] | None]:
        """Snapshot camera output tensors and metadata before a later render overwrites them."""
        data = self.camera.data
        outputs = {name: value.torch.clone() for name, value in data.output.items()}
        return outputs, copy.deepcopy(data.info)

    def render_camera(self) -> None:
        """Render and refresh the camera without advancing physics."""
        self.sim.render()
        self.camera.update(0.0, force_recompute=True)

    def stabilize_camera(self, render_updates: int = 5) -> None:
        """Prime renderer history without advancing physics."""
        if self.camera.cfg.renderer_cfg.renderer_type == "isaac_rtx":
            import omni.usd

            omni.usd.get_context().reset_renderer_accumulation()
        for _ in range(render_updates):
            self.render_camera()


@contextmanager
def build_rendering_scene(
    scene_cfg: InteractiveSceneCfg,
    physics_backend: str,
    *,
    renderer: str,
    data_types: Sequence[str] = (),
    background_color: tuple[float, float, float] | None = None,
    physics_cfg: PhysicsCfg | None = None,
    device: str = "cuda:0",
    visualizer_cfgs: Sequence[VisualizerCfg] = (),
    async_rendering: bool = False,
) -> Iterator[RenderingScene]:
    """Build a caller-owned rendering scene without an RL/task environment."""
    scene_cfg = copy.deepcopy(scene_cfg)
    scene_cfg.camera.data_types = list(data_types)
    scene_cfg.camera.background_color = background_color
    scene_cfg.camera.renderer_cfg = {
        "isaac_rtx": IsaacRtxRendererCfg(semantic_filter=["class"], semantic_segmentation_mapping=SEMANTIC_COLORS),
        "newton_warp": NewtonWarpRendererCfg(
            render_order="pixel_priority", semantic_filter=["class"], semantic_segmentation_mapping=SEMANTIC_COLORS
        ),
        "ovrtx": OVRTXRendererCfg(enable_shadows=True, async_rendering=async_rendering),
    }[renderer]
    if physics_cfg is None:
        physics_cfg = {
            "physx": PhysxCfg(enable_enhanced_determinism=True, enable_external_forces_every_iteration=True),
            "ovphysx": OvPhysxCfg(enable_enhanced_determinism=True, enable_external_forces_every_iteration=True),
            "newton": NewtonCfg(solver_cfg=MJWarpSolverCfg(), num_substeps=1, debug_mode=False, use_cuda_graph=True),
        }[physics_backend]
    sim_cfg = SimulationCfg(
        dt=1.0 / 60.0,
        device=device,
        physics=copy.deepcopy(physics_cfg),
        visualizer_cfgs=list(visualizer_cfgs),
    )
    with (
        launch_simulation(SimpleNamespace(sim=sim_cfg, scene=scene_cfg)),
        build_simulation_context(sim_cfg=sim_cfg) as sim,
    ):
        runtime = RenderingScene(sim, instantiate(scene_cfg))
        sim.reset()
        runtime.scene.reset()
        # Spawned root transforms already include the asset's local frame. Initialize only dynamic state;
        # writing an init_state root pose here would discard that frame on fixed-base articulations.
        for asset in runtime.scene.articulations.values():
            asset.write_joint_position_to_sim_index(position=asset.data.default_joint_pos.torch)
            asset.write_joint_velocity_to_sim_index(velocity=asset.data.default_joint_vel.torch)
            asset.set_joint_position_target_index(target=asset.data.default_joint_pos.torch)
            asset.set_joint_velocity_target_index(target=asset.data.default_joint_vel.torch)
        for asset in runtime.scene.rigid_objects.values():
            asset.write_root_velocity_to_sim_index(root_velocity=asset.data.default_root_vel.torch)
        runtime.scene.write_data_to_sim()
        sim.forward()
        runtime.scene.update(0.0)
        yield runtime
