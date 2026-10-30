# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct scene lifecycle for renderer correctness tests."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.app import launch_simulation
from isaaclab.physics import PhysicsCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import Camera
from isaaclab.sim import SimulationCfg, SimulationContext, build_simulation_context
from isaaclab.utils import instantiate
from isaaclab.visualizers import VisualizerCfg


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
        self.camera.update(self.sim.get_rendering_dt(), force_recompute=True)

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
    data_types: Sequence[str] = (),
    background_color: tuple[float, float, float] | None = None,
    physics_cfg: PhysicsCfg | None = None,
    device: str = "cuda:0",
    visualizer_cfgs: Sequence[VisualizerCfg] = (),
    dt: float = 1.0 / 60.0,
    on_scene_created: Callable[[InteractiveScene], None] | None = None,
) -> Iterator[RenderingScene]:
    """Build a caller-owned rendering scene without an RL/task environment.

    ``on_scene_created`` runs on the instantiated scene before the simulation resets, for stage edits that renderers
    and physics backends must see when they initialize.
    """
    scene_cfg = copy.deepcopy(scene_cfg)
    scene_cfg.camera.data_types = list(data_types)
    scene_cfg.camera.background_color = background_color
    if physics_cfg is None:
        physics_cfg = {
            "physx": PhysxCfg(enable_enhanced_determinism=True, enable_external_forces_every_iteration=True),
            "ovphysx": OvPhysxCfg(enable_enhanced_determinism=True, enable_external_forces_every_iteration=True),
            "newton": NewtonCfg(solver_cfg=MJWarpSolverCfg(), num_substeps=1, debug_mode=False, use_cuda_graph=True),
        }[physics_backend]
    sim_cfg = SimulationCfg(
        dt=dt,
        device=device,
        physics=copy.deepcopy(physics_cfg),
        visualizer_cfgs=list(visualizer_cfgs),
    )
    with (
        launch_simulation(SimpleNamespace(sim=sim_cfg, scene=scene_cfg)),
        build_simulation_context(sim_cfg=sim_cfg) as sim,
    ):
        runtime = RenderingScene(sim, instantiate(scene_cfg))
        if on_scene_created is not None:
            on_scene_created(runtime.scene)
        sim.reset()
        runtime.scene.reset()
        for asset in runtime.scene.rigid_objects.values():
            asset.write_root_velocity_to_sim_index(root_velocity=asset.data.default_root_vel.torch)
        runtime.scene.write_data_to_sim()
        sim.forward()
        runtime.scene.update(0.0)
        yield runtime
