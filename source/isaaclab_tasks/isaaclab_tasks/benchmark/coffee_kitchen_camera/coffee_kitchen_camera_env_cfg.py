# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rigid coffee kitchen benchmark using the demo's scene and the existing analytic animation."""

from __future__ import annotations

import os

import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix

from isaaclab_tasks.benchmark.render_benchmark.render_benchmark_env_cfg import (
    BENCHMARK_MODE,
    RenderBenchmarkFrankaCabinetEnvCfg,
)
from isaaclab_tasks.contrib.rizon_sharpa_coffee.demo_env_cfg import (
    KITCHEN_CAMERA_EYE,
    KITCHEN_CAMERA_FOCAL_LENGTH,
    KITCHEN_CAMERA_LOOKAT,
    CoffeeKitchenCfg,
    CoffeeSceneCfg,
    configure_scene_rigid,
)
from isaaclab_tasks.utils import PresetCfg, preset
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg


def _camera(data_type: str) -> CameraCfg:
    rotation = create_rotation_matrix_from_view(
        torch.tensor([KITCHEN_CAMERA_EYE]), torch.tensor([KITCHEN_CAMERA_LOOKAT]), "Z", device="cpu"
    )
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=CameraCfg.OffsetCfg(
            pos=KITCHEN_CAMERA_EYE, rot=tuple(quat_from_matrix(rotation)[0].tolist()), convention="opengl"
        ),
        data_types=[data_type],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=KITCHEN_CAMERA_FOCAL_LENGTH, horizontal_aperture=15.2908, vertical_aperture=15.2908
        ),
        width=int(os.getenv("BENCHMARK_RENDER_RESOLUTION", "256")),
        height=int(os.getenv("BENCHMARK_RENDER_RESOLUTION", "256")),
        # The robot uses UDIM textures that the Newton renderer does not load.
        renderer_cfg=MultiBackendRendererCfg(
            newton_renderer=NewtonWarpRendererCfg(enable_textures=False),
            default=NewtonWarpRendererCfg(enable_textures=False),
        ),
    )


@configclass
class CoffeeKitchenSceneCfg(CoffeeSceneCfg):
    """One rigid coffee workstation, kitchen and camera in each environment."""

    kitchen: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Kitchen",
        cloning_contexts=preset(default=None, ovrtx=(), isaacsim_rtx=(), rtx=()),
        spawn=CoffeeKitchenCfg(disable_lights=True, remove_ceiling=True),
    )
    tiled_camera: PresetCfg = preset(
        default=_camera("rgb"),
        **{
            data_type: _camera(data_type)
            for data_type in (
                "rgb",
                "albedo",
                "depth",
                "simple_shading_constant_diffuse",
                "simple_shading_diffuse_mdl",
                "simple_shading_full_mdl",
                "semantic_segmentation",
            )
        },
    )

    def __post_init__(self) -> None:
        configure_scene_rigid(self)


_PHYSICS = NewtonCfg(
    solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=False, integrator="implicitfast", njmax=1024, nconmax=128),
    num_substeps=1,
    use_cuda_graph=True,
)


@configclass
class CoffeeKitchenCameraEnvCfg(RenderBenchmarkFrankaCabinetEnvCfg):
    """Replicated geometry throughput; coffee/water particles and the demo's exact motion are excluded.

    Camera images are consumed on every step by the existing render benchmark. The arm follows its
    deterministic sinusoid; the hand retains the demo's default grip. Both modes step rigid physics.
    """

    scene: CoffeeKitchenSceneCfg = CoffeeKitchenSceneCfg(num_envs=4, env_spacing=8.0)
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 800.0,
        render_interval=8,
        physics=preset(default=_PHYSICS, newton_mjwarp=_PHYSICS),
        use_newton_actuators=True,
    )
    decimation: int = 8
    seed: int = 42
    benchmark_mode = BENCHMARK_MODE or "render"
    joint_animation_amplitude: float = 0.1
    """Peak arm-joint offset from the demo's ready pose [rad]."""
