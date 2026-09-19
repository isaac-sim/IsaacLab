# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.utils import PresetCfg
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG


@configclass
class RenderBenchmarkPhysicsCfg(PresetCfg):
    """Physics backend presets.

    Pick via ``presets=newton_mjwarp`` (default) or ``presets=physx``, which resolves to the
    concrete PhysX backend at launch. Use ``presets=isaacsim_physx`` to pin Isaac Sim PhysX.
    The BVH constructors are read from the environment so ``benchmark_renderer.py`` can sweep
    them without a separate preset per combination.
    """

    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(solver="newton", integrator="implicitfast", njmax=200, nconmax=70),
        num_substeps=2,
        bvh_constructor_geometry=os.getenv("NEWTON_BVH_GEOMETRY", "cubql"),
        bvh_constructor_gaussian=os.getenv("NEWTON_BVH_GAUSSIAN", "cubql"),
        bvh_constructor_scene=os.getenv("NEWTON_BVH_SCENE", "sah"),
        use_cuda_graph=os.getenv("NEWTON_USE_CUDA_GRAPH", "0") == "1",
    )
    isaacsim_physx: PhysxCfg = PhysxCfg()
    physx: PhysxAutoCfg = PhysxAutoCfg(isaacsim_physx=isaacsim_physx)
    default = newton_mjwarp


@configclass
class RenderBenchmarkTiledCameraCfg(PresetCfg):
    """Render-target presets — pick via ``presets=rgb`` (default), ``presets=depth``, and so on."""

    @configclass
    class BaseRenderBenchmarkCameraCfg(CameraCfg):
        """Front view of the workspace, pitch-only (no yaw, no roll).

        The camera sits on the drawer-facing ``+X`` side of the cabinet, elevated and pitched
        ~34.5 deg down so the workspace reads as a slight bird's-eye. ``rot`` is ``(qx, qy, qz, qw)``
        in ``convention="world"`` (``+X`` camera-forward): with ``qw=0`` and only ``qx``/``qz`` set it
        is a 180 deg rotation about the ``(-X, 0, +Z)`` axis, aiming the camera at ``(0.4, 0, 0.4)``.
        """

        prim_path: str = "{ENV_REGEX_NS}/Camera"
        offset: CameraCfg.OffsetCfg = CameraCfg.OffsetCfg(
            pos=(2.0, 0.0, 1.5), rot=(-0.296, 0.0, 0.955, 0.0), convention="world"
        )
        data_types: list[str] = []
        spawn: sim_utils.PinholeCameraCfg = sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 50.0)
        )
        width: int = int(os.getenv("BENCHMARK_RENDER_RESOLUTION", "256"))
        height: int = int(os.getenv("BENCHMARK_RENDER_RESOLUTION", "256"))
        renderer_cfg: MultiBackendRendererCfg = MultiBackendRendererCfg(
            newton_renderer=NewtonWarpRendererCfg(enable_shadows=True),
        )

    default = BaseRenderBenchmarkCameraCfg(data_types=["rgb"])
    rgb = default
    albedo = BaseRenderBenchmarkCameraCfg(data_types=["albedo"])
    depth = BaseRenderBenchmarkCameraCfg(data_types=["depth"])
    simple_shading_constant_diffuse = BaseRenderBenchmarkCameraCfg(data_types=["simple_shading_constant_diffuse"])
    simple_shading_diffuse_mdl = BaseRenderBenchmarkCameraCfg(data_types=["simple_shading_diffuse_mdl"])
    simple_shading_full_mdl = BaseRenderBenchmarkCameraCfg(data_types=["simple_shading_full_mdl"])


@configclass
class RenderBenchmarkSceneCfg(InteractiveSceneCfg):
    """Franka, cabinet, ground, camera, and lighting for renderer benchmarking."""

    # Use a simulation mesh that both renderers see, sized to tile the default environments without overlap.
    ground: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ground",
        spawn=sim_utils.CuboidCfg(
            size=(3.0, 3.0, 0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5), metallic=0.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
    )
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(
            pos=(1.0, 0.0, 0.0),
            rot=(0.0, 0.0, 1.0, 0.0),
        ),
    )
    cabinet: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.4),
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.0,
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                joint_effort_limit=87.0,
                stiffness=10.0,
                damping=1.0,
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                joint_effort_limit=87.0,
                stiffness=10.0,
                damping=2.5,
            ),
        },
    )
    tiled_camera: RenderBenchmarkTiledCameraCfg = RenderBenchmarkTiledCameraCfg()
    dome_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )
    directional_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/LightDirectional",
        spawn=sim_utils.DistantLightCfg(
            intensity=200.0,
            exposure=0.0,
            angle=0.0,
            color=(1.0, 1.0, 1.0),
            normalize=True,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.3251, 0.3251, 0.0, 0.8881)),
    )


@configclass
class RenderBenchmarkFrankaCabinetEnvCfg(DirectRLEnvCfg):
    """Franka Panda and Sektion cabinet, animated for renderer benchmarking.

    The cabinet contributes four articulated joints (two drawers, two doors) on top of the
    Franka's seven, and a sinusoidal animation drives all of them so every rendered frame has
    moving articulated geometry rather than a static scene. There is no policy: actions are
    ignored, rewards are zero, and the episode only ends on time-out.
    """

    decimation: int = 2
    episode_length_s: float = 60.0

    action_space: int = 1
    observation_space: int = 1
    state_space: int = 0

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=2, physics=RenderBenchmarkPhysicsCfg())

    scene: RenderBenchmarkSceneCfg = RenderBenchmarkSceneCfg(num_envs=4, env_spacing=3.0, replicate_physics=True)

    joint_animation_amplitude: float = 0.4
    """Peak sinusoidal offset from each joint's default position [m or rad, depending on joint type].

    Values at or below zero disable the animation, leaving a static scene.
    """

    joint_animation_freq_hz: float = 0.35
    """Frequency of the joint animation [Hz]."""

    write_image_to_file: bool = os.getenv("BENCHMARK_SAVE_IMAGE", "0") == "1"
    """Whether to dump each rendered frame to a PNG, for eyeballing renderer output."""
