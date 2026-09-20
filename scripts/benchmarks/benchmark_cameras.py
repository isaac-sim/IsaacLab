# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark camera sensors in a sample scene or an Isaac Lab task.

The benchmark supports the RTX-backed :class:`isaaclab.sensors.Camera` and the
ray-caster camera. It can also increase the camera count in a task until a
configured system-utilization threshold is reached.

.. code-block:: bash

    # Benchmark 20 RTX cameras in Cartpole without a visualizer.
    uv run python scripts/benchmarks/benchmark_cameras.py \
        --task Isaac-Cartpole --num_cameras 20 --viz none

    # Benchmark two ray-caster cameras in the sample scene.
    uv run python scripts/benchmarks/benchmark_cameras.py \
        --num_ray_caster_cameras 2 --camera_data_types distance_to_image_plane

"""

from __future__ import annotations

# Parse command-line arguments before launching the simulation runtime.
import argparse
import copy
import sys
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab.app import add_launcher_args, launch_simulation

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene
    from isaaclab.sensors import Camera, RayCasterCamera


@dataclass(frozen=True)
class CameraSelection:
    """Resolved camera benchmark parameters."""

    kind: str
    count: int
    data_types: tuple[str, ...]


def _build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    task_group = parser.add_argument_group("task and autotune")
    task_group.add_argument("--task", help="Registered task in which to inject cameras.")
    task_group.add_argument(
        "--task_num_cameras_per_env",
        type=int,
        default=1,
        help="Number of injected camera sensors in each task environment.",
    )
    fabric_group = task_group.add_mutually_exclusive_group()
    fabric_group.add_argument(
        "--disable_fabric", action="store_true", help="Disable Fabric and use USD I/O operations."
    )
    fabric_group.add_argument(
        "--use_fabric",
        action="store_true",
        help="Deprecated compatibility option; Fabric is enabled by default.",
    )
    task_group.add_argument(
        "--autotune",
        action="store_true",
        help="Increase the camera count until a utilization threshold or maximum count is reached.",
    )
    task_group.add_argument(
        "--autotune_max_percentage_util",
        nargs=4,
        type=float,
        default=(100.0, 80.0, 80.0, 80.0),
        metavar=("CPU", "RAM", "GPU", "GPU_MEMORY"),
        help="CPU, RAM, GPU compute, and GPU memory utilization limits in percent.",
    )
    task_group.add_argument("--autotune_max_camera_count", type=int, default=4096, help="Maximum camera count to try.")
    task_group.add_argument(
        "--autotune_camera_count_interval",
        type=int,
        default=25,
        help="Number of cameras added after each successful autotune trial.",
    )

    camera_group = parser.add_argument_group("camera")
    camera_group.add_argument(
        "--num_cameras",
        type=int,
        default=0,
        help="Number of RTX Camera instances. For autotuning, this is the starting count.",
    )
    camera_group.add_argument(
        "--num_ray_caster_cameras",
        type=int,
        default=0,
        help="Number of ray-caster camera instances. For autotuning, this is the starting count.",
    )
    camera_group.add_argument(
        "--camera_data_types",
        nargs="+",
        default=None,
        help="Camera outputs to render (default: rgb depth for RTX, distance_to_image_plane for ray caster).",
    )
    camera_group.add_argument(
        "--ray_caster_visible_mesh_prim_paths",
        nargs="+",
        default=["/World/ground"],
        help="Static mesh prim paths visible to a ray-caster camera.",
    )
    camera_group.add_argument(
        "--keep_raw_depth",
        action="store_true",
        help="Do not convert distance_to_camera output to image-plane depth before unprojection.",
    )
    camera_group.add_argument("--height", type=int, default=120, help="Camera image height [pixels].")
    camera_group.add_argument("--width", type=int, default=140, help="Camera image width [pixels].")

    # Compatibility aliases for the pre-3.0 benchmark interface. Camera and TiledCamera
    # now use the same implementation, so both aliases resolve to --num_cameras.
    compatibility_group = parser.add_argument_group("deprecated camera options")
    compatibility_group.add_argument(
        "--num_tiled_cameras", type=int, default=None, help="Deprecated alias for --num_cameras."
    )
    compatibility_group.add_argument(
        "--num_standard_cameras", type=int, default=None, help="Deprecated alias for --num_cameras."
    )
    compatibility_group.add_argument(
        "--tiled_camera_data_types", nargs="+", default=None, help="Deprecated alias for --camera_data_types."
    )
    compatibility_group.add_argument(
        "--standard_camera_data_types", nargs="+", default=None, help="Deprecated alias for --camera_data_types."
    )
    compatibility_group.add_argument(
        "--ray_caster_camera_data_types", nargs="+", default=None, help="Deprecated alias for --camera_data_types."
    )
    compatibility_group.add_argument(
        "--convert_depth_to_camera_to_image_plane",
        action="store_true",
        help="Deprecated no-op; conversion is enabled unless --keep_raw_depth is passed.",
    )

    experiment_group = parser.add_argument_group("experiment")
    experiment_group.add_argument(
        "--warm_start_length", type=int, default=3, help="Number of warmup steps excluded from measurements."
    )
    experiment_group.add_argument(
        "--experiment_length", type=int, default=15, help="Number of measured simulation steps."
    )
    experiment_group.add_argument("--num_objects", type=int, default=10, help="Number of objects in the sample scene.")
    experiment_group.add_argument(
        "--benchmark_formatter",
        default="omniperf",
        choices=("json", "osmo", "omniperf", "summary"),
        help="Benchmark output formatter.",
    )
    experiment_group.add_argument("--output_path", default=".", help="Directory for benchmark results.")

    add_launcher_args(parser)
    return parser


def _resolve_camera_selection(parser: argparse.ArgumentParser, args: argparse.Namespace) -> CameraSelection:
    """Resolve current and deprecated camera options into one selection."""
    camera_counts = {
        "--num_cameras": args.num_cameras,
        "--num_ray_caster_cameras": args.num_ray_caster_cameras,
        "--num_tiled_cameras": args.num_tiled_cameras,
        "--num_standard_cameras": args.num_standard_cameras,
    }
    for option, count in camera_counts.items():
        if count is not None and count < 0:
            parser.error(f"{option} cannot be negative.")

    legacy_counts = [
        ("--num_tiled_cameras", args.num_tiled_cameras, args.tiled_camera_data_types),
        ("--num_standard_cameras", args.num_standard_cameras, args.standard_camera_data_types),
    ]
    selected_legacy = [(name, count, data_types) for name, count, data_types in legacy_counts if count is not None]
    if len(selected_legacy) > 1:
        parser.error("--num_tiled_cameras and --num_standard_cameras cannot be used together.")
    if selected_legacy and args.num_cameras != 0:
        parser.error("Use --num_cameras or a deprecated RTX camera count option, not both.")

    rtx_count = args.num_cameras
    rtx_data_types = args.camera_data_types
    if selected_legacy:
        option, rtx_count, legacy_data_types = selected_legacy[0]
        unrelated_data_option = (
            args.standard_camera_data_types if option == "--num_tiled_cameras" else args.tiled_camera_data_types
        )
        if unrelated_data_option is not None:
            parser.error("A deprecated camera count can only be used with its corresponding data type option.")
        warnings.warn(
            f"{option} is deprecated because Camera and TiledCamera are now the same sensor; use --num_cameras.",
            DeprecationWarning,
            stacklevel=2,
        )
        if legacy_data_types is not None and args.camera_data_types is not None:
            parser.error("Use only one camera data type option.")
        if legacy_data_types is not None:
            rtx_data_types = legacy_data_types
        elif rtx_data_types is None:
            rtx_data_types = (
                ["rgb", "depth"]
                if option == "--num_tiled_cameras"
                else ["rgb", "distance_to_image_plane", "distance_to_camera"]
            )

    legacy_data_options = (
        ("--tiled_camera_data_types", args.tiled_camera_data_types),
        ("--standard_camera_data_types", args.standard_camera_data_types),
        ("--ray_caster_camera_data_types", args.ray_caster_camera_data_types),
    )
    supplied_legacy_data = [name for name, value in legacy_data_options if value is not None]
    unrelated_legacy_data = [
        name
        for name in supplied_legacy_data
        if name != "--ray_caster_camera_data_types" or args.num_ray_caster_cameras == 0
    ]
    if unrelated_legacy_data and not selected_legacy:
        parser.error(f"{unrelated_legacy_data[0]} requires its corresponding deprecated camera count option.")

    if args.num_ray_caster_cameras > 0:
        if rtx_count > 0:
            parser.error("Benchmark one camera kind at a time.")
        if args.ray_caster_camera_data_types is not None:
            warnings.warn(
                "--ray_caster_camera_data_types is deprecated; use --camera_data_types.",
                DeprecationWarning,
                stacklevel=2,
            )
            if args.camera_data_types is not None:
                parser.error("Use only one camera data type option.")
            data_types = args.ray_caster_camera_data_types
        else:
            data_types = args.camera_data_types or ["distance_to_image_plane"]
        selection = CameraSelection("ray_caster", args.num_ray_caster_cameras, tuple(data_types))
    else:
        data_types = rtx_data_types or ["rgb", "depth"]
        selection = CameraSelection("camera", rtx_count, tuple(data_types))

    if selection.count <= 0:
        parser.error("Select at least one camera with --num_cameras or --num_ray_caster_cameras.")
    if not selection.data_types:
        parser.error("Select at least one camera data type.")
    return selection


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace, selection: CameraSelection) -> None:
    """Validate argument relationships and numeric bounds."""
    positive_fields = {
        "--height": args.height,
        "--width": args.width,
        "--experiment_length": args.experiment_length,
        "--task_num_cameras_per_env": args.task_num_cameras_per_env,
    }
    for option, value in positive_fields.items():
        if value <= 0:
            parser.error(f"{option} must be greater than zero.")
    if args.warm_start_length < 0 or args.num_objects < 0:
        parser.error("--warm_start_length and --num_objects cannot be negative.")
    if args.autotune and args.task is None:
        parser.error("--autotune requires --task.")
    if args.task and selection.count % args.task_num_cameras_per_env != 0:
        parser.error("The camera count must be divisible by --task_num_cameras_per_env.")
    if args.autotune:
        if args.autotune_camera_count_interval <= 0:
            parser.error("--autotune_camera_count_interval must be greater than zero.")
        if args.autotune_camera_count_interval % args.task_num_cameras_per_env != 0:
            parser.error("The autotune interval must be divisible by --task_num_cameras_per_env.")
        if args.autotune_max_camera_count < selection.count:
            parser.error("--autotune_max_camera_count cannot be smaller than the starting camera count.")
        if any(value < 0.0 or value > 100.0 for value in args.autotune_max_percentage_util):
            parser.error("Autotune utilization thresholds must be between 0 and 100 percent.")
    if args.use_fabric:
        warnings.warn(
            "--use_fabric is deprecated because Fabric is enabled by default.", DeprecationWarning, stacklevel=2
        )
    if args.convert_depth_to_camera_to_image_plane:
        if args.keep_raw_depth:
            parser.error("--convert_depth_to_camera_to_image_plane and --keep_raw_depth cannot be used together.")
        warnings.warn(
            "--convert_depth_to_camera_to_image_plane is deprecated because conversion is enabled by default.",
            DeprecationWarning,
            stacklevel=2,
        )


parser = _build_parser()
args_cli, hydra_overrides = parser.parse_known_args()
unknown_options = [argument for argument in hydra_overrides if argument.startswith("-")]
if unknown_options:
    parser.error(f"unrecognized arguments: {' '.join(unknown_options)}")
sys.argv = [sys.argv[0], *hydra_overrides]
camera_selection = _resolve_camera_selection(parser, args_cli)
_validate_args(parser, args_cli, camera_selection)
# Camera rendering extensions are required even when no visualizer is selected.
args_cli.enable_cameras = camera_selection.kind == "camera"
# RayCasterCamera queries USD meshes, which requires a Kit runtime even though it does not render through RTX.
args_cli.require_kit = camera_selection.kind == "ray_caster"

# Import configuration types before launch, but defer sensor and scene runtime classes
# because they load Kit modules that must only be imported after AppLauncher starts.

import random
import time
from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import psutil
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.benchmark import BaseIsaacLabBenchmark, DictMeasurement, SingleMeasurement
from isaaclab.physics import PhysicsCfg
from isaaclab.sensors import CameraCfg, RayCasterCameraCfg, patterns
from isaaclab.utils.math import orthogonalize_perspective_depth, unproject_depth

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


class UtilizationMonitor:
    """Track peak CPU, RAM, GPU compute, and GPU memory utilization."""

    def __init__(self, monitor_gpu: bool):
        self._pynvml = None
        self._gpu_handles = []
        self._maximum = [0.0, 0.0, 0.0, 0.0]
        psutil.cpu_percent(interval=None)
        if monitor_gpu and torch.cuda.is_available():
            try:
                import pynvml
            except ImportError as exc:
                raise ImportError("Autotuning requires pynvml. Install it with 'uv pip install nvidia-ml-py'.") from exc
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._gpu_handles = [
                pynvml.nvmlDeviceGetHandleByIndex(index) for index in range(pynvml.nvmlDeviceGetCount())
            ]

    def reset(self) -> None:
        """Reset all peak values."""
        self._maximum = [0.0, 0.0, 0.0, 0.0]
        psutil.cpu_percent(interval=None)

    def sample(self) -> None:
        """Sample system utilization and update peak values."""
        self._maximum[0] = max(self._maximum[0], psutil.cpu_percent(interval=None))
        self._maximum[1] = max(self._maximum[1], psutil.virtual_memory().percent)
        if self._pynvml is None:
            return
        for handle in self._gpu_handles:
            gpu_utilization = self._pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            memory = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
            self._maximum[2] = max(self._maximum[2], float(gpu_utilization))
            self._maximum[3] = max(self._maximum[3], memory.used / memory.total * 100.0)

    @property
    def maximum(self) -> list[float]:
        """Return a copy of the peak utilization percentages."""
        return self._maximum.copy()

    def close(self) -> None:
        """Release NVML when it was initialized."""
        if self._pynvml is not None:
            self._pynvml.nvmlShutdown()
            self._pynvml = None


def _create_camera_cfg(prim_path: str, selection: CameraSelection) -> CameraCfg | RayCasterCameraCfg:
    """Create a camera sensor configuration."""
    if selection.kind == "camera":
        return CameraCfg(
            prim_path=prim_path,
            update_period=0.0,
            height=args_cli.height,
            width=args_cli.width,
            data_types=list(selection.data_types),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e4),
            ),
        )
    return RayCasterCameraCfg(
        prim_path=prim_path,
        mesh_prim_paths=args_cli.ray_caster_visible_mesh_prim_paths,
        update_period=0.0,
        offset=RayCasterCameraCfg.OffsetCfg(),
        data_types=list(selection.data_types),
        debug_vis=False,
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            height=args_cli.height,
            width=args_cli.width,
        ),
    )


def _add_cameras_to_task_cfg(env_cfg, selection: CameraSelection) -> None:
    """Add the selected camera sensors to a task environment configuration."""

    for index in range(args_cli.task_num_cameras_per_env):
        name = "benchmark_camera" if index == 0 else f"benchmark_camera_{index}"
        prim_path = f"{{ENV_REGEX_NS}}/{name}"
        if selection.kind == "ray_caster":
            setattr(
                env_cfg.scene,
                f"{name}_mount",
                RigidObjectCfg(
                    prim_path=prim_path,
                    spawn=sim_utils.CuboidCfg(
                        size=(0.01, 0.01, 0.01),
                        visible=False,
                        rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
                        mass_props=sim_utils.MassCfg(mass=0.001),
                    ),
                ),
            )
        setattr(env_cfg.scene, name, _create_camera_cfg(prim_path, selection))


def _create_task_cfg(selection: CameraSelection):
    """Resolve a task config and inject the selected number of cameras."""
    num_envs = selection.count // args_cli.task_num_cameras_per_env
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=num_envs,
        use_fabric=not args_cli.disable_fabric,
        overrides=hydra_overrides,
    )
    _add_cameras_to_task_cfg(env_cfg, selection)
    return env_cfg


def _create_sample_scene_cfg(selection: CameraSelection):
    """Create the standalone scene config before the simulation runtime launches."""
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveSceneCfg

    scene_cfg = InteractiveSceneCfg(num_envs=selection.count, env_spacing=4.0, replicate_physics=True)
    scene_cfg.ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    scene_cfg.light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    for index in range(args_cli.num_objects):
        position = (np.random.rand(3) - np.asarray([0.05, 0.05, -1.0])) * np.asarray([1.5, 1.5, 0.5])
        color = (random.random(), random.random(), random.random())
        shape_type = random.choice(("Cube", "Cone", "Cylinder"))
        properties = {
            "rigid_props": sim_utils.UsdPhysicsRigidBodyCfg(),
            "mass_props": sim_utils.MassCfg(mass=5.0),
            "collision_props": sim_utils.UsdPhysicsCollisionCfg(),
            "visual_material": sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=0.5),
            "semantic_tags": [("class", shape_type)],
        }
        if shape_type == "Cube":
            shape_cfg = sim_utils.CuboidCfg(size=(0.25, 0.25, 0.25), **properties)
        elif shape_type == "Cone":
            shape_cfg = sim_utils.ConeCfg(radius=0.1, height=0.25, **properties)
        else:
            shape_cfg = sim_utils.CylinderCfg(radius=0.25, height=0.25, **properties)
        object_cfg = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Objects/Obj_{index:02d}",
            spawn=shape_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(position)),
        )
        setattr(scene_cfg, f"rigid_object_{index}", object_cfg)

    camera_prim_path = "{ENV_REGEX_NS}/BenchmarkCamera"
    if selection.kind == "ray_caster":
        scene_cfg.benchmark_camera_mount = RigidObjectCfg(
            prim_path=camera_prim_path,
            spawn=sim_utils.CuboidCfg(
                size=(0.01, 0.01, 0.01),
                visible=False,
                rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassCfg(mass=0.001),
            ),
        )
    scene_cfg.benchmark_camera = _create_camera_cfg(camera_prim_path, selection)
    return scene_cfg


def _create_sample_scene(
    sim: sim_utils.SimulationContext, scene_cfg
) -> tuple[InteractiveScene, list[Camera | RayCasterCamera]]:
    """Instantiate the standalone scene after the simulation runtime launches."""
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim.utils.stage import use_stage

    with use_stage(sim.stage):
        scene = InteractiveScene(scene_cfg)
    sim.register_interactive_scene(scene)
    return scene, [scene["benchmark_camera"]]


def _get_task_cameras(scene: InteractiveScene) -> list[Camera | RayCasterCamera]:
    """Return the injected camera sensors from a task scene."""
    cameras = []
    for index in range(args_cli.task_num_cameras_per_env):
        name = "benchmark_camera" if index == 0 else f"benchmark_camera_{index}"
        cameras.append(scene[name])
    return cameras


def _as_torch(value) -> torch.Tensor:
    """Return a Torch tensor, materializing a zero-copy ProxyArray view when needed."""
    return value.torch if hasattr(value, "torch") else value


def _process_camera_outputs(camera: Camera | RayCasterCamera, data_types: Sequence[str]) -> None:
    """Read configured outputs and perform the benchmark's depth-to-point-cloud work."""
    for data_type in data_types:
        output = camera.data.output[data_type]
        if data_type not in {"depth", "distance_to_camera", "distance_to_image_plane"}:
            continue
        depth = _as_torch(output)
        intrinsics = _as_torch(camera.data.intrinsic_matrices)
        if data_type == "distance_to_camera" and not args_cli.keep_raw_depth:
            depth = orthogonalize_perspective_depth(depth, intrinsics)
        unproject_depth(depth=depth, intrinsics=intrinsics)


def _synchronize_cuda() -> None:
    """Synchronize CUDA so timings include asynchronous camera processing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _run_simulator(
    cameras: Sequence[Camera | RayCasterCamera],
    monitor: UtilizationMonitor,
    *,
    sim: sim_utils.SimulationContext | None = None,
    scene: InteractiveScene | None = None,
    env: gym.Env | None = None,
) -> dict[str, dict | list[float]]:
    """Run warmup and measured steps and return timing and utilization metrics."""
    if (sim is None) == (env is None):
        raise ValueError("Provide exactly one of sim or env.")

    if sim is not None:
        for camera in cameras:
            targets = (
                scene.env_origins if scene is not None else torch.zeros((camera.num_instances, 3), device=sim.device)
            )
            positions = targets + 2.5
            camera.set_world_poses_from_view(positions, targets)
        physics_dt = sim.get_physics_dt()
        actions = None
    else:
        physics_dt = 0.0
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

    def step() -> tuple[float, float]:
        _synchronize_cuda()
        iteration_start = time.perf_counter()
        if sim is not None:
            if scene is not None:
                scene.write_data_to_sim()
            sim.step()
            if scene is not None:
                scene.update(dt=physics_dt)
            else:
                for camera in cameras:
                    camera.update(dt=physics_dt)
        else:
            with torch.inference_mode():
                env.step(actions)
        _synchronize_cuda()
        simulation_duration = time.perf_counter() - iteration_start
        for camera in cameras:
            _process_camera_outputs(camera, camera_selection.data_types)
        _synchronize_cuda()
        return time.perf_counter() - iteration_start, simulation_duration

    for step_index in range(args_cli.warm_start_length):
        print(f"Warmup step {step_index + 1}/{args_cli.warm_start_length}")
        step()

    monitor.reset()
    total_duration = 0.0
    simulation_duration = 0.0
    for step_index in range(args_cli.experiment_length):
        print(f"Measured step {step_index + 1}/{args_cli.experiment_length}")
        step_duration, sim_step_duration = step()
        total_duration += step_duration
        simulation_duration += sim_step_duration
        monitor.sample()

    timing = {
        "average_timestep_duration": total_duration / args_cli.experiment_length,
        "average_sim_step_duration": simulation_duration / args_cli.experiment_length,
        "total_simulation_time": simulation_duration,
        "total_experiment_duration": total_duration,
    }
    utilization = monitor.maximum
    print("--- Benchmark Results ---")
    print(f"Average timestep duration: {timing['average_timestep_duration']:.6f} seconds")
    print(f"Average simulation step duration: {timing['average_sim_step_duration']:.6f} seconds")
    print(f"Total experiment duration: {timing['total_experiment_duration']:.6f} seconds")
    print("\nSystem Utilization Statistics:")
    print(
        f"| CPU: {utilization[0]:.1f}% | RAM: {utilization[1]:.1f}% | "
        f"GPU Compute: {utilization[2]:.1f}% | GPU Memory: {utilization[3]:.1f}% |"
    )
    if not args_cli.autotune:
        print("GPU utilization sampling is enabled only during autotuning.")
    return {"timing_analytics": timing, "system_utilization_analytics": utilization}


def _run_sample_scene(sim_cfg: sim_utils.SimulationCfg, scene_cfg, monitor: UtilizationMonitor):
    """Run one benchmark in the standalone sample scene."""
    from isaaclab.sim.utils.stage import use_stage

    sim = sim_utils.SimulationContext(sim_cfg)
    scene, cameras = _create_sample_scene(sim, scene_cfg)
    with use_stage(sim.stage):
        sim.reset()
    scene.update(dt=sim.get_physics_dt())
    scene.reset()
    print("[INFO] Sample scene setup complete.")
    return _run_simulator(cameras, monitor, sim=sim, scene=scene)


def _run_task_trial(env_cfg, monitor: UtilizationMonitor):
    """Create, benchmark, and close one task environment."""
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        env.reset()
        cameras = _get_task_cameras(env.unwrapped.scene)
        return _run_simulator(cameras, monitor, env=env)
    finally:
        env.close()


def _run_task_benchmarks(template_cfg, monitor: UtilizationMonitor):
    """Run one task benchmark or the requested autotune sequence."""
    thresholds = args_cli.autotune_max_percentage_util
    count = camera_selection.count
    last_passing_count = None
    final_analysis = None
    camera_label = "RTX" if camera_selection.kind == "camera" else "ray-caster"

    while count <= (args_cli.autotune_max_camera_count if args_cli.autotune else camera_selection.count):
        env_cfg = copy.deepcopy(template_cfg)
        env_cfg.scene.num_envs = count // args_cli.task_num_cameras_per_env
        print(f"[INFO] Testing {count} {camera_label} cameras across {env_cfg.scene.num_envs} environments.")
        final_analysis = _run_task_trial(env_cfg, monitor)
        utilization = final_analysis["system_utilization_analytics"]
        within_thresholds = all(value <= limit for value, limit in zip(utilization, thresholds))
        if not args_cli.autotune or not within_thresholds:
            break
        last_passing_count = count
        count += args_cli.autotune_camera_count_interval
        if count <= args_cli.autotune_max_camera_count:
            sim_utils.create_new_stage()

    if args_cli.autotune:
        if last_passing_count is None:
            print("[INFO] The starting camera count exceeded at least one utilization threshold.")
        else:
            print(f"[INFO] Largest tested camera count within all thresholds: {last_passing_count}.")
        print("[INFO] These results exclude training workload; reserve resources for the learning process.")
    return final_analysis


def _record_results(benchmark: BaseIsaacLabBenchmark, analysis) -> None:
    """Add benchmark measurements and write the selected output format."""
    timing = analysis["timing_analytics"]
    utilization = analysis["system_utilization_analytics"]
    benchmark.add_measurement(
        "runtime",
        SingleMeasurement(
            name="Average Timestep Duration", value=timing["average_timestep_duration"] * 1000.0, unit="ms"
        ),
    )
    benchmark.add_measurement(
        "runtime",
        SingleMeasurement(
            name="Average Simulation Step Duration",
            value=timing["average_sim_step_duration"] * 1000.0,
            unit="ms",
        ),
    )
    benchmark.add_measurement(
        "runtime",
        SingleMeasurement(name="Total Simulation Time", value=timing["total_simulation_time"] * 1000.0, unit="ms"),
    )
    benchmark.add_measurement(
        "runtime",
        DictMeasurement(
            name="System Utilization",
            value={
                "cpu_percent": utilization[0],
                "ram_percent": utilization[1],
                "gpu_compute_percent": utilization[2],
                "gpu_memory_percent": utilization[3],
            },
        ),
    )
    benchmark.update_manual_recorders()
    benchmark.finalize()


def main() -> None:
    """Launch the required runtime and execute the camera benchmark."""
    if args_cli.task is None:
        sample_scene_cfg = _create_sample_scene_cfg(camera_selection)
        sample_sim_cfg = sim_utils.SimulationCfg(
            device=args_cli.device,
            use_fabric=not args_cli.disable_fabric,
            physics=PhysicsCfg(),
        )
        launch_cfg = {"sim": sample_sim_cfg, "scene": sample_scene_cfg}
        task_cfg = None
    else:
        sample_scene_cfg = None
        sample_sim_cfg = None
        task_cfg = _create_task_cfg(camera_selection)
        launch_cfg = task_cfg

    with launch_simulation(launch_cfg, args_cli) as physics_cfg:
        benchmark = BaseIsaacLabBenchmark(
            benchmark_name="benchmark_cameras",
            formatter_type=args_cli.benchmark_formatter,
            output_path=args_cli.output_path,
            use_recorders=True,
            frametime_recorders=args_cli.benchmark_formatter in ("summary", "omniperf"),
            output_prefix="benchmark_cameras",
            workflow_metadata={
                "metadata": [
                    {"name": "task", "data": args_cli.task},
                    {"name": "camera_type", "data": camera_selection.kind},
                    {"name": "num_cameras", "data": camera_selection.count},
                    {"name": "height", "data": args_cli.height},
                    {"name": "width", "data": args_cli.width},
                    {"name": "experiment_length", "data": args_cli.experiment_length},
                    {"name": "autotune", "data": args_cli.autotune},
                ]
            },
        )
        monitor = UtilizationMonitor(monitor_gpu=args_cli.autotune)
        try:
            if task_cfg is None:
                sample_sim_cfg.physics = physics_cfg
                analysis = _run_sample_scene(sample_sim_cfg, sample_scene_cfg, monitor)
            else:
                analysis = _run_task_benchmarks(task_cfg, monitor)
            _record_results(benchmark, analysis)
        finally:
            monitor.close()


if __name__ == "__main__":
    main()
