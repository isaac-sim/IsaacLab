# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark the PhysX FrameTransformer update path.

Each environment contains a kinematic source body and target body. Multiple
target frames share the target body with distinct offsets, exercising the
FrameTransformer kernel without adding unnecessary PhysX bodies.

Usage:
    uv run python source/isaaclab_physx/benchmark/sensors/benchmark_frame_transformer.py --num_envs 4096
"""

from __future__ import annotations

import argparse
import traceback
from functools import partial

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Benchmark the PhysX FrameTransformer update path.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments.")
parser.add_argument("--num_target_frames", type=int, default=4, help="Number of target frames per environment.")
parser.add_argument("--num_steps", type=int, default=500, help="Number of timed updates.")
parser.add_argument("--warmup_steps", type=int, default=50, help="Number of untimed warm-up updates.")
parser.add_argument("--label", type=str, default="current", help="Label printed with the benchmark results.")
parser.add_argument("--output_path", type=str, default=".", help="Output directory for benchmark results.")
parser.add_argument(
    "--benchmark_formatter",
    type=str,
    default="summary",
    choices=["json", "osmo", "omniperf", "summary"],
    help="Formatter used for benchmark results.",
)
parser.add_argument(
    "--disable_recorded_launch",
    action="store_true",
    help="Use cached PhysX views with ordinary eager Warp launches.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_target_frames <= 0:
    parser.error("--num_target_frames must be greater than zero")
if args_cli.num_envs <= 0:
    parser.error("--num_envs must be greater than zero")
if args_cli.num_steps <= 0:
    parser.error("--num_steps must be greater than zero")
if args_cli.warmup_steps < 0:
    parser.error("--warmup_steps must be non-negative")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below follows application launch."""

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.benchmark import LatencyBenchmarkRunner, SingleMeasurement
from isaaclab.benchmark.micro import measure_latency
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg
from isaaclab.utils.configclass import configclass


@configclass
class FrameTransformerBenchmarkSceneCfg(InteractiveSceneCfg):
    """Two kinematic rigid bodies and one FrameTransformer per environment."""

    source = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Source",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    )
    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.5)),
    )
    frame_transformer: FrameTransformerCfg = None


def main() -> None:
    """Run the benchmark and print latency statistics."""
    sim_dt = 1.0 / 120.0
    sim_cfg = sim_utils.SimulationCfg(dt=sim_dt, device=args_cli.device, gravity=(0.0, 0.0, 0.0))
    sim = sim_utils.SimulationContext(sim_cfg)

    scene_cfg = FrameTransformerBenchmarkSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=1.0,
        lazy_sensor_update=True,
    )
    scene_cfg.frame_transformer = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Source",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                name=f"target_{index}",
                prim_path="{ENV_REGEX_NS}/Target",
                offset=OffsetCfg(pos=(0.0, 0.01 * index, 0.0)),
            )
            for index in range(args_cli.num_target_frames)
        ],
    )
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.reset()

    sensor = scene["frame_transformer"]

    if args_cli.disable_recorded_launch:
        sensor._use_recorded_launch = False
    for _ in range(args_cli.warmup_steps):
        sim.step(render=False)
        sensor.update(sim_dt, force_recompute=True)
    wp.synchronize_device(sim.device)

    synchronize = partial(wp.synchronize_device, sim.device)
    samples = []
    for _ in range(args_cli.num_steps):
        sim.step(render=False)
        samples.append(
            measure_latency(
                operation=lambda: sensor.update(sim_dt, force_recompute=True),
                synchronize=synchronize,
            )
        )

    observer_samples = [
        measure_latency(operation=lambda: None, synchronize=synchronize) for _ in range(args_cli.num_steps)
    ]

    target_positions = sensor.data.target_pos_w.torch
    finite_frames = int(torch.isfinite(target_positions).all(dim=-1).sum().item())
    expected_frames = args_cli.num_envs * args_cli.num_target_frames

    if finite_frames != expected_frames:
        raise RuntimeError(f"Expected {expected_frames} finite target frames, received {finite_frames}.")

    benchmark = LatencyBenchmarkRunner(
        benchmark_name="physx_frame_transformer_sensor",
        formatter_type=args_cli.benchmark_formatter,
        output_path=args_cli.output_path,
        metadata={
            "label": args_cli.label,
            "device": str(sim.device),
            "num_envs": args_cli.num_envs,
            "target_frames_per_env": args_cli.num_target_frames,
            "num_steps": args_cli.num_steps,
            "warmup_steps": args_cli.warmup_steps,
        },
    )
    benchmark.add_latency_samples("sensor_update", samples)
    benchmark.add_synchronized_samples(
        "observer", "Synchronized Observer Floor", [s.synchronized_s for s in observer_samples]
    )
    benchmark.add_measurement(
        "validation",
        measurement=[
            SingleMeasurement(name="Finite Target Frames", value=finite_frames, unit="count"),
            SingleMeasurement(name="Expected Target Frames", value=expected_frames, unit="count"),
        ],
    )
    benchmark.finalize()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        if simulation_app.config.get("fast_shutdown", False):
            traceback.print_exc()
        simulation_app.close(exit_code=1)
        raise
    else:
        simulation_app.close()
