# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark the OVPhysX FrameTransformer update path.

Mirrors ``isaaclab_physx/benchmark/sensors/benchmark_frame_transformer.py`` but
runs kitless against the OVPhysX backend. Also times the per-body blocking
``RIGID_BODY_POSE`` reads in isolation.

Usage:
    ./isaaclab.sh -p source/isaaclab_ovphysx/benchmark/sensors/benchmark_frame_transformer.py --num_envs 4096
"""

from __future__ import annotations

import argparse
from functools import partial

from isaaclab.benchmark._cli import parse_non_negative_int, parse_positive_int

parser = argparse.ArgumentParser(description="Benchmark the OVPhysX FrameTransformer update path.")
parser.add_argument("--physics_variant", choices=("ovphysx",), default="ovphysx", help="Exact physics variant.")
parser.add_argument("--num_envs", type=parse_positive_int, default=4096, help="Number of environments.")
parser.add_argument(
    "--num_target_frames", type=parse_positive_int, default=4, help="Number of target frames per environment."
)
parser.add_argument("--num_steps", type=parse_positive_int, default=500, help="Number of timed updates.")
parser.add_argument(
    "--warmup_steps", type=parse_non_negative_int, default=50, help="Number of untimed warm-up updates."
)
parser.add_argument("--label", type=str, default="current", help="Label printed with the benchmark results.")
parser.add_argument("--output_path", type=str, default=".", help="Output directory for benchmark results.")
parser.add_argument(
    "--benchmark_formatter",
    type=str,
    default="summary",
    choices=["json", "osmo", "omniperf", "summary"],
    help="Formatter used for benchmark results.",
)
parser.add_argument("--device", type=str, default="cuda:0", help="Simulation device.")
args_cli = parser.parse_args()

import isaaclab_ovphysx.tensor_types as TT
import torch
import warp as wp
from isaaclab_ovphysx.physics import OvPhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.benchmark import LatencyBenchmarkRunner, SingleMeasurement
from isaaclab.benchmark.micro import measure_latency
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.utils.configclass import configclass

wp.init()


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
    frame_transformer: FrameTransformerCfg | None = None


def main() -> None:
    """Run the benchmark and print latency statistics."""
    sim_dt = 1.0 / 120.0
    sim_cfg = SimulationCfg(dt=sim_dt, device=args_cli.device, physics=OvPhysxCfg(), gravity=(0.0, 0.0, 0.0))
    with build_simulation_context(device=args_cli.device, sim_cfg=sim_cfg) as sim:
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

        for _ in range(args_cli.warmup_steps):
            sim.step()
            sensor.update(sim_dt, force_recompute=True)
        wp.synchronize_device(sim.device)

        synchronize_device = partial(wp.synchronize_device, sim.device)
        samples = []
        for _ in range(args_cli.num_steps):
            sim.step()
            samples.append(
                measure_latency(
                    operation=lambda: sensor.update(sim_dt, force_recompute=True),
                    synchronize=synchronize_device,
                )
            )

        observer_samples = [
            measure_latency(operation=lambda: None, synchronize=synchronize_device) for _ in range(args_cli.num_steps)
        ]

        # Read-only phase: the blocking per-body native fetches without Warp kernels.
        def read_only() -> None:
            for view, read_buf in zip(sensor._body_views, sensor._body_read_bufs):
                view.read_into(TT.RIGID_BODY_POSE, read_buf)

        read_only_samples = [
            measure_latency(operation=read_only, synchronize=synchronize_device) for _ in range(args_cli.num_steps)
        ]

        target_positions = sensor.data.target_pos_w.torch
        finite_frames = int(torch.isfinite(target_positions).all(dim=-1).sum().item())
        expected_frames = args_cli.num_envs * args_cli.num_target_frames
        if finite_frames != expected_frames:
            raise RuntimeError(f"Expected {expected_frames} finite target frames, received {finite_frames}.")

        benchmark = LatencyBenchmarkRunner(
            benchmark_name="ovphysx_frame_transformer_sensor",
            formatter_type=args_cli.benchmark_formatter,
            output_path=args_cli.output_path,
            metadata={
                "physics_variant": args_cli.physics_variant,
                "label": args_cli.label,
                "device": str(sim.device),
                "num_envs": args_cli.num_envs,
                "target_frames_per_env": args_cli.num_target_frames,
                "tracked_body_views": len(sensor._body_views),
                "num_steps": args_cli.num_steps,
                "warmup_steps": args_cli.warmup_steps,
            },
        )
        full_stats = benchmark.add_latency_samples("sensor_update", samples)
        read_stats = benchmark.add_latency_samples("native_read", read_only_samples)
        benchmark.add_synchronized_samples(
            "observer", "Synchronized Observer Floor", [s.synchronized_s for s in observer_samples]
        )
        benchmark.add_measurement(
            "sensor_update",
            measurement=SingleMeasurement(
                name="Estimated Synchronized Non-read Time",
                value=(full_stats.mean_s - read_stats.mean_s) * 1000.0,
                unit="ms",
            ),
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
    main()
