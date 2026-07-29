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

from isaaclab.benchmark._cli import parse_positive_int
from isaaclab.benchmark.sensor_suites import add_sensor_benchmark_args

parser = argparse.ArgumentParser(description="Benchmark the OVPhysX FrameTransformer update path.")
add_sensor_benchmark_args(
    parser,
    physics_variants=("ovphysx",),
    default_physics_variant="ovphysx",
    add_device=True,
)
parser.add_argument(
    "--num_target_frames", type=parse_positive_int, default=4, help="Number of target frames per environment."
)
args_cli = parser.parse_args()

import isaaclab_ovphysx.tensor_types as TT
import torch
import warp as wp
from isaaclab_ovphysx.physics import OvPhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.benchmark import LatencyBenchmarkRunner, SingleMeasurement
from isaaclab.benchmark.sensor_suites import add_sensor_latency_measurements, collect_sensor_latency_samples
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

        # Read-only phase: the blocking per-body native fetches without Warp kernels.
        def read_only() -> None:
            for view, read_buf in zip(sensor._body_views, sensor._body_read_bufs):
                view.read_into(TT.RIGID_BODY_POSE, read_buf)

        samples = collect_sensor_latency_samples(
            num_steps=args_cli.num_steps,
            step=sim.step,
            update=lambda: sensor.update(sim_dt, force_recompute=True),
            synchronize=synchronize_device,
            native_read=read_only,
        )

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
        add_sensor_latency_measurements(
            benchmark,
            samples=samples,
            validation=[
                SingleMeasurement(name="Finite Target Frames", value=finite_frames, unit="count"),
                SingleMeasurement(name="Expected Target Frames", value=expected_frames, unit="count"),
            ],
            update_phase="sensor_update",
            observer_phase="observer",
            validation_phase="validation",
        )
        benchmark.finalize()


if __name__ == "__main__":
    main()
