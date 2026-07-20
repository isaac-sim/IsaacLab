# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark per-articulation actuator dispatch cost on the Newton backend.

Actuator models are computed once per articulation group per physics step, so their
host-side dispatch cost scales with the number of articulations in the scene rather
than the number of environments. This benchmark spawns many copies of ANYmal-D per
environment, cycling through the analytic actuator models (DC motor, ideal PD, and
solver-implicit PD), and measures the steady-state per-step time.

Every environment has an identical composition, so the scene runs on the stock
Newton MJWarp solver without any multi-world grouping.

.. code-block:: bash

    # Default composition: 8 ANYmal-D per environment, 1024 environments.
    ./isaaclab.sh -p scripts/benchmarks/benchmark_actuator_dispatch.py --headless

    # Scale the number of articulations per environment.
    ./isaaclab.sh -p scripts/benchmarks/benchmark_actuator_dispatch.py --headless \\
        --num_envs 1024 --num_articulations 16

The simulation app is launched through :func:`isaaclab.app.launch_simulation` with a
Newton MJWarp physics configuration.
"""

"""Configure the simulation launch first."""

import argparse
import math
import time
from typing import TYPE_CHECKING

from isaaclab.app import add_launcher_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Benchmark per-articulation actuator dispatch cost.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments.")
parser.add_argument(
    "--num_articulations", type=int, default=8, help="Number of ANYmal-D articulations per environment."
)
parser.add_argument("--num_frames", type=int, default=500, help="Number of timed physics steps.")
parser.add_argument(
    "--warmup_frames", type=int, default=100, help="Untimed physics steps before measurement (covers graph capture)."
)
parser.add_argument(
    "--benchmark_formatter",
    type=str,
    default="summary",
    choices=["json", "osmo", "omniperf", "summary"],
    help="Benchmark output formatter.",
)
parser.add_argument("--output_path", type=str, default=".", help="Path to output benchmark results.")
# append launcher cli args
add_launcher_args(parser)
# parse the arguments
args_cli, _ = parser.parse_known_args()
if args_cli.num_frames < 1 or args_cli.warmup_frames < 0:
    parser.error("--num_frames must be >= 1 and --warmup_frames must be >= 0.")

"""Rest everything follows."""

import torch

# Note: only configuration classes are imported at module scope. The ``isaaclab.sim`` package uses lazy
# exports, so importing it (and the ``*Cfg`` classes below) does not pull in USD/``omni`` bindings. The
# runtime classes (``SimulationContext``, ``InteractiveScene``) eagerly import ``pxr`` and must therefore
# be imported only after the simulation app has been launched (see :func:`main`).
import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorBaseCfg, IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.app import launch_simulation
from isaaclab.assets import AssetBaseCfg
from isaaclab.physics import PhysicsCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.test.benchmark import BaseIsaacLabBenchmark, SingleMeasurement
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationContext

##
# Pre-defined configs
##
from isaaclab_assets import ANYDRIVE_3_SIMPLE_ACTUATOR_CFG, ANYMAL_D_CFG  # isort:skip
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg  # isort:skip


# Analytic actuator models cycled across the articulation copies, in order: scenes with
# fewer articulations than variants use the leading entries. All variants share the
# ANYdrive 3 gains and effort limit; only the actuator model computing the torque differs.
ACTUATOR_VARIANTS: dict[str, ActuatorBaseCfg] = {
    "dc_motor": ANYDRIVE_3_SIMPLE_ACTUATOR_CFG,
    "ideal_pd": IdealPDActuatorCfg(
        joint_names_expr=[".*"], stiffness=40.0, damping=5.0, effort_limit=80.0, velocity_limit=7.5
    ),
    "implicit": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=40.0, damping=5.0, effort_limit_sim=80.0),
}


@configclass
class ActuatorDispatchSceneCfg(InteractiveSceneCfg):
    """Scene with a grid of ANYmal-D robots per environment; articulations are added dynamically."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )


def build_scene_cfg(num_envs: int, num_articulations: int) -> ActuatorDispatchSceneCfg:
    """Place ``num_articulations`` ANYmal-D copies on a grid inside every environment.

    Args:
        num_envs: Number of environments.
        num_articulations: Number of robot copies per environment.

    Returns:
        The populated scene configuration.
    """
    grid_side = math.ceil(math.sqrt(num_articulations))
    grid_pitch = 1.8
    scene_cfg = ActuatorDispatchSceneCfg(num_envs=num_envs, env_spacing=grid_side * grid_pitch + 2.0)
    variants = list(ACTUATOR_VARIANTS.items())
    for i in range(num_articulations):
        variant_name, actuator_cfg = variants[i % len(variants)]
        pos_x = (i % grid_side - (grid_side - 1) / 2) * grid_pitch
        pos_y = (i // grid_side - (grid_side - 1) / 2) * grid_pitch
        robot_cfg = ANYMAL_D_CFG.replace(
            prim_path=f"{{ENV_REGEX_NS}}/Robot_{i:02d}_{variant_name}",
            init_state=ANYMAL_D_CFG.init_state.replace(pos=(pos_x, pos_y, 0.6)),
            actuators={"legs": actuator_cfg.replace()},
        )
        setattr(scene_cfg, f"robot_{i:02d}_{variant_name}", robot_cfg)
    return scene_cfg


def run_simulator(sim: "SimulationContext", scene: "InteractiveScene", benchmark: BaseIsaacLabBenchmark):
    """Step the scene and measure the steady-state per-step time."""
    robots = [scene[name] for name in scene.articulations]
    default_targets = [robot.data.default_joint_pos.torch.clone() for robot in robots]
    sim_dt = sim.get_physics_dt()
    sync_device = torch.device(sim.device) if "cuda" in sim.device else None

    num_steps = args_cli.warmup_frames + args_cli.num_frames
    step_time_begin = 0
    for count in range(num_steps):
        if count == args_cli.warmup_frames:
            if sync_device is not None:
                torch.cuda.synchronize(sync_device)
            step_time_begin = time.perf_counter_ns()
        # command a small joint-space sway around the standing pose
        offset = 0.05 * math.sin(2.0 * math.pi * 0.5 * count * sim_dt)
        for robot, targets in zip(robots, default_targets):
            robot.set_joint_position_target_index(target=targets + offset)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
    if sync_device is not None:
        torch.cuda.synchronize(sync_device)
    step_time_end = time.perf_counter_ns()

    per_step_time_ms = (step_time_end - step_time_begin) / args_cli.num_frames / 1e6
    env_steps_per_second = args_cli.num_envs / (per_step_time_ms / 1e3)
    benchmark.add_measurement(
        "runtime", measurement=SingleMeasurement(name="Per Step Time", value=per_step_time_ms, unit="ms")
    )
    benchmark.add_measurement(
        "runtime",
        measurement=SingleMeasurement(name="Env Steps Per Second", value=env_steps_per_second, unit="steps/s"),
    )
    print(
        f"[RESULT] num_envs={args_cli.num_envs} num_articulations={args_cli.num_articulations}"
        f" per_step_ms={per_step_time_ms:.3f} env_steps_per_second={env_steps_per_second:.0f}"
    )


def main(scene_cfg: ActuatorDispatchSceneCfg, physics_cfg: PhysicsCfg, benchmark: BaseIsaacLabBenchmark):
    """Create the simulation and scene, then run the benchmark loop."""
    # Import runtime classes only now that the simulation app has been launched. These modules import
    # USD/``omni`` bindings at import time, so importing them before the app is running crashes the simulator.
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationContext

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 200.0, device=args_cli.device, physics=physics_cfg)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([4.0, 0.0, 5.0], [0.0, 0.0, 1.0])

    setup_time_begin = time.perf_counter_ns()
    scene = InteractiveScene(scene_cfg)
    setup_time_end = time.perf_counter_ns()
    benchmark.add_measurement(
        "startup",
        measurement=SingleMeasurement(
            name="Scene Creation Time", value=(setup_time_end - setup_time_begin) / 1e6, unit="ms"
        ),
    )

    reset_time_begin = time.perf_counter_ns()
    sim.reset()
    reset_time_end = time.perf_counter_ns()
    benchmark.add_measurement(
        "startup",
        measurement=SingleMeasurement(
            name="Simulation Start Time", value=(reset_time_end - reset_time_begin) / 1e6, unit="ms"
        ),
    )

    run_simulator(sim, scene, benchmark)

    benchmark.update_manual_recorders()
    benchmark._finalize_impl()


if __name__ == "__main__":
    # Build the scene configuration before launching the app so config errors surface early.
    scene_cfg = build_scene_cfg(args_cli.num_envs, args_cli.num_articulations)

    # Contact capacity scales with the number of robots per world; all other solver
    # settings stay at their defaults.
    solver_cfg = MJWarpSolverCfg(nconmax=12 * args_cli.num_articulations, njmax=48 * args_cli.num_articulations)
    with launch_simulation(NewtonCfg(solver_cfg=solver_cfg), args_cli) as physics_cfg:
        if physics_cfg is None:
            raise RuntimeError("No physics backend was selected.")
        benchmark = BaseIsaacLabBenchmark(
            benchmark_name="benchmark_actuator_dispatch",
            formatter_type=args_cli.benchmark_formatter,
            output_path=args_cli.output_path,
            use_recorders=True,
            # The frametime recorder extension activates Isaac Sim's bundled Warp, which conflicts
            # with the newer Warp version required by Newton.
            frametime_recorders=False,
            output_prefix="benchmark_actuator_dispatch",
            workflow_metadata={
                "metadata": [
                    {"name": "num_envs", "data": args_cli.num_envs},
                    {"name": "num_articulations", "data": args_cli.num_articulations},
                ]
            },
        )
        main(scene_cfg, physics_cfg, benchmark)
