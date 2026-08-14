# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to benchmark loading multiple copies of a robot.

.. code-block:: bash

    uv run python scripts/benchmarks/benchmark_load_robot.py --num_envs 2048 --robot g1

    uv run python scripts/benchmarks/benchmark_load_robot.py --physics newton_mjwarp --robot h1

The simulation app is launched through :func:`isaaclab.app.launch_simulation`, which selects the requested
physics backend while the benchmark uses the full ``isaaclab.python.kit`` experience for consistent services.

"""

"""Configure the simulation launch first."""

import argparse
import time
from typing import TYPE_CHECKING

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Benchmark loading different robots.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of robots to simulate.")
parser.add_argument("--physics", default="physx", choices=["physx", "newton_mjwarp"], help="Physics backend.")
parser.add_argument(
    "--robot",
    type=str,
    choices=["anymal_d", "h1", "g1"],
    default="h1",
    help="Choose which robot to load: anymal_d, h1, or g1.",
)
parser.add_argument(
    "--benchmark_formatter",
    type=str,
    default="omniperf",
    choices=["json", "osmo", "omniperf", "summary"],
    help="Benchmark output formatter, defaults omniperf",
)
parser.add_argument("--output_path", type=str, default=".", help="Path to output benchmark results.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, _ = parser.parse_known_args()

"""Rest everything follows."""

# Start the timer for imports
imports_time_begin = time.perf_counter_ns()

import torch

# Note: only configuration classes are imported at module scope. The ``isaaclab.sim`` package uses lazy
# exports, so importing it (and the ``*Cfg`` classes below) does not pull in USD/``omni`` bindings. The
# runtime classes (``SimulationContext``, ``InteractiveScene``) eagerly import ``pxr`` and must therefore
# be imported only after the simulation app has been launched (see :func:`main`).
import isaaclab.sim as sim_utils
from isaaclab.app import launch_simulation
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.benchmark import BaseIsaacLabBenchmark, SingleMeasurement
from isaaclab.physics import PhysicsCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationContext

##
# Pre-defined configs
##
from isaaclab_assets import ANYMAL_D_CFG, G1_MINIMAL_CFG, H1_MINIMAL_CFG  # isort:skip
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg  # isort:skip


# Stop the timer for imports
imports_time_end = time.perf_counter_ns()


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for a simple scene with a robot."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    if args_cli.robot == "h1":
        robot: ArticulationCfg = H1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "g1":
        robot: ArticulationCfg = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "anymal_d":
        robot: ArticulationCfg = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError(f"Unsupported robot type: {args_cli.robot}.")


def run_simulator(sim: "SimulationContext", scene: "InteractiveScene", benchmark: BaseIsaacLabBenchmark):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    # Start the timer for creating the scene
    step_time_begin = time.perf_counter_ns()
    num_steps = 2000

    # Simulation loop
    for count in range(num_steps):
        # Reset
        if count % 500 == 0:
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_pose = robot.data.default_root_pose.torch.clone()
            root_pose[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim_index(root_pose=root_pose)
            root_vel = robot.data.default_root_vel.torch.clone()
            robot.write_root_velocity_to_sim_index(root_velocity=root_vel)
            # set joint positions with some noise
            joint_pos, joint_vel = (
                robot.data.default_joint_pos.torch.clone(),
                robot.data.default_joint_vel.torch.clone(),
            )
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_position_to_sim_index(position=joint_pos)
            robot.write_joint_velocity_to_sim_index(velocity=joint_vel)
            # clear internal buffers
            scene.reset()
        # Apply random action
        # -- generate random joint efforts
        efforts = torch.randn_like(robot.data.joint_pos.torch) * 5.0
        # -- apply action to the robot
        robot.set_joint_effort_target_index(target=efforts)
        # -- write data to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Update buffers
        scene.update(sim_dt)

    # Stop the timer for reset
    step_time_end = time.perf_counter_ns()
    per_step_time_ms = (step_time_end - step_time_begin) / num_steps / 1e6

    # Log per step time to benchmark
    benchmark.add_measurement(
        "runtime", measurement=SingleMeasurement(name="Per Step Time", value=per_step_time_ms, unit="ms")
    )


def main(
    scene_cfg: RobotSceneCfg,
    physics_cfg: PhysicsCfg,
    app_start_time_ms: float,
    benchmark: BaseIsaacLabBenchmark,
):
    """Main function."""
    # Import runtime classes only now that the simulation app has been launched. These modules import
    # USD/``omni`` bindings at import time, so importing them before the app is running crashes the simulator.
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationContext

    # Load kit helper
    # The default MJWarp configuration needs additional constraint capacity and solver tuning for bipeds.
    if (
        args_cli.robot in ("h1", "g1")
        and isinstance(physics_cfg, NewtonCfg)
        and isinstance(physics_cfg.solver_cfg, MJWarpSolverCfg)
    ):
        physics_cfg.solver_cfg.njmax = 70
        physics_cfg.solver_cfg.nconmax = 70
        physics_cfg.solver_cfg.ls_iterations = 40
        physics_cfg.solver_cfg.cone = "elliptic"
        physics_cfg.solver_cfg.impratio = 100
        physics_cfg.solver_cfg.ls_parallel = False
        physics_cfg.solver_cfg.integrator = "implicitfast"
        physics_cfg.num_substeps = 2

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, physics=physics_cfg)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    # Start the timer for creating the scene
    setup_time_begin = time.perf_counter_ns()
    # Design scene
    scene = InteractiveScene(scene_cfg)
    # Stop the timer for creating the scene
    setup_time_end = time.perf_counter_ns()

    # Log scene creation time
    scene_creation_time_ms = (setup_time_end - setup_time_begin) / 1e6
    benchmark.add_measurement(
        "startup", measurement=SingleMeasurement(name="Scene Creation Time", value=scene_creation_time_ms, unit="ms")
    )

    # Start the timer for reset
    reset_time_begin = time.perf_counter_ns()
    # Play the simulator
    sim.reset()
    # Stop the timer for reset
    reset_time_end = time.perf_counter_ns()

    # Log simulation start time
    sim_start_time_ms = (reset_time_end - reset_time_begin) / 1e6
    benchmark.add_measurement(
        "startup", measurement=SingleMeasurement(name="Simulation Start Time", value=sim_start_time_ms, unit="ms")
    )

    # Log app start and imports time
    imports_time_ms = (imports_time_end - imports_time_begin) / 1e6
    benchmark.add_measurement(
        "startup", measurement=SingleMeasurement(name="App Launch Time", value=app_start_time_ms, unit="ms")
    )
    benchmark.add_measurement(
        "startup", measurement=SingleMeasurement(name="Imports Time", value=imports_time_ms, unit="ms")
    )

    # Run the simulator
    run_simulator(sim, scene, benchmark)

    # Finalize benchmark
    benchmark.update_manual_recorders()
    benchmark._finalize_impl()


if __name__ == "__main__":
    # Use the full Kit experience in both headless and windowed modes for a consistent runtime.
    # An explicit CLI override still takes precedence.
    if not args_cli.experience:
        args_cli.experience = "isaaclab.python.kit"

    # Build the scene configuration before measuring app startup so scene-config imports are measured separately.
    scene_cfg = RobotSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)

    # Start the timer for app start
    app_start_time_begin = time.perf_counter_ns()
    # Launch the selected physics backend in the benchmark's Kit experience.
    with launch_simulation(PhysicsCfg(), args_cli) as physics_cfg:
        if physics_cfg is None:
            raise RuntimeError("No physics backend was selected.")
        # End the timer for app start
        app_start_time_end = time.perf_counter_ns()
        app_start_time_ms = (app_start_time_end - app_start_time_begin) / 1e6
        # Create the benchmark after Kit startup, matching the runtime benchmark lifecycle.
        formatter_types = [value.strip() for value in args_cli.benchmark_formatter.split(",") if value.strip()]
        formatter_types = formatter_types or ["omniperf"]
        benchmark = BaseIsaacLabBenchmark(
            benchmark_name="benchmark_load_robot",
            formatter_type=args_cli.benchmark_formatter,
            output_path=args_cli.output_path,
            use_recorders=True,
            # Isaac Sim's frametime recorder extension activates its bundled Warp, which conflicts with
            # the newer Warp version required by Newton. Other benchmark recorders remain backend-independent.
            frametime_recorders=args_cli.physics == "physx"
            and any(t in ("summary", "omniperf") for t in formatter_types),
            output_prefix="benchmark_load_robot",
            workflow_metadata={
                "metadata": [
                    {"name": "robot", "data": args_cli.robot},
                    {"name": "physics", "data": args_cli.physics},
                    {"name": "num_envs", "data": args_cli.num_envs},
                ]
            },
        )
        # run the main function
        main(scene_cfg, physics_cfg, app_start_time_ms, benchmark)
