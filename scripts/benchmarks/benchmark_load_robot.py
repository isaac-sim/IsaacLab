# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to benchmark loading multiple copies of a robot.

.. code-block python

    ./isaaclab.sh -p scripts/benchmarks/benchmark_load_robot.py --num_envs 2048 --robot g1 --headless

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import time

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Benchmark loading different robots.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of robots to simulate.")
parser.add_argument(
    "--robot",
    type=str,
    choices=["anymal_d", "h1", "g1"],
    default="h1",
    help="Choose which robot to load: anymal_d, h1, or g1.",
)
parser.add_argument(
    "--benchmark_backend",
    type=str,
    default="omniperf",
    choices=["json", "osmo", "omniperf"],
    help="Benchmarking backend options, defaults omniperf",
)
parser.add_argument("--output_path", type=str, default=".", help="Path to output benchmark results.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, _ = parser.parse_known_args()

# Start the timer for app start
app_start_time_begin = time.perf_counter_ns()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# End the timer for app start
app_start_time_end = time.perf_counter_ns()

"""Rest everything follows."""

# Start the timer for imports
imports_time_begin = time.perf_counter_ns()

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.test.benchmark import BaseIsaacLabBenchmark, SingleMeasurement
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaaclab_assets import ANYMAL_D_CFG, G1_MINIMAL_CFG, H1_MINIMAL_CFG  # isort:skip


# Stop the timer for imports
imports_time_end = time.perf_counter_ns()

# Create the benchmark
benchmark = BaseIsaacLabBenchmark(
    benchmark_name="benchmark_load_robot",
    backend_type=args_cli.benchmark_backend,
    output_path=args_cli.output_path,
    use_recorders=True,
    output_prefix="benchmark_load_robot",
    workflow_metadata={
        "metadata": [
            {"name": "robot", "data": args_cli.robot},
            {"name": "num_envs", "data": args_cli.num_envs},
        ]
    },
)


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


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
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
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions with some noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
        # Apply random action
        # -- generate random joint efforts
        efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        # -- apply action to the robot
        robot.set_joint_effort_target(efforts)
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


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device="cuda:0")
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    # Start the timer for creating the scene
    setup_time_begin = time.perf_counter_ns()
    # Design scene
    scene_cfg = RobotSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
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
    app_start_time_ms = (app_start_time_end - app_start_time_begin) / 1e6
    imports_time_ms = (imports_time_end - imports_time_begin) / 1e6
    benchmark.add_measurement(
        "startup", measurement=SingleMeasurement(name="App Launch Time", value=app_start_time_ms, unit="ms")
    )
    benchmark.add_measurement(
        "startup", measurement=SingleMeasurement(name="Imports Time", value=imports_time_ms, unit="ms")
    )

    # Run the simulator
    run_simulator(sim, scene)

    # Finalize benchmark
    benchmark.update_manual_recorders()
    benchmark._finalize_impl()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
