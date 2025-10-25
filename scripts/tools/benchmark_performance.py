# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Benchmarking tool to measure simulation performance across different robot counts and configurations.
This script helps identify performance bottlenecks and optimal configurations for your simulations.
It measures physics update time, rendering time, and total simulation throughput.
Usage:
    # Benchmark default configuration
    ./isaaclab.sh -p scripts/tools/benchmark_performance.py --robot isaaclab_assets.CRAZYFLIE_CFG
    
    # Benchmark with specific counts
    ./isaaclab.sh -p scripts/tools/benchmark_performance.py --robot isaaclab_assets.ANYMAL_D_CFG --counts 1,10,50,100 --iterations 1000
    
    # Export results to CSV
    ./isaaclab.sh -p scripts/tools/benchmark_performance.py --robot isaaclab_assets.CRAZYFLIE_CFG --output benchmark_results.csv
"""
"""Launch Isaac Sim Simulator first."""
import argparse
import csv
import time
from pathlib import Path
from isaaclab.app import AppLauncher
# add argparse arguments
parser = argparse.ArgumentParser(description="Benchmark simulation performance with different robot counts.")
parser.add_argument(
    "--robot",
    type=str,
    required=True,
    help="Robot configuration to benchmark (e.g., 'isaaclab_assets.CRAZYFLIE_CFG')"
)
parser.add_argument(
    "--counts",
    type=str,
    default="1,10,50,100",
    help="Comma-separated list of robot counts to test (default: 1,10,50,100)"
)
parser.add_argument(
    "--iterations",
    type=int,
    default=500,
    help="Number of simulation steps per benchmark (default: 500)"
)
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Output CSV file path for results (optional)"
)
parser.add_argument(
    "--warmup",
    type=int,
    default=100,
    help="Number of warmup steps before measurement (default: 100)"
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
"""Rest everything follows."""
import importlib
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab.utils.timer import Timer
def load_robot_config(config_path: str):
    """Load robot configuration from module path."""
    parts = config_path.rsplit('.', 1)
    if len(parts) != 2:
        raise ValueError(f"Config must be 'module.CONFIG_NAME', got: {config_path}")
    
    module_name, config_name = parts
    module = importlib.import_module(module_name)
    return getattr(module, config_name)
def benchmark_config(robot_cfg, num_robots: int, iterations: int, warmup: int = 100):
    """Benchmark simulation with specified number of robots.
    
    Args:
        robot_cfg: Robot configuration object
        num_robots: Number of robots to spawn
        iterations: Number of simulation steps to measure
        warmup: Number of warmup steps before measurement
        
    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking with {num_robots} robot(s)")
    print(f"{'='*60}")
    
    # Create simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.5])
    
    # Setup scene
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)
    
    # Spawn robots in a grid
    robots = []
    grid_size = int(num_robots ** 0.5) + 1
    spacing = 2.0
    
    for i in range(num_robots):
        x = (i % grid_size) * spacing
        y = (i // grid_size) * spacing
        
        robot_cfg_instance = robot_cfg.replace(prim_path=f"/World/Robot_{i}")
        pos = [x, y, 0.5]
        robot_cfg_instance.spawn.func(
            f"/World/Robot_{i}",
            robot_cfg_instance.spawn,
            translation=pos
        )
        robots.append(Articulation(robot_cfg_instance))
    
    # Reset simulation
    sim.reset()
    
    # Warmup phase
    print(f"Warming up ({warmup} steps)...")
    for _ in range(warmup):
        for robot in robots:
            robot.write_data_to_sim()
        sim.step()
        for robot in robots:
            robot.update(sim_cfg.dt)
    
    # Benchmark phase
    print(f"Running benchmark ({iterations} steps)...")
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()
    
    for _ in range(iterations):
        for robot in robots:
            robot.write_data_to_sim()
        sim.step()
        for robot in robots:
            robot.update(sim_cfg.dt)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.perf_counter()
    
    # Calculate metrics
    total_time = end_time - start_time
    avg_step_time = total_time / iterations
    fps = iterations / total_time
    
    results = {
        'num_robots': num_robots,
        'total_time': total_time,
        'avg_step_time_ms': avg_step_time * 1000,
        'fps': fps,
        'iterations': iterations
    }
    
    print(f"\n[SUCCESS] Results:")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Avg step time: {avg_step_time*1000:.3f}ms")
    print(f"  FPS: {fps:.2f}")
    print(f"  Steps/sec: {fps:.2f}")
    
    return results
def main():
    """Main benchmark function."""
    # Parse robot counts
    counts = [int(x.strip()) for x in args_cli.counts.split(',')]
    
    print("\n" + "="*60)
    print("IsaacLab Performance Benchmark")
    print("="*60)
    print(f"Robot: {args_cli.robot}")
    print(f"Counts: {counts}")
    print(f"Iterations per count: {args_cli.iterations}")
    print(f"Device: {args_cli.device}")
    print("="*60)
    
    # Load robot configuration
    robot_cfg = load_robot_config(args_cli.robot)
    
    # Run benchmarks
    all_results = []
    for count in counts:
        try:
            results = benchmark_config(
                robot_cfg,
                count,
                args_cli.iterations,
                args_cli.warmup
            )
            all_results.append(results)
        except Exception as e:
            print(f"\n[ERROR] Error benchmarking {count} robots: {e}")
            continue
    
    # Print summary
    print("\n" + "="*60)
    print("Benchmark Summary")
    print("="*60)
    print(f"{'Robots':<10} {'Avg Step (ms)':<15} {'FPS':<10} {'Steps/sec':<12}")
    print("-"*60)
    
    for result in all_results:
        print(
            f"{result['num_robots']:<10} "
            f"{result['avg_step_time_ms']:<15.3f} "
            f"{result['fps']:<10.2f} "
            f"{result['fps']:<12.2f}"
        )
    
    # Save to CSV if requested
    if args_cli.output:
        output_path = Path(args_cli.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['num_robots', 'total_time', 'avg_step_time_ms', 'fps', 'iterations'])
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"\n[SUCCESS] Results saved to: {output_path}")
    
    print("\n" + "="*60)
if __name__ == "__main__":
    main()
    simulation_app.close()
