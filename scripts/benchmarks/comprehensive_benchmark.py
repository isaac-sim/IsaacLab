# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Comprehensive benchmark script for multiple environments with different GPU configurations."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import glob


def parse_tf_logs(log_dir: str):
    """Search for the latest tfevents file in log_dir folder and returns
    the tensorboard logs in a dictionary.

    Args:
        log_dir: directory used to search for tfevents files
    """

    # search log directory for latest log file
    list_of_files = glob.glob(f"{log_dir}/events*")  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)

    log_data = {}
    ea = event_accumulator.EventAccumulator(latest_file)
    ea.Reload()
    tags = ea.Tags()["scalars"]
    for tag in tags:
        log_data[tag] = []
        for event in ea.Scalars(tag):
            log_data[tag].append(event.value)

    return log_data


class ComprehensiveBenchmark:
    """Comprehensive benchmarking class for multiple tasks, environments, and GPU configurations."""

    def __init__(self, max_iterations: int = 100, output_dir: str = "benchmark_results"):
        """Initialize the benchmark.

        Args:
            max_iterations: Number of training iterations to run
            output_dir: Directory to store benchmark results
        """
        self.max_iterations = max_iterations
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Define task configurations
        self.task_configs = {
            # new batch of benchmark environments
            #COMPARISON ENVIRONMENTS FOR ManagerBased vs Direct
            # "Isaac-Velocity-Rough-Anymal-C-v0": {  # ManagerBased
            #     "enable_cameras": False,
            #     "env_counts": [2048, 4096, 8192, 16384],
            #     "training_scripts": ["rsl_rl"],
            # },
            # "Isaac-Velocity-Rough-Anymal-C-Direct-v0": {  # Direct
            #     "enable_cameras": False,
            #     "env_counts": [2048, 4096, 8192, 16384],
            #     "training_scripts": ["rsl_rl"],
            # },
            # "Isaac-Open-Drawer-Franka-v0": {  # ManagerBased
            #     "enable_cameras": False,
            #     "env_counts": [2048, 4096, 8192, 16384],
            #     "training_scripts": ["rsl_rl"],
            # },
            # "Isaac-Franka-Cabinet-Direct-v0": {  # Direct
            #     "enable_cameras": False,
            #     "env_counts": [2048, 4096, 8192, 16384],
            #     "training_scripts": ["rsl_rl"],
            # },
            # "Isaac-Velocity-Rough-Digit-v0": {
            #     "enable_cameras": False,
            #     "env_counts": [2048, 4096, 8192, 16384],
            #     "training_scripts": ["rsl_rl"],
            # },
            # "Isaac-Factory-GearMesh-Direct-v0": {
            #     "enable_cameras": False,
            #     "env_counts": [1024, 2048, 4096],
            #     "training_scripts": ["rl_games"],
            #     "max_iterations": 25,
            # },            

            # # Camera-enabled task
            # "Isaac-Dexsuite-Kuka-Allegro-Lift-Depth-TiledCamera-v0": {
            #     "enable_cameras": True,
            #     "env_counts": [1024, 2048, 4096],
            #     "training_scripts": ["rl_games"],
            #     "extra": "--rendering_mode performance"
            # },
            # "Isaac-Dexsuite-Kuka-Allegro-Lift-Depth-RayCasterCamera-v0": {
            #     "enable_cameras": True,
            #     "env_counts": [1024, 2048, 4096],
            #     "training_scripts": ["rl_games"],
            #     "extra": "--rendering_mode performance"
            # },
            # "Isaac-Navigation-Anymal-C-RayCaster-v0": {
            #     "enable_cameras": True,
            #     "env_counts": [1024, 2048, 4096],
            #     "training_scripts": ["rsl_rl"],
            # },
            "Isaac-Navigation-Anymal-C-Tiled-v0": {
                "enable_cameras": True,
                "env_counts": [1024, 2048, 4096],
                "training_scripts": ["rsl_rl"],
            },
            # # Non-camera tasks
            # "Isaac-Dexsuite-Kuka-Allegro-Reorient-v0": {
            #     "enable_cameras": False,
            #     "env_counts": [2048, 4096, 8192, 16384],
            #     "training_scripts": ["rsl_rl"],
            # },
            # "Isaac-Dexsuite-Kuka-Allegro-Lift-v0": {
            #     "enable_cameras": False,
            #     "env_counts": [2048, 4096, 8192, 16384],
            #     "training_scripts": ["rsl_rl"],
            # },
            # "Isaac-Velocity-Rough-Anymal-D-v0": {
            #     "enable_cameras": False,
            #     "env_counts": [2048, 4096, 8192, 16384],
            #     "training_scripts": ["rsl_rl"],
            # },
            # "Isaac-Velocity-Rough-G1-v0": {
            #     "enable_cameras": False,
            #     "env_counts": [2048, 4096, 8192, 16384],
            #     "training_scripts": ["rsl_rl"],
            # },
            # "Isaac-Repose-Cube-Shadow-Direct-v0": {
            #     "enable_cameras": False,
            #     "env_counts": [2048, 4096, 8192, 16384],
            #     "training_scripts": ["rsl_rl", "rl_games"],
            # },
            # "Isaac-Repose-Cube-Shadow-OpenAI-LSTM-Direct-v0": {
            #     "enable_cameras": False,
            #     "env_counts": [2048, 4096, 8192, 16384],
            #     "training_scripts": ["rl_games"],
            # },
            
            
        }

        self.gpu_counts = [4, 8]
        self.results = []

    def _build_command(self, task: str, num_envs: int, num_gpus: int, training_script: str) -> list[str]:
        """Build the command to run the benchmark.

        Args:
            task: Task name
            num_envs: Number of environments
            num_gpus: Number of GPUs
            training_script: Training script type ("rsl_rl" or "rl_games")

        Returns:
            Command as list of strings
        """
        config = self.task_configs[task]
        
        # Use task-specific max_iterations if available, otherwise use global default
        max_iterations = config.get("max_iterations", self.max_iterations)

        base_script_path = f"scripts/reinforcement_learning/{training_script}/train.py"

        if num_gpus == 1:
            # Single GPU command
            cmd = [
                "./isaaclab.sh",
                "-p",
                base_script_path,
                f"--task={task}",
                f"--num_envs={num_envs}",
                f"--max_iterations={max_iterations}",
                "--headless",
            ]
        else:
            # Multi-GPU command
            cmd = [
                "./isaaclab.sh",
                "-p",
                "-m",
                "torch.distributed.run",
                "--nnodes=1",
                f"--nproc_per_node={num_gpus}",
                base_script_path,
                f"--task={task}",
                f"--num_envs={num_envs}",
                f"--max_iterations={max_iterations}",
                "--headless",
                "--distributed",
            ]

        # Add camera flag if required
        if config["enable_cameras"]:
            cmd.append("--enable_cameras")
        
        if "extra" in config:
            cmd.extend(config["extra"].split())

        return cmd

    def _extract_fps_from_logs(self, log_dir: str, training_script: str, num_gpus: int) -> dict:
        """Extract FPS statistics from training logs.

        Args:
            log_dir: Directory containing the logs
            training_script: Training script type ("rsl_rl" or "rl_games")
            num_gpus: Number of GPUs used (for RSL RL scaling)

        Returns:
            Dictionary with FPS statistics (min, max, mean, std), or None if not found
        """
        try:
            # Import here to avoid issues with app launcher
            sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

            print(f"DEBUG: Extracting FPS from {log_dir} for {training_script}")
            
            if not os.path.exists(log_dir):
                print(f"DEBUG: Log directory does not exist: {log_dir}")
                return None

            if training_script == "rsl_rl":
                # Look for RSL-RL logs
                print(f"DEBUG: Looking for tfevents files in: {log_dir}")
                
                # Check if there are any tfevents files
                import glob
                import numpy as np
                log_folders = glob.glob(f"{log_dir}/*")
                # find most recent folder
                if not log_folders:
                    print("DEBUG: No log folders found")
                    return None
                
                latest_log_folder = max(log_folders, key=os.path.getmtime)
                tfevents_files = glob.glob(f"{latest_log_folder}/events*")
                print(f"DEBUG: Found tfevents files: {tfevents_files}")
                
                log_data = parse_tf_logs(latest_log_folder)
                print(f"DEBUG: Available log keys: {list(log_data.keys())}")
                
                if "Perf/total_fps" in log_data and log_data["Perf/total_fps"]:
                    fps_values = np.array(log_data["Perf/total_fps"]) * num_gpus  # Scale by number of GPUs
                    fps_stats = {
                        "min_fps": float(np.min(fps_values)),
                        "max_fps": float(np.max(fps_values)),
                        "mean_fps": float(np.mean(fps_values)),
                        "std_fps": float(np.std(fps_values)),
                        "sample_count": len(fps_values)
                    }
                    print(f"DEBUG: Found Perf/total_fps with stats (scaled by {num_gpus} GPUs): {fps_stats}")
                    return fps_stats
                else:
                    print("DEBUG: Perf/total_fps not found or empty in log data")
                    
            elif training_script == "rl_games":
                # Check if there are any tfevents files
                import glob
                import numpy as np

                print(f"DEBUG: Found latest folder: {log_dir}")

                # Look for RL-Games logs in summaries subdirectory
                summaries_dir = os.path.join(log_dir, "summaries")
                print(f"DEBUG: Looking for RL-Games logs in: {summaries_dir}")
                
                if os.path.exists(summaries_dir):
                    import glob
                    tfevents_files = glob.glob(f"{summaries_dir}/events*")
                    print(f"DEBUG: Found tfevents files in summaries: {tfevents_files}")
                    
                    if not tfevents_files:
                        print("DEBUG: No tfevents files found in summaries")
                        return None
                    
                    log_data = parse_tf_logs(summaries_dir)
                    print(f"DEBUG: Available log keys: {list(log_data.keys())}")
                    
                    if (
                        "performance/step_inference_rl_update_fps" in log_data
                        and log_data["performance/step_inference_rl_update_fps"]
                    ):
                        fps_values = np.array(log_data["performance/step_inference_rl_update_fps"])
                        fps_stats = {
                            "min_fps": float(np.min(fps_values)),
                            "max_fps": float(np.max(fps_values)),
                            "mean_fps": float(np.mean(fps_values)),
                            "std_fps": float(np.std(fps_values)),
                            "sample_count": len(fps_values)
                        }
                        print(f"DEBUG: Found performance/step_inference_rl_update_fps with stats: {fps_stats}")
                        return fps_stats
                    else:
                        print("DEBUG: performance/step_inference_rl_update_fps not found or empty in log data")
                else:
                    print(f"DEBUG: Summaries directory does not exist: {summaries_dir}")

            return None
        except Exception as e:
            print(f"Error extracting FPS from logs in {log_dir}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _run_single_benchmark(self, task: str, num_envs: int, num_gpus: int, training_script: str) -> dict:
        """Run a single benchmark configuration.

        Args:
            task: Task name
            num_envs: Number of environments
            num_gpus: Number of GPUs
            training_script: Training script type

        Returns:
            Dictionary with benchmark results
        """
        print(f"Running benchmark: {task} | {num_envs} envs | {num_gpus} GPUs | {training_script}")

        cmd = self._build_command(task, num_envs, num_gpus, training_script)

        # Create unique run identifier
        run_id = f"{task}_{num_envs}envs_{num_gpus}gpus_{training_script}_{int(time.time())}"

        # Set environment variable for logging
        env = os.environ.copy()

        result = {
            "task": task,
            "num_envs": num_envs,
            "num_gpus": num_gpus,
            "training_script": training_script,
            "fps": -1,
            "fps_stats": None,
            "status": "failed",
            "command": " ".join(cmd),
            "run_id": run_id,
        }

        try:
            # Run the command
            print(f"Executing: {' '.join(cmd)}")
            start_time = time.time()

            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=os.getcwd()
            )

            stdout, stderr = process.communicate(timeout=3600)  # 1 hour timeout
            end_time = time.time()

            if process.returncode == 0:
                result["status"] = "completed"
                result["duration"] = end_time - start_time

                # Try to extract FPS from logs
                # Look for the most recent log directory
                if training_script == "rsl_rl":
                    log_base_dir = "logs/rsl_rl"
                else:
                    log_base_dir = "logs/rl_games"

                if os.path.exists(log_base_dir):
                    # Find the most recent log directory
                    log_dirs = [d for d in os.listdir(log_base_dir) if os.path.isdir(os.path.join(log_base_dir, d))]
                    if log_dirs:
                        # Get the most recent directory by modification time
                        latest_log_dir = max(log_dirs, key=lambda x: os.path.getmtime(os.path.join(log_base_dir, x)))
                        full_log_path = os.path.join(log_base_dir, latest_log_dir)

                        # For RL-Games, look for subdirectories
                        if training_script == "rl_games":
                            subdirs = [
                                d for d in os.listdir(full_log_path) if os.path.isdir(os.path.join(full_log_path, d))
                            ]
                            if subdirs:
                                latest_subdir = max(
                                    subdirs, key=lambda x: os.path.getmtime(os.path.join(full_log_path, x))
                                )
                                full_log_path = os.path.join(full_log_path, latest_subdir)

                        fps_stats = self._extract_fps_from_logs(full_log_path, training_script, num_gpus)
                        if fps_stats:
                            result["fps_stats"] = fps_stats
                            result["fps"] = fps_stats["mean_fps"]  # For backward compatibility
                        else:
                            result["fps_stats"] = None
                            result["fps"] = -1
                        result["log_path"] = full_log_path

                if result['fps_stats']:
                    stats = result['fps_stats']
                    print(f"✓ Completed successfully. FPS Stats - Mean: {stats['mean_fps']:.1f}, Min: {stats['min_fps']:.1f}, Max: {stats['max_fps']:.1f}, Std: {stats['std_fps']:.1f}")
                else:
                    print(f"✓ Completed successfully. FPS: {result['fps']}")
            else:
                result["error"] = stderr
                print(f"✗ Failed with return code {process.returncode}")
                print(f"Error: {stderr}")

        except subprocess.TimeoutExpired:
            process.kill()
            result["error"] = "Timeout after 1 hour"
            print("✗ Failed: Timeout after 1 hour")
        except Exception as e:
            result["error"] = str(e)
            print(f"✗ Failed with exception: {e}")

        return result

    def run_all_benchmarks(self) -> None:
        """Run all benchmark configurations."""
        total_configs = sum(
            len(config["env_counts"]) * len(self.gpu_counts) * len(config["training_scripts"])
            for config in self.task_configs.values()
        )

        print(f"Starting comprehensive benchmark with {total_configs} configurations...")
        print(f"Max iterations per run: {self.max_iterations}")
        print(f"Results will be saved to: {self.output_dir}/benchmark_results.csv")
        print("=" * 80)

        config_count = 0

        for task, config in self.task_configs.items():
            for num_envs in config["env_counts"]:
                for num_gpus in self.gpu_counts:
                    for training_script in config["training_scripts"]:
                        config_count += 1
                        print(f"\nConfiguration {config_count}/{total_configs}")
                        print("-" * 40)

                        result = self._run_single_benchmark(task, num_envs, num_gpus, training_script)
                        self.results.append(result)

                        # Save intermediate results
                        self._save_results()

        print("\n" + "=" * 80)
        print("Benchmark completed!")
        self._generate_report()

    def _save_results(self) -> None:
        """Save results to CSV file."""
        # Create DataFrame and save as CSV
        df = pd.DataFrame(self.results)
        csv_file = self.output_dir / "benchmark_results.csv"
        df.to_csv(csv_file, index=False)
        print(f"Results saved: {csv_file}")

    def _generate_report(self) -> None:
        """Generate console summary."""
        # Create DataFrame for analysis
        df = pd.DataFrame(self.results)
        
        # Calculate statistics
        successful_runs = len(df[df["status"] == "completed"])
        success_rate = (successful_runs / len(df)) * 100

        # Print summary to console
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        print(f"Total configurations: {len(self.results)}")
        print(f"Successful runs: {successful_runs}")
        print(f"Success rate: {success_rate:.1f}%")

        if not df[df["fps"] > 0].empty:
            fps_data = df[df["fps"] > 0]["fps"]
            avg_fps = fps_data.mean()
            min_fps = fps_data.min()
            max_fps = fps_data.max()
            std_fps = fps_data.std()
            
            print("\nFPS Statistics:")
            print(f"  Average FPS: {avg_fps:.1f}")
            print(f"  Standard Deviation: {std_fps:.1f}")
            print(f"  Minimum FPS: {min_fps:.1f}")
            print(f"  Maximum FPS: {max_fps:.1f}")
            
            best_overall = df[df["fps"] > 0].loc[df["fps"].idxmax()]
            print("\nBest performance:")
            print(f"  Task: {best_overall['task']}")
            print(f"  FPS: {best_overall['fps']:.1f}")
            print(
                f"  Configuration: {best_overall['num_envs']} envs, {best_overall['num_gpus']} GPUs,"
                f" {best_overall['training_script']}"
            )


def main():
    """Main function to run the comprehensive benchmark."""
    parser = argparse.ArgumentParser(description="Comprehensive benchmark for multiple tasks and configurations")
    parser.add_argument(
        "--max_iterations", type=int, default=100, help="Number of training iterations per run (default: 5000)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/benchmarks",
        help="Directory to store results (default: benchmark_results)",
    )
    parser.add_argument("--tasks", nargs="+", help="Specific tasks to benchmark (default: all tasks)")
    parser.add_argument("--gpus", nargs="+", type=int, help="Specific GPU counts to test (default: 1,2,4,8)")

    args = parser.parse_args()

    # Create benchmark instance
    benchmark = ComprehensiveBenchmark(max_iterations=args.max_iterations, output_dir=args.output_dir)

    # Filter tasks if specified
    if args.tasks:
        benchmark.task_configs = {task: config for task, config in benchmark.task_configs.items() if task in args.tasks}

    # Filter GPU counts if specified
    if args.gpus:
        benchmark.gpu_counts = args.gpus

    # Run benchmarks
    benchmark.run_all_benchmarks()


if __name__ == "__main__":
    main()
