# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Script to benchmark RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

# add argparse arguments
parser = argparse.ArgumentParser(description="Benchmark RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=10, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument(
    "--benchmark_backend",
    type=str,
    default="OmniPerfKPIFile",
    choices=["LocalLogMetrics", "JSONFileMetrics", "OsmoKPIFile", "OmniPerfKPIFile"],
    help="Benchmarking backend options, defaults OmniPerfKPIFile",
)
parser.add_argument(
    "--output_folder",
    type=str,
    default="/tmp",
    help="Output folder for the benchmark metrics.",
)
parser.add_argument(
    "--kit",
    action="store_true",
    default=False,
    help="Enable Isaac Sim Kit and use isaacsim.benchmark.services. Default: False (uses standalone benchmark).",
)

# Conditionally add RSL-RL and AppLauncher args only if --kit is enabled
if "--kit" in sys.argv:
    import scripts.reinforcement_learning.rsl_rl.cli_args as cli_args

    cli_args.add_rsl_rl_args(parser)

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)

# to ensure kit args don't break the benchmark arg parsing
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

app_start_time_begin = time.perf_counter_ns()

# Conditionally launch Isaac Sim
simulation_app = None
app_launcher = None
if args_cli.kit:
    # Force Omniverse mode by setting environment variable
    # This ensures SimulationApp is launched even without explicit visualizers
    os.environ["LAUNCH_OV_APP"] = "1"

    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app


app_start_time_end = time.perf_counter_ns()

imports_time_begin = time.perf_counter_ns()

import gymnasium as gym
import numpy as np
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

imports_time_end = time.perf_counter_ns()

# Import benchmark infrastructure based on kit flag
if args_cli.kit:
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.benchmark.services")

    # Set the benchmark settings according to the inputs
    import carb

    settings = carb.settings.get_settings()
    settings.set("/exts/isaacsim.benchmark.services/metrics/metrics_output_folder", args_cli.output_folder)
    settings.set("/exts/isaacsim.benchmark.services/metrics/randomize_filename_prefix", True)

    from isaacsim.benchmark.services import BaseIsaacBenchmark

    from scripts.benchmarks.utils.benchmark_utils import create_kit_logging_functions, get_timer_value

    # Get all logging functions for kit mode
    log_funcs = create_kit_logging_functions()
else:
    # Use standalone benchmark services
    from scripts.benchmarks.utils.benchmark_utils import create_standalone_logging_functions, get_timer_value
    from scripts.benchmarks.utils.standalone_benchmark import StandaloneBenchmark

    # Get all logging functions for standalone mode
    log_funcs = create_standalone_logging_functions()

# Extract individual functions from the dictionary for easier use
get_isaaclab_version = log_funcs["get_isaaclab_version"]
get_mujoco_warp_version = log_funcs["get_mujoco_warp_version"]
get_newton_version = log_funcs["get_newton_version"]
log_app_start_time = log_funcs["log_app_start_time"]
log_python_imports_time = log_funcs["log_python_imports_time"]
log_task_start_time = log_funcs["log_task_start_time"]
log_scene_creation_time = log_funcs["log_scene_creation_time"]
log_simulation_start_time = log_funcs["log_simulation_start_time"]
log_newton_finalize_builder_time = log_funcs["log_newton_finalize_builder_time"]
log_newton_initialize_solver_time = log_funcs["log_newton_initialize_solver_time"]
log_total_start_time = log_funcs["log_total_start_time"]
log_runtime_step_times = log_funcs["log_runtime_step_times"]
log_rl_policy_rewards = log_funcs["log_rl_policy_rewards"]
log_rl_policy_episode_lengths = log_funcs["log_rl_policy_episode_lengths"]
parse_tf_logs = log_funcs["parse_tf_logs"]

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# Create the benchmark
if args_cli.kit:
    benchmark = BaseIsaacBenchmark(
        benchmark_name="benchmark_rsl_rl_train",
        workflow_metadata={
            "metadata": [
                {"name": "task", "data": args_cli.task},
                {"name": "seed", "data": args_cli.seed},
                {"name": "num_envs", "data": args_cli.num_envs},
                {"name": "max_iterations", "data": args_cli.max_iterations},
                {"name": "Mujoco Warp Info", "data": get_mujoco_warp_version()},
                {"name": "Isaac Lab Info", "data": get_isaaclab_version()},
                {"name": "Newton Info", "data": get_newton_version()},
            ],
        },
        backend_type=args_cli.benchmark_backend,
    )
else:
    benchmark = StandaloneBenchmark(
        benchmark_name="benchmark_rsl_rl_train",
        workflow_metadata={
            "metadata": [
                {"name": "task", "data": args_cli.task},
                {"name": "seed", "data": args_cli.seed},
                {"name": "num_envs", "data": args_cli.num_envs},
                {"name": "max_iterations", "data": args_cli.max_iterations},
                {"name": "Mujoco Warp Info", "data": get_mujoco_warp_version()},
                {"name": "Isaac Lab Info", "data": get_isaaclab_version()},
                {"name": "Newton Info", "data": get_newton_version()},
            ],
        },
        backend_type=args_cli.benchmark_backend,
        output_folder=args_cli.output_folder,
        randomize_filename_prefix=True,
    )


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # parse configuration
    if args_cli.kit:
        benchmark.set_phase("loading", start_recording_frametime=False, start_recording_runtime=True)
    else:
        benchmark.set_phase("loading", start_recording_frametime=False, start_recording_runtime=False)

    # override configurations with non-hydra CLI arguments
    if args_cli.kit:
        import scripts.reinforcement_learning.rsl_rl.cli_args as cli_args

        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    if args_cli.kit and hasattr(args_cli, "device") and args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # multi-gpu training configuration
    world_rank = 0
    world_size = 1
    if args_cli.distributed:
        if args_cli.kit:
            env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
            agent_cfg.device = f"cuda:{app_launcher.local_rank}"

            # set seed to have diversity in different threads
            seed = agent_cfg.seed + app_launcher.local_rank
            env_cfg.seed = seed
            agent_cfg.seed = seed
            world_rank = app_launcher.global_rank
            world_size = int(os.getenv("WORLD_SIZE", 1))
        else:
            print("[WARNING] Distributed mode is only supported with --kit flag.")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg.max_iterations = args_cli.max_iterations

    task_startup_time_begin = time.perf_counter_ns()

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    task_startup_time_end = time.perf_counter_ns()

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # set seed of the environment
    env.seed(agent_cfg.seed)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    benchmark.set_phase("sim_runtime")

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    if world_rank == 0:
        benchmark.store_measurements()

        # parse tensorboard file stats
        log_data = parse_tf_logs(log_dir)

        # prepare RL timing dict
        collection_fps = (
            1
            / (np.array(log_data.get("Perf/collection time", [1])))
            * env.unwrapped.num_envs
            * agent_cfg.num_steps_per_env
            * world_size
        )
        rl_training_times = {
            "Collection Time": (np.array(log_data.get("Perf/collection time", [])) / 1000).tolist(),
            "Learning Time": (np.array(log_data.get("Perf/learning_time", [])) / 1000).tolist(),
            "Collection FPS": collection_fps.tolist(),
            "Total FPS": [x * world_size for x in log_data.get("Perf/total_fps", [])],
        }

        # log additional metrics to benchmark services
        log_app_start_time(benchmark, (app_start_time_end - app_start_time_begin) / 1e6)
        log_python_imports_time(benchmark, (imports_time_end - imports_time_begin) / 1e6)
        log_task_start_time(benchmark, (task_startup_time_end - task_startup_time_begin) / 1e6)

        # Timer may not be available in standalone mode
        scene_creation_time = get_timer_value("scene_creation")
        simulation_start_time = get_timer_value("simulation_start")
        newton_finalize_builder_time = get_timer_value("newton_finalize_builder")
        newton_initialize_solver_time = get_timer_value("newton_initialize_solver")

        log_scene_creation_time(benchmark, scene_creation_time * 1000 if scene_creation_time else None)
        log_simulation_start_time(benchmark, simulation_start_time * 1000 if simulation_start_time else None)
        log_newton_finalize_builder_time(
            benchmark, newton_finalize_builder_time * 1000 if newton_finalize_builder_time else None
        )
        log_newton_initialize_solver_time(
            benchmark, newton_initialize_solver_time * 1000 if newton_initialize_solver_time else None
        )
        log_total_start_time(benchmark, (task_startup_time_end - app_start_time_begin) / 1e6)
        log_runtime_step_times(benchmark, rl_training_times, compute_stats=True)
        log_rl_policy_rewards(benchmark, log_data.get("Train/mean_reward", []))
        log_rl_policy_episode_lengths(benchmark, log_data.get("Train/mean_episode_length", []))

        benchmark.stop()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    if simulation_app is not None:
        simulation_app.close()
