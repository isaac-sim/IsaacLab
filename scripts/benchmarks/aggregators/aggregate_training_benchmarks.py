# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import json
import numpy as np

from utils import (
    bar_plot,
    check_info_consistency,
    compute_lab_overhead,
    load_json_files,
    load_physx_non_rl_reference_data,
    rework_data,
)

parser = argparse.ArgumentParser()
# A list of benchmark folders
parser.add_argument(
    "--benchmark_folders", type=str, nargs="+", required=True, help="List of paths to benchmark folders"
)
parser.add_argument("--include_physx_reference", action="store_true", help="Include PhysX reference data")
parser.add_argument("--plot_scaling_curves", action="store_true", help="Plot scaling curves")
args = parser.parse_args()

benchmark_folders = args.benchmark_folders

aggregated_data = {}
for benchmark_folder in benchmark_folders:
    json_data = load_json_files(benchmark_folder)
    info = check_info_consistency(json_data)
    aggregated_data[benchmark_folder] = {"Metadata": info}
    aggregated_data[benchmark_folder].update(rework_data(json_data, phases=["startup", "runtime", "train"]))


if args.include_physx_reference:
    reference_data = load_physx_non_rl_reference_data()
    aggregated_data.update(reference_data)

num_envs = json_data[0]["runtime"]["num_envs"]
gpu_name = json_data[0]["sim_runtime"]["gpu_device_name"]
if gpu_name == "NVIDIA GeForce RTX 5090":
    cpu_name = "Intel i9-14900K"
else:
    cpu_name = "Horde"
num_iterations = json_data[0]["runtime"]["max_iterations"]

compute_lab_overhead(aggregated_data, num_envs=num_envs, decimations=[2, 2, 2, 4, 4, 4])

# Plot the data
out_1 = bar_plot(
    aggregated_data,
    [
        "EMA 0.95 Episode Length",
        "EMA 0.95 Reward",
        "Mean Total FPS",
        "Mean Collection FPS",
        "Total Start Time (Launch to Train)",
        "Simulate Time",
        "Env Step Time",
    ],
    titles=[
        f"EMA terminal episode length, {num_envs} envs @ {num_iterations} iterations",
        f"EMA terminal episode reward, {num_envs} envs @ {num_iterations} iterations",
        f"Mean total FPS, {num_envs} envs, {gpu_name}",
        f"Mean collection FPS, {num_envs} envs, {gpu_name}",
        f"Total Start Time (Launch to Train), {num_envs} envs, {cpu_name}",
        f"Simulate Time, {num_envs} envs, {gpu_name}",
        f"Env Step Time, {num_envs} envs, {gpu_name}",
    ],
    y_axes_labels=[
        "steps",
        "reward",
        "FPS",
        "FPS",
        "time (s)",
        "time (ms)",
        "time (ms)",
    ],
    scale_y_axis=[1, 1, 1, 1, 0.001, 1, 1],
    num_envs=num_envs,
)
for metric in out_1.keys():
    print("##" * 20)
    print(metric)
    print("##" * 20)
    for run in out_1[metric].keys():
        print(" *", run)
        for task in out_1[metric][run].keys():
            print("  -", task, ":", out_1[metric][run][task])


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
        except Exception as e:
            print(f"[ERROR] Error encoding {obj} to JSON: {e}")
        return super().default(obj)


with open("aggregated_data.json", "w") as f:
    json.dump(aggregated_data, f, cls=NumpyEncoder)
