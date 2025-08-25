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
    load_json_files,
    load_physx_non_rl_reference_data,
    plot_scaling_curves,
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
    aggregated_data[benchmark_folder].update(rework_data(json_data, phases=["startup", "runtime"]))


if args.include_physx_reference:
    reference_data = load_physx_non_rl_reference_data()
    aggregated_data.update(reference_data)

# Plot the data
bar_plot(
    aggregated_data,
    ["Mean Environment step effective FPS", "Mean Environment step times"],
    titles=[
        "Simulated time steps per second, 4096 envs, RTX5090",
        "Time required to run a single time step, 4096 envs, RTX5090",
    ],
    y_axes_labels=["time steps per second", "time (ms)"],
    num_envs=4096,
)

if args.plot_scaling_curves:
    plot_scaling_curves(
        aggregated_data,
        ["Mean Environment step effective FPS", "Mean Environment step times"],
        titles=[
            "Simulated time steps per second, 4096 envs, RTX5090",
            "Time required to run a single time step, 4096 envs, RTX5090",
        ],
        y_axes_labels=["time steps per second", "time (ms)"],
        num_envs=[1024, 2048, 4096, 8192, 16384],
    )


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
