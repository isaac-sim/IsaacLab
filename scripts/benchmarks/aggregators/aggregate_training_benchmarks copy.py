# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import json
import numpy as np
import os
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--benchmark_folder", type=str, required=True)
args = parser.parse_args()

benchmark_folder = args.benchmark_folder


def load_json_files(benchmark_folder) -> list[dict]:
    # Get all the json files in the benchmark_folder
    json_files = [f.path for f in os.scandir(benchmark_folder) if f.is_file() and f.name.endswith(".json")]

    # Load all the json files in memory
    json_data = [json.load(open(f)) for f in json_files]

    return json_data


def check_info_consistency(json_data: list[dict]) -> dict:
    # Check that APP INFO is the same in all the json files
    app_info = json_data[0]["App Info"]
    for data in json_data:
        if data["App Info"] != app_info:
            raise ValueError("APP INFO is not the same in all the json files")

    # Check that NEWTON INFO is the same in all the json files
    newton_info = json_data[0]["Newton Info"]
    for data in json_data:
        if data["Newton Info"] != newton_info:
            raise ValueError("NEWTON INFO is not the same in all the json files")

    # Check that ISAAC LAB INFO is the same in all the json files
    isaac_lab_info = json_data[0]["Isaac Lab Info"]
    # for data in json_data:
    #    if data["Isaac Lab Info"] != isaac_lab_info:
    #        raise ValueError("ISAAC LAB INFO is not the same in all the json files")

    # Check that MUJOCO WARP INFO is the same in all the json files
    mujoco_warp_info = json_data[0]["Mujoco Warp Info"]
    for data in json_data:
        if data["Mujoco Warp Info"] != mujoco_warp_info:
            raise ValueError("MUJOCO WARP INFO is not the same in all the json files")

    return {
        "App Info": app_info,
        "Newton Info": newton_info,
        "Isaac Lab Info": isaac_lab_info,
        "Mujoco Warp Info": mujoco_warp_info,
    }


json_data = load_json_files(benchmark_folder)
info = check_info_consistency(json_data)


def rework_data(json_data: list[dict]) -> dict:
    phases = ["startup", "train", "runtime"]
    default_excludes = ["workflow_name", "task", "seed", "num_envs", "phase", "max_iterations"]
    excludes = {
        "startup": default_excludes,
        "runtime": default_excludes + ["Min Collection FPS", "Max Collection FPS", "Mean Collection FPS"],
        "train": default_excludes,
    }

    reworked_data = {}
    for data in json_data:
        task = data["sim_runtime"]["task"]
        if task not in reworked_data:
            reworked_data[task] = {}
        num_envs = data["sim_runtime"]["num_envs"]
        if num_envs not in reworked_data[task]:
            reworked_data[task][num_envs] = {}
        for phase in phases:
            if phase not in reworked_data[task][num_envs]:
                reworked_data[task][num_envs][phase] = {}
            for metric in data[phase].keys():
                if metric not in excludes[phase]:
                    if metric not in reworked_data[task][num_envs][phase]:
                        reworked_data[task][num_envs][phase][metric] = []
                    reworked_data[task][num_envs][phase][metric].append(data[phase][metric])

    # Compute the mean, min, max, and std of the metrics
    for task in reworked_data.keys():
        for num_envs in reworked_data[task].keys():
            for phase in phases:
                for metric in reworked_data[task][num_envs][phase].keys():
                    reworked_data[task][num_envs][phase][metric] = {
                        "mean": np.mean(reworked_data[task][num_envs][phase][metric]),
                        "min": np.min(reworked_data[task][num_envs][phase][metric]),
                        "max": np.max(reworked_data[task][num_envs][phase][metric]),
                        "std": np.std(reworked_data[task][num_envs][phase][metric]),
                    }

    return reworked_data


def rework_by_metric(reworked_data: dict, metric: str, quantities: list[str] = ["mean", "min", "max"]):
    metric_data = {}
    for task in reworked_data.keys():
        if task == "Metadata":
            continue
        for num_envs in reworked_data[task].keys():
            for phase in reworked_data[task][num_envs].keys():
                if metric in reworked_data[task][num_envs][phase]:
                    if task not in metric_data:
                        metric_data[task] = {}
                    if num_envs not in metric_data[task]:
                        metric_data[task][int(num_envs)] = {}
                    for quantity in quantities:
                        if quantity not in metric_data[task][num_envs]:
                            metric_data[task][int(num_envs)][quantity] = reworked_data[task][num_envs][phase][metric][
                                quantity
                            ]
    metric_data_refactor = {}
    for task in metric_data.keys():
        if task not in metric_data_refactor.keys():
            metric_data_refactor[task] = {}
        for quantity in quantities:
            # if task not in metric_data_refactor[quantity]:
            data = dict(sorted(metric_data[task].items()))
            data_reduced = {k: v[quantity] for k, v in data.items()}
            tuple_data = (list(data_reduced.keys()), list(data_reduced.values()))
            # tuple_data = (list(data.keys()), list(data.values()))
            metric_data_refactor[task][quantity] = tuple_data

    return metric_data_refactor


aggregated_data = {"Metadata": info}
aggregated_data.update(rework_data(json_data))
#
# data = rework_by_metric(aggregated_data, "Mean Total FPS", quantities=["mean"])
# data = rework_by_metric(aggregated_data, "EMA 0.95 Reward", quantities=["mean", "min", "max"])
# data = rework_by_metric(aggregated_data, "EMA 0.95 Episode Length", quantities=["mean", "min", "max"])
# plot(aggregated_data, ["Mean Total FPS"], ["mean"])
# Save the aggregated data to a json file
with open("aggregated_data.json", "w") as f:
    json.dump(aggregated_data, f)
