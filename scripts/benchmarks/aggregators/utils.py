# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import numpy as np
import os
from matplotlib import pyplot as plt


def load_json_files(benchmark_folder) -> list[dict]:
    # Get all the json files in the benchmark_folder
    json_files = [f.path for f in os.scandir(benchmark_folder) if f.is_file() and f.name.endswith(".json")]

    # Load all the json files in memory
    json_data = []
    for f in json_files:
        with open(f) as f:
            json_data.append(json.load(f))

    return json_data


def check_info_consistency(json_data: list[dict]) -> dict:
    try:
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
    except Exception as e:
        print(e)
        return {
            "App Info": None,
            "Newton Info": None,
            "Isaac Lab Info": None,
            "Mujoco Warp Info": None,
        }


def rework_data(json_data: list[dict], phases: list[str] = ["startup", "train", "runtime"]) -> dict:
    default_excludes = [
        "workflow_name",
        "task",
        "seed",
        "num_envs",
        "phase",
        "max_iterations",
        "Mujoco Warp Info",
        "Isaac Lab Info",
        "Newton Info",
    ]
    excludes = {
        "startup": default_excludes,
        "runtime": default_excludes,
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


def bar_table_single_env(
    data_s: dict, metrics: list[str], num_envs: int = 4096, titles: list[str] = None, y_axes_labels: list[str] = None
):
    reworked_data = {}
    for benchmark_names, benchmark_data in data_s.items():
        for task_name, task_data in benchmark_data.items():
            if task_name == "Metadata":
                continue
            for num_envs_key, num_envs_data in task_data.items():
                if int(num_envs_key) != num_envs:
                    continue
                data = {}
                for phase, phase_data in num_envs_data.items():
                    data.update({k: v for k, v in phase_data.items()})
                for metric_name, metric_data in data.items():
                    if metric_name not in metrics:
                        continue
                    if metric_name not in reworked_data:
                        reworked_data[metric_name] = {}
                    if benchmark_names not in reworked_data[metric_name]:
                        reworked_data[metric_name][benchmark_names] = {}
                    reworked_data[metric_name][benchmark_names][task_name] = metric_data["mean"]

    bar_width = 0.25

    # For each metric, plot the data in an individual figure
    num_benchmarks = len(list(reworked_data.values())[0].keys())
    benchmark_names = list(list(reworked_data.values())[0].keys())
    task_names = list(list(list(reworked_data.values())[0].values())[0].keys())
    task_names.sort()

    for i, metric_name in enumerate(metrics):
        metric_data = reworked_data[metric_name]
        plt.figure(figsize=(10, 5))
        data = []
        for benchmark_name in metric_data.keys():
            d = []
            for task_name in task_names:
                d.append(metric_data[benchmark_name][task_name])
            data.append(d)

        for k, d in enumerate(data):
            offset = [bar_width * (k + (num_benchmarks + 1) * j) for j in range(len(task_names))]
            if clean_benchmark_name(benchmark_names[k]) == "PhysX":
                plt.bar(offset, d, width=bar_width, label=clean_benchmark_name(benchmark_names[k]), color="gold")
            else:
                plt.bar(offset, d, width=bar_width, label=clean_benchmark_name(benchmark_names[k]))


def bar_plot(
    data_s: dict,
    metrics: list[str],
    num_envs: int = 4096,
    titles: list[str] = None,
    y_axes_labels: list[str] = None,
    scale_y_axis: list[float] = None,
):
    reworked_data = {}
    for benchmark_names, benchmark_data in data_s.items():
        for task_name, task_data in benchmark_data.items():
            if task_name == "Metadata":
                continue
            for num_envs_key, num_envs_data in task_data.items():
                if int(num_envs_key) != num_envs:
                    continue
                data = {}
                for phase, phase_data in num_envs_data.items():
                    data.update({k: v for k, v in phase_data.items()})
                for metric_name, metric_data in data.items():
                    if metric_name not in metrics:
                        continue
                    if metric_name not in reworked_data:
                        reworked_data[metric_name] = {}
                    if benchmark_names not in reworked_data[metric_name]:
                        reworked_data[metric_name][benchmark_names] = {}
                    reworked_data[metric_name][benchmark_names][task_name] = metric_data["mean"]

    bar_width = 0.25

    # For each metric, plot the data in an individual figure
    num_benchmarks = len(list(reworked_data.values())[0].keys())
    benchmark_names = list(list(reworked_data.values())[0].keys())
    task_names = list(list(list(reworked_data.values())[0].values())[0].keys())
    task_names.sort()
    cleaned_task_names = clean_task_names(task_names)
    metric_data_list = {}
    for i, metric_name in enumerate(metrics):
        metric_data = reworked_data[metric_name]
        metric_data_list[metric_name] = {}
        plt.figure(figsize=(10, 5))
        data = []
        for benchmark_name in metric_data.keys():
            d = []
            metric_data_list[metric_name][clean_benchmark_name(benchmark_name)] = {}
            for task_name in task_names:
                d.append(metric_data[benchmark_name][task_name] * (scale_y_axis[i] if scale_y_axis is not None else 1))
                metric_data_list[metric_name][clean_benchmark_name(benchmark_name)][
                    clean_task_names([task_name])[0]
                ] = metric_data[benchmark_name][task_name]

            data.append(d)
        for k, d in enumerate(data):
            offset = [bar_width * (k + (num_benchmarks + 1) * j) for j in range(len(task_names))]
            if (clean_benchmark_name(benchmark_names[k]) == "PhysX") or (
                clean_benchmark_name(benchmark_names[k]) == "PhysX_train"
            ):
                plt.bar(offset, d, width=bar_width, label=clean_benchmark_name(benchmark_names[k]), color="gold")
            else:
                bars = plt.bar(offset, d, width=bar_width, label=clean_benchmark_name(benchmark_names[k]))
                if ("PhysX" in [clean_benchmark_name(bn) for bn in benchmark_names]) or (
                    "PhysX_train" in [clean_benchmark_name(bn) for bn in benchmark_names]
                ):
                    try:
                        try:
                            physx_idx = [clean_benchmark_name(bn) for bn in benchmark_names].index("PhysX")
                        except Exception as e:
                            print(e)
                            physx_idx = [clean_benchmark_name(bn) for bn in benchmark_names].index("PhysX_train")
                        physx_data = data[physx_idx]
                        for j, bar in enumerate(bars):
                            physx_val = physx_data[j]
                            this_val = d[j]
                            if physx_val == 0:
                                delta_pct = 0.0
                            else:
                                delta_pct = (this_val - physx_val) / physx_val * 100
                            # Only annotate if not PhysX itself
                            if (
                                clean_benchmark_name(benchmark_names[k]) != "PhysX"
                                and clean_benchmark_name(benchmark_names[k]) != "PhysX_train"
                            ):
                                plt.text(
                                    bar.get_x() + bar.get_width() / 2,
                                    bar.get_height(),
                                    f"{delta_pct:+.1f}%",
                                    ha="center",
                                    va="bottom",
                                    fontsize=8,
                                    color="black",
                                )
                    except Exception:
                        pass
        plt.xlabel("Tasks")
        if y_axes_labels is not None:
            plt.ylabel(y_axes_labels[i])
        else:
            plt.ylabel(metric_name)
        if titles is not None:
            plt.title(titles[i])
        else:
            plt.title(metric_name)
        k = (len(benchmark_names) - 1) / 2.0
        offset = [bar_width * (k + (num_benchmarks + 1) * j) for j in range(len(task_names))]
        plt.xticks(offset, cleaned_task_names)
        plt.legend()
    plt.show()
    return metric_data_list


def plot_scaling_curves(
    data_s: dict,
    metrics: list[str],
    num_envs: list[int] = [1024, 2048, 4096, 8192, 16384],
    titles: list[str] = None,
    y_axes_labels: list[str] = None,
):
    reworked_data = {}
    for benchmark_names, benchmark_data in data_s.items():
        for task_name, task_data in benchmark_data.items():
            if task_name == "Metadata":
                continue
            for num_envs_key, num_envs_data in task_data.items():
                if int(num_envs_key) not in num_envs:
                    continue
                data = {}
                for phase, phase_data in num_envs_data.items():
                    data.update({k: v for k, v in phase_data.items()})
                for metric_name, metric_data in data.items():
                    if metric_name not in metrics:
                        continue
                    if metric_name not in reworked_data:
                        reworked_data[metric_name] = {}
                    if task_name not in reworked_data[metric_name]:
                        reworked_data[metric_name][task_name] = {}
                    if benchmark_names not in reworked_data[metric_name][task_name]:
                        reworked_data[metric_name][task_name][benchmark_names] = {}
                    reworked_data[metric_name][task_name][benchmark_names][int(num_envs_key)] = metric_data["mean"]

    task_names = list(list(reworked_data.values())[0].keys())
    cleaned_task_names = clean_task_names(task_names)
    num_benchmarks = len(list(list(reworked_data.values())[0].values())[0].keys())
    benchmark_names = list(list(list(reworked_data.values())[0].values())[0].keys())
    num_num_envs = len(num_envs)

    for i, metric_name in enumerate(metrics):
        metric_data = reworked_data[metric_name]
        # Make a plot for each task
        fig, axs = plt.subplots(3, 2, figsize=(10, 5))
        for j, task_name in enumerate(task_names):
            data = []
            for benchmark_name in metric_data[task_name].keys():
                values = [reworked_data[metric_name][task_name][benchmark_name][num_envs] for num_envs in num_envs]
                data.append(values)
            bar_width = 0.25
            x = j % 3
            y = j // 3
            for k, d in enumerate(data):
                offset = [bar_width * (k + (num_benchmarks + 1) * j) for j in range(num_num_envs)]
                if clean_benchmark_name(benchmark_names[k]) == "PhysX":
                    axs[x, y].bar(
                        offset, d, width=bar_width, label=clean_benchmark_name(benchmark_names[k]), color="gold"
                    )
                else:
                    bars = axs[x, y].bar(offset, d, width=bar_width, label=clean_benchmark_name(benchmark_names[k]))
                    # Add delta % above bar compared to PhysX
                    # Only if PhysX is present in benchmark_names
                    if "PhysX" in [clean_benchmark_name(bn) for bn in benchmark_names]:
                        # Find PhysX index in benchmark_names
                        try:
                            physx_idx = [clean_benchmark_name(bn) for bn in benchmark_names].index("PhysX")
                            physx_data = data[physx_idx]
                            for m, bar in enumerate(bars):
                                physx_val = physx_data[m]
                                this_val = d[m]
                                if physx_val == 0:
                                    delta_pct = 0.0
                                else:
                                    delta_pct = (this_val - physx_val) / physx_val * 100
                                # Only annotate if not PhysX itself
                                if clean_benchmark_name(benchmark_names[k]) != "PhysX":
                                    axs[x, y].text(
                                        bar.get_x() + bar.get_width() / 2,
                                        bar.get_height(),
                                        f"{delta_pct:+.1f}%",
                                        ha="center",
                                        va="bottom",
                                        fontsize=8,
                                        color="black",
                                    )
                        except Exception:
                            pass
            axs[x, y].set_xlabel("Number of environments")
            if y_axes_labels is not None:
                axs[x, y].set_ylabel(y_axes_labels[i])
            else:
                axs[x, y].set_ylabel(metric_name)
            axs[x, y].set_title(cleaned_task_names[j])
            k = (len(benchmark_names) - 1) / 2.0
            offset = [bar_width * (k + (num_benchmarks + 1) * j) for j in range(num_num_envs)]
            axs[x, y].set_xticks(offset)
            axs[x, y].set_xticklabels([str(num_envs) for num_envs in num_envs])
            axs[x, y].legend()
        if titles is not None:
            plt.suptitle(titles[i])
        else:
            plt.suptitle(metric_name)
        plt.tight_layout()
    plt.show()


def compute_lab_overhead(data_s: dict, num_envs: int = 4096, decimations: list[int] | None = None):

    metrics = ["Simulate Time", "Env Step Time"]

    reworked_data = {}
    for benchmark_names, benchmark_data in data_s.items():
        for task_name, task_data in benchmark_data.items():
            if task_name == "Metadata":
                continue
            for num_envs_key, num_envs_data in task_data.items():
                if int(num_envs_key) != num_envs:
                    continue
                data = {}
                for phase, phase_data in num_envs_data.items():
                    data.update({k: v for k, v in phase_data.items()})
                for metric_name, metric_data in data.items():
                    if metric_name not in metrics:
                        continue
                    if metric_name not in reworked_data:
                        reworked_data[metric_name] = {}
                    if benchmark_names not in reworked_data[metric_name]:
                        reworked_data[metric_name][benchmark_names] = {}
                    reworked_data[metric_name][benchmark_names][task_name] = metric_data["mean"]

    bar_width = 0.25

    # For each metric, plot the data in an individual figure
    num_benchmarks = len(list(reworked_data.values())[0].keys())
    benchmark_names = list(list(reworked_data.values())[0].keys())
    task_names = list(list(list(reworked_data.values())[0].values())[0].keys())
    task_names.sort()
    cleaned_task_names = clean_task_names(task_names)

    env_step_time = reworked_data["Env Step Time"]
    simulate_time = reworked_data["Simulate Time"]

    print(env_step_time)

    data = []
    for benchmark_name in benchmark_names:
        d = []
        for j, task_name in enumerate(task_names):
            env_step_time_mean = env_step_time[benchmark_name][task_name]
            simulate_time_mean = simulate_time[benchmark_name][task_name]
            lab_overhead = env_step_time_mean - simulate_time_mean * decimations[j]
            d.append(lab_overhead)
        data.append(d)

    for k, d in enumerate(data):
        offset = [bar_width * (k + (num_benchmarks + 1) * j) for j in range(len(task_names))]
        plt.bar(offset, d, width=bar_width, label=clean_benchmark_name(benchmark_names[k]))
        plt.xlabel("Tasks")
        plt.ylabel("Time (ms)")
        plt.title("Lab Overhead")
        k = (len(benchmark_names) - 1) / 2.0
        offset = [bar_width * (k + (num_benchmarks + 1) * j) for j in range(len(task_names))]
        plt.xticks(offset, cleaned_task_names)
        plt.legend()
    plt.show()


def clean_task_names(task_names: list[str]) -> list[str]:
    cleaned_task_names = []
    for task_name in task_names:
        task_name = task_name.split("-v0")[0]
        task_name = task_name.split("-Direct")[0]
        task_name = task_name.split("Isaac-")[-1]
        task_name = task_name.split("Velocity-Flat-")[-1]
        cleaned_task_names.append(task_name)
    return cleaned_task_names


def load_physx_non_rl_reference_data() -> dict:
    # Get the path to the parent of this python file
    import os

    current_file_path = os.path.abspath(__file__)
    parent_dir = os.path.dirname(current_file_path)
    reference_data_path = os.path.join(parent_dir, "physx_golden_values_non_rl.json")
    with open(reference_data_path) as f:
        reference_data = json.load(f)
    return reference_data


def clean_benchmark_name(benchmark_name: str) -> str:
    benchmark_name = benchmark_name.split("/")[-1]
    return benchmark_name
