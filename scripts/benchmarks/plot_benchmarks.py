import os
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Plot benchmarks.")
parser.add_argument("--results_dirs", type=str, nargs="+", help="Results directories.")
parser.add_argument("--output_dir", type=str, help="Output directory.")
parser.add_argument("--save", action="store_true", help="Save the plots.")
args_cli = parser.parse_args()


# Load the data
data = {}
for results_dir in args_cli.results_dirs:
    raw_data = {}
    for file in os.listdir(results_dir):
        if file.endswith('.json'):
            with open(os.path.join(results_dir, file), 'r') as f:
                raw_data[file] = json.load(f)
    # Process the data
    exp_data = {}
    for file_name, raw_data in raw_data.items():
        # Initialize task dict if it doesn't exist
        if raw_data["task"] not in exp_data:
            exp_data[raw_data["task"]] = {}
            
        # Add data for this num_envs
        exp_data[raw_data["task"]][raw_data["num_envs"]] = {
            "time_per_step_num_envs": raw_data["time_per_step_num_envs"],
            "mean_time_per_step_batch_num_envs": raw_data["avg_time_per_step_batch"] / raw_data["num_steps"],
            "std_time_per_step_batch_num_envs": raw_data["std_time_per_step_batch"] / raw_data["num_steps"], 
            "mean_time_per_step_batch": raw_data["avg_time_per_step_batch"] / raw_data["step_interval"],
            "std_time_per_step_batch": raw_data["std_time_per_step_batch"] / raw_data["step_interval"]
        }

    data[results_dir] = exp_data

# Rework the data stucture
reworked_data = {}
for results_dir, exp_data in data.items():
    reworked_data[results_dir] = {}
    for task, task_data in exp_data.items():
        reworked_data[results_dir][task] = {}
        reworked_data[results_dir][task]["num_envs"] = [int(num_envs) for num_envs in task_data.keys()]
        reworked_data[results_dir][task]["time_per_step_num_envs"] = [task_data[num_envs]["time_per_step_num_envs"] for num_envs in task_data.keys()]
        reworked_data[results_dir][task]["mean_time_per_step_batch_num_envs"] = [task_data[num_envs]["mean_time_per_step_batch_num_envs"] for num_envs in task_data.keys()]        
        reworked_data[results_dir][task]["std_time_per_step_batch_num_envs"] = [task_data[num_envs]["std_time_per_step_batch_num_envs"] for num_envs in task_data.keys()]
        reworked_data[results_dir][task]["mean_time_per_step_batch"] = [task_data[num_envs]["mean_time_per_step_batch"] for num_envs in task_data.keys()]
        reworked_data[results_dir][task]["std_time_per_step_batch"] = [task_data[num_envs]["std_time_per_step_batch"] for num_envs in task_data.keys()]
        # Sort the data by num_envs
        sorted_indices = np.argsort(reworked_data[results_dir][task]["num_envs"])
        reworked_data[results_dir][task]["num_envs"] = np.array(reworked_data[results_dir][task]["num_envs"])[sorted_indices]
        reworked_data[results_dir][task]["time_per_step_num_envs"] = np.array(reworked_data[results_dir][task]["time_per_step_num_envs"])[sorted_indices]
        reworked_data[results_dir][task]["mean_time_per_step_batch_num_envs"] = np.array(reworked_data[results_dir][task]["mean_time_per_step_batch_num_envs"])[sorted_indices]
        reworked_data[results_dir][task]["std_time_per_step_batch_num_envs"] = np.array(reworked_data[results_dir][task]["std_time_per_step_batch_num_envs"])[sorted_indices]
        reworked_data[results_dir][task]["mean_time_per_step_batch"] = np.array(reworked_data[results_dir][task]["mean_time_per_step_batch"])[sorted_indices]
        reworked_data[results_dir][task]["std_time_per_step_batch"] = np.array(reworked_data[results_dir][task]["std_time_per_step_batch"])[sorted_indices]


# Plot the data
# time_per_step_num_envs vs num_envs
plt.figure(figsize=(10, 5), dpi=200)
for results_dir, results_data in reworked_data.items():
    for task, task_data in results_data.items():
        plt.plot(task_data["num_envs"], task_data["time_per_step_num_envs"], label=f"{task} {results_dir}")
plt.xlabel("Number of Environments")
plt.ylabel("Time per Step / num_envs (s)")
plt.title("Time per Step vs Number of Environments (divided by num_envs)")
plt.legend()
plt.savefig("time_per_step_num_envs.png")

# mean and std of time per step batch vs num_envs
plt.figure(figsize=(10, 5), dpi=150)
for results_dir, results_data in reworked_data.items():
    for task, task_data in results_data.items():
        plt.plot(task_data["num_envs"], task_data["mean_time_per_step_batch"], label=f"{task} {results_dir}")
        plt.fill_between(task_data["num_envs"], task_data["mean_time_per_step_batch"] - task_data["std_time_per_step_batch"], task_data["mean_time_per_step_batch"] + task_data["std_time_per_step_batch"], alpha=0.2)
plt.xlabel("Number of Environments")
plt.ylabel("Time per Step (s)")
plt.title("Time per Step vs Number of Environments (with std-dev 50 runs)" )
plt.legend()
plt.savefig("time_per_step_batch_num_envs.png")
plt.legend()

# mean and std of time per step batch vs num_envs
plt.figure(figsize=(10, 5), dpi=150)
for results_dir, results_data in reworked_data.items():
    for task, task_data in results_data.items():
        plt.plot(task_data["num_envs"], task_data["mean_time_per_step_batch"] / task_data["mean_time_per_step_batch"][0], label=f"{task} {results_dir}")
        #plt.plot(task_data["num_envs"], task_data["mean_time_per_step_batch"] / task_data["mean_time_per_step_batch"][0], marker="^", label=f"{task} {results_dir}")
        tdm_rs = task_data["mean_time_per_step_batch"] / task_data["mean_time_per_step_batch"][0]
        rds_rs = task_data["std_time_per_step_batch"] / task_data["mean_time_per_step_batch"][0]
        plt.fill_between(task_data["num_envs"], tdm_rs - rds_rs, tdm_rs + rds_rs, alpha=0.2)
plt.xlabel("Number of Environments")
plt.ylabel("Increase in time relatively to 64 envs")
plt.title("Step time increase vs num_envs (with std-dev 50 runs)" )
plt.legend()
plt.savefig("step_time_increase_vs_num_envs.png")
plt.legend()

plt.show()