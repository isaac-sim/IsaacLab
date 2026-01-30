# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import glob
import json
import math
import os
import re

import numpy as np
import yaml


def _get_repo_path():
    """Get the repository root by searching for marker files.

    Searches upward from the current file for IsaacLab repository markers
    (isaaclab.sh or setup.py) to robustly find the repo root.
    """
    current = os.path.abspath(__file__)
    # Look for isaaclab.sh or setup.py as markers (max 10 levels up)
    for _ in range(10):
        current = os.path.dirname(current)
        if os.path.exists(os.path.join(current, "isaaclab.sh")):
            return current
        # Fallback marker
        if os.path.exists(os.path.join(current, "setup.py")) and os.path.exists(os.path.join(current, "source")):
            return current
    raise RuntimeError("Could not find IsaacLab repository root. Expected to find 'isaaclab.sh' in parent directories.")


def get_env_configs(configs_path):
    """Get environment configurations from yaml filepath."""
    with open(configs_path) as env_configs_file:
        env_configs = yaml.safe_load(env_configs_file)
    return env_configs


def get_env_config(env_configs, mode, workflow, task):
    """Get the environment configuration."""
    if mode not in env_configs:
        raise ValueError(f"Mode {mode} is not supported in the config file.")

    extended_task = f"{workflow}:{task}"
    # return a direct match with extended task name
    if extended_task in env_configs[mode]:
        return env_configs[mode][extended_task]

    # else, return a direct match with task name
    if task in env_configs[mode]:
        return env_configs[mode][task]

    # else, return a regex match with extended task name
    for env_config_key in env_configs[mode].keys():
        if re.match(env_config_key, extended_task):
            return env_configs[mode][env_config_key]

    # else, return a regex match with task name
    for env_config_key in env_configs[mode].keys():
        if re.match(env_config_key, task):
            return env_configs[mode][env_config_key]

    # if no match is found, return None
    return None


def evaluate_job(workflow, task, env_config, duration):
    """Evaluate the job."""
    log_data = _retrieve_logs(workflow, task)

    kpi_payload = {"success": True, "msg": ""}

    # handle case where no log files are found
    if not log_data:
        kpi_payload["success"] = False
        kpi_payload["msg"] = "error: training did not finish!"
        return kpi_payload

    thresholds = {**env_config.get("lower_thresholds", {}), **env_config.get("upper_thresholds", {})}

    # evaluate all thresholds from the config
    for threshold_name, threshold_val in thresholds.items():
        uses_lower_threshold = threshold_name in env_config.get("lower_thresholds", {})
        if threshold_name == "duration":
            val = duration
        else:
            val = _extract_log_val(threshold_name, log_data, uses_lower_threshold, workflow)
        # skip non-numeric values
        if val is None or not isinstance(val, (int, float)) or (isinstance(val, float) and math.isnan(val)):
            continue
        val = round(val, 4)
        threshold_val_rounded = round(threshold_val, 4)
        if uses_lower_threshold:
            if val < threshold_val:
                kpi_payload["success"] = False
                if not kpi_payload["msg"]:
                    kpi_payload["msg"] = f"{threshold_name} below threshold: {val} < {threshold_val_rounded}"
        else:
            if val > threshold_val:
                kpi_payload["success"] = False
                if not kpi_payload["msg"]:
                    kpi_payload["msg"] = f"{threshold_name} above threshold: {val} > {threshold_val_rounded}"
        kpi_payload[threshold_name] = val
        if threshold_name == "reward":
            normalized_reward = val / threshold_val
            kpi_payload[f"{threshold_name}_normalized"] = normalized_reward
        kpi_payload[f"{threshold_name}_threshold"] = threshold_val

    # add max iterations to the payload
    max_iterations = env_config.get("max_iterations")
    if max_iterations is not None:
        kpi_payload["max_iterations"] = max_iterations

    return kpi_payload


def process_kpi_data(kpi_payloads, tag, timestamp):
    """Combine and augment the KPI payloads.

    Args:
        kpi_payloads: Dictionary of KPI payloads for each job.
        tag: Tag for the KPI payload.
        timestamp: Timestamp to use (ISO format).
    """
    # accumulate workflow outcomes
    totals = {}
    successes = {}
    failures_did_not_finish = {}
    failures_did_not_pass_thresholds = {}
    for job_id, kpi_payload in kpi_payloads.items():
        workflow = job_id.split(":")[0]
        if workflow not in totals:
            totals[workflow] = 0
            successes[workflow] = 0
            failures_did_not_finish[workflow] = 0
            failures_did_not_pass_thresholds[workflow] = 0
        totals[workflow] += 1
        if kpi_payload["success"]:
            successes[workflow] += 1
        else:
            if kpi_payload["msg"] == "error: training did not finish!":
                failures_did_not_finish[workflow] += 1
            else:
                failures_did_not_pass_thresholds[workflow] += 1

    kpi_payloads["overall"] = {
        "totals": totals,
        "successes": successes,
        "failures_did_not_finish": failures_did_not_finish,
        "failures_did_not_pass_thresholds": failures_did_not_pass_thresholds,
        "timestamp": timestamp,
        "tag": tag,
    }

    return kpi_payloads


def output_payloads(payloads):
    """Output the KPI payloads to a json file."""
    # first grab all log files
    repo_path = _get_repo_path()
    output_path = os.path.join(repo_path, "logs/kpi.json")
    # create directory if it doesn't exist
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    # save file
    with open(output_path, "w") as payload_file:
        json.dump(payloads, payload_file, indent=4)


def _retrieve_logs(workflow, task):
    """Retrieve training logs."""
    # first grab all log files
    repo_path = _get_repo_path()

    # Defer Isaac Sim version import to avoid preloading USD before SimulationApp starts.
    from isaaclab.utils.version import get_isaac_sim_version

    if get_isaac_sim_version().major < 5:
        repo_path = os.path.join(repo_path, "..")
    if workflow == "rl_games":
        log_files_path = os.path.join(repo_path, f"logs/{workflow}/{task}/*/summaries/*")
    elif workflow == "sb3":
        log_files_path = os.path.join(repo_path, f"logs/{workflow}/{task}/*/*/*.tfevents.*")
    else:
        log_files_path = os.path.join(repo_path, f"logs/{workflow}/{task}/*/*.tfevents.*")
    log_files = glob.glob(log_files_path)
    # handle case where no log files are found
    if not log_files:
        return None
    # find most recent
    latest_log_file = max(log_files, key=os.path.getctime)
    # parse tf file into a dictionary
    log_data = _parse_tf_logs(latest_log_file)

    # validate that log data contains entries
    if not log_data:
        print(f"Warning: Log file {latest_log_file} parsed but contains no data")
        return None

    return log_data


def _parse_tf_logs(log):
    """Parse the tensorflow filepath into a dictionary."""
    # Defer tensorboard import to avoid side effects during pytest collection.
    from tensorboard.backend.event_processing import event_accumulator

    log_data = {}
    ea = event_accumulator.EventAccumulator(log)
    ea.Reload()
    tags = ea.Tags()["scalars"]
    for tag in tags:
        log_data[tag] = []
        for event in ea.Scalars(tag):
            log_data[tag].append((event.step, event.value))
    return log_data


def _extract_log_val(name, log_data, uses_lower_threshold, workflow):
    """Extract the value from the log data."""
    try:
        if name == "reward":
            reward_tags = {
                "rl_games": "rewards/iter",
                "rsl_rl": "Train/mean_reward",
                "sb3": "rollout/ep_rew_mean",
                "skrl": "Reward / Total reward (mean)",
            }
            tag = reward_tags.get(workflow)
            if tag:
                return _extract_reward(log_data, tag)

        elif name == "episode_length":
            episode_tags = {
                "rl_games": "episode_lengths/iter",
                "rsl_rl": "Train/mean_episode_length",
                "sb3": "rollout/ep_len_mean",
                "skrl": "Episode / Total timesteps (mean)",
            }
            tag = episode_tags.get(workflow)
            if tag:
                return _extract_feature(log_data, tag, uses_lower_threshold)
    except KeyError as e:
        print(f"Warning: Metric '{name}' not found in logs for workflow '{workflow}': {e}")
        return None
    except Exception as e:
        print(f"Error extracting '{name}' for workflow '{workflow}': {e}")
        return None

    raise ValueError(f"Env Config name {name} is not supported.")


def _extract_feature(log_data, feature, uses_lower_threshold):
    """Extract the feature from the log data."""
    log_data = np.array(log_data[feature])[:, 1]

    if uses_lower_threshold:
        return max(log_data)
    else:
        return min(log_data)


def _extract_reward(log_data, feature, k=8):
    """Extract the averaged max reward from the log data."""
    log_data = np.array(log_data[feature])[:, 1]

    # find avg of k max values
    k = min(len(log_data), k)
    averaged_reward = np.mean(np.partition(log_data, -k)[-k:])

    return averaged_reward
