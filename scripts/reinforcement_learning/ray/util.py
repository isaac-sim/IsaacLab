# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import argparse
import os
import re
import select
import subprocess
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from math import isclose
from time import time
from typing import Any

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from tensorboard.backend.event_processing.directory_watcher import DirectoryDeletedError
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tensorboard_logs(directory: str) -> dict:
    """From a tensorboard directory, get the latest scalar values. If the logs can't be
    found, check the summaries sublevel.

    Args:
        directory: The directory of the tensorboard logging.

    Returns:
        The latest available scalar values.
    """

    # replace any non-alnum/underscore/dot with "_", then collapse runs of "_"
    def replace_invalid_chars(t):
        t2 = re.sub(r"[^0-9A-Za-z_./]", "_", t)
        t2 = re.sub(r"_+", "_", t2)
        return t2.strip("_")

    # Initialize the event accumulator with a size guidance for only the latest entry
    def get_latest_scalars(path: str) -> dict:
        event_acc = EventAccumulator(path, size_guidance={"scalars": 1})
        try:
            event_acc.Reload()
            if event_acc.Tags()["scalars"]:
                return {
                    replace_invalid_chars(tag): event_acc.Scalars(tag)[-1].value
                    for tag in event_acc.Tags()["scalars"]
                    if event_acc.Scalars(tag)
                }
        except (KeyError, OSError, RuntimeError, DirectoryDeletedError):
            return {}

    scalars = get_latest_scalars(directory)
    return scalars or get_latest_scalars(os.path.join(directory, "summaries"))


def get_invocation_command_from_cfg(
    cfg: dict,
    python_cmd: str = "/workspace/isaaclab/isaaclab.sh -p",
    workflow: str = "scripts/reinforcement_learning/rl_games/train.py",
) -> str:
    """Generate command with proper Hydra arguments"""
    runner_args = []
    hydra_args = []

    def process_args(args, target_list, is_hydra=False):
        for key, value in args.items():
            if not is_hydra:
                if key.endswith("_singleton"):
                    target_list.append(value)
                elif key.startswith("--"):
                    target_list.append(f"{key} {value}")  # Space instead of = for runner args
                else:
                    target_list.append(f"{value}")
            else:
                if isinstance(value, list):
                    # Check the type of the first item to determine formatting
                    if value and isinstance(value[0], dict):
                        # Handle list of dictionaries (e.g., CNN convs)
                        formatted_items = [f"{{{','.join(f'{k}:{v}' for k, v in item.items())}}}" for item in value]
                    else:
                        # Handle list of primitives (e.g., MLP units)
                        formatted_items = [str(x) for x in value]
                    target_list.append(f"'{key}=[{','.join(formatted_items)}]'")
                elif isinstance(value, str) and ("{" in value or "}" in value):
                    target_list.append(f"'{key}={value}'")
                else:
                    target_list.append(f"{key}={value}")

    print(f"[INFO]: Starting workflow {workflow}")
    process_args(cfg["runner_args"], runner_args)
    print(f"[INFO]: Retrieved workflow runner args: {runner_args}")
    process_args(cfg["hydra_args"], hydra_args, is_hydra=True)
    print(f"[INFO]: Retrieved hydra args: {hydra_args}")

    invoke_cmd = f"{python_cmd} {workflow} "
    invoke_cmd += " ".join(runner_args) + " " + " ".join(hydra_args)
    return invoke_cmd


@ray.remote
def remote_execute_job(
    job_cmd: str, identifier_string: str, test_mode: bool = False, extract_experiment: bool = False
) -> str | dict:
    """This method has an identical signature to :meth:`execute_job`, with the ray remote decorator"""
    return execute_job(
        job_cmd=job_cmd, identifier_string=identifier_string, test_mode=test_mode, extract_experiment=extract_experiment
    )


class LogExtractionError(Exception):
    """Raised when we cannot extract experiment_name/logdir from the trainer output."""

    pass


def execute_job(
    job_cmd: str,
    identifier_string: str = "job 0",
    test_mode: bool = False,
    extract_experiment: bool = False,
    persistent_dir: str | None = None,
    log_all_output: bool = False,
    max_lines_to_search_logs: int = 1000,
    max_time_to_search_logs: float = 200.0,
) -> str | dict:
    """Issue a job (shell command).

    Args:
        job_cmd: The shell command to run.
        identifier_string: What prefix to add to make logs easier to differentiate
            across clusters or jobs. Defaults to "job 0".
        test_mode: When true, only run 'nvidia-smi'. Defaults to False.
        extract_experiment: When true, search for experiment details from a training run. Defaults to False.
        persistent_dir: When supplied, change to run the directory in a persistent
            directory. Can be used to avoid losing logs in the /tmp directory. Defaults to None.
        log_all_output: When true, print all output to the console. Defaults to False.
        max_lines_to_search_logs: Maximum number of lines to search for experiment info. Defaults to 1000.
        max_time_to_search_logs: Maximum time to wait for experiment info before giving up. Defaults to 200.0 seconds.
    Raises:
        ValueError: If the job is unable to start, or throws an error. Most likely to happen
            due to running out of memory.

    Returns:
        Relevant information from the job
    """
    start_time = datetime.now().strftime("%H:%M:%S.%f")
    result_details = [f"{identifier_string}: ---------------------------------\n"]
    result_details.append(f"{identifier_string}:[INFO]: Invocation {job_cmd} \n")
    node_id = ray.get_runtime_context().get_node_id()
    result_details.append(f"{identifier_string}:[INFO]: Ray Node ID: {node_id} \n")

    if test_mode:
        import torch

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.free,serial", "--format=csv,noheader,nounits"],
                capture_output=True,
                check=True,
                text=True,
            )
            output = result.stdout.strip().split("\n")
            for gpu_info in output:
                name, memory_free, serial = gpu_info.split(", ")
                result_details.append(
                    f"{identifier_string}[INFO]: Name: {name}|Memory Available: {memory_free} MB|Serial Number"
                    f" {serial} \n"
                )

            # Get GPU count from PyTorch
            num_gpus_detected = torch.cuda.device_count()
            result_details.append(f"{identifier_string}[INFO]: Detected GPUs from PyTorch: {num_gpus_detected} \n")

            # Check CUDA_VISIBLE_DEVICES and count the number of visible GPUs
            cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            if cuda_visible_devices:
                visible_devices_count = len(cuda_visible_devices.split(","))
                result_details.append(
                    f"{identifier_string}[INFO]: GPUs visible via CUDA_VISIBLE_DEVICES: {visible_devices_count} \n"
                )
            else:
                visible_devices_count = len(output)  # All GPUs visible if CUDA_VISIBLE_DEVICES is not set
                result_details.append(
                    f"{identifier_string}[INFO]: CUDA_VISIBLE_DEVICES not set; all GPUs visible"
                    f" ({visible_devices_count}) \n"
                )

            # If PyTorch GPU count disagrees with nvidia-smi, reset CUDA_VISIBLE_DEVICES and rerun detection
            if num_gpus_detected != len(output):
                result_details.append(
                    f"{identifier_string}[WARNING]: PyTorch and nvidia-smi disagree on GPU count! Re-running with all"
                    " GPUs visible. \n"
                )
                result_details.append(f"{identifier_string}[INFO]: This shows that GPU resources were isolated.\n")
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(len(output))])
                num_gpus_detected_after_reset = torch.cuda.device_count()
                result_details.append(
                    f"{identifier_string}[INFO]: After setting CUDA_VISIBLE_DEVICES, PyTorch detects"
                    f" {num_gpus_detected_after_reset} GPUs \n"
                )

        except subprocess.CalledProcessError as e:
            print(f"Error calling nvidia-smi: {e.stderr}")
            result_details.append({"error": "Failed to retrieve GPU information"})
    else:
        if persistent_dir:
            og_dir = os.getcwd()
            os.chdir(persistent_dir)
        process = subprocess.Popen(
            job_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
        )
        process_file_descriptor = process.stdout.fileno()

        if persistent_dir:
            os.chdir(og_dir)
        experiment_name = None
        logdir = None
        experiment_info_pattern = re.compile("Exact experiment name requested from command line: (.+)")
        logdir_pattern = re.compile(r"\[INFO\] Logging experiment in directory: (.+)$")
        err_pattern = re.compile("There was an error (.+)$")

        def stream_reader(stream, identifier_string, result_details):
            for line in iter(stream.readline, ""):
                line = line.strip()
                result_details.append(f"{identifier_string}: {line}\n")
                if log_all_output:
                    print(f"{identifier_string}: {line}")

        # Read stdout until we find exp. info, up to max_lines_to_search_logs lines, max_time_to_search_logs, or EOF.
        # Do some careful handling prevent overflowing the pipe reading buffer with error 141
        lines_read = 0
        search_duration = 0.0
        search_start_time = time()
        while True:
            new_line_ready, _, _ = select.select([process_file_descriptor], [], [], 1.0)  # Wait up to 1s for stdout
            if new_line_ready:
                line = process.stdout.readline()
                if not line:  # EOF
                    break

                lines_read += 1
                line = line.strip()
                result_details.append(f"{identifier_string}: {line} \n")

                if log_all_output:
                    print(f"{identifier_string}: {line}")

                if extract_experiment:
                    exp_match = experiment_info_pattern.search(line)
                    log_match = logdir_pattern.search(line)
                    err_match = err_pattern.search(line)

                    if err_match:
                        raise ValueError(f"Encountered an error during trial run. {' '.join(result_details)}")

                    if exp_match:
                        experiment_name = exp_match.group(1)
                    if log_match:
                        logdir = log_match.group(1)

                    if experiment_name and logdir:
                        # Start stderr reader after finding experiment info
                        stderr_thread = threading.Thread(
                            target=stream_reader, args=(process.stderr, identifier_string, result_details)
                        )
                        stderr_thread.daemon = True
                        stderr_thread.start()

                        # Start stdout reader to continue reading to flush buffer
                        stdout_thread = threading.Thread(
                            target=stream_reader, args=(process.stdout, identifier_string, result_details)
                        )
                        stdout_thread.daemon = True
                        stdout_thread.start()

                        return {
                            "experiment_name": experiment_name,
                            "logdir": logdir,
                            "proc": process,
                            "result": " ".join(result_details),
                        }

            if extract_experiment:  # if we are looking for experiment info, check for timeouts and line limits
                search_duration = time() - search_start_time
                if search_duration > max_time_to_search_logs:
                    print(f"[ERROR]: Could not find experiment logs within {max_time_to_search_logs} seconds.")
                    break
                if lines_read >= max_lines_to_search_logs:
                    print(f"[ERROR]: Could not find experiment logs within first {max_lines_to_search_logs} lines.")
                    break

        # If we reach here, we didn't find experiment info in the output
        if extract_experiment and not (experiment_name and logdir):
            error_msg = (
                "Could not extract experiment_name/logdir from trainer output "
                f"(experiment_name={experiment_name!r}, logdir={logdir!r}).\n"
                "\tMake sure your training script prints the following correctly:\n"
                "\t\tExact experiment name requested from command line: <name>\n"
                "\t\t[INFO] Logging experiment in directory: <logdir>\n\n"
            )
            print(f"[ERROR]: {error_msg}")
            raise LogExtractionError("Could not extract experiment_name/logdir from training workflow output.")
        process.wait()
        now = datetime.now().strftime("%H:%M:%S.%f")
        completion_info = f"\n[INFO]: {identifier_string}: Job Started at {start_time}, completed at {now}\n"
        print(completion_info)
        result_details.append(completion_info)
        return " ".join(result_details)


def ray_init(ray_address: str = "auto", runtime_env: dict[str, Any] | None = None, log_to_driver: bool = False):
    """Initialize Ray with the given address and runtime environment."""
    if not ray.is_initialized():
        print(
            f"[INFO] Initializing Ray with address {ray_address}, log_to_driver={log_to_driver},"
            f" runtime_env={runtime_env}"
        )
        ray.init(address=ray_address, runtime_env=runtime_env, log_to_driver=log_to_driver)


def get_gpu_node_resources(
    total_resources: bool = False,
    one_node_only: bool = False,
    include_gb_ram: bool = False,
    include_id: bool = False,
) -> list[dict] | dict:
    """Get information about available GPU node resources.

    Args:
        total_resources: When true, return total available resources. Defaults to False.
        one_node_only: When true, return resources for a single node. Defaults to False.
        include_gb_ram: Set to true to convert MB to GB in result
        include_id: Set to true to include node ID
        ray_address: The ray address to connect to.

    Returns:
        Resource information for all nodes, sorted by descending GPU count, then descending CPU
        count, then descending RAM capacity, and finally by node ID in ascending order if available,
        or simply the resource for a single node if requested.
    """
    if not ray.is_initialized():
        ray_init()
    nodes = ray.nodes()
    node_resources = []
    total_cpus = 0
    total_gpus = 0
    total_memory = 0  # in bytes

    for node in nodes:
        if node["Alive"] and "GPU" in node["Resources"]:
            node_id = node["NodeID"]
            resources = node["Resources"]
            cpus = resources.get("CPU", 0)
            gpus = resources.get("GPU", 0)
            memory = resources.get("memory", 0)
            node_resources.append({"CPU": cpus, "GPU": gpus, "memory": memory})

            if include_id:
                node_resources[-1]["id"] = node_id
            if include_gb_ram:
                node_resources[-1]["ram_gb"] = memory / 1024**3

            total_cpus += cpus
            total_gpus += gpus
            total_memory += memory
    node_resources = sorted(node_resources, key=lambda x: (-x["GPU"], -x["CPU"], -x["memory"], x.get("id", "")))

    if total_resources:
        # Return summed total resources
        return {"CPU": total_cpus, "GPU": total_gpus, "memory": total_memory}

    if one_node_only and node_resources:
        return node_resources[0]

    return node_resources


def add_resource_arguments(
    arg_parser: argparse.ArgumentParser,
    defaults: list | None = None,
    cluster_create_defaults: bool = False,
) -> argparse.ArgumentParser:
    """Add resource arguments to a cluster; this is shared across both
    wrapping resources and launching clusters.

    Args:
        arg_parser: the argparser to add the arguments to. This argparser is mutated.
        defaults: The default values for GPUs, CPUs, RAM, and Num Workers
        cluster_create_defaults: Set to true to populate reasonable defaults for creating clusters.
    Returns:
        The argparser with the standard resource arguments.
    """
    if defaults is None:
        if cluster_create_defaults:
            defaults = [[1], [8], [16], [1]]
        else:
            defaults = [None, None, None, [1]]
    arg_parser.add_argument(
        "--gpu_per_worker",
        nargs="+",
        type=int,
        default=defaults[0],
        help="Number of GPUs per worker node. Supply more than one for heterogeneous resources",
    )
    arg_parser.add_argument(
        "--cpu_per_worker",
        nargs="+",
        type=int,
        default=defaults[1],
        help="Number of CPUs per worker node. Supply more than one for heterogeneous resources",
    )
    arg_parser.add_argument(
        "--ram_gb_per_worker",
        nargs="+",
        type=int,
        default=defaults[2],
        help="RAM in GB per worker node. Supply more than one for heterogeneous resources.",
    )
    arg_parser.add_argument(
        "--num_workers",
        nargs="+",
        type=int,
        default=defaults[3],
        help="Number of desired workers. Supply more than one for heterogeneous resources.",
    )
    return arg_parser


def fill_in_missing_resources(
    args: argparse.Namespace, resources: dict | None = None, cluster_creation_flag: bool = False, policy: callable = max
):
    """Normalize the lengths of resource lists based on the longest list provided."""
    print("[INFO]: Filling in missing command line arguments with best guess...")
    if resources is None:
        resources = {
            "gpu_per_worker": args.gpu_per_worker,
            "cpu_per_worker": args.cpu_per_worker,
            "ram_gb_per_worker": args.ram_gb_per_worker,
            "num_workers": args.num_workers,
        }
        if cluster_creation_flag:
            cluster_creation_resources = {"worker_accelerator": args.worker_accelerator}
            resources.update(cluster_creation_resources)

    # Calculate the maximum length of any list
    max_length = max(len(v) for v in resources.values())
    print("[INFO]: Resource list lengths:")
    for key, value in resources.items():
        print(f"[INFO] {key}: {len(value)} values {value}")

    # Extend each list to match the maximum length using the maximum value in each list
    for key, value in resources.items():
        potential_value = getattr(args, key)
        if potential_value is not None:
            max_value = policy(policy(value), policy(potential_value))
        else:
            max_value = policy(value)
        extension_length = max_length - len(value)
        if extension_length > 0:  # Only extend if the current list is shorter than max_length
            print(f"\n[WARNING]: Resource '{key}' needs extension:")
            print(f"[INFO] Current length: {len(value)}")
            print(f"[INFO] Target length: {max_length}")
            print(f"[INFO] Filling in {extension_length} missing values with {max_value}")
            print(f"[INFO] To avoid auto-filling, provide {extension_length} more {key} value(s)")
            value.extend([max_value] * extension_length)
        setattr(args, key, value)
        resources[key] = value
        print(f"[INFO] Final {key} values: {getattr(args, key)}")
    print("[INFO]: Done filling in command line arguments...\n\n")
    return args


def populate_isaac_ray_cfg_args(cfg: dict = {}) -> dict:
    """Small utility method to create empty fields if needed for a configuration."""
    if "runner_args" not in cfg:
        cfg["runner_args"] = {}
    if "hydra_args" not in cfg:
        cfg["hydra_args"] = {}
    return cfg


def _dicts_equal(d1: dict, d2: dict, tol=1e-9) -> bool:
    """Check if two dicts are equal; helps ensure only new logs are returned."""
    if d1.keys() != d2.keys():
        return False
    for key in d1:
        if isinstance(d1[key], float) and isinstance(d2[key], float):
            if not isclose(d1[key], d2[key], abs_tol=tol):
                return False
        elif d1[key] != d2[key]:
            return False
    return True


@dataclass
class JobResource:
    """A dataclass to represent a resource request for a job."""

    num_gpus: float | None = None
    num_cpus: float | None = None
    memory: int | None = None  # in bytes

    def to_opt(self) -> dict[str, Any]:
        """Convert the resource request to a dictionary."""
        opt = {}
        if self.num_gpus is not None:
            opt["num_gpus"] = self.num_gpus
        if self.num_cpus is not None:
            opt["num_cpus"] = self.num_cpus
        if self.memory is not None:
            opt["memory"] = self.memory
        return opt

    def to_pg_resources(self) -> dict[str, Any]:
        """Convert the resource request to a dictionary suitable for placement groups."""
        res = {}
        if self.num_gpus is not None:
            res["GPU"] = self.num_gpus
        if self.num_cpus is not None:
            res["CPU"] = self.num_cpus
        if self.memory is not None:
            res["memory"] = self.memory
        return res


@dataclass
class JobNode:
    """A dataclass to represent a node for job affinity."""

    specific: str | None = None
    hostname: str | None = None
    node_id: str | None = None

    def to_opt(self, nodes: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Convert node affinity settings into a dictionary of Ray actor scheduling options.

        Args:
            nodes (list[dict[str, Any]]): List of node metadata from `ray.nodes()` which looks like this:
            [{
                'NodeID': 'xxx',
                'Alive': True,
                'NodeManagerAddress': 'x.x.x.x',
                'NodeManagerHostname': 'ray-head-mjzzf',
                'NodeManagerPort': 44039,
                'ObjectManagerPort': 35689,
                'ObjectStoreSocketName': '/tmp/ray/session_xxx/sockets/plasma_store',
                'RayletSocketName': '/tmp/ray/session_xxx/sockets/raylet',
                'MetricsExportPort': 8080,
                'NodeName': 'x.x.x.x',
                'RuntimeEnvAgentPort': 63725,
                'DeathReason': 0,
                'DeathReasonMessage': '',
                'alive': True,
                'Resources': {
                    'node:__internal_head__': 1.0,
                    'object_store_memory': 422449279795.0,
                    'memory': 1099511627776.0,
                    'GPU': 8.0,
                    'node:x.x.x.x': 1.0,
                    'CPU': 192.0,
                    'accelerator_type:H20': 1.0
                    },
                'Labels': {
                    'ray.io/node_id': 'xxx'
                    }
                },...]

        Returns:
            dict[str, Any]: A dictionary with possible scheduling options:
                - Empty if no specific placement requirement.
                - "scheduling_strategy" key set to `NodeAffinitySchedulingStrategy`
                  if hostname or node_id placement is specified.

        Raises:
            ValueError: If hostname/node_id is specified but not found in the cluster
                        or the node is not alive.
        """
        opt = {}
        if self.specific is None or self.specific == "any":
            return opt
        elif self.specific == "hostname":
            if self.hostname is None:
                raise ValueError("Hostname must be specified when specific is 'hostname'")
            for node in nodes:
                if node["NodeManagerHostname"] == self.hostname:
                    if node["alive"] is False:
                        raise ValueError(f"Node {node['NodeID']} is not alive")
                    opt["scheduling_strategy"] = NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
                    return opt
            raise ValueError(f"Hostname {self.hostname} not found in nodes: {nodes}")
        elif self.specific == "node_id":
            if self.node_id is None:
                raise ValueError("Node ID must be specified when specific is 'node_id'")
            for node in nodes:
                if node["NodeID"] == self.node_id:
                    if node["alive"] is False:
                        raise ValueError(f"Node {node['NodeID']} is not alive")
                    opt["scheduling_strategy"] = NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
                    return opt
            raise ValueError(f"Node ID {self.node_id} not found in nodes: {nodes}")
        else:
            raise ValueError(f"Invalid specific value: {self.specific}. Must be 'any', 'hostname', or 'node_id'.")


@dataclass
class Job:
    """A dataclass to represent a job to be submitted to Ray."""

    # job command
    cmd: str | None = None
    py_args: str | None = None
    # identifier string for the job, e.g., "job 0"
    name: str = ""
    # job resources, e.g., {"CPU": 4, "GPU": 1}
    resources: JobResource | None = None
    # specify the node to run the job on, if needed to run on a specific node
    node: JobNode | None = None

    def to_opt(self, nodes: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Convert the job definition into a dictionary of Ray scheduling options.

        Args:
            nodes (list[dict[str, Any]]): Node information from `ray.nodes()`.

        Returns:
            dict[str, Any]: Combined scheduling options from:
                - `JobResource.to_opt()` for resource requirements
                - `JobNode.to_opt()` for node placement constraints
        """
        opt = {}
        if self.resources is not None:
            opt.update(self.resources.to_opt())
        if self.node is not None:
            opt.update(self.node.to_opt(nodes))
        return opt


@ray.remote
class JobActor:
    """Actor to run job in Ray cluster."""

    def __init__(self, job: Job, test_mode: bool, log_all_output: bool, extract_experiment: bool = False):
        self.job = job
        self.test_mode = test_mode
        self.log_all_output = log_all_output
        self.extract_experiment = extract_experiment
        self.done = True

    def ready(self) -> bool:
        """Check if the job is ready to run."""
        return self.done

    def run(self):
        """Run the job."""
        cmd = self.job.cmd if self.job.cmd else " ".join([sys.executable, *self.job.py_args.split()])
        return execute_job(
            job_cmd=cmd,
            identifier_string=self.job.name,
            test_mode=self.test_mode,
            extract_experiment=self.extract_experiment,
            log_all_output=self.log_all_output,
        )


def submit_wrapped_jobs(
    jobs: Sequence[Job],
    log_realtime: bool = True,
    test_mode: bool = False,
    concurrent: bool = False,
) -> None:
    """
    Submit a list of jobs to the Ray cluster and manage their execution.

    Args:
        jobs (Sequence[Job]): A sequence of Job objects to execute on Ray.
        log_realtime (bool): Whether to log stdout/stderr in real-time. Defaults to True.
        test_mode (bool): If True, run in GPU sanity-check mode instead of actual jobs. Defaults to False.
        concurrent (bool): Whether to launch tasks simultaneously as a batch,
                           or independently as resources become available. Defaults to False.

    Returns:
        None
    """
    if jobs is None or len(jobs) == 0:
        print("[WARNING]: No jobs to submit")
        return
    if not ray.is_initialized():
        raise Exception("Ray is not initialized. Please initialize Ray before submitting jobs.")
    nodes = ray.nodes()
    actors = []
    for i, job in enumerate(jobs):
        opts = job.to_opt(nodes)
        name = job.name or f"job_{i + 1}"
        print(f"[INFO] Create {name} with opts={opts}")
        job_actor = JobActor.options(**opts).remote(job, test_mode, log_realtime)
        actors.append(job_actor)
    try:
        if concurrent:
            ray.get([actor.ready.remote() for actor in actors])
            print("[INFO] All actors are ready to run.")
        future = [actor.run.remote() for actor in actors]
        while future:
            ready, not_ready = ray.wait(future, timeout=5)
            for result in ray.get(ready):
                print(f"\n{result}\n")
            future = not_ready
        print("[INFO] all jobs completed.")
    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt received, cancelling …")
        for actor in actors:
            ray.cancel(actor, force=True)
        sys.exit(0)
