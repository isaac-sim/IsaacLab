# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

from ray import job_submission

"""
This script submits aggregate job(s) to cluster(s) described in a
config file containing ``name: <NAME> address: http://<IP>:<PORT>`` on
a new line for each cluster. For KubeRay clusters, this file
can be automatically created with :file:`grok_cluster_with_kubectl.py`

Aggregate job(s) are matched with cluster(s) via the following relation:
cluster_line_index_submitted_to = job_index % total_cluster_count

Aggregate jobs are separated by the * delimiter. The ``--aggregate_jobs`` argument must be
the last argument supplied to the script.

An aggregate job could be a :file:`../tuner.py` tuning job, which automatically
creates several individual jobs when started on a cluster. Alternatively, an aggregate job
could be a :file:'../wrap_resources.py` resource-wrapped job,
which may contain several individual sub-jobs separated by
the + delimiter.

If there are more aggregate jobs than cluster(s), aggregate jobs will be submitted
as clusters become available via the defined relation above. If there are less aggregate job(s)
than clusters, some clusters will not receive aggregate job(s). The maximum number of
aggregate jobs that can be run simultaneously is equal to the number of workers created by
default by a ThreadPoolExecutor on the machine submitting jobs due to fetching the log output after
jobs finish, which is unlikely to constrain overall-job submission.

Usage:

.. code-block:: bash

    # Example; submitting a tuning job
    python3 scripts/reinforcement_learning/ray/submit_job.py \
    --aggregate_jobs /workspace/isaaclab/scripts/reinforcement_learning/ray/tuner.py \
        --cfg_file hyperparameter_tuning/vision_cartpole_cfg.py \
        --cfg_class CartpoleTheiaJobCfg --mlflow_uri <ML_FLOW_URI>

    # Example: Submitting resource wrapped job
    python3 scripts/reinforcement_learning/ray/submit_job.py --aggregate_jobs wrap_resources.py --test

    # For all command line arguments
    python3 scripts/reinforcement_learning/ray/submit_job.py -h
"""
script_directory = os.path.dirname(os.path.abspath(__file__))
CONFIG = {"working_dir": script_directory, "executable": "/workspace/isaaclab/isaaclab.sh -p"}


def read_cluster_spec(fn: str | None = None) -> list[dict]:
    if fn is None:
        cluster_spec_path = os.path.expanduser("~/.cluster_config")
    else:
        cluster_spec_path = os.path.expanduser(fn)

    if not os.path.exists(cluster_spec_path):
        raise FileNotFoundError(f"Cluster spec file not found at {cluster_spec_path}")

    clusters = []
    with open(cluster_spec_path) as f:
        for line in f:
            parts = line.strip().split(" ")
            http_address = parts[3]
            cluster_info = {"name": parts[1], "address": http_address}
            print(f"[INFO] Setting {cluster_info['name']}")  # with {cluster_info['num_gpu']} GPUs.")
            clusters.append(cluster_info)

    return clusters


def submit_job(cluster: dict, job_command: str) -> None:
    """
    Submits a job to a single cluster, prints the final result and Ray dashboard URL at the end.
    """
    address = cluster["address"]
    cluster_name = cluster["name"]
    print(f"[INFO]: Submitting job to cluster '{cluster_name}' at {address}")  # with {num_gpus} GPUs.")
    client = job_submission.JobSubmissionClient(address)
    runtime_env = {"working_dir": CONFIG["working_dir"], "executable": CONFIG["executable"]}
    print(f"[INFO]: Checking contents of the directory: {CONFIG['working_dir']}")
    try:
        dir_contents = os.listdir(CONFIG["working_dir"])
        print(f"[INFO]: Directory contents: {dir_contents}")
    except Exception as e:
        print(f"[INFO]: Failed to list directory contents: {str(e)}")
    entrypoint = f"{CONFIG['executable']} {job_command}"
    print(f"[INFO]: Attempting entrypoint {entrypoint=} in cluster {cluster}")
    job_id = client.submit_job(entrypoint=entrypoint, runtime_env=runtime_env)
    status = client.get_job_status(job_id)
    while status in [job_submission.JobStatus.PENDING, job_submission.JobStatus.RUNNING]:
        time.sleep(5)
        status = client.get_job_status(job_id)

    final_logs = client.get_job_logs(job_id)
    print("----------------------------------------------------")
    print(f"[INFO]: Cluster {cluster_name} Logs: \n")
    print(final_logs)
    print("----------------------------------------------------")


def submit_jobs_to_clusters(jobs: list[str], clusters: list[dict]) -> None:
    """
    Submit all jobs to their respective clusters, cycling through clusters if there are more jobs than clusters.
    """
    if not clusters:
        raise ValueError("No clusters available for job submission.")

    if len(jobs) < len(clusters):
        print("[INFO]: Less jobs than clusters, some clusters will not receive jobs")
    elif len(jobs) == len(clusters):
        print("[INFO]: Exactly one job per cluster")
    else:
        print("[INFO]: More jobs than clusters, jobs submitted as clusters become available.")
    with ThreadPoolExecutor() as executor:
        for idx, job_command in enumerate(jobs):
            # Cycle through clusters using modulus to wrap around if there are more jobs than clusters
            cluster = clusters[idx % len(clusters)]
            executor.submit(submit_job, cluster, job_command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit multiple GPU jobs to multiple Ray clusters.")
    parser.add_argument("--config_file", default="~/.cluster_config", help="The cluster config path.")
    parser.add_argument(
        "--aggregate_jobs",
        type=str,
        nargs=argparse.REMAINDER,
        help="This should be last argument. The aggregate jobs to submit separated by the * delimiter.",
    )
    args = parser.parse_args()
    if args.aggregate_jobs is not None:
        jobs = " ".join(args.aggregate_jobs)
        formatted_jobs = jobs.split("*")
        if len(formatted_jobs) > 1:
            print("Warning; Split jobs by cluster with the * delimiter")
    else:
        formatted_jobs = []
    print(f"[INFO]: Isaac Ray Wrapper received jobs {formatted_jobs=}")
    clusters = read_cluster_spec(args.config_file)
    submit_jobs_to_clusters(formatted_jobs, clusters)
