# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

"""
This script requires that kubectl is installed and KubeRay was used to create the cluster.

Creates a config file containing ``name: <NAME> address: http://<IP>:<PORT>`` on
a new line for each cluster, and also fetches the MLFlow URI.

Usage:

.. code-block:: bash

    python3 scripts/reinforcement_learning/ray/grok_cluster_with_kubectl.py
    # For options, supply -h arg
"""


def get_namespace() -> str:
    """Get the current Kubernetes namespace from the context, fallback to default if not set"""
    try:
        namespace = (
            subprocess.check_output(["kubectl", "config", "view", "--minify", "--output", "jsonpath={..namespace}"])
            .decode()
            .strip()
        )
        if not namespace:
            namespace = "default"
    except subprocess.CalledProcessError:
        namespace = "default"
    return namespace


def get_pods(namespace: str = "default") -> list[tuple]:
    """Get a list of all of the pods in the namespace"""
    cmd = ["kubectl", "get", "pods", "-n", namespace, "--no-headers"]
    output = subprocess.check_output(cmd).decode()
    pods = []
    for line in output.strip().split("\n"):
        fields = line.split()
        pod_name = fields[0]
        status = fields[2]
        pods.append((pod_name, status))
    return pods


def get_clusters(pods: list, cluster_name_prefix: str) -> set:
    """
    Get unique cluster name(s). Works for one or more clusters, based off of the number of head nodes.
    Excludes MLflow deployments.
    """
    clusters = set()
    for pod_name, _ in pods:
        # Skip MLflow pods
        if "-mlflow" in pod_name:
            continue

        match = re.match(r"(" + re.escape(cluster_name_prefix) + r"[-\w]+)", pod_name)
        if match:
            # Get base name without head/worker suffix (skip workers)
            if "head" in pod_name:
                base_name = match.group(1).split("-head")[0]
                clusters.add(base_name)
    return sorted(clusters)


def get_mlflow_info(namespace: str = None, cluster_prefix: str = "isaacray") -> str:
    """
    Get MLflow service information if it exists in the namespace with the given prefix.
    Only works for a single cluster instance.
    Args:
        namespace: Kubernetes namespace
        cluster_prefix: Base cluster name (without -head/-worker suffixes)
    Returns:
        MLflow service URL
    """
    # Strip any -head or -worker suffixes to get base name
    if namespace is None:
        namespace = get_namespace()
    pods = get_pods(namespace=namespace)
    clusters = get_clusters(pods=pods, cluster_name_prefix=cluster_prefix)
    if len(clusters) > 1:
        raise ValueError("More than one cluster matches prefix, could not automatically determine mlflow info.")
    mlflow_name = f"{cluster_prefix}-mlflow"

    cmd = ["kubectl", "get", "svc", mlflow_name, "-n", namespace, "--no-headers"]
    try:
        output = subprocess.check_output(cmd).decode()
        fields = output.strip().split()

        # Get cluster IP
        cluster_ip = fields[2]
        port = "5000"  # Default MLflow port
        # This needs to be http to be resolved. HTTPS can't be resolved
        # This should be fine as it is on a subnet on the cluster regardless
        return f"http://{cluster_ip}:{port}"
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Could not grok MLflow: {e}")  # Fixed f-string


def check_clusters_running(pods: list, clusters: set) -> bool:
    """
    Check that all of the pods in all provided clusters are running.

    Args:
        pods (list): A list of tuples where each tuple contains the pod name and its status.
        clusters (set): A set of cluster names to check.

    Returns:
        bool: True if all pods in any of the clusters are running, False otherwise.
    """
    clusters_running = False
    for cluster in clusters:
        cluster_pods = [p for p in pods if p[0].startswith(cluster)]
        total_pods = len(cluster_pods)
        running_pods = len([p for p in cluster_pods if p[1] == "Running"])
        if running_pods == total_pods and running_pods > 0:
            clusters_running = True
            break
    return clusters_running


def get_ray_address(head_pod: str, namespace: str = "default", ray_head_name: str = "head") -> str:
    """
    Given a cluster head pod, check its logs, which should include the ray address which can accept job requests.

    Args:
        head_pod (str): The name of the head pod.
        namespace (str, optional): The Kubernetes namespace. Defaults to "default".
        ray_head_name (str, optional): The name of the ray head container. Defaults to "head".

    Returns:
        str: The ray address if found, None otherwise.

    Raises:
        ValueError: If the logs cannot be retrieved or the ray address is not found.
    """
    cmd = ["kubectl", "logs", head_pod, "-c", ray_head_name, "-n", namespace]
    try:
        output = subprocess.check_output(cmd).decode()
    except subprocess.CalledProcessError as e:
        raise ValueError(
            f"Could not enter head container with cmd {cmd}: {e}Perhaps try a different namespace or ray head name."
        )
    match = re.search(r"RAY_ADDRESS='([^']+)'", output)
    if match:
        return match.group(1)
    else:
        return None


def process_cluster(cluster_info: dict, ray_head_name: str = "head") -> str:
    """
    For each cluster, check that it is running, and get the Ray head address that will accept jobs.

    Args:
        cluster_info (dict): A dictionary containing cluster information with keys 'cluster', 'pods', and 'namespace'.
        ray_head_name (str, optional): The name of the ray head container. Defaults to "head".

    Returns:
        str: A string containing the cluster name and its Ray head address, or an error message if the head pod or Ray address is not found.
    """
    cluster, pods, namespace = cluster_info
    head_pod = None
    for pod_name, status in pods:
        if pod_name.startswith(cluster + "-head"):
            head_pod = pod_name
            break
    if not head_pod:
        return f"Error: Could not find head pod for cluster {cluster}\n"

    # Get RAY_ADDRESS and status
    ray_address = get_ray_address(head_pod, namespace=namespace, ray_head_name=ray_head_name)
    if not ray_address:
        return f"Error: Could not find RAY_ADDRESS for cluster {cluster}\n"

    # Return only cluster and ray address
    return f"name: {cluster} address: {ray_address}\n"


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process Ray clusters and save their specifications.")
    parser.add_argument("--prefix", default="isaacray", help="The prefix for the cluster names.")
    parser.add_argument("--output", default="~/.cluster_config", help="The file to save cluster specifications.")
    parser.add_argument("--ray_head_name", default="head", help="The metadata name for the ray head container")
    parser.add_argument(
        "--namespace", help="Kubernetes namespace to use. If not provided, will detect from current context."
    )
    args = parser.parse_args()

    # Get namespace from args or detect it
    current_namespace = args.namespace if args.namespace else get_namespace()
    print(f"Using namespace: {current_namespace}")

    cluster_name_prefix = args.prefix
    cluster_spec_file = os.path.expanduser(args.output)

    # Get all pods
    pods = get_pods(namespace=current_namespace)

    # Get clusters
    clusters = get_clusters(pods, cluster_name_prefix)
    if not clusters:
        print(f"No clusters found with prefix {cluster_name_prefix}")
        return

    # Wait for clusters to be running
    while True:
        pods = get_pods(namespace=current_namespace)
        if check_clusters_running(pods, clusters):
            break
        print("Waiting for all clusters to spin up...")
        time.sleep(5)

    print("Checking for MLflow:")
    # Check MLflow status for each cluster
    for cluster in clusters:
        try:
            mlflow_address = get_mlflow_info(current_namespace, cluster)
            print(f"MLflow address for {cluster}: {mlflow_address}")
        except ValueError as e:
            print(f"ML Flow not located: {e}")
    print()

    # Prepare cluster info for parallel processing
    cluster_infos = []
    for cluster in clusters:
        cluster_pods = [p for p in pods if p[0].startswith(cluster)]
        cluster_infos.append((cluster, cluster_pods, current_namespace))

    # Use ThreadPoolExecutor to process clusters in parallel
    results = []
    results_lock = threading.Lock()

    with ThreadPoolExecutor() as executor:
        future_to_cluster = {
            executor.submit(process_cluster, info, args.ray_head_name): info[0] for info in cluster_infos
        }
        for future in as_completed(future_to_cluster):
            cluster_name = future_to_cluster[future]
            try:
                result = future.result()
                with results_lock:
                    results.append(result)
            except Exception as exc:
                print(f"{cluster_name} generated an exception: {exc}")

    # Sort results alphabetically by cluster name
    results.sort()

    # Write sorted results to the output file (Ray info only)
    with open(cluster_spec_file, "w") as f:
        for result in results:
            f.write(result)

    print(f"Cluster spec information saved to {cluster_spec_file}")
    # Display the contents of the config file
    with open(cluster_spec_file) as f:
        print(f.read())


if __name__ == "__main__":
    main()
