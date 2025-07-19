# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import pathlib
import subprocess
import yaml

import util
from jinja2 import Environment, FileSystemLoader
from kubernetes import config

"""This script helps create one or more KubeRay clusters.

Usage:

.. code-block:: bash
    # If the head node is stuck on container creating, make sure to create a secret
    python3 scripts/reinforcement_learning/ray/launch.py -h

    # Examples

    # The following creates 8 GPUx1 nvidia l4 workers
    python3 scripts/reinforcement_learning/ray/launch.py --cluster_host google_cloud \
    --namespace <NAMESPACE> --image <YOUR_ISAAC_RAY_IMAGE> \
    --num_workers 8 --num_clusters 1 --worker_accelerator nvidia-l4 --gpu_per_worker 1

    # The following creates 1 GPUx1 nvidia l4 worker, 2 GPUx2 nvidia-tesla-t4 workers,
    # and 2 GPUx4 nvidia-tesla-t4 GPU workers
    python3 scripts/reinforcement_learning/ray/launch.py --cluster_host google_cloud \
    --namespace <NAMESPACE> --image <YOUR_ISAAC_RAY_IMAGE> \
    --num_workers 1 2 --num_clusters 1 \
    --worker_accelerator nvidia-l4 nvidia-tesla-t4 --gpu_per_worker 1 2 4
"""
RAY_DIR = pathlib.Path(__file__).parent


def apply_manifest(args: argparse.Namespace) -> None:
    """Provided a Jinja templated ray.io/v1alpha1 file,
    populate the arguments and create the cluster. Additionally, create
    kubernetes containers for resources separated by '---' from the rest
    of the file.

    Args:
        args: Possible arguments concerning cluster parameters.
    """
    # Load Kubernetes configuration
    config.load_kube_config()

    # Set up Jinja2 environment for loading templates
    templates_dir = RAY_DIR / "cluster_configs" / args.cluster_host
    file_loader = FileSystemLoader(str(templates_dir))
    jinja_env = Environment(loader=file_loader, keep_trailing_newline=True, autoescape=True)

    # Define template filename
    template_file = "kuberay.yaml.jinja"

    # Convert args namespace to a dictionary
    template_params = vars(args)

    # Load and render the template
    template = jinja_env.get_template(template_file)
    file_contents = template.render(template_params)

    # Parse all YAML documents in the rendered template
    all_yamls = []
    for doc in yaml.safe_load_all(file_contents):
        all_yamls.append(doc)

    # Convert back to YAML string, preserving multiple documents
    cleaned_yaml_string = ""
    for i, doc in enumerate(all_yamls):
        if i > 0:
            cleaned_yaml_string += "\n---\n"
        cleaned_yaml_string += yaml.dump(doc)

    # Apply the Kubernetes manifest using kubectl
    try:
        print(cleaned_yaml_string)
        subprocess.run(["kubectl", "apply", "-f", "-"], input=cleaned_yaml_string, text=True, check=True)
    except subprocess.CalledProcessError as e:
        exit(f"An error occurred while running `kubectl`: {e}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for Kubernetes deployment script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    arg_parser = argparse.ArgumentParser(
        description="Script to apply manifests to create Kubernetes objects for Ray clusters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    arg_parser.add_argument(
        "--cluster_host",
        type=str,
        default="google_cloud",
        choices=["google_cloud"],
        help=(
            "In the cluster_configs directory, the name of the folder where a tune.yaml.jinja"
            "file exists defining the KubeRay config. Currently only google_cloud is supported."
        ),
    )

    arg_parser.add_argument(
        "--name",
        type=str,
        required=False,
        default="isaacray",
        help="Name of the Kubernetes deployment.",
    )

    arg_parser.add_argument(
        "--namespace",
        type=str,
        required=False,
        default="default",
        help="Kubernetes namespace to deploy the Ray cluster.",
    )

    arg_parser.add_argument(
        "--service_acount_name", type=str, required=False, default="default", help="The service account name to use."
    )

    arg_parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Docker image for the Ray cluster pods.",
    )

    arg_parser.add_argument(
        "--worker_accelerator",
        nargs="+",
        type=str,
        default=["nvidia-l4"],
        help="GPU accelerator name. Supply more than one for heterogeneous resources.",
    )

    arg_parser = util.add_resource_arguments(arg_parser, cluster_create_defaults=True)

    arg_parser.add_argument(
        "--num_clusters",
        type=int,
        default=1,
        help="How many Ray Clusters to create.",
    )
    arg_parser.add_argument(
        "--num_head_cpu",
        type=float,  # to be able to schedule partial CPU heads
        default=8,
        help="The number of CPUs to give the Ray head.",
    )

    arg_parser.add_argument("--head_ram_gb", type=int, default=8, help="How many gigs of ram to give the Ray head")
    args = arg_parser.parse_args()
    return util.fill_in_missing_resources(args, cluster_creation_flag=True)


def main():
    args = parse_args()

    if "head" in args.name:
        raise ValueError("For compatibility with other scripts, do not include head in the name")
    if args.num_clusters == 1:
        apply_manifest(args)
    else:
        default_name = args.name
        for i in range(args.num_clusters):
            args.name = default_name + "-" + str(i)
            apply_manifest(args)


if __name__ == "__main__":
    main()
