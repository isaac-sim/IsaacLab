# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import pathlib
import subprocess
import yaml

from jinja2 import Environment, FileSystemLoader
from kubernetes import config

RAY_DIR = pathlib.Path(__file__).parent


def apply_manifest(args: argparse.Namespace) -> None:
    """Provided a Jinja templated ray.io/v1alpha1 file,
    populate the arguments and create the cluster.

    Args:
        args: Possible arguments concerning cluster parameters.
    """
    # Load Kubernetes configuration
    config.load_kube_config()

    # Set up Jinja2 environment for loading templates
    templates_dir = RAY_DIR / "cluster_configs" / args.cluster_host
    print(templates_dir)
    file_loader = FileSystemLoader(str(templates_dir))
    jinja_env = Environment(loader=file_loader, keep_trailing_newline=True)

    # Define template filename
    template_file = "kuberay.yaml.jinja"

    # Convert args namespace to a dictionary
    template_params = vars(args)

    # Load and render the template
    template = jinja_env.get_template(template_file)
    file_contents = template.render(template_params)

    # Parse the rendered YAML
    parsed_yaml = yaml.safe_load(file_contents)
    cleaned_yaml_string = yaml.dump(parsed_yaml)

    # Apply the Kubernetes manifest using kubectl
    try:
        print("Populated the following Ray Configuration from the config file and CLI args")
        print("<START OF CONFIG BELOW")
        print(cleaned_yaml_string)
        print("<END OF CONFIG ABOVE")
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
        choices=["google_cloud", "local"],
        help=(
            "In the cluster_configs directory, the name of the folder where a tune.yaml.jinja"
            "file exists defining the KubeRay config. Currently only google_cloud is supported."
        ),
    )

    arg_parser.add_argument(
        "--name",
        type=str,
        required=False,
        default="isaac-lab-hyperparameter-tuner",
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
        type=str,
        default="nvidia-l4",
        help="The name of a gpu accelerator available with Google.",
    )

    arg_parser.add_argument(
        "--min_workers",
        type=int,
        required=False,
        default=2,
        help="Minimum number of workers.",
    )

    arg_parser.add_argument(
        "--max_workers",
        type=int,
        required=False,
        default=8,
        help="Maximum number of workers.",
    )

    arg_parser.add_argument(
        "--starting_worker_count",
        type=int,
        default=1,
        help="Initial number of workers to start with.",
    )

    arg_parser.add_argument("--cpu_per_worker", type=int, default=8, help="Number of CPUs to assign to each worker pod")

    arg_parser.add_argument(
        "--gpu_per_worker",
        type=int,
        default=1,
        help="Number of GPUs to assign to each worker pod.",
    )

    arg_parser.add_argument("--worker_ram_gb", type=int, default=16, help="How many gigs of RAM to use")

    arg_parser.add_argument(
        "--num_clusters",
        type=int,
        default=1,
        help=(
            "How many Ray Clusters to create."
            "This should be greater than 1 "
            "ONLY IF you are tuning several agents "
            "OR IF you are tuning agents "
            "in separate environments at the same time."
        ),
    )
    arg_parser.add_argument(
        "--num_head_cpu",
        type=float,  # to be able to schedule partial CPU heads
        default=4,
        help="The number of CPUs to give the Ray head.",
    )
    arg_parser.add_argument(
        "--secret_name", default="bucket-access", type=str, help="The name of the secret that allows for bucket access."
    )

    arg_parser.add_argument("--head_ram_gb", type=int, default=4, help="How many gigs of ram to give the Ray head")

    return arg_parser.parse_args()


if __name__ == "__main__":
    """
    Usage:
    python3 launch.py -h
    """
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
