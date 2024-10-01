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

SCRIPT_DIR = pathlib.Path(__file__).parent


def apply_manifest(args: argparse.Namespace) -> None:
    # Load Kubernetes configuration
    config.load_kube_config()

    # Set up Jinja2 environment for loading templates
    templates_dir = SCRIPT_DIR / "templates"
    file_loader = FileSystemLoader(str(templates_dir))
    jinja_env = Environment(loader=file_loader, keep_trailing_newline=True)

    # Define template filename
    template_file = "tune.yaml.jinja"

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
        print(cleaned_yaml_string)
        print("not running yet, for inspection")
        # subprocess.run(["kubectl", "apply", "-f", "-"], input=cleaned_yaml_string, text=True, check=True)
    except subprocess.CalledProcessError as e:
        exit(f"An error occurred while running `kubectl`: {e}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for Kubernetes deployment script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    arg_parser = argparse.ArgumentParser(
        description="Script to apply manifests to create Kubernetes objects for Ray clusters."
    )

    arg_parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of the Kubernetes deployment.",
    )

    arg_parser.add_argument(
        "--namespace",
        type=str,
        required=True,
        help="Kubernetes namespace to deploy the Ray cluster.",
    )

    arg_parser.add_argument("--service_acount_name", type=str, required=True, help="The service account name to use.")

    arg_parser.add_argument(
        "--external_dns_hostname", type=str, required=True, help="The external hostname to use to view logs in progress"
    )

    arg_parser.add_argument(
        "head_entry_point",
        nargs="+",
        default=["python", "no_entrypoint_supplied_silly.py"],  # TODO: Make this whatever the tune is
        help="The program to run from the head node to facilitate a hyperparam run",
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
        "--min-workers",
        type=int,
        required=False,
        default=2,
        help="Minimum number of workers.",
    )

    arg_parser.add_argument(
        "--max-workers",
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

    arg_parser.add_argument("--cpu_per_worker", type=int, default=1, help="Number of CPUs to assign to each worker pod")

    arg_parser.add_argument(
        "--gpu_per_worker",
        type=int,
        default=1,
        help="Number of GPUs to assign to each worker pod.",
    )

    arg_parser.add_argument("--ram_gb", type=int, default=50, help="How many gigs of RAM to use")

    return arg_parser.parse_args()


if __name__ == "__main__":
    """
    Usage:
    python3 launch.py -h
    """
    args = parse_args()
    apply_manifest(args)
