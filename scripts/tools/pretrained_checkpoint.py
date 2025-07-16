# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to manage pretrained checkpoints for our environments.
"""

import argparse

from isaaclab.app import AppLauncher

# Initialize the parser
parser = argparse.ArgumentParser(
    description="""
Script used for the training and publishing of pre-trained checkpoints for Isaac Lab.

Examples :
    # Train an agent using the rl_games workflow on the Isaac-Cartpole-v0 environment.
    pretrained_checkpoint.py --train rl_games:Isaac-Cartpole-v0
    # Train and publish the checkpoints for all workflows on only the direct Cartpole environments.
    pretrained_checkpoint.py -tp "*:Isaac-Cartpole-*Direct-v0" \\
      --/persistent/isaaclab/asset_root/pretrained_checkpoints="/some/path"
    # Review all repose cube jobs, excluding the Play tasks and skrl
    pretrained_checkpoint.py -r "*:*Repose-Cube*" --exclude "*:*Play*" --exclude skrl:*
    # Publish all results (that have been reviewed and approved).
    pretrained_checkpoint.py --publish --all \\
      --/persistent/isaaclab/asset_root/pretrained_checkpoints="/some/path"
""",
    formatter_class=argparse.RawTextHelpFormatter,
)

# Add positional arguments that can accept zero or more values
parser.add_argument(
    "jobs",
    nargs="*",
    help="""
A job consists of a workflow and a task name separated by a colon (wildcards optional), for example :
    rl_games:Isaac-Humanoid-*v0
    rsl_rl:Isaac-Ant-*-v0
    *:Isaac-Velocity-Flat-Spot-v0
""",
)
parser.add_argument("-t", "--train", action="store_true", help="Train checkpoints for later publishing.")
parser.add_argument("-p", "--publish_checkpoint", action="store_true", help="Publish pre-trained checkpoints.")
parser.add_argument("-r", "--review", action="store_true", help="Review checkpoints.")
parser.add_argument("-l", "--list", action="store_true", help="List all available environments and workflows.")
parser.add_argument("-f", "--force", action="store_true", help="Force training when results already exist.")
parser.add_argument("-a", "--all", action="store_true", help="Run all valid workflow task pairs.")
parser.add_argument(
    "-E",
    "--exclude",
    action="append",
    type=str,
    default=[],
    help="Excludes jobs matching the argument, with wildcard support.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--force_review", action="store_true", help="Forces review when one already exists.")
parser.add_argument("--force_publish", action="store_true", help="Publish checkpoints without review.")
parser.add_argument("--headless", action="store_true", help="Run training without the UI.")

args, _ = parser.parse_known_args()

# Need something to do
if len(args.jobs) == 0 and not args.all:
    parser.error("Jobs must be provided, or --all.")

# Must train, publish, review or list
if not (args.train or args.publish_checkpoint or args.review or args.list):
    parser.error("A train, publish, review or list flag must be given.")

# List excludes train and publish
if args.list and (args.train or args.publish_checkpoint):
    parser.error("Can't train or publish when listing.")

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


import csv

# Now everything else
import fnmatch
import gymnasium as gym
import json
import numpy as np
import os
import subprocess
import sys

import omni.client
from omni.client._omniclient import CopyBehavior

from isaaclab.utils.pretrained_checkpoint import (
    WORKFLOW_EXPERIMENT_NAME_VARIABLE,
    WORKFLOW_PLAYER,
    WORKFLOW_TRAINER,
    WORKFLOWS,
    get_log_root_path,
    get_pretrained_checkpoint_path,
    get_pretrained_checkpoint_publish_path,
    get_pretrained_checkpoint_review,
    get_pretrained_checkpoint_review_path,
    has_pretrained_checkpoint_job_finished,
    has_pretrained_checkpoint_job_run,
    has_pretrained_checkpoints_asset_root_dir,
)

# Need somewhere to publish
if args.publish_checkpoint and not has_pretrained_checkpoints_asset_root_dir():
    raise Exception("A /persistent/isaaclab/asset_root/pretrained_checkpoints setting is required to publish.")


def train_job(workflow, task_name, headless=False, force=False, num_envs=None):
    """
    This trains a task using the workflow's train.py script, overriding the experiment name to ensure unique
    log directories.  By default it will return if an experiment has already been run.

    Args:
        workflow: The workflow.
        task_name: The task name.
        headless: Should the training run without the UI.
        force: Run training even if previous experiments have been run.
        num_envs: How many simultaneous environments to simulate, overriding the config.
    """

    log_root_path = get_log_root_path(workflow, task_name)

    # We already ran this
    if not force and os.path.exists(log_root_path) and len(os.listdir(log_root_path)) > 0:
        print(f"Skipping training of {workflow}:{task_name}, already has been run")
        return

    print(f"Training {workflow}:{task_name}")

    # Construct our command
    cmd = [
        sys.executable,
        WORKFLOW_TRAINER[workflow],
        "--task",
        task_name,
        "--enable_cameras",
    ]

    # Changes the directory name for logging
    if WORKFLOW_EXPERIMENT_NAME_VARIABLE[workflow]:
        cmd.append(f"{WORKFLOW_EXPERIMENT_NAME_VARIABLE[workflow]}={task_name}")

    if headless:
        cmd.append("--headless")
    if num_envs:
        cmd.extend(["--num_envs", str(num_envs)])

    print("Running : " + " ".join(cmd))

    subprocess.run(cmd)


def review_pretrained_checkpoint(workflow, task_name, force_review=False, num_envs=None):
    """
    This initiates a review of the pretrained checkpoint.  The play.py script for the workflow is run, and the user
    inspects the results.  When done they close the simulator and will be prompted for their review.

    Args:
        workflow: The workflow.
        task_name: The task name.
        force_review: Performs the review even if a review already exists.
        num_envs: How many simultaneous environments to simulate, overriding the config.
    """

    # This workflow task pair hasn't been trained
    if not has_pretrained_checkpoint_job_run(workflow, task_name):
        print(f"Skipping review of {workflow}:{task_name}, hasn't been trained yet")
        return

    # Couldn't find the checkpoint
    if not has_pretrained_checkpoint_job_finished(workflow, task_name):
        print(f"Training not complete for {workflow}:{task_name}")
        return

    review = get_pretrained_checkpoint_review(workflow, task_name)

    if not force_review and review and review["reviewed"]:
        print(f"Review already complete for {workflow}:{task_name}")
        return

    print(f"Reviewing {workflow}:{task_name}")

    # Construct our command
    cmd = [
        sys.executable,
        WORKFLOW_PLAYER[workflow],
        "--task",
        task_name,
        "--checkpoint",
        get_pretrained_checkpoint_path(workflow, task_name),
        "--enable_cameras",
    ]

    if num_envs:
        cmd.extend(["--num_envs", str(num_envs)])

    print("Running : " + " ".join(cmd))

    subprocess.run(cmd)

    # Give user a chance to leave the old review
    if force_review and review and review["reviewed"]:
        result = review["result"]
        notes = review.get("notes")
        print(f"A review already exists for {workflow}:{task_name}, it was marked as '{result}'.")
        print(f"  Notes: {notes}")
        answer = input("Would you like to replace it?  Please answer yes or no (y/n) [n]: ").strip().lower()
        if answer != "y":
            return

    # Get the verdict from the user
    print(f"Do you accept this checkpoint for {workflow}:{task_name}?")

    answer = input("Please answer yes, no or undetermined (y/n/u) [u]: ").strip().lower()
    if answer not in {"y", "n", "u"}:
        answer = "u"
    answer_map = {
        "y": "accepted",
        "n": "rejected",
        "u": "undetermined",
    }

    # Create the review dict
    review = {
        "reviewed": True,
        "result": answer_map[answer],
    }

    # Maybe add some notes
    notes = input("Please add notes or hit enter: ").strip().lower()
    if notes:
        review["notes"] = notes

    # Save the review JSON file
    path = get_pretrained_checkpoint_review_path(workflow, task_name)
    if not path:
        raise Exception("This shouldn't be possible, something went very wrong.")

    with open(path, "w") as f:
        json.dump(review, f, indent=4)


def publish_pretrained_checkpoint(workflow, task_name, force_publish=False):
    """
    This publishes the pretrained checkpoint to Nucleus using the asset path in the
    /persistent/isaaclab/asset_root/pretrained_checkpoints Carb variable.

    Args:
        workflow: The workflow.
        task_name: The task name.
        force_publish: Publish without review.
    """

    # This workflow task pair hasn't been trained
    if not has_pretrained_checkpoint_job_run(workflow, task_name):
        print(f"Skipping publishing of {workflow}:{task_name}, hasn't been trained yet")
        return

    # Couldn't find the checkpoint
    if not has_pretrained_checkpoint_job_finished(workflow, task_name):
        print(f"Training not complete for {workflow}:{task_name}")
        return

    # Get local pretrained checkpoint path
    local_path = get_pretrained_checkpoint_path(workflow, task_name)
    if not local_path:
        raise Exception("This shouldn't be possible, something went very wrong.")

    # Not forcing, need to check review results
    if not force_publish:

        # Grab the review if it exists
        review = get_pretrained_checkpoint_review(workflow, task_name)

        if not review or not review["reviewed"]:
            print(f"Skipping publishing of {workflow}:{task_name}, hasn't been reviewed yet")
            return

        result = review["result"]
        if result != "accepted":
            print(f'Skipping publishing of {workflow}:{task_name}, review result was "{result}"')
            return

    print(f"Publishing {workflow}:{task_name}")

    # Copy the file
    publish_path = get_pretrained_checkpoint_publish_path(workflow, task_name)
    omni.client.copy_file(local_path, publish_path, CopyBehavior.OVERWRITE)


def get_job_summary_row(workflow, task_name):
    """Returns a single row summary of the job"""

    has_run = has_pretrained_checkpoint_job_run(workflow, task_name)
    has_finished = has_pretrained_checkpoint_job_finished(workflow, task_name)
    review = get_pretrained_checkpoint_review(workflow, task_name)

    if review:
        result = review.get("result", "undetermined")
        notes = review.get("notes", "")
    else:
        result = ""
        notes = ""

    return [workflow, task_name, has_run, has_finished, result, notes]


def main():

    # Figure out what workflows and tasks we'll be using
    if args.all:
        jobs = ["*:*"]
    else:
        jobs = args.jobs

    if args.list:
        print()
        print("# Workflow, Task, Ran, Finished, Review, Notes")

    summary_rows = []

    # Could be implemented more efficiently, but the performance gain would be inconsequential
    for workflow in WORKFLOWS:
        for task_spec in sorted(gym.registry.values(), key=lambda t: t.id):
            job_id = f"{workflow}:{task_spec.id}"

            # We've excluded this job
            if any(fnmatch.fnmatch(job_id, e) for e in args.exclude):
                continue

            # None of our jobs match this pair
            if not np.any(np.array([fnmatch.fnmatch(job_id, job) for job in jobs])):
                continue

            # No config for this workflow
            if workflow + "_cfg_entry_point" not in task_spec.kwargs:
                continue

            if args.list:
                summary_rows.append(get_job_summary_row(workflow, task_spec.id))
                continue

            # Training reviewing and publishing
            if args.train:
                train_job(workflow, task_spec.id, args.headless, args.force, args.num_envs)

            if args.review:
                review_pretrained_checkpoint(workflow, task_spec.id, args.force_review, args.num_envs)

            if args.publish_checkpoint:
                publish_pretrained_checkpoint(workflow, task_spec.id, args.force_publish)

    if args.list:
        writer = csv.writer(sys.stdout, quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerows(summary_rows)


if __name__ == "__main__":

    try:
        # Run the main function
        main()
    except Exception as e:
        raise e
    finally:
        # Close the app
        simulation_app.close()
