# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import logging
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from torch.utils.tensorboard import SummaryWriter

import mlflow
from mlflow.tracking import MlflowClient


def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def get_existing_runs(download_dir: str) -> set[str]:
    """Get set of run IDs that have already been downloaded."""
    existing_runs = set()
    tensorboard_dir = os.path.join(download_dir, "tensorboard")
    if os.path.exists(tensorboard_dir):
        for entry in os.listdir(tensorboard_dir):
            if entry.startswith("run_"):
                existing_runs.add(entry[4:])
    return existing_runs


def process_run(args):
    """Convert MLflow run to TensorBoard format."""
    run_id, download_dir, tracking_uri = args

    try:
        # Set up MLflow client
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()
        run = client.get_run(run_id)

        # Create TensorBoard writer
        tensorboard_log_dir = os.path.join(download_dir, "tensorboard", f"run_{run_id}")
        writer = SummaryWriter(log_dir=tensorboard_log_dir)

        # Log parameters
        for key, value in run.data.params.items():
            writer.add_text(f"params/{key}", str(value))

        # Log metrics with history
        for key in run.data.metrics.keys():
            history = client.get_metric_history(run_id, key)
            for m in history:
                writer.add_scalar(f"metrics/{key}", m.value, m.step)

        # Log tags
        for key, value in run.data.tags.items():
            writer.add_text(f"tags/{key}", str(value))

        writer.close()
        return run_id, True
    except Exception:
        return run_id, False


def download_experiment_tensorboard_logs(uri: str, experiment_name: str, download_dir: str) -> None:
    """Download MLflow experiment logs and convert to TensorBoard format."""
    logger = logging.getLogger(__name__)

    try:
        # Set up MLflow
        mlflow.set_tracking_uri(uri)
        logger.info(f"Connected to MLflow tracking server at {uri}")

        # Get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found at URI '{uri}'.")

        # Get all runs
        runs = mlflow.search_runs([experiment.experiment_id])
        logger.info(f"Found {len(runs)} total runs in experiment '{experiment_name}'")

        # Check existing runs
        existing_runs = get_existing_runs(download_dir)
        logger.info(f"Found {len(existing_runs)} existing runs in {download_dir}")

        # Create directory structure
        os.makedirs(os.path.join(download_dir, "tensorboard"), exist_ok=True)

        # Process new runs
        new_run_ids = [run.run_id for _, run in runs.iterrows() if run.run_id not in existing_runs]

        if not new_run_ids:
            logger.info("No new runs to process")
            return

        logger.info(f"Processing {len(new_run_ids)} new runs...")

        # Process runs in parallel
        num_processes = min(mp.cpu_count(), len(new_run_ids))
        processed = 0

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            future_to_run = {
                executor.submit(process_run, (run_id, download_dir, uri)): run_id for run_id in new_run_ids
            }

            for future in as_completed(future_to_run):
                run_id = future_to_run[future]
                try:
                    run_id, success = future.result()
                    processed += 1
                    if success:
                        logger.info(f"[{processed}/{len(new_run_ids)}] Successfully processed run {run_id}")
                    else:
                        logger.error(f"[{processed}/{len(new_run_ids)}] Failed to process run {run_id}")
                except Exception as e:
                    logger.error(f"Error processing run {run_id}: {e}")

        logger.info(f"\nAll data saved to {download_dir}/tensorboard")

    except Exception as e:
        logger.error(f"Error during download: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Download MLflow experiment logs for TensorBoard visualization.")
    parser.add_argument("--uri", required=True, help="The MLflow tracking URI (e.g., http://localhost:5000)")
    parser.add_argument("--experiment-name", required=True, help="Name of the experiment to download")
    parser.add_argument("--download-dir", required=True, help="Directory to save TensorBoard logs")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)

    try:
        download_experiment_tensorboard_logs(args.uri, args.experiment_name, args.download_dir)
        print("\nSuccess! To view the logs, run:")
        print(f"tensorboard --logdir {os.path.join(args.download_dir, 'tensorboard')}")
    except Exception as e:
        logging.error(f"Failed to download experiment logs: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
