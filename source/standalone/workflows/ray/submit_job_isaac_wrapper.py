import argparse
import logging
import os
import re
import subprocess
import shlex
import time
import ray
from ray import job_submission

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Path to the cluster spec file
CLUSTER_SPEC_FILE = os.path.expanduser("~/.cluster_spec")


def read_cluster_spec(file_path):
    """Reads the cluster spec file and returns a dictionary with the cluster details."""
    cluster_specs = {}
    file_path = os.path.expanduser(file_path)

    if not os.path.exists(file_path):
        logging.error(f"Cluster spec file not found at {file_path}. Please ensure it exists.")
        return cluster_specs

    with open(file_path) as file:
        for line in file:
            # Expected format: name: CLUSTER_NAME address: ADDRESS num_cpu: NUM_CPU num_gpu: NUM_GPU ram_gb: GB_RAM
            match = re.match(r'name: (\S+) address: (\S+) num_cpu: (\S+) num_gpu: (\S+) ram_gb: (\S+)', line.strip())
            if match:
                cluster_name, address, num_cpu, num_gpu, ram_gb = match.groups()
                cluster_specs[cluster_name] = {
                    "address": address,
                    "num_cpu": int(float(num_cpu)),
                    "num_gpu": int(float(num_gpu)),
                    "ram_gb": float(ram_gb)
                }
            else:
                logging.warning(f"Could not parse line in cluster spec file: {line.strip()}")

    return cluster_specs


# Function to submit a job using Ray's job submission API
def submit_job(job, job_args, address, working_dir, executable, num_gpus, aws_s3_bucket=None, gcs_bucket=None, let_subscript_grab_logs=False):
    """Submit a job to a Ray cluster using Ray's job submission API."""
    job_submission_client = job_submission.JobSubmissionClient(f"http://{address.split(':')[0]}:8265")

    entrypoint = f"{executable} {job} {' '.join(job_args)}"

    logging.info(f"Submitting job: {entrypoint} to cluster at address: {address} with {num_gpus} GPUs")

    job_id = job_submission_client.submit_job(
        entrypoint=entrypoint,
        runtime_env={"working_dir": working_dir},
    )

    if aws_s3_bucket:
        logging.info(f"Logs will be uploaded to AWS S3 bucket: {aws_s3_bucket}")
    elif gcs_bucket:
        logging.info(f"Logs will be uploaded to Google Cloud Storage bucket: {gcs_bucket}")
    elif let_subscript_grab_logs:
        logging.info(f"Subscript will handle logs as per '--let-subscript-grab-logs'.")
    else:
        logging.info("Logs will be handled locally.")

    dashboard_address = f"http://{address.split(':')[0]}:8265"
    logging.info(f"Job '{job_id}' submitted successfully. Ray dashboard: {dashboard_address}")


def parse_arguments():
    """Parse command-line arguments for job submission."""
    parser = argparse.ArgumentParser(description="Run a script with allocated GPUs or test GPU availability.")
    parser.add_argument("script_and_args", type=str, help="Script and arguments within quotes to run")
    parser.add_argument("--gpus_per_job", type=int, default=1, help="Number of GPUs to allocate per job")
    parser.add_argument("--test_gpu", action="store_true", help="Test GPU availability instead of running the script")
    parser.add_argument(
        "--working-dir",
        default="source/standalone/workflows/ray",
        help="Working directory for the Ray job",
    )
    parser.add_argument(
        "--executable",
        default="/workspace/isaaclab/isaaclab.sh -p",
        help="Executable to run the jobs",
    )
    parser.add_argument("--aws_s3_bucket", type=str, help="AWS S3 bucket for log upload", default=None)
    parser.add_argument("--gcs_bucket", type=str, help="Google Cloud Storage bucket for log upload", default=None)
    parser.add_argument("--let-subscript-grab-logs", action="store_true", help="Let the subscript handle logs.")
    return parser.parse_args()


def main():
    """Main function to handle reading Ray addresses and submitting jobs."""
    args = parse_arguments()

    # Read cluster specifications
    cluster_specs = read_cluster_spec(CLUSTER_SPEC_FILE)

    if not cluster_specs:
        logging.error("No clusters found in the cluster spec file. Exiting.")
        return

    # Submit the job to all clusters, ensuring enough GPUs are available
    for cluster_name, spec in cluster_specs.items():
        available_gpus = spec["num_gpu"]
        if available_gpus >= args.gpus_per_job:
            submit_job(
                args.script_and_args, [], spec["address"], args.working_dir, args.executable,
                num_gpus=args.gpus_per_job, aws_s3_bucket=args.aws_s3_bucket, gcs_bucket=args.gcs_bucket,
                let_subscript_grab_logs=args.let_subscript_grab_logs
            )
        else:
            logging.warning(f"Cluster {cluster_name} does not have enough GPUs. Required: {args.gpus_per_job}, Available: {available_gpus}")


if __name__ == "__main__":
    main()

