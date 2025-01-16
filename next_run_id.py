import glob
import os
import sys

def get_latest_run_id(log_path: str, env_name) -> int:
    """
    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.

    :param log_path: path to log folder
    :param env_name:
    :return: latest run number
    """
    max_run_id = 0
    for path in glob.glob(os.path.join(log_path, env_name + "_[0-9]*")):
        run_id = path.split("_")[-1]
        path_without_run_id = path[: -len(run_id) - 1]
        if path_without_run_id.endswith(env_name) and run_id.isdigit() and int(run_id) > max_run_id:
            max_run_id = int(run_id)
    return max_run_id


if len(sys.argv) < 3:
    print("Usage: python next_run_id.py <algo> <env>")
    sys.exit(1)

algo = sys.argv[1]
env = sys.argv[2]
log_path = f"logs/rsl_rl/{algo}/"

print(get_latest_run_id(log_path, env) + 1)