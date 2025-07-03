import yaml
import ray
import sys
import argparse
import subprocess
import threading
from enum import Enum

"""
This script dispatches one or more user-defined Python tasks to workers in a Ray cluster.
Each task, with its resource requirements and execution parameters, is described in a YAML configuration file.
You may specify the desired number of CPUs, GPUs, and memory allocation for each task in the config file.

Key features:
- Flexible resource management per task via config fields (`num_gpus`, `num_cpus`, `memory`).
- Real-time output streaming (stdout/stderr) for each task.
- Parallel execution of multiple tasks across cluster resources.

Tasks are distributed and scheduled according to Ray’s built-in resource manager.

Typical usage:
---------------

.. code-block:: bash

    # Print help and argument details:
    python task_runner.py -h

    # Submit tasks defined in a YAML file to the Ray cluster (auto-detects Ray head address):
    python task_runner.py --task_cfg /path/to/tasks.yaml

YAML configuration example:
---------------------------
.. code-block:: yaml
    pip: ["xxx"]
    py_modules: ["my_package/my_package"]
    tasks:
      - name: "task1"
        py_args: "-m torch.distributed.run --nnodes=1 ..."
        num_gpus: 2
        num_cpus: 10
        memory: 10737418240
      - name: "task2"
        py_args: "script.py --option arg"
        num_gpus: 0
        num_cpus: 1
        memory: 10*1024*1024*1024

- `pip`: List of pip packages to install.
- `py_args`: Arguments passed to the Python executable for this task.
- `num_gpus`, `num_cpus`: Number of GPUs/CPUs to allocate. Can be integer or a string like `"2*2"`.
- `memory`: Amount of memory (bytes) to allocate. Can be integer or a string like `"10*1024*1024*1024"`.

To stop all tasks early, press Ctrl+C; the script will cancel all running Ray tasks.
"""

class OutputType(str, Enum):
    STDOUT = "stdout"
    STDERR = "stderr"

def parse_args():
    parser = argparse.ArgumentParser(description="Run tasks from a YAML config file.")
    parser.add_argument("--task_cfg", type=str, required=True, help="Path to the YAML task file.")
    parser.add_argument("--ray_address", type=str, default="auto", help="the Ray address.")
    return parser.parse_args()
    
@ray.remote
def task_wrapper(task):
    task_name = task["name"]
    task_py_args = task["py_args"]

    # build command
    cmd = [sys.executable, *task_py_args.split()]
    print(f"[INFO]: {task_name} run: {' '.join(cmd)}")
    def handle_stream(stream, output_type):
        for line in iter(stream.readline, ''):
            stripped_line = line.rstrip('\n')
            if output_type == OutputType.STDOUT:
                print(stripped_line)
            elif output_type == OutputType.STDERR:
                print(stripped_line, file=sys.stderr)
        stream.close()
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # None for best performance and 1 for realtime output
        )

        # start tow threads to read stdout and stderr
        stdout_thread = threading.Thread(
            target=handle_stream, args=(process.stdout, OutputType.STDOUT)
        )
        stderr_thread = threading.Thread(
            target=handle_stream, args=(process.stderr, OutputType.STDERR)
        )
        stdout_thread.start()
        stderr_thread.start()
        # wait for process to finish
        process.wait()
        # wait for threads to finish
        stdout_thread.join()
        stderr_thread.join()

        returncode = process.returncode
    except Exception as e:
        print(f"[ERROR]: error while running task {task_name}: {str(e)}" )
        raise e

    print(f"[INFO]: task {task_name} finished with return code {returncode}")
    return True


def submit_tasks(ray_address,pip,py_modules,tasks):
    if not tasks:
        print("[WARNING]: no tasks to submit")
        return

    if not ray.is_initialized():
        try:
            ray.init(address=ray_address, log_to_driver=True, runtime_env={
                "pip": pip,
                "py_modules": py_modules,
            })
        except Exception as e:
            raise RuntimeError(f"initialize ray failed: {str(e)}")
    task_results = []
    for  task in tasks:
        num_gpus = eval(task["num_gpus"]) if isinstance(task["num_gpus"], str) else task["num_gpus"]
        num_cpus = eval(task["num_cpus"]) if isinstance(task["num_cpus"], str) else task["num_cpus"]
        memory = eval(task["memory"]) if isinstance(task["memory"], str) else task["memory"]
        print(f"[INFO]: submitting task {task['name']} with num_gpus={num_gpus}, num_cpus={num_cpus}, memory={memory}")
        task_results.append(task_wrapper.options(
            num_gpus=num_gpus,
            num_cpus=num_cpus,
            memory=memory,
        ).remote(task))
    
    try:
        results = ray.get(task_results)
        for i, _ in enumerate(results):
            print(f"[INFO]: Task {tasks[i]['name']} finished")
        print("[INFO]: all tasks completed.")
    except KeyboardInterrupt:
        print("[INFO]: dealing with keyboard interrupt")
        for future in task_results:
            ray.cancel(future,force=True)
        print("[INFO]: all tasks cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR]: error while running tasks: {str(e)}")
        raise e


def main():
    args = parse_args()
    try:
        with open(args.task_cfg, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise SystemExit(f"error while loading task config: {str(e)}")
    tasks = config["tasks"]
    py_modules = config.get("py_modules",None)
    pip = config.get("pip",None)
    submit_tasks(
            ray_address=args.ray_address,
            pip=pip,
            py_modules=py_modules,
            tasks=tasks,
        )

if __name__ == "__main__":
    main()

