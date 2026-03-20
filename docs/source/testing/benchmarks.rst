.. _testing_benchmarks:

Benchmarking Framework
======================

Isaac Lab provides a comprehensive benchmarking framework for measuring the performance
of simulations, training workflows, and system resources. The framework is designed to
work without depending on Isaac Sim's benchmark services, enabling standalone benchmarking
with pluggable output backends.

Overview
--------

The benchmarking framework consists of several key components:

.. code-block:: text

   ┌─────────────────────────────────┐
   │    BaseIsaacLabBenchmark        │
   │    (benchmark_core.py)          │
   └───────────────┬─────────────────┘
                   │
       ┌───────────┼───────────┐
       │           │           │
       ▼           ▼           ▼
   ┌───────┐  ┌─────────┐  ┌──────────┐
   │Phases │  │Recorders│  │ Backends │
   └───────┘  └─────────┘  └──────────┘

**Key Components:**

- **BaseIsaacLabBenchmark**: Main class for orchestrating benchmark execution
- **Measurements**: Data classes for recording metrics (timing, counts, statistics)
- **Metadata**: Data classes for recording context (hardware, versions, parameters)
- **TestPhase**: Container for organizing measurements into logical groups
- **Recorders**: System information collectors (CPU, GPU, memory, versions)
- **Backends**: Output formatters (JSON, Osmo, OmniPerf, Summary)

.. seealso::

   For method-level micro-benchmarks that measure asset setter/writer and property
   performance using mock interfaces (without running full simulations), see
   :ref:`testing_micro_benchmarks`.

Quick Start
-----------

Basic usage with :class:`~isaaclab.test.benchmark.BaseIsaacLabBenchmark`:

.. code-block:: python

   from isaaclab.test.benchmark import (
       BaseIsaacLabBenchmark,
       SingleMeasurement,
       StatisticalMeasurement,
       StringMetadata,
   )

   # Initialize benchmark
   benchmark = BaseIsaacLabBenchmark(
       benchmark_name="MyBenchmark",
       backend_type="json",
       output_path="./results",
   )

   # Record measurements
   benchmark.add_measurement(
       phase_name="simulation",
       measurement=SingleMeasurement(
           name="fps",
           value=1234.5,
           unit="frames/sec"
       ),
   )

   benchmark.add_measurement(
       phase_name="simulation",
       measurement=StatisticalMeasurement(
           name="step_time",
           mean=0.82,
           std=0.05,
           n=1000,
           unit="ms"
       ),
   )

   # Add metadata
   benchmark.add_measurement(
       phase_name="simulation",
       metadata=StringMetadata(name="task", data="Isaac-Cartpole-v0"),
   )

   # Finalize and write output
   benchmark._finalize_impl()

Running Benchmark Scripts
-------------------------

Isaac Lab provides shell scripts for running benchmark suites:

Non-RL Benchmarks
~~~~~~~~~~~~~~~~~

Measure environment stepping performance without training:

.. code-block:: bash

   # Run all non-RL benchmarks
   ./scripts/benchmarks/run_non_rl_benchmarks.sh ./output_dir

   # Run a single benchmark manually
   ./isaaclab.sh -p scripts/benchmarks/benchmark_non_rl.py \
       --task Isaac-Cartpole-v0 \
       --num_envs 4096 \
       --num_frames 100 \
       --headless \
       --benchmark_backend json \
       --output_path ./results

RL Training Benchmarks
~~~~~~~~~~~~~~~~~~~~~~

Measure training performance with RSL-RL:

.. code-block:: bash

   # Run training benchmarks
   ./scripts/benchmarks/run_training_benchmarks.sh ./output_dir

   # Run manually with RSL-RL
   ./isaaclab.sh -p scripts/benchmarks/benchmark_rsl_rl.py \
       --task Isaac-Cartpole-v0 \
       --num_envs 4096 \
       --max_iterations 500 \
       --headless \
       --benchmark_backend json \
       --output_path ./results

PhysX Micro-Benchmarks
~~~~~~~~~~~~~~~~~~~~~~

Measure asset method and property performance using mock interfaces:

.. code-block:: bash

   # Run PhysX micro-benchmarks
   ./scripts/benchmarks/run_physx_benchmarks.sh ./output_dir

   # Run articulation benchmarks manually
   ./isaaclab.sh -p source/isaaclab_physx/benchmark/assets/benchmark_articulation.py \
       --num_iterations 1000 \
       --num_instances 4096

For detailed documentation on micro-benchmarks, including available benchmark files,
input modes, and how to add new benchmarks, see :ref:`testing_micro_benchmarks`.

Startup Profiling Benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Profile the startup sequence of an IsaacLab environment using ``cProfile``. Each
startup stage is wrapped in its own profiling session and the top functions by
own-time are reported. This is useful for investigating startup regressions and
understanding where time is spent during initialization.

.. code-block:: bash

   # Basic usage — reports top 30 functions per phase
   ./isaaclab.sh -p scripts/benchmarks/benchmark_startup.py \
       --task Isaac-Ant-v0 \
       --num_envs 4096 \
       --headless \
       --benchmark_backend summary

The script profiles four phases independently:

- **app_launch**: ``launch_simulation()`` context entry (Kit/USD/PhysX init)
- **python_imports**: importing gymnasium, torch, isaaclab_tasks, etc.
- **env_creation**: ``gym.make()`` + ``env.reset()`` (scene creation, sim start)
- **first_step**: a single ``env.step()`` call

Each phase records a wall-clock time plus per-function own-time and cumulative
time as ``SingleMeasurement`` entries. Only IsaacLab functions and first-level
calls into external libraries are included (deep internals of torch, USD, etc.
are filtered out).

**Whitelist mode** — For dashboard time-series comparisons across runs, use a
YAML whitelist config to report a fixed set of functions instead of top-N.
Patterns use ``fnmatch`` syntax (``*`` and ``?`` wildcards):

.. code-block:: yaml

   # Example whitelist config
   app_launch:
     - "isaaclab.utils.configclass:_custom_post_init"
     - "isaaclab.sim.*:__init__"
   env_creation:
     - "isaaclab.cloner.*:usd_replicate"
     - "isaaclab.cloner.*:filter_collisions"
     - "isaaclab.scene.*:_init_scene"
   first_step:
     - "isaaclab.actuators.*:compute"
     - "warp.*:launch"

.. code-block:: bash

   ./isaaclab.sh -p scripts/benchmarks/benchmark_startup.py \
       --task Isaac-Ant-v0 \
       --num_envs 4096 \
       --headless \
       --benchmark_backend omniperf \
       --whitelist_config scripts/benchmarks/startup_whitelist.yaml

Phases listed in the YAML use the whitelist; phases not listed fall back to
``--top_n`` (default: 5 in whitelist mode, 30 otherwise). Patterns that match
no profiled function emit ``0.0`` placeholders so the output always contains
the same keys.

A default whitelist is provided at ``scripts/benchmarks/startup_whitelist.yaml``.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Argument
     - Default
     - Description
   * - ``--task``
     - required
     - Environment task name
   * - ``--num_envs``
     - from config
     - Number of parallel environments
   * - ``--top_n``
     - 30 (5 with whitelist)
     - Max functions per non-whitelisted phase
   * - ``--whitelist_config``
     - None
     - Path to YAML whitelist file
   * - ``--benchmark_backend``
     - ``omniperf``
     - Output backend (``json``, ``osmo``, ``omniperf``, ``summary``)
   * - ``--output_path``
     - ``.``
     - Directory for output files

Command Line Arguments
----------------------

Common Arguments
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Argument
     - Default
     - Description
   * - ``--benchmark_backend``
     - ``json``
     - Output backend: ``json``, ``osmo``, ``omniperf``, or ``summary``
   * - ``--output_path``
     - ``./``
     - Directory for output files
   * - ``--headless``
     - ``false``
     - Run without rendering

Non-RL Benchmark Arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Argument
     - Default
     - Description
   * - ``--task``
     - required
     - Environment task name (e.g., ``Isaac-Cartpole-v0``)
   * - ``--num_envs``
     - ``4096``
     - Number of parallel environments
   * - ``--num_frames``
     - ``100``
     - Number of simulation frames to run
   * - ``--enable_cameras``
     - ``false``
     - Enable camera rendering (for RGB/depth tasks)

RL Training Arguments
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Argument
     - Default
     - Description
   * - ``--task``
     - required
     - Environment task name
   * - ``--num_envs``
     - ``4096``
     - Number of parallel environments
   * - ``--max_iterations``
     - ``500``
     - Number of training iterations

Measurement Types
-----------------

The framework provides several measurement types for different data:

SingleMeasurement
~~~~~~~~~~~~~~~~~

For single numeric values:

.. code-block:: python

   from isaaclab.test.benchmark import SingleMeasurement

   measurement = SingleMeasurement(
       name="total_frames",
       value=100000,
       unit="frames"
   )

StatisticalMeasurement
~~~~~~~~~~~~~~~~~~~~~~

For statistical summaries:

.. code-block:: python

   from isaaclab.test.benchmark import StatisticalMeasurement

   measurement = StatisticalMeasurement(
       name="step_time",
       mean=0.82,
       std=0.05,
       n=1000,
       unit="ms"
   )

BooleanMeasurement
~~~~~~~~~~~~~~~~~~

For pass/fail status:

.. code-block:: python

   from isaaclab.test.benchmark import BooleanMeasurement

   measurement = BooleanMeasurement(
       name="converged",
       bvalue=True
   )

DictMeasurement
~~~~~~~~~~~~~~~

For structured data:

.. code-block:: python

   from isaaclab.test.benchmark import DictMeasurement

   measurement = DictMeasurement(
       name="config",
       value={"learning_rate": 0.001, "batch_size": 64}
   )

ListMeasurement
~~~~~~~~~~~~~~~

For sequences of values:

.. code-block:: python

   from isaaclab.test.benchmark import ListMeasurement

   measurement = ListMeasurement(
       name="rewards_per_episode",
       value=[100.5, 102.3, 98.7, 105.1]
   )

Test Phases
-----------

:class:`~isaaclab.test.benchmark.TestPhase` organizes measurements and metadata
into logical groups. Common phases include:

- ``benchmark_info``: Workflow name, timestamp, configuration
- ``hardware_info``: CPU, GPU, memory information
- ``version_info``: Software versions (Isaac Sim, PyTorch, etc.)
- ``simulation``: Environment stepping metrics
- ``training``: RL training metrics
- ``runtime``: Execution time and resource usage

Example:

.. code-block:: python

   # Measurements are automatically grouped by phase
   benchmark.add_measurement("simulation", measurement=fps_measurement)
   benchmark.add_measurement("simulation", metadata=task_metadata)
   benchmark.add_measurement("training", measurement=reward_measurement)

Output Backends
---------------

JSON Backend
~~~~~~~~~~~~

Full output with all phases, measurements, and metadata:

.. code-block:: bash

   ./isaaclab.sh -p ... --benchmark_backend json --output_path ./results

Output structure:

.. code-block:: json

   [
     {
       "phase_name": "simulation",
       "measurements": [
         {
           "name": "MyBenchmark simulation fps",
           "value": 1234.5,
           "unit": "frames/sec",
           "type": "single"
         }
       ],
       "metadata": [
         {"name": "MyBenchmark simulation task", "data": "Isaac-Cartpole-v0", "type": "string"}
       ]
     }
   ]

Osmo Backend
~~~~~~~~~~~~

Simplified key-value format for CI/CD integration:

.. code-block:: bash

   ./isaaclab.sh -p ... --benchmark_backend osmo --output_path ./results

Output structure:

.. code-block:: json

   {
     "workflow_name": "MyBenchmark",
     "phase": "simulation",
     "fps": 1234.5,
     "task": "Isaac-Cartpole-v0"
   }

OmniPerf Backend
~~~~~~~~~~~~~~~~

Format for database upload and performance tracking:

.. code-block:: bash

   ./isaaclab.sh -p ... --benchmark_backend omniperf --output_path ./results

Output structure:

.. code-block:: json

   {
     "simulation": {
       "workflow_name": "MyBenchmark",
       "fps": 1234.5,
       "step_time_mean": 0.82,
       "step_time_std": 0.05
     }
   }

Summary Backend
~~~~~~~~~~~~~~~

Human-readable console report plus JSON file. Prints a formatted summary to the
terminal while also writing the same data as JSON. Standard phases (runtime,
startup, train, frametime, system info) are rendered with specialized formatting;
any additional phases (e.g., from the startup profiling benchmark) are rendered
automatically with their ``SingleMeasurement`` and ``StatisticalMeasurement``
entries. Use when you want a quick readout without opening the JSON:

.. code-block:: bash

   ./isaaclab.sh -p ... --benchmark_backend summary --output_path ./results

When ``summary`` is selected, frametime recorders are enabled automatically when
running with Isaac Sim (Kit).

BenchmarkMonitor
----------------

:class:`~isaaclab.test.benchmark.BenchmarkMonitor` enables continuous system
monitoring during blocking operations like RL training loops:

.. code-block:: python

   from isaaclab.test.benchmark import BaseIsaacLabBenchmark, BenchmarkMonitor

   benchmark = BaseIsaacLabBenchmark(
       benchmark_name="TrainingBenchmark",
       backend_type="json",
       output_path="./results",
   )

   # Monitor system resources during blocking training call
   with BenchmarkMonitor(benchmark, interval=1.0):
       runner.learn(num_learning_iterations=1000)  # Blocking call

   benchmark._finalize_impl()

The monitor runs in a background thread and periodically calls
``update_manual_recorders()`` to capture CPU, GPU, and memory usage samples.

System Recorders
----------------

The framework includes built-in recorders for system information:

CPUInfoRecorder
~~~~~~~~~~~~~~~

Captures CPU model, core count, and usage statistics.

GPUInfoRecorder
~~~~~~~~~~~~~~~

Captures GPU model, memory, and utilization via ``nvidia-smi``.

MemoryInfoRecorder
~~~~~~~~~~~~~~~~~~

Captures system and GPU memory usage over time.

VersionInfoRecorder
~~~~~~~~~~~~~~~~~~~

Captures software versions:

- Isaac Sim version
- Isaac Lab version
- PyTorch version
- CUDA version
- Python version

Creating Custom Benchmarks
--------------------------

Step 1: Initialize Benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import argparse
   from isaaclab.test.benchmark import BaseIsaacLabBenchmark

   parser = argparse.ArgumentParser()
   parser.add_argument("--benchmark_backend", default="json")
   parser.add_argument("--output_path", default="./")
   args = parser.parse_args()

   benchmark = BaseIsaacLabBenchmark(
       benchmark_name="CustomBenchmark",
       backend_type=args.benchmark_backend,
       output_path=args.output_path,
   )

Step 2: Run Your Workload
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time

   start_time = time.time()

   # Your workload here
   for i in range(num_iterations):
       env.step(actions)

   elapsed = time.time() - start_time

Step 3: Record Measurements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from isaaclab.test.benchmark import SingleMeasurement, StringMetadata

   benchmark.add_measurement(
       phase_name="runtime",
       measurement=SingleMeasurement(
           name="total_time",
           value=elapsed,
           unit="seconds"
       ),
   )

   benchmark.add_measurement(
       phase_name="runtime",
       metadata=StringMetadata(name="num_iterations", data=str(num_iterations)),
   )

Step 4: Finalize
~~~~~~~~~~~~~~~~

.. code-block:: python

   benchmark._finalize_impl()

Integration with CI/CD
----------------------

The shell scripts in ``scripts/benchmarks/`` are designed for CI/CD integration:

.. code-block:: bash

   # GitHub Actions / GitLab CI example
   - name: Run Benchmarks
     run: |
       ./scripts/benchmarks/run_non_rl_benchmarks.sh ./benchmark_results

   - name: Upload Results
     uses: actions/upload-artifact@v3
     with:
       name: benchmark-results
       path: ./benchmark_results/

For Osmo integration, use the ``osmo`` backend:

.. code-block:: bash

   ./scripts/benchmarks/run_non_rl_benchmarks.sh ./results
   # Results are in Osmo-compatible JSON format

Troubleshooting
---------------

Import Errors
~~~~~~~~~~~~~

Ensure you're running through the Isaac Lab launcher:

.. code-block:: bash

   ./isaaclab.sh -p your_benchmark.py

Not:

.. code-block:: bash

   python your_benchmark.py  # Missing environment setup

Missing GPU Metrics
~~~~~~~~~~~~~~~~~~~

Verify ``nvidia-smi`` is available and CUDA is configured:

.. code-block:: bash

   nvidia-smi  # Should show GPU info

Empty Output Files
~~~~~~~~~~~~~~~~~~

Ensure ``_finalize_impl()`` is called before the script exits:

.. code-block:: python

   try:
       # Your benchmark code
       pass
   finally:
       benchmark._finalize_impl()

Backend Not Recognized
~~~~~~~~~~~~~~~~~~~~~~

Valid backend types are: ``json``, ``osmo``, ``omniperf``, ``summary``

.. code-block:: bash

   # Correct
   --benchmark_backend json

   # Incorrect
   --benchmark_backend JSON  # Case sensitive
