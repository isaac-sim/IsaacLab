.. _testing_benchmark_framework:

Benchmark Framework API
=======================

This advanced reference explains how to call benchmark workflows from Python,
record custom measurements, select output formatters, and extend the benchmark
framework. To run and interpret the supported CLI workflows, see
:ref:`testing_benchmarks`.

Use The Python API
------------------

The canonical runtime, startup, training, and play workflows live in
:mod:`isaaclab.benchmark`.

Programmatic dispatch uses typed requests and returns the result bundle together
with every formatter output path:

.. code-block:: python

   from isaaclab.benchmark import (
       BenchmarkLauncherConfig,
       BenchmarkOutputConfig,
       BenchmarkTrainingRequest,
       run_benchmark,
   )

   result = run_benchmark(
       BenchmarkTrainingRequest(
           backend="rsl_rl",
           task="Isaac-Cartpole-Direct",
           num_envs=4096,
           max_iterations=500,
           presets=("newton_mjwarp",),
           output=BenchmarkOutputConfig(formatters=("schema", "summary")),
           launcher=BenchmarkLauncherConfig(visualizers=()),
       )
   )

   print(result.bundle.runtime.total_fps.mean)
   print(result.output_paths)

See the full :mod:`isaaclab.benchmark` API reference for all supported
requests, result bundles, and framework types.

Benchmark Core
--------------

Basic usage with :class:`~isaaclab.benchmark.BaseIsaacLabBenchmark`:

.. code-block:: python

   from isaaclab.benchmark import (
       BaseIsaacLabBenchmark,
       SingleMeasurement,
       StatisticalMeasurement,
       StringMetadata,
   )

   # Initialize benchmark
   benchmark = BaseIsaacLabBenchmark(
       benchmark_name="MyBenchmark",
       formatter_type="json",
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
       metadata=StringMetadata(name="task", data="Isaac-Cartpole"),
   )

   # Finalize and write output
   benchmark.finalize()

Measurement Types
-----------------

The framework provides several measurement types for different data:

SingleMeasurement
~~~~~~~~~~~~~~~~~

For single numeric values:

.. code-block:: python

   from isaaclab.benchmark import SingleMeasurement

   measurement = SingleMeasurement(
       name="total_frames",
       value=100000,
       unit="frames"
   )

StatisticalMeasurement
~~~~~~~~~~~~~~~~~~~~~~

For statistical summaries:

.. code-block:: python

   from isaaclab.benchmark import StatisticalMeasurement

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

   from isaaclab.benchmark import BooleanMeasurement

   measurement = BooleanMeasurement(
       name="converged",
       bvalue=True
   )

DictMeasurement
~~~~~~~~~~~~~~~

For structured data:

.. code-block:: python

   from isaaclab.benchmark import DictMeasurement

   measurement = DictMeasurement(
       name="config",
       value={"learning_rate": 0.001, "batch_size": 64}
   )

ListMeasurement
~~~~~~~~~~~~~~~

For sequences of values:

.. code-block:: python

   from isaaclab.benchmark import ListMeasurement

   measurement = ListMeasurement(
       name="rewards_per_episode",
       value=[100.5, 102.3, 98.7, 105.1]
   )

Test Phases
-----------

:class:`~isaaclab.benchmark.TestPhase` organizes measurements and metadata
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

Output Formatters
-----------------

JSON Formatter
~~~~~~~~~~~~~~

Full output with all phases, measurements, and metadata:

.. code-block:: bash

   ./isaaclab.sh -p ... --benchmark_formatter json --output_path ./results

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
         {"name": "MyBenchmark simulation task", "data": "Isaac-Cartpole", "type": "string"}
       ]
     }
   ]

Osmo Formatter
~~~~~~~~~~~~~~

Simplified key-value format for CI/CD integration:

.. code-block:: bash

   ./isaaclab.sh -p ... --benchmark_formatter osmo --output_path ./results

Output structure:

.. code-block:: json

   {
     "workflow_name": "MyBenchmark",
     "phase": "simulation",
     "fps": 1234.5,
     "task": "Isaac-Cartpole"
   }

OmniPerf Formatter
~~~~~~~~~~~~~~~~~~

Format for database upload and performance tracking:

.. code-block:: bash

   ./isaaclab.sh -p ... --benchmark_formatter omniperf --output_path ./results

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

Schema Formatter
~~~~~~~~~~~~~~~~

Writes a schema-v1 bundle attached with
:meth:`~isaaclab.benchmark.BaseIsaacLabBenchmark.attach_bundle`. Use it
with a ``RuntimeBundle``, ``TrainingBundle``, or ``StartupBundle`` when a
typed, stable output contract is required.

.. code-block:: bash

   ./isaaclab.sh -p ... --benchmark_formatter schema --output_path ./results

Summary Formatter
~~~~~~~~~~~~~~~~~

Human-readable console report plus JSON file. Prints a formatted summary to the
terminal while also writing the same data as JSON. Standard phases (runtime,
startup, train, frametime, system info) are rendered with specialized formatting;
any additional phases (e.g., from the startup profiling benchmark) are rendered
automatically with their ``SingleMeasurement`` and ``StatisticalMeasurement``
entries. Use when you want a quick readout without opening the JSON:

.. code-block:: bash

   ./isaaclab.sh -p ... --benchmark_formatter summary --output_path ./results

When ``summary`` is selected, frametime recorders are enabled automatically when
running with Isaac Sim (Kit).

Benchmark Monitor
-----------------

:class:`~isaaclab.benchmark.BenchmarkMonitor` enables continuous system
monitoring during blocking operations like RL training loops:

.. code-block:: python

   from isaaclab.benchmark import BaseIsaacLabBenchmark, BenchmarkMonitor

   benchmark = BaseIsaacLabBenchmark(
       benchmark_name="TrainingBenchmark",
       formatter_type="json",
       output_path="./results",
   )

   # Monitor system resources during blocking training call
   with BenchmarkMonitor(benchmark, interval=1.0):
       runner.learn(num_learning_iterations=1000)  # Blocking call

   benchmark.finalize()

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

Create A Custom Benchmark
-------------------------

Step 1: Initialize Benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import argparse
   from isaaclab.benchmark import BaseIsaacLabBenchmark

   parser = argparse.ArgumentParser()
   parser.add_argument("--benchmark_formatter", default="json")
   parser.add_argument("--output_path", default="./")
   args = parser.parse_args()

   benchmark = BaseIsaacLabBenchmark(
       benchmark_name="CustomBenchmark",
       formatter_type=args.benchmark_formatter,
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

   from isaaclab.benchmark import SingleMeasurement, StringMetadata

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

   benchmark.finalize()

CI Integration
--------------

The ``isaaclab benchmark`` entrypoints are designed for CI/CD integration:

.. code-block:: bash

   # GitHub Actions / GitLab CI example
   - name: Run Runtime Benchmark
     run: |
       uv run isaaclab benchmark runtime \
           --task Isaac-Cartpole --num_envs 4096 --num_frames 1000 \
           --benchmark_formatter schema,summary --output_path ./benchmark_results

   - name: Run Training Benchmark
     run: |
       uv run isaaclab benchmark training \
           --rl_library rsl_rl --task Isaac-Cartpole --num_envs 4096 \
           --max_iterations 500 --benchmark_formatter schema,summary \
           --output_path ./benchmark_results

   - name: Upload Results
     uses: actions/upload-artifact@v3
     with:
       name: benchmark-results
       path: ./benchmark_results/

For Osmo integration, use the ``osmo`` formatter:

.. code-block:: bash

   uv run isaaclab benchmark runtime \
       --task Isaac-Cartpole --num_envs 4096 --num_frames 1000 \
       --benchmark_formatter osmo --output_path ./results
   # Results are in Osmo-compatible JSON format

Framework Troubleshooting
-------------------------

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

Ensure ``finalize()`` is called before the script exits:

.. code-block:: python

   try:
       # Your benchmark code
       pass
   finally:
       benchmark.finalize()

Formatter Not Recognized
~~~~~~~~~~~~~~~~~~~~~~~~

Valid formatter types are: ``schema``, ``json``, ``osmo``, ``omniperf``, or ``summary``

.. code-block:: bash

   # Correct
   --benchmark_formatter json

   # Incorrect
   --benchmark_formatter JSON  # Case sensitive
