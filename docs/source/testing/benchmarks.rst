.. _testing_benchmarks:

Benchmarking Framework
======================

Isaac Lab provides a comprehensive benchmarking framework for measuring the performance
of simulations, training workflows, and system resources. The framework is designed to
work without depending on Isaac Sim's benchmark services, enabling standalone benchmarking
with pluggable output formatters.

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
   ┌───────┐  ┌─────────┐  ┌────────────┐
   │Phases │  │Recorders│  │ Formatters │
   └───────┘  └─────────┘  └────────────┘

**Key Components:**

- **BaseIsaacLabBenchmark**: Main class for orchestrating benchmark execution
- **Measurements**: Data classes for recording metrics (timing, counts, statistics)
- **Metadata**: Data classes for recording context (hardware, versions, parameters)
- **TestPhase**: Container for organizing measurements into logical groups
- **Recorders**: System information collectors (CPU, GPU, memory, versions)
- **Formatters**: Output formatters (JSON, Osmo, OmniPerf, Summary, Schema)

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
   benchmark._finalize_impl()

Running Benchmark Scripts
-------------------------

Isaac Lab provides unified ``runtime.py``, ``startup.py``, ``training.py``, and ``play.py``
entry points under ``scripts/benchmarks/``. They default to ``--benchmark_formatter schema``, which
emits a schema-v1 JSON bundle via :mod:`isaaclab.test.benchmark`.
``--benchmark_formatter`` accepts a comma-separated list (e.g.
``schema,omniperf``) to emit several formats in a single run. Each selected
formatter writes timestamped output; the Osmo formatter writes one
phase-suffixed JSON file per phase.

The examples below use ``uv run isaaclab benchmark``. From an existing
Isaac Lab environment, run the same workflows directly instead:

* Runtime: ``./isaaclab.sh -p scripts/benchmarks/runtime.py <arguments>``
* Startup: ``./isaaclab.sh -p scripts/benchmarks/startup.py <arguments>``
* Training: ``./isaaclab.sh -p scripts/benchmarks/training.py <arguments>``
* Play: ``./isaaclab.sh -p scripts/benchmarks/play.py <arguments>``

Non-RL / Runtime Benchmarks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Measure environment stepping performance without any RL library:

.. code-block:: bash

   uv run isaaclab benchmark runtime \
       --task Isaac-Cartpole \
       --num_envs 4096 \
       --num_frames 1000 \
       --warmup_frames 50 \
       --benchmark_formatter json \
       --output_path ./results

RL Training Benchmarks
~~~~~~~~~~~~~~~~~~~~~~

Measure training performance.  Use ``--rl_library`` to select the RL library
(``rsl_rl``, ``rl_games``, ``skrl``, or ``sb3``):

.. code-block:: bash

   # Benchmark with RSL-RL
   uv run isaaclab benchmark training \
       --rl_library rsl_rl \
       --task Isaac-Cartpole \
       --num_envs 4096 \
       --max_iterations 500 \
       --benchmark_formatter json \
       --output_path ./results

   # Benchmark with RL Games
   uv run isaaclab benchmark training \
       --rl_library rl_games \
       --task Isaac-Cartpole \
       --num_envs 4096 \
       --max_iterations 500 \
       --benchmark_formatter json \
       --output_path ./results

RL Play Benchmarks
~~~~~~~~~~~~~~~~~~

Load a trained checkpoint and benchmark policy inference (the *play* workflow).
The same ``--rl_library`` dispatch selects the RL library (``rsl_rl``, ``rl_games``,
``skrl``, or ``sb3``).  In addition to the inference throughput, the emitted
``PlayBundle`` reports the rolled-out policy's reward, episode length, and success
rate.  The checkpoints consumed here are produced by ``training.py``.

.. code-block:: bash

   # Benchmark inference of a trained RSL-RL policy
   uv run isaaclab benchmark play \
       --rl_library rsl_rl \
       --task Isaac-Cartpole \
       --num_envs 4096 \
       --num_frames 1000 \
       --checkpoint /path/to/model.pt \
       --benchmark_formatter json \
       --output_path ./results

The checkpoint is resolved in the following order:

#. ``--checkpoint`` — a local filesystem path or a Nucleus URI.
#. Otherwise, the published Nucleus checkpoint for the task is downloaded
   (a warning is logged).
#. If neither is available, an error is raised.

.. note::

   ``reward``, ``ep_length``, and ``success_rate`` aggregate only **completed**
   episodes.  Set ``--num_frames`` larger than the task's episode length so at
   least one episode finishes during the rollout; otherwise these fields remain
   ``null`` (the inference throughput is still reported).

Environment-Step Timing Semantics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Runtime, training, and play benchmarks report
:class:`~isaaclab.test.benchmark.EnvironmentStepTiming`. The default
``host_return`` measurement mode records wall time until ``env.step()`` returns
without forcing queued device work to complete. It describes the host-visible
call boundary, not a device-complete boundary; asynchronously queued work can be
paid at a later synchronization point.

Pass ``--measure_sync_step`` to collect the
``serialized_synchronized`` diagnostic. It synchronizes at every measured
environment and simulation boundary, then partitions synchronized
environment-step wall time into:

* time inside nested ``SimulationContext.step()`` calls; and
* time outside those calls.

The outside-simulation remainder includes required action and actuator
processing, state updates, observations, rewards, terminations, resets,
manager execution, wrappers, and measurement synchronization. It is therefore
not an estimate of removable Isaac Lab overhead.

.. warning::

   The synchronized diagnostic serializes device work and changes the execution
   schedule it measures. Every timing and throughput field in such a run,
   including ``collection_fps``, ``total_fps``, ``iteration_time_s``, and
   ``total_wall_time_s``, is observer-perturbed. Flat formatters prefix those
   fields with ``Serialized Diagnostic``. Compare results only when
   ``measurement_mode`` is the same, and do not treat the instrumented rates as
   production throughput.
   Estimating a specific removable overhead requires a paired counterfactual
   benchmark with that feature enabled and disabled under the same workload.

For example, add the diagnostic flag to any runtime, training, or play command:

.. code-block:: bash

   uv run isaaclab benchmark runtime \
       --task Isaac-Ant-Direct \
       --num_envs 4096 \
       --measure_sync_step

PhysX Micro-Benchmarks
~~~~~~~~~~~~~~~~~~~~~~

Measure asset method and property performance using mock interfaces:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

         # Run articulation benchmarks
         uv run python source/isaaclab_physx/benchmark/assets/benchmark_articulation.py \
             --num_iterations 1000 \
             --num_instances 4096

   .. tab-item:: isaaclab.sh / isaaclab.bat

      .. code-block:: bash

         # Run articulation benchmarks
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
   uv run isaaclab benchmark startup \
       --task Isaac-Ant \
       --num_envs 4096 \
       --benchmark_formatter summary

The script profiles five phases independently:

- **app_launch**: ``launch_simulation()`` context entry (simulation runtime initialization)
- **python_imports**: importing gymnasium, torch, isaaclab_tasks, etc.
- **task_config**: ``resolve_task_config()`` (Hydra config resolution)
- **env_creation**: ``gym.make()`` + ``env.reset()`` (scene creation, sim start)
- **first_step**: a single ``env.step()`` call

Schema output records each phase wall-clock time and per-function own-time,
cumulative time, and call count. Flat formatters project the same data into
measurements. Only Isaac Lab functions and first-level calls into external
libraries are included (deep internals of torch, USD, etc. are filtered out).

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

   uv run isaaclab benchmark startup \
       --task Isaac-Ant \
       --num_envs 4096 \
       --benchmark_formatter omniperf \
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
   * - ``--benchmark_formatter``
     - ``schema``
     - Output formatter(s), comma-separated (``schema``, ``json``, ``osmo``, ``omniperf``, ``summary``)
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
   * - ``--benchmark_formatter``
     - ``schema``
     - Output formatter(s), comma-separated (``schema``, ``json``, ``osmo``, ``omniperf``, ``summary``)
   * - ``--output_path``
     - ``./``
     - Directory for output files

Non-RL / Runtime Benchmark Arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Argument
     - Default
     - Description
   * - ``--task``
     - required
     - Environment task name (e.g., ``Isaac-Cartpole``)
   * - ``--num_envs``
     - ``None`` (task config)
     - Number of parallel environments
   * - ``--num_frames``
     - ``1000``
     - Number of environment steps to measure
   * - ``--warmup_frames``
     - ``50``
     - Exact number of environment steps to exclude from timing; zero measures the first step
   * - ``--measure_sync_step``
     - ``false``
     - Collect the serialized synchronized diagnostic described above

RL Training Arguments
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Argument
     - Default
     - Description
   * - ``--rl_library``
     - required
     - RL library: ``rsl_rl``, ``rl_games``, ``skrl``, or ``sb3``
   * - ``--task``
     - required
     - Environment task name
   * - ``--num_envs``
     - ``None`` (task config)
     - Number of parallel environments
   * - ``--max_iterations``
     - ``None`` (task config)
     - Number of training iterations
   * - ``--warmup_steps``
     - ``1``
     - Number of initial environment steps to exclude before recording timing
   * - ``--measure_sync_step``
     - ``false``
     - Collect the serialized synchronized diagnostic described above

RL Play Arguments
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Argument
     - Default
     - Description
   * - ``--task``
     - required
     - Environment task name
   * - ``--rl_library``
     - required
     - RL library that produced the checkpoint: ``rsl_rl``, ``rl_games``, ``skrl``, or ``sb3``
   * - ``--num_envs``
     - ``None`` (task config)
     - Number of parallel environments
   * - ``--num_frames``
     - ``100``
     - Number of measured inference steps
   * - ``--warmup_frames``
     - ``1``
     - Number of preceding environment steps to exclude from timing and throughput
   * - ``--checkpoint``
     - ``None`` (published Nucleus checkpoint)
     - Local path or Nucleus URI of the checkpoint to roll out
   * - ``--benchmark_formatter``
     - ``schema``
     - Output formatter(s), comma-separated (``schema``, ``json``, ``osmo``, ``omniperf``, ``summary``)
   * - ``--measure_sync_step``
     - ``false``
     - Collect the serialized synchronized diagnostic described above

Runtime and play execute ``warmup_frames + num_frames`` environment steps. Thus,
the requested ``num_frames`` is always the exact number of measured steps. Play
warmup frames are excluded only from timing and throughput; they still contribute
to reward, episode-length, success-rate, and resource measurements.

Runtime warmup frames are excluded from steady-state timing, throughput, and
synchronized environment-step measurements. When warmup is nonzero, the first
warmup frame ordinary wall-clock time is retained separately as the ``first_step`` startup diagnostic.
With zero warmup, the first measured frame supplies that diagnostic.

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

Output Formatters
-----------------

JSON Formatter
~~~~~~~~~~~~~~

Full output with all phases, measurements, and metadata:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

         uv run python ... --benchmark_formatter json --output_path ./results

   .. tab-item:: isaaclab.sh / isaaclab.bat

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

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

         uv run python ... --benchmark_formatter osmo --output_path ./results

   .. tab-item:: isaaclab.sh / isaaclab.bat

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

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

         uv run python ... --benchmark_formatter omniperf --output_path ./results

   .. tab-item:: isaaclab.sh / isaaclab.bat

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
:meth:`~isaaclab.test.benchmark.BaseIsaacLabBenchmark.attach_bundle`. Use it
with a ``RuntimeBundle``, ``TrainingBundle``, or ``StartupBundle`` when a
typed, stable output contract is required.

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

         uv run python ... --benchmark_formatter schema --output_path ./results

   .. tab-item:: isaaclab.sh / isaaclab.bat

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

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

         uv run python ... --benchmark_formatter summary --output_path ./results

   .. tab-item:: isaaclab.sh / isaaclab.bat

      .. code-block:: bash

         ./isaaclab.sh -p ... --benchmark_formatter summary --output_path ./results

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
       formatter_type="json",
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

The benchmark entry points under ``scripts/benchmarks/`` are designed for CI/CD integration:

.. code-block:: bash

   # GitHub Actions / GitLab CI example
   - name: Run Runtime Benchmark
     run: |
       uv run isaaclab benchmark runtime \
           --task Isaac-Cartpole --num_envs 4096 --num_frames 1000 \
           --benchmark_formatter json --output_path ./benchmark_results

   - name: Run Training Benchmark
     run: |
       uv run isaaclab benchmark training \
           --rl_library rsl_rl --task Isaac-Cartpole --num_envs 4096 \
           --max_iterations 500 --benchmark_formatter json \
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

Troubleshooting
---------------

Import Errors
~~~~~~~~~~~~~

Ensure you're running through the Isaac Lab launcher:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

         uv run python your_benchmark.py

   .. tab-item:: isaaclab.sh / isaaclab.bat

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

Formatter Not Recognized
~~~~~~~~~~~~~~~~~~~~~~~~

Valid formatter types are: ``schema``, ``json``, ``osmo``, ``omniperf``, or ``summary``

.. code-block:: bash

   # Correct
   --benchmark_formatter json

   # Incorrect
   --benchmark_formatter JSON  # Case sensitive
