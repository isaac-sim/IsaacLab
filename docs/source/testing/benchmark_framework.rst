.. _testing_benchmark_framework:

Benchmark Framework API
=======================

This advanced guide is for automation authors who need to run supported
benchmarks from Python, consume typed results, or add a benchmark producer. For
day-to-day benchmarking, start with the CLI workflows and interpretation
guidance in :ref:`testing_benchmarks`.

.. seealso::

   To isolate one asset method, cached property, or sensor update, see
   :ref:`testing_micro_benchmarks`. Those workloads use the framework described
   here, but have a different timing protocol and interpretation.

Choose The Supported API
------------------------

Use the typed workflow API whenever an existing workflow answers the question.
It applies the same launch, task, timing, schema, and output behavior as the CLI.

.. list-table::
   :header-rows: 1
   :widths: 25 37 38

   * - Question
     - Request
     - Runner
   * - Environment-step capacity
     - :class:`~isaaclab.benchmark.BenchmarkRuntimeRequest`
     - :func:`~isaaclab.benchmark.run_runtime_benchmark`
   * - Application and task startup
     - :class:`~isaaclab.benchmark.BenchmarkStartupRequest`
     - :func:`~isaaclab.benchmark.run_startup_benchmark`
   * - RL training and learning
     - :class:`~isaaclab.benchmark.BenchmarkTrainingRequest`
     - :func:`~isaaclab.benchmark.run_training_benchmark`
   * - Trained-policy playback
     - :class:`~isaaclab.benchmark.BenchmarkPlayRequest`
     - :func:`~isaaclab.benchmark.run_play_benchmark`

:func:`~isaaclab.benchmark.run_benchmark` accepts any of these request types
when generic dispatch is more convenient. Use
:class:`~isaaclab.benchmark.BaseIsaacLabBenchmark` only when implementing a
measurement workflow that the supported requests do not cover.

Run A Workflow From Python
--------------------------

The following complete runtime example measures 1000 steps after 50 warm-up
steps, disables visualizers, and requests both the stable schema and the
human-readable summary:

.. code-block:: python

   from pathlib import Path

   from isaaclab.benchmark import (
       BenchmarkLauncherConfig,
       BenchmarkOutputConfig,
       BenchmarkRuntimeRequest,
       run_runtime_benchmark,
   )

   result = run_runtime_benchmark(
       BenchmarkRuntimeRequest(
           task="Isaac-Cartpole-Direct",
           num_envs=4096,
           num_steps=1000,
           warmup_steps=50,
           seed=42,
           presets=("newton_mjwarp",),
           output=BenchmarkOutputConfig(
               path=Path("results/runtime"),
               formatters=("schema", "summary"),
           ),
           launcher=BenchmarkLauncherConfig(visualizers=()),
       )
   )

   print(f"Total FPS: {result.bundle.runtime.total_fps.mean:,.0f}")
   for output_path in result.output_paths:
       print(output_path)

Run the file through the Isaac Lab Python wrapper:

.. code-block:: bash

   ./isaaclab.sh -p runtime_benchmark.py

The terminal receives the summary report. The two paths returned in
``result.output_paths`` identify the schema and summary JSON files actually
written; do not reconstruct their timestamped names in automation.

The other workflows use the same output and launcher objects. These focused
examples show their workflow-specific inputs and results.

Startup profiling records wall time and the most expensive functions for each
phase:

.. code-block:: python

   from pathlib import Path

   from isaaclab.benchmark import (
       BenchmarkLauncherConfig,
       BenchmarkOutputConfig,
       BenchmarkStartupRequest,
       run_startup_benchmark,
   )

   result = run_startup_benchmark(
       BenchmarkStartupRequest(
           task="Isaac-Cartpole-Direct",
           num_envs=4096,
           top_n=20,
           presets=("newton_mjwarp",),
           output=BenchmarkOutputConfig(
               path=Path("results/startup"),
               formatters=("schema", "summary"),
           ),
           launcher=BenchmarkLauncherConfig(visualizers=()),
       )
   )
   print(f"Environment creation: {result.bundle.phases['env_creation'].total_time_s:.3f} s")

Training returns the checkpoint produced by the selected RL library:

.. code-block:: python

   from pathlib import Path

   from isaaclab.benchmark import (
       BenchmarkLauncherConfig,
       BenchmarkOutputConfig,
       BenchmarkTrainingRequest,
       run_training_benchmark,
   )

   result = run_training_benchmark(
       BenchmarkTrainingRequest(
           backend="rsl_rl",
           task="Isaac-Cartpole-Direct",
           num_envs=4096,
           max_iterations=500,
           warmup_steps=50,
           seed=42,
           presets=("newton_mjwarp",),
           output=BenchmarkOutputConfig(
               path=Path("results/training"),
               formatters=("schema", "summary"),
           ),
           launcher=BenchmarkLauncherConfig(visualizers=()),
       )
   )
   print(f"Collection FPS: {result.bundle.runtime.collection_fps.mean:,.0f}")
   print(f"Checkpoint: {result.bundle.checkpoint_path}")

Pass the persisted ``checkpoint_path`` to a play request in a **new process**:

.. code-block:: python

   from pathlib import Path

   from isaaclab.benchmark import (
       BenchmarkLauncherConfig,
       BenchmarkOutputConfig,
       BenchmarkPlayRequest,
       run_play_benchmark,
   )

   checkpoint = Path("checkpoint.txt").read_text().strip()
   result = run_play_benchmark(
       BenchmarkPlayRequest(
           backend="rsl_rl",
           task="Isaac-Cartpole-Direct",
           checkpoint=checkpoint,
           num_envs=4096,
           num_steps=1000,
           warmup_steps=50,
           seed=42,
           presets=("newton_mjwarp",),
           output=BenchmarkOutputConfig(
               path=Path("results/play"),
               formatters=("schema", "summary"),
           ),
           launcher=BenchmarkLauncherConfig(visualizers=()),
       )
   )
   print(f"Environment + inference FPS: {result.bundle.runtime.collection_fps.mean:,.0f}")

Persist the training checkpoint path in an orchestration artifact, configuration
store, or the training schema file before starting play. A simulator application
owns process-level state and is not designed to be repeatedly launched and
closed by several workflow calls in one Python process.

Configure Requests
------------------

All requests share the task identifier, optional environment count and seed,
task configuration, output configuration, and launcher configuration.

.. list-table:: Shared request fields
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Purpose
   * - ``task``
     - Registered Gym task identifier, for example ``Isaac-Cartpole-Direct``.
   * - ``num_envs`` and ``seed``
     - Environment count and random seed. Keep both fixed for comparisons.
   * - ``presets``
     - Typed task presets, such as ``("newton_mjwarp",)``.
   * - ``hydra_args``
     - Additional Hydra overrides, represented as a tuple of strings.
   * - ``output``
     - :class:`~isaaclab.benchmark.BenchmarkOutputConfig` controlling the output
       directory and formatter set.
   * - ``launcher``
     - :class:`~isaaclab.benchmark.BenchmarkLauncherConfig` controlling device,
       cameras, visualizers, Kit, livestream, and logging options.

Workflow requests add the following measurement controls:

.. list-table:: Workflow-specific fields
   :header-rows: 1
   :widths: 22 26 52

   * - Workflow
     - Fields
     - Meaning
   * - Runtime
     - ``num_steps``, ``warmup_steps``
     - Measured environment steps and preceding excluded steps.
   * - Startup
     - ``top_n``, ``whitelist_config``
     - Number and optional filter of profiled functions retained per phase.
   * - Training
     - ``backend``, ``max_iterations``, ``warmup_steps``
     - RL library, learning iterations, and initial environment steps excluded
       from environment-step timing.
   * - Play
     - ``backend``, ``checkpoint``, ``num_steps``, ``warmup_steps``
     - RL library, policy, measured inference steps, and initial excluded steps.

Training exposes additional learning, video, sensor capture, and convergence
options. Training and play also expose ``backend_args`` for options that truly
belong to one RL library. Prefer typed fields, presets, and ``hydra_args`` when
they express the option; ``backend_args`` bypasses the common request contract.
See the :mod:`isaaclab.benchmark` API reference for the exhaustive field list.

An empty ``visualizers`` tuple explicitly disables every visualizer. ``None``
preserves task and environment defaults. Enabling cameras, rendering,
livestreaming, deterministic behavior, or animation recording changes the
workload and must be kept identical across compared runs.

Choose Output Formats
~~~~~~~~~~~~~~~~~~~~~

Select one or more formatter names in
:class:`~isaaclab.benchmark.BenchmarkOutputConfig`:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Formatter
     - Contract
   * - ``schema``
     - Writes the stable typed bundle. Use this for programmatic comparison and
       long-lived result storage.
   * - ``summary``
     - Prints a compact terminal report and writes a flat metrics JSON file.
   * - ``json``
     - Writes the lower-level phase, measurement, and metadata representation.
   * - ``osmo``
     - Writes phase-oriented KPI files for Osmo consumers.
   * - ``omniperf``
     - Writes the OmniPerf KPI representation.

Use ``("schema", "summary")`` for results intended for review or publication:
the summary gives immediate feedback while the schema preserves typed data for
analysis. With multiple formatters, each output filename includes its formatter
name so JSON outputs cannot overwrite each other.

Read Results
------------

Every workflow returns :class:`~isaaclab.benchmark.BenchmarkResult` with two
members:

* ``bundle`` is a :class:`~isaaclab.benchmark.RuntimeBundle`,
  :class:`~isaaclab.benchmark.StartupBundle`,
  :class:`~isaaclab.benchmark.TrainingBundle`, or
  :class:`~isaaclab.benchmark.PlayBundle`.
* ``output_paths`` is the exact tuple of :class:`pathlib.Path` objects written
  by the selected formatters.

Consume typed fields rather than scraping terminal text or depending on keys in
``bundle.extra``:

.. code-block:: python

   # Runtime or play
   total_fps = result.bundle.runtime.total_fps.mean
   step_fps = result.bundle.runtime.environment_step_timing.environment_step_fps.mean

   # Startup
   scene_creation_s = result.bundle.phases["env_creation"].total_time_s

   # Training
   collection_fps = result.bundle.runtime.collection_fps.mean
   final_reward = result.bundle.learning.reward.final_raw
   checkpoint = result.bundle.checkpoint_path

   # Play, when at least one episode completed
   mean_return = result.bundle.reward.mean if result.bundle.reward is not None else None

The common ``run``, ``versions``, and ``hardware`` fields carry the comparison
context. ``runtime`` separates collection throughput, total throughput, startup
time, and environment-step timing. See :ref:`testing_benchmarks` for the meaning
of those scopes and the public API reference for every schema field.

Handle Errors And Process Lifetime
----------------------------------

Typed dispatch converts a workflow parser rejection into ``ValueError``. A
workflow that exits without returning a result raises ``RuntimeError``. Invalid
output configuration, such as an empty formatter tuple or an unknown formatter,
also fails rather than silently producing an incomplete result.

Run each workflow in a separate process. This gives every benchmark a clean
simulator lifecycle, prevents state from one workflow contaminating another,
and matches the CLI execution model. Process startup is reported separately and
is not included in steady-state throughput.

.. warning::

   ``measure_synchronized_step_breakdown=True`` inserts device
   synchronizations around environment and simulation step boundaries. This
   serializes work and changes the schedule being measured, especially on
   Newton. Every rate from that run is diagnostic and must not be reported as
   throughput. Time outside ``SimulationContext.step()`` includes required
   action, actuator, state, manager, reset, wrapper, and synchronization work;
   it is not equivalent to removable Isaac Lab overhead.

Extend The Framework
--------------------

Use :class:`~isaaclab.benchmark.BaseIsaacLabBenchmark` when adding a producer
whose lifecycle cannot be represented by a supported workflow. The framework
provides phases, typed measurement and metadata records, system recorders, and
formatters. It does not define the timing boundary for you.

The main record types are:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Type
     - Use
   * - :class:`~isaaclab.benchmark.SingleMeasurement`
     - One scalar with a unit, such as total duration or completed operations.
   * - :class:`~isaaclab.benchmark.StatisticalMeasurement`
     - Mean, standard deviation, sample count, and unit for repeated samples.
   * - :class:`~isaaclab.benchmark.BooleanMeasurement`
     - A status or correctness outcome.
   * - :class:`~isaaclab.benchmark.DictMeasurement`
     - Structured measurement data that is not a scalar sample.
   * - :class:`~isaaclab.benchmark.ListMeasurement`
     - A retained sample or learning series.
   * - ``StringMetadata``, ``IntMetadata``, ``FloatMetadata``, ``DictMetadata``
     - Configuration and identity needed to compare the measurement.

Complete Custom Example
~~~~~~~~~~~~~~~~~~~~~~~

This executable example measures a CPU workload, records the sample
distribution and configuration, updates system recorders, and always finalizes
the benchmark:

.. code-block:: python

   import statistics
   import time

   from isaaclab.benchmark import (
       BaseIsaacLabBenchmark,
       IntMetadata,
       StatisticalMeasurement,
   )

   num_iterations = 100
   benchmark = BaseIsaacLabBenchmark(
       benchmark_name="custom_cpu_workload",
       formatter_type=["json", "summary"],
       output_path="results/custom",
       output_prefix="custom_cpu_workload",
       use_recorders=False,
       workflow_metadata={
           "metadata": [{"name": "num_iterations", "data": num_iterations}]
       },
   )

   output_paths = ()
   try:
       for _ in range(10):
           sum(value * value for value in range(10_000))

       samples_ms = []
       for _ in range(num_iterations):
           start_ns = time.perf_counter_ns()
           sum(value * value for value in range(10_000))
           samples_ms.append((time.perf_counter_ns() - start_ns) / 1.0e6)

       benchmark.add_measurement(
           "workload",
           measurement=StatisticalMeasurement(
               name="duration",
               mean=statistics.fmean(samples_ms),
               std=statistics.stdev(samples_ms),
               n=len(samples_ms),
               unit="ms",
           ),
           metadata=IntMetadata(name="warmup_iterations", data=10),
       )
   finally:
       output_paths = benchmark.finalize()

   for output_path in output_paths:
       print(output_path)

Choose a meaningful phase name, state units explicitly, warm up before sampling,
and report the sample count with the mean and standard deviation. For GPU work,
define whether the measurement is submission latency or synchronized completion
latency and place synchronization at deliberate boundaries.

When a custom producer creates one of the stable workflow bundles, call
``attach_bundle(bundle)`` before ``finalize()`` and select ``schema``. Otherwise,
use the lower-level ``json`` representation. ``finalize()`` stops optional Kit
frametime recorders, gathers recorder data, writes every selected formatter,
and returns the paths written.

System And Frametime Recorders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``use_recorders=True`` (the default), the base class captures CPU, GPU,
memory, and version metadata. Call ``update_manual_recorders()`` at least once
before ``finalize()``, and update during long workloads when current utilization
samples are needed.

Set ``frametime_recorders=True`` only when a Kit application is running and the
workload needs physics, render, application, or GPU frametime data. The
framework enables the available Isaac Sim benchmark services and gracefully
omits unavailable recorders. Frametime instrumentation is part of the workload;
keep its setting fixed across comparisons.

Isolate One Method Or Sensor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~isaaclab.benchmark.MethodBenchmarkRunner` builds on the base class to
repeat one method or property across input modes, warm-up steps, and instance
counts. It is the common bridge used by the asset micro-benchmarks. Sensor
benchmarks use live scenes, :func:`~isaaclab.benchmark.measure_latency` for
paired timing boundaries, and
:class:`~isaaclab.benchmark.LatencyBenchmarkRunner` for structured output.

Do not infer end-to-end environment or training throughput from either kind of
isolated result. The command matrix, backend prerequisites, mock-versus-live
distinction, output examples, and extension protocol are documented in
:ref:`testing_micro_benchmarks`.

Troubleshooting
---------------

Run scripts through ``./isaaclab.sh -p`` so the Isaac Lab and simulator Python
environment is active. If output is missing, inspect the returned
``output_paths`` and make sure custom producers call ``finalize()``. Formatter
names are lowercase and case-sensitive: ``schema``, ``summary``, ``json``,
``osmo``, and ``omniperf``.

Missing GPU metadata usually means ``nvidia-smi`` or CUDA device discovery is
unavailable. Missing Kit frametime measurements can instead mean the relevant
benchmark service is not present; it does not invalidate non-frametime phases.

For a new supported end-to-end workflow, keep typed request construction,
dispatch, schema output, summary output, and CLI behavior aligned. For an
isolated operation, continue with :ref:`testing_micro_benchmarks` before adding
a new runner or timing convention.
