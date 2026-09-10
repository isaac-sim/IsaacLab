.. _developer_tools_benchmarking_api:

Use the benchmark API
=====================

This guide is for automation authors who need to run supported benchmarks from
Python, consume typed results, or add a benchmark producer. For day-to-day
benchmarking, start with the CLI workflows and interpretation guidance in
:ref:`developer_tools_benchmarking_run`. To isolate one asset method, cached
property, or sensor update, use :ref:`developer_tools_benchmarking_micro`.

Run one workflow
----------------

Use a typed request for a supported workflow. It applies the same launch, task,
timing, schema, and output behavior as the CLI. The following runtime example
measures 1000 steps after 50 warm-up steps. It disables visualizers and requests
both stable schema and human-readable summary output:

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

The command prints the summary report. The paths in ``result.output_paths``
identify the schema and summary JSON files that were written. Use these paths in
automation instead of reconstructing the timestamped names.

Choose a request
----------------

Choose the request and dedicated runner that match the measurement question.
:func:`~isaaclab.benchmark.run_benchmark` accepts any request type when generic
dispatch is more convenient.

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

The examples below reuse the runtime workflow's output and launcher objects.

.. dropdown:: Run startup profiling

   Startup profiling records wall time and the most expensive functions for
   each phase:

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

.. dropdown:: Run training

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

.. dropdown:: Run playback

   Pass the persisted ``checkpoint_path`` to a play request in a **new
   process**:

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

Configure a request
-------------------

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
belong to one RL library. Use typed fields, presets, and ``hydra_args`` for
supported options. ``backend_args`` bypasses the common request contract.
See the :mod:`isaaclab.benchmark` API reference for the exhaustive field list.

An empty ``visualizers`` tuple explicitly disables every visualizer. ``None``
preserves task and environment defaults. Enabling cameras, rendering,
livestreaming, deterministic behavior, or animation recording changes the
workload and must be kept identical across compared runs.

Choose output formats
---------------------

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

Use ``("schema", "summary")`` for results intended for review or publication.
The summary gives immediate feedback. The schema preserves typed data for
analysis. With multiple formatters, each output filename includes its formatter
name, so JSON outputs cannot overwrite each other.

Read the result
---------------

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
time, and environment-step timing. For measurement definitions, see
:ref:`developer_tools_benchmarking_run`. See the public API reference for every
schema field.

Handle errors and process lifetime
----------------------------------

If a workflow parser rejects a typed request, the API raises ``ValueError``. If
a workflow exits without returning a result, the API raises ``RuntimeError``.
Invalid output configuration also raises an error. Examples include an empty
formatter tuple and an unknown formatter.

Run each workflow in a separate process. This gives each benchmark a clean
simulator lifecycle and prevents state from carrying over between workflows. It
also matches the CLI execution model. Process startup is reported separately
from steady-state throughput. Save ``result.bundle.checkpoint_path`` before
starting playback.

.. warning::

   ``measure_synchronized_step_breakdown=True`` inserts device
   synchronizations around environment and simulation step boundaries. This
   serializes work and changes the schedule being measured, especially on
   Newton. Every rate from that run is diagnostic and must not be reported as
   throughput. Time outside ``SimulationContext.step()`` includes required
   action, actuator, state, manager, reset, wrapper, and synchronization work.
   It is not equivalent to removable Isaac Lab overhead.

Add a custom producer
---------------------

Use :class:`~isaaclab.benchmark.BaseIsaacLabBenchmark` when a producer's
lifecycle cannot be represented by a supported workflow. The framework provides
phases, typed measurement and metadata records, system recorders, and
formatters. You must define the timing boundary.

.. list-table:: Measurement and metadata types
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
       use_recorders=True,
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
       benchmark.update_manual_recorders()
   finally:
       output_paths = benchmark.finalize()

   for output_path in output_paths:
       print(output_path)

Choose a meaningful phase name and state units explicitly. Warm up before
sampling. Report the sample count with the mean and standard deviation. For GPU
work, state whether the metric is submission latency or synchronized completion
latency. Place synchronization at deliberate boundaries.

When a custom producer creates one of the stable workflow bundles, call
``attach_bundle(bundle)`` before ``finalize()`` and select ``schema``. Otherwise,
use the lower-level ``json`` representation. ``finalize()`` stops optional Kit
frametime recorders and gathers recorder data. It writes every selected
formatter and returns the paths written.

With ``use_recorders=True`` (the default), the base class captures CPU, GPU,
memory, and version metadata. Call ``update_manual_recorders()`` at least once
before ``finalize()``, and update during long workloads when current utilization
samples are needed.

Set ``frametime_recorders=True`` only when a Kit application is running and the
workload needs physics, render, application, or GPU frametime data. The
framework enables the available Isaac Sim benchmark services and skips
unavailable recorders. Frametime instrumentation is part of the workload. Keep
its setting fixed across comparisons.

Choose a lower-level runner
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~isaaclab.benchmark.MethodBenchmarkRunner` repeats one method or
property across input modes, warm-up steps, and instance counts. Use
:func:`~isaaclab.benchmark.measure_latency` to define paired host-submission
and device-synchronized timing boundaries. Use
:class:`~isaaclab.benchmark.LatencyBenchmarkRunner` to report those structured
latency samples. Neither isolated result predicts end-to-end environment or
training throughput. The command matrices and extension protocol are in
:ref:`developer_tools_benchmarking_micro`.

Troubleshooting
---------------

Run scripts through ``./isaaclab.sh -p`` so the Isaac Lab and simulator Python
environment is active. If output is missing, inspect returned ``output_paths``
and make sure custom producers call ``finalize()``. Formatter names are lowercase
and case-sensitive. Valid names are ``schema``, ``summary``, ``json``, ``osmo``,
and ``omniperf``.

Missing GPU metadata usually means ``nvidia-smi`` or CUDA device discovery is
unavailable. Missing Kit frametime measurements can instead mean the relevant
benchmark service is not present. It does not invalidate non-frametime phases.

For a new supported end-to-end workflow, keep typed request construction,
dispatch, schema output, summary output, and CLI behavior aligned. For an
isolated operation, follow :ref:`developer_tools_benchmarking_micro` before
adding a new runner or timing convention.
