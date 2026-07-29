.. _testing_benchmarks:

Benchmarking Isaac Lab
======================

Isaac Lab provides supported benchmark workflows for environment stepping,
trained-policy playback, reinforcement-learning training, and startup profiling.
This guide explains which workflow to use, how to run it, and how to interpret
its results.

.. seealso::

   To call workflows from Python or extend the framework, see
   :ref:`testing_benchmark_framework`. To isolate individual asset or sensor
   operations, see :ref:`testing_micro_benchmarks`.

Choose A Workflow
-----------------

.. list-table::
   :header-rows: 1
   :widths: 55 20

   * - Question
     - Workflow
   * - Large environment-step capacity change
     - ``runtime``
   * - Trained-policy behavior or deployment throughput
     - ``play``
   * - End-to-end learning throughput or learning behavior
     - ``training``
   * - Launch, import, configuration, scene creation, or first-step latency
     - ``startup``
   * - One asset or sensor operation
     - :ref:`testing_micro_benchmarks`

Use these rules for every comparison:

* Compare identical scopes. A collection rate, an environment-step rate, and an
  end-to-end rate answer different questions even when they share FPS units.
* Separate startup from steady state. Cold launch and first-step costs are not
  part of steady-state throughput.
* Control the CPU, GPU, software revision and versions, task workload, seed,
  power mode, and background activity. Record changes to any of them.

Runtime Quick Start
-------------------

Use It When
~~~~~~~~~~~

Use ``runtime`` to measure a task's environment-step capacity without policy
inference or learning. This is the shortest supported path for screening a
large change in simulation or task throughput.

Command
~~~~~~~

From a source installation, run:

.. code-block:: bash

   ./isaaclab.sh benchmark runtime \
       --task Isaac-Cartpole-Direct \
       --num_envs 4096 \
       --warmup_steps 50 \
       --num_steps 1000 \
       --seed 42 \
       --visualizer none \
       --benchmark_formatter schema,summary \
       --output_path ./benchmark_results \
       physics=isaacsim_physx

``summary`` prints a human-readable terminal report and writes its flat metrics
JSON. ``schema`` writes the stable, typed JSON bundle used for programmatic
comparison. With multiple formatters, their filenames include ``_summary`` and
``_schema`` respectively.

Warm-Up
~~~~~~~

``--warmup_steps`` runs the exact number of excluded environment steps before
the measured window. Runtime executes ``warmup_steps + num_steps`` total
``env.step()`` calls, while the throughput window always contains exactly
``num_steps`` calls. With nonzero warm-up, the first warm-up step supplies the
separate ``first_step`` startup diagnostic. With zero warm-up, the first measured
step supplies that diagnostic without being removed from the measured window.

Read The Result
~~~~~~~~~~~~~~~

Read ``runtime.environment_step_timing.environment_step_fps`` for the aggregate
environment-step rate. Runtime samples random actions before starting the
``env.step()`` timer, so random-action generation is outside this timing. For a
runtime run, ``runtime.collection_fps`` and ``runtime.total_fps`` describe the
same random-action stepping workload.

.. dropdown:: Canonical workstation output and provenance

   Headless runtime summary (output abbreviated):

   .. code-block:: text

      |                                   Summary Report                                   |
      | workflow_name: benchmark_runtime                                                   |
      | task: Isaac-Cartpole-Direct                                                        |
      | num_envs: 4096                                                                     |
      |   Collection FPS (mean/std/max): 857289.11 / 88395.50 / 983383.55 FPS              |
      |   Total FPS (mean/std/max): 857289.11 / 88395.50 / 983383.55 FPS                   |
      |   Environment Step Host-Return FPS (mean/std/max): 857648.65 / 88448.67 /          |
      | 983822.18 FPS                                                                      |
      [... output abbreviated ...]

   Headless typed-schema excerpt:

   .. code-block:: json

      {
        "schema_version": "1.3",
        "run": {
          "config": {"physics_backend": "physx", "rendering_backend": "none", "presets": ["physx"]},
          "task": "Isaac-Cartpole-Direct", "seed": 42, "status": "completed", "num_envs": 4096
        },
        "runtime": {
          "iterations_completed": 1000,
          "collection_fps": {"mean": 857289.1054855952},
          "total_fps": {"mean": 857289.1054855952},
          "environment_step_timing": {
            "environment_step_fps": {"mean": 857648.6464583826},
            "environment_step_calls": 1000, "measurement_mode": "host_return"
          }
        }
      }

   Rendered runtime summary (output abbreviated):

   .. code-block:: text

      |                                   Summary Report                                   |
      | workflow_name: benchmark_runtime                                                   |
      | task: Isaac-Cartpole-Camera-Direct                                                 |
      | num_envs: 1024                                                                     |
      |   Collection FPS (mean/std/max): 31587.33 / 2203.25 / 36188.75 FPS                 |
      |   Total FPS (mean/std/max): 31587.33 / 2203.25 / 36188.75 FPS                      |
      |   Environment Step Host-Return FPS (mean/std/max): 31590.46 / 2203.47 / 36192.00   |
      | FPS                                                                                |
      [... output abbreviated ...]

   Rendered typed-schema excerpt:

   .. code-block:: json

      {
        "schema_version": "1.3",
        "run": {
          "config": {
            "physics_backend": "physx", "rendering_backend": "isaacsim_rtx",
            "presets": ["physx", "isaacsim_rtx", "rgb"]
          },
          "task": "Isaac-Cartpole-Camera-Direct", "seed": 42,
          "status": "completed", "num_envs": 1024
        },
        "runtime": {
          "iterations_completed": 1000,
          "collection_fps": {"mean": 31587.33014449644},
          "total_fps": {"mean": 31587.33014449644},
          "environment_step_timing": {
            "environment_step_fps": {"mean": 31590.455581417973},
            "environment_step_calls": 1000, "measurement_mode": "host_return"
          }
        }
      }

Capture provenance: Intel(R) Core(TM) i9-14900K CPU, NVIDIA GeForce RTX 5090
GPU, Ubuntu 24.04.3, revision
f02ca894a91f9db3a9ab0d42fcf23a5bc5eae22d. Both runs used PhysX, seed 42,
50 warm-up steps, and a 1000-step measured window. The headless run used
Isaac-Cartpole-Direct with 4096 environments; the rendered run used
Isaac-Cartpole-Camera-Direct with 1024 environments, RTX rendering, and the RGB
preset. The approved balanced power profile used intel_pstate with powersave
governors and balance_performance energy preferences.

Do Not Infer
~~~~~~~~~~~~

Do not infer policy-serving or training performance from a runtime result. It
contains neither policy inference nor a policy update. Do not compare its FPS
against a rendered, differently sized, or differently instrumented workload.

Measurement Boundaries
----------------------

The primary rates have these boundaries:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Field and workflow
     - Measured scope
   * - Runtime environment-step FPS
     - ``env.step()`` under random actions; random-action generation is excluded.
   * - Play collection FPS
     - Policy inference and rollout, including ``env.step()``.
   * - Play environment-step FPS
     - ``env.step()`` only; policy inference is excluded.
   * - Training collection FPS
     - Rollout collection; policy update is excluded.
   * - Training total FPS
     - Collection and policy update together.
   * - Training environment-step FPS
     - ``env.step()`` only; inference and learning are excluded.
   * - Startup phase time
     - Cold startup work; it is not steady-state throughput.

The same schema field name and the same
``runtime.environment_step_timing.measurement_mode`` are prerequisites for a
valid comparison. Matching FPS units alone is insufficient.

Play
----

Use It When
~~~~~~~~~~~

Use ``play`` to measure a trained policy's rollout throughput or evaluate its
behavior. First run ``training`` to produce a checkpoint for the same RL library
and task, or supply another compatible checkpoint.

Command
~~~~~~~

.. code-block:: bash

   ./isaaclab.sh benchmark play \
       --rl_library rsl_rl \
       --task Isaac-Cartpole-Direct \
       --num_envs 4096 \
       --num_steps 1000 \
       --warmup_steps 50 \
       --checkpoint /path/to/model.pt \
       --seed 42 \
       --visualizer none \
       --benchmark_formatter schema,summary \
       --output_path ./benchmark_results/play \
       physics=isaacsim_physx

Warm-Up
~~~~~~~

``--warmup_steps`` excludes only the first N ``env.step()`` calls from
``runtime.environment_step_timing``. It does not exclude those steps from
collection FPS, total FPS, policy evaluation, episode statistics, or wall time.

Read The Result
~~~~~~~~~~~~~~~

Use ``runtime.collection_fps`` or ``runtime.total_fps`` for policy inference
plus rollout; they describe the same scope in play. Use
``runtime.environment_step_timing.environment_step_fps`` to isolate the
``env.step()`` boundary. ``reward``, ``ep_length``, and ``success_rate`` are
computed only from completed episodes and can be ``null`` when the measured
window completes no episodes or the task reports no success value.

The typed result adds play-specific ``reward``, ``ep_length``,
``success_rate``, and ``checkpoint_path`` fields to the canonical runtime
envelope shown above. Capture the same hardware, software revision, workload,
seed, warm-up, rendering, and power-profile provenance for comparisons.

Do Not Infer
~~~~~~~~~~~~

Do not interpret play collection FPS as environment-only performance, and do
not interpret absent episode metrics as zero reward or zero success. A play run
does not measure learning or policy-update throughput.

Training
--------

Use It When
~~~~~~~~~~~

Use ``training`` to measure end-to-end learning throughput and learning
behavior. The run writes the RL library's normal logs and checkpoints; use a
compatible saved checkpoint as the input to ``play``.

Command
~~~~~~~

.. code-block:: bash

   ./isaaclab.sh benchmark training \
       --rl_library rsl_rl \
       --task Isaac-Cartpole-Direct \
       --num_envs 4096 \
       --max_iterations 100 \
       --warmup_steps 50 \
       --seed 42 \
       --visualizer none \
       --benchmark_formatter schema,summary \
       --output_path ./benchmark_results/training \
       physics=isaacsim_physx

Warm-Up
~~~~~~~

``--warmup_steps`` excludes only the first N ``env.step()`` calls from
``runtime.environment_step_timing``. It does not exclude work from collection
FPS, total FPS, learning curves, policy evaluation, episode statistics, or wall
time.

Read The Result
~~~~~~~~~~~~~~~

Use ``runtime.collection_fps`` for rollout collection without policy update,
``runtime.total_fps`` for collection plus update, and
``runtime.environment_step_timing.environment_step_fps`` for environment-only
stepping. Inspect ``learning.reward`` and ``learning.ep_length`` for learning
behavior instead of reducing training to one throughput value.

The typed result adds training-specific ``run.framework``,
``run.max_iterations``, ``learning``, ``success_rate``, and
``checkpoint_path`` fields to the canonical runtime envelope shown above.
Capture the same provenance plus the RL library and training-iteration count.

Do Not Infer
~~~~~~~~~~~~

Do not treat a faster environment-step rate as proof of faster end-to-end
training or equivalent learning. Do not compare short training curves as if
they established final policy quality.

Startup Profiling
-----------------

Use It When
~~~~~~~~~~~

Use ``startup`` when launch or cold initialization is the subject of the
investigation. Startup latency is paid in edit-run-debug cycles, so reducing
it shortens developer iteration and matters for quick prototyping. It
separates five cold phases:

* ``app_launch`` enters the simulation launcher and initializes its runtime.
* ``python_imports`` imports launcher, task-registration, and runtime libraries.
* ``task_config`` resolves the requested task configuration.
* ``env_creation`` runs ``gym.make()`` and the initial ``env.reset()``.
* ``first_step`` runs the first ``env.step()`` and waits for device completion.

Command
~~~~~~~

.. code-block:: bash

   ./isaaclab.sh benchmark startup \
       --task Isaac-Cartpole-Direct \
       --num_envs 4096 \
       --top_n 30 \
       --seed 42 \
       --visualizer none \
       --benchmark_formatter schema,summary \
       --output_path ./benchmark_results/startup \
       physics=isaacsim_physx

Warm-Up
~~~~~~~

Startup deliberately has no warm-up: cold work is the measurement. Run it in a
fresh process and control cache state and execution order when comparing runs.

Read The Result
~~~~~~~~~~~~~~~

Read the wall time and attributed functions under each entry in ``phases``.
Pass ``--whitelist_config scripts/benchmarks/startup_whitelist.yaml`` to select
stable ``fnmatch`` patterns for specific phases. Whitelist mode ignores
``--top_n`` for listed phases. A pattern that matches no profiled function is
still emitted with zero own time, cumulative time, and calls, which preserves
stable dashboard keys.

The typed result replaces runtime throughput fields with startup-specific
``config`` and ``phases`` mappings. Each phase reports wall time and selected
profile entries. Capture the same provenance plus cache state, process order,
``top_n``, and whitelist configuration.

Do Not Infer
~~~~~~~~~~~~

``cProfile`` is an attribution tool with observer cost. Treat phase wall times
and function attribution as cold-start diagnostics, not as steady-state
throughput or an unperturbed timing trace.

Rendered Workloads
------------------

Rendering changes the workload. Use a camera task, camera enablement, renderer,
and sensor preset explicitly:

.. code-block:: bash

   ./isaaclab.sh benchmark runtime \
       --task Isaac-Cartpole-Camera-Direct \
       --num_envs 1024 \
       --warmup_steps 50 \
       --num_steps 1000 \
       --seed 42 \
       --enable_cameras \
       --visualizer none \
       --benchmark_formatter schema,summary \
       --output_path ./benchmark_results/rendered \
       physics=isaacsim_physx renderer=isaacsim_rtx presets=rgb

Selecting ``summary`` enables the available Kit physics, rendering,
application, and GPU frame-time recorders. Some recorders may be unavailable in
a particular installation or backend combination. The camera task, 1024
environments, RTX renderer, and RGB preset make this a different workload from
the headless ``Isaac-Cartpole-Direct`` walkthrough; do not compare their FPS as
a backend-only delta.

Physics Backends
----------------

Keep the command fixed and substitute one physics selector:

.. code-block:: text

   physics=isaacsim_physx
   physics=newton_mjwarp
   physics=ovphysx

For a backend comparison, keep the task, environment count, seed, presets,
renderer, warm-up, measured window, and measurement mode identical. A renderer
or task that is valid for one backend may be incompatible with another; choose
a common supported workload before collecting the comparison.

Interpret The Output
--------------------

``--benchmark_formatter`` accepts a comma-separated list. The formatters serve
different consumers:

* ``schema`` writes the stable typed bundle. Use it for analysis, comparison,
  and archival evidence.
* ``summary`` prints a terminal report and also writes flat metrics JSON. It
  enables available Kit frame-time recorders for runtime, play, and training.
* ``json`` writes all legacy flat phases, measurements, and metadata.
* ``osmo`` writes per-phase, single-value KPI documents for Osmo ingestion.
* ``omniperf`` writes phase-grouped KPI JSON for performance tracking and
  database upload.

Use ``schema,summary`` for interactive runs. These ``jq`` queries address the
typed schema paths:

.. code-block:: bash

   jq '.run, .runtime.collection_fps, .runtime.total_fps' benchmark_*_schema.json
   jq '.runtime.environment_step_timing' benchmark_*_schema.json
   jq '.phases' startup_*_schema.json

With multiple formatters, the ``_schema`` suffix identifies the typed bundle.

Evidence Levels
---------------

Choose the evidence level before collecting data:

* **One-run exploration:** use one controlled run to check direction, find a
  bottleneck, or validate the benchmark setup. Do not publish a performance
  claim from it.
* **Gross regression screening:** use at least three paired runs with rotated
  seeds and execution order. Pair baseline and candidate runs collected under
  the same conditions.
* **Performance claims:** use longer, repeated training runs that cover the
  behavior being claimed, not only a short throughput probe.

Report individual run values, paired deltas, and dispersion or confidence
intervals. Label a result inconclusive when the interval for the delta crosses
zero. Avoid reporting only the best run or only an aggregate.

Every advertised result must identify the CPU model, physical core count, RAM,
GPU, OS, physics backend, software versions, revision, task, environment count,
seed, warm-up, measured window, and execution order. Benchmark performance is a
property of the whole CPU, GPU, software, and workload configuration, not the
GPU alone. Process CPU utilization is aggregated across cores and can therefore
exceed 100%.

Synchronized Step Diagnostic
----------------------------

Runtime, play, and training accept this optional diagnostic flag:

.. code-block:: bash

   --measure_synchronized_step_breakdown

It changes ``measurement_mode`` from ``host_return`` to
``serialized_synchronized``. The diagnostic synchronizes before environment
and simulation boundaries, serializes normally asynchronous work, and can
materially slow Newton. Every timing and rate collected inside the instrumented
workflow observes that changed schedule. It is observer-perturbed and must not
be used for authoritative throughput measurements.

The diagnostic partitions synchronized environment-step time into simulation
time and an **outside-simulation remainder**. That remainder contains required
action, actuator, state, manager, reset, wrapper, and synchronization work. It
is an arithmetic remainder for the instrumented schedule, not removable
framework overhead. Never subtract it to predict attainable throughput.

Troubleshooting
---------------

Missing checkpoint
   Run training first, or pass a checkpoint created by the selected RL library
   for the same task and compatible agent configuration. Check the path and the
   ``--rl_library`` selection.

Incomplete play episodes
   Increase ``--num_steps``. Reward, episode length, and success aggregates
   require completed episodes and may otherwise be ``null``.

Invalid counts
   ``--num_steps`` and an explicitly supplied ``--max_iterations`` must be
   greater than zero. ``--warmup_steps`` must be non-negative.

Missing resource or frame-time metrics
   Very short runs may finish before the periodic resource monitor samples
   enough data. Frame-time metrics require the corresponding Kit recorder to be
   available and are enabled by ``summary`` or ``omniperf``.

Incompatible renderer
   Check that the physics backend, renderer, camera task, sensor preset, and
   visualizer combination is supported. Use an identical common configuration
   for comparisons.

Mismatched measurement modes
   Compare ``host_return`` only with ``host_return`` and
   ``serialized_synchronized`` only with the same diagnostic mode. Also verify
   that the schema field names match.

Foreign GPU workloads
   Stop unrelated GPU jobs and background activity, then repeat the paired
   runs. Record execution order and power mode so thermal or contention effects
   are visible.
