.. _developer_tools_benchmarking_run:

Run benchmarks
==============

This guide covers the ``runtime``, ``play``, ``training``, and ``startup``
benchmark workflows. For isolated operations, see
:ref:`developer_tools_benchmarking_micro`. For Python use and framework
extensions, see :ref:`developer_tools_benchmarking_api`. See
:ref:`developer_tools_benchmarking_multigpu` for workflows that use several
GPUs.

Choose a workflow
-----------------

.. list-table::
   :header-rows: 1
   :widths: 15 35 25 25

   * - Workflow
     - Use it to measure
     - Primary result
     - Excludes from the primary result
   * - ``runtime``
     - Environment-step capacity under random actions
     - Environment-step FPS
     - Policy inference and learning
   * - ``play``
     - Trained-policy rollout
     - Collection FPS
     - Learning updates
   * - ``training``
     - End-to-end learning
     - Total FPS and learning metrics
     - Startup
   * - ``startup``
     - Launch, imports, configuration, scene creation, and first step
     - Phase duration [s]
     - Steady-state throughput
   * - ``*-multigpu``
     - A supported workflow with one rank per GPU
     - The corresponding workflow result, with its scope recorded in ``extra``
     - Single-GPU and cross-scope inference

For every comparison:

* Keep the workload and provenance fixed.
* Compare identical schema fields and measurement modes.
* Separate cold startup from steady state.

Runtime
-------

When to use it
~~~~~~~~~~~~~~

Use ``runtime`` to measure a task's environment-step capacity without policy
inference or learning. Use it to screen changes in simulation or task
throughput.

Run it
~~~~~~

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

Read the result
~~~~~~~~~~~~~~~

The command prints a throughput and resource summary when the run finishes. The
JSON output contains the full result.

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
        "schema_version": "1.4",
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
        "schema_version": "1.4",
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
   Isaac-Cartpole-Direct with 4096 environments. The rendered run used
   Isaac-Cartpole-Camera-Direct with 1024 environments, RTX rendering, and the
   RGB preset. The approved balanced power profile used intel_pstate with
   powersave governors and balance_performance energy preferences.

``--warmup_steps`` runs the exact number of excluded environment steps before
the measured window. Runtime executes ``warmup_steps + num_steps`` total
``env.step()`` calls, while the throughput window always contains exactly
``num_steps`` calls. With nonzero warm-up, the first warm-up step supplies the
separate ``first_step`` startup diagnostic. With zero warm-up, the first measured
step supplies that diagnostic without being removed from the measured window.

What not to infer
~~~~~~~~~~~~~~~~~

Do not infer policy-serving or training performance from a runtime result. It
contains neither policy inference nor a policy update. Do not compare its FPS
against a rendered, differently sized, or differently instrumented workload.

Play
----

When to use it
~~~~~~~~~~~~~~

Use ``play`` to measure a trained policy's rollout throughput or evaluate its
behavior. First run ``training`` to produce a checkpoint for the same RL library
and task, or supply another compatible checkpoint.

Run it
~~~~~~

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

Read the result
~~~~~~~~~~~~~~~

Use ``runtime.collection_fps`` or ``runtime.total_fps`` for policy inference
plus rollout. Both fields describe the same scope in play. Use
``runtime.environment_step_timing.environment_step_fps`` to isolate the
``env.step()`` boundary. ``reward``, ``ep_length``, and ``success_rate`` are
computed only from completed episodes and can be ``null`` when the measured
window completes no episodes or the task reports no success value.

.. code-block:: text

   runtime.collection_fps
   runtime.total_fps
   runtime.environment_step_timing.environment_step_fps

``--warmup_steps`` excludes only the first N ``env.step()`` calls from
``runtime.environment_step_timing``. It does not exclude those steps from
collection FPS, total FPS, policy evaluation, episode statistics, or wall time.

The typed result adds play-specific ``reward``, ``ep_length``,
``success_rate``, and ``checkpoint_path`` fields to the canonical runtime
envelope shown above. Capture the same hardware, software revision, workload,
seed, warm-up, rendering, and power-profile provenance for comparisons.

What not to infer
~~~~~~~~~~~~~~~~~

Do not interpret play collection FPS as environment-only performance. Absent
episode metrics do not mean zero reward or zero success. A play run does not
measure learning or policy-update throughput.

Training
--------

When to use it
~~~~~~~~~~~~~~

Use ``training`` to measure end-to-end learning throughput and learning
behavior. The run writes the RL library's normal logs and checkpoints. Pass a
compatible saved checkpoint to ``play``.

Run it
~~~~~~

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

Read the result
~~~~~~~~~~~~~~~

The command prints a throughput, resource, and learning summary when the run
finishes. The JSON output contains the full result.

Use ``runtime.collection_fps`` for rollout collection without policy update,
``runtime.total_fps`` for collection plus update, and
``runtime.environment_step_timing.environment_step_fps`` for environment-only
stepping. Inspect ``learning.reward`` and ``learning.ep_length`` for learning
behavior instead of reducing training to one throughput value.

.. code-block:: text

   runtime.collection_fps
   runtime.total_fps
   runtime.environment_step_timing.environment_step_fps
   learning.reward
   learning.ep_length

``--warmup_steps`` excludes only the first N ``env.step()`` calls from
``runtime.environment_step_timing``. It does not exclude work from collection
FPS, total FPS, learning curves, policy evaluation, episode statistics, or wall
time.

The typed result adds training-specific ``run.framework``,
``run.max_iterations``, ``learning``, ``success_rate``, and
``checkpoint_path`` fields to the canonical runtime envelope shown above.
Capture the same provenance plus the RL library and training-iteration count.

What not to infer
~~~~~~~~~~~~~~~~~

Do not treat a faster environment-step rate as proof of faster end-to-end
training or equivalent learning. Do not compare short training curves as if
they established final policy quality.

.. _developer_tools_benchmarking_multigpu:

Multi-GPU
---------

When to use it
~~~~~~~~~~~~~~

Append ``-multigpu`` to ``startup``, ``runtime``, or ``training`` to use one rank
per GPU. Use it to measure synchronized multi-GPU training throughput. You can
also measure how much a workflow slows down when every GPU on the node is busy.

Run it
~~~~~~

.. code-block:: bash

   ./isaaclab.sh benchmark training-multigpu \
       --rl_library rsl_rl \
       --num_gpus 2 \
       --task Isaac-Cartpole-Direct \
       --num_envs 4096 \
       --max_iterations 100 \
       --seed 42 \
       --visualizer none \
       --output_path ./benchmark_results/multigpu \
       physics=isaacsim_physx

The launcher accepts ``--num_gpus``, ``--nnodes``, ``--node_rank``, and the
``torchrun`` rendezvous options, just like :ref:`train-multigpu-command`. Every
other argument is passed to the single-GPU workflow unchanged. Add ``--dry_run``
to print the ``torchrun`` command without running it. Add ``--log_all_ranks`` to
show output from every rank instead of local rank 0 only.

For a multi-node run, issue the same command on every node with a distinct
``--node_rank``:

.. code-block:: bash

   ./isaaclab.sh benchmark training-multigpu \
       --rl_library rsl_rl --nnodes 2 --node_rank 0 --num_gpus 8 \
       --rdzv_backend c10d --rdzv_endpoint host0:29400 --rdzv_id bench \
       --task Isaac-Cartpole-Direct

``training-multigpu`` supports ``rsl_rl``, ``rl_games``, and ``skrl`` with
Torch. It does not support skrl JAX or SB3. It also rejects ``--video``,
``--capture_env_sensors``, and ``--check_success`` because these options do not
apply across ranks. Use :ref:`train-multigpu-command` for general distributed
training.

Read the result
~~~~~~~~~~~~~~~

``--num_envs`` is the number of environments **per rank**. Each rank creates its
own Isaac Lab instance on its own GPU. Only global rank 0 writes a bundle. The
``extra`` fields record what that bundle covers:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - ``extra`` field
     - Meaning
   * - ``world_size``, ``local_world_size``, ``num_nodes``
     - Rank layout of the job.
   * - ``num_envs_per_rank``
     - Environments hosted by each rank.
   * - ``workload_scope``
     - ``global`` for ``training-multigpu``: ranks train in lockstep, so ``run.num_envs``,
       ``runtime.steps_per_iteration``, and every FPS field cover all ranks.
       ``rank0`` for ``startup-multigpu`` and ``runtime-multigpu``: those ranks run
       independent workloads, so the reported values are rank 0's own, measured while
       the other ranks contend for the same host.
   * - ``measurement_scope``
     - ``rank0_process`` — timings, learning curves, CPU, and RAM come from rank 0 alone.
   * - ``gpu_measurement_scope``
     - ``rank0_node`` — ``resources.devices`` reports every GPU visible to rank 0, so a
       single-node run shows all ranks. ``resources.gpu_util_pct`` and
       ``resources.gpu_mem_gb`` remain scoped to rank 0's own device.

What not to infer
~~~~~~~~~~~~~~~~~

Do not compare a multi-GPU result against a single-GPU result at the same
``--num_envs``. The multi-GPU run has ``world_size`` times as many environments.
To measure scaling, compare the global throughput of an ``N``-GPU run against
``N`` times the throughput of a single-GPU run. Keep the per-rank environment
count fixed. Do not read ``startup-multigpu`` or ``runtime-multigpu`` throughput
as a global rate. Their ``workload_scope`` is ``rank0``, and the other ranks were
not measured.

Startup
-------

When to use it
~~~~~~~~~~~~~~

Use ``startup`` to investigate launch or cold initialization. Faster startup
shortens the edit-run-debug cycle. The workflow separates five cold phases:

* ``app_launch`` enters the simulation launcher and initializes its runtime.
* ``python_imports`` imports launcher, task-registration, and runtime libraries.
* ``task_config`` resolves the requested task configuration.
* ``env_creation`` runs ``gym.make()`` and the initial ``env.reset()``.
* ``first_step`` runs the first ``env.step()`` and waits for device completion.

Run it
~~~~~~

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

Read the result
~~~~~~~~~~~~~~~

A per-phase wall-time summary is printed to the console when the run finishes,
including the timers that ran during ``env_creation``. The JSON output holds the
full profile.

Read the wall time and attributed functions under each entry in ``phases``.

.. code-block:: text

   phases.<phase>.total_time_s
   phases.<phase>.top_functions

Pass ``--whitelist_config scripts/benchmarks/startup_whitelist.yaml`` to select
stable ``fnmatch`` patterns for specific phases. Whitelist mode ignores
``--top_n`` for listed phases. The output still includes unmatched patterns.
Their own time, cumulative time, and call count are zero. This keeps dashboard
keys stable. The command also logs a warning that names each unmatched pattern.

Patterns match profile labels built relative to each installed package root.
In-repo functions have no package prefix
(``utils.assets:_find_asset_dependencies``). External packages keep their full
dotted path (``warp._src.context:launch``).

The typed result replaces runtime throughput fields with startup-specific
``config`` and ``phases`` mappings. Each phase reports wall time and selected
profile entries. Capture the same provenance plus cache state, process order,
``top_n``, and whitelist configuration.

Startup deliberately has no warm-up: cold work is the measurement. Run it in a
fresh process and control cache state and execution order when comparing runs.

What not to infer
~~~~~~~~~~~~~~~~~

``cProfile`` is an attribution tool with observer cost. Treat phase wall times
and function attribution as cold-start diagnostics, not as steady-state
throughput or an unperturbed timing trace.

Measurement boundaries
----------------------

The primary rates have these boundaries:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Field and workflow
     - Measured scope
   * - ``runtime.environment_step_timing.environment_step_fps`` (runtime)
     - ``env.step()`` under random actions. Random-action generation is excluded.
   * - ``runtime.collection_fps`` (play)
     - Policy inference and rollout, including ``env.step()``.
   * - ``runtime.environment_step_timing.environment_step_fps`` (play)
     - ``env.step()`` only. Policy inference is excluded.
   * - ``runtime.collection_fps`` (training)
     - Rollout collection. Policy update is excluded.
   * - ``runtime.total_fps`` (training)
     - Collection and policy update together.
   * - ``runtime.environment_step_timing.environment_step_fps`` (training)
     - ``env.step()`` only. Inference and learning are excluded.
   * - ``phases.<phase>.total_time_s`` (startup)
     - Cold startup work. This is not steady-state throughput.

Compare runs only when the schema field name and
``runtime.environment_step_timing.measurement_mode`` match. Matching FPS units
alone is not enough.

Rendered workloads
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
the headless ``Isaac-Cartpole-Direct`` walkthrough. Do not compare their FPS as
a backend-only delta.

Physics backends
----------------

Keep the command fixed and substitute one physics selector:

.. code-block:: text

   physics=isaacsim_physx
   physics=newton_mjwarp
   physics=ovphysx

For a backend comparison, keep the task, environment count, seed, presets,
renderer, warm-up, measured window, and measurement mode identical. A renderer
or task that is valid for one backend may be incompatible with another. Choose
a common supported workload before collecting the comparison.

Read the output
---------------

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
   jq '.phases' benchmark_startup_*_schema.json

With multiple formatters, filenames include ``_summary`` and ``_schema`` for
the summary and typed bundle respectively.

Evidence levels
~~~~~~~~~~~~~~~

Choose the evidence level before collecting data:

* **One-run exploration:** use one controlled run to check direction, find a
  bottleneck, or validate the benchmark setup. Do not publish a performance
  claim from it.
* **Gross regression screening:** use at least three paired independent
  processes with rotated seeds and execution order. Pair baseline and candidate
  runs collected under the same conditions.
* **Performance claims:** use at least three independent processes and longer,
  repeated training runs. The runs must cover the behavior being claimed. A
  short throughput probe is not enough.

Compare runs
------------

Report individual run values, paired deltas, and dispersion or confidence
intervals. Label a result inconclusive when the interval for the delta crosses
zero. Avoid reporting only the best run or only an aggregate.

Control background activity. Record any changes alongside each run.

Every advertised result must include this provenance:

.. list-table:: Required provenance
   :header-rows: 1
   :widths: 22 78

   * - Group
     - Required fields
   * - Hardware
     - CPU model, physical core count, RAM, and GPU.
   * - Software
     - OS, physics backend, software versions, and revision.
   * - Workload
     - Task, environment count, seed, warm-up, measured window, and rendering
       configuration.
   * - Run conditions
     - Power profile and execution order.

Performance depends on the full CPU, GPU, software, and workload configuration.
It is not a property of the GPU alone. Process CPU utilization is summed across
cores, so it can exceed 100%.

Synchronized-step diagnostics
---------------------------------

Runtime, play, and training accept this optional diagnostic flag:

.. code-block:: bash

   --measure_sync_step

It changes ``measurement_mode`` from ``host_return`` to
``serialized_synchronized``. The diagnostic synchronizes before environment
and simulation boundaries. It also serializes work that normally runs
asynchronously and can greatly slow Newton. Every timing and rate in the
instrumented workflow uses this changed schedule. Do not report these results as
throughput.

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
   runs. Record execution order, power mode, and background-activity changes so
   thermal or contention effects are visible.
