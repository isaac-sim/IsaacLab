.. Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _developer_tools_benchmarking_micro:

Write micro-benchmarks
======================

Micro-benchmarks answer isolated performance questions. Asset benchmarks use
backend-specific **mock views** to measure an asset method or data property.
Sensor benchmarks use **live simulation scenes** to measure a production sensor
update after physics has completed. Neither type predicts end-to-end environment
or training throughput. Use :ref:`developer_tools_benchmarking_run` for
environment logic, policy inference, learning, or application startup.

.. seealso::

   For the typed Python API, formatter and recorder internals, or custom
   producers, see :ref:`developer_tools_benchmarking_api`.

Choose a suite and backend
--------------------------

Run commands from the repository root through ``./isaaclab.sh``. The active
Python environment must contain the backend being measured. PhysX and Newton
asset benchmarks run kitless with mock views. PhysX sensor benchmarks launch
Isaac Sim. Newton sensor benchmarks run kitless with the installed Newton
runtime. OVPhysX runs kitless with its optional ``ovphysx`` runtime wheel. Use
CUDA for representative GPU numbers. CPU execution is useful for correctness or
profiling, but it is a different workload. Do not mix CPU and CUDA results.

.. list-table::
   :header-rows: 1
   :widths: 24 28 22 26

   * - Question
     - Workload
     - Simulation mode
     - Supported backends
   * - How fast is one asset method or data property?
     - ``articulation``
     - Backend-specific mock view with no physics
     - ``physics=physx``, ``physics=newton_mjwarp``,
       ``physics=newton_kamino``, ``physics=ovphysx``
   * - How fast is one asset method or data property?
     - ``rigid_object``
     - Backend-specific mock view with no physics
     - ``physics=physx``, ``physics=newton_mjwarp``,
       ``physics=newton_kamino``, ``physics=ovphysx``
   * - How fast is one asset method or data property?
     - ``rigid_object_collection``
     - Backend-specific mock view with no physics
     - ``physics=physx``, ``physics=newton_mjwarp``,
       ``physics=newton_kamino``, ``physics=ovphysx``
   * - How fast is one production sensor update?
     - ``contact_sensor``
     - Live scene with an untimed physics step
     - ``physics=physx``, ``physics=newton_mjwarp``,
       ``physics=newton_kamino``, ``physics=ovphysx``
   * - How fast is one production sensor update?
     - ``frame_transformer``
     - Live scene with an untimed physics step
     - ``physics=physx``, ``physics=newton_mjwarp``,
       ``physics=newton_kamino``, ``physics=ovphysx``
   * - How fast is one production sensor update?
     - ``imu`` or ``pva``
     - Live scene with an untimed physics step
     - ``physics=physx``, ``physics=newton_mjwarp``,
       ``physics=newton_kamino``, ``physics=ovphysx``
   * - How fast is one production sensor update?
     - ``joint_wrench``
     - Live scene with an untimed physics step
     - ``physics=physx``, ``physics=newton_mjwarp``,
       ``physics=newton_kamino``, ``physics=ovphysx``
   * - How fast is one production sensor update?
     - ``ray_caster``
     - Live scene with an untimed physics step
     - ``physics=physx``, ``physics=newton_mjwarp``,
       ``physics=newton_kamino``, ``physics=ovphysx``

Asset entry points are under each backend's ``benchmark/assets`` directory.
Sensor entry points are under ``benchmark/sensors``. The top-level command
selects one exact variant and component. Use the same component, dimensions,
mode, and selector when comparing results.

Run an asset benchmark
----------------------

Asset benchmarks isolate Python input handling, tensor or index conversion,
Isaac Lab method logic, backend binding calls, and data-property computation.
They run method then data-property phases and write separate historical method
and data artifacts. Existing result ingestion continues to work. Methods cover
supported state writes, targets, forces, and material or mass properties. Data
benchmarks cover backend-supported cached and derived properties.

Equivalent sets exist under:

* ``source/isaaclab_physx/benchmark/assets/``
* ``source/isaaclab_newton/benchmark/assets/``
* ``source/isaaclab_ov/benchmark/assets/``

.. list-table::
   :header-rows: 1
   :widths: 34 22 22 22

   * - Benchmark file
     - PhysX
     - Newton
     - OVPhysX
   * - ``benchmark_articulation.py``
     - Method + data
     - Method + data
     - Method + data
   * - ``benchmark_rigid_object.py``
     - Method + data
     - Method + data
     - Method + data
   * - ``benchmark_rigid_object_collection.py``
     - Method + data
     - Method + data
     - Method + data

Use the same file, dimensions, and mode when comparing backends or commits.

For example, run the complete PhysX articulation workload:

.. code-block:: bash

   ./isaaclab.sh microbenchmark --component articulation physics=physx \
       --num_instances 4096 \
       --num_bodies 12 \
       --num_joints 11 \
       --warmup_steps 10 \
       --num_iterations 1000 \
       --mode all \
       --backend json \
       --output_dir results/physx_articulation

Use ``rigid_object`` or ``rigid_object_collection`` for the other asset
components. Other backends use these commands:

.. dropdown:: Additional asset backend commands

   .. code-block:: bash

      ./isaaclab.sh microbenchmark --component articulation physics=newton_mjwarp \
          --num_instances 4096 --warmup_steps 10 --num_iterations 1000

   .. code-block:: bash

      ./isaaclab.sh microbenchmark --component articulation physics=newton_kamino \
          --num_instances 4096 --warmup_steps 10 --num_iterations 1000

   .. code-block:: bash

      ./isaaclab.sh microbenchmark --component articulation physics=ovphysx \
          --num_instances 4096 --warmup_steps 10 --num_iterations 1000

Run a sensor benchmark
----------------------

Sensor benchmarks build a live scene and exercise production sensors. The
defaults reduce timing noise: 4096 environments, 50 warm-up updates, and 500
timed updates. The benchmark steps the selected backend to create fresh source
data. It does not time that physics step.

For example, run the complete PhysX contact-sensor workload:

.. code-block:: bash

   ./isaaclab.sh microbenchmark --component contact_sensor physics=physx \
       --num_envs 4096 --warmup_steps 50 --num_steps 500 \
       --decimation 4 --history_length 0

The contact commands below use backend-specific cadence controls. Other
components are ``frame_transformer``, ``imu``, ``pva``, ``joint_wrench``, and
``ray_caster``.

.. dropdown:: Additional sensor backend and component commands

   .. code-block:: bash

      ./isaaclab.sh microbenchmark --component contact_sensor physics=newton_mjwarp \
          --num_envs 4096 --warmup_steps 50 --num_steps 500 \
          --decimation 4 --history_length 0

   .. code-block:: bash

      ./isaaclab.sh microbenchmark --component contact_sensor physics=ovphysx \
          --num_envs 4096 --warmup_steps 50 --num_steps 500

   .. code-block:: bash

      ./isaaclab.sh microbenchmark --component frame_transformer physics=newton_mjwarp \
          --num_envs 4096 --num_target_frames 4 --warmup_steps 50 --num_steps 500

   .. code-block:: bash

      ./isaaclab.sh microbenchmark --component imu physics=newton_kamino \
          --num_envs 4096 --warmup_steps 50 --num_steps 500

   .. code-block:: bash

      ./isaaclab.sh microbenchmark --component pva physics=physx \
          --num_envs 4096 --warmup_steps 50 --num_steps 500

   .. code-block:: bash

      ./isaaclab.sh microbenchmark --component joint_wrench physics=ovphysx \
          --num_envs 4096 --warmup_steps 50 --num_steps 500 \
          --benchmark_formatter summary --output_path results/sensors

   .. code-block:: bash

      ./isaaclab.sh microbenchmark --component ray_caster physics=newton_mjwarp \
          --num_envs 4096 --grid_size 1.0 --grid_resolution 0.25 \
          --warmup_steps 50 --num_steps 500

By default, ray-caster commands run a plane workload and a deterministic,
seeded rough-terrain workload. They report ``plane_sensor_update`` and
``rough_sensor_update`` separately. Each workload has matching observer and
validation phases. Pass ``--terrain plane`` or ``--terrain rough`` to run only
one workload.

Change the workload
-------------------

Asset arguments
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 18 57

   * - Argument
     - Default
     - Meaning
   * - ``--num_iterations``
     - 1000
     - Timed calls per method or property
   * - ``--warmup_steps``
     - 10
     - Untimed calls that compile code and warm caches
   * - ``--num_instances``
     - 4096
     - Asset instances represented by the mock view
   * - ``--mode``
     - ``all``
     - Input or index modes to run for method benchmarks
   * - ``--backend``
     - ``json``
     - Output formatter: ``json``, ``osmo``, ``omniperf``, or ``summary``
   * - ``--output_dir``
     - current directory
     - Directory for timestamped method and data result files
   * - ``--no_shape_checks``
     - false
     - Disable method input shape checks when supported

Asset-specific dimensions include ``--num_bodies`` and ``--num_joints``.
Defaults differ by file. Use ``--help`` before creating a comparison command.

Asset input modes
~~~~~~~~~~~~~~~~~

To isolate one selector representation, name its exact mode:

.. code-block:: bash

   ./isaaclab.sh microbenchmark --component articulation physics=physx \
       --mode torch_tensor_int64

Methods indexed only by ``env_ids`` support five selector modes:

``torch_list``
   Pass environment IDs as Python lists. This includes list-to-tensor
   conversion and represents common convenience-API usage.

``torch_tensor_int32`` and ``torch_tensor_int64``
   Pass pre-allocated Torch tensors using the corresponding signed index width.

``warp_int32`` and ``warp_int64``
   Pass pre-allocated Warp arrays using the corresponding signed index width.

Writers with ``joint_ids`` or ``body_ids`` support the same-type modes above
plus four mixed-width modes:

``torch_tensor_int32_int64`` and ``torch_tensor_int64_int32``
   Pass Torch environment and item selectors with the widths named in order.

``warp_int32_int64`` and ``warp_int64_int32``
   Pass Warp environment and item selectors with the widths named in order.

Supported Newton and OVPhysX mask APIs are benchmarked separately with
pre-allocated Warp boolean masks.

Not every backend or method supports every mode. Compare a mode only when it has
the same meaning on both sides.

Articulation finder phases
~~~~~~~~~~~~~~~~~~~~~~~~~~

Articulation method artifacts also contain actual ``find_bodies`` and
``find_joints`` workloads:

* ``default`` measures the legacy list return.
* ``proxy_cold`` clears the asset-local selector cache before each call, outside
  the timer, so the measured finder call includes creation of the device-local
  proxy selector.
* ``proxy_cached`` warms the cache during preflight and measures repeated lookup
  of the same proxy selector.

Run timing comparisons only while the selected GPU is idle. Another process
using the device can dominate microsecond-scale differences. Do not present
correctness-only CPU runs or results from a busy GPU as speed comparisons.

Sensor arguments
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 27 18 55

   * - Argument
     - Default
     - Meaning
   * - ``--num_envs``
     - 4096
     - Number of live sensor instances
   * - ``--num_steps``
     - 500
     - Timed sensor updates or contact cadences
   * - ``--warmup_steps``
     - 50
     - Untimed simulation and sensor updates before measurement
   * - ``--device``
     - ``cuda:0``
     - Simulation and sensor device
   * - ``--benchmark_formatter``
     - ``summary``
     - Output formatter: ``summary``, ``json``, ``osmo``, or ``omniperf``
   * - ``--output_path``
     - current directory
     - Directory for timestamped result files
   * - ``--label``
     - ``current``
     - Run label stored in metadata by scripts that support it
   * - ``--sensor``
     - required
     - Select ``imu`` or ``pva`` in ``benchmark_imu_pva.py``
   * - ``--num_target_frames``
     - 4
     - Target frames per environment in the frame-transformer workload
   * - ``--grid_size``
     - 1.0
     - Ray-grid width and length [m]
   * - ``--terrain``
     - ``all``
     - Select both ray-caster terrain workloads, ``plane``, or ``rough``
   * - ``--grid_resolution``
     - 0.25
     - Ray-grid spacing [m]

The default sensor ``summary`` formatter prints a terminal report and writes
JSON. Add ``--output_path results/sensors`` to keep artifacts outside the
repository root. ``--benchmark_formatter json`` writes JSON without the terminal
summary. Use ``osmo`` or ``omniperf`` to select their ingestion formats. PhysX
scripts also expose ``--disable_graph`` or ``--disable_recorded_launch`` where
applicable. These diagnostic controls compare production cached or graph paths
with eager launches. Leave them disabled when measuring default production
behavior.

Understand the timing boundary
------------------------------

Sensor benchmarks use this timing boundary:

.. code-block:: text

   sim.step() [untimed]
       -> synchronize [exclude earlier device work]
       -> start clock
       -> sensor operation [record host-return time]
       -> synchronize [wait for submitted device work]
       -> stop clock
       -> validate output [untimed]

The shared :func:`~isaaclab.benchmark.measure_latency` helper enforces both
synchronization boundaries and returns host-submission and
synchronized-completion times. The first synchronization keeps pending
simulation, policy, or unrelated kernels out of the sensor sample. The runner
measures the same synchronized no-op boundary separately to show observer cost.
Asset benchmarks instead time each method or property call after an untimed
warm-up.

Read the result
---------------

Asset scripts print the mean and standard deviation for every method or mode
pair in microseconds. They also print mode comparisons when applicable. Each
script writes a timestamped JSON file to ``--output_dir``. The file contains the
configuration, hardware and software metadata, phase names, and measurements.
The script prints the output path when it finishes. Articulation method artifacts
also contain the finder and raw index-kernel phases described above.

.. dropdown:: Illustrative sensor terminal summary (not reference performance)

   .. code-block:: text

      Results written to: results/sensors/newton_joint_wrench_sensor_2026-07-20_16-09-38.json
      +------------------------------------------------------------------------------------+
      |                                   Summary Report                                   |
      +------------------------------------------------------------------------------------+
      | workflow_name: newton_joint_wrench_sensor                                         |
      | num_envs: 4096                                                                    |
      +------------------------------------------------------------------------------------+
      | Phase: sensor_update                                                              |
      | Synchronized Completion: 0.120 ms                                                 |
      | Synchronized Completion p50: 0.118 ms                                             |
      | Synchronized Completion p95: 0.126 ms                                             |
      | Host Submission: 0.115 ms                                                         |
      | Host Submission p50: 0.113 ms                                                     |
      | Host Submission p95: 0.122 ms                                                     |
      +------------------------------------------------------------------------------------+
      | Phase: observer                                                                   |
      | Synchronized Observer Floor: 0.002 ms                                             |
      +------------------------------------------------------------------------------------+
      | Phase: validation                                                                 |
      | Finite Wrenches: 8192 count                                                       |
      | Nonzero Wrenches: 8192 count                                                      |
      +------------------------------------------------------------------------------------+

.. dropdown:: Illustrative JSON measurement (not reference performance)

   .. code-block:: json

      {
          "name": "newton_joint_wrench_sensor sensor_update Synchronized Completion",
          "mean": 0.038223,
          "std": 0.00010748023074035611,
          "n": 2,
          "unit": "ms",
          "type": "statistical"
      }

Generated sensor artifacts contain ``mean``, sample ``std``, ``n``, and
``unit`` for each statistical measurement. The p50 and p95 values are separate
measurements in the same phase.

``Synchronized Completion``
   Wall-clock latency from immediately before the sensor operation until all
   work it submitted completes. It is the primary comparison metric.

``Host Submission``
   Host time until the operation returns, before post-boundary synchronization.
   It measures enqueue and dispatch cost, not GPU execution.

``Synchronized Observer Floor``
   Cost of the same synchronized timing boundary around a no-op. It quantifies
   measurement overhead and is never subtracted automatically. Contact cadence
   uses the same ``decimation + 1`` boundaries as its sensor sample. Ray-caster
   benchmarks report a matching observer phase for every selected terrain.

``p50`` and ``p95``
   Median and 95th-percentile latency within one process. They expose jitter a
   mean can hide. Use JSON ``std`` for within-process variation.

``Synchronized Native Read``
   An OVPhysX phase for an isolated blocking backend read. A missing phase means
   that no equivalent read is exposed. It does not mean the read costs zero.

``Estimated Synchronized Non-read Time``
   The OVPhysX full synchronized-update mean minus the native-read mean. The two
   phases are sampled separately. Noise can dominate the estimate when their
   values are close. This is not direct kernel timing.

``validation``
   Counts demonstrating expected contacts, finite frames, sensor outputs,
   nonzero wrenches, or ray hits. Invalid output exits with an error rather than
   producing a valid-looking result. Ray-caster validation is terrain-specific.
   It checks plane hits against z=0. For rough terrain, it reports the finite
   hit-height range.

Compare runs
------------

For a performance claim:

1. Use the same workstation, GPU and CPU conditions, software environment,
   device, benchmark file, dimensions, warm-up count, and timed count. The CPU
   model, frequency, and load affect Python, dispatch, and synchronization costs.
   This remains true when the measured tensors are on the GPU. Compare the same
   ray-caster terrain phase.
2. Run baseline and candidate configurations in separate clean processes.
3. Use at least three repetitions per configuration and report the mean plus
   between-run standard deviation.
4. Check validation output and retain the raw JSON artifacts.
5. Compare the same metric: not asset microseconds with sensor milliseconds,
   host submission with synchronized latency, or sensor latency with environment
   FPS.
6. Treat startup separately. Compilation, scene creation, physics
   initialization, and CUDA graph construction are outside reported sensor
   update latency but affect total command duration.

Published Isaac Lab performance comparisons must be collected on the project
designated benchmark workstation with complete hardware and run provenance.
Local runs are appropriate for correctness checks and investigation, but must
not be presented as official reference numbers.

.. important::

   Contact protocols are not yet identical across backends. PhysX and Newton
   measure a configurable ``--decimation`` physics-step cadence plus a data
   read. OVPhysX measures one forced update after every physics step. Use contact
   results for within-backend regressions unless the protocols are aligned.

Add a benchmark
---------------

Add an asset case
~~~~~~~~~~~~~~~~~

Define shared methods with
:class:`~isaaclab.benchmark.asset_suites.AssetMethodSpec` and shared data
properties with :class:`~isaaclab.benchmark.asset_suites.AssetPropertySpec`.
Declare shared property prerequisites through ``AssetPropertySpec.dependencies``.
Use adapter ``property_dependency_overrides`` only when a backend needs
different dependencies. Keep backend-specific target construction, refresh
behavior, capabilities, and generator overrides in the backend adapter.
Allocate inputs before the timed call. Keep equivalent backend behavior aligned
where the API is shared. Do not register a mode or property that a backend
cannot implement meaningfully.

Add a sensor workload
~~~~~~~~~~~~~~~~~~~~~

1. Build the smallest live scene that exercises the production sensor path.
2. Warm simulation and sensor updates before collecting samples.
3. Keep ``sim.step()`` and validation untimed.
4. Collect samples with :func:`~isaaclab.benchmark.measure_latency`. Do not
   duplicate clock, synchronization, percentile, or unit-conversion logic.
5. Publish samples through
   :class:`~isaaclab.benchmark.LatencyBenchmarkRunner` with a matched
   synchronized observer floor.
6. Add workload dimensions and modes as metadata, and validation values as
   measurements.
7. Fail when output shapes, finite values, or physical signals are invalid.
8. Document each backend-specific timing phase or protocol difference.

Troubleshooting
---------------

Import or backend errors
~~~~~~~~~~~~~~~~~~~~~~~~

Confirm the backend is installed in the Python environment selected by
``./isaaclab.sh``. OVPhysX requires its optional runtime wheel. PhysX sensor
benchmarks require Isaac Sim.

CUDA out of memory
~~~~~~~~~~~~~~~~~~

Reduce ``--num_instances`` for assets or ``--num_envs`` for sensors. Record the
reduced size because latency scaling changes with workload size.

Slow first process
~~~~~~~~~~~~~~~~~~

The command can compile Warp kernels, build a scene, initialize physics, or
capture CUDA graphs before measurement. Warm-up excludes this work from reported
operation latency, but not total command duration.

Noisy results
~~~~~~~~~~~~~

Ensure no other GPU workload is active. Increase timed iterations, repeat in
independent processes, and report between-run variation. A difference smaller
than normal variation is not evidence of a regression or improvement.
