.. _testing_micro_benchmarks:

Micro-Benchmarks for Performance Testing
========================================

Micro-benchmarks measure a narrow Isaac Lab operation while excluding as much
unrelated work as possible. They answer questions such as:

* Did an asset API change make a property read or state write slower?
* How much host conversion or backend-binding overhead does an API call add?
* How expensive is a production sensor update after physics has completed?
* Is a regression in Isaac Lab code, a backend read, or the workload around it?

Isolation makes these benchmarks quick to repeat and easier to diagnose than a
full training run. It also limits what they prove: micro-benchmarks do **not**
predict end-to-end environment or training throughput. Use
:ref:`testing_benchmarks` when the question includes environment logic, policy
inference, learning, or application startup.

.. seealso::

   For the typed Python API, formatter and recorder internals, or custom
   producers, see :ref:`testing_benchmark_framework`.

Choosing a Benchmark
--------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Question
     - Use
     - Deliberately excluded
   * - How fast is one asset method or data property?
     - Asset method/data micro-benchmark
     - Scene creation, simulation, sensors, and environment logic
   * - How fast is one production sensor update?
     - Sensor update micro-benchmark
     - Timed physics stepping, environment logic, and learning
   * - How fast does an environment step or policy train?
     - Runtime, play, or training benchmark
     - Nothing required by that workflow

Benchmark Families
------------------

Asset Method and Data Benchmarks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Asset benchmarks instantiate Isaac Lab asset classes against backend-specific
**mock views**. The mocks reproduce relevant data shapes and binding behavior
without running physics. Measurements isolate Python input handling,
tensor/index conversion, Isaac Lab method logic, backend binding calls, and
data-property computation.

Equivalent sets exist under:

* ``source/isaaclab_physx/benchmark/assets/``
* ``source/isaaclab_newton/benchmark/assets/``
* ``source/isaaclab_ovphysx/benchmark/assets/``

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

Each retained asset script runs its method phase followed by its data-property
phase. It writes two result artifacts using the historical method and data
workflow names, so existing result ingestion remains valid. Method benchmarks
cover state writes, targets, forces, and material or mass properties supported
by the asset. Data benchmarks time backend-supported cached and derived
properties. Use the same file, dimensions, and mode when comparing backends or
commits.

Sensor Update Benchmarks
~~~~~~~~~~~~~~~~~~~~~~~~

Sensor benchmarks build **live simulation scenes** and exercise production
sensor implementations. They step the selected backend so fresh source data
exists, but place ``sim.step()`` outside the timed region. Reported latency
therefore measures sensor update work rather than the solver.

Equivalent workloads exist under each backend ``benchmark/sensors`` directory:

.. list-table::
   :header-rows: 1
   :widths: 34 22 22 22

   * - Benchmark file
     - PhysX
     - Newton
     - OVPhysX
   * - ``benchmark_contact_sensor.py``
     - Yes
     - Yes
     - Yes
   * - ``benchmark_frame_transformer.py``
     - Yes
     - Yes
     - Yes
   * - ``benchmark_imu_pva.py``
     - IMU and PVA
     - IMU and PVA
     - IMU and PVA
   * - ``benchmark_joint_wrench.py``
     - Yes
     - Yes
     - Yes
   * - ``benchmark_ray_caster.py``
     - Yes
     - Yes
     - Yes

Every sensor benchmark validates output after timing. A fast run with missing
contacts, non-finite transforms, zero wrenches, or invalid ray hits fails instead
of reporting a misleading performance result.

Prerequisites
-------------

Run commands from the repository root through ``./isaaclab.sh -p``. The active
Python environment must contain the backend being measured.

.. list-table::
   :header-rows: 1
   :widths: 18 41 41

   * - Backend
     - Asset benchmarks
     - Sensor benchmarks
   * - PhysX
     - Run kitless with mocked PhysX views
     - Launch Isaac Sim and build a live PhysX scene
   * - Newton
     - Run kitless with mocked Newton views
     - Run kitless with the installed Newton runtime
   * - OVPhysX
     - Run kitless; require the optional ``ovphysx`` runtime wheel
     - Run kitless; require the optional ``ovphysx`` runtime wheel

Use a CUDA device for representative GPU numbers. CPU execution is useful for
correctness or profiling, but is a different workload and must not be mixed with
CUDA results.

Running Asset Benchmarks
------------------------

Use the top-level component command with an exact physics selector:

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

   ./isaaclab.sh microbenchmark --component articulation physics=newton_mjwarp \
       --num_instances 4096 --warmup_steps 10 --num_iterations 1000

   ./isaaclab.sh microbenchmark --component articulation physics=newton_kamino \
       --num_instances 4096 --warmup_steps 10 --num_iterations 1000

   ./isaaclab.sh microbenchmark --component articulation physics=ovphysx \
       --num_instances 4096 --warmup_steps 10 --num_iterations 1000

To isolate one item-selector representation, select its mode explicitly:

.. code-block:: bash

   ./isaaclab.sh microbenchmark --component articulation physics=physx \
       --mode torch_tensor_int64

Replace ``articulation`` with ``rigid_object`` or
``rigid_object_collection``. Each command measures both API surfaces and emits
separate method and data artifacts. The three retained scripts under each
backend's ``benchmark/assets`` directory remain available for direct execution.

Asset Arguments
~~~~~~~~~~~~~~~

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
     - Untimed calls used to compile and warm caches
   * - ``--num_instances``
     - 4096
     - Asset instances represented by the mock view
   * - ``--mode``
     - ``all``
     - Input/index modes to run; method benchmarks only
   * - ``--backend``
     - ``json``
     - Output formatter: ``json``, ``osmo``, ``omniperf``, or ``summary``
   * - ``--output_dir``
     - current directory
     - Directory for the timestamped method and data result files
   * - ``--no_shape_checks``
     - false
     - Disable method input shape checks when supported

Asset-specific dimensions include ``--num_bodies`` and ``--num_joints``.
Defaults differ by file; use ``--help`` before creating a comparison command.

Asset Input Modes
~~~~~~~~~~~~~~~~~

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

Articulation Finder Phases
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
using the device can dominate microsecond-scale differences; correctness-only
CPU runs and results collected on a busy GPU must not be presented as speed
comparisons.

Running Sensor Benchmarks
-------------------------

Common sensor defaults are substantial enough to amortize timing noise: 4096
environments, 50 warm-up updates, and 500 timed updates. Override them for a
quick smoke test or scaling study.

Select the same component across exact physics variants through the top-level
command. Contact examples differ because PhysX and Newton expose cadence
controls:

.. code-block:: bash

   ./isaaclab.sh microbenchmark --component contact_sensor physics=physx \
       --num_envs 4096 --warmup_steps 50 --num_steps 500 \
       --decimation 4 --history_length 0

   ./isaaclab.sh microbenchmark --component contact_sensor physics=newton_mjwarp \
       --num_envs 4096 --warmup_steps 50 --num_steps 500 \
       --decimation 4 --history_length 0

   ./isaaclab.sh microbenchmark --component contact_sensor physics=ovphysx \
       --num_envs 4096 --warmup_steps 50 --num_steps 500

Other component names are ``frame_transformer``, ``imu``, ``pva``,
``joint_wrench``, and ``ray_caster``:

.. code-block:: bash

   ./isaaclab.sh microbenchmark --component frame_transformer physics=newton_mjwarp \
       --num_envs 4096 --num_target_frames 4 --warmup_steps 50 --num_steps 500

   ./isaaclab.sh microbenchmark --component imu physics=newton_kamino \
       --num_envs 4096 --warmup_steps 50 --num_steps 500

   ./isaaclab.sh microbenchmark --component pva physics=physx \
       --num_envs 4096 --warmup_steps 50 --num_steps 500

   ./isaaclab.sh microbenchmark --component joint_wrench physics=ovphysx \
       --num_envs 4096 --warmup_steps 50 --num_steps 500 \
       --benchmark_formatter summary --output_path results/sensors

   ./isaaclab.sh microbenchmark --component ray_caster physics=newton_mjwarp \
       --num_envs 4096 --grid_size 1.0 --grid_resolution 0.25 \
       --warmup_steps 50 --num_steps 500

The retained scripts under each backend's ``benchmark/sensors`` directory
remain available for direct execution.

Ray-caster commands run matched plane and deterministic seeded rough-terrain
workloads by default. Results are reported separately as
``plane_sensor_update`` and ``rough_sensor_update`` phases with matching
observer and validation phases. Pass ``--terrain plane`` or ``--terrain rough``
to run only one workload.

Sensor Arguments
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

The default ``summary`` formatter prints a terminal report and writes JSON. Add
``--output_path results/sensors`` to keep artifacts outside the repository root.
PhysX scripts also expose ``--disable_graph`` or
``--disable_recorded_launch`` where applicable. These are diagnostic controls
for comparing production cached or graph paths with eager launches; leave them
disabled when measuring default production behavior.

Understanding the Outputs
-------------------------

Asset Output
~~~~~~~~~~~~

Asset scripts print mean and standard deviation for every method/mode pair in
microseconds. Articulation method artifacts additionally contain the finder and
raw index-kernel phases described above:

.. code-block:: text

   [1/30] [TORCH_LIST] write_root_state_to_sim... 132.02 +/- 6.79 us
   [1/30] [TORCH_TENSOR_INT64] write_root_state_to_sim... 65.44 +/- 3.06 us

They also write a timestamped JSON file to ``--output_dir`` by default. It
contains benchmark configuration, hardware/software metadata, phase names, and
recorded measurements. Select ``--backend osmo`` or ``--backend omniperf`` for
those ingestion formats. The script prints the exact output path at completion.

Sensor Output
~~~~~~~~~~~~~

The default ``summary`` formatter prints the aggregate result and writes the
same phases to a timestamped JSON file. An abbreviated report looks like:

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

The numbers above illustrate the output shape; they are not reference
performance values. Use the generated artifact for analysis. Its statistical
measurements contain ``mean``, sample ``std``, ``n``, and ``unit`` fields, while
p50 and p95 are separate measurements in the same phase. For example:

.. code-block:: json

   {
       "name": "newton_joint_wrench_sensor sensor_update Synchronized Completion",
       "mean": 0.038223,
       "std": 0.00010748023074035611,
       "n": 2,
       "unit": "ms",
       "type": "statistical"
   }

``Synchronized Completion``
   Wall-clock latency from immediately before the sensor operation until all
   work it submitted completes. A device synchronization before the timer
   excludes earlier simulation, policy, or unrelated kernels. The matching
   post-boundary synchronization makes this the primary comparison metric.

``Host Submission``
   Host time until the sensor operation returns, before the post-boundary
   synchronization. This measures enqueue and dispatch cost, not GPU execution.

``Synchronized Observer Floor``
   Cost of the same synchronized timing boundary around a no-op. It quantifies
   measurement overhead and is never subtracted automatically. Contact cadence
   reports an observer sample with the same ``decimation + 1`` boundaries as its
   sensor sample. Ray-caster benchmarks report a matching observer phase for
   each selected terrain workload.

``p50`` and ``p95``
   Median and 95th-percentile latency within one process. They expose jitter that
   a mean can hide. Use the JSON ``std`` to quantify within-process variation.

``Synchronized Native Read``
   OVPhysX scripts report this phase when the blocking backend read can be
   isolated. A missing phase means that backend does not expose an equivalent
   read, not that the read costs zero.

``Estimated Synchronized Non-read Time``
   OVPhysX may report full synchronized update mean minus native-read mean. It is
   derived from separately sampled phases and can be dominated by noise when the
   values are close; it is not direct kernel timing.

``validation``
   Counts prove that the workload produced expected contacts, finite frames,
   sensor outputs, nonzero wrenches, or ray hits. The script exits with an error
   instead of writing a valid-looking result when these checks fail. Ray-caster
   validation is terrain-specific: plane hits are checked against z=0, while
   rough-terrain validation reports the finite hit-height range.

Use ``--benchmark_formatter json`` for JSON without the terminal summary, or
``osmo`` and ``omniperf`` for their ingestion formats.

Making Fair Comparisons
-----------------------

For a performance claim:

1. Use the same workstation, GPU and CPU conditions, software environment,
   device, benchmark file, dimensions, warm-up count, and timed count. CPU model,
   frequency, and load still affect Python, dispatch, and synchronization costs
   when the measured tensors live on the GPU. For ray-caster results, compare
   the same plane or rough-terrain phase.
2. Run baseline and candidate configurations from separate clean processes.
3. Use at least three repetitions per configuration and report the mean plus
   between-run standard deviation.
4. Check validation output and retain the raw JSON files.
5. Compare the same metric. Do not compare asset microseconds with sensor
   milliseconds, submission with synchronized latency, or sensor latency with
   environment FPS.
6. Treat startup separately. Compilation, scene creation, physics initialization,
   and CUDA graph construction occur outside reported sensor update latency but
   still affect total command duration.

Published Isaac Lab performance comparisons must be collected on the project
designated benchmark workstation with complete hardware and run provenance.
Local runs are appropriate for correctness checks and investigation, but should
not be presented as official reference numbers.

.. important::

   Contact workloads are not yet identical across all backends. PhysX and
   Newton measure a configurable cadence of ``--decimation`` physics steps plus
   a data read. OVPhysX measures one forced update after each physics step. Use
   contact results for within-backend regressions unless protocols are aligned.

Architecture
------------

Asset benchmarks isolate method/data code with mocks:

.. code-block:: text

   generated inputs -> Isaac Lab asset method/property -> backend mock view
                    -> MethodBenchmarkRunner -> console + metrics formatter

Sensor benchmarks isolate a live sensor update from simulation:

.. code-block:: text

   sim.step() [untimed]
       -> synchronize [exclude earlier device work]
       -> start clock
       -> sensor operation [record host-return time]
       -> synchronize [wait for submitted device work]
       -> stop clock
       -> validate output [untimed]

The shared :func:`~isaaclab.benchmark.measure_latency` helper enforces both
boundaries and returns paired host-submission and synchronized-completion times.
The runner measures a synchronized no-op separately to expose observer cost.
The pre-boundary synchronization is intentional: without it, pending simulation,
policy, or unrelated kernels could be charged to the sensor sample.

Adding New Benchmarks
---------------------

Adding an Asset Method or Property
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add shared input generators and
:class:`~isaaclab.benchmark.MethodBenchmarkDefinition` entries to
``isaaclab.benchmark.asset_suites``. Put only backend-specific target
construction, refresh behavior, capabilities, and generator overrides in the
backend adapter. Allocate inputs before the timed call. For a data property,
list prerequisite properties through ``derived_from`` so dependencies are
populated before timing.

Keep equivalent backends aligned where the API is shared, but do not register a
mode or property that a backend cannot implement meaningfully.

Adding a Sensor Workload
~~~~~~~~~~~~~~~~~~~~~~~~

1. Build the smallest live scene that exercises the production sensor path.
2. Warm simulation and sensor updates before collecting samples.
3. Keep ``sim.step()`` and validation outside the timed region.
4. Collect samples with :func:`~isaaclab.benchmark.measure_latency`; do not
   duplicate clock, synchronization, percentile, or unit-conversion logic.
5. Publish the samples through
   :class:`~isaaclab.benchmark.LatencyBenchmarkRunner` and include a matched
   synchronized observer floor.
6. Add workload dimensions and modes as metadata, and validation values as
   measurements.
7. Fail when output shapes, finite values, or physical signals are invalid.
8. Document every backend-specific timing phase or protocol difference.

Troubleshooting
---------------

Import or Backend Errors
~~~~~~~~~~~~~~~~~~~~~~~~

Confirm the backend is installed in the Python environment selected by
``./isaaclab.sh -p``. OVPhysX requires its optional runtime wheel. PhysX
sensor benchmarks require Isaac Sim.

CUDA Out of Memory
~~~~~~~~~~~~~~~~~~

Reduce ``--num_instances`` for assets or ``--num_envs`` for sensors. Record the
reduced size because latency scaling changes with workload size.

Slow First Process
~~~~~~~~~~~~~~~~~~

The command may compile Warp kernels, build a scene, initialize physics, or
capture CUDA graphs before measurement. Warm-up excludes this work from reported
operation latency, but not from total command duration.

Noisy Results
~~~~~~~~~~~~~

Ensure no other GPU workload is active. Increase timed iterations, repeat the
command in independent processes, and report between-run variation. A difference
smaller than normal variation is not evidence of a regression or improvement.
