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
     - Yes
     - Yes
     - Yes
   * - ``benchmark_articulation_data.py``
     - Yes
     - Yes
     - Yes
   * - ``benchmark_rigid_object.py``
     - Yes
     - Yes
     - Yes
   * - ``benchmark_rigid_object_data.py``
     - Yes
     - Yes
     - Yes
   * - ``benchmark_rigid_object_collection.py``
     - Yes
     - Yes
     - Yes
   * - ``benchmark_rigid_object_collection_data.py``
     - Yes
     - Yes
     - Yes

Method benchmarks cover state writes, targets, forces, and material or mass
properties supported by the asset. Data benchmarks time backend-supported
cached and derived properties. Use the same file, dimensions, and mode when
comparing backends or commits.

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
     - Launch Isaac Sim and use mocked PhysX views
     - Launch Isaac Sim and build a live PhysX scene
   * - Newton
     - Launch Isaac Sim and use mocked Newton views
     - Run kitless with the installed Newton runtime
   * - OVPhysX
     - Run kitless; require the optional ``ovphysx`` runtime wheel
     - Run kitless; require the optional ``ovphysx`` runtime wheel

Use a CUDA device for representative GPU numbers. CPU execution is useful for
correctness or profiling, but is a different workload and must not be mixed with
CUDA results.

Running Asset Benchmarks
------------------------

Choose a backend by changing the package directory:

.. code-block:: bash

   ./isaaclab.sh -p source/isaaclab_physx/benchmark/assets/benchmark_articulation.py \
       --num_instances 4096 \
       --num_bodies 12 \
       --num_joints 11 \
       --warmup_steps 10 \
       --num_iterations 1000 \
       --mode all \
       --backend json \
       --output_dir results/physx_articulation

   ./isaaclab.sh -p source/isaaclab_newton/benchmark/assets/benchmark_articulation.py \
       --num_instances 4096 --warmup_steps 10 --num_iterations 1000

   ./isaaclab.sh -p source/isaaclab_ovphysx/benchmark/assets/benchmark_articulation.py \
       --num_instances 4096 --warmup_steps 10 --num_iterations 1000

Replace ``benchmark_articulation.py`` with the corresponding rigid-object,
collection, or ``*_data.py`` file to measure another API surface.

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
     - Output formatter: ``json``, ``osmo``, or ``omniperf``
   * - ``--output_dir``
     - current directory
     - Directory for the timestamped result file
   * - ``--no_shape_checks``
     - false
     - Disable method input shape checks when supported

Asset-specific dimensions include ``--num_bodies`` and ``--num_joints``.
Defaults differ by file; use ``--help`` before creating a comparison command.

Asset Input Modes
~~~~~~~~~~~~~~~~~

``torch_list``
   Pass selection IDs as Python lists. This includes list-to-tensor conversion
   and represents common convenience-API usage.

``torch_tensor``
   Pass pre-allocated Torch tensors. This removes repeated list conversion and
   represents the optimized Torch path.

``warp_mask``
   Pass pre-allocated Warp boolean masks. This mode is available for supported
   Newton and OVPhysX mask APIs.

Not every backend or method supports every mode. Compare a mode only when it has
the same meaning on both sides.

Running Sensor Benchmarks
-------------------------

Common sensor defaults are substantial enough to amortize timing noise: 4096
environments, 50 warm-up updates, and 500 timed updates. Override them for a
quick smoke test or scaling study.

Contact examples are backend-specific because PhysX and Newton expose a cadence
workload with decimation and history controls:

.. code-block:: bash

   ./isaaclab.sh -p source/isaaclab_physx/benchmark/sensors/benchmark_contact_sensor.py \
       --num_envs 4096 --warmup_steps 50 --num_steps 500 \
       --decimation 4 --history_length 0

   ./isaaclab.sh -p source/isaaclab_newton/benchmark/sensors/benchmark_contact_sensor.py \
       --num_envs 4096 --warmup_steps 50 --num_steps 500 \
       --decimation 4 --history_length 0

   ./isaaclab.sh -p source/isaaclab_ovphysx/benchmark/sensors/benchmark_contact_sensor.py \
       --num_envs 4096 --warmup_steps 50 --num_steps 500

For other sensors, set ``BACKEND`` to ``physx``, ``newton``, or ``ovphysx``:

.. code-block:: bash

   BACKEND=newton

   ./isaaclab.sh -p source/isaaclab_${BACKEND}/benchmark/sensors/benchmark_frame_transformer.py \
       --num_envs 4096 --num_target_frames 4 --warmup_steps 50 --num_steps 500

   ./isaaclab.sh -p source/isaaclab_${BACKEND}/benchmark/sensors/benchmark_imu_pva.py \
       --sensor imu --num_envs 4096 --warmup_steps 50 --num_steps 500

   ./isaaclab.sh -p source/isaaclab_${BACKEND}/benchmark/sensors/benchmark_imu_pva.py \
       --sensor pva --num_envs 4096 --warmup_steps 50 --num_steps 500

   ./isaaclab.sh -p source/isaaclab_${BACKEND}/benchmark/sensors/benchmark_joint_wrench.py \
       --num_envs 4096 --warmup_steps 50 --num_steps 500

   ./isaaclab.sh -p source/isaaclab_${BACKEND}/benchmark/sensors/benchmark_ray_caster.py \
       --num_envs 4096 --grid_size 1.0 --grid_resolution 0.25 \
       --warmup_steps 50 --num_steps 500

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
   * - ``--label``
     - ``current``
     - Label printed by scripts that support it
   * - ``--sensor``
     - required
     - Select ``imu`` or ``pva`` in ``benchmark_imu_pva.py``
   * - ``--num_target_frames``
     - 4
     - Target frames per environment in the frame-transformer workload
   * - ``--grid_size``
     - 1.0
     - Ray-grid width and length [m]
   * - ``--grid_resolution``
     - 0.25
     - Ray-grid spacing [m]

PhysX scripts also expose ``--disable_graph`` or
``--disable_recorded_launch`` where applicable. These are diagnostic controls
for comparing production cached/graph paths with eager launches; leave them
disabled when measuring default production behavior.

Understanding the Outputs
-------------------------

Asset Output
~~~~~~~~~~~~

Asset scripts print mean and standard deviation for every method/mode pair in
microseconds, followed by mode comparisons when applicable:

.. code-block:: text

   [1/24] [TORCH_LIST] write_root_state_to_sim... 132.02 +/- 6.79 us
   [1/24] [TORCH_TENSOR] write_root_state_to_sim... 65.44 +/- 3.06 us

They also write a timestamped JSON file to ``--output_dir`` by default. It
contains benchmark configuration, hardware/software metadata, phase names, and
recorded measurements. Select ``--backend osmo`` or ``--backend omniperf`` for
those ingestion formats. The script prints the exact output path at completion.

Sensor Output
~~~~~~~~~~~~~

Sensor scripts currently report human-readable statistics to standard output:

.. code-block:: text

   synchronized mean      : 0.041 ms
   synchronized p50       : 0.040 ms
   synchronized p95       : 0.043 ms
   submission mean        : 0.037 ms
   finite target frames   : 16384 / 16384

``synchronized`` latency
   Wall-clock latency from immediately before ``sensor.update()`` until all
   submitted device work completes. A device synchronization before the timer
   prevents earlier simulation or policy kernels from being charged to the
   sensor.

``submission`` latency
   Host time spent issuing ``sensor.update()`` before post-update device
   synchronization. This is enqueue/dispatch cost, not pure GPU execution time.

``p50`` and ``p95``
   Median and 95th-percentile latency within one process. They expose jitter
   that a mean can hide.
   Most workloads report both; the PhysX/Newton contact cadence reports p50 and
   minimum instead.

``read-only`` latency
   OVPhysX scripts also time the blocking backend fetch without sensor Warp
   processing. Newton sensor inputs are already Warp arrays, and PhysX scripts
   do not consistently expose an equivalent phase.

``implied kernel tail``
   OVPhysX reports ``synchronized mean - read-only mean`` as an estimate of
   remaining processing. It derives from separately measured phases, so treat
   small differences as noise rather than direct kernel timing.

Validation counts
   Final lines prove that the workload produced expected contacts, finite
   frames, sensor outputs, nonzero wrenches, or ray hits. Never use a result
   whose validation failed.

Sensor scripts do not currently write a structured result file. Redirect stdout
to retain raw runs:

.. code-block:: bash

   ./isaaclab.sh -p source/isaaclab_newton/benchmark/sensors/benchmark_ray_caster.py \
       --num_envs 4096 --warmup_steps 50 --num_steps 500 \
       > results/newton_ray_caster_run_1.log 2>&1

Making Fair Comparisons
-----------------------

For a performance claim:

1. Use the same workstation, GPU conditions, environment, device, benchmark
   file, dimensions, warm-up count, and timed count.
2. Run baseline and candidate configurations from separate clean processes.
3. Use at least three repetitions per configuration and report the mean plus
   between-run standard deviation.
4. Check validation output and retain raw logs or JSON files.
5. Compare the same metric. Do not compare asset microseconds with sensor
   milliseconds, submission with synchronized latency, or sensor latency with
   environment FPS.
6. Treat startup separately. Compilation, scene creation, physics initialization,
   and CUDA graph construction occur outside reported sensor update latency but
   still affect total command duration.

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
       -> synchronize [timing boundary]
       -> sensor.update() [timed]
       -> synchronize [timing boundary]
       -> validate output [untimed]

The pre-boundary synchronization is intentional. GPU work is asynchronous; if
the timer started while earlier work was pending, synchronized sensor latency
could incorrectly include that work.

Adding New Benchmarks
---------------------

Adding an Asset Method or Property
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add input generators and a
:class:`~isaaclab.test.benchmark.MethodBenchmarkDefinition` to the corresponding
backend script. Allocate inputs before the timed call. For a data property, list
prerequisite properties through ``derived_from`` so dependencies are populated
before timing.

Keep equivalent backends aligned where the API is shared, but do not register a
mode or property that a backend cannot implement meaningfully.

Adding a Sensor Workload
~~~~~~~~~~~~~~~~~~~~~~~~

1. Build the smallest live scene that exercises the production sensor path.
2. Warm simulation and sensor updates before collecting samples.
3. Keep ``sim.step()`` and validation outside the timed region.
4. Synchronize the device immediately before and after ``sensor.update()``.
5. Report synchronized and submission distributions with clear units.
6. Fail when output shapes, finite values, or physical signals are invalid.
7. Document every backend-specific timing phase or protocol difference.

Troubleshooting
---------------

Import or Backend Errors
~~~~~~~~~~~~~~~~~~~~~~~~

Confirm the backend is installed in the Python environment selected by
``./isaaclab.sh -p``. OVPhysX requires its optional runtime wheel. PhysX
benchmarks require Isaac Sim.

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
