.. _testing_micro_benchmarks:

Micro-Benchmarks for Performance Testing
========================================

Isaac Lab provides micro-benchmarking tools to measure the performance of asset
setter/writer methods and data property accessors without requiring Isaac Sim.

Overview
--------

The benchmarks use **mock interfaces** to simulate PhysX views, allowing performance
measurement of Python-level overhead in isolation. This is useful for:

- Comparing list vs tensor index performance
- Identifying bottlenecks in hot code paths
- Tracking performance regressions
- Optimizing custom methods

Quick Start
-----------

Run benchmarks using the Isaac Lab launcher:

.. code-block:: bash

   # Run Articulation method benchmarks
   ./isaaclab.sh -p source/isaaclab_physx/benchmark/assets/benchmark_articulation.py

   # With custom parameters
   ./isaaclab.sh -p source/isaaclab_physx/benchmark/assets/benchmark_articulation.py \
       --num_iterations 1000 \
       --num_instances 64 \
       --num_bodies 5 \
       --num_joints 4

Available Benchmarks
--------------------

Asset Method Benchmarks
~~~~~~~~~~~~~~~~~~~~~~~

These benchmark setter and writer methods on asset classes:

.. list-table::
   :header-rows: 1
   :widths: 35 25 40

   * - Benchmark File
     - Asset Class
     - Methods Covered
   * - ``benchmark_articulation.py``
     - ``Articulation``
     - 24 methods (root/joint state, mass props, forces)
   * - ``benchmark_rigid_object.py``
     - ``RigidObject``
     - 13 methods (root state, mass props, forces)
   * - ``benchmark_rigid_object_collection.py``
     - ``RigidObjectCollection``
     - 13 methods (body state, mass props, forces)

Data Property Benchmarks
~~~~~~~~~~~~~~~~~~~~~~~~

These benchmark property accessors on data classes:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Benchmark File
     - Data Class
     - Properties
   * - ``benchmark_articulation_data.py``
     - ``ArticulationData``
     - 59 properties
   * - ``benchmark_rigid_object_data.py``
     - ``RigidObjectData``
     - 40 properties
   * - ``benchmark_rigid_object_collection_data.py``
     - ``RigidObjectCollectionData``
     - 40 properties

All benchmarks are located in ``source/isaaclab_physx/benchmark/assets/``.

Command Line Arguments
----------------------

Common Arguments
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Argument
     - Default
     - Description
   * - ``--num_iterations``
     - 1000
     - Number of timed iterations
   * - ``--warmup_steps``
     - 10
     - Warmup iterations (not timed)
   * - ``--num_instances``
     - 4096
     - Number of asset instances
   * - ``--device``
     - ``cuda:0``
     - Device for tensors
   * - ``--mode``
     - ``all``
     - ``all``, ``torch_list``, or ``torch_tensor``
   * - ``--output``
     - auto
     - Output JSON filename
   * - ``--no_csv``
     - false
     - Disable CSV output

Asset-Specific Arguments
~~~~~~~~~~~~~~~~~~~~~~~~

**Articulation benchmarks:**

- ``--num_bodies``: Number of links (default: 13)
- ``--num_joints``: Number of DOFs (default: 12)

**RigidObjectCollection benchmarks:**

- ``--num_bodies``: Number of bodies in collection (default: 5)

Benchmark Modes
---------------

Each method is benchmarked under two input scenarios:

**torch_list**
   Environment/body IDs passed as Python lists. Measures the overhead of
   list-to-tensor conversion, which is common in user code.

**torch_tensor**
   Environment/body IDs passed as pre-allocated tensors. Represents the
   optimal baseline with minimal overhead.

Example output:

.. code-block:: text

   [1/24] [TORCH_LIST] write_root_state_to_sim... 132.02 ± 6.79 µs
   [1/24] [TORCH_TENSOR] write_root_state_to_sim... 65.44 ± 3.06 µs

The comparison shows tensor indices are ~2x faster than list indices.

Output Format
-------------

Console Output
~~~~~~~~~~~~~~

.. code-block:: text

   Benchmarking Articulation (PhysX) with 64 instances, 5 bodies, 4 joints...
   Device: cuda:0
   Iterations: 100, Warmup: 10

   Benchmarking 24 methods...
   [1/24] [TORCH_LIST] write_root_state_to_sim... 132.02 ± 6.79 µs
   [1/24] [TORCH_TENSOR] write_root_state_to_sim... 65.44 ± 3.06 µs
   ...

   ================================================================================
   COMPARISON: Torch_list vs Torch_tensor
   ================================================================================
   Method Name                         Torch_list   Torch_tensor   Speedup
   ------------------------------------------------------------------------
   write_root_state_to_sim               132.02        65.44        2.02x

Export Files
~~~~~~~~~~~~

Results are automatically exported to:

- ``{benchmark_name}_{timestamp}.json`` - Full results with hardware info
- ``{benchmark_name}_{timestamp}.csv`` - Tabular results for analysis

JSON Structure
~~~~~~~~~~~~~~

.. code-block:: json

   {
     "config": {
       "num_iterations": 100,
       "num_instances": 64,
       "device": "cuda:0"
     },
     "hardware": {
       "cpu": "Intel Core i9-13950HX",
       "gpu": "NVIDIA RTX 5000",
       "pytorch": "2.7.0",
       "cuda": "12.8"
     },
     "results": [
       {
         "name": "write_root_state_to_sim",
         "mode": "torch_list",
         "mean_us": 132.02,
         "std_us": 6.79,
         "iterations": 100
       }
     ]
   }

Architecture
------------

The benchmarks use mock interfaces to simulate PhysX views without Isaac Sim:

.. code-block:: text

   ┌─────────────────────┐     ┌──────────────────────┐
   │   Asset Class       │────>│   MockArticulationView│
   │   (Articulation)    │     │   (mock_interfaces)   │
   └─────────────────────┘     └──────────────────────┘
            │
            v
   ┌─────────────────────┐
   │   Benchmark         │
   │   Framework         │
   └─────────────────────┘

Key Components
~~~~~~~~~~~~~~

1. **Mock Views** (``isaaclab_physx/test/mock_interfaces/``)

   - ``MockArticulationView`` - Mimics PhysX ArticulationView
   - ``MockRigidBodyView`` - Mimics PhysX RigidBodyView

2. **Benchmark Utilities** (``isaaclab/test/benchmark/``)

   - ``BenchmarkConfig`` - Configuration dataclass
   - ``MethodBenchmark`` - Benchmark definition
   - ``benchmark_method()`` - Core timing function
   - Export utilities for JSON/CSV

3. **Module Mocking**

   Each benchmark file mocks Isaac Sim dependencies (``isaacsim``, ``omni``, ``pxr``)
   to allow the asset classes to be instantiated without simulation.

Adding New Benchmarks
---------------------

Adding a Method Benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~

1. Create input generator functions:

.. code-block:: python

   def gen_my_method_torch_list(config: BenchmarkConfig) -> dict:
       return {
           "param1": torch.rand(config.num_instances, 3, device=config.device),
           "env_ids": list(range(config.num_instances)),
       }

   def gen_my_method_torch_tensor(config: BenchmarkConfig) -> dict:
       return {
           "param1": torch.rand(config.num_instances, 3, device=config.device),
           "env_ids": make_tensor_env_ids(config.num_instances, config.device),
       }

2. Add to the ``BENCHMARKS`` list:

.. code-block:: python

   MethodBenchmark(
       name="my_method",
       method_name="my_method",
       input_generators={
           "torch_list": gen_my_method_torch_list,
           "torch_tensor": gen_my_method_torch_tensor,
       },
       category="my_category",
   ),

Adding a Property Benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For data class properties, add to the ``PROPERTIES`` list:

.. code-block:: python

   ("my_property", {"derived_from": ["dependency1", "dependency2"]}),

The ``derived_from`` key indicates dependencies that should be pre-computed
before timing the property access.

Performance Tips
----------------

Based on benchmark results:

1. **Use tensor indices** instead of lists for 30-50% speedup
2. **Pre-allocate index tensors** and reuse them across calls
3. **Batch operations** where possible (e.g., set all joint positions at once)
4. **Mass properties are CPU-bound** - PhysX requires CPU tensors for these

Example optimization:

.. code-block:: python

   # Slow: Create new list each call
   for _ in range(1000):
       robot.write_joint_state_to_sim(state, env_ids=list(range(64)))

   # Fast: Pre-allocate tensor and reuse
   env_ids = torch.arange(64, device="cuda:0")
   for _ in range(1000):
       robot.write_joint_state_to_sim(state, env_ids=env_ids)

Troubleshooting
---------------

Import Errors
~~~~~~~~~~~~~

Ensure you're running through ``isaaclab.sh``:

.. code-block:: bash

   ./isaaclab.sh -p source/isaaclab_physx/benchmark/assets/benchmark_articulation.py

CUDA Out of Memory
~~~~~~~~~~~~~~~~~~

Reduce ``--num_instances``:

.. code-block:: bash

   ./isaaclab.sh -p ... --num_instances 1024

Slow First Run
~~~~~~~~~~~~~~

The first run compiles Warp kernels. Subsequent runs will be faster.
