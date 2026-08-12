.. Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _developer_tools_benchmarking:

Benchmarking
============

Measure Isaac Lab workloads, compare changes, and catch performance
regressions with the benchmark CLI and Python API.

Run a benchmark
---------------

Start with a runtime benchmark. This command measures 1000 environment steps
after 50 warm-up steps and prints a summary:

.. code-block:: bash

   ./isaaclab.sh benchmark runtime \
       --task Isaac-Cartpole-Direct \
       --num_envs 4096 \
       --warmup_steps 50 \
       --num_steps 1000 \
       --benchmark_formatter summary \
       --output_path ./benchmark_results \
       physics=isaacsim_physx

Choose a workflow
-----------------

.. list-table::
   :header-rows: 1
   :widths: 38 20 42

   * - Question
     - Tool
     - Continue with
   * - How fast does an environment step?
     - ``runtime``
     - :ref:`testing_benchmarks`
   * - How fast does a trained policy run?
     - ``play``
     - :ref:`testing_benchmarks`
   * - How fast does a policy train?
     - ``training``
     - :ref:`testing_benchmarks`
   * - Where is startup time spent?
     - ``startup``
     - :ref:`testing_benchmarks`
   * - How fast is one asset or sensor operation?
     - Micro-benchmark
     - :ref:`testing_micro_benchmarks`
   * - How do I automate or extend a benchmark?
     - Python API
     - :ref:`testing_benchmark_framework`

From workload to comparison
---------------------------

.. code-block:: text

   workload -> warm-up -> measurement -> summary/schema output -> comparison

Use the same workload, hardware, software revision, and measurement mode on
both sides of a comparison. The detailed guides define each timing boundary
and the provenance required for a valid result.

Guides
------

.. grid:: 1 1 3 3
   :gutter: 2

   .. grid-item-card:: Run benchmarks
      :link: testing_benchmarks
      :link-type: ref

      Measure runtime, policy playback, training, or startup.

   .. grid-item-card:: Write micro-benchmarks
      :link: testing_micro_benchmarks
      :link-type: ref

      Isolate one asset method, data property, or sensor update.

   .. grid-item-card:: Use the benchmark API
      :link: testing_benchmark_framework
      :link-type: ref

      Run workflows from Python or add a benchmark producer.

Published results
-----------------

See :doc:`/source/overview/reinforcement-learning/performance_benchmarks` for
published reinforcement-learning performance results.

.. toctree::
   :hidden:
   :maxdepth: 1

   Run benchmarks <../testing/benchmarks>
   Write micro-benchmarks <../testing/micro_benchmarks>
   Use the benchmark API <../testing/benchmark_framework>
