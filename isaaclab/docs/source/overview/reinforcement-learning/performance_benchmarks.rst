Performance Benchmarks
======================

Isaac Lab leverages end-to-end GPU training for reinforcement learning workflows,
allowing for fast parallel training across thousands of environments.
In this section, we provide runtime performance benchmark results for reinforcement learning
training of various example environments on different GPU setups.
Multi-GPU and multi-node training performance results are also outlined.


Benchmark Results
-----------------

All benchmarking results were performed with the RL Games library with ``--headless`` flag on Ubuntu 22.04.
``Isaac-Velocity-Rough-G1-v0`` environment benchmarks were performed with the RSL RL library.


Memory Consumption
^^^^^^^^^^^^^^^^^^

+------------------------------------+----------------+-------------------+----------+-----------+
| Environment Name                   |                | # of Environments | RAM (GB) | VRAM (GB) |
+====================================+================+===================+==========+===========+
| Isaac-Cartpole-Direct-v0           | |cartpole|     | 4096              | 3.7      | 3.3       |
+------------------------------------+----------------+-------------------+----------+-----------+
| Isaac-Cartpole-RGB-Camera-Direct-v0| |cartpole-cam| | 1024              | 7.5      | 16.7      |
+------------------------------------+----------------+-------------------+----------+-----------+
| Isaac-Velocity-Rough-G1-v0         | |g1|           | 4096              | 6.5      | 6.1       |
+------------------------------------+----------------+-------------------+----------+-----------+
| Isaac-Repose-Cube-Shadow-Direct-v0 | |shadow|       | 8192              | 6.7      | 6.4       |
+------------------------------------+----------------+-------------------+----------+-----------+

.. |cartpole| image:: ../../_static/benchmarks/cartpole.jpg
    :width: 80
    :height: 45
.. |cartpole-cam| image:: ../../_static/benchmarks/cartpole_camera.jpg
    :width: 80
    :height: 45
.. |g1| image:: ../../_static/benchmarks/g1_rough.jpg
    :width: 80
    :height: 45
.. |shadow| image:: ../../_static/benchmarks/shadow.jpg
    :width: 80
    :height: 45


Single GPU - RTX 4090
^^^^^^^^^^^^^^^^^^^^^

CPU: AMD Ryzen 9 7950X 16-Core Processor

+-------------------------------------+-------------------+--------------+-------------------+--------------------+
| Environment Name                    | # of Environments | Environment  | Environment Step  | Environment Step,  |
|                                     |                   | Step FPS     | and               | Inference,         |
|                                     |                   |              | Inference FPS     | and Train FPS      |
+=====================================+===================+==============+===================+====================+
| Isaac-Cartpole-Direct-v0            | 4096              | 1100000      | 910000            | 510000             |
+-------------------------------------+-------------------+--------------+-------------------+--------------------+
| Isaac-Cartpole-RGB-Camera-Direct-v0 | 1024              | 50000        | 45000             | 32000              |
+-------------------------------------+-------------------+--------------+-------------------+--------------------+
| Isaac-Velocity-Rough-G1-v0          | 4096              | 94000        | 88000             | 82000              |
+-------------------------------------+-------------------+--------------+-------------------+--------------------+
| Isaac-Repose-Cube-Shadow-Direct-v0  | 8192              | 200000       | 190000            | 170000             |
+-------------------------------------+-------------------+--------------+-------------------+--------------------+


Single GPU - L40
^^^^^^^^^^^^^^^^

CPU: Intel(R) Xeon(R) Platinum 8362 CPU @ 2.80GHz

+-------------------------------------+-------------------+--------------+-------------------+--------------------+
| Environment Name                    | # of Environments | Environment  | Environment Step  | Environment Step,  |
|                                     |                   | Step FPS     | and               | Inference,         |
|                                     |                   |              | Inference FPS     | and Train FPS      |
+=====================================+===================+==============+===================+====================+
| Isaac-Cartpole-Direct-v0            | 4096              | 620000       | 490000            | 260000             |
+-------------------------------------+-------------------+--------------+-------------------+--------------------+
| Isaac-Cartpole-RGB-Camera-Direct-v0 | 1024              | 30000        | 28000             | 21000              |
+-------------------------------------+-------------------+--------------+-------------------+--------------------+
| Isaac-Velocity-Rough-G1-v0          | 4096              | 72000        | 64000             | 62000              |
+-------------------------------------+-------------------+--------------+-------------------+--------------------+
| Isaac-Repose-Cube-Shadow-Direct-v0  | 8192              | 170000       | 140000            | 120000             |
+-------------------------------------+-------------------+--------------+-------------------+--------------------+


Single-Node, 4 x L40 GPUs
^^^^^^^^^^^^^^^^^^^^^^^^^

CPU: Intel(R) Xeon(R) Platinum 8362 CPU @ 2.80GHz

+-------------------------------------+-------------------+--------------+-------------------+--------------------+
| Environment Name                    | # of Environments | Environment  | Environment Step  | Environment Step,  |
|                                     |                   | Step FPS     | and               | Inference,         |
|                                     |                   |              | Inference FPS     | and Train FPS      |
+=====================================+===================+==============+===================+====================+
| Isaac-Cartpole-Direct-v0            | 4096              | 2700000      | 2100000           | 950000             |
+-------------------------------------+-------------------+--------------+-------------------+--------------------+
| Isaac-Cartpole-RGB-Camera-Direct-v0 | 1024              | 130000       | 120000            | 90000              |
+-------------------------------------+-------------------+--------------+-------------------+--------------------+
| Isaac-Velocity-Rough-G1-v0          | 4096              | 290000       | 270000            | 250000             |
+-------------------------------------+-------------------+--------------+-------------------+--------------------+
| Isaac-Repose-Cube-Shadow-Direct-v0  | 8192              | 440000       | 420000            | 390000             |
+-------------------------------------+-------------------+--------------+-------------------+--------------------+


4 Nodes, 4 x L40 GPUs per node
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CPU: Intel(R) Xeon(R) Platinum 8362 CPU @ 2.80GHz

+-------------------------------------+-------------------+--------------+-------------------+--------------------+
| Environment Name                    | # of Environments | Environment  | Environment Step  | Environment Step,  |
|                                     |                   | Step FPS     | and               | Inference,         |
|                                     |                   |              | Inference FPS     | and Train FPS      |
+=====================================+===================+==============+===================+====================+
| Isaac-Cartpole-Direct-v0            | 4096              | 10200000     | 8200000           | 3500000            |
+-------------------------------------+-------------------+--------------+-------------------+--------------------+
| Isaac-Cartpole-RGB-Camera-Direct-v0 | 1024              | 530000       | 490000            | 260000             |
+-------------------------------------+-------------------+--------------+-------------------+--------------------+
| Isaac-Velocity-Rough-G1-v0          | 4096              | 1200000      | 1100000           | 960000             |
+-------------------------------------+-------------------+--------------+-------------------+--------------------+
| Isaac-Repose-Cube-Shadow-Direct-v0  | 8192              | 2400000      | 2300000           | 1800000            |
+-------------------------------------+-------------------+--------------+-------------------+--------------------+


Benchmark Scripts
-----------------

For ease of reproducibility, we provide benchmarking scripts available at ``scripts/benchmarks``.
This folder contains individual benchmark scripts that resemble the ``train.py`` script for RL-Games
and RSL RL. In addition, we also provide a benchmarking script that runs only the environment implementation
without any reinforcement learning library.

Example scripts can be run similarly to training scripts:

.. code-block:: bash

   # benchmark with RSL RL
   python scripts/benchmarks/benchmark_rsl_rl.py --task=Isaac-Cartpole-v0 --headless

   # benchmark with RL Games
   python scripts/benchmarks/benchmark_rlgames.py --task=Isaac-Cartpole-v0 --headless

   # benchmark without RL libraries
   python scripts/benchmarks/benchmark_non_rl.py --task=Isaac-Cartpole-v0 --headless

Each script will generate a set of KPI files at the end of the run, which includes data on the
startup times, runtime statistics, such as the time taken for each simulation or rendering step,
as well as overall environment FPS for stepping the environment, performing inference during
rollout, as well as training.
