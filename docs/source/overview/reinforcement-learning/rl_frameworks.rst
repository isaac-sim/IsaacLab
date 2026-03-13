.. _rl-frameworks:

Reinforcement Learning Library Comparison
=========================================

In this section, we provide an overview of the supported reinforcement learning libraries in Isaac Lab,
along with performance benchmarks across the libraries.

The supported libraries are:

- `SKRL <https://skrl.readthedocs.io>`__
- `RSL-RL <https://github.com/leggedrobotics/rsl_rl>`__
- `RL-Games <https://github.com/Denys88/rl_games>`__
- `Stable-Baselines3 <https://stable-baselines3.readthedocs.io/en/master/index.html>`__

Feature Comparison
------------------

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 1

   * - Feature
     - RL-Games
     - RSL RL
     - SKRL
     - Stable Baselines3
   * - Algorithms Included
     - PPO, SAC, A2C
     - PPO, Distillation
     - `Extensive List <https://skrl.readthedocs.io/en/latest/#agents>`__
     - `Extensive List <https://github.com/DLR-RM/stable-baselines3?tab=readme-ov-file#implemented-algorithms>`__
   * - Vectorized Training
     - Yes
     - Yes
     - Yes
     - No
   * - Distributed Training
     - Yes
     - Yes
     - Yes
     - No
   * - ML Frameworks Supported
     - PyTorch
     - PyTorch
     - PyTorch, JAX
     - PyTorch
   * - Multi-Agent Support
     - PPO
     - PPO
     - PPO + Multi-Agent algorithms
     - External projects support
   * - Documentation
     - Low
     - Low
     - Comprehensive
     - Extensive
   * - Community Support
     - Small Community
     - Small Community
     - Small Community
     - Large Community
   * - Available Examples in Isaac Lab
     - Large
     - Large
     - Large
     - Small


Training Performance
--------------------

We performed training with each RL library on the same ``Isaac-Humanoid-v0`` environment
with ``--headless`` on a single NVIDIA GeForce RTX 4090 and logged the total training time
for 65.5M steps (4096 environments x 32 rollout steps x 500 iterations).

+--------------------+-----------------+
| RL Library         | Time in seconds |
+====================+=================+
| RL-Games           | 201             |
+--------------------+-----------------+
| SKRL               | 201             |
+--------------------+-----------------+
| RSL RL             | 198             |
+--------------------+-----------------+
| Stable-Baselines3  | 287             |
+--------------------+-----------------+

Training commands (check for the *'Training time: XXX seconds'* line in the terminal output):

.. code:: bash

    python scripts/reinforcement_learning/rl_games/train.py --task Isaac-Humanoid-v0 --max_iterations 500 --headless
    python scripts/reinforcement_learning/skrl/train.py --task Isaac-Humanoid-v0 --max_iterations 500 --headless
    python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Humanoid-v0 --max_iterations 500 --headless
    python scripts/reinforcement_learning/sb3/train.py --task Isaac-Humanoid-v0 --max_iterations 500 --headless
