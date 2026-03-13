.. _warp-environments:

Warp Experimental Environments
==============================

.. note::

   The warp environment infrastructure lives in ``isaaclab_experimental`` and
   ``isaaclab_tasks_experimental``. It's an experimental feature.

The experimental extensions introduce **warp-first** environment infrastructure with CUDA graph capture
support. All environment-side computation (observations, rewards, resets, actions) runs as pure Warp
kernels, eliminating Python overhead and enabling CUDA graph capture for maximum throughput.


Workflows
~~~~~~~~~

Two environment workflows are supported:

**Direct workflow** — ``DirectRLEnvWarp`` base class. You implement the step loop, observations,
rewards, and resets directly in your env class using Warp kernels.

**Manager-based workflow** — ``ManagerBasedRLEnvWarp`` base class. You define MDP terms as
standalone Warp-kernel functions and compose them via configuration.


Available Environments
~~~~~~~~~~~~~~~~~~~~~~

Direct Warp Environments
^^^^^^^^^^^^^^^^^^^^^^^^

- ``Isaac-Cartpole-Direct-Warp-v0`` — Cartpole balance
- ``Isaac-Ant-Direct-Warp-v0`` — Ant locomotion
- ``Isaac-Humanoid-Direct-Warp-v0`` — Humanoid locomotion
- ``Isaac-Repose-Cube-Allegro-Direct-Warp-v0`` — Allegro hand cube repose


Manager-Based Warp Environments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Classic**

- ``Isaac-Cartpole-Warp-v0``
- ``Isaac-Ant-Warp-v0``
- ``Isaac-Humanoid-Warp-v0``

**Locomotion (Flat)**

- ``Isaac-Velocity-Flat-Anymal-B-Warp-v0``
- ``Isaac-Velocity-Flat-Anymal-C-Warp-v0``
- ``Isaac-Velocity-Flat-Anymal-D-Warp-v0``
- ``Isaac-Velocity-Flat-Cassie-Warp-v0``
- ``Isaac-Velocity-Flat-G1-Warp-v0``
- ``Isaac-Velocity-Flat-G1-Warp-v1``
- ``Isaac-Velocity-Flat-H1-Warp-v0``
- ``Isaac-Velocity-Flat-Unitree-A1-Warp-v0``
- ``Isaac-Velocity-Flat-Unitree-Go1-Warp-v0``
- ``Isaac-Velocity-Flat-Unitree-Go2-Warp-v0``

**Manipulation**

- ``Isaac-Reach-Franka-Warp-v0``
- ``Isaac-Reach-UR10-Warp-v0``


Quick Start
~~~~~~~~~~~

.. code-block:: bash

    # Direct workflow
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Cartpole-Direct-Warp-v0 --num_envs 4096 --headless

    # Manager-based workflow
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Velocity-Flat-Anymal-C-Warp-v0 --num_envs 4096 --headless

All RL libraries with warp-compatible wrappers are supported: RSL-RL, RL Games, SKRL, and
Stable-Baselines3.


Performance Comparison
~~~~~~~~~~~~~~~~~~~~~~

Step time comparison between the stable (torch/manager) and warp (CUDA graph captured) variants,
both running on the Newton physics backend. Measured over 300 iterations with 4096 environments.

.. note::

   The warp migration is an ongoing effort. Several components (e.g. scene write, actuator models)
   have not yet been migrated to Warp kernels and still run through torch. Further performance
   improvements are expected as these components are migrated.

.. list-table::
   :header-rows: 1
   :widths: 30 12 15 15 12

   * - Env
     - Type
     - Stable Step (us)
     - Warp Step (us)
     - Change
   * - Cartpole-Direct
     - Direct
     - 5,274
     - 4,331
     - -17.88%
   * - Ant-Direct
     - Direct
     - 6,368
     - 3,128
     - -50.88%
   * - Humanoid-Direct
     - Direct
     - 13,937
     - 10,783
     - -22.63%
   * - Allegro-Direct
     - Direct
     - 82,950
     - 74,570
     - -10.10%
   * - Cartpole
     - Manager
     - 7,971
     - 3,642
     - -54.31%
   * - Ant
     - Manager
     - 9,781
     - 4,672
     - -52.23%
   * - Humanoid
     - Manager
     - 17,653
     - 12,505
     - -29.16%
   * - Reach-Franka
     - Manager
     - 11,458
     - 7,813
     - -31.83%
   * - Anymal-B
     - Manager
     - 29,188
     - 21,781
     - -25.38%
   * - Anymal-C
     - Manager
     - 30,938
     - 22,228
     - -28.15%
   * - Anymal-D
     - Manager
     - 32,294
     - 23,977
     - -25.75%
   * - Cassie
     - Manager
     - 17,320
     - 10,706
     - -38.19%
   * - G1
     - Manager
     - 34,487
     - 27,300
     - -20.84%
   * - H1
     - Manager
     - 22,202
     - 15,864
     - -28.55%
   * - A1
     - Manager
     - 15,257
     - 9,907
     - -35.07%
   * - Go1
     - Manager
     - 16,515
     - 11,869
     - -28.13%
   * - Go2
     - Manager
     - 15,221
     - 9,966
     - -34.52%


Adding New Warp Environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To add a new warp environment:

1. Create a task config in ``isaaclab_tasks_experimental`` mirroring the stable config structure.
2. Import MDP terms from ``isaaclab_experimental.envs.mdp`` instead of ``isaaclab.envs.mdp``.
3. Configure Newton physics with ``use_cuda_graph=True``.
4. Register the task with a ``-Warp-`` suffix in the gym ID.

For a detailed guide on converting each component (observations, rewards, events, actions),
see :doc:`warp-env-migration`.
