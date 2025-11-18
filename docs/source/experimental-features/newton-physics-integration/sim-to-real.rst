.. _sim2real:

Sim-to-Real Policy Transfer
===========================
Deploying policies from simulation to real robots involves important nuances that must be addressed.
This section provides a high-level guide for training policies that can be deployed on a real Unitree G1 robot.
The key challenge is that not all observations available in simulation can be directly measured by real robot sensors.
This means RL-trained policies cannot be directly deployed unless they use only sensor-available observations. For example, while real robot IMU sensors provide angular acceleration (which can be integrated to get angular velocity), they cannot directly measure linear velocity. Therefore, if a policy relies on base linear velocity during training, this information must be removed before real robot deployment.


Requirements
~~~~~~~~~~~~

We assume that policies from this workflow are first verified through sim-to-sim transfer before real robot deployment.
Please see :ref:`here <sim2sim>` for more information.


Overview
--------

This section demonstrates a sim-to-real workflow using teacher–student distillation for the Unitree G1
velocity-tracking task with the Newton backend.

The teacher–student distillation workflow consists of three stages:

1. Train a teacher policy with privileged observations that are not available in real-world sensors.
2. Distill a student policy that excludes privileged terms (e.g., root linear velocity) by behavior cloning from the teacher policy.
3. Fine-tune the student policy with RL using only real-sensor observations.

The teacher and student observation groups are implemented in the velocity task configuration. See the following source for details:

- Teacher observations: ``PolicyCfg(ObsGroup)`` in `velocity_env_cfg.py <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py>`__
- Student observations: ``StudentPolicyCfg(ObsGroup)`` in `velocity_env_cfg.py <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py>`__


1. Train the teacher policy
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Train the teacher policy for the G1 velocity task using the Newton backend. The task ID is ``Isaac-Velocity-Flat-G1-v1``

.. code-block:: bash

   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Flat-G1-v1 --num_envs=4096 --headless

The teacher policy includes privileged observations (e.g., root linear velocity) defined in ``PolicyCfg(ObsGroup)``.


2. Distill the student policy (remove privileged terms)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

During distillation, the student policy learns to mimic the teacher through behavior cloning by minimizing the mean squared error
between their actions: :math:`loss = MSE(\pi(O_{teacher}), \pi(O_{student}))`.

The student policy only uses observations available from real sensors (see ``StudentPolicyCfg(ObsGroup)``
in `velocity_env_cfg.py <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py>`__).
Specifically: **Root angular velocity** and **Projected gravity** come from the IMU sensor, **Joint positions and velocities** come from joint encoders, and **Actions** are the joint torques applied by the controller.

Run the student distillation task ``Velocity-G1-Distillation-v1`` using ``--load_run`` and ``--checkpoint`` to specify the teacher policy you want to distill from.

.. code-block:: bash

   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Velocity-G1-Distillation-v1 --num_envs=4096 --headless --load_run 2025-08-13_23-53-28 --checkpoint model_1499.pt

.. note::

   Use the correct ``--load_run`` and ``--checkpoint`` to ensure you distill from the intended teacher policy.


3. Fine-tune the student policy with RL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fine-tune the distilled student policy using RL with the ``Velocity-G1-Student-Finetune-v1`` task.
Use ``--load_run`` and ``--checkpoint`` to initialize from the distilled policy.

.. code-block:: bash

   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Velocity-G1-Student-Finetune-v1 --num_envs=4096 --headless --load_run 2025-08-20_16-06-52_distillation --checkpoint model_1499.pt

This starts from the distilled student policy and improves it further with RL training.

.. note::

   Make sure ``--load_run`` and ``--checkpoint`` point to the correct initial policy (usually the latest checkpoint from the distillation step).

You can replay the student policy via:

.. code-block:: bash

   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task=Velocity-G1-Student-Finetune-v1 --num_envs=32


This exports the policy as ``.pt`` and ``.onnx`` files in the run's export directory, ready for real robot deployment.
