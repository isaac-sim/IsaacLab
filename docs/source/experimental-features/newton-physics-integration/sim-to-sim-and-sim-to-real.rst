.. _sim2sim_and_sim2real:

Sim-to-Sim and Sim-to-Real
==========================
This section provides examples of sim-to-sim as well as sim-to-real policy transfer using the Newton backend.


Sim-to-Sim Policy Transfer (PhysX to Newton) Overview
-------------------------------------------------------------

This guide explains how to replay a policy trained in Isaac Lab with the PhysX backend on the Newton backend. The method is applicable to any robot and physics engine, but has been validated only on Unitree G1, Unitree H1, and ANYmal-D, and only for policies trained with PhysX.

Policies trained with PhysX assume a specific joint/link ordering defined by the PhysX-parsed robot model. The Newton backend may parse the same robot with a different joint and link ordering. To execute a PhysX-trained policy under Newton, we remap observations and actions between the two orderings.

This remapping is configured via YAML files that list the joint names in PhysX order (source) and Newton order (target). During inference, the mapping is used to reorder the input observations and output actions so they are interpreted correctly by the Newton-based simulation.


What you need
~~~~~~~~~~~~~

- A policy checkpoint trained with PhysX (RSL-RL).
- A joint mapping YAML for your robot under ``scripts/newton_sim2sim/mappings/``.
- The provided player script: ``scripts/newton_sim2sim/rsl_rl_transfer.py``.


Available mappings
~~~~~~~~~~~~~~~~~~

The repository includes the following ready-to-use mappings:

- ``scripts/newton_sim2sim/mappings/sim2sim_g1.yaml``
- ``scripts/newton_sim2sim/mappings/sim2sim_h1.yaml``
- ``scripts/newton_sim2sim/mappings/sim2sim_anymal_d.yaml``

If you add a new robot, create a YAML with two lists, where each joint name appears exactly once in both lists:

.. code-block:: yaml

   # Example structure
   source_joint_names:  # PhysX joint order
     - joint_1
     - joint_2
     # ...
   target_joint_names:  # Newton joint order
     - joint_1
     - joint_2
     # ...

The player will compute bidirectional mappings and an observation remap suitable for locomotion tasks.


How to run
~~~~~~~~~~

Use the following command template to play a PhysX-trained policy with the Newton backend of IsaacLab:

.. code-block:: bash

   ./isaaclab.sh -p scripts/newton_sim2sim/rsl_rl_transfer.py \
       --task=<TASK_ID> \
       --num_envs=32 \
       --checkpoint <PATH_TO_PHYSX_CHECKPOINT> \
       --policy_transfer_file <PATH_TO_MAPPING_YAML>

The following examples show how to run this transfer for various robots.

1. Unitree G1

.. code-block:: bash

   ./isaaclab.sh -p scripts/newton_sim2sim/rsl_rl_transfer.py \
       --task=Isaac-Velocity-Flat-G1-v0 \
       --num_envs=32 \
       --checkpoint <PATH_TO_PHYSX_CHECKPOINT> \
       --policy_transfer_file scripts/newton_sim2sim/mappings/sim2sim_g1.yaml


2. Unitree H1


.. code-block:: bash

   ./isaaclab.sh -p scripts/newton_sim2sim/rsl_rl_transfer.py \
       --task=Isaac-Velocity-Flat-H1-v0 \
       --num_envs=32 \
       --checkpoint <PATH_TO_PHYSX_CHECKPOINT> \
       --policy_transfer_file scripts/newton_sim2sim/mappings/sim2sim_h1.yaml


3. ANYmal-D


.. code-block:: bash

   ./isaaclab.sh -p scripts/newton_sim2sim/rsl_rl_transfer.py \
       --task=Isaac-Velocity-Flat-Anymal-D-v0 \
       --num_envs=32 \
       --checkpoint <PATH_TO_PHYSX_CHECKPOINT> \
       --policy_transfer_file scripts/newton_sim2sim/mappings/sim2sim_anymal_d.yaml


Notes and limitations
~~~~~~~~~~~~~~~~~~~~~

- The transfer has been tested only for Unitree G1, Unitree H1, and ANYmal-D with PhysX-trained policies.
- The observation remapping implemented in ``scripts/newton_sim2sim/policy_mapping.py`` assumes a locomotion observation layout (a fixed base segment followed by joint-wise segments). If your observation layout differs, adjust the mapping accordingly.
- For new robots/backends, ensure the joint name sets are identical between source and target and that their orders in the YAML reflect each backend’s parsing.


Sim-to-Real Policy Transfer Overview
----------------------------------------------------------------

This section demonstrates a sim-to-real workflow through the teacher–student distillation approach for the Unitree G1 velocity-tracking task with the Newton backend.

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

The distillation stage performs behavior cloning from teacher to student by minimizing mean squared error between actions, i.e. :math:`loss = MSE(\pi(O_{teacher}), \pi(O_{student}))`.

The student policy uses only terms available from real sensors. See ``StudentPolicyCfg(ObsGroup)`` in `velocity_env_cfg.py <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py>`__. Specifically, **Root angular velocity** and **Projected gravity** are obtained from the IMU sensor, **Joint positions and velocities** are obtained from joint encoders and **Actions** are joint torques applied by the controller.

Run the student distillation task ``Velocity-G1-Distillation-v1`` and point ``--load_run``/``--checkpoint`` to the teacher run/checkpoint you want to distill from.

.. code-block:: bash

   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Velocity-G1-Distillation-v1 --num_envs=4096 --headless --load_run 2025-08-13_23-53-28 --checkpoint model_1499.pt

.. note::

   Use the correct ``--load_run`` and ``--checkpoint`` to ensure you distill from the intended teacher policy.


3. Fine-tune the student policy with RL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fine-tune the distilled student policy using RL with the ``Velocity-G1-Student-Finetune-v1`` task. Initialize from a checkpoint using ``--load_run``/``--checkpoint``.

.. code-block:: bash

   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Velocity-G1-Student-Finetune-v1 --num_envs=4096 --headless --load_run 2025-08-20_16-06-52_distillation --checkpoint model_1499.pt

This uses the distilled student policy as the starting point and fine-tunes it with RL.

.. note::

   Ensure ``--load_run`` and ``--checkpoint`` point to the intended initial policy (typically the latest student checkpoint from the distillation run).

You can replay the student policy via

.. code-block:: bash

   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task=Velocity-G1-Student-Finetune-v1 --num_envs=32


which will export the policy to ``.pt``/``.onnx`` files in the exported directory of the run. These policies can be deployed to the real robot.
