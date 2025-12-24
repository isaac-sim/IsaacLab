.. _sim2sim:

Sim-to-Sim Policy Transfer
==========================
This section provides examples of sim-to-sim policy transfer between PhysX and Newton backends. Sim-to-sim transfer is an essential step before real robot deployment because it verifies that policies work across different simulators. Policies that pass sim-to-sim verification are much more likely to succeed on real robots.


Overview
--------

This guide shows how to transfer policies between PhysX and Newton backends in both directions. The main challenge is that different physics engines may parse the same robot model with different joint and link ordering.

Policies trained in one backend expect joints and links in a specific order determined by how that backend parses the robot model. When transferring to another backend, the joint ordering may be different, requiring remapping of observations and actions.

In the future, we plan to solve this using **robot schema** that standardizes joint and link ordering across different backends.

Currently, we solve this by remapping observations and actions using joint mappings defined in YAML files. These files specify joint names in both source and target backend orders. During policy execution, we use this mapping to reorder observations and actions so they work correctly with the target backend.

The method has been tested with Unitree G1, Unitree Go2, Unitree H1, and ANYmal-D robots for both transfer directions.


What you need
~~~~~~~~~~~~~

- A policy checkpoint trained with either PhysX or Newton (RSL-RL).
- A joint mapping YAML for your robot under ``scripts/sim2sim_transfer/config/``.
- The provided player script: ``scripts/sim2sim_transfer/rsl_rl_transfer.py``.

To add a new robot, create a YAML file with two lists where each joint name appears exactly once in both:

.. code-block:: yaml

   # Example structure
   source_joint_names:  # Source backend joint order
     - joint_1
     - joint_2
     # ...
   target_joint_names:  # Target backend joint order
     - joint_1
     - joint_2
     # ...

The script automatically computes the necessary mappings for locomotion tasks.


PhysX-to-Newton Transfer
~~~~~~~~~~~~~~~~~~~~~~~~

To run a PhysX-trained policy with the Newton backend, use this command template:

.. code-block:: bash

   ./isaaclab.sh -p scripts/sim2sim_transfer/rsl_rl_transfer.py \
       --task=<TASK_ID> \
       --num_envs=32 \
       --checkpoint <PATH_TO_PHYSX_CHECKPOINT> \
       --policy_transfer_file <PATH_TO_MAPPING_YAML> \
       --visualizer newton

Here are examples for different robots:

1. Unitree G1

.. code-block:: bash

   ./isaaclab.sh -p scripts/sim2sim_transfer/rsl_rl_transfer.py \
       --task=Isaac-Velocity-Flat-G1-v0 \
       --num_envs=32 \
       --checkpoint <PATH_TO_PHYSX_CHECKPOINT> \
       --policy_transfer_file scripts/sim2sim_transfer/config/physx_to_newton_g1.yaml \
       --visualizer newton

2. Unitree H1


.. code-block:: bash

   ./isaaclab.sh -p scripts/sim2sim_transfer/rsl_rl_transfer.py \
       --task=Isaac-Velocity-Flat-H1-v0 \
       --num_envs=32 \
       --checkpoint <PATH_TO_PHYSX_CHECKPOINT> \
       --policy_transfer_file scripts/sim2sim_transfer/config/physx_to_newton_h1.yaml \
       --visualizer newton


3. Unitree Go2

.. code-block:: bash

   ./isaaclab.sh -p scripts/sim2sim_transfer/rsl_rl_transfer.py \
       --task=Isaac-Velocity-Flat-Go2-v0 \
       --num_envs=32 \
       --checkpoint <PATH_TO_PHYSX_CHECKPOINT> \
       --policy_transfer_file scripts/sim2sim_transfer/config/physx_to_newton_go2.yaml \
       --visualizer newton


4. ANYmal-D


.. code-block:: bash

   ./isaaclab.sh -p scripts/sim2sim_transfer/rsl_rl_transfer.py \
       --task=Isaac-Velocity-Flat-Anymal-D-v0 \
       --num_envs=32 \
       --checkpoint <PATH_TO_PHYSX_CHECKPOINT> \
       --policy_transfer_file scripts/sim2sim_transfer/config/physx_to_newton_anymal_d.yaml \
       --visualizer newton

Note that to run this, you need to checkout the Newton-based branch of IsaacLab such as ``feature/newton``.

Newton-to-PhysX Transfer
~~~~~~~~~~~~~~~~~~~~~~~~

To transfer Newton-trained policies to PhysX-based IsaacLab, use the reverse mapping files:

Here are examples for different robots:

1. Unitree G1

.. code-block:: bash

   ./isaaclab.sh -p scripts/sim2sim_transfer/rsl_rl_transfer.py \
       --task=Isaac-Velocity-Flat-G1-v0 \
       --num_envs=32 \
       --checkpoint <PATH_TO_NEWTON_CHECKPOINT> \
       --policy_transfer_file scripts/sim2sim_transfer/config/newton_to_physx_g1.yaml


2. Unitree H1

.. code-block:: bash

   ./isaaclab.sh -p scripts/sim2sim_transfer/rsl_rl_transfer.py \
       --task=Isaac-Velocity-Flat-H1-v0 \
       --num_envs=32 \
       --checkpoint <PATH_TO_NEWTON_CHECKPOINT> \
       --policy_transfer_file scripts/sim2sim_transfer/config/newton_to_physx_h1.yaml


3. Unitree Go2

.. code-block:: bash

   ./isaaclab.sh -p scripts/sim2sim_transfer/rsl_rl_transfer.py \
       --task=Isaac-Velocity-Flat-Go2-v0 \
       --num_envs=32 \
       --checkpoint <PATH_TO_NEWTON_CHECKPOINT> \
       --policy_transfer_file scripts/sim2sim_transfer/config/newton_to_physx_go2.yaml


4. ANYmal-D

.. code-block:: bash

   ./isaaclab.sh -p scripts/sim2sim_transfer/rsl_rl_transfer.py \
       --task=Isaac-Velocity-Flat-Anymal-D-v0 \
       --num_envs=32 \
       --checkpoint <PATH_TO_NEWTON_CHECKPOINT> \
       --policy_transfer_file scripts/sim2sim_transfer/config/newton_to_physx_anymal_d.yaml

The key difference is using the ``newton_to_physx_*.yaml`` mapping files instead of ``physx_to_newton_*.yaml`` files. Also note that you need to checkout a PhysX-based IsaacLab branch such as ``main``.

Notes and Limitations
~~~~~~~~~~~~~~~~~~~~~

- Both transfer directions have been tested with Unitree G1, Unitree Go2, Unitree H1, and ANYmal-D robots.
- PhysX-to-Newton transfer uses ``physx_to_newton_*.yaml`` mapping files.
- Newton-to-PhysX transfer requires the corresponding ``newton_to_physx_*.yaml`` mapping files and the PhysX branch of IsaacLab.
- The observation remapping assumes a locomotion layout with base observations followed by joint observations. For different observation layouts, you'll need to modify the ``get_joint_mappings`` function in ``scripts/sim2sim_transfer/rsl_rl_transfer.py``.
- When adding new robots or backends, make sure both source and target have identical joint names, and that the YAML lists reflect how each backend orders these joints.
