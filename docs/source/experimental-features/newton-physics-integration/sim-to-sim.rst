.. _sim2sim:

Sim-to-Sim Policy Transfer
==========================
This section provides examples of sim-to-sim policy transfer using the Newton backend. Sim-to-sim transfer is an essential step before real robot deployment because it verifies that policies work across different simulators. Policies that pass sim-to-sim verification are much more likely to succeed on real robots.


Overview
--------

This guide shows how to run a PhysX-trained policy on the Newton backend. While the method works for any robot and physics engine, it has only been tested with Unitree G1, Unitree H1, and ANYmal-D robots using PhysX-trained policies.

PhysX-trained policies expect joints and links in a specific order determined by how PhysX parses the robot model. However, Newton may parse the same robot with different joint and link ordering.

In the future, we plan to solve this using **robot schema** that standardizes joint and link ordering across different backends.

Currently, we solve this by remapping observations and actions using joint mappings defined in YAML files. These files specify joint names in both PhysX order (source) and Newton order (target). During policy execution, we use this mapping to reorder observations and actions so they work correctly with Newton.


What you need
~~~~~~~~~~~~~

- A policy checkpoint trained with PhysX (RSL-RL).
- A joint mapping YAML for your robot under ``scripts/newton_sim2sim/mappings/``.
- The provided player script: ``scripts/newton_sim2sim/rsl_rl_transfer.py``.

To add a new robot, create a YAML file with two lists where each joint name appears exactly once in both:

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

The script automatically computes the necessary mappings for locomotion tasks.


How to run
~~~~~~~~~~

Use this command template to run a PhysX-trained policy with Newton:

.. code-block:: bash

   ./isaaclab.sh -p scripts/newton_sim2sim/rsl_rl_transfer.py \
       --task=<TASK_ID> \
       --num_envs=32 \
       --checkpoint <PATH_TO_PHYSX_CHECKPOINT> \
       --policy_transfer_file <PATH_TO_MAPPING_YAML>

Here are examples for different robots:

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

- This transfer method has only been tested with Unitree G1, Unitree H1, and ANYmal-D using PhysX-trained policies.
- The observation remapping assumes a locomotion layout with base observations followed by joint observations. For different observation layouts, you'll need to modify ``scripts/newton_sim2sim/policy_mapping.py``.
- When adding new robots or backends, make sure both source and target have identical joint names, and that the YAML lists reflect how each backend orders these joints.
