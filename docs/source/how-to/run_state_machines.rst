.. _run-scripted-state-machines:

Run Scripted State Machines
===========================

Isaac Lab includes hand-written state-machine examples for inspecting an
environment's observations and action interface without training a policy. The
state transitions run in parallel across environments as Warp kernels, which
keeps the examples efficient at larger environment counts.

Run these commands from the Isaac Lab repository root. Use ``--num_envs`` to
change the number of parallel environments and ``--viz`` to select a
visualizer.

Pick and lift a rigid cube
--------------------------

This example approaches a cube, closes the gripper, and lifts the cube to its
goal pose:

.. code-block:: bash

   uv run python scripts/environments/state_machine/lift_cube_sm.py \
      --num_envs 32 --viz kit

Lift a deformable object
------------------------

This example uses the Newton backend to grasp and lift a soft object. The
Newton visualizer opens by default:

.. code-block:: bash

   uv run python scripts/environments/state_machine/lift_franka_soft.py \
      --num_envs 1

Open a cabinet drawer
---------------------

This example approaches the drawer handle, grasps it, pulls the drawer open,
and releases it:

.. code-block:: bash

   uv run python scripts/environments/state_machine/open_cabinet_sm.py \
      --num_envs 32 --viz kit

Each script defines its states, wait times, transition kernel, and action loop
in one file. Start with ``lift_cube_sm.py`` when adapting the pattern to a new
manipulation task.
