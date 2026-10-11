:orphan:

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

This example uses the Newton backend to grasp and lift a soft object. Pass
``--viz newton_gl`` to open the Newton visualizer:

.. code-block:: bash

   uv run --extra tetrahedralization python scripts/environments/state_machine/lift_franka_soft.py \
      --num_envs 1 --viz newton_gl

Open a cabinet drawer
---------------------

This example approaches the drawer handle, grasps it, pulls the drawer open,
and releases it:

.. code-block:: bash

   uv run python scripts/environments/state_machine/open_cabinet_sm.py \
      --num_envs 32 --viz kit

.. _tablecloth-example:

Compare five humanoid tablecloth pulls
--------------------------------------

This standalone example uses :class:`~isaaclab.scene.InteractiveScene` to build
five side-by-side H2 Plus / SHARPA trials with the default Isaac Lab ground plane,
wooden dining tables, and SimReady kitchenware. Newton VBD simulates the
articulations, cloth, and tableware in a single solver. The Warp state machine
and Newton IK run on the GPU inside the physics CUDA graph. It does not register
an RL environment or require a trained policy.

.. figure:: ../_static/examples/tablecloth_h2.jpg
   :alt: Five side-by-side H2 Plus robots grasping red tablecloths beside kitchen tables.
   :width: 100%

   Contact-only pulls at peak speeds of 4.0, 2.0, 1.5, 0.75, and 0.5 m/s,
   from left to right, rendered with Kit.

.. code-block:: bash

   uv run isaaclab example tablecloth --device cuda:0
   uv run --extra importers isaaclab example tablecloth --robot h1 --device cuda:0

The default robot uses the manufacturer's articulated `H2 Plus USD with SHARPA hands
<https://github.com/unitreerobotics/unitree_model/tree/b6a8942b0803b6c137e58cef12beb4b03e4a2fa7/H2_Plus>`__,
downloaded and cached by Isaac Lab. ``--robot h1`` explicitly selects the existing H1 fallback;
it requires the ``importers`` extra. Both use physical contact-only grips, without cloth attachments.
The H2 index fingers start beneath the hem; the thumbs close against their native
rubber surfaces before the lift and pull. This is a calibrated pinch-and-pull example,
not a general-purpose grasp planner. The H2 configuration uses 64 physics ticks
per 60 Hz control frame to resolve the fastest pull. Native fingertip contacts
are computationally expensive; this configuration does not target real-time
playback. Its position drives are simulation settings: direct VBD with
compliant-ALM rigid dynamics does not enforce the asset's hardware torque limits
or armature.

Use ``--speeds 2.0`` for one robot, or ``--viz none`` for a complete headless
run. The default run duration includes the slowest pull and a final hold.
All H2 lanes use the same object masses, friction, cloth mesh, and 0.48 × 0.84 m
tabletop. Only the peak pull speed changes: the fastest pull leaves the arrangement
nearly unchanged, the middle pulls produce more sliding, and the two slowest
pulls carry the tableware off the table. The dining-table USD supplies the visuals;
simple box proxies supply the tabletop and leg collisions.
The pelvis is fixed with the feet at ground level: this demonstrates
manipulation, not humanoid balance. Asset and cloth changes require revalidating
the calibrated physical grasp. This experimental example checks material-point
retention after a complete run and reports grasp failures; successful IK tracking
alone does not establish a grasp. Kit uses Catmull-Clark subdivision for cloth
rendering without refining the physics mesh.
