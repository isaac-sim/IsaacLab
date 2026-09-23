.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

Browser simulations
===================

.. raw:: html

   <link rel="stylesheet" href="../../_static/css/browser-demo.css">
   <script type="module" src="../../_static/css/browser-demo.js"></script>

These small Newton simulations run locally in your browser. They load compiled WebAssembly
only when scrolled into view. Use **Pause** and **Reset** to inspect each example.
The 3D views use Isaac Lab's default metric checker ground textures; Newton supplies the
collision plane in the browser.

Material tuning with VBD
------------------------

Three tetrahedral cubes fall onto a plane under Newton's VBD solver. One viewer controls
the middle cube's two Lamé stiffness values, all cubes' material damping, and gravity.
The outer cubes retain their stiffness values as references. Change one control and press
**Reset** to compare the same drop. The exported scene uses a 3 × 3 × 3 cell grid and
eight solver iterations per step. Increasing the solver to 16 iterations barely changed
the soft cube's motion in this setup; damping reduced its lingering oscillation after contact.
Drag the view to orbit and scroll over it to zoom. For the corresponding Isaac Lab material
configuration, see :doc:`deformables`.

.. raw:: html

   <isaaclab-browser-demo src="../../_static/browser_demos/stiffness/manifest.json" parameters="stiffness,damping,gravity" demo-title="VBD material tuning"></isaaclab-browser-demo>

Cloth bending with VBD
----------------------

One cloth sheet spans two upright cylindrical supports. Its end rows are fixed
at the cylinder tops, so it bends between the supports without sliding off.
Change the cloth's bending stiffness on a logarithmic scale from 0.1 to
1000 N·m, or change gravity, then press **Reset** to compare the same drop.
The scene uses a 16 × 4 cell cloth mesh, particle-cylinder contact, and ten VBD
iterations per step. See :doc:`solver-tuning/tune_vbd` for the corresponding
surface-deformable material fields and contact settings.

.. raw:: html

   <isaaclab-browser-demo src="../../_static/browser_demos/cloth_bending/manifest.json"></isaaclab-browser-demo>

Inclined friction with MJWarp
-----------------------------

Three equal rigid boxes start at rest near the top of a 0.22 rad incline.
Gravity pulls them downhill. Change the middle box's friction coefficient and
press **Reset** to compare its travel against fixed 0.05 and 0.8 references.
Newton MJWarp advances the contact simulation at 120 Hz. See
:doc:`solver-tuning/tune_mjwarp` for contact and friction tuning in Isaac Lab tasks.

.. raw:: html

   <isaaclab-browser-demo src="../../_static/browser_demos/rigid_friction/manifest.json"></isaaclab-browser-demo>

Cartpole policy with MJWarp
---------------------------

The published ``Isaac-Cartpole-Direct`` Newton MJWarp policy balances the pole from
a repeatable 0.2 rad initial angle. Hold the slider to add up to 300 N of cart
force, then release it to watch the policy respond.
**Reset** restores the starting state. The cart resets beyond 3 m of travel.
See :doc:`solver-tuning/tune_mjwarp` for
the corresponding solver tuning guide.

.. raw:: html

   <isaaclab-browser-demo src="../../_static/browser_demos/cartpole/manifest.json"></isaaclab-browser-demo>

G1 velocity control
-------------------

This example uses the ``Isaac-Velocity-Flat-G1`` RSL-RL checkpoint trained with Newton MJWarp.
Drag the X/Y and yaw pads to set forward, sideways, and turning velocity commands. Release a pad
to stop; the arrow keys also work when a pad is focused. Drag the 3D view to orbit
the robot and scroll over it to zoom. Its visual meshes come from the matching 37-joint G1
description and load only when the example enters view. The policy runs every 20 ms;
Newton advances at 5 ms per physics step. The browser export uses the task's default pose and
joint actuator gains, and a minimal collision asset. It uses two solver iterations for browser
responsiveness, while the training task uses 100. It is an interactive illustration of the
policy, rather than a replacement for Isaac Lab evaluation.

.. raw:: html

   <isaaclab-browser-demo src="../../_static/browser_demos/g1/manifest.json"></isaaclab-browser-demo>

ANYmal-D velocity control
-------------------------

This example uses the published ``Isaac-Velocity-Flat-AnymalD`` RSL-RL checkpoint trained with
Newton MJWarp. The same X/Y and yaw pads command its three body velocities. Its 38 visual meshes
come from the ANYmal-D USD and load only when scrolled into view. The 12-action policy runs every
20 ms, with 5 ms physics steps and two substeps per step. The task uses an LSTM ANYdrive actuator;
this compact browser simulation uses a tuned PD drive instead. Its motion can therefore differ from
the full task.

.. raw:: html

   <isaaclab-browser-demo src="../../_static/browser_demos/anymal/manifest.json"></isaaclab-browser-demo>

See the `export script <https://github.com/isaac-sim/IsaacLab/blob/main/docs/browser_demos/export.py>`_
and `rebuild instructions <https://github.com/isaac-sim/IsaacLab/blob/main/docs/browser_demos/README.md>`_.
The browser bundles contain compiled simulation code, policy weights, and packed robot
visual geometry. They run from Isaac Lab documentation without a connection to the export tool.
For full task metrics and rendering, run ``uv run isaaclab play`` with the relevant task and
``--checkpoint pretrained``.
