.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _interactive-browser-demos:

Interactive Demos
=================

Explore physics and pretrained policies without installing Isaac Lab. Choose a
scene to run its simulation here, adjust its controls, and pause or reset at any
point. Only the selected demo loads. Physics runs locally on your CPU through
WebAssembly, and policy inference also runs in your browser.

.. isaaclab-demos::
   :title: Explore interactive demos

.. dropdown:: Demo index

   **Tuning and mechanisms**

   .. _interactive-demo-stiffness:

   **VBD material stiffness** — Compare deformable cubes while changing shear
   stiffness, volume stiffness, damping, and gravity.

   .. _interactive-demo-cloth-bending:

   **VBD cloth bending** — Compare sheets draping over rollers while changing
   bending stiffness and gravity.

   .. _interactive-demo-rigid-friction:

   **MJWarp sliding friction** — Change friction on an incline and compare the
   sliding box with two fixed material references.

   .. _interactive-demo-joint-pd:

   **Joint PD tuning** — Tune a suspended three-joint arm with step and sine
   commands, and compare target and measured angles on a live plot.

   .. _interactive-demo-mpm:

   **MPM material response** — Drop 2,744 particles into a tub; try qualitative
   sand, snow, clay, and water references, or tune individual material parameters.

   .. _interactive-demo-cable-bending:

   **VBD cable bending** — Bend a clamped 0.45 m cable by changing its bending
   stiffness and downward tip load.

   .. _interactive-demo-pulley:

   **Pulley workshop** — Choose the wheel count and payload, pull the handle,
   and compare ideal force with the cable reaction. Choose a massless rope, a
   VBD cable, or a Newton tendon with friction and stretch. VBD and Newton
   offer one, two, or four wheels; the massless rope also offers six.

   .. _interactive-demo-dominoes:

   **Domino cascade** — Topple 24 slabs with a short push, change friction, and
   replay their simulated contacts.

   **Policy inference**

   .. _interactive-demo-cartpole:

   **Cartpole balancing** — Perturb a pretrained balancing policy with up to
   300 N in either direction, then release the slider to watch recovery.

   .. _interactive-demo-g1:

   **G1 velocity control** — Command forward, sideways, and turning motion
   with travel and yaw pads using the published WBC-AGILE policy.

   .. _interactive-demo-anymal:

   **ANYmal-D velocity control** — Drive an ANYmal-D using a published Isaac
   Lab Newton checkpoint and a PD approximation of its learned actuator.

   .. _interactive-demo-microduck:

   **MicroDuck walking** — Walk through a bounded warehouse with travel and
   yaw sticks, or drag the robot to apply a 2 N perturbation.

   .. _interactive-demo-franka-reach:

   .. _interactive-demo-ur10-reach:

   **UR10 reach** — Move target sliders to test a pretrained reach policy, or
   switch to Manual end-effector control with inverse kinematics.

   .. _interactive-demo-franka-lift:

   **Franka lift** — Watch a pretrained policy lift a cube from a pregrasp
   reset, adjust its target, or control the arm manually.

   .. _interactive-demo-franka-soft:

   **Franka soft-body policy** — Grasp and lift a flexible beam through
   contact. Change stiffness or target height, or use Manual control.

   .. _interactive-demo-franka-cloth:

   **Franka cloth policy** — Grasp a VBD sheet with a pretrained policy.
   Change bending stiffness or target height, or use Manual control.

   .. _interactive-demo-franka-cable:

   **Franka cable policy** — Grasp and lift a 12-segment VBD cable with a
   published policy, change target height, or use Manual control.

   .. _interactive-demo-alphanist:

   **AlphaNIST peg insertion** — Watch a pretrained policy insert one round
   and two rectangular pegs, or drag a loose part to perturb its approach.
   Small convex pegs and compound socket walls simplify the contacts.

   **Scripted manipulation**

   .. _interactive-demo-franka-stack:

   **Franka cube stacking** — Stack two 50 mm cubes with scripted IK and
   simulated gripper contacts, or take over in Manual mode.

About these demos
-----------------

These compact scenes illustrate physics and control. Scene preparation and
adapted policy playback can differ from the native tasks; use the full
:ref:`environments <environment-browser>` for training and measured comparisons.

The gallery runtime is published in the official
`interactive-demos release <https://github.com/isaac-sim/IsaacLab/releases/tag/interactive-demos>`_.
Earlier snapshots and their exporter sources are also retained there.
