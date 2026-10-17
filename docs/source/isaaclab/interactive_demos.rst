.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _interactive-browser-demos:

Interactive Demos
=================

Explore physics and trained policies in your browser—no installation required.
Select a demo below to get started.

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: VBD material stiffness
      :img-top: ../_static/browser_demos/previews/stiffness.jpg
      :link: ../../../demos/interactive/?demo=stiffness

      .. _interactive-demo-stiffness:

      :bdg-info:`Tuning`

      Compare three deformable cubes. Adjust shear, volume stiffness,
      damping, and gravity, then repeat the same drop.

   .. grid-item-card:: VBD cloth bending
      :img-top: ../_static/browser_demos/previews/cloth_bending.jpg
      :link: ../../../demos/interactive/?demo=cloth_bending

      .. _interactive-demo-cloth-bending:

      :bdg-info:`Tuning`

      Compare cloth sheets draping over rollers. Change the middle sheet's
      bending stiffness and gravity.

   .. grid-item-card:: MJWarp sliding friction
      :img-top: ../_static/browser_demos/previews/rigid_friction.jpg
      :link: ../../../demos/interactive/?demo=rigid_friction

      .. _interactive-demo-rigid-friction:

      :bdg-info:`Tuning`

      Change a box's friction on an incline and compare its sliding motion
      with two fixed material references.

   .. grid-item-card:: Joint PD tuning
      :img-top: ../_static/browser_demos/previews/joint_pd.jpg
      :link: ../../../demos/interactive/?demo=joint_pd

      .. _interactive-demo-joint-pd:

      :bdg-info:`Tuning`

      Tune a suspended three-joint arm with step and sine commands.
      Compare target and measured angles on the live tracking plot.

   .. grid-item-card:: MPM material response
      :img-top: ../_static/browser_demos/previews/mpm.jpg
      :link: ../../../demos/interactive/?demo=mpm

      .. _interactive-demo-mpm:

      :bdg-info:`Tuning`

      Drop 2,744 fine particles onto a cylinder inside a catch tub. Try sand,
      snow, clay, and water references, or tune individual parameters.

   .. grid-item-card:: VBD cable bending
      :img-top: ../_static/browser_demos/previews/cable_bending.jpg
      :link: ../../../demos/interactive/?demo=cable_bending

      .. _interactive-demo-cable-bending:

      :bdg-info:`Tuning`

      Bend a clamped 0.45 m, ten-segment cable. Adjust bending stiffness
      and downward tip load, then reset to compare the response.

   .. grid-item-card:: Domino cascade
      :img-top: ../_static/browser_demos/previews/dominoes.jpg
      :link: ../../../demos/interactive/?demo=dominoes

      .. _interactive-demo-dominoes:

      :bdg-info:`Tuning`

      Topple 24 slabs with a short push. Change friction and replay the
      cascade using Newton XPBD and the Isaac Lab checker ground.

   .. grid-item-card:: Cartpole balancing
      :img-top: ../_static/browser_demos/previews/cartpole.jpg
      :link: ../../../demos/interactive/?demo=cartpole

      .. _interactive-demo-cartpole:

      :bdg-success:`Policy`

      Push a pretrained balancing policy with up to 300 N in either
      direction. Release the slider to watch it recover.

   .. grid-item-card:: G1 velocity control
      :img-top: ../_static/browser_demos/previews/g1.jpg
      :link: ../../../demos/interactive/?demo=g1

      .. _interactive-demo-g1:

      :bdg-success:`Policy`

      Command forward, sideways, and turning motion with travel and yaw
      pads using the published WBC-AGILE policy.

   .. grid-item-card:: ANYmal-D velocity control
      :img-top: ../_static/browser_demos/previews/anymal.jpg
      :link: ../../../demos/interactive/?demo=anymal

      .. _interactive-demo-anymal:

      :bdg-success:`Policy`

      Drive an ANYmal-D with travel and yaw pads using a published
      Isaac Lab Newton checkpoint.

   .. grid-item-card:: MicroDuck walking
      :img-top: ../_static/browser_demos/previews/microduck.jpg
      :link: ../../../demos/interactive/?demo=microduck

      .. _interactive-demo-microduck:

      :bdg-success:`Policy`

      Walk in a bounded warehouse. Drag sticks to their edges to lock commands;
      pull back or press Escape to stop. Drag the robot for a 2 N push.

   .. grid-item-card:: Franka reach
      :img-top: ../_static/browser_demos/previews/franka_reach.jpg
      :link: ../../../demos/interactive/?demo=franka_reach

      .. _interactive-demo-franka-reach:

      :bdg-success:`Policy`

      Move the target position and watch a pretrained joint-position
      policy reach for it.

   .. grid-item-card:: Franka lift
      :img-top: ../_static/browser_demos/previews/franka_lift.jpg
      :link: ../../../demos/interactive/?demo=franka_lift

      .. _interactive-demo-franka-lift:

      :bdg-success:`Policy`

      Watch a pretrained policy lift a cube from a pregrasp reset.
      Adjust its target position, pause, or replay the episode.

   .. grid-item-card:: Franka drawer opening
      :img-top: ../_static/browser_demos/previews/franka_drawer.jpg
      :link: ../../../demos/interactive/?demo=franka_drawer

      .. _interactive-demo-franka-drawer:

      :bdg-success:`Policy`

      Play, pause, and reset a pretrained drawer-opening policy.
      Orbit the camera to inspect the grasp and drawer motion.

   .. grid-item-card:: Franka soft-body policy
      :img-top: ../_static/browser_demos/previews/franka_soft.jpg
      :link: ../../../demos/interactive/?demo=franka_soft

      .. _interactive-demo-franka-soft:

      :bdg-success:`Policy`

      Lift a flexible beam with a pretrained VBD policy. Change its target
      height or stiffness, or pull the beam to test recovery.

   .. grid-item-card:: Franka cloth policy
      :img-top: ../_static/browser_demos/previews/franka_cloth.jpg
      :link: ../../../demos/interactive/?demo=franka_cloth

      .. _interactive-demo-franka-cloth:

      :bdg-success:`Policy`

      Watch a pretrained VBD policy grasp a cloth sheet through contact.
      Adjust bending stiffness or target height, or pull the cloth.

   .. grid-item-card:: Franka cube stacking
      :img-top: ../_static/browser_demos/previews/franka_stack.jpg
      :link: ../../../demos/interactive/?demo=franka_stack

      .. _interactive-demo-franka-stack:

      :bdg-secondary:`Scripted`

      Stack two 50 mm cubes with a scripted IK controller and simulated
      gripper contacts, or take over the arm in Manual mode.

About these demos
-----------------

Each demo runs a Newton simulation locally on your CPU through WebAssembly, with camera controls,
pause, and reset. Simplified scenes and adapted policy playback can behave differently from native tasks.
Use the full Isaac Lab :ref:`environments <environment-browser>` for training and measured comparisons.

Simulation bundles, policy checkpoints, exporters, and third-party license notices are distributed
through the `interactive-demos release <https://github.com/isaac-sim/IsaacLab/releases/tag/interactive-demos>`_.
See the `original implementation <https://github.com/isaac-sim/IsaacLab/pull/7974>`_.
