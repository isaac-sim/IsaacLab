.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _interactive-browser-demos:

Interactive Demos
=================

Explore physics and trained policies directly in your browser. Each card opens
one self-contained Newton simulation with camera controls, pause, and reset.
The simulation runs locally on the CPU through WebAssembly; no Isaac Lab
installation or remote simulation server is needed.

Open the `hosted gallery <../../../demos/interactive/>`_ to filter all 17 demos by
activity and physics: rigid bodies, VBD, or MPM.

These compact scenes illustrate individual effects. Use the full Isaac Lab
examples and tasks for evaluation, training, and measured comparisons.

Tuning
------

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: VBD material stiffness
      :img-top: ../_static/browser_demos/previews/stiffness.jpg
      :link: ../../../demos/interactive/?demo=stiffness

      .. _interactive-demo-stiffness:

      Compare three deformable cubes. Adjust shear, volume stiffness,
      damping, and gravity, then repeat the same drop.

   .. grid-item-card:: VBD cloth bending
      :img-top: ../_static/browser_demos/previews/cloth_bending.jpg
      :link: ../../../demos/interactive/?demo=cloth_bending

      .. _interactive-demo-cloth-bending:

      Compare cloth sheets draping over rollers. Change the middle sheet's
      bending stiffness and gravity.

   .. grid-item-card:: MJWarp sliding friction
      :img-top: ../_static/browser_demos/previews/rigid_friction.jpg
      :link: ../../../demos/interactive/?demo=rigid_friction

      .. _interactive-demo-rigid-friction:

      Change a box's friction on an incline and compare its sliding motion
      with two fixed material references.

   .. grid-item-card:: Joint PD tuning
      :img-top: ../_static/browser_demos/previews/joint_pd.jpg
      :link: ../../../demos/interactive/?demo=joint_pd

      .. _interactive-demo-joint-pd:

      Tune a suspended three-joint arm with step and sine commands.
      Compare target and measured angles on the live tracking plot.

   .. grid-item-card:: MPM material response
      :img-top: ../_static/browser_demos/previews/mpm.jpg
      :link: ../../../demos/interactive/?demo=mpm

      .. _interactive-demo-mpm:

      Drop 2,744 fine particles onto a cylinder inside a catch tub. Try sand,
      snow, clay, and water references, or tune individual parameters.

   .. grid-item-card:: VBD cable bending
      :img-top: ../_static/browser_demos/previews/cable_bending.jpg
      :link: ../../../demos/interactive/?demo=cable_bending

      .. _interactive-demo-cable-bending:

      Bend a clamped 0.45 m, ten-segment cable. Adjust bending stiffness
      and downward tip load, then reset to compare the response.

   .. grid-item-card:: Domino cascade
      :img-top: ../_static/browser_demos/previews/dominoes.jpg
      :link: ../../../demos/interactive/?demo=dominoes

      .. _interactive-demo-dominoes:

      Topple 24 slabs with a short push. Change friction and replay the
      cascade using Newton XPBD and the Isaac Lab checker ground.

For the corresponding native settings, see :ref:`newton-using-vbd`,
:ref:`mjwarp-solver-tuning`, :ref:`newton-tuning-mpm`, :ref:`overview-actuators`,
and :ref:`deformables`.
The MPM presets are qualitative references on a coarse grid, rather than
calibrated materials. Material changes apply on reset with fresh plastic history.

The cable and domino scenes isolate small comparisons from the native examples;
use the full examples for larger scenes and measured comparisons.

Policy inference
----------------

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: Cartpole balancing
      :img-top: ../_static/browser_demos/previews/cartpole.jpg
      :link: ../../../demos/interactive/?demo=cartpole

      .. _interactive-demo-cartpole:

      Push a pretrained balancing policy with up to 300 N in either
      direction. Release the slider to watch it recover.

   .. grid-item-card:: G1 velocity control
      :img-top: ../_static/browser_demos/previews/g1.jpg
      :link: ../../../demos/interactive/?demo=g1

      .. _interactive-demo-g1:

      Command forward, sideways, and turning motion with travel and yaw
      pads using the published WBC-AGILE policy.

   .. grid-item-card:: ANYmal-D velocity control
      :img-top: ../_static/browser_demos/previews/anymal.jpg
      :link: ../../../demos/interactive/?demo=anymal

      .. _interactive-demo-anymal:

      Drive an ANYmal-D with travel and yaw pads using a published
      Isaac Lab Newton checkpoint.

   .. grid-item-card:: MicroDuck walking
      :img-top: ../_static/browser_demos/previews/microduck.jpg
      :link: ../../../demos/interactive/?demo=microduck

      .. _interactive-demo-microduck:

      Walk in a bounded warehouse. Drag sticks to their edges to lock commands;
      pull back or press Escape to stop. Drag the robot for a 2 N push.

   .. grid-item-card:: Franka reach
      :img-top: ../_static/browser_demos/previews/franka_reach.jpg
      :link: ../../../demos/interactive/?demo=franka_reach

      .. _interactive-demo-franka-reach:

      Move the target position and watch a pretrained joint-position
      policy reach for it.

   .. grid-item-card:: Franka lift
      :img-top: ../_static/browser_demos/previews/franka_lift.jpg
      :link: ../../../demos/interactive/?demo=franka_lift

      .. _interactive-demo-franka-lift:

      Watch a pretrained policy lift a cube from a pregrasp reset.
      Adjust its target position, pause, or replay the episode.

   .. grid-item-card:: Franka drawer opening
      :img-top: ../_static/browser_demos/previews/franka_drawer.jpg
      :link: ../../../demos/interactive/?demo=franka_drawer

      .. _interactive-demo-franka-drawer:

      Play, pause, and reset a pretrained drawer-opening policy.
      Orbit the camera to inspect the grasp and drawer motion.

   .. grid-item-card:: Franka soft-body policy
      :img-top: ../_static/browser_demos/previews/franka_soft.jpg
      :link: ../../../demos/interactive/?demo=franka_soft

      .. _interactive-demo-franka-soft:

      Lift a flexible beam with a pretrained VBD policy. Change its target
      height or stiffness, or pull the beam to test recovery.

   .. grid-item-card:: Franka cloth policy
      :img-top: ../_static/browser_demos/previews/franka_cloth.jpg
      :link: ../../../demos/interactive/?demo=franka_cloth

      .. _interactive-demo-franka-cloth:

      Watch a pretrained VBD policy grasp a cloth sheet through contact.
      Adjust bending stiffness or target height, or pull the cloth.

Cartpole uses the published ``Isaac-Cartpole-Direct`` Newton policy. G1 uses
the external WBC-AGILE ``Velocity-G1-v0`` policy. ANYmal-D uses the
``Isaac-Velocity-Flat-AnymalD`` RSL-RL checkpoint with a tuned PD drive in
place of the native task's LSTM actuator, so its motion can differ from a full task run.

MicroDuck uses Pollen Robotics' published v5 ``velstand`` policy and native BAM servos
from `task PR #8270 <https://github.com/isaac-sim/IsaacLab/pull/8270>`_. Its warehouse
adds 12 movable props with nominal mass and inertia; other fixtures remain fixed.
Turning from rest includes a visible 0.25 m/s forward command. Playback uses one
captured reset with selected observation randomization disabled.

The five Franka policies use published Newton RSL-RL checkpoints for
``Isaac-Reach-Franka`` (``joint_pos``), ``Isaac-Lift-Franka`` (``cube``),
``Isaac-Open-Drawer-Franka``, ``Isaac-Lift-Soft-Franka``, and ``Isaac-Lift-Cloth-Franka``.
The VBD scenes use compact simulation meshes; changing their material tests policy
generalization. Drawer opening retains the captured task's legacy absolute-action
controller and clears its initial overlap through a short simulated retreat;
the current task uses a different EMA controller.

Franka policy playback repeats one captured reset every eight seconds for reach,
lift, and drawer opening, or five seconds for soft-body and cloth lifting.
Python episode and reset managers are not exported. See :ref:`reinforcement-learning`
and :ref:`environment-browser` for native playback, training, and full task options.

Scripted scenes
---------------

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: Franka cube stacking
      :img-top: ../_static/browser_demos/previews/franka_stack.jpg
      :link: ../../../demos/interactive/?demo=franka_stack

      .. _interactive-demo-franka-stack:

      Stack two 50 mm cubes with a scripted IK controller and simulated
      gripper contacts, or take over the arm in Manual mode.

Manual Franka control
---------------------

All six Franka demos offer **Manual** alongside **Policy** or **Scripted**.
Click the viewport, then use **WASD / arrow keys** to move relative to the camera,
**R/F** for height, **Q/E** for yaw, and **G** for the gripper where available.
Hold **Shift** for finer movement; **Escape** stops. Touch screens show a movement pad.
Inverse kinematics supplies actuator targets; the same simulation handles contacts.

Reset keeps the selected mode. Returning to Policy hands over the current state;
selecting Scripted restarts the stacking sequence. Manual mode continues until reset.

About these demos
-----------------

The demos are published alongside Isaac Lab documentation on GitHub Pages. The site serves
simulation bundles, policies, geometry, and third-party license notices.
Newton's captured Warp kernels advance the physics, Three.js displays the state,
and a shared JavaScript evaluator runs the policy networks in the browser.
The documentation gallery contains small thumbnails and links to these pages.

The exporters and prebuilt demo application are distributed through Isaac Lab's
`interactive-demos release <https://github.com/isaac-sim/IsaacLab/releases/tag/interactive-demos>`_.
The docs deployment downloads a reviewed, commit-named archive and verifies its checksum
before publishing. Compiled assets stay outside the development source tree.
The public `reference PR #7974 <https://github.com/isaac-sim/IsaacLab/pull/7974>`_
records the original implementation and scene adaptations.
Rebuilding requires the pinned internal Newton Web compiler and Emscripten.
The gallery uses a stable ``demos/interactive/`` URL. Earlier published version URLs
remain available.
