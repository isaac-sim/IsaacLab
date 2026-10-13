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

Open the `hosted gallery <../../../demos/v3/>`_ to browse all demos.

These compact scenes illustrate individual effects. Use the full Isaac Lab
examples and tasks for evaluation, training, and measured comparisons.

Solver and actuator tuning
--------------------------

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: VBD material stiffness
      :img-top: ../_static/browser_demos/previews/stiffness.jpg
      :link: ../../../demos/v1/?demo=stiffness

      .. _interactive-demo-stiffness:

      Compare three deformable cubes. Adjust shear, volume stiffness,
      damping, and gravity, then repeat the same drop.

   .. grid-item-card:: VBD cloth bending
      :img-top: ../_static/browser_demos/previews/cloth_bending.jpg
      :link: ../../../demos/v1/?demo=cloth_bending

      .. _interactive-demo-cloth-bending:

      Compare cloth sheets draping over rollers. Change the middle sheet's
      bending stiffness and gravity.

   .. grid-item-card:: MJWarp sliding friction
      :img-top: ../_static/browser_demos/previews/rigid_friction.jpg
      :link: ../../../demos/v1/?demo=rigid_friction

      .. _interactive-demo-rigid-friction:

      Change a box's friction on an incline and compare its sliding motion
      with two fixed material references.

   .. grid-item-card:: Joint PD tuning
      :img-top: ../_static/browser_demos/previews/joint_pd.jpg
      :link: ../../../demos/v1/?demo=joint_pd

      .. _interactive-demo-joint-pd:

      Tune a suspended three-joint arm with step and sine commands.
      Compare target and measured angles on the live tracking plot.

   .. grid-item-card:: MPM material response
      :img-top: ../_static/browser_demos/previews/mpm.jpg
      :link: ../../../demos/v1/?demo=mpm

      .. _interactive-demo-mpm:

      Drop 1,728 particles onto a cylinder inside a catch tub. Try sand,
      snow, clay, and water references, or tune individual parameters.

For the corresponding native settings, see :ref:`newton-using-vbd`,
:ref:`mjwarp-solver-tuning`, :ref:`newton-tuning-mpm`, and :ref:`overview-actuators`.
The MPM presets are qualitative references on a coarse grid, rather than
calibrated materials. Material changes apply on reset with fresh plastic history.

Policy control
--------------

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: Cartpole balancing
      :img-top: ../_static/browser_demos/previews/cartpole.jpg
      :link: ../../../demos/v1/?demo=cartpole

      .. _interactive-demo-cartpole:

      Push a pretrained balancing policy with up to 300 N in either
      direction. Release the slider to watch it recover.

   .. grid-item-card:: G1 velocity control
      :img-top: ../_static/browser_demos/previews/g1.jpg
      :link: ../../../demos/v3/?demo=g1

      .. _interactive-demo-g1:

      Command forward, sideways, and turning motion with travel and yaw
      pads using the published WBC-AGILE policy.

   .. grid-item-card:: ANYmal-D velocity control
      :img-top: ../_static/browser_demos/previews/anymal.jpg
      :link: ../../../demos/v3/?demo=anymal

      .. _interactive-demo-anymal:

      Drive an ANYmal-D with travel and yaw pads using a published
      Isaac Lab Newton checkpoint.

   .. grid-item-card:: MicroDuck walking
      :img-top: ../_static/browser_demos/previews/microduck.jpg
      :link: ../../../demos/v3/?demo=microduck

      .. _interactive-demo-microduck:

      Hold the viewport joysticks to walk through a miniature courtyard.
      Drag the robot to apply a 2 N push; release to let the policy recover.

Cartpole uses the published ``Isaac-Cartpole-Direct`` Newton policy. G1 uses
the external 29-joint WBC-AGILE ``Velocity-G1-v0`` policy. ANYmal-D uses the
``Isaac-Velocity-Flat-AnymalD`` RSL-RL checkpoint with a tuned PD drive in
place of the native task's LSTM actuator. Its motion can differ from a full
task run. MicroDuck uses Pollen Robotics' published v5 ``velstand`` policy
and native BAM servos from `task PR #8270 <https://github.com/isaac-sim/IsaacLab/pull/8270>`_.
Its courtyard is decorative scenery over flat ground; playback uses one
captured reset with selected observation randomization disabled.
See :ref:`reinforcement-learning` for native playback and training.

Manipulation
------------

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: Franka reach
      :img-top: ../_static/browser_demos/previews/franka_reach.jpg
      :link: ../../../demos/v1/?demo=franka_reach

      .. _interactive-demo-franka-reach:

      Move the target position and watch a pretrained joint-position
      policy reach for it.

   .. grid-item-card:: Franka lift
      :img-top: ../_static/browser_demos/previews/franka_lift.jpg
      :link: ../../../demos/v1/?demo=franka_lift

      .. _interactive-demo-franka-lift:

      Watch a pretrained policy lift a cube from a pregrasp reset.
      Adjust its target position, pause, or replay the episode.

   .. grid-item-card:: Franka drawer opening
      :img-top: ../_static/browser_demos/previews/franka_drawer.jpg
      :link: ../../../demos/v1/?demo=franka_drawer

      .. _interactive-demo-franka-drawer:

      Play, pause, and reset a pretrained drawer-opening policy.
      Orbit the camera to inspect the grasp and drawer motion.

These three demos use published Newton RSL-RL checkpoints for ``Isaac-Reach-Franka``
(``joint_pos``), ``Isaac-Lift-Franka`` (``cube``), and ``Isaac-Open-Drawer-Franka``.
Playback repeats one captured reset every eight seconds; Python episode and
reset managers are not exported. Browse their full task options in
:ref:`environment-browser`.

About these demos
-----------------

The demos are published alongside Isaac Lab documentation on GitHub Pages. The site serves
versioned simulation bundles, policies, geometry, and third-party license notices.
Newton's captured Warp kernels advance the physics, Three.js displays the state,
and a shared JavaScript evaluator runs the policy networks in the browser.
The documentation gallery contains small thumbnails and links to these pages.

The exporters and demo application are maintained separately from Isaac Lab.
The docs deployment downloads a versioned archive from Isaac Lab GitHub releases
and verifies its checksum before publishing; compiled assets are kept outside the source tree.
The public `reference PR #7974 <https://github.com/isaac-sim/IsaacLab/pull/7974>`_
records the exporters, shared viewer, and scene adaptations.
Rebuilding requires the pinned internal Newton Web compiler and Emscripten.
The locomotion demos use the version 3 viewer with controls inside the viewport.
Other demos retain version 1 links. Each archive preserves earlier version directories.
