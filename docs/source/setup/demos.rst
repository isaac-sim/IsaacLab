.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _demos:

Demos
=====

Demos are polished showcases of Isaac Lab capabilities. They ship in the ``isaaclab`` wheel, so you can run them
without a source checkout. Start with Zoo to see several robot families and simulation features in one scene, or run
``uvx isaaclab demo list`` to inspect the complete catalog.

All programs live under the repository-level ``examples/`` directory: curated showcases in ``examples/demos/``,
focused programs for learning an API or tuning a feature in directories such as ``examples/mpm/`` and
``examples/sensors/``, and shared data in ``examples/assets/``. List focused programs with
``uvx isaaclab example list`` and run one with ``uvx isaaclab example <name>``.

Demo and example commands show the same Isaac Lab startup screen as task playback while
their simulation initializes. Pass ``--info`` to keep startup messages visible.

In any packaged demo or example running with ``--viz newton_gl``, the **Isaac Lab Programs**
panel shows the core showcases first, with focused examples in a separate collapsed section.
Selecting one closes the current simulation and relaunches it with ``--viz newton_gl``.
For example, run ``uvx isaaclab demo zoo --viz newton_gl`` to explore the catalog.
Programs whose required modules are unavailable, hardware-dependent teleoperation, and
Kit-only demos are not shown.

Command Builder
---------------

.. raw:: html

   <div class="environment-browser demo-browser" data-demo-browser>
     <section class="environment-command-panel" aria-label="Isaac Lab demo command builder">
       <div class="environment-command-row environment-command-row-primary demo-command-row">
         <span class="environment-command-prefix" aria-hidden="true">uvx</span>
         <strong class="demo-command-selection" data-demo-name>Zoo</strong>
         <label class="environment-selector environment-selector-physics">
           <span>--physics</span>
           <select data-demo-field="physics" aria-label="Physics backend"></select>
         </label>
         <label class="environment-selector environment-selector-renderer">
           <span>--viz</span>
           <select data-demo-field="visualizer" aria-label="Visualizer"></select>
         </label>
       </div>
       <p class="demo-command-description" data-demo-description></p>
       <div class="environment-command-output">
         <code data-command-output></code>
         <div class="environment-command-actions">
           <span class="environment-copy-status" data-copy-status aria-live="polite"></span>
           <button type="button" class="environment-copy-button" data-copy-command
                   aria-label="Copy command" title="Copy command">
             <i class="fa-regular fa-copy" aria-hidden="true"></i>
           </button>
         </div>
       </div>
     </section>

     <div class="demo-card-grid" data-demo-list>
       <button type="button" class="demo-card is-selected" aria-pressed="true"
               data-demo-name="Zoo" data-demo-id="zoo"
               data-demo-physics="isaacsim_physx,newton_mjwarp"
               data-demo-visualizers="none,kit,newton_gl,newton_rtx,rerun,viser"
               data-demo-visualizers-isaacsim-physx="none,kit,newton_gl,rerun,viser"
               data-demo-visualizers-newton-mjwarp="none,kit,newton_gl,newton_rtx,rerun,viser"
               data-demo-description="Animate an arm, biped, quadruped, dexterous hand, quadcopter, and rigid props in one deterministic scene.">
         <img src="../../_static/demos/arms.jpg" alt="Robots in the Isaac Lab Zoo demo" loading="lazy">
         <span>Zoo</span>
       </button>
       <button type="button" class="demo-card" aria-pressed="false"
               data-demo-name="H1 Locomotion" data-demo-id="h1-locomotion"
               data-demo-physics="isaacsim_physx" data-demo-visualizers="kit"
               data-demo-description="Select H1 robots and control a trained rough-terrain policy with the keyboard and follow camera.">
         <img src="../../_static/demos/h1_locomotion.jpg" alt="H1 locomotion in Isaac Lab" loading="lazy">
         <span>H1 Locomotion</span>
       </button>
       <button type="button" class="demo-card" aria-pressed="false"
               data-demo-name="Pick and Place" data-demo-id="pick-and-place"
               data-demo-physics="isaacsim_physx" data-demo-visualizers="kit"
               data-demo-description="Interactively pick up a cube with a parallel robot and place it on a target.">
         <img src="../../_static/demos/pick_and_place.jpg" alt="Interactive pick and place demo" loading="lazy">
         <span>Pick and Place</span>
       </button>
       <button type="button" class="demo-card" aria-pressed="false"
               data-demo-name="Newton Block and Tackle" data-demo-id="newton-block-and-tackle"
               data-demo-physics="newton_vbd" data-demo-fixed-physics="true"
               data-demo-visualizers="newton_gl"
               data-demo-description="Drag a cable handle to lift a load through a 4:1 pulley system.">
         <span>Newton Block and Tackle</span>
       </button>
       <button type="button" class="demo-card" aria-pressed="false"
               data-demo-name="Snowball Smash" data-demo-id="snowball-smash"
               data-demo-physics="newton_mpm" data-demo-fixed-physics="true"
               data-demo-visualizers="none,kit,newton_gl,newton_rtx,rerun,viser"
               data-demo-args="--device cuda:0"
               data-demo-description="Smash rigid crates with coupled MPM snowballs.">
         <span>Snowball Smash</span>
       </button>
       <button type="button" class="demo-card" aria-pressed="false"
               data-demo-name="Teapot Fill" data-demo-id="teapot-fill"
               data-demo-physics="newton_mpm" data-demo-fixed-physics="true"
               data-demo-visualizers="none,kit,newton_gl,newton_rtx,rerun,viser"
               data-demo-args="--device cuda:0"
               data-demo-description="Fill a Utah teapot with MPM water particles and pour them into a bowl.">
         <img src="../../_static/demos/teapot_fill.jpg" alt="Utah teapot pouring simulated water" loading="lazy">
         <span>Teapot Fill</span>
       </button>
     </div>
   </div>

H1 locomotion uses a published policy and provides keyboard, mouse-selection, and camera controls
in Kit. Pick and place also requires Kit input. For autonomous H1 task playback, use ``isaaclab play``.
