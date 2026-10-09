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

For particle-material comparisons and solver guidance, see :ref:`newton-tuning-mpm`.

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
       <aside class="admonition note demo-command-note" data-demo-note="h1-locomotion" hidden>
         <p class="admonition-title">H1 locomotion</p>
         <p>H1 locomotion uses a published policy. For autonomous H1 task playback, use <code>isaaclab play</code>.</p>
         <p data-demo-note-visualizers="newton_gl,newton_rtx" hidden>In the Newton viewer, press <kbd>N</kbd> to select a robot,
           <kbd>I</kbd>/<kbd>J</kbd>/<kbd>L</kbd> to walk forward or turn, <kbd>K</kbd> to stop, and <kbd>C</kbd> to toggle the follow camera.</p>
       </aside>
       <aside class="admonition note demo-command-note" data-demo-note="pick-and-place" hidden>
         <p class="admonition-title">Pick and place</p>
         <p>Pick and place requires Kit input.</p>
       </aside>
     </section>

     <div class="demo-card-grid" data-demo-list>
       <button type="button" class="demo-card is-selected" aria-pressed="true"
               data-demo-name="Zoo" data-demo-id="zoo"
               data-demo-physics="isaacsim_physx,newton_mjwarp"
               data-demo-visualizers="kit,newton_gl,newton_rtx,rerun,viser"
               data-demo-visualizers-isaacsim-physx="kit,newton_gl,rerun,viser"
               data-demo-visualizers-newton-mjwarp="kit,newton_gl,newton_rtx,rerun,viser"
               data-demo-description="Animate an arm, biped, quadruped, dexterous hand, quadcopter, and rigid props in one deterministic scene.">
         <img src="../../_static/demos/zoo.jpg" alt="Robots in the Isaac Lab Zoo demo" loading="lazy">
         <span>Zoo</span>
       </button>
       <button type="button" class="demo-card" aria-pressed="false"
               data-demo-name="H1 Locomotion" data-demo-id="h1-locomotion"
               data-demo-physics="isaacsim_physx,newton_mjwarp"
               data-demo-visualizers="kit,newton_gl,newton_rtx,rerun,viser"
               data-demo-visualizers-isaacsim-physx="kit,newton_gl,rerun,viser"
               data-demo-visualizers-newton-mjwarp="kit,newton_gl,newton_rtx,rerun,viser"
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
         <img src="../../_static/demos/newton_block_and_tackle.jpg" alt="Block and tackle pulleys and a red load" loading="lazy">
         <span>Newton Block and Tackle</span>
       </button>
       <button type="button" class="demo-card" aria-pressed="false"
               data-demo-name="Snowball Smash" data-demo-id="snowball-smash"
               data-demo-physics="newton_mpm" data-demo-fixed-physics="true"
               data-demo-visualizers="kit,newton_gl,newton_rtx,rerun,viser"
               data-demo-args="--device cuda:0"
               data-demo-description="Smash rigid crates with coupled MPM snowballs.">
         <img src="../../_static/demos/snowball_smash.jpg" alt="Snowballs striking a stack of colored crates" loading="lazy">
         <span>Snowball Smash</span>
       </button>
       <button type="button" class="demo-card" aria-pressed="false"
               data-demo-name="Teapot Fill" data-demo-id="teapot-fill"
               data-demo-physics="newton_mpm" data-demo-fixed-physics="true"
               data-demo-visualizers="kit,newton_gl,newton_rtx,rerun,viser"
               data-demo-args="--device cuda:0"
               data-demo-description="Fill a Utah teapot with MPM water particles and pour them into a bowl.">
         <img src="../../_static/demos/teapot_fill.jpg" alt="Utah teapot pouring simulated water" loading="lazy">
         <span>Teapot Fill</span>
       </button>
       <button type="button" class="demo-card" aria-pressed="false"
               data-demo-name="Rizon Sharpa Teapot" data-demo-id="rizon-sharpa-teapot"
               data-demo-physics="newton_coupler" data-demo-fixed-physics="true"
               data-demo-visualizers="none,newton_gl,newton_rtx"
               data-demo-args="--device cuda:0"
               data-demo-description="Grasp the stock teapot physically, pour while rising, and inspect water particles and their reconstructed surface.">
         <img src="../../_static/demos/teapot_fill.jpg" alt="Teapot and simulated water" loading="lazy">
         <span>Rizon Sharpa Teapot</span>
       </button>
     </div>
   </div>

Rizon--Sharpa teapot
--------------------

This additional demo requires access to the NVIDIA-internal
`Fabrics-Sim Rizon4s--Sharpa asset <https://gitlab-master.nvidia.com/dex/fabrics-sim/-/tree/d0dbd1ddaefc4996db546949a7dfb37e39afcbeb/src/fabrics_sim/models/robots/urdf/rizon4s_sharpa/rizon4s_sharpa_no_spheres>`__.
The asset is proprietary and is not included in Isaac Lab. Obtain the generated USD and its
``textures/`` directory from the asset maintainer under your applicable access and license terms.
Keep ``rizon4s_sharpa_no_spheres_generated.usd`` at the bundle root, beside ``textures/``;
the demo verifies both the USD and its 14 referenced textures against the pinned bundle digest
``ae5d22792b44fb6d29a7691d4276bc061a5529132f01e7a0eb5795a482595d63``.
Then point the demo at that directory:

.. code-block:: bash

   export ISAACLAB_FABRICS_SIM_RIZON_SHARPA_ROOT=/path/to/rizon4s_sharpa_no_spheres
   uv run isaaclab demo rizon-sharpa-teapot

Without this asset, the original ``uv run isaaclab demo teapot-fill`` remains available.
The robot demo's startup smoke test is skipped when the asset directory is not configured.

The Rizon--Sharpa teapot demo reuses the teapot-fill simulation with a physical index-finger
grasp. The pot weighs 450 g with proportionally scaled inertia, reducing rocking during
pickup while retaining the stock geometry, center of mass, and principal axes. Use
``--teapot_mass 0.35`` to restore its original mass. The pot reaches a 54-degree tilt
smoothly over four seconds. During the 70 cm rise
over twelve seconds, it eases to 49 degrees before the high pour. The initial fill is
``0.35`` (approximately 168 ml at the default particle resolution), leaving room below
the stock teapot's open rim. The longer rising pour gives the water time to drain before
the pot returns upright. Override ``--fill_level`` or ``--pour_rise_time`` to adjust these defaults.
The closeup camera shows reconstructed water for two seconds during the rising pour,
then returns to particles entering the bowl. The robot USD and textures
remain in the existing licensed asset cache; the demo does not redistribute them.
Select ``--grasp_finger middle`` to use the original grasp, or ``--presentation default``
for the original camera and fixed fluid-rendering mode. Adjust ``--pour_angle``,
``--pour_tilt_time``, ``--pour_upper_angle``, ``--pour_rise_time``, and ``--pour_aim_offset_x`` to change the pour;
validate modified trajectories with ``--motion_report``.
This report samples contacts at outer-step boundaries; when additional substeps are selected,
it does not measure the intervening rigid-step peaks.

MJWarp resolves robot and teapot contacts. By default, MPM receives the measured teapot
pose and velocity at every fluid substep, without returning fluid forces to the robot.
``--fluid_coupling two_way`` selects the experimental proxy path, which requires its own
complete-playback validation. Its supported-pot mass scaling is applied only within MPM.

The default outer and fluid frequency is 800 Hz. For timestep experiments,
``--physics_hz`` selects the outer frequency and ``--physics_substeps`` divides
each outer step into coupled fluid/rigid steps. ``--rigid_substeps`` adds contact
steps inside each coupled step without advancing the fluid again.
``--controller_hz`` must divide the outer frequency exactly.
With one-way coupling, arm position targets are interpolated at each rigid
substep, and the measured start/end teapot poses determine the collider's
center-of-mass and angular velocities for each fluid interval.
Lower fluid frequencies change settling and delivered water, so validate the
complete pour before selecting a faster configuration. ``--benchmark`` reports
the actual solver frequencies and simulated time per wall-clock second;
use ``--visualizer none`` to exclude rendering.
These timing options use the installed Newton solver; they do not enable the
unmerged MPM stabilization settings used in experimental recordings.

Record with Newton RTX:

.. code-block:: bash

   uv run --extra ovrtx --with imageio-ffmpeg isaaclab demo rizon-sharpa-teapot \
       --visualizer newton_rtx --video logs/teapot/pour_closeups.mp4
