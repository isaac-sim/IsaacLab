.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _demos:

Demos
=====

Explore focused scripts that demonstrate Isaac Lab's robots, objects, sensors, and simulation features.
Choose a demo card, then select a supported physics backend and visualizer to build a ready-to-run command.
The command automatically includes the optional dependency groups required by the selection.

Command Builder
---------------

.. raw:: html

   <div class="environment-browser demo-browser" data-demo-browser>
     <section class="environment-command-panel" aria-label="Isaac Lab demo command builder">
       <div class="environment-command-row environment-command-row-primary demo-command-row">
         <span class="environment-command-prefix" aria-hidden="true">uv run</span>
         <strong class="demo-command-selection" data-demo-name>Arms</strong>
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
               data-demo-name="Arms" data-demo-path="scripts/demos/arms.py"
               data-demo-physics="isaacsim_physx,newton_mjwarp"
               data-demo-visualizers="none,kit,newton_gl,newton_rtx,rerun,viser"
               data-demo-description="Spawn different robot arms and apply random joint-position commands.">
         <img src="../../_static/demos/arms.jpg" alt="Robot arms in Isaac Lab" loading="lazy">
         <span>Arms</span>
       </button>
       <button type="button" class="demo-card" aria-pressed="false"
               data-demo-name="Bipeds" data-demo-path="scripts/demos/bipeds.py"
               data-demo-physics="isaacsim_physx,newton_mjwarp"
               data-demo-visualizers="none,kit,newton_gl,newton_rtx,rerun,viser"
               data-demo-description="Spawn a collection of biped robots.">
         <img src="../../_static/demos/bipeds.jpg" alt="Biped robots in Isaac Lab" loading="lazy">
         <span>Bipeds</span>
       </button>
       <button type="button" class="demo-card" aria-pressed="false"
               data-demo-name="Cables" data-demo-path="scripts/demos/cables.py"
               data-demo-physics="newton_vbd"
               data-demo-visualizers="none,kit,newton_gl,newton_rtx,rerun,viser"
               data-demo-description="Simulate a pile of colliding cables with Newton VBD.">
         <img src="../../_static/demos/cables.jpg" alt="Cable pile simulated with Newton VBD" loading="lazy">
         <span>Cables</span>
       </button>
       <button type="button" class="demo-card" aria-pressed="false"
               data-demo-name="Deformable Objects" data-demo-path="scripts/demos/deformables.py"
               data-demo-physics="isaacsim_physx,newton_vbd,ovphysx"
               data-demo-visualizers="none,kit,newton_gl,newton_rtx,rerun,viser"
               data-demo-visualizers-isaacsim-physx="none,kit"
               data-demo-visualizers-newton-vbd="none,kit,newton_gl,newton_rtx,rerun,viser"
               data-demo-visualizers-ovphysx="none" data-demo-extras="tetrahedralization"
               data-demo-description="Drop deformable primitive shapes from a height.">
         <img src="../../_static/demos/deformables.jpg" alt="Deformable objects in Isaac Lab" loading="lazy">
         <span>Deformable Objects</span>
       </button>
       <button type="button" class="demo-card" aria-pressed="false"
               data-demo-name="Teapot Fill" data-demo-path="scripts/demos/mpm/teapot_fill.py"
               data-demo-physics="newton_mpm" data-demo-fixed-physics="true"
               data-demo-visualizers="none,kit,newton_gl,newton_rtx,rerun,viser"
               data-demo-args="--device cuda:0"
               data-demo-description="Fill a Utah teapot with MPM water particles and pour them into a bowl.">
         <img src="../../_static/demos/teapot_fill.jpg" alt="Utah teapot pouring simulated water" loading="lazy">
         <span>Teapot Fill</span>
       </button>
       <button type="button" class="demo-card" aria-pressed="false"
               data-demo-name="H1 Locomotion" data-demo-path="scripts/demos/h1_locomotion.py"
               data-demo-physics="isaacsim_physx" data-demo-visualizers="kit"
               data-demo-description="Interactively control a trained H1 rough-terrain locomotion policy with the keyboard.">
         <img src="../../_static/demos/h1_locomotion.jpg" alt="H1 locomotion in Isaac Lab" loading="lazy">
         <span>H1 Locomotion</span>
       </button>
       <button type="button" class="demo-card" aria-pressed="false"
               data-demo-name="Dexterous Hands" data-demo-path="scripts/demos/hands.py"
               data-demo-physics="isaacsim_physx,newton_mjwarp"
               data-demo-visualizers="none,kit,newton_gl,newton_rtx,rerun,viser"
               data-demo-description="Spawn dexterous hands and command them to open and close.">
         <img src="../../_static/demos/hands.jpg" alt="Dexterous hands in Isaac Lab" loading="lazy">
         <span>Dexterous Hands</span>
       </button>
       <button type="button" class="demo-card" aria-pressed="false"
               data-demo-name="Markers" data-demo-path="scripts/demos/markers.py"
               data-demo-physics="isaacsim_physx"
               data-demo-visualizers="none,kit,newton_gl,newton_rtx,rerun,viser"
               data-demo-description="Create reusable visualization markers.">
         <img src="../../_static/demos/markers.jpg" alt="Visualization markers in Isaac Lab" loading="lazy">
         <span>Markers</span>
       </button>
       <button type="button" class="demo-card" aria-pressed="false"
               data-demo-name="Multi-Asset Scene" data-demo-path="scripts/demos/multi_asset.py"
               data-demo-physics="isaacsim_physx,newton_mjwarp"
               data-demo-visualizers="none,kit,newton_gl,newton_rtx,rerun,viser"
               data-demo-description="Spawn varying assets in separate environments of one interactive scene.">
         <img src="../../_static/demos/multi_asset.jpg" alt="Multiple assets in one simulation" loading="lazy">
         <span>Multi-Asset Scene</span>
       </button>
       <button type="button" class="demo-card" aria-pressed="false"
               data-demo-name="Heterogeneous Scene" data-demo-path="scripts/demos/heterogeneous_scene.py"
               data-demo-physics="isaacsim_physx"
               data-demo-visualizers="none,kit,newton_gl,newton_rtx,rerun,viser"
               data-demo-description="Compose several task scenes into one heterogeneous cloned simulation.">
         <img src="../../_static/demos/heterogeneous_scene.jpg" alt="Heterogeneous task scenes in Isaac Lab" loading="lazy">
         <span>Heterogeneous Scene</span>
       </button>
       <button type="button" class="demo-card" aria-pressed="false"
               data-demo-name="Bin Packing" data-demo-path="scripts/demos/bin_packing.py"
               data-demo-physics="isaacsim_physx"
               data-demo-visualizers="none,kit,newton_gl,newton_rtx,rerun,viser"
               data-demo-description="Demonstrate bin packing with a rigid-object collection.">
         <img src="../../_static/demos/bin_packing.jpg" alt="Randomized objects in bins" loading="lazy">
         <span>Bin Packing</span>
       </button>
       <button type="button" class="demo-card" aria-pressed="false"
               data-demo-name="Pick and Place" data-demo-path="scripts/demos/pick_and_place.py"
               data-demo-physics="isaacsim_physx" data-demo-visualizers="kit"
               data-demo-description="Interactively pick up a cube with a parallel robot and place it on a target.">
         <img src="../../_static/demos/pick_and_place.jpg" alt="Interactive pick and place demo" loading="lazy">
         <span>Pick and Place</span>
       </button>
       <button type="button" class="demo-card" aria-pressed="false"
               data-demo-name="Haply Teleoperation" data-demo-path="scripts/demos/haply_teleoperation.py"
               data-demo-physics="isaacsim_physx,newton_mjwarp"
               data-demo-visualizers="none,kit,newton_gl,newton_rtx,rerun,viser"
               data-demo-extras="teleop"
               data-demo-args="--websocket_uri ws://localhost:10001 --pos_sensitivity 1.65"
               data-demo-description="Teleoperate a Franka Panda with Haply Inverse3 and VerseGrip hardware.">
         <img src="../../_static/demos/haply_teleop_franka.jpg" alt="Haply teleoperation with force feedback" loading="lazy">
         <span>Haply Teleoperation</span>
       </button>
       <button type="button" class="demo-card" aria-pressed="false"
               data-demo-name="Procedural Terrain" data-demo-path="scripts/demos/procedural_terrain.py"
               data-demo-physics="isaacsim_physx"
               data-demo-visualizers="none,kit,newton_gl,newton_rtx,rerun,viser"
               data-demo-description="Create and spawn procedurally generated terrain configurations.">
         <img src="../../_static/demos/procedural_terrain.jpg" alt="Procedurally generated terrain" loading="lazy">
         <span>Procedural Terrain</span>
       </button>
       <button type="button" class="demo-card" aria-pressed="false"
               data-demo-name="Quadcopter" data-demo-path="scripts/demos/quadcopter.py"
               data-demo-physics="isaacsim_physx,newton_mjwarp"
               data-demo-visualizers="none,kit,newton_gl,newton_rtx,rerun,viser"
               data-demo-description="Spawn a quadcopter in the default environment.">
         <img src="../../_static/demos/quadcopter.jpg" alt="Quadcopter in Isaac Lab" loading="lazy">
         <span>Quadcopter</span>
       </button>
       <button type="button" class="demo-card" aria-pressed="false"
               data-demo-name="Quadrupeds" data-demo-path="scripts/demos/quadrupeds.py"
               data-demo-physics="isaacsim_physx,newton_mjwarp"
               data-demo-visualizers="none,kit,newton_gl,newton_rtx,rerun,viser"
               data-demo-description="Spawn quadruped robots and hold standing poses with position commands.">
         <img src="../../_static/demos/quadrupeds.jpg" alt="Quadruped robots in Isaac Lab" loading="lazy">
         <span>Quadrupeds</span>
       </button>
       <button type="button" class="demo-card" aria-pressed="false"
               data-demo-name="Multi-Mesh Ray Caster" data-demo-path="scripts/demos/sensors/multi_mesh_raycaster.py"
               data-demo-physics="isaacsim_physx,newton_mjwarp"
               data-demo-visualizers="none,kit,newton_gl,newton_rtx,rerun,viser"
               data-demo-visualizers-isaacsim-physx="none,kit,newton_gl,newton_rtx,rerun,viser"
               data-demo-visualizers-newton-mjwarp="none,newton_gl,newton_rtx,rerun,viser"
               data-demo-args="--num_envs 16 --asset_type objects"
               data-demo-description="Cast rays against multiple meshes with Warp kernels.">
         <img src="../../_static/demos/multi-mesh-raycast.jpg" alt="Multi-mesh ray casting in Isaac Lab" loading="lazy">
         <span>Multi-Mesh Ray Caster</span>
       </button>
     </div>
   </div>

The H1 locomotion and pick-and-place demos require interactive keyboard or mouse input. Haply teleoperation requires
Inverse3 and VerseGrip devices and a running Haply WebSocket service. Cables are a Newton-only asset; see
:doc:`../concepts/deformables` for details.
