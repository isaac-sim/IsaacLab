.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _environment-browser:

Environments
============

Build a command and browse the core and contributed tasks available in Isaac Lab.

Command Builder
---------------

.. raw:: html

   <div class="environment-browser" data-environment-browser>
     <section class="environment-command-panel" aria-label="Isaac Lab command builder">
       <div class="environment-command-row environment-command-row-primary">
         <span class="environment-command-prefix" aria-hidden="true">uv run isaaclab</span>
         <div class="environment-mode-switch" role="group" aria-label="Command mode">
           <button type="button" class="is-active" data-command-mode="train" aria-pressed="true">Train</button>
           <button type="button" data-command-mode="play" aria-pressed="false">Play</button>
         </div>
         <label class="environment-inline-field">
           <span>--rl_library</span>
           <select data-environment-field="rl" aria-label="RL library"></select>
         </label>
         <label class="environment-inline-field environment-task-field">
           <span>--task</span>
           <select data-environment-field="task" aria-label="Task"></select>
         </label>
       </div>
       <div class="environment-command-row environment-command-row-options">
         <label class="environment-selector environment-selector-physics">
           <span>physics=</span>
           <select data-environment-field="physics" aria-label="Physics preset"></select>
         </label>
         <label class="environment-selector environment-selector-renderer">
           <span>renderer=</span>
           <select data-environment-field="renderer" aria-label="Renderer preset"></select>
         </label>
         <label class="environment-selector environment-selector-preset">
           <span>presets=</span>
           <select data-environment-field="presets" aria-label="Domain preset"></select>
         </label>
         <label class="environment-checkpoint-toggle">
           <input type="checkbox" data-environment-field="checkpoint">
           <span>Use Pre-trained Checkpoint</span>
         </label>
       </div>
       <p class="environment-non-rl-note" data-non-rl-note role="status" hidden>
         <i class="fa-solid fa-circle-info" aria-hidden="true"></i>
         This environment does not support RL training or playback. The command runs it with the
         zero-action agent instead.
       </p>
       <p class="environment-backend-warning" data-backend-warning role="status" hidden>
         This combination of physics, renderer, and preset can't launch. Choose a different option.
       </p>
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
   </div>

Task Preview
------------

.. raw:: html

   <div class="environment-browser" data-environment-preview>
     <section class="environment-preview-panel" aria-live="polite">
       <div class="environment-preview-stage">
         <video data-preview-video muted autoplay loop playsinline preload="metadata" hidden></video>
         <img data-preview-image
              src="../../_static/environment-previews/cartpole.jpg"
              alt="Isaac-Cartpole preview">
         <div class="environment-preview-caption">
           <span class="environment-preview-label">Selected task</span>
           <strong data-preview-task>Isaac-Cartpole</strong>
         </div>
       </div>
       <dl class="environment-preview-details">
         <div><dt>Mode</dt><dd data-preview-mode>Train</dd></div>
         <div><dt>RL Library</dt><dd data-preview-rl>rsl_rl</dd></div>
         <div><dt>Physics</dt><dd data-preview-physics>newton_mjwarp</dd></div>
         <div><dt>Renderer</dt><dd data-preview-renderer>Default</dd></div>
         <div><dt>Preset</dt><dd data-preview-presets>Default</dd></div>
         <div><dt>VRAM</dt><dd data-preview-vram>Not available</dd></div>
       </dl>
     </section>
   </div>

Benchmarks
----------

.. isaaclab-benchmark-data::

.. raw:: html

   <div class="environment-browser" data-environment-benchmarks
        data-benchmark-release-dates="2026-09-09:EA 3.0,2026-10-12:GA 3.0"
        data-benchmark-develop-dates="2026-08-23,2026-08-30,2026-09-06,2026-09-13,2026-09-20,2026-09-27,2026-10-04,2026-10-11">
     <section class="environment-benchmark-panel">
       <div class="environment-benchmark-toolbar">
         <div class="environment-benchmark-switch" role="group" aria-label="Benchmark channel">
           <button type="button" data-benchmark-channel="release" aria-pressed="false">Release</button>
           <button type="button" class="is-active" data-benchmark-channel="develop" aria-pressed="true">Develop</button>
         </div>
         <div class="environment-benchmark-switch" role="group" aria-label="Benchmark workload">
           <button type="button" class="is-active" data-benchmark-workload="collection" aria-pressed="true">Collection</button>
           <button type="button" data-benchmark-workload="training" aria-pressed="false">Training</button>
         </div>
       </div>
       <div class="environment-benchmark-results">
         <div class="environment-benchmark-chart" data-benchmark-chart></div>
         <div class="environment-benchmark-table" data-benchmark-table tabindex="0"
              role="region" aria-label="Benchmark FPS measurements, scroll for more configurations" hidden></div>
       </div>
       <div class="environment-benchmark-stub" data-benchmark-empty hidden>
         <i class="fa-solid fa-chart-line" aria-hidden="true"></i>
         <div>
           <strong>Benchmark data is not available yet</strong>
         </div>
       </div>
       <div class="environment-benchmark-error" data-benchmark-error hidden>Benchmark data could not be loaded.</div>
     </section>
   </div>

Available Tasks
---------------

.. raw:: html

   <div class="environment-browser" data-environment-tasks>
     <div class="environment-task-toolbar">
       <label class="environment-task-search">
         <i class="fa-solid fa-magnifying-glass" aria-hidden="true"></i>
         <span class="visually-hidden">Search tasks and capabilities</span>
         <input type="search" data-task-search placeholder="Search tasks or capabilities" autocomplete="off">
       </label>
       <label class="environment-task-filter">
         <span class="visually-hidden">Task category</span>
         <select data-task-category aria-label="Task category">
           <option value="all">All categories</option>
           <option value="classic">Classic control</option>
           <option value="manipulation">Manipulation</option>
           <option value="locomotion">Locomotion</option>
         </select>
       </label>
       <div class="environment-scope-switch" role="group" aria-label="Task collection">
         <button type="button" class="is-active" data-task-scope="core" aria-pressed="true">Core</button>
         <button type="button" data-task-scope="contrib" aria-pressed="false">Contrib</button>
         <button type="button" data-task-scope="warp" aria-pressed="false">Warp</button>
       </div>
       <span class="environment-task-count" data-task-count></span>
     </div>
     <div class="environment-task-list" data-task-list></div>
     <p class="environment-empty-state" data-task-empty hidden>No tasks match this search.</p>
   </div>

NIST assembly variants
----------------------

``IsaacContrib-Factory-Variant-Franka`` trains a single-component policy across
twenty assembly pairs. Each environment contains one fixed part and one movable
part. Resets select a matching pair and restore its geometry, inertial properties,
and cached starting state. This task does not assemble several components in one
episode.

The variant task selects Newton MJWarp internally; no ``physics=`` selector is
needed. For example::

   uv run isaaclab train --task IsaacContrib-Factory-Variant-Franka --rl_library rsl_rl
   uv run isaaclab play --task IsaacContrib-Factory-Variant-Franka --rl_library rsl_rl --checkpoint /path/to/model.pt --visualizer newton_gl

Training samples cached loose-part, grasped, and partially assembled states with
beta sampling. Play mode uses uniformly sampled random-start states. Checkpoints
must match this task's observation and policy configuration; multi-component
checkpoints are not interchangeable with this task.

Point-cloud samples also form part of the checkpoint's observation configuration.
Matching the random seed alone is insufficient if collider geometry or its sampling
order has changed. The ``scene_point_cloud`` term in the ``perception`` observation
group accepts these parameters:

* ``sampling_seed``: the point sampler seed, independent of the environment reset
  seed. The default is 42; robot body streams use this seed plus the body index.
* ``sampling_variant_order``: a full permutation of the scene's assembly names.
  Points are sampled in this order, then mapped back to the live variant IDs.
  When adding variants to a checkpoint's catalog, place them after the original
  names to preserve the original collider sampling streams. The default uses the
  scene's catalog order.
* ``robot_geometry_usd_path``: an optional USD geometry source for robot observation
  points. It must retain the trained rigid-body names and local frames. This does
  not change the physical robot; the default samples its current asset.

These settings regenerate points from USD geometry at initialization, without a
saved point-cloud tensor. They do not compensate for other changes to a
checkpoint's observation or action configuration.

Reset-time geometry switching uses Newton's external contact pipeline with
sleeping disabled. Mesh resources are loaded once; changing the selected variant
does not require a custom Newton solver or its precaching API. MuJoCo's compiled
geometry is not replaced, so internal MuJoCo contacts and geometry-dependent
MuJoCo queries are not supported for switched assets. Newton OpenGL and Viser
display the selected variants; RTX rendering of switched variants is not supported.

The existing ``IsaacContrib-Factory-Franka`` task remains available for a fixed
assembly chosen through a domain preset, such as ``presets=peg_insert_8mm``.
