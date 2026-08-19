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
         <img data-preview-image src="../../_images/cartpole.jpg" alt="Isaac-Cartpole preview">
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

.. raw:: html

   <div class="environment-browser" data-environment-benchmarks
        data-benchmark-source="../../_static/environment-performance.csv">
     <section class="environment-benchmark-panel">
       <div class="environment-benchmark-toolbar">
         <div class="environment-benchmark-legend" aria-label="Benchmark series"></div>
         <div class="environment-benchmark-switch" role="group" aria-label="Benchmark workload">
           <button type="button" class="is-active" data-benchmark-workload="runtime" aria-pressed="true">Collection</button>
           <button type="button" data-benchmark-workload="training" aria-pressed="false">Training</button>
         </div>
       </div>
       <div class="environment-benchmark-chart" data-benchmark-chart></div>
       <div class="environment-benchmark-stub" data-benchmark-empty hidden>
         <i class="fa-solid fa-chart-line" aria-hidden="true"></i>
         <div>
           <strong>Benchmark data is not available yet</strong>
           <p>Performance history will appear here when benchmark collection is enabled.</p>
         </div>
       </div>
       <p class="environment-benchmark-error" data-benchmark-error hidden>Benchmark data could not be loaded.</p>
     </section>
   </div>

Available Tasks
---------------

.. raw:: html

   <div class="environment-browser" data-environment-tasks>
     <div class="environment-task-toolbar">
       <label class="environment-task-search">
         <i class="fa-solid fa-magnifying-glass" aria-hidden="true"></i>
         <span class="visually-hidden">Search tasks</span>
         <input type="search" data-task-search placeholder="Search tasks" autocomplete="off">
       </label>
       <label class="environment-task-filter">
         <span class="visually-hidden">Task category</span>
         <select data-task-category aria-label="Task category">
           <option value="all">All categories</option>
           <option value="classic">Classic control</option>
           <option value="manipulation">Manipulation</option>
           <option value="locomotion">Locomotion</option>
           <option value="contrib">Contrib</option>
         </select>
       </label>
       <span class="environment-task-count" data-task-count></span>
     </div>
     <div class="environment-task-list" data-task-list></div>
     <p class="environment-empty-state" data-task-empty hidden>No tasks match this search.</p>
   </div>
